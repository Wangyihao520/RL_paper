# TensorFlow MAPPO training entry point
# train_e2_v2.py - TensorFlow version
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import csv
import logging
import random
import tensorflow as tf

try:
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.losses as losses
except ImportError:
    # 兼容较旧版本的TensorFlow
    from tensorflow.python.keras.models import load_model, Model
    from tensorflow.python.keras.layers import Dense, Input
    from tensorflow.python.keras.optimizers import Adam
    import tensorflow.python.keras.losses as losses
from sklearn.preprocessing import StandardScaler
import joblib
import time
import gc

from rsm_env import (
    RunningMeanStd, set_seed, huber_loss, relative_loss, r2_metric, rmse_metric,
    RSMEmissionEnv
)

set_seed(42)

class MAPPOAgent:
    """Multi-Agent PPO智能体 - 集中式训练，分布式执行（改进版）"""

    def __init__(self, num_agents, local_obs_dim, global_obs_dim, action_dim,
                 lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # 🎯 改进的探索参数
        self.exploration_noise = 0.3  # 增加初始探索噪声
        self.noise_decay = 0.998  # 更慢的噪声衰减
        self.min_noise = 0.08  # 提高最小噪声

        # �� 新增：多样化探索机制
        self.exploration_bonus_weight = 0.1  # 探索奖励权重
        self.action_history = []  # 记录历史动作
        self.max_history_length = 1000  # 历史记录最大长度

        # 🎲 新增：好奇心驱动探索
        self.curiosity_weight = 0.05  # 好奇心权重
        self.state_visit_count = {}  # 状态访问计数
        self.action_diversity_bonus = 0.02  # 动作多样性奖励

        # 🌟 新增：自适应探索策略
        self.performance_history = []  # 性能历史
        self.exploration_boost_threshold = 5  # 性能停滞阈值
        self.exploration_boost_factor = 2.0  # 探索增强因子
        self.current_exploration_boost = 1.0  # 当前探索增强

        # 🎯 新增：区间探索策略
        self.exploration_ranges = {
            'conservative': (0.85, 1.0),  # 保守减排：减排0-15%
            'moderate': (0.7, 0.85),  # 适度减排：减排15-30%
            'aggressive': (0.5, 0.7),  # 激进减排：减排30-50%
            'extreme': (0.3, 0.5)  # 极端减排：减排50-70%
        }
        self.current_exploration_mode = 'moderate'  # 当前探索模式
        self.exploration_mode_counter = 0  # 探索模式计数器
        self.exploration_mode_switch_interval = 10  # 探索模式切换间隔

        # 🔧 新增：奖励标准化
        self.reward_normalizer = RunningMeanStd()
        self.normalize_rewards = True

        # 🔧 新增：梯度裁剪
        self.max_grad_norm = 0.5

        # 构建改进的共享Actor网络（分布式执行）
        self.shared_actor = self._build_enhanced_shared_actor()
        self.shared_actor_old = self._build_enhanced_shared_actor()
        self.shared_actor_old.set_weights(self.shared_actor.get_weights())

        # 构建集中式Critic网络（集中式训练）
        self.centralized_critic = self._build_centralized_critic()

        # 优化器（降低学习率）
        self.actor_optimizer = Adam(learning_rate=lr)
        self.critic_optimizer = Adam(learning_rate=lr * 0.5)  # Critic使用更低的学习率

        # 存储轨迹
        self.memory = []

        print(f"改进版探索策略MAPPO智能体初始化完成:")
        print(f"  智能体数量: {num_agents}")
        print(f"  局部观察维度: {local_obs_dim}")
        print(f"  全局观察维度: {global_obs_dim}")
        print(f"  动作维度: {action_dim}")
        print(f"  初始探索噪声: {self.exploration_noise}")
        print(f"  奖励标准化: {self.normalize_rewards}")
        print(f"  梯度裁剪: {self.max_grad_norm}")
        print(f"  探索策略: 多样化 + 好奇心 + 自适应 + 区间探索")

    def _build_enhanced_shared_actor(self):
        """构建增强的共享Actor网络"""
        # 输入：局部观察 + 智能体ID
        local_obs = Input(shape=(self.local_obs_dim,), name='local_obs')
        agent_id = Input(shape=(self.num_agents,), name='agent_id')  # one-hot编码

        # 🎯 改进1：分离处理局部观察和智能体ID
        # 处理局部观察
        obs_branch = Dense(128, activation='relu', name='obs_branch1')(local_obs)
        obs_branch = Dense(64, activation='relu', name='obs_branch2')(obs_branch)

        # 处理智能体ID（增强省份特异性）
        id_branch = Dense(64, activation='relu', name='id_branch1')(agent_id)
        id_branch = Dense(32, activation='relu', name='id_branch2')(id_branch)

        # 🎯 改进2：注意力机制式的特征融合
        # 合并分支
        combined = tf.concat([obs_branch, id_branch], axis=-1)

        # 增强的特征提取
        x = Dense(256, activation='relu', name='actor_hidden1')(combined)
        x = Dense(256, activation='relu', name='actor_hidden2')(x)
        x = Dense(128, activation='relu', name='actor_hidden3')(x)

        # 🎯 改进3：双头输出（均值和方差）
        # 动作均值
        action_mean = Dense(self.action_dim, activation='sigmoid', name='action_mean')(x)

        # 动作标准差（用于更好的探索）
        action_std = Dense(self.action_dim, activation='softplus', name='action_std')(x)
        action_std = tf.clip_by_value(action_std, 0.01, 0.5)  # 限制标准差范围

        model = Model(inputs=[local_obs, agent_id], outputs=[action_mean, action_std], name='EnhancedSharedActor')
        return model

    def _build_centralized_critic(self):
        """构建集中式Critic网络"""
        # 输入：全局状态（包含所有智能体的信息）
        global_obs = Input(shape=(self.global_obs_dim,), name='global_obs')

        # Critic网络
        x = Dense(512, activation='relu', name='critic_hidden1')(global_obs)
        x = Dense(512, activation='relu', name='critic_hidden2')(x)
        x = Dense(256, activation='relu', name='critic_hidden3')(x)
        x = Dense(128, activation='relu', name='critic_hidden4')(x)

        # 输出状态价值
        value = Dense(1, activation='linear', name='value_output')(x)

        model = Model(inputs=global_obs, outputs=value, name='CentralizedCritic')
        return model

    def _get_exploration_mode_range(self):
        """获取当前探索模式的动作范围"""
        return self.exploration_ranges[self.current_exploration_mode]

    def _update_exploration_mode(self, episode):
        """更新探索模式"""
        if episode % self.exploration_mode_switch_interval == 0:
            modes = list(self.exploration_ranges.keys())
            # 循环切换探索模式
            current_idx = modes.index(self.current_exploration_mode)
            next_idx = (current_idx + 1) % len(modes)
            self.current_exploration_mode = modes[next_idx]

            print(f"🎲 探索模式切换: {self.current_exploration_mode}")
            print(f"   动作范围: {self.exploration_ranges[self.current_exploration_mode]}")

    def _calculate_curiosity_bonus(self, local_observations):
        """计算好奇心奖励（基于状态访问频率）"""
        curiosity_bonuses = []

        for obs in local_observations:
            # 将观察转换为状态键（简化版）
            state_key = tuple(np.round(obs[:10], 2))  # 只使用前10个特征作为状态表示

            # 更新访问计数
            if state_key not in self.state_visit_count:
                self.state_visit_count[state_key] = 0
            self.state_visit_count[state_key] += 1

            # 计算好奇心奖励（访问次数越少，奖励越高）
            visit_count = self.state_visit_count[state_key]
            curiosity_bonus = 1.0 / (visit_count + 1)  # 反比例关系
            curiosity_bonuses.append(curiosity_bonus)

        return np.array(curiosity_bonuses)

    def _calculate_diversity_bonus(self, actions):
        """计算动作多样性奖励"""
        if len(self.action_history) < 10:
            return 0.0

        # 计算当前动作与历史动作的差异
        recent_actions = np.array(self.action_history[-10:])  # 最近10个动作
        current_actions = actions.reshape(1, -1)

        # 计算欧氏距离
        distances = np.linalg.norm(recent_actions - current_actions, axis=1)
        avg_distance = np.mean(distances)

        # 距离越大，多样性奖励越高
        diversity_bonus = min(avg_distance * self.action_diversity_bonus, 0.1)

        return diversity_bonus

    def _update_performance_tracking(self, reward):
        """更新性能跟踪，用于自适应探索"""
        self.performance_history.append(reward)

        # 只保留最近的性能记录
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

        # 检查是否需要增强探索
        if len(self.performance_history) >= self.exploration_boost_threshold:
            recent_performance = self.performance_history[-self.exploration_boost_threshold:]
            performance_std = np.std(recent_performance)

            # 如果性能变化很小，增强探索
            if performance_std < 0.5:  # 性能停滞阈值
                self.current_exploration_boost = min(self.current_exploration_boost * 1.1,
                                                     self.exploration_boost_factor)
                print(f"🚀 检测到性能停滞，增强探索: {self.current_exploration_boost:.2f}")
            else:
                self.current_exploration_boost = max(self.current_exploration_boost * 0.95, 1.0)

    def select_actions(self, local_observations, agent_ids, episode=0, step_reward=None):
        """为所有智能体选择动作（增强探索版）"""
        # 更新探索模式
        self._update_exploration_mode(episode)

        # 更新性能跟踪
        if step_reward is not None:
            self._update_performance_tracking(step_reward)

        # 使用增强的共享Actor网络
        action_means, action_stds = self.shared_actor_old([local_observations, agent_ids])

        # 🎯 多层次探索策略

        # 1. 基础自适应噪声
        current_noise = max(self.exploration_noise, self.min_noise) * self.current_exploration_boost

        # 2. 好奇心驱动探索
        curiosity_bonuses = self._calculate_curiosity_bonus(local_observations)
        curiosity_noise = np.mean(curiosity_bonuses) * self.curiosity_weight

        # 3. 区间探索策略
        exploration_range = self._get_exploration_mode_range()
        range_center = (exploration_range[0] + exploration_range[1]) / 2
        range_width = exploration_range[1] - exploration_range[0]

        # 4. 基于网络输出的标准差进行采样
        base_noise = tf.random.normal(tf.shape(action_means), stddev=current_noise)
        adaptive_noise = action_stds * base_noise

        # 5. 添加差异化探索（为不同智能体添加不同的探索偏好）
        agent_specific_noise = tf.random.normal(tf.shape(action_means), stddev=0.1)
        for i in range(self.num_agents):
            # 为每个智能体添加轻微的探索偏好
            bias = 0.02 * tf.sin(float(i) * 2.0 * np.pi / self.num_agents)
            agent_specific_noise = tf.tensor_scatter_nd_update(
                agent_specific_noise,
                [[i, j] for j in range(self.action_dim)],
                [bias] * self.action_dim
            )

        # 6. 🎲 区间引导探索
        # 在当前探索模式的区间内添加额外的探索偏向
        range_bias = tf.random.uniform(tf.shape(action_means),
                                       minval=exploration_range[0] - 0.1,
                                       maxval=exploration_range[1] + 0.1)
        range_bias = tf.clip_by_value(range_bias, 0.3, 1.0)

        # 7. 🌟 随机探索增强（偶尔进行完全随机探索）
        random_exploration_prob = 0.1 * self.current_exploration_boost
        should_random_explore = tf.random.uniform([self.num_agents, 1]) < random_exploration_prob
        random_actions = tf.random.uniform(tf.shape(action_means), minval=0.4, maxval=1.0)

        # 综合所有探索策略
        # 基础动作
        base_actions = action_means + adaptive_noise + agent_specific_noise

        # 区间引导动作
        range_guided_actions = 0.7 * base_actions + 0.3 * range_bias

        # 添加好奇心噪声
        curiosity_enhanced_actions = range_guided_actions + curiosity_noise

        # 最终动作选择
        actions = tf.where(should_random_explore,
                           random_actions,
                           curiosity_enhanced_actions)

        # 🎯 动态动作范围（根据探索模式调整）
        final_actions = tf.clip_by_value(actions, 0.3, 1.0)  # 扩大总体动作范围

        # 计算对数概率（基于正态分布）
        log_probs = -0.5 * tf.reduce_sum(
            tf.square((final_actions - action_means) / (action_stds + 1e-8)) +
            tf.math.log(2 * np.pi * tf.square(action_stds + 1e-8)),
            axis=-1
        )

        # 更新动作历史
        self.action_history.append(final_actions.numpy().flatten())
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)

        # 计算多样性奖励
        diversity_bonus = self._calculate_diversity_bonus(final_actions.numpy())

        # 衰减探索噪声
        self.exploration_noise *= self.noise_decay

        print(f"🎯 增强探索统计:")
        print(f"  探索模式: {self.current_exploration_mode} {self.exploration_ranges[self.current_exploration_mode]}")
        print(f"  动作均值范围: [{tf.reduce_min(action_means):.3f}, {tf.reduce_max(action_means):.3f}]")
        print(f"  动作标准差范围: [{tf.reduce_min(action_stds):.3f}, {tf.reduce_max(action_stds):.3f}]")
        print(f"  最终动作范围: [{tf.reduce_min(final_actions):.3f}, {tf.reduce_max(final_actions):.3f}]")
        print(f"  当前探索噪声: {current_noise:.3f}")
        print(f"  好奇心噪声: {curiosity_noise:.3f}")
        print(f"  探索增强因子: {self.current_exploration_boost:.2f}")
        print(f"  多样性奖励: {diversity_bonus:.4f}")
        print(f"  随机探索概率: {random_exploration_prob:.3f}")

        return final_actions.numpy(), log_probs.numpy(), diversity_bonus

    def evaluate_actions(self, local_observations, agent_ids, actions):
        """评估动作的价值和概率（改进版）"""
        # 使用当前Actor网络重新计算动作概率
        action_means, action_stds = self.shared_actor([local_observations, agent_ids])

        # 计算新的对数概率
        new_log_probs = -0.5 * tf.reduce_sum(
            tf.square((actions - action_means) / (action_stds + 1e-8)) +
            tf.math.log(2 * np.pi * tf.square(action_stds + 1e-8)),
            axis=-1
        )

        # 计算熵（用于探索）
        entropy = 0.5 * tf.reduce_sum(
            tf.math.log(2 * np.pi * np.e * tf.square(action_stds + 1e-8)),
            axis=-1
        )

        return new_log_probs, entropy

    def get_values(self, global_observations):
        """获取全局状态的价值"""
        values = self.centralized_critic(global_observations)
        return tf.squeeze(values, axis=-1)

    def update(self):
        """MAPPO更新"""
        if len(self.memory) == 0:
            return

        print(f"开始MAPPO更新，经验池大小: {len(self.memory)}")

        # 准备数据
        local_obs_batch = []
        global_obs_batch = []
        agent_ids_batch = []
        actions_batch = []
        rewards_batch = []
        old_log_probs_batch = []

        for experience in self.memory:
            local_obs_batch.append(experience['local_obs'])
            global_obs_batch.append(experience['global_obs'])
            agent_ids_batch.append(experience['agent_ids'])
            actions_batch.append(experience['actions'])
            rewards_batch.append(experience['rewards'])
            old_log_probs_batch.append(experience['old_log_probs'])

        # 转换为张量
        local_obs_batch = tf.constant(local_obs_batch, dtype=tf.float32)
        global_obs_batch = tf.constant(global_obs_batch, dtype=tf.float32)
        agent_ids_batch = tf.constant(agent_ids_batch, dtype=tf.float32)
        actions_batch = tf.constant(actions_batch, dtype=tf.float32)
        rewards_batch = tf.constant(rewards_batch, dtype=tf.float32)
        old_log_probs_batch = tf.constant(old_log_probs_batch, dtype=tf.float32)

        print(f"原始数据形状:")
        print(f"  local_obs_batch: {local_obs_batch.shape}")
        print(f"  global_obs_batch: {global_obs_batch.shape}")
        print(f"  agent_ids_batch: {agent_ids_batch.shape}")
        print(f"  actions_batch: {actions_batch.shape}")
        print(f"  rewards_batch: {rewards_batch.shape}")
        print(f"  old_log_probs_batch: {old_log_probs_batch.shape}")

        # 重新整形数据：将(batch_size, num_agents, feature_dim)转换为(batch_size*num_agents, feature_dim)
        batch_size = tf.shape(local_obs_batch)[0]
        num_agents = tf.shape(local_obs_batch)[1]

        local_obs_batch = tf.reshape(local_obs_batch, [-1, local_obs_batch.shape[-1]])
        agent_ids_batch = tf.reshape(agent_ids_batch, [-1, agent_ids_batch.shape[-1]])
        actions_batch = tf.reshape(actions_batch, [-1, actions_batch.shape[-1]])
        old_log_probs_batch = tf.reshape(old_log_probs_batch, [-1])

        print(f"重新整形后数据形状:")
        print(f"  local_obs_batch: {local_obs_batch.shape}")
        print(f"  agent_ids_batch: {agent_ids_batch.shape}")
        print(f"  actions_batch: {actions_batch.shape}")
        print(f"  old_log_probs_batch: {old_log_probs_batch.shape}")

        # 计算折扣奖励和优势
        values = self.get_values(global_obs_batch)
        print(f"  values: {values.shape}")

        # 对于MAPPO，每个时间步的奖励是所有智能体的平均奖励
        # rewards_batch形状应该是(batch_size,)，而不是(batch_size*num_agents,)
        # 所以我们需要重新整形rewards

        # 如果rewards_batch是1D的，说明每个时间步只有一个奖励值
        if len(rewards_batch.shape) == 1:
            # 直接使用
            step_rewards = rewards_batch
        else:
            # 如果是2D的，需要处理
            step_rewards = tf.reduce_mean(tf.reshape(rewards_batch, [batch_size, num_agents]), axis=1)

        print(f"  step_rewards: {step_rewards.shape}")

        returns, advantages = self._compute_gae(step_rewards, values)

        # 将advantages扩展到所有智能体
        # advantages形状：(batch_size,) -> (batch_size*num_agents,)
        advantages = tf.repeat(advantages, num_agents)

        print(f"  返回的advantages: {advantages.shape}")
        print(f"  返回的returns: {returns.shape}")

        # 标准化优势
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        # PPO更新
        for epoch in range(self.k_epochs):
            # 更新Actor
            with tf.GradientTape() as actor_tape:
                new_log_probs, entropy = self.evaluate_actions(local_obs_batch, agent_ids_batch, actions_batch)

                # 计算比率
                ratio = tf.exp(new_log_probs - old_log_probs_batch)

                # PPO目标函数
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # Actor损失 = -PPO目标 - 熵奖励
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - 0.01 * tf.reduce_mean(entropy)

            # 更新Critic
            with tf.GradientTape() as critic_tape:
                current_values = self.get_values(global_obs_batch)
                critic_loss = tf.reduce_mean(tf.square(returns - current_values))

            # 应用梯度
            actor_grads = actor_tape.gradient(actor_loss, self.shared_actor.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss, self.centralized_critic.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(actor_grads, self.shared_actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.centralized_critic.trainable_variables))

        # 更新旧策略
        self.shared_actor_old.set_weights(self.shared_actor.get_weights())

        # 清空经验池
        self.memory = []

        print(f"MAPPO更新完成，Actor损失: {actor_loss:.4f}, Critic损失: {critic_loss:.4f}")

    def _compute_gae(self, rewards, values):
        """计算GAE（Generalized Advantage Estimation）"""
        gae_lambda = 0.95

        # 转换为numpy数组进行计算
        if isinstance(rewards, tf.Tensor):
            rewards = rewards.numpy()
        if isinstance(values, tf.Tensor):
            values = values.numpy()

        # 确保是1D数组
        rewards = np.atleast_1d(rewards)
        values = np.atleast_1d(values)

        # 获取序列长度
        seq_len = len(rewards)

        # 添加最后一个价值（用于计算最后一步的优势）
        values_with_next = np.concatenate([values, [0.0]])

        advantages = []
        gae = 0

        # 反向计算GAE
        for t in reversed(range(seq_len)):
            delta = rewards[t] + self.gamma * values_with_next[t + 1] - values_with_next[t]
            gae = delta + self.gamma * gae_lambda * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + values

        # 转换回TensorFlow张量
        return tf.constant(returns, dtype=tf.float32), tf.constant(advantages, dtype=tf.float32)

    def store_experience(self, local_obs, global_obs, agent_ids, actions, rewards, old_log_probs):
        """存储经验"""
        experience = {
            'local_obs': local_obs,
            'global_obs': global_obs,
            'agent_ids': agent_ids,
            'actions': actions,
            'rewards': rewards,
            'old_log_probs': old_log_probs
        }
        self.memory.append(experience)


def train_mappo():
    """训练MAPPO算法"""
    print("开始训练MAPPO算法...")

    # 检查文件
    required_files = [
        ("./models_unet", "U-Net模型目录"),
        ("./conc/base/base.csv", "基准浓度数据"),
        ("./conc/clean/clean.csv", "清洁情景数据"),
        ("./prov_grid_map/36kmprov.csv", "省份映射数据"),
        ("./other data/cost.csv", "成本系数数据"),
        ("./input_emi/base", "基准排放数据目录")
    ]

    for file_path, description in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 找不到{description}: {file_path}")
            return None, None

    try:
        # 创建环境
        env = RSMEmissionEnv(
            model_path="./models_unet",
            scaler_path="./models_unet",
            base_conc_path="./conc/base/base.csv",
            clean_conc_path="./conc/clean/clean.csv",
            province_map_path="./prov_grid_map/36kmprov.csv",
            base_emission_path="./input_emi/base",
            cost_data_path="./other data/cost.csv",
            transport_matrix_path="./other data/transport.csv",
            fairness_weight=1000,
            fairness_metric='l1',
            fairness_external_only=False,
            max_steps=8
        )

        # 🎯 更新观察维度计算（适应新增的省份特征）
        local_obs_dim = env.action_dim + env.num_provinces + env.province_feature_dim  # 25 + 32 + 8 = 65
        global_obs_dim = env.num_provinces * env.action_dim + env.num_provinces * 2 + 1  # 800 + 32 + 32 + 1 = 865

        print(f"🎯 观察空间维度（更新后）:")
        print(
            f"  局部观察维度: {local_obs_dim} (动作{env.action_dim} + 省份编码{env.num_provinces} + 省份特征{env.province_feature_dim})")
        print(f"  全局观察维度: {global_obs_dim}")

        # 创建MAPPO智能体
        mappo_agent = MAPPOAgent(
            num_agents=env.num_provinces,
            local_obs_dim=local_obs_dim,
            global_obs_dim=global_obs_dim,
            action_dim=env.action_dim,
            lr=2e-4  # 🎯 降低学习率，提高稳定性
        )

        # 🎯 改进的训练参数
        max_episodes = 100  # 增加训练回合数
        update_interval = 3  # 更频繁的更新

        # 创建输出目录
        os.makedirs("./results", exist_ok=True)

        episode_rewards = []
        best_reward = float('-inf')

        # 🎯 添加训练统计
        diversity_scores = []  # 记录策略多样性
        convergence_patience = 0
        max_patience = 20

        # 开始训练
        for episode in range(max_episodes):
            print(f"\n{'=' * 80}")
            print(f"🚀 开始训练回合 {episode + 1}/{max_episodes}")
            print(f"   当前探索噪声: {mappo_agent.exploration_noise:.4f}")
            print(f"{'=' * 80}")

            try:
                # 重置环境
                env.reset()

                # 获取初始观察
                local_observations = env.get_local_observations()
                global_observation = env.get_global_observation()
                agent_ids = env.get_agent_ids()

                print(f"环境重置完成，观察形状:")
                print(f"  局部观察: {local_observations.shape}")
                print(f"  全局观察: {global_observation.shape}")
                print(f"  智能体ID: {agent_ids.shape}")

                episode_reward = 0
                episode_experiences = []
                episode_actions = []  # 记录回合中的所有动作

                print(f"\n开始执行 {env.max_steps} 个环境步骤...")

                for step in range(env.max_steps):
                    print(f"\n🔄 步骤 {step + 1}/{env.max_steps}")

                    try:
                        # 使用MAPPO选择动作（传递episode和step_reward参数）
                        actions, log_probs, diversity_bonus = mappo_agent.select_actions(
                            local_observations, agent_ids, episode=episode,
                            step_reward=episode_reward if step > 0 else None
                        )
                        episode_actions.append(actions.copy())
                        print(f"MAPPO动作选择完成，动作形状: {actions.shape}, 多样性奖励: {diversity_bonus:.4f}")

                        # 执行环境步骤
                        next_local_obs, reward, done, info = env.step(actions)
                        next_global_obs = env.get_global_observation()

                        # 🎯 将多样性奖励加入总奖励
                        enhanced_reward = reward + diversity_bonus

                        print(
                            f"环境步骤执行完成，基础奖励: {reward:.4f}, 多样性奖励: {diversity_bonus:.4f}, 总奖励: {enhanced_reward:.4f}")

                        # 存储经验
                        experience = {
                            'local_obs': local_observations,
                            'global_obs': global_observation,
                            'agent_ids': agent_ids,
                            'actions': actions,
                            'rewards': enhanced_reward,  # 使用增强后的奖励
                            'old_log_probs': log_probs
                        }
                        episode_experiences.append(experience)

                        # 更新观察
                        local_observations = env.get_local_observations()
                        global_observation = next_global_obs

                        episode_reward += enhanced_reward  # 累积增强后的奖励

                        print(f"步骤 {step + 1} 完成，当前累积奖励: {episode_reward:.4f}")

                        if done:
                            print("✅ 回合提前结束")
                            break

                    except Exception as e:
                        print(f"❌ 步骤 {step + 1} 失败: {e}")
                        import traceback
                        print("详细错误堆栈:")
                        traceback.print_exc()
                        print(f"错误类型: {type(e).__name__}")
                        print(f"错误位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
                        break

                # 🎯 计算回合统计
                episode_diversity = 0
                if len(episode_actions) > 0:
                    # 计算动作多样性
                    all_actions = np.concatenate(episode_actions, axis=0)
                    action_std = np.std(all_actions, axis=0)
                    episode_diversity = np.mean(action_std)
                    diversity_scores.append(episode_diversity)

                print(f"\n📊 回合 {episode + 1} 统计:")
                print(f"   总步骤数: {len(episode_experiences)}")
                print(f"   总奖励: {episode_reward:.4f}")
                print(f"   动作多样性: {episode_diversity:.4f}")
                if len(episode_experiences) > 0:
                    print(f"   平均步骤奖励: {episode_reward / len(episode_experiences):.4f}")
                else:
                    print(f"   平均步骤奖励: 0.0000 (无有效步骤)")

                # 存储所有经验到MAPPO智能体
                print("💾 存储训练经验...")
                for experience in episode_experiences:
                    mappo_agent.store_experience(
                        experience['local_obs'],
                        experience['global_obs'],
                        experience['agent_ids'],
                        experience['actions'],
                        experience['rewards'],
                        experience['old_log_probs']
                    )

                print(f"成功存储 {len(episode_experiences)} 个经验")

                # 更新MAPPO智能体
                if (episode + 1) % update_interval == 0:
                    print(f"\n🔄 更新MAPPO智能体（每 {update_interval} 回合更新一次）...")
                    try:
                        mappo_agent.update()
                        print("✅ MAPPO更新成功")
                    except Exception as e:
                        print(f"❌ MAPPO更新失败: {e}")

                episode_rewards.append(episode_reward)

                # 🎯 改进的收敛检测
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    convergence_patience = 0
                    print(f"🏆 发现更好的策略！新最佳奖励: {best_reward:.4f}")

                    try:
                        mappo_agent.shared_actor.save_weights(f"./results/best_mappo_actor")
                        mappo_agent.centralized_critic.save_weights(f"./results/best_mappo_critic")
                        print("✅ 保存了最佳MAPPO模型")
                    except Exception as e:
                        print(f"❌ 保存模型失败: {e}")
                else:
                    convergence_patience += 1

                print(f"\n🎯 回合 {episode + 1} 完成:")
                print(f"   当前奖励: {episode_reward:.4f}")
                print(f"   历史最佳: {best_reward:.4f}")
                print(f"   收敛耐心: {convergence_patience}/{max_patience}")

                # 早停检测
                if convergence_patience >= max_patience:
                    print(f"🛑 达到收敛耐心限制，提前停止训练")
                    break

                # 定期保存结果
                if (episode + 1) % 10 == 0:
                    print(f"\n📈 保存训练进度（第 {episode + 1} 回合）...")
                    try:
                        plt.figure(figsize=(15, 10))

                        # 奖励曲线
                        plt.subplot(2, 2, 1)
                        plt.plot(episode_rewards, 'b-', linewidth=2)
                        plt.title('MAPPO训练奖励曲线', fontsize=14)
                        plt.xlabel('回合')
                        plt.ylabel('奖励')
                        plt.grid(True, alpha=0.3)

                        # 奖励移动平均
                        plt.subplot(2, 2, 2)
                        if len(episode_rewards) >= 10:
                            moving_avg = np.convolve(episode_rewards, np.ones(10) / 10, mode='valid')
                            plt.plot(moving_avg, 'r-', linewidth=2, label='10回合移动平均')
                            plt.legend()
                        plt.title('奖励移动平均', fontsize=14)
                        plt.xlabel('回合')
                        plt.ylabel('平均奖励')
                        plt.grid(True, alpha=0.3)

                        # 动作多样性
                        plt.subplot(2, 2, 3)
                        if len(diversity_scores) > 0:
                            plt.plot(diversity_scores, 'g-', linewidth=2)
                            plt.title('动作多样性变化', fontsize=14)
                            plt.xlabel('回合')
                            plt.ylabel('动作标准差')
                            plt.grid(True, alpha=0.3)

                        # 探索噪声衰减
                        plt.subplot(2, 2, 4)
                        noise_history = [
                            mappo_agent.exploration_noise * (1 / mappo_agent.noise_decay) ** (episode + 1 - i)
                            for i in range(min(episode + 1, 50))]
                        plt.plot(noise_history, 'm-', linewidth=2)
                        plt.title('探索噪声衰减', fontsize=14)
                        plt.xlabel('回合')
                        plt.ylabel('探索噪声')
                        plt.grid(True, alpha=0.3)

                        plt.tight_layout()
                        plt.savefig('./results/mappo_training_curve.png', dpi=300, bbox_inches='tight')
                        plt.close()

                        np.save('./results/mappo_episode_rewards.npy', np.array(episode_rewards))
                        np.save('./results/mappo_diversity_scores.npy', np.array(diversity_scores))

                        print(f"✅ 训练进度已保存")
                        print(f"   当前最佳奖励: {best_reward:.4f}")
                        print(f"   最近10回合平均: {np.mean(episode_rewards[-10:]):.4f}")
                        if len(diversity_scores) > 0:
                            print(f"   当前动作多样性: {diversity_scores[-1]:.4f}")

                    except Exception as e:
                        print(f"❌ 保存结果失败: {e}")

            except Exception as e:
                print(f"❌ 训练回合 {episode + 1} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("🎉 MAPPO训练完成！")
        return mappo_agent, episode_rewards

    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("🚀 开始MAPPO多智能体强化学习训练")
    print("=" * 80)

    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)

    try:
        mappo_agent, rewards = train_mappo()

        if mappo_agent is not None:
            print("\n🎉 MAPPO训练完成！")
            print("=" * 80)
            print("结果文件保存在 ./results/ 目录中")
            print("包含以下文件：")
            print("  - best_mappo_actor: 最佳演员网络权重")
            print("  - best_mappo_critic: 最佳评论家网络权重")
            print("  - mappo_training_curve.png: 训练曲线图")
            print("  - mappo_episode_rewards.npy: 回合奖励数据")

            if rewards:
                print(f"\n📈 训练统计:")
                print(f"  总回合数: {len(rewards)}")
                print(f"  最佳奖励: {max(rewards):.4f}")
                print(f"  平均奖励: {np.mean(rewards):.4f}")
                print(f"  最终奖励: {rewards[-1]:.4f}")
        else:
            print("\n❌ MAPPO训练失败")

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback

        traceback.print_exc()