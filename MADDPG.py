# train_e2_v5_pytorch_MADDPG.py - PyTorch MADDPG training script
# ============================================================
# Version: v14.12 - reward normalization updated for a larger reward range
# Last updated: 2024-12-16
# ============================================================
"""
This script trains the MADDPG baseline for multi-province emission control.

Main features:
- deterministic policy gradient updates
- replay buffer training
- soft target-network updates
- centralized training with decentralized execution
- OU or Gaussian exploration noise
"""
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import csv
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# Matplotlib display settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Use a non-interactive backend for file output
import matplotlib
matplotlib.use('Agg')

# JSON encoder for NumPy and PyTorch objects
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

from sklearn.preprocessing import StandardScaler
import joblib
import time
import sys
import gc
import copy

# Reuse environment utilities from the shared module
from rsm_env import (
    RunningMeanStd, set_seed, huber_loss, relative_loss, r2_metric, rmse_metric,
    RSMEmissionEnv
)
from experiment_configs import V14_MAIN_CONFIG

# ==================== Fairness reward settings (MADDPG) ====================
# The fairness reward is computed inside `rsm_env.RSMEmissionEnv`.
# Pass `transport_matrix_path` and the fairness parameters explicitly to keep it reproducible.
TRANSPORT_MATRIX_PATH = "./other data/transport.csv"
FAIRNESS_WEIGHT = 10000.0
FAIRNESS_METRIC = "l2"  # 'l1' or 'l2'
FAIRNESS_MODE = "penalty"  # 'penalty' or 'reward'
FAIRNESS_EXTERNAL_ONLY = False

# Set a default random seed
set_seed(42)


# Province monitor: track policy snapshots and rewards for a single province
class ProvinceMonitor:
    """Track policy changes and best rewards for a single province."""

    def __init__(self, province_idx, province_name, max_steps=8):
        self.province_idx = province_idx
        self.province_name = province_name
        self.max_steps = max_steps
        self.best_reward = float('-inf')
        self.best_episode = 0
        self.best_policy_state_dict = None
        self.best_actions_sequence = None
        self.reward_history = []
        self.episode_history = []
        self.policy_history = []
        self.actions_history = []
        self.is_fixed = False
        self.fixed_episode = None

    def update(self, episode, episode_reward, policy_state_dict, actions_sequence=None):
        """Update the monitor with the latest episode information."""
        self.reward_history.append(episode_reward)
        self.episode_history.append(episode)

        if self.is_fixed:
            return

        policy_snapshot = copy.deepcopy(policy_state_dict) if policy_state_dict else None
        self.policy_history.append(policy_snapshot)

        actions_snapshot = copy.deepcopy(actions_sequence) if actions_sequence else None
        self.actions_history.append(actions_snapshot)

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_episode = episode
            if policy_snapshot is not None:
                self.best_policy_state_dict = policy_snapshot
            if actions_snapshot is not None:
                self.best_actions_sequence = actions_snapshot

    def get_best_policy(self):
        """Return a deep copy of the best policy state."""
        if self.best_policy_state_dict is None:
            return None
        return copy.deepcopy(self.best_policy_state_dict)

    def get_best_actions_sequence(self):
        """Return the action sequence of the best episode."""
        if self.best_actions_sequence is None:
            return None
        return copy.deepcopy(self.best_actions_sequence)

    def fix_policy(self, episode):
        """Freeze the current policy for this province."""
        self.is_fixed = True
        self.fixed_episode = episode

    def get_stats(self):
        """Return summary statistics for the province."""
        return {
            'province_idx': self.province_idx,
            'province_name': self.province_name,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'is_fixed': self.is_fixed,
            'fixed_episode': self.fixed_episode,
            'total_episodes': len(self.reward_history),
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'std_reward': np.std(self.reward_history) if len(self.reward_history) > 1 else 0.0,
            'min_reward': np.min(self.reward_history) if self.reward_history else 0.0,
            'max_reward': np.max(self.reward_history) if self.reward_history else 0.0,
            'reward_history': self.reward_history
        }


# Reuse the detailed logger from the earlier implementation
from v13_copy import DetailedLogger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure PyTorch runtime
print(f"PyTorch版本: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

if torch.cuda.is_available():
    print(f"✅ 已启用GPU训练，使用 {torch.cuda.device_count()} 个GPU")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    print("❌ 未检测到GPU，使用CPU训练")
    torch.set_num_threads(16)


# ==================== MADDPG网络定义 ====================

class DeterministicActor(nn.Module):
    """MADDPG确定性Actor网络（输出确定性动作，不是概率分布）"""

    def __init__(self, local_obs_dim, num_agents, action_dim):
        super(DeterministicActor, self).__init__()

        # 局部观察分支
        self.obs_branch = nn.Sequential(
            nn.Linear(local_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 智能体ID分支
        self.id_branch = nn.Sequential(
            nn.Linear(num_agents, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 合并后的特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(64 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 确定性动作输出（使用tanh激活，然后缩放到目标范围）
        self.action_head = nn.Linear(64, action_dim)

    def forward(self, local_obs, agent_id):
        obs_features = self.obs_branch(local_obs)
        id_features = self.id_branch(agent_id)
        combined = torch.cat([obs_features, id_features], dim=-1)
        features = self.feature_extractor(combined)
        
        # 输出确定性动作：tanh激活 -> [0.01, 0.2]范围
        action = self.action_head(features)
        action = torch.sigmoid(action) * 0.19 + 0.01
        
        return action


class MADDPGCritic(nn.Module):
    """MADDPG集中式Critic网络（输入：全局状态+所有agent的动作）"""

    def __init__(self, global_obs_dim, action_dim, num_agents):
        super(MADDPGCritic, self).__init__()
        
        # 输入：全局观察 + 所有agent的动作
        input_dim = global_obs_dim + action_dim * num_agents
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_obs, all_actions):
        """
        Args:
            global_obs: (batch_size, global_obs_dim)
            all_actions: (batch_size, action_dim * num_agents) 所有agent的动作拼接
        """
        x = torch.cat([global_obs, all_actions], dim=-1)
        return self.critic(x).squeeze(-1)


# ==================== 经验回放缓冲区 ====================

class ReplayBuffer:
    """MADDPG经验回放缓冲区"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, local_obs, global_obs, agent_ids, actions, rewards, next_local_obs, next_global_obs, done):
        """添加经验"""
        self.buffer.append({
            'local_obs': local_obs,
            'global_obs': global_obs,
            'agent_ids': agent_ids,
            'actions': actions,
            'rewards': rewards,
            'next_local_obs': next_local_obs,
            'next_global_obs': next_global_obs,
            'done': done
        })

    def sample(self, batch_size):
        """随机采样batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'local_obs': np.array([e['local_obs'] for e in batch]),
            'global_obs': np.array([e['global_obs'] for e in batch]),
            'agent_ids': np.array([e['agent_ids'] for e in batch]),
            'actions': np.array([e['actions'] for e in batch]),
            'rewards': np.array([e['rewards'] for e in batch]),
            'next_local_obs': np.array([e['next_local_obs'] for e in batch]),
            'next_global_obs': np.array([e['next_global_obs'] for e in batch]),
            'done': np.array([e['done'] for e in batch])
        }

    def __len__(self):
        return len(self.buffer)


# ==================== Ornstein-Uhlenbeck噪声 ====================

class OUNoise:
    """Ornstein-Uhlenbeck噪声过程（用于连续动作空间的探索）"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# ==================== MADDPG智能体 ====================

class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient智能体"""

    def __init__(self, num_agents, local_obs_dim, global_obs_dim, action_dim,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005, device='cuda'):
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 学习参数
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau  # 软更新系数

        # 探索噪声
        self.exploration_noise = 0.3
        self.noise_decay = 0.999
        self.min_noise = 0.1
        self.ou_noise = [OUNoise(action_dim) for _ in range(num_agents)]

        # 奖励稳定性
        self.reward_normalizer = RunningMeanStd()
        self.reward_clipping = True
        self.clip_reward_threshold = 50.0

        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 256
        self.min_buffer_size = 1000  # 开始训练前的最小buffer大小

        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'critic_losses': [],
            'actor_losses': [],
            'actor_losses_per_province': [],
            'q_values': [],
            'exploration_noises': [],
            'reward_stds': [],
            'gradient_norms': [],
        }

        # 🎯 每个省份独立的Actor网络
        self.actor_list = nn.ModuleList(
            [DeterministicActor(local_obs_dim, num_agents, action_dim).to(self.device) 
             for _ in range(num_agents)]
        )
        self.actor_target_list = nn.ModuleList(
            [DeterministicActor(local_obs_dim, num_agents, action_dim).to(self.device) 
             for _ in range(num_agents)]
        )
        
        # 初始化目标网络
        for actor, actor_target in zip(self.actor_list, self.actor_target_list):
            actor_target.load_state_dict(actor.state_dict())
            actor_target.eval()

        # 集中式Critic（共享）
        self.centralized_critic = MADDPGCritic(global_obs_dim, action_dim, num_agents).to(self.device)
        self.centralized_critic_target = MADDPGCritic(global_obs_dim, action_dim, num_agents).to(self.device)
        self.centralized_critic_target.load_state_dict(self.centralized_critic.state_dict())
        self.centralized_critic_target.eval()

        # 优化器
        self.actor_optimizers = []
        for actor in self.actor_list:
            actor_optimizer = optim.Adam(actor.parameters(), lr=self.lr_actor)
            self.actor_optimizers.append(actor_optimizer)
        
        self.critic_optimizer = optim.Adam(self.centralized_critic.parameters(), lr=self.lr_critic)

        # 日志记录
        self.logger = None

        # 🎯 优先级训练相关
        self.fixed_policies = {}
        self.fixed_actor_networks = {}
        self.fixed_actions_sequences = {}
        self.province_monitors = {}
        self.current_training_province_idx = None

    def initialize_province_monitors(self, province_names):
        """初始化省份监控器"""
        for idx, name in enumerate(province_names):
            self.province_monitors[idx] = ProvinceMonitor(idx, name)

    def select_actions(self, local_observations, agent_ids, episode=0, add_noise=True):
        """选择动作（确定性策略 + 探索噪声）"""
        local_obs_tensor = torch.FloatTensor(local_observations).to(self.device)
        agent_ids_tensor = torch.FloatTensor(agent_ids).to(self.device)

        actions_np = np.zeros((len(local_observations), self.action_dim))

        for province_idx in range(len(local_observations)):
            actor = self.actor_list[province_idx]
            with torch.no_grad():
                action = actor(
                    local_obs_tensor[province_idx:province_idx + 1],
                    agent_ids_tensor[province_idx:province_idx + 1]
                )

            action_np = action[0].cpu().numpy()

            # 添加探索噪声
            if add_noise:
                noise = self.ou_noise[province_idx].sample() * self.exploration_noise
                action_np = action_np + noise
                action_np = np.clip(action_np, 0.01, 0.2)

            actions_np[province_idx] = action_np

        return actions_np

    def select_actions_with_fixed(self, local_observations, agent_ids, episode=0, step=0, add_noise=True):
        """选择动作：支持固定策略模式"""
        local_obs_tensor = torch.FloatTensor(local_observations).to(self.device)
        agent_ids_tensor = torch.FloatTensor(agent_ids).to(self.device)

        actions = np.zeros((len(local_observations), self.action_dim))

        for province_idx in range(len(local_observations)):
            # 检查是否有固定的动作序列
            if province_idx in self.fixed_actions_sequences:
                fixed_action = self.get_fixed_action(province_idx, step)
                if fixed_action is not None:
                    actions[province_idx] = np.array(fixed_action)
                    continue

            # 检查是否有固定策略网络
            if province_idx in self.fixed_policies:
                fixed_actor = self._get_fixed_actor(province_idx)
                if fixed_actor is None:
                    fixed_actor = self.actor_list[province_idx]
                with torch.no_grad():
                    action = fixed_actor(
                        local_obs_tensor[province_idx:province_idx + 1],
                        agent_ids_tensor[province_idx:province_idx + 1]
                    )
                actions[province_idx] = action[0].cpu().numpy()
            else:
                # 正常探索
                actor = self.actor_list[province_idx]
                with torch.no_grad():
                    action = actor(
                        local_obs_tensor[province_idx:province_idx + 1],
                        agent_ids_tensor[province_idx:province_idx + 1]
                    )
                
                action_np = action[0].cpu().numpy()

                # 添加探索噪声
                if add_noise:
                    noise = self.ou_noise[province_idx].sample() * self.exploration_noise
                    action_np = action_np + noise
                    action_np = np.clip(action_np, 0.01, 0.2)

                actions[province_idx] = action_np

        return actions

    def store_transition(self, local_obs, global_obs, agent_ids, actions, rewards, 
                        next_local_obs, next_global_obs, done):
        """存储transition到replay buffer"""
        self.replay_buffer.add(
            local_obs, global_obs, agent_ids, actions, rewards,
            next_local_obs, next_global_obs, done
        )

    def update(self, episode):
        """
        MADDPG更新
        
        注意：MADDPG是off-policy算法，使用每小步的即时奖励进行更新
        - 每个transition存储的是(s, a, r, s', done)，其中r是当前小步的即时奖励
        - 更新时使用Bellman方程: Q(s,a) = r + γ * Q(s',a') * (1-done)
        - 不是使用整个episode的累积奖励（那是on-policy方法如PPO的做法）
        """
        if len(self.replay_buffer) < self.min_buffer_size:
            return

        # 从replay buffer采样（每个样本都是一个小步的transition）
        batch = self.replay_buffer.sample(self.batch_size)

        local_obs_batch = torch.FloatTensor(batch['local_obs']).to(self.device)
        global_obs_batch = torch.FloatTensor(batch['global_obs']).to(self.device)
        agent_ids_batch = torch.FloatTensor(batch['agent_ids']).to(self.device)
        actions_batch = torch.FloatTensor(batch['actions']).to(self.device)
        
        # ✅ 确保rewards是正确的形状
        rewards_array = np.array(batch['rewards'])
        if len(rewards_array.shape) == 0:  # 标量
            rewards_array = np.array([rewards_array])
        rewards_batch = torch.FloatTensor(rewards_array).to(self.device)
        
        next_local_obs_batch = torch.FloatTensor(batch['next_local_obs']).to(self.device)
        next_global_obs_batch = torch.FloatTensor(batch['next_global_obs']).to(self.device)
        
        # ✅ 确保done是正确的形状
        done_array = np.array(batch['done'])
        if len(done_array.shape) == 0:
            done_array = np.array([done_array])
        done_batch = torch.FloatTensor(done_array).to(self.device)

        # ✅ 奖励标准化：针对每小步的即时奖励进行标准化
        # 注意：rewards_batch中的每个元素都是一个小步的即时奖励（不是累积奖励）
        # 
        # 奖励范围分析（基于实际日志）：
        # - 从episode_summary日志看，每小步即时奖励范围：354 ~ 729
        # - 整个episode总奖励范围：1800 ~ 4800（8小步累积）
        # - 平均每小步奖励：约 225 ~ 600
        # 
        # 标准化策略对比：
        # - 使用 /50 缩放：354/50 ~ 729/50 = 7.08 ~ 14.58（✅ 推荐：区分度好，范围合理）
        # - 使用 /10 缩放：35.4 ~ 72.9（❌ 不推荐：范围太大，可能导致Q值不稳定）
        # - 使用 /100 缩放：3.54 ~ 7.29（⚠️ 可以，但区分度稍小）
        # 
        # 选择 /50 的原因：
        # 1. 保留足够的区分度（范围约7.5，能清楚区分好坏：差奖励~7，好奖励~14）
        # 2. 不会导致Q值爆炸（范围在-10到15之间，安全）
        # 3. 能清楚区分好坏奖励（差奖励~7-9，好奖励~12-15）
        # 4. 符合MADDPG的奖励尺度（Q值通常在-10到20之间）
        
        # 记录原始奖励统计（每100个episode打印一次）
        if episode % 100 == 0 and len(rewards_batch) > 0:
            raw_rewards_np = rewards_batch.cpu().detach().numpy() if isinstance(rewards_batch, torch.Tensor) else rewards_batch
            print(f"📊 Episode {episode} 奖励统计:")
            print(f"   原始奖励范围: [{raw_rewards_np.min():.2f}, {raw_rewards_np.max():.2f}]")
            print(f"   原始奖励均值: {raw_rewards_np.mean():.2f}, 标准差: {raw_rewards_np.std():.2f}")
        
        rewards_batch = rewards_batch / 50.0  # ✅ 改为/50，基于实际奖励范围（354-729）
        rewards_batch = torch.clamp(rewards_batch, -10.0, 20.0)  # 防止极端值，上限设为20
        
        # 记录标准化后的奖励统计
        if episode % 100 == 0 and len(rewards_batch) > 0:
            norm_rewards_np = rewards_batch.cpu().detach().numpy() if isinstance(rewards_batch, torch.Tensor) else rewards_batch
            reward_range = norm_rewards_np.max() - norm_rewards_np.min()
            print(f"   标准化后范围: [{norm_rewards_np.min():.2f}, {norm_rewards_np.max():.2f}]")
            print(f"   标准化后均值: {norm_rewards_np.mean():.2f}, 标准差: {norm_rewards_np.std():.2f}")
            print(f"   奖励范围: {reward_range:.2f} ({'✅ 区分度良好' if reward_range > 3.0 else '⚠️ 区分度不足'})")

        batch_size = local_obs_batch.shape[0]
        num_agents = self.num_agents

        # Reshape
        local_obs_batch_flat = local_obs_batch.view(-1, local_obs_batch.shape[-1])
        agent_ids_batch_flat = agent_ids_batch.view(-1, agent_ids_batch.shape[-1])
        actions_batch_flat = actions_batch.view(-1, actions_batch.shape[-1])
        next_local_obs_batch_flat = next_local_obs_batch.view(-1, next_local_obs_batch.shape[-1])

        # ========== 更新Critic ==========
        # MADDPG Critic更新逻辑：
        # 1. 使用target Actor网络计算next state的actions
        # 2. 使用target Critic网络计算target Q值
        # 3. 使用当前Critic网络计算current Q值
        # 4. 最小化MSE损失 (current_q - target_q)^2
        
        self.critic_optimizer.zero_grad()

        # 计算target Q值
        with torch.no_grad():
            # 使用目标Actor网络计算下一个状态的动作
            next_actions_list = []
            for p_idx in range(num_agents):
                idxs = torch.arange(p_idx, batch_size * num_agents, num_agents, device=self.device)
                next_obs_p = next_local_obs_batch_flat[idxs]  # (batch_size, local_obs_dim)
                next_ids_p = agent_ids_batch_flat[idxs]  # (batch_size, num_agents)
                
                next_action_p = self.actor_target_list[p_idx](next_obs_p, next_ids_p)  # (batch_size, action_dim)
                next_actions_list.append(next_action_p)
            
            # 拼接所有agent的动作: (batch_size * num_agents, action_dim)
            next_actions = torch.cat(next_actions_list, dim=0)
            # Reshape: (batch_size, num_agents, action_dim)
            next_actions_reshaped = next_actions.view(batch_size, num_agents, -1)
            # Flatten: (batch_size, num_agents * action_dim) - 用于Critic输入
            next_actions_flat = next_actions_reshaped.view(batch_size, -1)

            # 计算target Q: Q_target(s', a') where a' = μ_target(s')
            target_q = self.centralized_critic_target(next_global_obs_batch, next_actions_flat)  # (batch_size,)
            # ✅ 修复：如果rewards_batch是1维的，直接使用；如果是2维的，取平均
            if len(rewards_batch.shape) == 1:
                reward_for_target = rewards_batch  # (batch_size,)
            else:
                reward_for_target = rewards_batch.mean(dim=1)  # (batch_size,)
            # ✅ 裁剪 target Q 值，防止爆炸
            target_q = torch.clamp(target_q, -100.0, 100.0)
            
            # Bellman方程: target_q = r + γ * Q_target(s', a') * (1 - done)
            target_q = reward_for_target + self.gamma * target_q * (1 - done_batch)
            
            # ✅ 再次裁剪最终的 target Q
            target_q = torch.clamp(target_q, -100.0, 100.0)

        # 当前Q值
        actions_batch_reshaped = actions_batch.view(batch_size, -1)
        current_q = self.centralized_critic(global_obs_batch, actions_batch_reshaped)

        # Critic损失：使用Huber Loss，对大误差更鲁棒（线性梯度而非二次梯度）
        critic_loss = F.smooth_l1_loss(current_q, target_q)
        
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ========== 更新Actor ==========
        # MADDPG Actor更新逻辑：
        # 1. 对每个agent i，使用当前Actor μ_i计算新动作 a_i = μ_i(s_i)
        # 2. 构建所有agent的动作: [a_1, a_2, ..., a_i(new), ..., a_N]
        #    其中其他agent使用batch中的旧动作，当前agent使用新动作
        # 3. 使用Critic计算Q值: Q(s, [a_1, a_2, ..., a_i(new), ..., a_N])
        # 4. 最大化Q值（即最小化 -Q）
        
        actor_losses = []
        per_prov_losses = []
        actor_grad_norms = []  # 收集所有actor的梯度范数

        for p_idx in range(num_agents):
            self.actor_optimizers[p_idx].zero_grad()

            # 获取该省份的样本
            idxs = torch.arange(p_idx, batch_size * num_agents, num_agents, device=self.device)
            obs_p = local_obs_batch_flat[idxs]  # (batch_size, local_obs_dim)
            ids_p = agent_ids_batch_flat[idxs]  # (batch_size, num_agents)

            # 使用当前Actor生成新动作: a_i_new = μ_i(s_i)
            new_action_p = self.actor_list[p_idx](obs_p, ids_p)  # (batch_size, action_dim)

            # 构建所有agent的动作（其他agent使用batch中的动作，当前agent使用新动作）
            # 这是MADDPG的关键：集中式训练，每个agent在更新时考虑所有agent的动作
            all_actions_for_q = []
            for other_p_idx in range(num_agents):
                if other_p_idx == p_idx:
                    all_actions_for_q.append(new_action_p)  # 当前agent使用新动作
                else:
                    other_idxs = torch.arange(other_p_idx, batch_size * num_agents, num_agents, device=self.device)
                    all_actions_for_q.append(actions_batch_flat[other_idxs])  # 其他agent使用batch中的动作
            
            # 拼接: (batch_size * num_agents, action_dim)
            all_actions = torch.cat(all_actions_for_q, dim=0)
            # Reshape: (batch_size, num_agents, action_dim)
            all_actions_reshaped = all_actions.view(batch_size, num_agents, -1)
            # Flatten: (batch_size, num_agents * action_dim) - 用于Critic输入
            all_actions_flat = all_actions_reshaped.view(batch_size, -1)

            # Actor损失：最大化Q值（即最小化-Q）
            # L_actor = -E[Q(s, [a_1, ..., a_i(new), ..., a_N])]
            q_value = self.centralized_critic(global_obs_batch, all_actions_flat)
            actor_loss = -q_value.mean()
            
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_list[p_idx].parameters(), 1.0)
            self.actor_optimizers[p_idx].step()

            actor_losses.append(actor_loss.item())
            per_prov_losses.append(actor_loss.item())
            actor_grad_norms.append(actor_grad_norm.item())  # ✅ 转换为item()存储

        # 软更新目标网络
        self._soft_update_target_networks()

        # 记录统计信息
        avg_actor_loss = np.mean(actor_losses)
        self.training_stats['actor_losses'].append(avg_actor_loss)
        self.training_stats['actor_losses_per_province'].append(per_prov_losses.copy())
        self.training_stats['critic_losses'].append(critic_loss.item())
        self.training_stats['q_values'].append(current_q.mean().item())
        self.training_stats['exploration_noises'].append(self.exploration_noise)
        self.training_stats['gradient_norms'].append({
            'actor': np.mean(actor_grad_norms) if len(actor_grad_norms) > 0 else 0.0,  # ✅ 使用收集的列表
            'critic': critic_grad_norm.item()
        })

        # 衰减探索噪声
        self.exploration_noise = max(self.min_noise, self.exploration_noise * self.noise_decay)

        if self.logger:
            self.logger.log(
                f"Episode {episode} MADDPG更新: "
                f"Actor loss = {avg_actor_loss:.6f}, "
                f"Critic loss = {critic_loss.item():.6f}, "
                f"Q = {current_q.mean().item():.4f}, "
                f"Target Q = {target_q.mean().item():.4f}, "
                f"Actor grad norm = {np.mean(actor_grad_norms) if len(actor_grad_norms) > 0 else 0.0:.4f}, "
                f"Critic grad norm = {critic_grad_norm.item():.4f}")

    def _soft_update_target_networks(self):
        """软更新目标网络"""
        # 更新Critic目标网络
        for target_param, param in zip(self.centralized_critic_target.parameters(), 
                                       self.centralized_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 更新Actor目标网络
        for actor_target, actor in zip(self.actor_target_list, self.actor_list):
            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger

    def get_province_policy(self, province_idx):
        """获取单个省份的策略"""
        return copy.deepcopy(self.actor_list[province_idx].state_dict())

    def set_fixed_policy(self, province_idx, policy_state_dict, actions_sequence=None):
        """保存固定策略"""
        if actions_sequence is not None:
            self.fixed_actions_sequences[province_idx] = copy.deepcopy(actions_sequence)

        if policy_state_dict is not None:
            policy_snapshot = copy.deepcopy(policy_state_dict)
            self.fixed_policies[province_idx] = policy_snapshot
            fixed_actor = DeterministicActor(self.local_obs_dim, self.num_agents, self.action_dim).to(self.device)
            fixed_actor.load_state_dict(policy_snapshot)
            fixed_actor.eval()
            self.fixed_actor_networks[province_idx] = fixed_actor

    def _get_fixed_actor(self, province_idx):
        """获取固定省份的独立Actor"""
        if province_idx in self.fixed_actor_networks:
            return self.fixed_actor_networks[province_idx]
        policy_state = self.fixed_policies.get(province_idx)
        if policy_state is None:
            return None
        fixed_actor = DeterministicActor(self.local_obs_dim, self.num_agents, self.action_dim).to(self.device)
        fixed_actor.load_state_dict(policy_state)
        fixed_actor.eval()
        self.fixed_actor_networks[province_idx] = fixed_actor
        return fixed_actor

    def get_fixed_action(self, province_idx, step):
        """获取固定省份在指定步骤的动作"""
        if province_idx not in self.fixed_actions_sequences:
            return None
        actions_seq = self.fixed_actions_sequences[province_idx]
        if step >= len(actions_seq):
            return actions_seq[-1] if len(actions_seq) > 0 else None
        return actions_seq[step]

    def get_exploration_strategy_status(self):
        """获取探索策略状态"""
        return {
            'exploration_noise': self.exploration_noise,
            'buffer_size': len(self.replay_buffer),
            'fixed_provinces_count': len(self.fixed_policies),
            'current_training_province_idx': self.current_training_province_idx
        }

    def save_best_policy(self, episode_reward, episode):
        """保存最佳策略权重"""
        # MADDPG使用replay buffer，不需要像MAPPO那样保存最佳策略用于恢复
        pass

    def check_and_update_best_policy(self, episode_reward, episode):
        """检查并更新最佳策略"""
        return False


# ==================== 训练函数 ====================

def train_maddpg(
        enable_realtime_console=True,
        random_seed=42,
        province_training_config=None,
        max_total_episodes=None,
        fine_tune_episodes=25,
        scenario_id=None,
):
    """训练MADDPG算法"""
    set_seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print(f"开始训练MADDPG算法（随机种子={random_seed}）...")
    if scenario_id:
        print(f"📋 情景ID: {scenario_id}")

    if scenario_id:
        log_dir = f"./log14/{scenario_id}"
    else:
        log_dir = "./log14"
    detailed_logger = DetailedLogger(log_dir=log_dir, console_output=enable_realtime_console)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if scenario_id:
        tensorboard_dir = f'./logs/{scenario_id}/tensorboard_{timestamp}'
    else:
        tensorboard_dir = f'./logs/tensorboard_{timestamp}'
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=tensorboard_dir, flush_secs=1)
    print(f"📊 TensorBoard日志保存到: {tensorboard_dir}")

    required_files = [
        ("./models_unet", "U-Net模型目录"),
        ("./conc/base/base.csv", "基准浓度数据"),
        ("./conc/clean/clean.csv", "清洁情景数据"),
        ("./prov_grid_map/36kmprov.csv", "省份映射数据"),
        ("./other data/cost.csv", "成本系数数据"),
        (TRANSPORT_MATRIX_PATH, "省际传输矩阵（公平性奖励）"),
        ("./input_emi/base", "基准排放数据目录")
    ]

    for file_path, description in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 找不到{description}: {file_path}")
            detailed_logger.close()
            writer.close()
            return None, None

    try:
        env = RSMEmissionEnv(
            model_path="./models_unet",
            scaler_path="./models_unet",
            base_conc_path="./conc/base/base.csv",
            clean_conc_path="./conc/clean/clean.csv",
            province_map_path="./prov_grid_map/36kmprov.csv",
            base_emission_path="./input_emi/base",
            cost_data_path="./other data/cost.csv",
            transport_matrix_path=TRANSPORT_MATRIX_PATH,
            fairness_weight=FAIRNESS_WEIGHT,
            fairness_metric=FAIRNESS_METRIC,
            fairness_mode=FAIRNESS_MODE,
            fairness_external_only=FAIRNESS_EXTERNAL_ONLY,
            max_steps=8
        )

        # ============================================================
        # ✅ 使用与v13相同的固定公式计算维度（避免环境返回不一致的问题）
        # ============================================================
        local_obs_dim = env.action_dim + env.num_provinces + env.province_feature_dim
        global_obs_dim = env.num_provinces * env.action_dim + env.num_provinces * 2 + 1

        print(f"\n🎯 观察空间维度（使用v13固定公式计算）:")
        print(f"   local_obs_dim = {local_obs_dim}")
        print(f"   = action_dim({env.action_dim}) + num_provinces({env.num_provinces}) + province_feature_dim({env.province_feature_dim})")
        print(f"   global_obs_dim = {global_obs_dim}")
        print(f"   env.action_dim = {env.action_dim}")
        print(f"   env.num_provinces = {env.num_provinces}")

        print(f"\n🔧 创建MADDPG智能体...")
        
        maddpg_agent = MADDPGAgent(
            num_agents=env.num_provinces,
            local_obs_dim=local_obs_dim,
            global_obs_dim=global_obs_dim,
            action_dim=env.action_dim,
            lr_actor=1e-5,   # ✅ 降低Actor学习率
            lr_critic=1e-4,  # ✅ 降低Critic学习率
            gamma=0.95,      # ✅ 降低折扣因子
            tau=0.001,       # ✅ 更小的软更新系数
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # ✅ 验证网络维度
        print(f"\n🔍 验证网络维度:")
        print(f"   maddpg_agent.local_obs_dim = {maddpg_agent.local_obs_dim}")
        first_layer = maddpg_agent.actor_list[0].obs_branch[0]
        print(f"   Actor[0] 第一层输入维度: {first_layer.in_features}")
        print(f"✅ 网络创建完成！\n")

        maddpg_agent.set_logger(detailed_logger)
        maddpg_agent.initialize_province_monitors(env.province_names)
        
        print(f"\n{'=' * 80}")
        print(f"🏗️  MADDPG模型架构:")
        print(f"{'=' * 80}")
        print(f"  📊 Actor网络: 确定性策略（Deterministic Policy）")
        print(f"     - 每个省份独立的Actor网络: {len(maddpg_agent.actor_list)} 个")
        print(f"     - 输出: 确定性动作（非概率分布）")
        print(f"  📊 Critic网络: 集中式Q函数")
        print(f"     - 输入: 全局状态 + 所有agent动作")
        print(f"  🔄 目标网络: Actor + Critic目标网络")
        print(f"     - 软更新系数 tau: {maddpg_agent.tau}")
        print(f"  💾 经验回放: Replay Buffer")
        print(f"     - 容量: 100000")
        print(f"     - Batch size: {maddpg_agent.batch_size}")
        print(f"  🎲 探索: Ornstein-Uhlenbeck噪声")
        print(f"     - 初始噪声: {maddpg_agent.exploration_noise}")
        print(f"{'=' * 80}\n")

        # 省份训练配置（与v13相同）
        if province_training_config is None:
            province_training_config = [
                {'province_idx': 0, 'province_name': env.province_names[0], 'episodes': 1200}
            ]
            for idx in range(1, env.num_provinces):
                province_training_config.append({
                    'province_idx': idx,

                    'province_name': env.province_names[idx],
                    'episodes': 60
                })

        configured_indices = [cfg['province_idx'] for cfg in province_training_config]
        if len(set(configured_indices)) != len(configured_indices):
            raise ValueError("❌ 错误：省份配置中有重复的province_idx")

        for cfg in province_training_config:
            if 'province_name' not in cfg or cfg['province_name'] is None:
                cfg['province_name'] = env.province_names[cfg['province_idx']]

        total_priority_episodes = sum(cfg['episodes'] for cfg in province_training_config)
        if max_total_episodes is not None:
            max_episodes = max_total_episodes
        else:
            max_episodes = total_priority_episodes + fine_tune_episodes

        province_config_map = {cfg['province_idx']: cfg for cfg in province_training_config}
        province_order = [cfg['province_idx'] for cfg in province_training_config]

        print(f"\n{'=' * 80}")
        print(f"🎯 优先级训练计划:")
        print(f"  总训练轮次: {max_episodes}")
        print(f"{'=' * 80}\n")

        if scenario_id:
            result_dir = f"./result14/{scenario_id}"
        else:
            result_dir = "./result14"
        os.makedirs(result_dir, exist_ok=True)

        episode_rewards = []
        best_reward = float('-inf')
        
        current_province_order_idx = 0
        current_province_idx = province_order[0] if len(province_order) > 0 else None
        province_episode_count = 0
        is_fine_tuning = False

        # 记录训练开始信息
        try:
            detailed_logger.log(f"\n{'=' * 80}")
            detailed_logger.log(f"🚀 开始MADDPG训练")
            detailed_logger.log(f"   总回合数: {max_episodes}")
            detailed_logger.log(f"   优先级训练回合数: {total_priority_episodes}")
            detailed_logger.log(f"   微调回合数: {max_episodes - total_priority_episodes}")
            detailed_logger.log(f"   TensorBoard目录: {tensorboard_dir}")
            detailed_logger.log(f"{'=' * 80}\n")
        except Exception as e:
            print(f"⚠️ 初始日志记录失败: {e}")

        # 训练循环
        for episode in range(max_episodes):
            # 确定当前训练阶段
            if episode < total_priority_episodes and current_province_order_idx < len(province_order):
                current_province_idx = province_order[current_province_order_idx]
                current_config = province_config_map[current_province_idx]
                required_episodes = current_config['episodes']

                if province_episode_count >= required_episodes:
                    monitor = maddpg_agent.province_monitors[current_province_idx]
                    best_policy = monitor.get_best_policy()
                    best_actions_seq = monitor.get_best_actions_sequence()

                    print(f"\n{'=' * 80}")
                    print(f"🔒 固定省份策略: {current_config['province_name']}")
                    print(f"   最佳奖励: {monitor.best_reward:.2f}")
                    print(f"{'=' * 80}\n")

                    if best_actions_seq is not None:
                        maddpg_agent.set_fixed_policy(current_province_idx, best_policy, best_actions_seq)
                    elif best_policy is not None:
                        maddpg_agent.set_fixed_policy(current_province_idx, best_policy, None)

                    maddpg_agent.province_monitors[current_province_idx].fix_policy(episode)

                    current_province_order_idx += 1
                    province_episode_count = 0

                    if current_province_order_idx >= len(province_order):
                        is_fine_tuning = True
                        current_province_idx = None
                    else:
                        current_province_idx = province_order[current_province_order_idx]

                maddpg_agent.current_training_province_idx = current_province_idx
                province_episode_count += 1
            else:
                is_fine_tuning = True
                maddpg_agent.current_training_province_idx = None
                current_province_idx = None

            print(f"\n{'=' * 60}")
            if is_fine_tuning:
                print(f"🔧 微调阶段 - 回合 {episode + 1}/{max_episodes}")
            else:
                print(f"🎯 优先级训练 - 回合 {episode + 1}/{max_episodes}")
                if current_province_idx is not None:
                    print(f"   当前优化省份: {province_config_map[current_province_idx]['province_name']}")
            print(f"{'=' * 60}")

            try:
                env.reset()
                local_observations = env.get_local_observations()
                global_observation = env.get_global_observation()
                agent_ids = env.get_agent_ids()
                
                # ✅ 维度适配：确保观察维度与网络一致
                local_observations = np.array(local_observations)
                global_observation = np.array(global_observation)
                
                # local_obs 维度适配
                if local_observations.shape[-1] != maddpg_agent.local_obs_dim:
                    expected_dim = maddpg_agent.local_obs_dim
                    actual_dim = local_observations.shape[-1]
                    if actual_dim < expected_dim:
                        padding = np.zeros((local_observations.shape[0], expected_dim - actual_dim))
                        local_observations = np.concatenate([local_observations, padding], axis=-1)
                    else:
                        local_observations = local_observations[:, :expected_dim]
                
                # global_obs 维度适配
                if len(global_observation) != maddpg_agent.global_obs_dim:
                    expected_dim = maddpg_agent.global_obs_dim
                    actual_dim = len(global_observation)
                    if actual_dim < expected_dim:
                        global_observation = np.concatenate([global_observation, np.zeros(expected_dim - actual_dim)])
                    else:
                        global_observation = global_observation[:expected_dim]

                episode_reward = 0
                episode_province_actions = {province_idx: [] for province_idx in range(len(env.province_names))}
                episode_experiences = []  # 用于日志记录
                episode_actions = []  # 用于计算动作多样性
                episode_reward_components = []  # 收集组件奖励
                episode_province_rewards = {province_idx: [] for province_idx in range(len(env.province_names))}

                for step in range(env.max_steps):
                    # 选择动作
                    actions = maddpg_agent.select_actions_with_fixed(
                        local_observations, agent_ids, episode=episode, step=step, add_noise=not is_fine_tuning
                    )

                    # 记录动作
                    for province_idx in range(len(env.province_names)):
                        if province_idx < len(actions):
                            episode_province_actions[province_idx].append(actions[province_idx].copy())

                    # 为“省份减排日志”保留 step 前状态
                    if env.current_step > 0 and hasattr(env, 'last_step_pm25') and env.last_step_pm25 is not None:
                        pm25_before = np.array(env.last_step_pm25).copy()
                    else:
                        pm25_before = np.array(
                            [env.province_base_conc.get(prov_name, 50.0) for prov_name in env.province_names],
                            dtype=np.float32
                        )
                    previous_cumulative_factors = env.cumulative_factors.copy() if env.cumulative_factors is not None else None

                    # 执行动作
                    next_local_obs, reward, done, info = env.step(actions)
                    next_global_obs = env.get_global_observation()
                    
                    # ✅ 维度适配：确保next观察维度与网络一致
                    next_local_obs = np.array(next_local_obs)
                    next_global_obs = np.array(next_global_obs)
                    
                    # next_local_obs 维度适配
                    if next_local_obs.shape[-1] != maddpg_agent.local_obs_dim:
                        expected_dim = maddpg_agent.local_obs_dim
                        actual_dim = next_local_obs.shape[-1]
                        if actual_dim < expected_dim:
                            padding = np.zeros((next_local_obs.shape[0], expected_dim - actual_dim))
                            next_local_obs = np.concatenate([next_local_obs, padding], axis=-1)
                        else:
                            next_local_obs = next_local_obs[:, :expected_dim]
                    
                    # next_global_obs 维度适配
                    if len(next_global_obs) != maddpg_agent.global_obs_dim:
                        expected_dim = maddpg_agent.global_obs_dim
                        actual_dim = len(next_global_obs)
                        if actual_dim < expected_dim:
                            next_global_obs = np.concatenate([next_global_obs, np.zeros(expected_dim - actual_dim)])
                        else:
                            next_global_obs = next_global_obs[:expected_dim]

                    episode_reward += reward

                    # 收集经验数据用于日志记录
                    reward_scalar = float(reward) if hasattr(reward, '__len__') and len(reward) == 1 else float(reward) if np.isscalar(reward) else reward
                    episode_experiences.append({
                        'rewards': reward_scalar,
                        'actions': actions.copy(),
                        'step': step + 1,
                        'info': info  # ✅ 保存info，便于DetailedLogger写reward_components
                    })
                    episode_actions.append(actions.copy())
                    
                    # ✅ 收集组件奖励信息
                    if isinstance(info, dict):
                        reward_components = info.get('reward_components', {})
                        if reward_components:
                            episode_reward_components.append({
                                'target': reward_components.get('total_target_reward', 0.0),
                                'cost': reward_components.get('total_cost_penalty', 0.0),
                                'health': reward_components.get('total_health_reward', 0.0),
                                'collaboration': reward_components.get('coordination_reward', 0.0),  # ✅ 修复：使用coordination_reward
                                'fairness': reward_components.get('total_fairness_reward', 0.0)
                            })
                        
                        # 收集省份级别的奖励（如果有）
                        if 'province_rewards' in info:
                            province_rewards = info['province_rewards']
                            for province_idx, province_name in enumerate(env.province_names):
                                if province_name in province_rewards:
                                    prov_reward = province_rewards[province_name]
                                    episode_province_rewards[province_idx].append(prov_reward)

                            # ✅ 记录省份级奖励到日志（包含Fairness）
                            try:
                                detailed_logger.log_province_rewards(episode + 1, province_rewards)
                            except Exception as e:
                                print(f"⚠️ 省份奖励日志记录失败: {e}")

                        # ✅ 记录省份减排信息到 province_reduction_*.log/csv
                        try:
                            predicted_pm25 = info.get('predicted_pm25', None)
                            step_targets, _ = env.get_step_target(step)

                            for province_idx, province_name in enumerate(env.province_names):
                                if province_idx >= len(actions):
                                    continue

                                # 单步减排率（与 v13 保持一致）
                                if step == 0:
                                    single_step_reduction = 1.0 - env.cumulative_factors[province_idx]
                                else:
                                    if previous_cumulative_factors is not None and province_idx < len(previous_cumulative_factors):
                                        prev_cumulative = previous_cumulative_factors[province_idx]
                                        curr_cumulative = env.cumulative_factors[province_idx]
                                        if np.any(prev_cumulative > 1e-8):
                                            single_step_reduction = (prev_cumulative - curr_cumulative) / prev_cumulative
                                        else:
                                            single_step_reduction = 1.0 - curr_cumulative
                                    else:
                                        single_step_reduction = 1.0 - env.cumulative_factors[province_idx]

                                cumulative_reduction = 1.0 - env.cumulative_factors[province_idx]
                                pm25_after_val = (
                                    predicted_pm25[province_idx]
                                    if predicted_pm25 is not None and province_idx < len(predicted_pm25)
                                    else env.province_base_conc.get(province_name, 50.0)
                                )

                                province_data = {
                                    'province_id': province_idx,
                                    'province_name': province_name,
                                    'single_step_reduction': single_step_reduction,
                                    'cumulative_reduction': cumulative_reduction,
                                    'action_value': actions[province_idx],
                                    'pm25_before': pm25_before[province_idx] if province_idx < len(pm25_before) else env.province_base_conc.get(province_name, 50.0),
                                    'pm25_after': pm25_after_val,
                                    'pm25_target': step_targets.get(province_name, env.province_base_conc.get(province_name, 50.0) * 0.5)
                                }
                                detailed_logger.log_province_reduction(episode + 1, step + 1, province_data)
                        except Exception as e:
                            print(f"⚠️ 省份减排日志记录失败: {e}")

                        # ✅ 记录step级奖励分解到日志与CSV（reward_data_*.csv）
                        try:
                            reward_change = 0.0
                            if len(episode_experiences) > 0:
                                prev_reward = episode_experiences[-1].get('rewards', 0.0)
                                reward_change = float(reward_scalar) - float(prev_reward)
                            detailed_logger.log_reward_analysis(
                                episode + 1, step + 1,
                                float(reward_scalar),
                                0.0,  # diversity_reward（MADDPG不使用）
                                float(reward_scalar),  # enhanced_reward（与总奖励一致）
                                float(reward_change),
                                info,
                                actions,
                                env.cumulative_factors
                            )
                        except Exception as e:
                            print(f"⚠️ reward_data日志记录失败: {e}")

                    # 存储transition（只在非fine-tuning阶段）
                    # ✅ MADDPG是off-policy算法，使用每小步的即时奖励（不是累积奖励）
                    # 每个transition: (s, a, r, s', done)，其中r是当前小步的即时奖励
                    # 更新时使用Bellman方程: Q(s,a) = r + γ * Q(s',a') * (1-done)
                    if not is_fine_tuning:
                        maddpg_agent.store_transition(
                            local_observations, global_observation, agent_ids, actions, reward_scalar,  # 即时奖励
                            next_local_obs, next_global_obs, done
                        )

                    local_observations = next_local_obs
                    global_observation = next_global_obs

                    if done:
                        break

                episode_rewards.append(episode_reward)
                
                # 计算动作多样性
                if len(episode_actions) > 0:
                    all_actions = np.array(episode_actions)
                    episode_diversity = float(np.std(all_actions))
                else:
                    episode_diversity = 0.0
                
                # 计算组件奖励统计
                if len(episode_reward_components) > 0:
                    total_target = sum(c.get('target', 0.0) for c in episode_reward_components)
                    total_cost = sum(c.get('cost', 0.0) for c in episode_reward_components)
                    total_health = sum(c.get('health', 0.0) for c in episode_reward_components)
                    total_collaboration = sum(c.get('collaboration', 0.0) for c in episode_reward_components)
                    total_fairness = sum(c.get('fairness', 0.0) for c in episode_reward_components)
                else:
                    total_target = 0.0
                    total_cost = 0.0
                    total_health = 0.0
                    total_collaboration = 0.0
                    total_fairness = 0.0

                # ✅ 每轮都输出详细的训练信息
                print(f"\n{'=' * 80}")
                print(f"📊 回合 {episode + 1}/{max_episodes} 训练信息")
                print(f"{'=' * 80}")
                print(f"🎯 总奖励: {episode_reward:.4f}")
                print(f"📈 组件奖励分解:")
                print(f"   - 目标奖励: {total_target:.4f}")
                print(f"   - 成本惩罚: {total_cost:.4f}")
                print(f"   - 健康效益: {total_health:.4f}")
                print(f"   - 协作奖励: {total_collaboration:.4f}")
                print(f"   - 公平性奖励: {total_fairness:.4f}")
                print(f"🔍 训练统计:")
                print(f"   - 动作多样性: {episode_diversity:.4f}")
                print(f"   - 探索噪声: {maddpg_agent.exploration_noise:.4f}")
                print(f"   - Replay Buffer大小: {len(maddpg_agent.replay_buffer)}")
                
                # 显示损失信息（如果已更新）
                if hasattr(maddpg_agent, 'training_stats') and maddpg_agent.training_stats:
                    if len(maddpg_agent.training_stats['actor_losses']) > 0:
                        avg_actor_loss = maddpg_agent.training_stats['actor_losses'][-1]
                        print(f"   - Actor损失: {avg_actor_loss:.6f}")
                    if len(maddpg_agent.training_stats['critic_losses']) > 0:
                        critic_loss = maddpg_agent.training_stats['critic_losses'][-1]
                        print(f"   - Critic损失: {critic_loss:.6f}")
                    if len(maddpg_agent.training_stats['q_values']) > 0:
                        q_value = maddpg_agent.training_stats['q_values'][-1]
                        print(f"   - Q值: {q_value:.4f}")
                    if len(maddpg_agent.training_stats['gradient_norms']) > 0:
                        grad_norms = maddpg_agent.training_stats['gradient_norms'][-1]
                        print(f"   - Actor梯度范数: {grad_norms.get('actor', 0.0):.4f}")
                        print(f"   - Critic梯度范数: {grad_norms.get('critic', 0.0):.4f}")
                
                if len(episode_rewards) >= 10:
                    print(f"   - 10回合平均奖励: {np.mean(episode_rewards[-10:]):.4f}")
                if len(episode_rewards) >= 100:
                    print(f"   - 100回合平均奖励: {np.mean(episode_rewards[-100:]):.4f}")
                print(f"{'=' * 80}\n")

                # 记录回合总结日志
                try:
                    detailed_logger.log_episode_summary(
                        episode + 1, episode_reward, episode_experiences, episode_actions, episode_diversity
                    )
                except Exception as e:
                    print(f"⚠️ 日志记录失败: {e}")

                # 更新省份监控器
                for province_idx in range(len(env.province_names)):
                    if province_idx in maddpg_agent.province_monitors:
                        province_actions_seq = episode_province_actions.get(province_idx, [])
                        if province_idx in maddpg_agent.fixed_policies:
                            current_policy = copy.deepcopy(maddpg_agent.fixed_policies[province_idx])
                        else:
                            current_policy = maddpg_agent.get_province_policy(province_idx)

                        # ✅ 使用该省份自己的episode累计奖励（优先total，已包含公平性/协作/竞争等）
                        prov_steps = episode_province_rewards.get(province_idx, [])
                        province_reward = 0.0
                        if prov_steps:
                            for r in prov_steps:
                                if isinstance(r, dict):
                                    province_reward += float(r.get('total', 0.0))
                                else:
                                    try:
                                        province_reward += float(r)
                                    except Exception:
                                        pass
                        maddpg_agent.province_monitors[province_idx].update(
                            episode + 1, province_reward, current_policy, province_actions_seq
                        )

                # 更新策略（从replay buffer学习）
                if not is_fine_tuning and len(maddpg_agent.replay_buffer) >= maddpg_agent.min_buffer_size:
                    # MADDPG在每个episode结束后更新多次
                    for _ in range(10):  # 每个episode更新10次
                        maddpg_agent.update(episode + 1)
                    
                    # 记录训练诊断信息（MADDPG专用，不包含entropy）
                    try:
                        if hasattr(maddpg_agent, 'training_stats') and maddpg_agent.training_stats:
                            # ✅ MADDPG是确定性策略，没有entropy，需要自定义日志记录
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            log_message = f"[{timestamp}] Episode {episode + 1} MADDPG Training Diagnostics\n"
                            
                            if len(maddpg_agent.training_stats.get('actor_losses', [])) > 0:
                                actor_loss = maddpg_agent.training_stats['actor_losses'][-1]
                                log_message += f"  Actor损失: {actor_loss:.6f}\n"
                            
                            if len(maddpg_agent.training_stats.get('critic_losses', [])) > 0:
                                critic_loss = maddpg_agent.training_stats['critic_losses'][-1]
                                log_message += f"  Critic损失: {critic_loss:.6f}\n"
                            
                            if len(maddpg_agent.training_stats.get('q_values', [])) > 0:
                                q_value = maddpg_agent.training_stats['q_values'][-1]
                                log_message += f"  Q值: {q_value:.4f}\n"
                            
                            if len(maddpg_agent.training_stats.get('gradient_norms', [])) > 0:
                                grad_norms = maddpg_agent.training_stats['gradient_norms'][-1]
                                log_message += f"  梯度范数: Actor={grad_norms.get('actor', 0.0):.6f}, Critic={grad_norms.get('critic', 0.0):.6f}\n"
                            
                            log_message += f"  探索噪声: {maddpg_agent.exploration_noise:.4f}\n"
                            log_message += "\n"
                            
                            # 使用logger的通用log方法
                            detailed_logger.log(log_message)
                    except Exception as e:
                        print(f"⚠️ 训练诊断日志记录失败: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # ✅ 记录组件奖励到日志
                    try:
                        if len(episode_reward_components) > 0:
                            # 计算平均组件奖励
                            avg_target = total_target / len(episode_reward_components) if len(episode_reward_components) > 0 else 0.0
                            avg_cost = total_cost / len(episode_reward_components) if len(episode_reward_components) > 0 else 0.0
                            avg_health = total_health / len(episode_reward_components) if len(episode_reward_components) > 0 else 0.0
                            avg_collaboration = total_collaboration / len(episode_reward_components) if len(episode_reward_components) > 0 else 0.0
                            
                            detailed_logger.log(
                                f"回合 {episode + 1} 组件奖励: "
                                f"目标={avg_target:.4f}, 成本={avg_cost:.4f}, "
                                f"健康={avg_health:.4f}, 协作={avg_collaboration:.4f}"
                            )
                    except Exception as e:
                        print(f"⚠️ 组件奖励日志记录失败: {e}")

                # 保存最佳模型
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    print(f"🏆 新最佳奖励: {best_reward:.4f}")
                    try:
                        torch.save(
                            {f"actor_{i}": actor.state_dict() for i, actor in enumerate(maddpg_agent.actor_list)},
                            os.path.join(result_dir, "best_maddpg_actors.pth")
                        )
                        torch.save(maddpg_agent.centralized_critic.state_dict(),
                                 os.path.join(result_dir, "best_maddpg_critic.pth"))
                        print("✅ 保存最佳模型")
                    except Exception as e:
                        print(f"❌ 保存模型失败: {e}")

                # ✅ TensorBoard实时写入（每轮都更新）
                if writer is not None:
                    try:
                        # 基础奖励指标
                        writer.add_scalar('Reward/Episode', float(episode_reward), episode + 1)
                        writer.add_scalar('Exploration/Noise', float(maddpg_agent.exploration_noise), episode + 1)
                        writer.add_scalar('Buffer/Size', len(maddpg_agent.replay_buffer), episode + 1)
                        writer.add_scalar('Diversity/Action', float(episode_diversity), episode + 1)

                        # 移动平均奖励
                        if len(episode_rewards) >= 10:
                            writer.add_scalar('Reward/MovingAvg_10', float(np.mean(episode_rewards[-10:])), episode + 1)
                        if len(episode_rewards) >= 100:
                            writer.add_scalar('Reward/MovingAvg_100', float(np.mean(episode_rewards[-100:])), episode + 1)

                        # ✅ 组件奖励
                        writer.add_scalar('RewardComponents/Target', float(total_target), episode + 1)
                        writer.add_scalar('RewardComponents/Cost', float(total_cost), episode + 1)
                        writer.add_scalar('RewardComponents/Health', float(total_health), episode + 1)
                        writer.add_scalar('RewardComponents/Collaboration', float(total_collaboration), episode + 1)
                        writer.add_scalar('RewardComponents/Fairness', float(total_fairness), episode + 1)

                        # ✅ 损失和Q值（实时记录）
                        if hasattr(maddpg_agent, 'training_stats') and maddpg_agent.training_stats:
                            if 'actor_losses' in maddpg_agent.training_stats and len(maddpg_agent.training_stats['actor_losses']) > 0:
                                writer.add_scalar('Loss/Actor', float(maddpg_agent.training_stats['actor_losses'][-1]), episode + 1)
                                # 记录每个省份的Actor损失
                                if 'actor_losses_per_province' in maddpg_agent.training_stats and len(maddpg_agent.training_stats['actor_losses_per_province']) > 0:
                                    per_prov_losses = maddpg_agent.training_stats['actor_losses_per_province'][-1]
                                    for prov_idx, prov_loss in enumerate(per_prov_losses):
                                        writer.add_scalar(f'Loss/Actor_Province_{prov_idx}', float(prov_loss), episode + 1)
                            
                            if 'critic_losses' in maddpg_agent.training_stats and len(maddpg_agent.training_stats['critic_losses']) > 0:
                                writer.add_scalar('Loss/Critic', float(maddpg_agent.training_stats['critic_losses'][-1]), episode + 1)
                            
                            if 'q_values' in maddpg_agent.training_stats and len(maddpg_agent.training_stats['q_values']) > 0:
                                writer.add_scalar('Value/Q', float(maddpg_agent.training_stats['q_values'][-1]), episode + 1)
                            
                            # ✅ 梯度范数
                            if 'gradient_norms' in maddpg_agent.training_stats and len(maddpg_agent.training_stats['gradient_norms']) > 0:
                                grad_norms = maddpg_agent.training_stats['gradient_norms'][-1]
                                if 'actor' in grad_norms:
                                    writer.add_scalar('Gradient/Actor', float(grad_norms['actor']), episode + 1)
                                if 'critic' in grad_norms:
                                    writer.add_scalar('Gradient/Critic', float(grad_norms['critic']), episode + 1)

                        # ✅ 实时flush，确保TensorBoard立即更新
                        writer.flush()
                        
                    except Exception as e:
                        print(f"❌ TensorBoard写入失败: {e}")
                        import traceback
                        traceback.print_exc()

            except Exception as e:
                print(f"❌ 训练回合 {episode + 1} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("🎉 MADDPG训练完成！")

        try:
            writer.flush()
            writer.close()
        except Exception as e:
            print(f"❌ TensorBoard关闭失败: {e}")

        detailed_logger.close()

        return maddpg_agent, episode_rewards, result_dir

    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        try:
            writer.flush()
            writer.close()
        except:
            pass
        detailed_logger.close()
        return None, None, None



if __name__ == "__main__":
    print("=" * 80)
    print("Starting MADDPG training")
    print("Version: v14.12 - reward normalization updated for the current reward range")
    print("=" * 80)

    runtime_config = copy.deepcopy(V14_MAIN_CONFIG)

    try:
        maddpg_agent, rewards, result_dir = train_maddpg(
            enable_realtime_console=runtime_config["enable_realtime_console"],
            random_seed=runtime_config["single_scenario_seed"],
            province_training_config=copy.deepcopy(runtime_config["province_training_config"]),
            max_total_episodes=runtime_config["max_total_episodes"],
            fine_tune_episodes=runtime_config["fine_tune_episodes"],
        )

        if maddpg_agent is not None:
            print("\nMADDPG training finished")
            print("=" * 80)
            if result_dir:
                print(f"Results are saved in {result_dir}/")
            print("Logs are saved in ./log14/")
            print("TensorBoard logs are saved in ./logs/")

            if rewards:
                print("\nTraining summary:")
                print(f"  Total episodes: {len(rewards)}")
                print(f"  Best reward: {max(rewards):.4f}")
                print(f"  Mean reward: {np.mean(rewards):.4f}")
                print(f"  Final reward: {rewards[-1]:.4f}")
        else:
            print("\nMADDPG training failed")
    except Exception as e:
        print(f"\nProgram execution failed: {e}")
        import traceback
        traceback.print_exc()
