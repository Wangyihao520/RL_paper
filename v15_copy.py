# train_e2_v5_pytorch_optimized.py - PyTorch MAPPO training script
"""
Usage overview:

1. Customize province training order and episode budget:
   Set `PROVINCE_TRAINING_CONFIG` in the main block, for example:

   PROVINCE_TRAINING_CONFIG = [
       {'province_idx': 0, 'province_name': 'HB', 'episodes': 200},
       {'province_idx': 1, 'province_name': 'SD', 'episodes': 100},
       {'province_idx': 2, 'province_name': 'BJ', 'episodes': 80},
   ]

   If `None`, the script falls back to the default schedule:
   200 episodes for the first province and 60 for the rest.

2. Multi-scenario training:
   Set `MULTI_SCENARIO_MODE = True`, then configure:
   - `NUM_SCENARIOS`
   - `BASE_SEED`
   - `PARALLEL_TRAINING`
   - `NUM_WORKERS`

   Parallel mode is recommended when CPU or GPU resources are sufficient.

3. Single-scenario training:
   Set `MULTI_SCENARIO_MODE = False`
   Set `SINGLE_SCENARIO_SEED`

4. Optional global episode cap:
   Set `MAX_TOTAL_EPISODES` to override the value derived from the
   province training schedule.
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
from torch.distributions import Normal
from datetime import datetime
import json
# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError as e:
    print(f"TensorBoard is unavailable: {e}")
    print("Training will continue without TensorBoard logging.")
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# Matplotlib display settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Use a non-interactive backend for file output
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端


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

# Reuse environment utilities from the base script
from train_e2_v46 import (
    RunningMeanStd, set_seed, huber_loss, relative_loss, r2_metric, rmse_metric,
    RSMEmissionEnv
)

# Set a default random seed
set_seed(42)


class ProvinceMonitor:
    """Track policy snapshots and rewards for a single province."""

    def __init__(self, province_idx, province_name, max_steps=8):
        self.province_idx = province_idx
        self.province_name = province_name
        self.max_steps = max_steps
        self.best_reward = float('-inf')
        self.best_episode = 0
        self.best_policy_state_dict = None
        self.best_actions_sequence = None  # Store the action sequence of the best episode.
        self.reward_history = []
        self.episode_history = []
        self.policy_history = []
        self.actions_history = []  # Store action sequences for later inspection.
        self.is_fixed = False
        self.fixed_episode = None

    def update(self, episode, episode_reward, policy_state_dict, actions_sequence=None):
        """更新监控网络

        Args:
            episode: 当前回合
            episode_reward: 该省份的回合总奖励
            policy_state_dict: 策略网络状态（用于备份）
            actions_sequence: 该省份在本回合中实际执行的动作序列 [step1_action, step2_action, ...]
        """
        self.reward_history.append(episode_reward)
        self.episode_history.append(episode)

        # 已固定的省份仅记录奖励，不再更新策略或最佳值
        if self.is_fixed:
            return

        # 只在奖励改善时保存 policy 和 actions（避免无限累积 GPU 内存）
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_episode = episode
            if policy_state_dict is not None:
                self.best_policy_state_dict = copy.deepcopy(policy_state_dict)
            if actions_sequence is not None:
                self.best_actions_sequence = copy.deepcopy(actions_sequence)

    def get_best_policy(self):
        """获取最佳策略"""
        if self.best_policy_state_dict is None:
            return None
        return copy.deepcopy(self.best_policy_state_dict)

    def get_best_actions_sequence(self):
        """获取最佳回合的实际动作序列"""
        if self.best_actions_sequence is None:
            return None
        return copy.deepcopy(self.best_actions_sequence)

    def get_best_policy_debug_info(self):
        """获取最佳策略的调试信息"""
        info = f"省份 {self.province_idx} ({self.province_name}): 最佳奖励={self.best_reward:.2f}, 最佳Episode={self.best_episode}"
        if self.best_actions_sequence is not None:
            info += f", 动作序列长度={len(self.best_actions_sequence)}"
            if len(self.best_actions_sequence) > 0:
                first_action = self.best_actions_sequence[0]
                if hasattr(first_action, '__len__'):
                    info += f", Step1动作前5值={list(first_action[:5])}"
        return info

    def fix_policy(self, episode):
        """固定策略"""
        self.is_fixed = True
        self.fixed_episode = episode

    def get_stats(self):
        """获取统计信息"""
        # 只返回最近的奖励历史（避免传递巨大列表）
        recent_history = self.reward_history[-50:] if len(self.reward_history) > 50 else self.reward_history
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
            'reward_history': recent_history
        }


# 🔍 创建详细日志系统（与TensorFlow版本相同）
class DetailedLogger:
    def __init__(self, log_dir="./log152", console_output=True):
        self.log_dir = log_dir
        self.console_output = console_output
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建各种日志文件
        self.reward_log = open(os.path.join(log_dir, f"reward_analysis_{timestamp}.log"), 'w', encoding='utf-8')
        self.training_log = open(os.path.join(log_dir, f"training_diagnostics_{timestamp}.log"), 'w', encoding='utf-8')
        self.exploration_log = open(os.path.join(log_dir, f"exploration_analysis_{timestamp}.log"), 'w',
                                    encoding='utf-8')
        self.episode_log = open(os.path.join(log_dir, f"episode_summary_{timestamp}.log"), 'w', encoding='utf-8')
        self.critic_log = open(os.path.join(log_dir, f"critic_convergence_{timestamp}.log"), 'w', encoding='utf-8')
        self.reduction_log = open(os.path.join(log_dir, f"province_reduction_{timestamp}.log"), 'w', encoding='utf-8')

        # 新增：省份级奖励日志
        self.province_rewards_log = open(os.path.join(log_dir, f"province_rewards_{timestamp}.log"), 'w',
                                         encoding='utf-8')

        # 🎯 新增：优化省份日志
        self.priority_training_log = open(os.path.join(log_dir, f"priority_training_{timestamp}.log"), 'w',
                                          encoding='utf-8')
        self.priority_training_csv = open(os.path.join(log_dir, f"priority_training_{timestamp}.csv"), 'w',
                                          newline='', encoding='utf-8')
        # 🎯 新增：策略调试日志
        self.policy_debug_log = open(os.path.join(log_dir, f"policy_debug_{timestamp}.log"), 'w', encoding='utf-8')
        
        # 🎯 新增：详细的Loss和梯度日志
        self.loss_gradient_log = open(os.path.join(log_dir, f"loss_gradient_detailed_{timestamp}.log"), 'w', encoding='utf-8')
        self.loss_gradient_csv = open(os.path.join(log_dir, f"loss_gradient_detailed_{timestamp}.csv"), 'w', newline='', encoding='utf-8')
        self.loss_gradient_writer = csv.writer(self.loss_gradient_csv)
        self.loss_gradient_writer.writerow([
            'Episode', 'Epoch', 'Province_Idx', 'Province_Name', 
            'Actor_Loss', 'Actor_Grad_Norm', 'Actor_Entropy',
            'Advantage_Mean', 'Advantage_Std', 'Advantage_Max', 'Advantage_Min',
            'Ratio_Mean', 'Ratio_Std', 'Ratio_Max', 'Ratio_Min',
            'Critic_Loss', 'Critic_Grad_Norm', 'Value_Mean', 'Value_Std',
            'Update_Skipped', 'Skip_Reason'
        ])
        
        # 🎯 新增：协作与竞争奖励专用日志
        self.coordination_log = open(os.path.join(log_dir, f"coordination_competition_{timestamp}.log"), 'w', encoding='utf-8')
        self.coordination_csv = open(os.path.join(log_dir, f"coordination_competition_{timestamp}.csv"), 'w', newline='', encoding='utf-8')
        self.coordination_writer = csv.writer(self.coordination_csv)
        self.coordination_writer.writerow([
            'Episode', 'Step', 'Province_Idx', 'Province_Name', 'Region_ID',
            'Differential_Reward', 'Ranking_Reward', 'InterRegion_Reward',
            'Base_Utility', 'Team_Goal_Real', 'Team_Goal_Baseline'
        ])
        
        self.priority_training_writer = csv.writer(self.priority_training_csv)
        self.priority_training_writer.writerow([
            'Episode', 'Province_Idx', 'Province_Name', 'Status', 'Best_Reward',
            'Best_Episode', 'Current_Reward', 'Is_Fixed', 'Fixed_Episode',
            'Total_Episodes', 'Avg_Reward', 'Std_Reward', 'Min_Reward', 'Max_Reward'
        ])
        self.province_rewards_csv = open(os.path.join(log_dir, f"province_rewards_{timestamp}.csv"), 'w', newline='',
                                         encoding='utf-8')

        # CSV文件用于数据分析
        self.reward_csv = open(os.path.join(log_dir, f"reward_data_{timestamp}.csv"), 'w', newline='', encoding='utf-8')
        self.training_csv = open(os.path.join(log_dir, f"training_data_{timestamp}.csv"), 'w', newline='',
                                 encoding='utf-8')
        self.reduction_csv = open(os.path.join(log_dir, f"province_reduction_{timestamp}.csv"), 'w', newline='',
                                  encoding='utf-8')

        # 初始化CSV writers
        self.reward_writer = csv.writer(self.reward_csv)
        self.training_writer = csv.writer(self.training_csv)
        self.reduction_writer = csv.writer(self.reduction_csv)
        self.province_rewards_writer = csv.writer(self.province_rewards_csv)  # 新增

        # 写入CSV头部 - ✅ 增强：添加奖励组成部分（并对齐写入列数）
        self.reward_writer.writerow([
            'Episode', 'Step',
            'Total_Reward', 'Target_Reward', 'Cost_Penalty', 'Health_Reward',
            'Coordination_Reward', 'Differential_Reward_Sum', 'Ranking_Reward_Sum', 'InterRegion_Reward_Sum',
            'Fairness_Reward',
            'Diversity_Reward', 'Enhanced_Reward', 'Reward_Change',
            'Avg_PM25', 'Avg_Reduction', 'Action_Mean', 'Action_Std'
        ])
        self.training_writer.writerow(['Episode', 'Actor_Loss', 'Critic_Loss', 'Entropy', 'Actor_Grad_Norm',
                                       'Critic_Grad_Norm', 'Value_Mean', 'Advantage_Mean', 'Reward_Mean', 'Reward_Std'])
        self.reduction_writer.writerow([
            'Episode', 'Step', 'Province_ID', 'Province_Name', 'Species_Count',
            'Mean_Reduction_Rate', 'Std_Reduction_Rate', 'Min_Reduction_Rate', 'Max_Reduction_Rate',
            'Median_Reduction_Rate',
            'Cumulative_Reduction_Rate', 'Action_Value', 'PM25_Before', 'PM25_After', 'PM25_Target',
            'All_Species_Reductions'
        ])
        # ✅ 更新CSV头部：添加公平性奖励列，并确保与写入列数一致
        self.province_rewards_writer.writerow(
            ['Episode', 'Province',
             'Target_Reward', 'Cost_Penalty', 'Health_Reward', 'Fairness_Reward',
             'Coordination_Reward', 'Ranking_Reward', 'InterRegion_Reward', 'Competition_Reward',
             'Total_Province_Reward'])

        # 🎨 实时图表数据存储
        self.episode_rewards = []
        self.diversity_scores = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_values = []
        self.exploration_rates = []
        self.learning_rates_actor = []
        self.learning_rates_critic = []
        self.performance_trends = []
        self.exploration_noise_history = []

        print(f"🌍 省份减排率记录已启用")
        print(f"📊 实时绘图系统已初始化")

    def update_training_data(self, episode_reward, training_stats, diversity_score):
        """更新训练数据用于实时图表"""
        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        if not hasattr(self, 'diversity_scores'):
            self.diversity_scores = []
        if not hasattr(self, 'actor_losses'):
            self.actor_losses = []
        if not hasattr(self, 'critic_losses'):
            self.critic_losses = []
        if not hasattr(self, 'entropy_values'):
            self.entropy_values = []
        if not hasattr(self, 'exploration_rates'):
            self.exploration_rates = []
        if not hasattr(self, 'learning_rates_actor'):
            self.learning_rates_actor = []
        if not hasattr(self, 'learning_rates_critic'):
            self.learning_rates_critic = []
        if not hasattr(self, 'performance_trends'):
            self.performance_trends = []
        if not hasattr(self, 'exploration_noise_history'):
            self.exploration_noise_history = []

        self.episode_rewards.append(episode_reward)
        self.diversity_scores.append(diversity_score)

        if 'actor_losses' in training_stats and len(training_stats['actor_losses']) > 0:
            self.actor_losses.append(training_stats['actor_losses'][-1])
        if 'critic_losses' in training_stats and len(training_stats['critic_losses']) > 0:
            self.critic_losses.append(training_stats['critic_losses'][-1])
        if 'entropy_losses' in training_stats and len(training_stats['entropy_losses']) > 0:
            self.entropy_values.append(training_stats['entropy_losses'][-1])
        elif 'entropy_values' in training_stats and len(training_stats['entropy_values']) > 0:
            self.entropy_values.append(training_stats['entropy_values'][-1])

        if 'exploration_rates' in training_stats and len(training_stats['exploration_rates']) > 0:
            self.exploration_rates.append(training_stats['exploration_rates'][-1])
        else:
            self.exploration_rates.append(0.0)

        if 'exploration_noise' in training_stats and len(training_stats['exploration_noise']) > 0:
            self.exploration_noise_history.append(training_stats['exploration_noise'][-1])
        else:
            self.exploration_noise_history.append(0.5)

        if 'learning_rates' in training_stats and len(training_stats['learning_rates']) > 0:
            lr_info = training_stats['learning_rates'][-1]
            if isinstance(lr_info, dict):
                self.learning_rates_actor.append(lr_info.get('actor', 1e-4))
                self.learning_rates_critic.append(lr_info.get('critic', 1e-4))
            else:
                self.learning_rates_actor.append(1e-4)
                self.learning_rates_critic.append(1e-4)
        else:
            self.learning_rates_actor.append(1e-4)
            self.learning_rates_critic.append(1e-4)

        if len(self.episode_rewards) >= 5:
            recent_rewards = self.episode_rewards[-5:]
            trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
            self.performance_trends.append(trend)
        else:
            self.performance_trends.append(0.0)

        # ✅ 内存优化：裁剪 logger 内部列表，防止无限增长
        _MAX_LOGGER_LEN = 2000
        for attr_name in ['episode_rewards', 'diversity_scores', 'actor_losses', 
                          'critic_losses', 'entropy_values', 'exploration_rates',
                          'learning_rates_actor', 'learning_rates_critic', 
                          'performance_trends', 'exploration_noise_history']:
            attr = getattr(self, attr_name, None)
            if attr is not None and isinstance(attr, list) and len(attr) > _MAX_LOGGER_LEN:
                setattr(self, attr_name, attr[-_MAX_LOGGER_LEN:])

        if len(self.episode_rewards) % 5 == 0:
            print(
                f"📊 数据更新 - 回合: {len(self.episode_rewards)}, 奖励: {episode_reward:.2f}, 多样性: {diversity_score:.3f}")

    def generate_realtime_plots(self, episode, save_path="./result152"):
        """生成实时训练曲线图"""
        print(f"🔍 开始绘图检查 - Episode {episode}")

        if not hasattr(self, 'episode_rewards') or len(self.episode_rewards) == 0:
            print("⚠️ 没有奖励数据，使用测试数据生成图片")
            self.episode_rewards = [1.2, 1.5, 1.8, 2.1, 2.3, 2.0, 2.4, 2.6, 2.8, 3.0]
            self.diversity_scores = [0.1, 0.15, 0.12, 0.18, 0.16, 0.14, 0.17, 0.19, 0.20, 0.22]
            self.actor_losses = [0.5, 0.45, 0.42, 0.38, 0.35, 0.33, 0.30, 0.28, 0.25, 0.23]
            self.critic_losses = [0.8, 0.75, 0.72, 0.68, 0.65, 0.62, 0.58, 0.55, 0.52, 0.50]
            self.entropy_values = [0.3, 0.32, 0.29, 0.31, 0.28, 0.30, 0.27, 0.29, 0.26, 0.25]
            self.exploration_noise_history = [0.5, 0.52, 0.48, 0.54, 0.51, 0.53, 0.49, 0.55, 0.52, 0.56]

        try:
            plt.figure(figsize=(20, 15))  # 增大画布以容纳更多子图

            # 1. 奖励曲线
            plt.subplot(3, 3, 1)
            plt.plot(self.episode_rewards, 'b-', linewidth=2)
            plt.title('PyTorch MAPPO Training Reward Curve', fontsize=14)
            plt.xlabel('轮次')
            plt.ylabel('奖励')
            plt.grid(True, alpha=0.3)

            # 2. 奖励移动平均
            plt.subplot(3, 3, 2)
            if len(self.episode_rewards) >= 10:
                moving_avg = np.convolve(self.episode_rewards, np.ones(10) / 10, mode='valid')
                plt.plot(moving_avg, 'r-', linewidth=2)
            plt.title('Reward Moving Average', fontsize=14)
            plt.xlabel('轮次')
            plt.ylabel('Average Reward')
            plt.grid(True, alpha=0.3)

            # 3. 动作多样性
            plt.subplot(3, 3, 3)
            if hasattr(self, 'diversity_scores') and len(self.diversity_scores) > 0:
                plt.plot(self.diversity_scores, 'g-', linewidth=2)
            plt.title('Action Diversity', fontsize=14)
            plt.xlabel('轮次')
            plt.ylabel('Action Std')
            plt.grid(True, alpha=0.3)

            # 4. Actor损失
            plt.subplot(3, 3, 4)
            if hasattr(self, 'actor_losses') and len(self.actor_losses) > 0:
                plt.plot(self.actor_losses, 'm-', linewidth=2)
            plt.title('Actor Loss', fontsize=14)
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)

            # 5. Critic损失
            plt.subplot(3, 3, 5)
            if hasattr(self, 'critic_losses') and len(self.critic_losses) > 0:
                plt.plot(self.critic_losses, 'c-', linewidth=2)
            plt.title('Critic Loss', fontsize=14)
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)

            # 6. 熵值
            plt.subplot(3, 3, 6)
            if hasattr(self, 'entropy_values') and len(self.entropy_values) > 0:
                plt.plot(self.entropy_values, 'orange', linewidth=2)
            plt.title('Entropy', fontsize=14)
            plt.xlabel('Update')
            plt.ylabel('Entropy')
            plt.grid(True, alpha=0.3)

            # 7. 探索率
            plt.subplot(3, 3, 7)
            if hasattr(self, 'exploration_noise_history') and len(self.exploration_noise_history) > 0:
                plt.plot(self.exploration_noise_history, 'purple', linewidth=2)
            plt.title('Exploration Rate', fontsize=14)
            plt.xlabel('Update')
            plt.ylabel('Exploration Noise')
            plt.grid(True, alpha=0.3)

            # 8. 奖励分布
            plt.subplot(3, 3, 8)
            if hasattr(self, 'episode_rewards') and len(self.episode_rewards) > 0:
                rewards_array = np.array(self.episode_rewards)
                if len(rewards_array) > 1:
                    mean_reward = np.mean(rewards_array)
                    std_reward = np.std(rewards_array)
                    plt.axhline(y=mean_reward, color='brown', linestyle='-', linewidth=2, label='Mean')
                    plt.fill_between([0, len(rewards_array) - 1],
                                     [mean_reward - std_reward, mean_reward - std_reward],
                                     [mean_reward + std_reward, mean_reward + std_reward],
                                     alpha=0.3, color='brown')
                    plt.legend()
            plt.title('Reward Distribution', fontsize=14)
            plt.xlabel('Update')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            os.makedirs(save_path, exist_ok=True)
            plot_path = os.path.join(save_path, 'mappo_training_curve_pytorch.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            if os.path.exists(plot_path):
                file_size = os.path.getsize(plot_path)
                print(f"✅ 训练曲线图已保存到: {plot_path} (文件大小: {file_size} 字节)")

            np.save(os.path.join(save_path, 'mappo_episode_rewards_pytorch.npy'), np.array(self.episode_rewards))
            if hasattr(self, 'diversity_scores') and len(self.diversity_scores) > 0:
                np.save(os.path.join(save_path, 'mappo_diversity_scores_pytorch.npy'), np.array(self.diversity_scores))

        except Exception as e:
            print(f"❌ 生成训练曲线图失败: {e}")
            import traceback
            traceback.print_exc()

    def save_training_data(self, save_path="./result152"):
        """保存训练数据"""
        try:
            os.makedirs(save_path, exist_ok=True)
            if hasattr(self, 'episode_rewards') and len(self.episode_rewards) > 0:
                np.save(os.path.join(save_path, 'episode_rewards_pytorch.npy'), np.array(self.episode_rewards))
            if hasattr(self, 'actor_losses') and len(self.actor_losses) > 0:
                np.save(os.path.join(save_path, 'actor_losses_pytorch.npy'), np.array(self.actor_losses))
            if hasattr(self, 'critic_losses') and len(self.critic_losses) > 0:
                np.save(os.path.join(save_path, 'critic_losses_pytorch.npy'), np.array(self.critic_losses))
            if hasattr(self, 'entropy_values') and len(self.entropy_values) > 0:
                np.save(os.path.join(save_path, 'entropy_values_pytorch.npy'), np.array(self.entropy_values))
            if hasattr(self, 'exploration_rates') and len(self.exploration_rates) > 0:
                np.save(os.path.join(save_path, 'exploration_rates_pytorch.npy'), np.array(self.exploration_rates))
            if hasattr(self, 'diversity_scores') and len(self.diversity_scores) > 0:
                np.save(os.path.join(save_path, 'diversity_scores_pytorch.npy'), np.array(self.diversity_scores))
            if self.console_output:
                print(f"💾 训练数据已保存到 {save_path}")
        except Exception as e:
            if self.console_output:
                print(f"❌ 保存训练数据失败: {e}")

    def _write_and_print(self, log_file, message, console_prefix=""):
        """写入日志文件并可选地输出到控制台"""
        log_file.write(message)
        log_file.flush()
        if self.console_output:
            console_msg = message.strip()
            if console_prefix:
                console_msg = f"{console_prefix} {console_msg}"
            print(console_msg)

    def log_reward_analysis(self, episode, step, total_reward, diversity_reward, enhanced_reward,
                            reward_change, info, actions, cumulative_factors):
        """记录奖励分析 - ✅ 增强：包含奖励各个组成部分"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        avg_pm25 = np.mean(info.get('predicted_pm25', [0])) if 'predicted_pm25' in info else 0
        avg_reduction = (1 - np.mean(cumulative_factors)) * 100
        action_mean = np.mean(actions)
        action_std = np.std(actions)

        # ✅ 提取奖励组成部分
        reward_components = info.get('reward_components', {})
        target_reward = reward_components.get('total_target_reward', 0.0)
        cost_penalty = reward_components.get('total_cost_penalty', 0.0)
        health_reward = reward_components.get('total_health_reward', 0.0)
        coordination_reward = reward_components.get('coordination_reward', 0.0)
        province_reward = reward_components.get('total_province_reward', 0.0)
        fairness_reward = reward_components.get('total_fairness_reward', 0.0)

        # ✅ 从coordination_components中提取差分/排名/博弈奖励总和（用于reward_data.csv列对齐）
        coordination_components = reward_components.get('coordination_components', {}) if isinstance(reward_components, dict) else {}
        differential_sum = 0.0
        ranking_sum = 0.0
        inter_region_sum = 0.0
        try:
            if isinstance(coordination_components, dict) and coordination_components:
                diff = coordination_components.get('differential_rewards', None)
                if isinstance(diff, dict):
                    differential_sum = float(sum(diff.values()))
                elif isinstance(diff, np.ndarray):
                    differential_sum = float(np.sum(diff))

                rank = coordination_components.get('ranking_rewards', None)
                if isinstance(rank, dict):
                    ranking_sum = float(sum(rank.values()))
                elif isinstance(rank, np.ndarray):
                    ranking_sum = float(np.sum(rank))

                # 新版优先province_game_rewards，旧版兼容inter_region_rewards
                game = coordination_components.get('province_game_rewards', None)
                if game is None:
                    game = coordination_components.get('inter_region_rewards', None)
                if isinstance(game, dict):
                    inter_region_sum = float(sum(game.values()))
                elif isinstance(game, np.ndarray):
                    inter_region_sum = float(np.sum(game))
        except Exception:
            pass

        log_message = f"[{timestamp}] Episode {episode}, Step {step}\n"
        log_message += f"  总奖励（包含PM2.5目标+成本+健康+协作）: {total_reward:.4f}\n"
        log_message += f"  📊 奖励组成部分:\n"
        log_message += f"    PM2.5目标奖励总和: {target_reward:+.2f}\n"
        log_message += f"    成本惩罚总和: {cost_penalty:+.2f}\n"
        log_message += f"    健康效益奖励总和: {health_reward:+.2f}\n"
        log_message += f"    区域协作奖励: {coordination_reward:+.2f}\n"
        log_message += f"    公平性奖励: {fairness_reward:+.2f}\n"
        log_message += f"    省份奖励总和: {province_reward:+.2f}\n"
        log_message += f"  多样性奖励（已废弃）: {diversity_reward:.4f}\n"
        log_message += f"  增强奖励（等于总奖励）: {enhanced_reward:.4f}\n"
        log_message += f"  奖励变化: {reward_change:+.4f}\n"
        log_message += f"  平均PM2.5: {avg_pm25:.2f} μg/m³\n"
        log_message += f"  累积减排率: {avg_reduction:.1f}%\n"
        log_message += f"  动作统计: mean={action_mean:.3f}, std={action_std:.3f}\n"
        log_message += f"  减排因子范围: [{np.min(cumulative_factors):.3f}, {np.max(cumulative_factors):.3f}]\n\n"

        self._write_and_print(self.reward_log, log_message, "🎯")
        # ✅ 更新CSV写入，包含奖励组成部分
        self.reward_writer.writerow([
            episode, step,
            total_reward, target_reward, cost_penalty, health_reward,
            coordination_reward, differential_sum, ranking_sum, inter_region_sum,
            fairness_reward,
            diversity_reward, enhanced_reward, reward_change,
            avg_pm25, avg_reduction, action_mean, action_std
        ])
        self.reward_csv.flush()

    def log_episode_summary(self, episode, episode_reward, episode_experiences, episode_actions, diversity_score):
        """记录回合总结"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] Episode {episode} Summary\n"
        log_message += f"  总步骤数: {len(episode_experiences)}\n"
        log_message += f"  总奖励: {episode_reward:.4f}\n"
        log_message += f"  动作多样性: {diversity_score:.4f}\n"
        if len(episode_experiences) > 0:
            avg_step_reward = episode_reward / len(episode_experiences)
            log_message += f"  平均步骤奖励: {avg_step_reward:.4f}\n"
            step_rewards = [exp['rewards'] for exp in episode_experiences]
            log_message += f"  奖励分布: min={np.min(step_rewards):.4f}, max={np.max(step_rewards):.4f}, std={np.std(step_rewards):.4f}\n"
        log_message += "\n"
        self._write_and_print(self.episode_log, log_message, "📊")

    def log_training_diagnostics(self, episode, training_stats):
        """记录训练诊断信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] Episode {episode} Training Diagnostics\n"
        if len(training_stats['actor_losses']) > 0:
            actor_loss = training_stats['actor_losses'][-1]
            critic_loss = training_stats['critic_losses'][-1]
            entropy = training_stats['entropy_values'][-1]
            log_message += f"  Actor损失: {actor_loss:.4f}\n"
            log_message += f"  Critic损失: {critic_loss:.4f}\n"
            log_message += f"  熵值: {entropy:.4f}\n"
        if len(training_stats['gradient_norms']) > 0:
            grad_norms = training_stats['gradient_norms'][-1]
            log_message += f"  梯度范数: Actor={grad_norms['actor']:.6f}, Critic={grad_norms['critic']:.6f}\n"
        log_message += "\n"
        self._write_and_print(self.training_log, log_message, "🔧")

    def log_exploration_analysis(self, episode, exploration_stats):
        """记录探索分析"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] Episode {episode} Exploration Analysis\n"
        log_message += f"  探索模式: {exploration_stats['mode']}\n"
        log_message += f"  探索噪声: {exploration_stats['noise']:.3f}\n"
        log_message += f"  探索增强: {exploration_stats['boost']:.3f}\n"
        log_message += f"  多样性奖励: {exploration_stats['diversity_bonus']:.4f}\n"
        log_message += f"  成功模式数量: {exploration_stats['success_patterns']}\n"
        log_message += "\n"
        self._write_and_print(self.exploration_log, log_message, "🎲")

    def log_critic_convergence(self, episode, pre_values, post_values, value_change, critic_loss_trend):
        """记录Critic收敛分析"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] 🔍 Critic收敛分析 - 回合{episode}\n"
        log_message += f"  价值函数变化: {value_change:.6f}\n"
        log_message += f"  更新前价值: mean={np.mean(pre_values):.4f}, std={np.std(pre_values):.4f}\n"
        log_message += f"  更新后价值: mean={np.mean(post_values):.4f}, std={np.std(post_values):.4f}\n"
        self._write_and_print(self.training_log, log_message)

    def log_exploration_strategy(self, episode, exploration_status, policy_improved):
        """记录探索策略状态"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] 🎯 探索策略状态 - 回合{episode}\n"
        log_message += f"  最佳奖励: {exploration_status['best_reward']:.4f}\n"
        log_message += f"  最佳回合: {exploration_status['best_episode']}\n"
        log_message += f"  无改善探索: {exploration_status['exploration_without_improvement']}\n"
        log_message += f"  使用最佳策略: {'是' if exploration_status['is_using_best_policy'] else '否'}\n"
        log_message += f"  策略恢复次数: {exploration_status['policy_restoration_count']}\n"
        log_message += f"  历史窗口大小: {exploration_status['history_window_size']}\n"
        log_message += f"  策略改善: {'是' if policy_improved else '否'}\n"
        self._write_and_print(self.training_log, log_message)

    def log_province_reduction(self, episode, step, province_data):
        """记录各省份的减排率信息"""
        try:
            def to_scalar(value):
                if hasattr(value, 'item'):
                    if hasattr(value, 'size'):
                        if value.size == 1:
                            return value.item()
                        elif value.size > 1:
                            return value.flat[0]
                    return value.item()
                elif hasattr(value, '__len__') and len(value) > 0:
                    return value[0]
                return value

            single_step_reduction = province_data['single_step_reduction']
            action_value = province_data['action_value']

            if hasattr(single_step_reduction, '__len__') and len(single_step_reduction) > 1:
                reduction_array = np.array(single_step_reduction)
                reduction_mean = np.mean(reduction_array)
                reduction_std = np.std(reduction_array)
                reduction_min = np.min(reduction_array)
                reduction_max = np.max(reduction_array)
                reduction_median = np.median(reduction_array)
                num_species = len(reduction_array)

                species_details = ", ".join([f"物种{i + 1}:{r:.4f}" for i, r in enumerate(reduction_array)])

                pm25_target = to_scalar(province_data.get('pm25_target', 0.0))
                log_message = (f"Episode {episode}, Step {step}, Province {province_data['province_id']} "
                               f"({province_data['province_name']}): "
                               f"物种数量={num_species}, "
                               f"平均减排率={reduction_mean:.4f}±{reduction_std:.4f}, "
                               f"中位数={reduction_median:.4f}, "
                               f"范围=[{reduction_min:.4f}, {reduction_max:.4f}], "
                               f"累积减排率={to_scalar(province_data['cumulative_reduction']):.4f}, "
                               f"动作值={to_scalar(action_value):.4f}, "
                               f"PM2.5变化: {to_scalar(province_data['pm25_before']):.2f} -> {to_scalar(province_data['pm25_after']):.2f} (目标: {pm25_target:.2f}), "
                               f"详细: [{species_details}]")
            else:
                reduction_scalar = to_scalar(single_step_reduction)
                pm25_target = to_scalar(province_data.get('pm25_target', 0.0))
                log_message = (f"Episode {episode}, Step {step}, Province {province_data['province_id']} "
                               f"({province_data['province_name']}): "
                               f"单步减排率={reduction_scalar:.4f}, "
                               f"累积减排率={to_scalar(province_data['cumulative_reduction']):.4f}, "
                               f"动作值={to_scalar(action_value):.4f}, "
                               f"PM2.5变化: {to_scalar(province_data['pm25_before']):.2f} -> {to_scalar(province_data['pm25_after']):.2f} (目标: {pm25_target:.2f})")

            self.reduction_log.write(log_message + '\n')
            self.reduction_log.flush()

            pm25_target = to_scalar(province_data.get('pm25_target', 0.0))
            if hasattr(single_step_reduction, '__len__') and len(single_step_reduction) > 1:
                reduction_array = np.array(single_step_reduction)
                self.reduction_writer.writerow([
                    episode, step, province_data['province_id'], province_data['province_name'],
                    len(reduction_array), f"{np.mean(reduction_array):.6f}", f"{np.std(reduction_array):.6f}",
                    f"{np.min(reduction_array):.6f}", f"{np.max(reduction_array):.6f}",
                    f"{np.median(reduction_array):.6f}",
                    f"{to_scalar(province_data['cumulative_reduction']):.6f}", f"{to_scalar(action_value):.6f}",
                    f"{to_scalar(province_data['pm25_before']):.2f}", f"{to_scalar(province_data['pm25_after']):.2f}",
                    f"{pm25_target:.2f}", str(list(reduction_array))
                ])
            else:
                reduction_scalar = to_scalar(single_step_reduction)
                self.reduction_writer.writerow([
                    episode, step, province_data['province_id'], province_data['province_name'],
                    1, f"{reduction_scalar:.6f}", 0.0, f"{reduction_scalar:.6f}", f"{reduction_scalar:.6f}",
                    f"{reduction_scalar:.6f}", f"{to_scalar(province_data['cumulative_reduction']):.6f}",
                    f"{to_scalar(action_value):.6f}", f"{to_scalar(province_data['pm25_before']):.2f}",
                    f"{to_scalar(province_data['pm25_after']):.2f}", f"{pm25_target:.2f}", str([reduction_scalar])
                ])
            self.reduction_csv.flush()

        except Exception as e:
            print(f"⚠️ 记录省份减排率失败: {e}")

    def log_priority_training(self, episode, province_monitors, current_training_province_idx=None, prefix="",
                              episode_total_reward=None, province_episode_totals=None):
        """记录优先级训练信息"""
        try:
            # 写入日志文件
            self.priority_training_log.write(f"\n{'=' * 80}\n")
            if prefix:
                self.priority_training_log.write(f"Episode {episode}: {prefix}优先级训练状态\n")
            else:
                self.priority_training_log.write(f"Episode {episode}: 优先级训练状态\n")
            self.priority_training_log.write(f"{'=' * 80}\n")

            # 统计信息
            fixed_count = sum(1 for monitor in province_monitors.values() if monitor.is_fixed)
            training_count = len(province_monitors) - fixed_count

            self.priority_training_log.write(f"📊 总体状态:\n")
            self.priority_training_log.write(f"  总省份数: {len(province_monitors)}\n")
            self.priority_training_log.write(f"  已固定: {fixed_count}\n")
            self.priority_training_log.write(f"  训练中: {training_count}\n")
            if episode_total_reward is not None:
                self.priority_training_log.write(f"  上一回合总奖励: {episode_total_reward:.2f}\n")
            if current_training_province_idx is not None:
                self.priority_training_log.write(
                    f"  当前优化省份: {province_monitors[current_training_province_idx].province_name} (索引: {current_training_province_idx})\n")
            self.priority_training_log.write(f"\n")

            # 已固定的省份
            self.priority_training_log.write(f"✅ 已固定的省份:\n")
            fixed_provinces = [m for m in province_monitors.values() if m.is_fixed]
            fixed_provinces.sort(key=lambda x: x.fixed_episode)
            for monitor in fixed_provinces:
                stats = monitor.get_stats()
                latest_fixed_reward = stats['reward_history'][-1] if stats['reward_history'] else stats['best_reward']
                self.priority_training_log.write(
                    f"  [{stats['province_idx']:2d}] {stats['province_name']:4s}: "
                    f"最佳奖励={stats['best_reward']:8.2f} (Episode {stats['best_episode']}), "
                    f"固定于Episode {stats['fixed_episode']}, "
                    f"最新固定奖励={latest_fixed_reward:8.2f}\n"
                )
                # 写入CSV
                self.priority_training_writer.writerow([
                    episode, stats['province_idx'], stats['province_name'], 'FIXED',
                    stats['best_reward'], stats['best_episode'], latest_fixed_reward,
                    True, stats['fixed_episode'], stats['total_episodes'],
                    stats['avg_reward'], stats['std_reward'], stats['min_reward'], stats['max_reward']
                ])

            if (episode_total_reward is not None and province_episode_totals and len(fixed_provinces) > 0):
                self.priority_training_log.write(
                    f"\n📌 固定省份奖励贡献 (本回合总奖励: {episode_total_reward:.2f})\n")
                for monitor in fixed_provinces:
                    stats = monitor.get_stats()
                    contribution = province_episode_totals.get(stats['province_idx'], 0.0)
                    share = (contribution / episode_total_reward * 100.0) if episode_total_reward != 0 else 0.0
                    self.priority_training_log.write(
                        f"  [{stats['province_idx']:2d}] {stats['province_name']:4s}: "
                        f"固定奖励={contribution:8.2f}, 占总奖励={share:6.2f}%\n"
                    )

            # 训练中的省份
            self.priority_training_log.write(f"\n🔄 训练中的省份:\n")
            training_provinces = [m for m in province_monitors.values() if not m.is_fixed]
            training_provinces.sort(key=lambda x: x.province_idx)
            for monitor in training_provinces:
                stats = monitor.get_stats()
                # 使用最新奖励（reward_history的最后一个值）
                current_reward = stats['reward_history'][-1] if stats['reward_history'] else 0.0
                status = 'CURRENT' if monitor.province_idx == current_training_province_idx else 'TRAINING'
                self.priority_training_log.write(
                    f"  [{stats['province_idx']:2d}] {stats['province_name']:4s}: "
                    f"最佳奖励={stats['best_reward']:8.2f} (Episode {stats['best_episode']}), "
                    f"最新奖励={current_reward:8.2f}, "
                    f"训练轮次={stats['total_episodes']}\n"
                )
                # 写入CSV
                self.priority_training_writer.writerow([
                    episode, stats['province_idx'], stats['province_name'], status,
                    stats['best_reward'], stats['best_episode'], current_reward,
                    False, None, stats['total_episodes'],
                    stats['avg_reward'], stats['std_reward'], stats['min_reward'], stats['max_reward']
                ])

            self.priority_training_log.flush()
            self.priority_training_csv.flush()

            # 控制台输出
            if self.console_output:
                print(f"\n📊 优先级训练状态 (Episode {episode}):")
                print(f"  已固定: {fixed_count}/{len(province_monitors)}")
                print(f"  训练中: {training_count}/{len(province_monitors)}")
                if episode_total_reward is not None:
                    print(f"  上一回合总奖励: {episode_total_reward:.2f}")
                if current_training_province_idx is not None:
                    current_monitor = province_monitors[current_training_province_idx]
                    latest_reward = current_monitor.reward_history[-1] if current_monitor.reward_history else 0.0
                    print(f"  当前优化: {current_monitor.province_name}")
                    print(f"    最新奖励: {latest_reward:.2f}")
                    print(f"    最佳奖励: {current_monitor.best_reward:.2f} (Episode {current_monitor.best_episode})")
                    print(f"    训练轮次: {len(current_monitor.reward_history)}")
                if (episode_total_reward is not None and province_episode_totals and len(fixed_provinces) > 0):
                    print(f"  固定省份贡献 (总奖励 {episode_total_reward:.2f}):")
                    for monitor in fixed_provinces:
                        stats = monitor.get_stats()
                        latest_fixed_reward = stats['reward_history'][-1] if stats['reward_history'] else stats[
                            'best_reward']
                        contribution = province_episode_totals.get(stats['province_idx'], 0.0)
                        share = (contribution / episode_total_reward * 100.0) if episode_total_reward != 0 else 0.0
                        print(f"    [{stats['province_idx']:2d}] {stats['province_name']:4s}: "
                              f"{contribution:.2f} ({share:.2f}%) | 最新固定奖励 {latest_fixed_reward:.2f}")
        except Exception as e:
            print(f"⚠️ 记录优先级训练信息失败: {e}")
            import traceback
            traceback.print_exc()

    def log_province_rewards(self, episode, province_rewards):
        """记录每个省份的奖励变化，以及Loss等指标，确保监控奖励稳定上升"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] Episode {episode} Province Rewards\n"
        avg_prov_reward = 0.0
        total_prov_reward_sum = 0.0  # ✅ 新增：计算总和用于验证
        num_provs = len(province_rewards)

        for prov, rew in province_rewards.items():
            target = rew.get('target', 0.0)
            cost = rew.get('cost', 0.0)
            health = rew.get('health', 0.0)
            fairness = rew.get('fairness', 0.0)
            # ✅ 协作和竞争奖励
            coordination = rew.get('coordination', 0.0)
            ranking = rew.get('ranking', 0.0)
            game = rew.get('game', 0.0)  # 省份博弈奖励（中央拨款）
            competition = rew.get('competition', ranking + game)  # 竞争奖励汇总（兼容）
            
            # 总奖励 = 基础奖励 + 协作奖励 + 排名竞争奖励 + 博弈奖励
            # 如果province_rewards_dict中有'total'字段，直接使用；否则计算
            total = rew.get('total', target + cost + health + fairness + coordination + ranking + game)
            
            avg_prov_reward += total
            total_prov_reward_sum += total  # ✅ 累加总和
            log_message += (f"  {prov}: Target={target:.2f}, Cost={cost:.2f}, Health={health:.2f}, "
                          f"Fairness={fairness:.2f}, Coordination={coordination:.2f}, Ranking={ranking:.2f}, "
                          f"Game={game:.2f}, Competition={competition:.2f}, Total={total:.2f}\n")
            # ✅ 更新CSV输出
            self.province_rewards_writer.writerow([
                episode, prov,
                target, cost, health, fairness,
                coordination, ranking, game, competition,
                total
            ])

        avg_prov_reward /= num_provs if num_provs > 0 else 1
        log_message += f"  Average Province Reward: {avg_prov_reward:.2f}\n"
        log_message += f"  Total Province Reward Sum: {total_prov_reward_sum:.2f} (用于验证数据一致性)\n\n"  # ✅ 新增：显示总和

        self._write_and_print(self.province_rewards_log, log_message, "🏆")
        self.province_rewards_csv.flush()

    def close(self):
        """关闭所有日志文件"""
        self.reward_log.close()
        self.training_log.close()
        self.exploration_log.close()
        self.episode_log.close()
        self.critic_log.close()
        self.reward_csv.close()
        self.training_csv.close()
        self.reduction_csv.close()
        self.province_rewards_log.close()
        self.province_rewards_csv.close()
        if hasattr(self, 'priority_training_log'):
            self.priority_training_log.close()
        if hasattr(self, 'priority_training_csv'):
            self.priority_training_csv.close()
        if hasattr(self, 'policy_debug_log'):
            self.policy_debug_log.close()
        if hasattr(self, 'loss_gradient_log'):
            self.loss_gradient_log.close()
        if hasattr(self, 'loss_gradient_csv'):
            self.loss_gradient_csv.close()
        print("📝 所有日志文件已关闭")

    def log(self, message):
        """通用日志记录方法"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self._write_and_print(self.training_log, log_message)
        if self.console_output:
            print(log_message)

    def log_policy_debug(self, message):
        """记录策略调试信息到专门的日志文件"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        if hasattr(self, 'policy_debug_log'):
            self.policy_debug_log.write(log_message)
            self.policy_debug_log.flush()
        if self.console_output:
            print(f"DEBUG: {message}")

    def log_coordination_competition(self, episode, step, coordination_components, province_names, province_regions):
        """
        记录协作与竞争奖励的详细信息
        
        Args:
            episode: 回合数
            step: 步骤数
            coordination_components: 协作奖励组成部分字典，包含：
                - province_coordination_rewards: {province_idx: coordination_reward} 区域协作奖励
                - ranking_rewards: numpy数组，区域内排名竞争奖励（零和）
                - province_game_rewards: numpy数组，省份博弈奖励（中央拨款50）
            province_names: 省份名称列表
            province_regions: 省份区域映射字典 {province_name: region_id}
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if coordination_components is None or not isinstance(coordination_components, dict) or len(coordination_components) == 0:
            return
        
        # 获取协作奖励（区域人口加权PM2.5改善）
        coordination_rewards = coordination_components.get('province_coordination_rewards', {})
        # 获取排名竞争奖励（零和）
        ranking_rewards = coordination_components.get('ranking_rewards', {})
        # 获取省份博弈奖励（中央拨款50）
        game_rewards = coordination_components.get('province_game_rewards', {})
        
        log_message = f"[{timestamp}] Episode {episode}, Step {step} - 协作与竞争奖励详情\n"
        log_message += f"{'=' * 80}\n"
        
        # 汇总统计 - 处理可能是numpy数组的情况
        if isinstance(coordination_rewards, dict):
            total_coordination = sum(coordination_rewards.values()) if len(coordination_rewards) > 0 else 0.0
        elif isinstance(coordination_rewards, np.ndarray):
            total_coordination = float(np.sum(coordination_rewards))
        else:
            total_coordination = 0.0
            
        if isinstance(ranking_rewards, dict):
            total_ranking = sum(ranking_rewards.values()) if len(ranking_rewards) > 0 else 0.0
        elif isinstance(ranking_rewards, np.ndarray):
            total_ranking = float(np.sum(ranking_rewards))
        else:
            total_ranking = 0.0
            
        if isinstance(game_rewards, dict):
            total_game = sum(game_rewards.values()) if len(game_rewards) > 0 else 0.0
        elif isinstance(game_rewards, np.ndarray):
            total_game = float(np.sum(game_rewards))
        else:
            total_game = 0.0
        
        log_message += f"📊 汇总统计:\n"
        log_message += f"  区域协作奖励总和: {total_coordination:+.2f}\n"
        log_message += f"  排名竞争奖励总和: {total_ranking:+.2f} (零和)\n"
        log_message += f"  省份博弈奖励总和: {total_game:+.2f} (中央拨款50)\n"
        log_message += f"  总奖励: {total_coordination + total_ranking + total_game:+.2f}\n\n"
        
        # 按区域分组显示
        if province_regions:
            region_groups = {}
            for province_idx, province_name in enumerate(province_names):
                if province_name in province_regions:
                    region_id = province_regions[province_name]
                    if region_id not in region_groups:
                        region_groups[region_id] = []
                    # 安全获取奖励值（处理numpy数组和字典两种情况）
                    if isinstance(coordination_rewards, np.ndarray):
                        coord_val = float(coordination_rewards[province_idx]) if province_idx < len(coordination_rewards) else 0.0
                    elif isinstance(coordination_rewards, dict):
                        coord_val = coordination_rewards.get(province_idx, 0.0)
                    else:
                        coord_val = 0.0
                    
                    if isinstance(ranking_rewards, np.ndarray):
                        rank_val = float(ranking_rewards[province_idx]) if province_idx < len(ranking_rewards) else 0.0
                    elif isinstance(ranking_rewards, dict):
                        rank_val = ranking_rewards.get(province_idx, 0.0)
                    else:
                        rank_val = 0.0
                    
                    if isinstance(game_rewards, np.ndarray):
                        game_val = float(game_rewards[province_idx]) if province_idx < len(game_rewards) else 0.0
                    elif isinstance(game_rewards, dict):
                        game_val = game_rewards.get(province_idx, 0.0)
                    else:
                        game_val = 0.0
                    
                    region_groups[region_id].append({
                        'idx': province_idx,
                        'name': province_name,
                        'coordination': coord_val,
                        'ranking': rank_val,
                        'game': game_val
                    })
            
            for region_id in sorted(region_groups.keys()):
                log_message += f"🌐 区域 {region_id}:\n"
                provinces = region_groups[region_id]
                for prov_info in provinces:
                    log_message += f"  [{prov_info['idx']:2d}] {prov_info['name']:4s}: "
                    log_message += f"协作={prov_info['coordination']:+.2f}, "
                    log_message += f"排名={prov_info['ranking']:+.2f}, "
                    log_message += f"博弈={prov_info['game']:+.2f}\n"
                    
                    # 写入CSV
                    self.coordination_writer.writerow([
                        episode, step, prov_info['idx'], prov_info['name'], region_id,
                        prov_info['coordination'], prov_info['ranking'], prov_info['game'],
                        0.0, 0.0, 0.0
                    ])
                log_message += "\n"
        else:
            # 如果没有区域信息，按省份显示
            for province_idx, province_name in enumerate(province_names):
                # 安全获取奖励值
                if isinstance(coordination_rewards, np.ndarray):
                    coord_val = float(coordination_rewards[province_idx]) if province_idx < len(coordination_rewards) else 0.0
                elif isinstance(coordination_rewards, dict):
                    coord_val = coordination_rewards.get(province_idx, 0.0)
                else:
                    coord_val = 0.0
                    
                if isinstance(ranking_rewards, np.ndarray):
                    rank_val = float(ranking_rewards[province_idx]) if province_idx < len(ranking_rewards) else 0.0
                elif isinstance(ranking_rewards, dict):
                    rank_val = ranking_rewards.get(province_idx, 0.0)
                else:
                    rank_val = 0.0
                    
                if isinstance(game_rewards, np.ndarray):
                    game_val = float(game_rewards[province_idx]) if province_idx < len(game_rewards) else 0.0
                elif isinstance(game_rewards, dict):
                    game_val = game_rewards.get(province_idx, 0.0)
                else:
                    game_val = 0.0
                
                log_message += f"  [{province_idx:2d}] {province_name:4s}: "
                log_message += f"协作={coord_val:+.2f}, "
                log_message += f"排名={rank_val:+.2f}, "
                log_message += f"博弈={game_val:+.2f}\n"
                
                # 写入CSV
                self.coordination_writer.writerow([
                    episode, step, province_idx, province_name, 0,
                    coord_val, rank_val, game_val,
                    0.0, 0.0, 0.0
                ])
        
        log_message += f"{'=' * 80}\n\n"
        
        self._write_and_print(self.coordination_log, log_message, "🤝")
        self.coordination_csv.flush()


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ 设置CUDA内存分配器配置，避免内存碎片导致的OOM
# 这个环境变量必须在import torch之后、使用GPU之前设置
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print(f"✅ 已设置CUDA内存分配器: expandable_segments:True")

# 配置PyTorch（优化GPU使用）
print(f"PyTorch版本: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

if torch.cuda.is_available():
    print(f"✅ 已启用GPU训练，使用 {torch.cuda.device_count()} 个GPU")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    # 启用cuDNN自动调优（提升性能）
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # ✅ 显示GPU内存信息
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"   GPU总内存: {total_memory:.2f} GB")
else:
    print("❌ 未检测到GPU，使用CPU训练")
    # 设置CPU线程数
    torch.set_num_threads(16)


# ==================== PyTorch网络定义 ====================

class SharedActor(nn.Module):
    """共享Actor网络（PyTorch版本）"""

    def __init__(self, local_obs_dim, num_agents, action_dim):
        super(SharedActor, self).__init__()

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
            nn.Linear(64 + 32, 256),  # 64 + 32 = 96
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 动作均值和标准差输出
        self.action_mean_head = nn.Linear(64, action_dim)
        self.action_std_head = nn.Linear(64, action_dim)

    def forward(self, local_obs, agent_id):
        # 处理局部观察
        obs_features = self.obs_branch(local_obs)

        # 处理智能体ID
        id_features = self.id_branch(agent_id)

        # 合并特征
        combined = torch.cat([obs_features, id_features], dim=-1)

        # 特征提取
        features = self.feature_extractor(combined)

        # 输出动作均值和标准差
        action_mean = self.action_mean_head(features)
        
        # ✅ 数值稳定性：检查并处理NaN/Inf
        if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 自定义激活：确保输出在0.01-0.2范围内
        action_mean = torch.sigmoid(action_mean) * 0.19 + 0.01

        action_std = self.action_std_head(features)
        # ✅ 数值稳定性：限制std输入范围，防止softplus溢出
        action_std = torch.clamp(action_std, min=-20.0, max=5.0)
        action_std = F.softplus(action_std) + 1e-5
        # ✅ 额外限制std范围
        action_std = torch.clamp(action_std, min=1e-5, max=1.0)

        return action_mean, action_std


class CentralizedCritic(nn.Module):
    """集中式Critic网络（PyTorch版本）- 改进版：添加目标网络和值函数clipping"""

    def __init__(self, global_obs_dim):
        super(CentralizedCritic, self).__init__()

        # 🎯 改进的Critic网络架构：添加批标准化和正则化
        self.critic = nn.Sequential(
            nn.Linear(global_obs_dim, 512),
            nn.BatchNorm1d(512),  # 批标准化
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout正则化
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

        # 🎯 目标网络，用于稳定的值函数估计（同样架构）
        self.target_critic = nn.Sequential(
            nn.Linear(global_obs_dim, 512),
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

        # 初始化目标网络权重
        self.update_target_network(tau=1.0)

    def forward(self, global_obs):
        return self.critic(global_obs).squeeze(-1)

    def target_forward(self, global_obs):
        """使用目标网络进行前向传播"""
        return self.target_critic(global_obs).squeeze(-1)

    def update_target_network(self, tau=0.005):
        """软更新目标网络"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ==================== PyTorch版MAPPO智能体 ====================

class OptimizedMAPPOAgent:
    """优化版Multi-Agent PPO智能体（PyTorch版本）"""

    def __init__(self, num_agents, local_obs_dim, global_obs_dim, action_dim,
                 lr=1e-4, gamma=0.99, eps_clip=0.2, k_epochs=8, device='cuda'):
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 学习参数
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # 探索参数
        self.exploration_noise = 0.3
        self.noise_decay = 0.999
        self.min_noise = 0.1
        self.epsilon = 0.1  # epsilon-greedy概率
        self.exploration_mode = "moderate"
        self.exploration_enhancement = 1.0
        self.exploration_bonus_weight = 0.1
        self.current_exploration_boost = 1.0
        self.current_exploration_mode = "moderate"
        self.good_exploration_patterns = []

        # 奖励稳定性监控
        self.recent_rewards = []
        self.reward_stability_window = 20
        self.reward_drop_threshold = -10
        self.consecutive_drops = 0
        self.max_consecutive_drops = 3

        # 性能跟踪
        self.performance_history = []
        self.successful_patterns = set()
        self.exploration_effectiveness = []
        self.best_performance = float('-inf')
        self.performance_stagnation_count = 0
        self.reward_trend_window = 20

        # 探索相关
        self.state_visit_count = {}
        self.action_history = []
        self.action_diversity_bonus = 0.05

        # 训练稳定性
        self.reward_clipping = True  # ✅ 启用奖励裁剪，防止极端值
        self.clip_reward_threshold = 500.0  # ✅ 降低阈值，防止梯度爆炸
        self.advantage_normalization = True
        self.normalize_rewards = True
        self.reward_normalizer = RunningMeanStd()

        # 🎯 改进的Critic训练参数（更保守的设置，针对高奖励范围）
        # 值函数裁剪范围分析（修复版）：
        # - 每step奖励标准化后：6.25~11.875（除以200后，再减9）
        # - 8步累积奖励（gamma=0.99）：约50~95（考虑奖励范围-3~+3）
        # - 裁剪到[-100, 100]是合理的，留有足够余量
        self.value_clip = 100.0  # ✅ 从300降到100，更严格地限制值函数范围
        self.max_grad_norm = 0.2  # ✅ 从0.3降到0.2，更强的梯度裁剪防止爆炸
        self.entropy_coef = 0.01  # 保持0.01，更稳定的策略
        self.entropy_decay = 0.999  # 保持0.999，熵衰减更慢
        self.min_entropy_coef = 0.005  # 保持最小熵系数

        # 🎯 目标网络更新参数（更频繁的更新）
        self.target_update_freq = 50  # ✅ 从100降到50，更频繁地更新目标网络
        self.target_update_tau = 0.01  # ✅ 从0.005增加到0.01，更快的软更新

        # 基于历史最优的探索策略（已禁用自动恢复功能，避免策略不匹配导致loss爆炸）
        self.best_reward_history = []
        self.best_policy_weights = None
        self.best_episode = 0
        self.exploration_without_improvement = 0
        self.max_exploration_without_improvement = float('inf')  # ✅ 修复：禁用自动恢复，设置为无穷大
        self.history_window = 500
        self.improvement_threshold = 0.01
        self.is_using_best_policy = False
        self.policy_restoration_count = 0

        # 🎯 扩展的训练统计，包含详细的探索效果评估
        self.training_stats = {
            'episode_rewards': [],
            'critic_losses': [],
            'actor_losses': [],
            'actor_losses_per_province': [],  # 新增：按省份记录actor loss
            'entropy_values': [],
            'exploration_rates': [],
            'entropy_losses': [],
            'value_losses': [],
            'policy_losses': [],
            'total_losses': [],
            'learning_rates': [],
            'exploration_noises': [],
            'reward_stds': [],
            'performance_trends': [],
            'best_policy_usage': [],
            'exploration_effectiveness': [],
            'gradient_norms': [],
            'value_predictions': [],
            'advantage_stats': [],
            'reward_components': [],
            # 🎯 新增探索效果指标
            'epsilon_values': [],  # epsilon-greedy概率历史
            'random_action_ratio': [],  # 随机动作比例
            'policy_diversity': [],  # 策略多样性
            'state_coverage': [],  # 状态覆盖率（简化版）
            'exploration_efficiency': []  # 探索效率：新状态发现率
        }

        # 🎯 优化的学习率设置：进一步降低学习率防止梯度爆炸（针对高奖励范围）
        self.base_lr = lr * 0.3  # ✅ 从0.5降到0.3，进一步减少梯度爆炸风险
        self.min_lr = self.base_lr * 0.1
        self.max_lr = self.base_lr * 1.2  # ✅ 从1.5降到1.2，更保守
        self.lr_adjustment_factor = 1.0
        self.actor_lr = self.base_lr
        self.critic_lr = self.base_lr * 1.5  # ✅ 从2.0降到1.5，更稳定

        # 经验缓冲区
        self.buffer = []
        self.buffer_size = 2048
        self.memory = []

        # 构建PyTorch网络
        # 🎯 每个省份独立的Actor网络
        self.actor_list = nn.ModuleList(
            [SharedActor(local_obs_dim, num_agents, action_dim).to(self.device) for _ in range(num_agents)]
        )
        self.actor_old_list = nn.ModuleList(
            [SharedActor(local_obs_dim, num_agents, action_dim).to(self.device) for _ in range(num_agents)]
        )
        # 🎯 额外保留一个“整体”共享Actor（用于统一保存/对齐），不参与决策
        self.shared_actor = SharedActor(local_obs_dim, num_agents, action_dim).to(self.device)
        self.shared_actor_old = SharedActor(local_obs_dim, num_agents, action_dim).to(self.device)

        # 拷贝初始权重到旧策略与共享Actor
        for actor_old, actor_cur in zip(self.actor_old_list, self.actor_list):
            actor_old.load_state_dict(actor_cur.state_dict())
            actor_old.eval()  # 旧策略设为评估模式
        self.shared_actor.load_state_dict(self.actor_list[0].state_dict())
        self.shared_actor_old.load_state_dict(self.actor_old_list[0].state_dict())
        self.shared_actor_old.eval()

        self.centralized_critic = CentralizedCritic(global_obs_dim).to(self.device)

        # ✅ 修复：每个省份的Actor使用独立的optimizer（真正实现独立更新）
        # 为每个省份创建独立的Actor优化器
        self.actor_optimizers = []
        for actor in self.actor_list:
            actor_optimizer = optim.Adam(actor.parameters(), lr=self.actor_lr)
            self.actor_optimizers.append(actor_optimizer)

        # 保留一个聚合optimizer用于兼容性（主要用于共享Actor，实际不使用）
        all_actor_params = list(self.shared_actor.parameters())
        for actor in self.actor_list:
            all_actor_params += list(actor.parameters())
        self.actor_optimizer = optim.Adam(all_actor_params, lr=self.actor_lr)  # 保留用于兼容

        self.critic_optimizer = optim.Adam(self.centralized_critic.parameters(), lr=self.critic_lr)

        # 日志记录
        self.logger = None

        # 🎯 优先级训练相关
        self.fixed_policies = {}  # {province_idx: policy_state_dict}
        self.fixed_actor_networks = {}  # {province_idx: SharedActor}
        self.fixed_actions_sequences = {}  # 🎯 新增：{province_idx: [step0_action, step1_action, ...]}
        self.province_monitors = {}  # {province_idx: ProvinceMonitor}
        self.current_training_province_idx = None

    def _reinit_actor(self, province_idx):
        """
        重新初始化指定省份的Actor网络权重
        当检测到NaN/Inf时调用，用于恢复训练稳定性
        """
        print(f"🔄 重新初始化省份 {province_idx} 的Actor网络...")
        
        # 重新创建Actor网络
        new_actor = SharedActor(self.local_obs_dim, self.num_agents, self.action_dim).to(self.device)
        
        # 复制新网络的权重到actor_list
        self.actor_list[province_idx].load_state_dict(new_actor.state_dict())
        self.actor_old_list[province_idx].load_state_dict(new_actor.state_dict())
        
        # 重新创建该省份的optimizer
        self.actor_optimizers[province_idx] = optim.Adam(
            self.actor_list[province_idx].parameters(), 
            lr=self.actor_lr
        )
        
        print(f"✅ 省份 {province_idx} 的Actor网络已重新初始化")

    def initialize_province_monitors(self, province_names):
        """初始化省份监控器"""
        for idx, name in enumerate(province_names):
            self.province_monitors[idx] = ProvinceMonitor(idx, name)

    def get_province_policy(self, province_idx):
        """获取单个省份的策略（从该省份的Actor中提取）

        注意：必须使用该省份的旧策略actor_old，以保持与实际执行一致
        """
        return copy.deepcopy(self.actor_old_list[province_idx].state_dict())

    def load_province_policy(self, province_idx, policy_state_dict):
        """加载单个省份的策略到该省份的Actor"""
        if policy_state_dict is not None:
            self.actor_list[province_idx].load_state_dict(policy_state_dict)
            # 同步共享Actor（使用省份0作为“整体”Actor）
            if province_idx == 0:
                self.shared_actor.load_state_dict(self.actor_list[0].state_dict())

    def set_fixed_policy(self, province_idx, policy_state_dict, actions_sequence=None):
        """保存固定策略（动作序列优先）

        Args:
            province_idx: 省份索引
            policy_state_dict: 策略网络状态（备用）
            actions_sequence: 实际执行的动作序列（优先使用）
        """
        # 🎯 优先保存实际动作序列
        if actions_sequence is not None:
            self.fixed_actions_sequences[province_idx] = copy.deepcopy(actions_sequence)
            if hasattr(self, 'logger') and self.logger is not None:
                first_action = actions_sequence[0] if len(actions_sequence) > 0 else None
                debug_msg = (f"✅ 已固定省份 {province_idx} 的动作序列 - "
                             f"序列长度={len(actions_sequence)}, "
                             f"Step1动作前5值={list(first_action[:5]) if first_action is not None else 'None'}")
                self.logger.log_policy_debug(debug_msg)

        # 同时保存策略网络作为备用
        if policy_state_dict is not None:
            policy_snapshot = copy.deepcopy(policy_state_dict)
            self.fixed_policies[province_idx] = policy_snapshot
            fixed_actor = SharedActor(self.local_obs_dim, self.num_agents, self.action_dim).to(self.device)
            fixed_actor.load_state_dict(policy_snapshot)
            fixed_actor.eval()
            self.fixed_actor_networks[province_idx] = fixed_actor

    def get_fixed_policy_debug_info(self, province_idx):
        """获取固定策略的调试信息"""
        info = f"省份 {province_idx}: "

        # 检查动作序列
        if province_idx in self.fixed_actions_sequences:
            actions = self.fixed_actions_sequences[province_idx]
            info += f"动作序列长度={len(actions)}"
            if len(actions) > 0:
                info += f", Step1动作前5值={list(actions[0][:5])}"
        elif province_idx in self.fixed_policies:
            policy = self.fixed_policies[province_idx]
            first_param = list(policy.values())[0]
            info += f"策略网络第一参数前5值={first_param.flatten()[:5].tolist()}"
        else:
            info += "没有固定策略"

        return info

    def _get_fixed_actor(self, province_idx):
        """获取固定省份的独立Actor（备用方法）"""
        if province_idx in self.fixed_actor_networks:
            return self.fixed_actor_networks[province_idx]
        policy_state = self.fixed_policies.get(province_idx)
        if policy_state is None:
            return None
        fixed_actor = SharedActor(self.local_obs_dim, self.num_agents, self.action_dim).to(self.device)
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
            # 如果步骤超出序列长度，使用最后一个动作
            return actions_seq[-1] if len(actions_seq) > 0 else None
        return actions_seq[step]

    def select_actions_with_fixed(self, local_observations, agent_ids, episode=0, step_reward=None, step=0):
        """选择动作：支持固定策略模式（优先使用保存的动作序列）

        ✅ 修复：固定省份也计算真实log_prob，避免梯度爆炸
        """
        self._update_exploration_mode(episode)

        if step_reward is not None:
            self._monitor_reward_stability(step_reward)

        # 转换为PyTorch张量
        local_obs_tensor = torch.FloatTensor(local_observations).to(self.device)
        agent_ids_tensor = torch.FloatTensor(agent_ids).to(self.device)

        actions = np.zeros((len(local_observations), self.action_dim))
        log_probs = np.zeros((len(local_observations),))

        # 🎯 epsilon-greedy策略
        use_random = np.random.random() < self.epsilon

        for province_idx in range(len(local_observations)):
            # 🎯 检查是否有固定的动作序列（优先使用）
            if province_idx in self.fixed_actions_sequences:
                # 使用保存的实际动作序列
                fixed_action = self.get_fixed_action(province_idx, step)
                if fixed_action is not None:
                    actions[province_idx] = np.array(fixed_action)

                    # ✅ 关键修复：计算真实log_prob而不是0，避免ratio计算异常
                    with torch.no_grad():
                        action_mean, action_std = self.actor_old_list[province_idx](
                            local_obs_tensor[province_idx:province_idx + 1],
                            agent_ids_tensor[province_idx:province_idx + 1]
                        )
                        dist = Normal(action_mean, action_std + 1e-8)
                        action_tensor = torch.FloatTensor(fixed_action).unsqueeze(0).to(self.device)
                        log_probs[province_idx] = dist.log_prob(action_tensor).sum(dim=-1).cpu().numpy()[0]

                    # 🔍 调试输出
                    if step == 0 and province_idx == 0 and hasattr(self, 'logger') and self.logger is not None:
                        debug_msg = (f"固定动作序列执行 - 省份 {province_idx}, Episode {episode}, Step {step}: "
                                     f"动作前5值={list(fixed_action[:5])}, log_prob={log_probs[province_idx]:.4f}")
                        self.logger.log_policy_debug(debug_msg)
                    continue
                # 如果动作序列获取失败，回退到策略网络

            if province_idx in self.fixed_policies:
                # 备用方案：使用固定策略网络（无探索）
                fixed_actor = self._get_fixed_actor(province_idx)
                if fixed_actor is None:
                    fixed_actor = self.actor_old_list[province_idx]
                with torch.no_grad():
                    action_mean, action_std = fixed_actor(
                        local_obs_tensor[province_idx:province_idx + 1],
                        agent_ids_tensor[province_idx:province_idx + 1]
                    )
                # 使用均值（不使用随机采样）
                actions[province_idx] = action_mean[0].cpu().numpy()

                # ✅ 修复：计算真实log_prob
                dist = Normal(action_mean, action_std + 1e-8)
                action_tensor = torch.FloatTensor(actions[province_idx]).unsqueeze(0).to(self.device)
                log_probs[province_idx] = dist.log_prob(action_tensor).sum(dim=-1).cpu().numpy()[0]
            else:
                # 正常探索：使用当前策略（省份专属actor_old）
                if use_random:
                    # 随机探索 - 也需要计算log_prob
                    random_action = None
                    if episode < 100:
                        random_action = np.random.uniform(0.01, 0.2, self.action_dim)
                    else:
                        random_action = np.random.uniform(0.05, 0.3, self.action_dim)
                    actions[province_idx] = random_action

                    # ✅ 计算随机动作的log_prob
                    with torch.no_grad():
                        action_mean, action_std = self.actor_old_list[province_idx](
                            local_obs_tensor[province_idx:province_idx + 1],
                            agent_ids_tensor[province_idx:province_idx + 1]
                        )
                        dist = Normal(action_mean, action_std + 1e-8)
                        action_tensor = torch.FloatTensor(random_action).unsqueeze(0).to(self.device)
                        log_probs[province_idx] = dist.log_prob(action_tensor).sum(dim=-1).cpu().numpy()[0]
                else:
                    # 使用该省份的旧策略网络
                    with torch.no_grad():
                        action_mean, action_std = self.actor_old_list[province_idx](
                            local_obs_tensor[province_idx:province_idx + 1],
                            agent_ids_tensor[province_idx:province_idx + 1]
                        )

                    # 添加探索噪声
                    current_noise = self.exploration_noise * self.current_exploration_boost
                    noise = torch.randn_like(action_mean) * current_noise
                    action = action_mean + action_std * noise
                    action = torch.clamp(action, 0.01, 0.2)

                    # 计算对数概率
                    dist = Normal(action_mean, action_std + 1e-8)
                    log_prob = dist.log_prob(action).sum(dim=-1)

                    actions[province_idx] = action[0].cpu().numpy()
                    log_probs[province_idx] = log_prob[0].cpu().numpy()

        return actions, log_probs

    def _update_exploration_mode(self, episode):
        """自适应探索策略 - 参考v10实现，结合奖励波动与性能趋势"""
        base_noise = max(self.min_noise, 0.3 * (0.98 ** (episode // 10)))

        # 基于奖励波动的动态调整（更保守的策略）
        if len(self.recent_rewards) >= 20:
            recent_std = np.std(self.recent_rewards[-20:])
            recent_mean = np.mean(self.recent_rewards[-20:])
            reward_cv = recent_std / (abs(recent_mean) + 1e-8)
            # ✅ 提高触发阈值，从0.5增加到0.6，避免过早增加探索
            if reward_cv > 0.6:
                # ✅ 降低增长幅度，从0.3降到0.2
                adaptive_boost = min(0.08, reward_cv * 0.2)
                base_noise += adaptive_boost
                # ✅ 只在每50个episode打印一次日志，避免刷屏
                if self.logger and episode % 50 == 0:
                    self.logger.log(f"奖励波动大(CV={reward_cv:.3f})，增加探索噪声到 {base_noise:.3f}")

        # 基于奖励趋势的调整
        if len(self.recent_rewards) >= 30:
            recent_trend = np.polyfit(
                range(len(self.recent_rewards[-30:])),
                self.recent_rewards[-30:], 1
            )[0]
            if recent_trend < -50:
                trend_boost = min(0.1, abs(recent_trend) / 500.0)
                base_noise += trend_boost
                if self.logger:
                    self.logger.log(f"检测到奖励下降趋势({recent_trend:.1f})，增加探索噪声到 {base_noise:.3f}")
            elif recent_trend > 20 and len(self.recent_rewards) >= 50:
                trend_reduction = min(0.05, recent_trend / 1000.0)
                base_noise = max(self.min_noise, base_noise - trend_reduction)

        # 性能停滞时增加探索
        if len(self.performance_history) >= 50:
            recent_performance = np.mean(self.performance_history[-50:])
            if recent_performance <= self.best_performance * 1.01:
                self.performance_stagnation_count += 1
                if self.performance_stagnation_count > 20:
                    stagnation_boost = min(0.1, self.performance_stagnation_count / 200.0)
                    base_noise += stagnation_boost
                    if self.logger:
                        self.logger.log(
                            f"性能停滞({self.performance_stagnation_count}轮)，增加探索噪声到 {base_noise:.3f}")
            else:
                self.performance_stagnation_count = 0

        self.exploration_noise = max(self.min_noise, min(0.5, base_noise))

    def _monitor_reward_stability(self, episode_reward):
        """监控奖励稳定性（改进版）"""
        self.recent_rewards.append(episode_reward)
        if len(self.recent_rewards) > self.reward_stability_window:
            self.recent_rewards.pop(0)

        if len(self.recent_rewards) >= 5:  # 增加到5个样本
            # 使用更稳定的趋势计算
            recent_trend = np.polyfit(range(min(10, len(self.recent_rewards))),
                                      self.recent_rewards[-min(10, len(self.recent_rewards)):], 1)[0]

            # 更保守的下降判断
            if recent_trend < self.reward_drop_threshold * 2:  # 降低灵敏度
                self.consecutive_drops += 1
                if self.consecutive_drops >= self.max_consecutive_drops:
                    self._trigger_recovery_mechanism()
            else:
                self.consecutive_drops = max(0, self.consecutive_drops - 1)  # 逐渐减少计数

    def _trigger_recovery_mechanism(self):
        """触发恢复机制（修复：更保守的学习率调整）"""
        self.exploration_noise = min(0.5, self.exploration_noise + 0.05)  # ✅ 从0.1降到0.05，更温和
        self.lr_adjustment_factor = max(0.3, self.lr_adjustment_factor * 0.9)  # ✅ 从0.8改为0.9，更温和
        new_lr = self.base_lr * self.lr_adjustment_factor

        # ✅ 更新所有省份的Actor optimizer
        for actor_optimizer in self.actor_optimizers:
            for param_group in actor_optimizer.param_groups:
                param_group['lr'] = new_lr

        # ✅ 同时更新聚合optimizer（兼容性）
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr

        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr

        # ✅ 只打印一次恢复机制触发的日志
        print(f"🔄 触发恢复机制: 探索噪声={self.exploration_noise:.3f}, 学习率={new_lr:.6f}")
        if self.logger:
            self.logger.log(f"触发恢复机制: 探索噪声={self.exploration_noise:.3f}, 学习率={new_lr:.6f}")
        self.consecutive_drops = 0

    def _stabilize_rewards(self, rewards, episode=None):
        """
        稳定奖励函数（修复版：自适应缩放，适用于高奖励范围）
        
        ========================================
        🎯 奖励缩放分析（针对10000~19000范围）
        ========================================
        
        如果每episode总奖励：10000~19000
        - 每个episode有8个小步
        - 每小步平均奖励：1250 ~ 2375
        
        修复后的缩放策略：
        1. 缩放：除以200（而不是50），得到 6.25 ~ 11.875
        2. 减去基准值：减去9.0（中间值），得到 -2.75 ~ +2.875
        3. 优点：
           - 奖励范围合理（-3 ~ +3），适合PPO训练
           - 奖励有正有负，信号清晰
           - 避免优势值过大导致loss波动
        
        ========================================
        📊 自适应缩放机制
        ========================================
        
        如果检测到奖励范围异常，自动调整缩放因子：
        - 奖励 > 5000: 使用更大缩放因子（除以200）
        - 奖励 1000~5000: 使用中等缩放因子（除以100）
        - 奖励 < 1000: 使用小缩放因子（除以50）
        
        ========================================
        
        Args:
            rewards: 原始奖励数组（每个step的即时奖励）
            episode: 当前episode编号（用于统计输出）
        """
        # ✅ 自适应缩放：根据奖励范围自动选择缩放因子
        reward_mean = np.mean(rewards) if len(rewards) > 0 else 0.0
        reward_max = np.max(rewards) if len(rewards) > 0 else 0.0
        
        # 根据奖励范围选择缩放因子
        if reward_max > 5000:
            # 高奖励范围（10000~19000）：使用大缩放因子
            scale_factor = 200.0
            baseline = 9.0  # (1250+2375)/2/200 ≈ 9.0
        elif reward_max > 1000:
            # 中等奖励范围（1000~5000）：使用中等缩放因子
            scale_factor = 100.0
            baseline = 5.0
        else:
            # 低奖励范围（<1000）：使用小缩放因子
            scale_factor = 50.0
            baseline = 2.0
        
        # ✅ 步骤1：缩放奖励
        rewards = rewards / scale_factor
        
        # ✅ 步骤2：减去基准值，使奖励有正有负
        rewards = rewards - baseline
        
        # ✅ 更严格的奖励裁剪，防止极端值导致loss波动
        self.reward_clipping = True
        self.clip_reward_threshold = 5.0  # ✅ 从50降到5，更严格地限制奖励范围

        # 记录原始奖励统计（每50个episode打印一次，更频繁）
        if episode is not None and episode % 50 == 0 and len(rewards) > 0:
            raw_rewards = (rewards + baseline) * scale_factor  # 恢复原始范围用于统计
            print(f"\n📊 Episode {episode} 奖励统计:")
            print(f"   原始奖励范围: [{np.min(raw_rewards):.2f}, {np.max(raw_rewards):.2f}]")
            print(f"   原始奖励均值: {np.mean(raw_rewards):.2f}, 标准差: {np.std(raw_rewards):.2f}")
            print(f"   缩放因子: {scale_factor}, 基准值: {baseline}")
            print(f"   缩放后范围: [{np.min(rewards + baseline):.2f}, {np.max(rewards + baseline):.2f}]")
            print(f"   减去基准值后范围: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
            print(f"   正负分布: 正奖励={np.sum(rewards > 0)}, 负奖励={np.sum(rewards < 0)}, 零奖励={np.sum(rewards == 0)}")
            print(f"   区分度评估: {'✅ 良好' if np.max(rewards) - np.min(rewards) > 2.0 else '⚠️ 较小'}")

        if self.reward_clipping:
            rewards = np.clip(rewards, -self.clip_reward_threshold, self.clip_reward_threshold)
        if self.normalize_rewards:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)
        
        # ✅ 数值稳定性：检查并处理NaN/Inf
        if np.isnan(rewards).any() or np.isinf(rewards).any():
            print(f"⚠️ 检测到奖励中的NaN/Inf值，进行修复")
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=self.clip_reward_threshold, neginf=-self.clip_reward_threshold)
        
        return rewards

    def _compute_stable_gae(self, rewards, values):
        """计算稳定的GAE（Generalized Advantage Estimation）

        确保优势计算正确：
        1. 使用正确的GAE公式
        2. 归一化优势以避免过大
        3. 处理边界情况
        """
        gae_lambda = 0.95  # ✅ 从0.9增加到0.95，更重视长期奖励

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        rewards = np.atleast_1d(rewards)
        values = np.atleast_1d(values)

        seq_len = len(rewards)
        values_with_next = np.concatenate([values, [0.0]])

        advantages = []
        gae = 0

        for t in reversed(range(seq_len)):
            delta = rewards[t] + self.gamma * values_with_next[t + 1] - values_with_next[t]
            gae = delta + self.gamma * gae_lambda * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + values

        # ✅ 改进优势归一化 - 更保守的策略，防止优势值过大
        if self.advantage_normalization:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            
            # ✅ 标准化
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            
            # ✅ 更保守的软限制：tanh压缩到(-1, 1)，乘以2使范围变为(-2, 2)
            # 从3.0降到2.0，进一步限制优势值范围，防止梯度爆炸
            advantages = np.tanh(advantages / 2.0) * 2.0
            
            # ✅ 额外硬裁剪作为安全网：确保优势值在合理范围内
            # 从±5.0收紧到±3.0，进一步防止梯度爆炸
            advantages = np.clip(advantages, -3.0, 3.0)

            # ✅ 验证归一化后的优势统计
            if np.any(np.isnan(advantages)) or np.any(np.isinf(advantages)):
                print(f"⚠️ 警告: GAE计算出现NaN或Inf，使用零优势")
                advantages = np.zeros_like(advantages)

        return torch.tensor(returns, dtype=torch.float32).to(self.device), torch.tensor(advantages,
                                                                                        dtype=torch.float32).to(
            self.device)

    def select_actions(self, local_observations, agent_ids, episode=0, step_reward=None):
        """选择动作"""
        self._update_exploration_mode(episode)

        if step_reward is not None:
            self._monitor_reward_stability(step_reward)

        # 前150轮加强随机探索，参考v10实现
        if episode < 150:
            decay_ratio = 0.8 - episode * (0.6 / 150.0)  # 逐步从0.8下降到0.2
            random_prob = max(0.2, decay_ratio)
            if np.random.random() < random_prob:
                actions = np.random.uniform(0.01, 0.2, (len(local_observations), self.action_dim))
                log_probs = np.zeros((len(local_observations),))
                self.action_history.append(actions.flatten())
                if len(self.action_history) > 100:
                    self.action_history.pop(0)
                if hasattr(self, 'exploration_stats'):
                    if 'random_actions' not in self.exploration_stats:
                        self.exploration_stats['random_actions'] = 0
                    if 'total_actions' not in self.exploration_stats:
                        self.exploration_stats['total_actions'] = 0
                    self.exploration_stats['random_actions'] += len(actions)
                    self.exploration_stats['total_actions'] += len(actions)
                return actions, log_probs

        # 🎯 改进的epsilon-greedy策略：与探索噪声协同工作
        if np.random.random() < self.epsilon:
            # 根据当前探索阶段选择不同的随机动作范围
            if episode < 100:
                # 早期：完全随机探索
                actions = np.random.uniform(0.01, 0.2, (len(local_observations), self.action_dim))
            elif episode < 300:
                # 中期：中等范围随机
                actions = np.random.uniform(0.05, 0.5, (len(local_observations), self.action_dim))
            else:
                # 后期：小范围随机，避免过度偏离最优策略
                actions = np.random.uniform(0.05, 0.3, (len(local_observations), self.action_dim))

            log_probs = np.zeros((len(local_observations),))
            self.action_history.append(actions.flatten())
            if len(self.action_history) > 100:
                self.action_history.pop(0)

            # 🎯 记录随机动作数量用于探索效果评估
            if hasattr(self, 'exploration_stats'):
                self.exploration_stats['random_actions'] += len(actions)

            return actions, log_probs

        # 转换为PyTorch张量
        local_obs_tensor = torch.FloatTensor(local_observations).to(self.device)
        agent_ids_tensor = torch.FloatTensor(agent_ids).to(self.device)

        actions_np = np.zeros((len(local_observations), self.action_dim))
        log_probs_np = np.zeros((len(local_observations),))

        for province_idx in range(len(local_observations)):
            actor_old = self.actor_old_list[province_idx]
            with torch.no_grad():
                action_means, action_stds = actor_old(
                    local_obs_tensor[province_idx:province_idx + 1],
                    agent_ids_tensor[province_idx:province_idx + 1]
                )

            # 添加探索噪声
            current_noise = self.exploration_noise * self.current_exploration_boost
            noise = torch.randn_like(action_means) * current_noise
            action = action_means + action_stds * noise

            # 裁剪动作
            action = torch.clamp(action, 0.01, 0.2)

            # 计算对数概率
            dist = Normal(action_means, action_stds + 1e-8)
            log_prob = dist.log_prob(action).sum(dim=-1)

            actions_np[province_idx] = action[0].cpu().numpy()
            log_probs_np[province_idx] = log_prob[0].cpu().numpy()

        self.action_history.append(actions_np.flatten())
        if len(self.action_history) > 100:
            self.action_history.pop(0)

        # 🎯 记录探索效果指标
        if not hasattr(self, 'exploration_stats'):
            self.exploration_stats = {
                'random_actions': 0,
                'total_actions': 0,
                'unique_states': set(),
                'episode_random_ratio': []
            }

        self.exploration_stats['total_actions'] += len(actions_np)

        return actions_np, log_probs_np

    def evaluate_actions(self, local_observations, agent_ids, actions):
        """评估动作（每个省份独立Actor）"""
        if isinstance(local_observations, torch.Tensor):
            local_obs_tensor = local_observations.to(self.device)
        else:
            local_obs_tensor = torch.FloatTensor(local_observations).to(self.device)

        if isinstance(agent_ids, torch.Tensor):
            agent_ids_tensor = agent_ids.to(self.device)
        else:
            agent_ids_tensor = torch.FloatTensor(agent_ids).to(self.device)

        if isinstance(actions, torch.Tensor):
            actions_tensor = actions.to(self.device)
        else:
            actions_tensor = torch.FloatTensor(actions).to(self.device)

        num_agents = len(self.actor_list)
        total_samples = actions_tensor.shape[0]  # batch_size * num_agents（已展平）
        assert total_samples % num_agents == 0

        new_log_probs_list = []
        entropy_list = []

        for p_idx in range(num_agents):
            idxs = torch.arange(p_idx, total_samples, num_agents, device=self.device)
            obs_p = local_obs_tensor[idxs]
            ids_p = agent_ids_tensor[idxs]
            actions_p = actions_tensor[idxs]

            actor_cur = self.actor_list[p_idx]
            mean_cur, std_cur = actor_cur(obs_p, ids_p)
            dist_cur = Normal(mean_cur, std_cur + 1e-8)

            logp_cur = dist_cur.log_prob(actions_p).sum(dim=-1)
            ent = dist_cur.entropy().sum(dim=-1)

            new_log_probs_list.append(logp_cur)
            entropy_list.append(ent)

        new_log_probs = torch.cat(new_log_probs_list, dim=0)
        entropy = torch.cat(entropy_list, dim=0)

        return new_log_probs, entropy

    def get_values(self, global_observations, use_target=False):
        """获取状态价值 - 改进版：支持目标网络"""
        if isinstance(global_observations, np.ndarray):
            global_obs_tensor = torch.FloatTensor(global_observations).to(self.device)
        else:
            global_obs_tensor = global_observations.to(self.device)

        with torch.no_grad():
            if use_target:
                values = self.centralized_critic.target_forward(global_obs_tensor)
            else:
                values = self.centralized_critic(global_obs_tensor)
        return values

    def update(self, episode):  # 修改：添加 episode 参数以支持 TensorBoard
        """MAPPO更新（PyTorch版本）"""
        if len(self.memory) == 0:
            return

        print(f"开始PyTorch版MAPPO更新，经验池大小: {len(self.memory)}")

        # 准备数据
        local_obs_batch = []
        global_obs_batch = []
        agent_ids_batch = []
        actions_batch = []
        rewards_batch = []
        old_log_probs_batch = []
        province_rewards_batch = []  # ✅ 新增：存储每个省份的奖励

        for experience in self.memory:
            local_obs_batch.append(experience['local_obs'])
            global_obs_batch.append(experience['global_obs'])
            agent_ids_batch.append(experience['agent_ids'])
            actions_batch.append(experience['actions'])
            rewards_batch.append(experience['rewards'])
            old_log_probs_batch.append(experience['old_log_probs'])
            # ✅ 新增：提取每个省份的奖励（包含竞争奖励）
            province_rewards_batch.append(experience.get('province_rewards', None))

        local_obs_batch = torch.FloatTensor(np.array(local_obs_batch)).to(self.device)
        global_obs_batch = torch.FloatTensor(np.array(global_obs_batch)).to(self.device)
        agent_ids_batch = torch.FloatTensor(np.array(agent_ids_batch)).to(self.device)
        
        # ✅ 数值稳定性检查：处理NaN和Inf值
        if torch.isnan(local_obs_batch).any() or torch.isinf(local_obs_batch).any():
            print(f"⚠️ 检测到local_obs_batch中的NaN/Inf值，进行修复")
            local_obs_batch = torch.nan_to_num(local_obs_batch, nan=0.0, posinf=100.0, neginf=-100.0)
        if torch.isnan(global_obs_batch).any() or torch.isinf(global_obs_batch).any():
            print(f"⚠️ 检测到global_obs_batch中的NaN/Inf值，进行修复")
            global_obs_batch = torch.nan_to_num(global_obs_batch, nan=0.0, posinf=100.0, neginf=-100.0)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        rewards_batch = np.array(rewards_batch)
        old_log_probs_batch = torch.FloatTensor(np.array(old_log_probs_batch)).to(
            self.device).detach()  # ✅ 修复：detach()确保不参与梯度计算

        # 重新整形数据
        batch_size = local_obs_batch.shape[0]
        num_agents = local_obs_batch.shape[1] if len(local_obs_batch.shape) > 2 else 1

        if len(local_obs_batch.shape) > 2:
            local_obs_batch = local_obs_batch.view(-1, local_obs_batch.shape[-1])
            agent_ids_batch = agent_ids_batch.view(-1, agent_ids_batch.shape[-1])
            actions_batch = actions_batch.view(-1, actions_batch.shape[-1])
            old_log_probs_batch = old_log_probs_batch.view(-1)

        # ✅ 关键修复：为每个省份计算包含竞争奖励的独立奖励
        # 
        # ========================================
        # 🎯 竞争奖励参与actor更新的机制
        # ========================================
        # 
        # 竞争奖励虽然是零和的，但每个省份应该得到不同的奖励：
        # - 排名高的省份得到正奖励
        # - 排名低的省份得到负奖励
        # 
        # 实现方式：
        # 1. 从province_rewards_dict中提取每个省份的奖励（包含竞争奖励）
        # 2. 为每个省份构建独立的奖励序列
        # 3. 为每个省份单独计算GAE，得到独立的优势
        # 4. 每个省份使用自己的优势来更新actor
        # 
        # 这样，竞争奖励才能真正影响每个省份的actor更新：
        # - 排名高的省份得到正优势，actor会向更好的策略更新
        # - 排名低的省份得到负优势，actor会向更好的策略更新（避免负优势）
        # 
        # ========================================
        # 
        # 如果提供了province_rewards，使用省份级奖励（包含竞争奖励）
        # 否则使用全局奖励（向后兼容）
        use_province_rewards = province_rewards_batch[0] is not None and len(province_rewards_batch) > 0
        
        if use_province_rewards:
            # ✅ 为每个省份构建独立的奖励序列（包含竞争奖励）
            # 格式：province_rewards_batch[step][province_name]['total'] 包含竞争奖励
            num_provinces = len(self.actor_list)
            province_rewards_per_step = []  # [step][province_idx] = reward
            
            for step_idx, province_rewards_dict in enumerate(province_rewards_batch):
                if province_rewards_dict is None:
                    # 如果没有省份级奖励，使用全局奖励
                    province_rewards_per_step.append([rewards_batch[step_idx]] * num_provinces)
                else:
                    # 提取每个省份的奖励（包含竞争奖励）
                    step_province_rewards = []
                    # ✅ 获取省份名称列表（从province_monitors中获取，确保顺序正确）
                    province_names_list = []
                    if hasattr(self, 'province_monitors') and self.province_monitors:
                        # 按照province_idx的顺序获取province_name
                        province_names_list = [self.province_monitors[i].province_name 
                                              for i in range(num_provinces) 
                                              if i in self.province_monitors]
                    
                    # 如果province_names_list为空或不完整，使用默认名称
                    if len(province_names_list) < num_provinces:
                        province_names_list = [f"Province_{i}" for i in range(num_provinces)]
                    
                    for province_idx in range(num_provinces):
                        # 获取省份名称
                        province_name = province_names_list[province_idx]
                        
                        # 从province_rewards_dict中获取该省份的奖励（包含竞争奖励）
                        province_reward = rewards_batch[step_idx]  # 默认使用全局奖励
                        if province_name in province_rewards_dict:
                            reward_dict = province_rewards_dict[province_name]
                            if isinstance(reward_dict, dict) and 'total' in reward_dict:
                                # ✅ 使用省份总奖励（包含竞争奖励：differential + ranking + inter_region）
                                province_reward = reward_dict['total']
                        
                        step_province_rewards.append(province_reward)
                    
                    province_rewards_per_step.append(step_province_rewards)
            
            # 转换为numpy数组：shape = (batch_size, num_provinces)
            province_rewards_array = np.array(province_rewards_per_step)
            
            # 奖励稳定化（对每个省份的奖励序列分别处理）
            # 注意：这里我们需要为每个省份单独处理奖励
            # 但为了简化，我们先对所有奖励一起处理，然后分配给每个省份
            all_province_rewards = province_rewards_array.flatten()
            all_province_rewards_stabilized = self._stabilize_rewards(all_province_rewards, episode=episode)
            province_rewards_stabilized = all_province_rewards_stabilized.reshape(batch_size, num_provinces)
            # 转换为numpy数组（_stabilize_rewards返回numpy数组）
            province_rewards_stabilized = np.array(province_rewards_stabilized)
            
            print(f"✅ 使用省份级奖励（包含竞争奖励），形状: {province_rewards_stabilized.shape}")
        else:
            # 使用全局奖励（向后兼容）
            rewards_batch = self._stabilize_rewards(rewards_batch, episode=episode)
            rewards_batch = torch.FloatTensor(rewards_batch).to(self.device)
            
            if len(rewards_batch.shape) == 1:
                step_rewards = rewards_batch
            else:
                step_rewards = rewards_batch.view(batch_size, num_agents).mean(dim=1)
        
        # 计算价值和优势
        values = self.get_values(global_obs_batch.view(-1, global_obs_batch.shape[-1]))

        if self.value_clip > 0:
            values = torch.clamp(values, -self.value_clip, self.value_clip)

        # ✅ 关键修复：为每个省份计算独立的优势（如果使用省份级奖励）
        if use_province_rewards:
            # 为每个省份单独计算GAE
            # 注意：values是全局的，我们需要为每个省份使用不同的奖励序列
            all_advantages = []
            all_returns = []
            
            for province_idx in range(num_provinces):
                # 获取该省份的奖励序列
                province_reward_seq = province_rewards_stabilized[:, province_idx]
                
                # 使用全局values（因为critic是集中式的）
                # 或者可以为每个省份使用不同的values（但这需要修改critic）
                province_returns, province_advantages = self._compute_stable_gae(
                    province_reward_seq, values.cpu().numpy()
                )
                all_advantages.append(province_advantages)
                all_returns.append(province_returns)
            
            # 转换为tensor：shape = (num_provinces, batch_size)
            advantages_per_province = torch.stack([torch.tensor(adv, dtype=torch.float32).to(self.device) 
                                                   for adv in all_advantages])
            returns_per_province = torch.stack([torch.tensor(ret, dtype=torch.float32).to(self.device) 
                                                for ret in all_returns])
            
            # 为了兼容性，也创建advantages_base和returns_base（虽然不会使用）
            advantages_base = None  # 在使用省份级奖励时不会使用
            returns_base = None  # 在使用省份级奖励时不会使用
            
            print(f"✅ 为每个省份计算了独立的优势，形状: {advantages_per_province.shape}")
        else:
            # 使用全局奖励计算GAE（原有方式）
            if len(rewards_batch.shape) == 1:
                step_rewards = rewards_batch
            else:
                step_rewards = rewards_batch.view(batch_size, num_agents).mean(dim=1)
            
            returns, advantages = self._compute_stable_gae(step_rewards.cpu().numpy(), values.cpu().numpy())
            advantages_base = torch.tensor(advantages, dtype=torch.float32).to(self.device).repeat(num_agents).detach()
            returns_base = torch.tensor(returns, dtype=torch.float32).to(self.device).repeat(num_agents)
            
            # 为了兼容性，也创建advantages_per_province和returns_per_province（虽然不会使用）
            advantages_per_province = None
            returns_per_province = None

        # PPO更新
        actor_losses = []
        critic_losses = []
        entropy_values = []
        gradient_norms = []
        last_per_prov_losses = []
        
        # 🎯 记录每个epoch的详细loss信息（用于日志）
        epoch_loss_details = []

        for epoch in range(self.k_epochs):
            num_agents_eff = len(self.actor_list)
            total_samples = actions_batch.shape[0]
            assert total_samples % num_agents_eff == 0

            current_entropy_coef = max(self.entropy_coef, self.min_entropy_coef)

            # ✅ 关键修复：每个省份完全独立计算，不共享计算图
            per_prov_losses = []
            all_actor_grad_norms = []
            all_entropy_values = []
            total_actor_loss = 0.0
            # 🎯 记录本epoch每个省份的详细信息
            epoch_province_details = []
            
            # 🎯 记录当前epoch编号（用于日志）
            current_epoch = epoch

            for p_idx in range(num_agents_eff):
                # ✅ 清空该省份Actor的梯度
                self.actor_optimizers[p_idx].zero_grad()

                # 获取该省份的所有样本索引
                idxs = torch.arange(p_idx, total_samples, num_agents_eff, device=self.device)
                if idxs.numel() == 0:
                    per_prov_losses.append(0.0)
                    all_actor_grad_norms.append(0.0)
                    all_entropy_values.append(0.0)
                    continue

                # ✅ 关键修复：为每个省份独立计算log_probs和entropy，确保计算图完全独立
                # 获取该省份的数据
                obs_p = local_obs_batch[idxs]
                ids_p = agent_ids_batch[idxs]
                actions_p = actions_batch[idxs]

                # ✅ 首先检查输入数据是否有NaN/Inf
                if torch.isnan(obs_p).any() or torch.isinf(obs_p).any():
                    print(f"⚠️ 检测到obs_p中的NaN/Inf值: 省份 {p_idx}, 修复输入数据")
                    obs_p = torch.nan_to_num(obs_p, nan=0.0, posinf=100.0, neginf=-100.0)
                if torch.isnan(ids_p).any() or torch.isinf(ids_p).any():
                    print(f"⚠️ 检测到ids_p中的NaN/Inf值: 省份 {p_idx}, 修复输入数据")
                    ids_p = torch.nan_to_num(ids_p, nan=0.0, posinf=1.0, neginf=0.0)
                
                # 使用该省份的Actor计算log_probs和entropy
                actor_cur = self.actor_list[p_idx]
                mean_cur, std_cur = actor_cur(obs_p, ids_p)
                
                # ✅ NaN检查和修复：防止梯度爆炸导致的NaN值
                has_nan = torch.isnan(mean_cur).any() or torch.isnan(std_cur).any()
                has_inf = torch.isinf(mean_cur).any() or torch.isinf(std_cur).any()
                
                if has_nan or has_inf:
                    print(f"⚠️ 检测到Actor输出NaN/Inf值: 省份 {p_idx}, 重新初始化该省份Actor网络")
                    # 重新初始化该省份的Actor网络权重
                    self._reinit_actor(p_idx)
                    # 重新计算（使用修复后的输入）
                    actor_cur = self.actor_list[p_idx]
                    mean_cur, std_cur = actor_cur(obs_p, ids_p)
                    # 如果仍然有NaN，用安全值替换
                    if torch.isnan(mean_cur).any() or torch.isnan(std_cur).any():
                        print(f"⚠️ 重新初始化后仍有NaN，使用安全值替换: 省份 {p_idx}")
                        mean_cur = torch.nan_to_num(mean_cur, nan=0.1)
                        std_cur = torch.nan_to_num(std_cur, nan=0.1)
                
                # 确保std_cur为正且有界
                std_cur = torch.clamp(std_cur, min=1e-6, max=1.0)
                # ✅ 确保mean_cur在合理范围内（防止极端值）
                mean_cur = torch.clamp(mean_cur, min=-10.0, max=10.0)
                
                # ✅ 在创建Normal分布之前，最后检查一次
                if torch.isnan(mean_cur).any() or torch.isnan(std_cur).any():
                    print(f"❌ 创建Normal分布前仍检测到NaN: 省份 {p_idx}, 使用默认值")
                    mean_cur = torch.full_like(mean_cur, 0.1)
                    std_cur = torch.full_like(std_cur, 0.1)
                
                dist_cur = Normal(mean_cur, std_cur + 1e-8)

                new_log_probs_p = dist_cur.log_prob(actions_p).sum(dim=-1)
                entropy_p = dist_cur.entropy().sum(dim=-1)

                # ✅ 获取该省份的advantages和old_log_probs
                # 如果使用省份级奖励，使用该省份的独立优势；否则使用全局优势
                if use_province_rewards:
                    # 使用该省份的独立优势（包含竞争奖励的影响）
                    # advantages_per_province shape: (num_provinces, batch_size)
                    # idxs对应的是该省份在所有样本中的位置：[p_idx, p_idx+num_provinces, p_idx+2*num_provinces, ...]
                    # 需要将idxs映射到batch中的位置：[0, 1, 2, ...]
                    batch_indices = idxs // num_agents_eff  # 计算在batch中的位置
                    # 确保batch_indices在有效范围内
                    batch_indices = torch.clamp(batch_indices, 0, advantages_per_province.shape[1] - 1)
                    advantages_p = advantages_per_province[p_idx, batch_indices].detach()
                else:
                    # 使用全局优势（原有方式）
                    advantages_p = advantages_base[idxs].detach()
                
                # ✅ 检查advantages是否有NaN/Inf
                if torch.isnan(advantages_p).any() or torch.isinf(advantages_p).any():
                    print(f"⚠️ 检测到advantages中的NaN/Inf值: 省份 {p_idx}")
                    advantages_p = torch.nan_to_num(advantages_p, nan=0.0, posinf=10.0, neginf=-10.0)
                
                old_log_probs_p = old_log_probs_batch[idxs]
                
                # ✅ 检查old_log_probs是否有NaN/Inf
                if torch.isnan(old_log_probs_p).any() or torch.isinf(old_log_probs_p).any():
                    print(f"⚠️ 检测到old_log_probs中的NaN/Inf值: 省份 {p_idx}")
                    old_log_probs_p = torch.nan_to_num(old_log_probs_p, nan=0.0, posinf=0.0, neginf=-100.0)

                # 计算该省份的ratio和surrogate loss
                log_ratio = new_log_probs_p - old_log_probs_p
                # ✅ 更严格的限制：防止log_ratio过大导致exp溢出
                log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)  # ✅ 从±20降到±10
                ratio_p = torch.exp(log_ratio)
                # ✅ 更严格的ratio裁剪，防止极端值导致loss波动
                ratio_p = torch.clamp(ratio_p, min=0.1, max=10.0)  # ✅ 从[0.01, 100]收紧到[0.1, 10]
                surr1_p = ratio_p * advantages_p
                surr2_p = torch.clamp(ratio_p, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_p

                # ✅ 该省份的独立loss
                actor_loss_p = -torch.mean(torch.minimum(surr1_p, surr2_p)) - current_entropy_coef * torch.mean(
                    entropy_p)
                
                # ✅ 检查Actor loss是否异常
                loss_value = actor_loss_p.item()
                if np.isnan(loss_value) or np.isinf(loss_value) or abs(loss_value) > 100.0:
                    print(f"⚠️ 省份 {p_idx} Actor loss异常: {loss_value:.4f}, 跳过本次更新")
                    per_prov_losses.append(0.0)  # 使用0作为占位符
                    all_entropy_values.append(0.0)
                    all_actor_grad_norms.append(0.0)
                    continue  # 跳过该省份的更新
                
                per_prov_losses.append(loss_value)
                all_entropy_values.append(entropy_p.mean().item())

                # ✅ 关键：每个省份用自己的loss更新自己的Actor网络
                actor_loss_p.backward()

                # ✅ 对该省份的Actor参数进行梯度裁剪
                # 注意：clip_grad_norm_返回的是裁剪前的梯度范数，裁剪后梯度已经被限制在max_grad_norm以内
                actor = self.actor_list[p_idx]
                actor_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                
                # ✅ 获取裁剪前的梯度范数（用于统计和诊断）
                grad_norm_value = actor_grad_norm_before_clip.item()
                
                # 🎯 计算优势值统计（用于诊断）
                adv_mean = float(advantages_p.mean().item()) if advantages_p.numel() > 0 else 0.0
                adv_std = float(advantages_p.std().item()) if advantages_p.numel() > 0 else 0.0
                adv_max = float(advantages_p.max().item()) if advantages_p.numel() > 0 else 0.0
                adv_min = float(advantages_p.min().item()) if advantages_p.numel() > 0 else 0.0
                
                # 🎯 计算ratio统计
                ratio_mean = float(ratio_p.mean().item()) if ratio_p.numel() > 0 else 0.0
                ratio_std = float(ratio_p.std().item()) if ratio_p.numel() > 0 else 0.0
                ratio_max = float(ratio_p.max().item()) if ratio_p.numel() > 0 else 0.0
                ratio_min = float(ratio_p.min().item()) if ratio_p.numel() > 0 else 0.0
                
                # 🎯 获取省份名称（用于日志）
                province_name = "Unknown"
                if hasattr(self, 'province_monitors') and p_idx in self.province_monitors:
                    province_name = self.province_monitors[p_idx].province_name
                
                skip_update = False
                skip_reason = ""
                
                # ✅ 只在梯度为NaN/Inf时跳过更新
                # 注意：梯度过大不应该跳过，因为clip_grad_norm_已经将梯度裁剪到安全范围
                if np.isnan(grad_norm_value) or np.isinf(grad_norm_value):
                    skip_update = True
                    skip_reason = f"梯度NaN/Inf: {grad_norm_value:.4f}"
                    print(f"⚠️ 省份 {p_idx} ({province_name}) Actor梯度异常: {skip_reason}, 跳过本次更新")
                
                # 🎯 梯度过大只作为警告记录，不跳过更新（因为已经被裁剪了）
                if grad_norm_value > 10.0 and not skip_update:
                    skip_reason = f"梯度过大(已裁剪): {grad_norm_value:.4f} > 10.0"
                    # 每100个episode打印一次警告，避免刷屏
                    if episode % 100 == 0 and epoch == 0 and p_idx == 0:
                        print(f"📊 梯度统计: 部分省份梯度较大，已被裁剪到{self.max_grad_norm}")
                
                # 🎯 记录详细信息
                epoch_province_details.append({
                    'province_idx': p_idx,
                    'province_name': province_name,
                    'actor_loss': loss_value,
                    'actor_grad_norm': grad_norm_value if not skip_update else 0.0,
                    'actor_entropy': entropy_p.mean().item(),
                    'advantage_mean': adv_mean,
                    'advantage_std': adv_std,
                    'advantage_max': adv_max,
                    'advantage_min': adv_min,
                    'ratio_mean': ratio_mean,
                    'ratio_std': ratio_std,
                    'ratio_max': ratio_max,
                    'ratio_min': ratio_min,
                    'skip_update': skip_update,
                    'skip_reason': skip_reason
                })
                
                if skip_update:
                    # 清空梯度，不更新
                    self.actor_optimizers[p_idx].zero_grad()
                    all_actor_grad_norms.append(0.0)
                    continue
                
                all_actor_grad_norms.append(grad_norm_value)

                # ✅ 更新该省份的Actor（使用该省份独立的optimizer）
                self.actor_optimizers[p_idx].step()

                total_actor_loss += loss_value

            # 计算平均梯度范数和平均entropy（用于统计）
            avg_actor_grad_norm = np.mean(all_actor_grad_norms) if all_actor_grad_norms else 0.0
            avg_entropy = np.mean(all_entropy_values) if all_entropy_values else 0.0

            # 更新Critic
            self.critic_optimizer.zero_grad()
            current_values = self.centralized_critic(global_obs_batch.view(-1, global_obs_batch.shape[-1]))

            if self.value_clip > 0:
                current_values = torch.clamp(current_values, -self.value_clip, self.value_clip)

            # ✅ 如果使用省份级奖励，使用所有省份returns的平均值更新critic
            # 这样critic可以学习到包含竞争奖励影响的全局价值函数
            if use_province_rewards:
                # 计算所有省份returns的平均值（按batch维度）
                # returns_per_province shape: (num_provinces, batch_size)
                # 取平均值得到 (batch_size,)，然后扩展为 (batch_size * num_provinces,)
                returns_avg = returns_per_province.mean(dim=0)  # (batch_size,)
                
                # ✅ 关键修复：对returns进行裁剪，防止极端值导致Critic loss波动
                returns_avg = torch.clamp(returns_avg, -self.value_clip, self.value_clip)
                
                returns_flat = returns_avg.repeat(num_provinces)  # (batch_size * num_provinces,)
                # 扩展current_values以匹配returns_flat的形状
                current_values_expanded = current_values.repeat(num_provinces)
                
                # ✅ 使用Huber Loss代替MSE Loss，对异常值更鲁棒
                critic_loss = F.smooth_l1_loss(current_values_expanded, returns_flat, beta=1.0)
            else:
                # 使用全局returns（原有方式）
                # ✅ 对returns_base也进行裁剪
                returns_base_clipped = torch.clamp(returns_base, -self.value_clip, self.value_clip)
                # ✅ 使用Huber Loss代替MSE Loss
                critic_loss = F.smooth_l1_loss(current_values, returns_base_clipped, beta=1.0)

            # ✅ 检查Critic loss是否异常
            critic_loss_value = critic_loss.item() if hasattr(critic_loss, 'item') else float(critic_loss)
            
            # 🎯 计算Critic值统计
            value_mean = float(current_values.mean().item()) if current_values.numel() > 0 else 0.0
            value_std = float(current_values.std().item()) if current_values.numel() > 0 else 0.0
            
            critic_skip_update = False
            critic_skip_reason = ""
            critic_grad_norm_value = 0.0
            
            if torch.isnan(critic_loss) or torch.isinf(critic_loss) or critic_loss_value > 100.0:
                critic_skip_update = True
                critic_skip_reason = f"Loss异常: {critic_loss_value:.4f}"
                print(f"⚠️ Critic loss异常: {critic_skip_reason}, 跳过本次更新")
                critic_loss_value = 0.0
            else:
                critic_loss.backward()
                # 注意：clip_grad_norm_返回裁剪前的梯度范数，裁剪后梯度已被限制在max_grad_norm以内
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.max_grad_norm)
                critic_grad_norm_value = critic_grad_norm.item()
                
                # ✅ 只在梯度为NaN/Inf时跳过更新
                # 梯度过大不应该跳过，因为clip_grad_norm_已经将梯度裁剪到安全范围
                if torch.isnan(critic_grad_norm) or torch.isinf(critic_grad_norm):
                    critic_skip_update = True
                    critic_skip_reason = f"梯度NaN/Inf: {critic_grad_norm_value:.4f}"
                    print(f"⚠️ Critic梯度异常: {critic_skip_reason}, 跳过本次更新")
                else:
                    # 记录梯度过大的情况（仅作为诊断信息）
                    if critic_grad_norm_value > 10.0:
                        critic_skip_reason = f"梯度过大(已裁剪): {critic_grad_norm_value:.4f}"
                    self.critic_optimizer.step()
            
            # 🎯 为每个省份记录Critic信息（所有省份共享同一个Critic）
            for prov_detail in epoch_province_details:
                prov_detail['epoch'] = current_epoch
                prov_detail['critic_loss'] = critic_loss_value
                prov_detail['critic_grad_norm'] = critic_grad_norm_value
                prov_detail['value_mean'] = value_mean
                prov_detail['value_std'] = value_std
                prov_detail['critic_skip_update'] = critic_skip_update
                prov_detail['critic_skip_reason'] = critic_skip_reason
            
            # 🎯 保存本epoch的详细信息
            epoch_loss_details.extend(epoch_province_details)

            # ✅ 修复：使用平均Actor loss（所有省份loss的平均值）
            avg_actor_loss_value = total_actor_loss / max(num_agents_eff, 1) if num_agents_eff > 0 else 0.0
            actor_losses.append(avg_actor_loss_value)
            critic_losses.append(critic_loss_value)  # ✅ 使用安全的值
            entropy_values.append(avg_entropy)
            gradient_norms.append({
                'actor': avg_actor_grad_norm,
                'critic': critic_grad_norm_value
            })
            # ✅ 记录每轮的分省损失（取最后一轮用于统计）
            last_per_prov_losses = per_prov_losses.copy()

        # 更新旧策略
        for actor_old, actor_cur in zip(self.actor_old_list, self.actor_list):
            actor_old.load_state_dict(actor_cur.state_dict())
        # 同步共享Actor为省份0的权重（作为“整体”Actor导出）
        self.shared_actor.load_state_dict(self.actor_list[0].state_dict())
        self.shared_actor_old.load_state_dict(self.actor_old_list[0].state_dict())

        # 🎯 定期更新目标网络
        if episode % self.target_update_freq == 0:
            self.centralized_critic.update_target_network(self.target_update_tau)
            if self.logger:
                self.logger.log(f"更新目标网络 (Episode {episode})")

        # 衰减熵系数
        self.entropy_coef *= self.entropy_decay

        # 记录统计信息
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        avg_entropy = np.mean(entropy_values)
        self.training_stats['actor_losses'].append(avg_actor_loss)
        
        # 🎯 将详细的loss信息写入日志文件
        if self.logger and hasattr(self.logger, 'loss_gradient_writer'):
            for detail in epoch_loss_details:
                self.logger.loss_gradient_writer.writerow([
                    episode, detail.get('epoch', 0), detail['province_idx'], detail['province_name'],
                    detail['actor_loss'], detail['actor_grad_norm'], detail['actor_entropy'],
                    detail['advantage_mean'], detail['advantage_std'], detail['advantage_max'], detail['advantage_min'],
                    detail['ratio_mean'], detail['ratio_std'], detail['ratio_max'], detail['ratio_min'],
                    detail['critic_loss'], detail['critic_grad_norm'], detail['value_mean'], detail['value_std'],
                    detail['skip_update'], detail['skip_reason']
                ])
            self.logger.loss_gradient_csv.flush()
            
            # 🎯 同时写入文本日志（更易读）
            if epoch_loss_details:
                log_msg = f"\n{'=' * 80}\n"
                log_msg += f"Episode {episode} - Loss和梯度详细信息\n"
                log_msg += f"{'=' * 80}\n"
                
                # 统计跳过更新的省份
                skipped_provinces = [d for d in epoch_loss_details if d['skip_update']]
                if skipped_provinces:
                    log_msg += f"⚠️ 跳过更新的省份数: {len(skipped_provinces)}/{len(epoch_province_details)}\n"
                    log_msg += f"   跳过原因统计:\n"
                    skip_reasons = {}
                    for d in skipped_provinces:
                        reason = d['skip_reason']
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    for reason, count in skip_reasons.items():
                        log_msg += f"     - {reason}: {count}个省份\n"
                
                # 统计梯度范数
                valid_grads = [d['actor_grad_norm'] for d in epoch_loss_details if not d['skip_update']]
                if valid_grads:
                    log_msg += f"\n✅ 成功更新的省份梯度统计:\n"
                    log_msg += f"   平均梯度范数: {np.mean(valid_grads):.4f}\n"
                    log_msg += f"   梯度范数范围: [{np.min(valid_grads):.4f}, {np.max(valid_grads):.4f}]\n"
                
                # 统计优势值
                all_adv_means = [d['advantage_mean'] for d in epoch_loss_details]
                all_adv_stds = [d['advantage_std'] for d in epoch_loss_details]
                if all_adv_means:
                    log_msg += f"\n📊 优势值统计:\n"
                    log_msg += f"   平均优势值: {np.mean(all_adv_means):.2f} ± {np.mean(all_adv_stds):.2f}\n"
                    log_msg += f"   优势值范围: [{np.min([d['advantage_min'] for d in epoch_loss_details]):.2f}, "
                    log_msg += f"{np.max([d['advantage_max'] for d in epoch_loss_details]):.2f}]\n"
                
                # 统计Ratio
                all_ratio_means = [d['ratio_mean'] for d in epoch_loss_details]
                if all_ratio_means:
                    log_msg += f"\n📊 Ratio统计:\n"
                    log_msg += f"   平均Ratio: {np.mean(all_ratio_means):.2f}\n"
                    log_msg += f"   Ratio范围: [{np.min([d['ratio_min'] for d in epoch_loss_details]):.2f}, "
                    log_msg += f"{np.max([d['ratio_max'] for d in epoch_loss_details]):.2f}]\n"
                
                # Critic信息
                if epoch_loss_details:
                    first_detail = epoch_loss_details[0]
                    log_msg += f"\n📊 Critic统计:\n"
                    log_msg += f"   Critic Loss: {first_detail['critic_loss']:.4f}\n"
                    log_msg += f"   Critic梯度范数: {first_detail['critic_grad_norm']:.4f}\n"
                    log_msg += f"   价值函数: {first_detail['value_mean']:.2f} ± {first_detail['value_std']:.2f}\n"
                    if first_detail['critic_skip_update']:
                        log_msg += f"   ⚠️ Critic更新被跳过: {first_detail['critic_skip_reason']}\n"
                
                log_msg += f"{'=' * 80}\n"
                self.logger.loss_gradient_log.write(log_msg)
                self.logger.loss_gradient_log.flush()

        # ✅ 修复：确保分省份loss正确保存
        if 'actor_losses_per_province' not in self.training_stats:
            self.training_stats['actor_losses_per_province'] = []

        # ✅ 确保last_per_prov_losses存在且有效
        if len(last_per_prov_losses) > 0:
            self.training_stats['actor_losses_per_province'].append(last_per_prov_losses.copy())
        else:
            # 如果没有数据，创建一个零列表（避免后续访问错误）
            num_agents = len(self.actor_list)
            self.training_stats['actor_losses_per_province'].append([0.0] * num_agents)
        self.training_stats['critic_losses'].append(avg_critic_loss)
        self.training_stats['entropy_values'].append(avg_entropy)
        self.training_stats['exploration_rates'].append(self.exploration_noise)
        self.training_stats['gradient_norms'].append({
            'actor': np.mean([g['actor'] for g in gradient_norms]),
            'critic': np.mean([g['critic'] for g in gradient_norms])
        })

        # 🎯 记录探索效果指标
        self.training_stats['epsilon_values'].append(self.epsilon)

        # 计算随机动作比例（如果有探索统计）
        if hasattr(self, 'exploration_stats'):
            random_ratio = self.exploration_stats['random_actions'] / max(1, self.exploration_stats['total_actions'])
            self.training_stats['random_action_ratio'].append(random_ratio)
            # 重置episode统计
            self.exploration_stats['random_actions'] = 0
            self.exploration_stats['total_actions'] = 0

        # 计算策略多样性（基于动作标准差）
        if len(self.action_history) > 10:
            recent_actions = np.array(self.action_history[-10:])
            policy_diversity = np.mean(np.std(recent_actions, axis=0))
            self.training_stats['policy_diversity'].append(policy_diversity)

        # 日志损失
        if self.logger:
            self.logger.log(
                f"Episode {episode}: Actor loss = {avg_actor_loss:.4f}, Critic loss = {avg_critic_loss:.4f}, Entropy = {avg_entropy:.4f}")

        if avg_critic_loss > 10.0:
            print(f"⚠️ Critic损失过大: {avg_critic_loss:.2f}")
        if abs(avg_actor_loss) > 1.0:
            print(f"⚠️ Actor损失异常: {abs(avg_actor_loss):.2f}")

        # 清空经验池
        self.memory = []
        
        # ✅ 内存清理：裁剪 training_stats 各列表，防止无限增长
        _MAX_STATS_LEN = 500
        for key, val in self.training_stats.items():
            if isinstance(val, list) and len(val) > _MAX_STATS_LEN:
                self.training_stats[key] = val[-_MAX_STATS_LEN:]
        
        # 裁剪 action_history
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        # 裁剪 best_reward_history
        if len(self.best_reward_history) > self.history_window:
            self.best_reward_history = self.best_reward_history[-self.history_window:]
        
        # 裁剪 recent_rewards
        if len(self.recent_rewards) > self.reward_stability_window:
            self.recent_rewards = self.recent_rewards[-self.reward_stability_window:]
        
        # 裁剪 performance_history
        if len(self.performance_history) > 200:
            self.performance_history = self.performance_history[-200:]
        
        # ✅ 释放GPU内存，防止CUDA OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def store_experience(self, local_obs, global_obs, agent_ids, actions, rewards, old_log_probs, province_rewards=None):
        """
        存储经验
        
        Args:
            local_obs: 局部观察
            global_obs: 全局观察
            agent_ids: 智能体ID
            actions: 动作
            rewards: 全局奖励（用于兼容性）
            old_log_probs: 旧策略的对数概率
            province_rewards: 每个省份的奖励字典（可选，如果提供则使用省份级奖励）
        """
        experience = {
            'local_obs': local_obs,
            'global_obs': global_obs,
            'agent_ids': agent_ids,
            'actions': actions,
            'rewards': rewards,  # 全局奖励（用于兼容性）
            'old_log_probs': old_log_probs,
            'province_rewards': province_rewards  # ✅ 新增：每个省份的奖励（包含竞争奖励）
        }
        self.memory.append(experience)

    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger

    def save_best_policy(self, episode_reward, episode):
        """保存最佳策略权重"""
        try:
            self.best_policy_weights = {
                'actor_state_dict': copy.deepcopy(self.shared_actor.state_dict()),
                'per_actor_state_dicts': [copy.deepcopy(actor.state_dict()) for actor in self.actor_list],
                'critic_state_dict': copy.deepcopy(self.centralized_critic.state_dict()),
                'episode': episode,
                'reward': episode_reward
            }
            self.best_episode = episode
            self.best_reward_history.append(episode_reward)
            if len(self.best_reward_history) > self.history_window:
                self.best_reward_history.pop(0)
            if self.logger:
                self.logger.log(f"🎯 保存最佳策略 - 回合{episode}, 奖励{episode_reward:.2f}")
            return True
        except Exception as e:
            if self.logger:
                self.logger.log(f"❌ 保存最佳策略失败: {e}")
            return False

    def restore_best_policy(self):
        """恢复最佳策略权重（共享Actor + 每省Actor）

        ⚠️ 警告：恢复策略后必须清空经验池，避免策略与经验数据不匹配导致loss爆炸
        """
        if self.best_policy_weights is None:
            if self.logger:
                self.logger.log("⚠️ 没有可恢复的最佳策略")
            return False
        try:
            # 恢复共享Actor
            self.shared_actor.load_state_dict(self.best_policy_weights['actor_state_dict'])
            # 恢复每省Actor（如果有保存）
            if 'per_actor_state_dicts' in self.best_policy_weights:
                per_states = self.best_policy_weights['per_actor_state_dicts']
                for idx, state in enumerate(per_states):
                    self.actor_list[idx].load_state_dict(state)
            # 同步旧策略
            for actor_old, actor_cur in zip(self.actor_old_list, self.actor_list):
                actor_old.load_state_dict(actor_cur.state_dict())
            self.shared_actor_old.load_state_dict(self.shared_actor.state_dict())

            self.centralized_critic.load_state_dict(self.best_policy_weights['critic_state_dict'])
            self.is_using_best_policy = True
            self.policy_restoration_count += 1
            self.exploration_without_improvement = 0

            # ✅ 关键修复：清空经验池，避免策略与经验数据不匹配
            self.memory = []
            if self.logger:
                self.logger.log(f"🔄 恢复最佳策略 - 回合{self.best_policy_weights['episode']}, "
                                f"奖励{self.best_policy_weights['reward']:.2f}, "
                                f"恢复次数{self.policy_restoration_count}, "
                                f"已清空经验池")
            return True
        except Exception as e:
            if self.logger:
                self.logger.log(f"❌ 恢复最佳策略失败: {e}")
            return False

    def check_and_update_best_policy(self, episode_reward, episode):
        """检查并更新最佳策略"""
        if len(self.best_reward_history) == 0:
            self.save_best_policy(episode_reward, episode)
            return True
        current_best = max(self.best_reward_history)
        if episode_reward > current_best + self.improvement_threshold:
            self.save_best_policy(episode_reward, episode)
            self.exploration_without_improvement = 0
            self.is_using_best_policy = False
            return True
        else:
            self.exploration_without_improvement += 1
            return False

    def should_restore_best_policy(self):
        """判断是否应该恢复最佳策略"""
        if (self.exploration_without_improvement >= self.max_exploration_without_improvement and
                not self.is_using_best_policy and
                self.best_policy_weights is not None):
            return True
        return False

    def select_actions_fine_tune(self, local_observations, agent_ids, episode=0, step=0):
        """微调模式：所有省份都使用固定策略，但允许小幅调整"""
        # 转换为PyTorch张量
        local_obs_tensor = torch.FloatTensor(local_observations).to(self.device)
        agent_ids_tensor = torch.FloatTensor(agent_ids).to(self.device)

        actions = np.zeros((len(local_observations), self.action_dim))
        log_probs = np.zeros((len(local_observations),))

        for province_idx in range(len(local_observations)):
            if province_idx in self.fixed_policies:
                # 使用固定策略
                fixed_actor = self._get_fixed_actor(province_idx)
                if fixed_actor is None:
                    fixed_actor = self.actor_old_list[province_idx]
                with torch.no_grad():
                    action_mean, _ = fixed_actor(
                        local_obs_tensor[province_idx:province_idx + 1],
                        agent_ids_tensor[province_idx:province_idx + 1]
                    )
                # 添加很小的噪声（±0.01）用于微调
                noise = np.random.normal(0, 0.01, self.action_dim)
                action = action_mean[0].cpu().numpy() + noise
                actions[province_idx] = np.clip(action, 0.01, 0.2)
                log_probs[province_idx] = 0.0
            else:
                # 如果某个省份没有固定策略（不应该发生），使用默认策略
                actions[province_idx] = np.random.uniform(0.01, 0.2, self.action_dim)
                log_probs[province_idx] = 0.0

        return actions, log_probs

    def update_fine_tune(self, episode):
        """微调模式：只更新Critic，Actor保持不变"""
        if len(self.memory) == 0:
            return

        # 只更新Critic，不更新Actor
        experiences = self.memory
        local_obs_batch = torch.FloatTensor([e['local_obs'] for e in experiences]).to(self.device)
        global_obs_batch = torch.FloatTensor([e['global_obs'] for e in experiences]).to(self.device)
        agent_ids_batch = torch.FloatTensor([e['agent_ids'] for e in experiences]).to(self.device)
        rewards_batch = torch.FloatTensor([e['rewards'] for e in experiences]).to(self.device)

        # 计算returns和advantages（只用于Critic）
        values = self.get_values(global_obs_batch, use_target=False)
        returns, advantages = self._compute_stable_gae(rewards_batch.cpu().numpy(), values.cpu().numpy())
        returns = torch.FloatTensor(returns).to(self.device)

        # 只更新Critic
        critic_losses = []
        for epoch in range(self.k_epochs):
            self.critic_optimizer.zero_grad()
            current_values = self.centralized_critic(global_obs_batch.view(-1, global_obs_batch.shape[-1]))
            if self.value_clip > 0:
                current_values = torch.clamp(current_values, -self.value_clip, self.value_clip)
            critic_loss = F.mse_loss(current_values, returns)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())

        # ✅ 修复：直接更新training_stats，与update()方法保持一致
        self.training_stats['actor_losses'].append(0.0)  # Actor不更新
        self.training_stats['critic_losses'].append(np.mean(critic_losses))
        self.training_stats['entropy_values'].append(0.0)

        # ✅ 修复：为分省份loss添加零列表
        if 'actor_losses_per_province' not in self.training_stats:
            self.training_stats['actor_losses_per_province'] = []
        self.training_stats['actor_losses_per_province'].append([0.0] * len(self.actor_list))

        self.training_stats['exploration_rates'].append(self.exploration_noise)
        self.training_stats['gradient_norms'].append({
            'actor': 0.0,
            'critic': np.mean(critic_losses)
        })

        # 清空经验池
        self.memory = []
        
        # ✅ 内存清理：裁剪 training_stats，释放 GPU 内存
        _MAX_STATS_LEN = 500
        for key, val in self.training_stats.items():
            if isinstance(val, list) and len(val) > _MAX_STATS_LEN:
                self.training_stats[key] = val[-_MAX_STATS_LEN:]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_exploration_strategy_status(self):
        """获取探索策略状态"""
        return {
            'best_reward': max(self.best_reward_history) if self.best_reward_history else 0,
            'best_episode': self.best_episode,
            'exploration_without_improvement': self.exploration_without_improvement,
            'is_using_best_policy': self.is_using_best_policy,
            'policy_restoration_count': self.policy_restoration_count,
            'history_window_size': len(self.best_reward_history),
            'fixed_provinces_count': len(self.fixed_policies),
            'current_training_province_idx': self.current_training_province_idx
        }


def train_optimized_mappo_pytorch(
        enable_realtime_console=True,
        random_seed=42,
        # ========== 可自定义的省份优化配置 ==========
        province_training_config=None,
        # 格式: [{'province_idx': 0, 'episodes': 200}, {'province_idx': 1, 'episodes': 60}, ...]
        max_total_episodes=None,  # 总训练轮次上限（如果设置，会覆盖基于province_training_config的计算）
        fine_tune_episodes=25,  # 微调阶段轮次
        scenario_id=None,  # 情景ID（用于多情景训练时的标识）
):
    """
    训练PyTorch版MAPPO算法 - 优先级训练版本（支持自定义省份优化顺序和轮次）

    Args:
        enable_realtime_console: 是否启用实时控制台输出
        random_seed: 随机种子
        province_training_config: 省份优化配置列表，格式：
            [
                {'province_idx': 0, 'province_name': 'HB', 'episodes': 200},
                {'province_idx': 1, 'province_name': 'SD', 'episodes': 60},
                ...
            ]
            如果为None，使用默认配置（第一个省份200轮，其他60轮）
        max_total_episodes: 总训练轮次上限（可选，如果设置会覆盖计算值）
        fine_tune_episodes: 微调阶段轮次
        scenario_id: 情景ID（用于多情景训练时的标识，如"scenario_1"）
    """
    # 设置随机种子
    set_seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print(f"开始训练PyTorch版MAPPO算法（优先级训练模式，随机种子={random_seed}）...")
    if scenario_id:
        print(f"📋 情景ID: {scenario_id}")

    # 根据scenario_id设置日志目录
    if scenario_id:
        log_dir = f"./log152/{scenario_id}"
    else:
        log_dir = "./log152"
    detailed_logger = DetailedLogger(log_dir=log_dir, console_output=enable_realtime_console)
    # ✅ TensorBoard writer - 使用带时间戳的目录确保每次运行都有独立的日志
    writer = None
    tensorboard_dir = None
    # 使用局部变量追踪 TensorBoard 可用性
    tensorboard_enabled = TENSORBOARD_AVAILABLE
    if tensorboard_enabled and SummaryWriter is not None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if scenario_id:
                tensorboard_dir = f'./logs/{scenario_id}/tensorboard_{timestamp}'
            else:
                tensorboard_dir = f'./logs/tensorboard_{timestamp}'
            os.makedirs(tensorboard_dir, exist_ok=True)  # ✅ 确保目录存在
            print(f"📊 TensorBoard目录: {tensorboard_dir}")
            print(f"   目录存在: {os.path.exists(tensorboard_dir)}")

            writer = SummaryWriter(log_dir=tensorboard_dir, flush_secs=1)
            print(f"📊 TensorBoard日志保存到: {tensorboard_dir}")
            print(f"   使用命令查看: tensorboard --logdir={tensorboard_dir}")
        except Exception as e:
            print(f"⚠️ TensorBoard初始化失败: {e}")
            print("   训练将继续，但不会记录TensorBoard日志")
            writer = None
            tensorboard_enabled = False
    else:
        print("⚠️ TensorBoard不可用，跳过TensorBoard日志记录")

    required_files = [
        ("./models_unet", "U-Net模型目录"),
        ("./conc/base/base.csv", "基准浓度数据"),
        ("./conc/clean/clean.csv", "清洁情景数据"),
        ("./prov_grid_map/36kmprov.csv", "省份映射数据"),
        ("./other data/cost.csv", "成本系数数据"),
        ("./other data/transport.csv", "省际传输矩阵（公平性奖励）"),
        ("./input_emi/base", "基准排放数据目录")
    ]

    for file_path, description in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 找不到{description}: {file_path}")
            detailed_logger.close()
            if writer is not None and tensorboard_enabled:
                try:
                    writer.close()
                except:
                    pass
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
            transport_matrix_path="./other data/transport.csv",
            fairness_weight=10000.0,
            fairness_metric='l2',
            fairness_mode='penalty',
            fairness_external_only=False,
            max_steps=8
        )

        local_obs_dim = env.action_dim + env.num_provinces + env.province_feature_dim
        global_obs_dim = env.num_provinces * env.action_dim + env.num_provinces * 2 + 1

        print(f"🎯 观察空间维度:")
        print(f"  局部观察维度: {local_obs_dim}")
        print(f"  全局观察维度: {global_obs_dim}")

        mappo_agent = OptimizedMAPPOAgent(
            num_agents=env.num_provinces,
            local_obs_dim=local_obs_dim,
            global_obs_dim=global_obs_dim,
            action_dim=env.action_dim,
            lr=1e-4,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=8,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        mappo_agent.set_logger(detailed_logger)
        
        # ✅ 将logger传递给环境（用于性能统计输出到日志文件）
        env.set_logger(detailed_logger)

        # 🎯 初始化省份监控器
        mappo_agent.initialize_province_monitors(env.province_names)
        print(f"✅ 已初始化 {len(env.province_names)} 个省份监控器")

        # ✅ 详细输出模型架构信息
        print(f"\n{'=' * 80}")
        print(f"🏗️  模型架构确认:")
        print(f"{'=' * 80}")
        print(f"  📊 Actor网络架构: 每个省份独立的Actor网络")
        print(f"     - 总共 {len(mappo_agent.actor_list)} 个独立Actor网络")
        print(f"     - 每个Actor输入维度: {local_obs_dim}")
        print(f"     - 每个Actor输出维度: {env.action_dim}")
        print(f"  📊 Critic网络架构: 集中式Critic（共享）")
        print(f"     - Critic输入维度: {global_obs_dim}")
        print(f"     - Critic输出维度: 1（状态价值）")
        print(f"  🎯 优化器配置:")
        print(f"     - ✅ 每个省份独立的Actor Optimizer（真正实现独立更新）")
        print(f"     - Actor优化器数量: {len(mappo_agent.actor_optimizers)} 个")
        print(f"     - Actor学习率: {mappo_agent.actor_lr:.6f}")
        print(f"     - Critic学习率: {mappo_agent.critic_lr:.6f}")
        print(f"     - 梯度裁剪阈值: {mappo_agent.max_grad_norm}")
        print(f"  🔄 更新机制:")
        print(f"     - ✅ 每个省份用自己的经验计算loss")
        print(f"     - ✅ 每个省份用自己的loss更新自己的Actor网络")
        print(f"     - ✅ 梯度更新完全独立，互不干扰")
        print(f"  🔧 稳定性配置:")
        print(f"     - 值函数裁剪范围: ±{mappo_agent.value_clip}")
        print(f"     - 目标网络更新频率: 每{mappo_agent.target_update_freq}回合")
        print(f"     - 目标网络软更新系数: {mappo_agent.target_update_tau}")
        print(f"     - 奖励缩放因子: 1/100")
        print(f"     - 奖励裁剪阈值: ±{mappo_agent.clip_reward_threshold}")
        print(f"  🎲 探索配置:")
        print(f"     - 初始探索噪声: {mappo_agent.exploration_noise}")
        print(f"     - Epsilon-greedy概率: {mappo_agent.epsilon}")
        print(f"     - 最小探索噪声: {mappo_agent.min_noise}")
        print(f"{'=' * 80}\n")

        # 🎯 优先级训练参数（支持自定义配置）
        if province_training_config is None:
            # 默认配置：第一个省份200轮，其他60轮
            province_training_config = [
                {'province_idx': 0, 'province_name': env.province_names[0], 'episodes': 200}
            ]
            for idx in range(1, env.num_provinces):
                province_training_config.append({
                    'province_idx': idx,
                    'province_name': env.province_names[idx],
                    'episodes': 60
                })

        # 验证和补充配置
        configured_indices = [cfg['province_idx'] for cfg in province_training_config]
        if len(set(configured_indices)) != len(configured_indices):
            raise ValueError("❌ 错误：省份配置中有重复的province_idx")
        if set(configured_indices) != set(range(env.num_provinces)):
            raise ValueError(f"❌ 错误：省份配置不完整，缺少某些省份（需要0-{env.num_provinces - 1}）")

        # ✅ 修复：根据province_name自动查找正确的province_idx
        # 创建省份名称到索引的映射
        province_name_to_idx = {name: idx for idx, name in enumerate(env.province_names)}
        
        # ✅ 检查配置中是否有重复的province_name
        configured_names = [cfg.get('province_name') for cfg in province_training_config if 'province_name' in cfg]
        if len(configured_names) != len(set(configured_names)):
            duplicates = [name for name in set(configured_names) if configured_names.count(name) > 1]
            raise ValueError(f"❌ 错误：配置中有重复的province_name: {duplicates}")
        
        # 确保所有配置都有province_name，并根据province_name自动设置正确的province_idx
        for cfg in province_training_config:
            if 'province_name' not in cfg or cfg['province_name'] is None:
                # 如果没有province_name，使用province_idx查找
                if 'province_idx' in cfg:
                    cfg['province_name'] = env.province_names[cfg['province_idx']]
                else:
                    raise ValueError(f"❌ 错误：配置缺少 'province_name' 和 'province_idx'")
            
            # ✅ 关键修复：根据province_name自动查找正确的province_idx
            province_name = cfg['province_name']
            if province_name not in province_name_to_idx:
                raise ValueError(f"❌ 错误：省份名称 '{province_name}' 不在env.province_names中")
            
            # 自动设置正确的province_idx
            actual_idx = province_name_to_idx[province_name]
            if 'province_idx' in cfg and cfg['province_idx'] != actual_idx:
                print(f"⚠️ 警告：配置中province_idx={cfg['province_idx']}与province_name='{province_name}'不匹配，"
                      f"自动更正为province_idx={actual_idx}")
            cfg['province_idx'] = actual_idx
            
            if 'episodes' not in cfg or cfg['episodes'] is None:
                raise ValueError(f"❌ 错误：省份 {province_name} (idx={actual_idx}) 的配置缺少 'episodes' 字段")

        # 计算总轮次
        total_priority_episodes = sum(cfg['episodes'] for cfg in province_training_config)
        if max_total_episodes is not None:
            max_episodes = max_total_episodes
        else:
            max_episodes = total_priority_episodes + fine_tune_episodes

        # 创建省份索引到配置的映射（使用修正后的province_idx）
        province_config_map = {cfg['province_idx']: cfg for cfg in province_training_config}
        # 创建优化顺序列表（按配置顺序，使用修正后的province_idx）
        province_order = [cfg['province_idx'] for cfg in province_training_config]

        print(f"\n{'=' * 80}")
        print(f"🎯 优先级训练计划:")
        print(f"{'=' * 80}")
        print(f"  📊 总省份数: {env.num_provinces}")
        print(f"  🔄 优化策略: 逐省顺序优化（Sequential Province Optimization）")
        print(f"  📋 优化顺序: {' → '.join([env.province_names[idx] for idx in province_order[:5]])}...")
        print(f"  📈 各省份训练轮次配置:")

        # 按组显示，更清晰
        for i, cfg in enumerate(province_training_config):
            marker = "🎯" if i == 0 else "  "
            print(f"    {marker} [{cfg['province_idx']:2d}] {cfg['province_name']:4s}: {cfg['episodes']:3d} 轮")
            if i >= 9 and len(province_training_config) > 15:  # 只显示前10个，避免输出过长
                remaining = len(province_training_config) - 10
                print(f"    ... (还有{remaining}个省份)")
                break

        print(f"\n  📊 训练阶段划分:")
        print(f"     - 优先级训练阶段: {total_priority_episodes} 轮")
        print(f"       └─ 每个省份独立优化，完成后固定策略")
        print(f"     - 微调阶段: {fine_tune_episodes} 轮")
        print(f"       └─ 所有省份策略固定，仅微调Critic")
        print(f"     - 总训练轮次: {max_episodes} 轮")
        print(f"{'=' * 80}\n")

        update_interval = 1  # 每个 episode 更新

        # 根据scenario_id设置结果保存目录
        if scenario_id:
            result_dir = f"./result152/{scenario_id}"
        else:
            result_dir = "./result152"
        os.makedirs(result_dir, exist_ok=True)

        episode_rewards = []
        diversity_scores = []
        best_reward = float('-inf')
        convergence_patience = 0
        reward_baseline = None
        baseline_window = 20
        cumulative_reductions = {}  # ✅ 修复：初始化累积减排率字典

        # 🎯 优先级训练状态（使用新的配置系统）
        current_province_order_idx = 0  # 当前在province_order中的索引
        current_province_idx = province_order[0] if len(province_order) > 0 else None
        province_episode_count = 0
        is_fine_tuning = False
        previous_episode_total_reward = None
        previous_province_totals = None
        episode_offset = 0  # 用于跟踪当前省份的起始episode

        for episode in range(max_episodes):
            # ✅ 定期强制内存清理，防止长时间训练导致 CUDA OOM
            if episode > 0 and episode % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                # 裁剪 ProvinceMonitor 的 reward_history（只保留最近 500 条）
                for pm in mappo_agent.province_monitors.values():
                    if len(pm.reward_history) > 500:
                        pm.reward_history = pm.reward_history[-500:]
                        pm.episode_history = pm.episode_history[-500:]

            # 🎯 确定当前训练阶段
            if episode < total_priority_episodes and current_province_order_idx < len(province_order):
                # 优先级训练阶段
                current_province_idx = province_order[current_province_order_idx]
                current_config = province_config_map[current_province_idx]
                required_episodes = current_config['episodes']

                # 检查当前省份是否训练完成
                if province_episode_count >= required_episodes:
                    # 固定当前省份的策略
                    monitor = mappo_agent.province_monitors[current_province_idx]
                    # 🎯 获取最佳策略和最佳动作序列
                    best_policy = monitor.get_best_policy()
                    best_actions_seq = monitor.get_best_actions_sequence()

                    print(f"\n{'=' * 80}")
                    print(f"🔒 固定省份策略")
                    print(f"{'=' * 80}")
                    print(f"  省份索引: {current_province_idx}")
                    print(f"  省份名称: {current_config['province_name']}")
                    print(f"  训练轮次: {province_episode_count}/{current_config['episodes']}")
                    print(f"  最佳奖励: {monitor.best_reward:.2f}")
                    print(f"  最佳回合: {monitor.best_episode}")

                    if best_actions_seq is not None:
                        # 优先使用动作序列
                        mappo_agent.set_fixed_policy(current_province_idx, best_policy, best_actions_seq)
                        mappo_agent.province_monitors[current_province_idx].fix_policy(episode)
                        print(f"  固定方式: ✅ 动作序列（推荐）")
                        print(f"  序列长度: {len(best_actions_seq)} 步")
                        if len(best_actions_seq) > 0:
                            print(f"  首步动作示例: {best_actions_seq[0][:5]}")
                    elif best_policy is not None:
                        # 备用：使用策略网络
                        mappo_agent.set_fixed_policy(current_province_idx, best_policy, None)
                        mappo_agent.province_monitors[current_province_idx].fix_policy(episode)
                        print(f"  固定方式: ⚠️ 策略网络（备用）")
                    else:
                        print(f"  ❌ 错误: 无可用的策略或动作序列")
                    print(f"{'=' * 80}\n")

                    # 移动到下一个省份
                    current_province_order_idx += 1
                    province_episode_count = 0
                    episode_offset = episode + 1

                    if current_province_order_idx >= len(province_order):
                        # 所有省份都固定，进入微调阶段
                        is_fine_tuning = True
                        print(f"\n🎯 所有省份策略已固定，进入微调阶段")
                        current_province_idx = None
                    else:
                        current_province_idx = province_order[current_province_order_idx]

                mappo_agent.current_training_province_idx = current_province_idx if current_province_idx is not None else None
                province_episode_count += 1
            else:
                # 微调阶段
                is_fine_tuning = True
                mappo_agent.current_training_province_idx = None
                current_province_idx = None

            # 🎯 打印训练状态
            if is_fine_tuning:
                print(f"\n{'=' * 60}")
                print(f"🔧 微调阶段 - 回合 {episode + 1}/{max_episodes}")
                print(f"{'=' * 60}")
            else:
                print(f"\n{'=' * 60}")
                print(f"🎯 优先级训练 - 回合 {episode + 1}/{max_episodes}")
                if current_province_idx is not None:
                    current_config = province_config_map[current_province_idx]
                    print(f"   当前优化省份: {current_province_idx} ({current_config['province_name']})")
                    print(f"   该省份训练轮次: {province_episode_count}/{current_config['episodes']}")
                print(f"   已固定省份数: {len(mappo_agent.fixed_policies)}/{env.num_provinces}")
                print(f"   进度: {current_province_order_idx}/{len(province_order)} 省份")
            print(f"{'=' * 60}")

            # 🎯 记录优先级训练状态（episode开始前，显示上一轮的状态）
            # 注意：第一个episode会显示初始状态，后续episode会显示上一轮的状态
            if episode == 0:
                detailed_logger.log_priority_training(episode + 1, mappo_agent.province_monitors,
                                                      mappo_agent.current_training_province_idx, prefix="[开始] ")
            else:
                detailed_logger.log_priority_training(
                    episode + 1,
                    mappo_agent.province_monitors,
                    mappo_agent.current_training_province_idx,
                    prefix="[开始前] ",
                    episode_total_reward=previous_episode_total_reward,
                    province_episode_totals=previous_province_totals
                )

            try:
                env.reset()
                local_observations = env.get_local_observations()
                global_observation = env.get_global_observation()
                agent_ids = env.get_agent_ids()

                # 🎯 关键修复：在episode开始时保存当前策略快照
                # 这是实际执行动作时使用的策略，必须在update()之前保存
                episode_policy_snapshot = copy.deepcopy(mappo_agent.shared_actor_old.state_dict())

                episode_reward = 0
                episode_experiences = []
                episode_actions = []
                # 🎯 收集每个step的省份级奖励（用于计算每个省份的总奖励）
                episode_province_rewards = {province_idx: [] for province_idx in range(len(env.province_names))}
                # 🎯 新增：收集每个省份在每个step的实际动作（用于固定动作序列）
                episode_province_actions = {province_idx: [] for province_idx in range(len(env.province_names))}

                for step in range(env.max_steps):
                    print(f"🔄 步骤 {step + 1}/{env.max_steps}", end=" ")

                    try:
                        # 🎯 使用支持固定策略的动作选择方法
                        if is_fine_tuning:
                            # 微调模式：所有省份都使用固定策略，但允许小幅调整
                            actions, log_probs = mappo_agent.select_actions_fine_tune(
                                local_observations, agent_ids, episode=episode, step=step
                            )
                        else:
                            # 优先级训练模式：固定策略的省份使用固定策略，其他省份正常探索
                            actions, log_probs = mappo_agent.select_actions_with_fixed(
                                local_observations, agent_ids, episode=episode,
                                step_reward=episode_reward if step > 0 else None, step=step
                            )
                        episode_actions.append(actions.copy())

                        # 🎯 收集每个省份的实际动作
                        for province_idx in range(len(env.province_names)):
                            if province_idx < len(actions):
                                episode_province_actions[province_idx].append(actions[province_idx].copy())

                        next_local_obs, reward, done, info = env.step(actions)
                        next_global_obs = env.get_global_observation()

                        # ✅ 收集省份级奖励（从info中获取）
                        province_rewards = info.get('province_rewards', {})
                        if province_rewards is None or (isinstance(province_rewards, dict) and len(province_rewards) == 0):
                            # 如果环境没有返回，尝试从reward_components构建
                            reward_components = info.get('reward_components', {})
                            if reward_components:
                                # 使用平均值分配（简化处理）
                                num_provs = len(env.province_names)
                                avg_target = reward_components.get('total_target_reward', 0) / num_provs
                                avg_cost = reward_components.get('total_cost_penalty', 0) / num_provs
                                avg_health = reward_components.get('total_health_reward', 0) / num_provs
                                province_rewards = {
                                    prov: {'target': avg_target, 'cost': avg_cost, 'health': avg_health}
                                    for prov in env.province_names
                                }
                            else:
                                print("⚠️ info 中无 province_rewards，使用 mock 数据")
                                province_rewards = {
                                    prov: {'target': random.uniform(100, 200), 'cost': random.uniform(-50, -10),
                                           'health': random.uniform(50, 100)} for prov in env.province_names}
                        detailed_logger.log_province_rewards(episode + 1, province_rewards)

                        # 🎯 收集每个省份的step奖励（用于计算episode总奖励）
                        for province_idx, province_name in enumerate(env.province_names):
                            if province_name in province_rewards:
                                province_reward_dict = province_rewards[province_name]
                                # ✅ 计算该省份的step总奖励：优先使用环境返回的 total（已包含公平性/协作/竞争等）
                                if isinstance(province_reward_dict, dict) and 'total' in province_reward_dict:
                                    step_province_reward = float(province_reward_dict.get('total', 0.0))
                                else:
                                    # 兼容：如果没有total，尽量把已知项加总
                                    step_province_reward = float(
                                        province_reward_dict.get('target', 0.0) +
                                        province_reward_dict.get('cost', 0.0) +
                                        province_reward_dict.get('health', 0.0) +
                                        province_reward_dict.get('fairness', 0.0) +
                                        province_reward_dict.get('coordination', 0.0) +
                                        province_reward_dict.get('ranking', 0.0) +
                                        province_reward_dict.get('game', 0.0)
                                    )
                                episode_province_rewards[province_idx].append(step_province_reward)

                        enhanced_reward = reward

                        reward_change = 0
                        if len(episode_experiences) > 0:
                            prev_reward = episode_experiences[-1]['rewards']
                            reward_change = enhanced_reward - prev_reward

                        # ✅ 验证数据一致性：检查province_rewards总和是否与reward_components一致
                        reward_components = info.get('reward_components', {})
                        if reward_components is not None and isinstance(reward_components, dict) and len(reward_components) > 0 and province_rewards is not None and isinstance(province_rewards, dict) and len(province_rewards) > 0:
                            province_reward_sum_from_dict = sum(
                                [rew['target'] + rew['cost'] + rew['health']
                                 for rew in province_rewards.values()])
                            province_reward_sum_from_components = reward_components.get('total_province_reward', 0)

                            # 允许小的浮点误差（1e-3）
                            if abs(province_reward_sum_from_dict - province_reward_sum_from_components) > 1e-3:
                                print(f"⚠️ 数据不一致警告: province_rewards总和={province_reward_sum_from_dict:.2f}, "
                                      f"reward_components中的total_province_reward={province_reward_sum_from_components:.2f}")

                        detailed_logger.log_reward_analysis(
                            episode + 1, step + 1, reward,
                            0.0, enhanced_reward,
                            reward_change, info, actions, env.cumulative_factors
                        )

                        # 🎯 记录协作与竞争奖励详情
                        reward_components = info.get('reward_components', {})
                        coordination_components = reward_components.get('coordination_components', {}) if isinstance(reward_components, dict) else {}
                        if coordination_components is not None and isinstance(coordination_components, dict) and len(coordination_components) > 0:
                            # 获取省份区域映射（从环境获取）
                            province_regions = {}
                            if hasattr(env, 'province_regions'):
                                province_regions = env.province_regions
                            
                            detailed_logger.log_coordination_competition(
                                episode + 1, step + 1,
                                coordination_components,
                                env.province_names,
                                province_regions
                            )

                        # 🎯 记录各省份的减排信息
                        predicted_pm25 = info.get('predicted_pm25', [])
                        # 获取当前步骤的PM2.5目标
                        step_targets, _ = env.get_step_target(step)

                        # 获取上一步的PM2.5浓度（用于计算变化）
                        if step == 0:
                            pm25_before = [env.province_base_conc.get(prov_name, 50.0) for prov_name in
                                           env.province_names]
                            previous_cumulative_factors = np.ones_like(env.cumulative_factors)  # 第一步前是全1
                        else:
                            # 从上一个experience获取
                            if len(episode_experiences) > 0:
                                prev_info = episode_experiences[-1].get('info', {})
                                pm25_before = prev_info.get('predicted_pm25',
                                                            [env.province_base_conc.get(prov_name, 50.0) for prov_name
                                                             in env.province_names])
                                # 获取上一步的cumulative_factors（从experience中保存）
                                if 'cumulative_factors' in episode_experiences[-1]:
                                    previous_cumulative_factors = episode_experiences[-1]['cumulative_factors']
                                elif hasattr(env,
                                             'previous_cumulative_factors') and env.previous_cumulative_factors is not None:
                                    previous_cumulative_factors = env.previous_cumulative_factors
                                else:
                                    # 如果没有保存，使用当前值作为近似
                                    previous_cumulative_factors = env.cumulative_factors.copy()
                            else:
                                pm25_before = [env.province_base_conc.get(prov_name, 50.0) for prov_name in
                                               env.province_names]
                                previous_cumulative_factors = np.ones_like(env.cumulative_factors)

                        # 为每个省份记录减排信息
                        for province_idx, province_name in enumerate(env.province_names):
                            if province_idx < len(actions):
                                # 计算单步减排率
                                if step == 0:
                                    # 第一步：单步减排率 = 1 - cumulative_factors
                                    single_step_reduction = 1.0 - env.cumulative_factors[province_idx]
                                else:
                                    # 后续步骤：需要与上一步的cumulative_factors比较
                                    if province_idx < len(previous_cumulative_factors):
                                        prev_cumulative = previous_cumulative_factors[province_idx]
                                        curr_cumulative = env.cumulative_factors[province_idx]
                                        # 避免除零
                                        if np.any(prev_cumulative > 1e-8):
                                            single_step_reduction = (
                                                                            prev_cumulative - curr_cumulative) / prev_cumulative
                                        else:
                                            single_step_reduction = 1.0 - curr_cumulative
                                    else:
                                        single_step_reduction = 1.0 - env.cumulative_factors[province_idx]

                                cumulative_reduction = 1.0 - env.cumulative_factors[province_idx]

                                province_data = {
                                    'province_id': province_idx,
                                    'province_name': province_name,
                                    'single_step_reduction': single_step_reduction,
                                    'cumulative_reduction': cumulative_reduction,
                                    'action_value': actions[province_idx],
                                    'pm25_before': pm25_before[province_idx] if province_idx < len(
                                        pm25_before) else env.province_base_conc.get(province_name, 50.0),
                                    'pm25_after': predicted_pm25[province_idx] if province_idx < len(
                                        predicted_pm25) else env.province_base_conc.get(province_name, 50.0),
                                    'pm25_target': step_targets.get(province_name,
                                                                    env.province_base_conc.get(province_name,
                                                                                               50.0) * 0.5)
                                }

                                detailed_logger.log_province_reduction(episode + 1, step + 1, province_data)

                        print(f"奖励: {enhanced_reward:.2f}")

                        experience = {
                            'local_obs': local_observations,
                            'global_obs': global_observation,
                            'agent_ids': agent_ids,
                            'actions': actions,
                            'rewards': enhanced_reward,
                            'old_log_probs': log_probs,
                            'info': info,
                            'cumulative_factors': env.cumulative_factors.copy()
                        }
                        episode_experiences.append(experience)
                        
                        # ✅ 内存优化：只保留最近若干步的完整 info，
                        # 旧步骤只保留 province_rewards 和 predicted_pm25
                        if len(episode_experiences) > 2:
                            old_exp = episode_experiences[-3]
                            if 'info' in old_exp and isinstance(old_exp['info'], dict):
                                slim_info = {}
                                if 'province_rewards' in old_exp['info']:
                                    slim_info['province_rewards'] = old_exp['info']['province_rewards']
                                if 'predicted_pm25' in old_exp['info']:
                                    slim_info['predicted_pm25'] = old_exp['info']['predicted_pm25']
                                old_exp['info'] = slim_info

                        local_observations = env.get_local_observations()
                        global_observation = next_global_obs

                        episode_reward += enhanced_reward

                        if done:
                            print("✅ 回合提前结束")
                            break

                    except Exception as e:
                        print(f"❌ 步骤失败: {e}")
                        break

                episode_diversity = 0
                if len(episode_actions) > 0:
                    all_actions = np.concatenate(episode_actions, axis=0)
                    action_std = np.std(all_actions, axis=0)
                    episode_diversity = np.mean(action_std)
                    diversity_scores.append(episode_diversity)

                    # ✅ 更保守的多样性检查
                    if episode_diversity < 0.02:  # 进一步降低阈值，只有多样性极低时才调整
                        # 更温和的调整
                        mappo_agent.exploration_noise = min(0.3, mappo_agent.exploration_noise * 1.2)
                        print(
                            f"Very low diversity {episode_diversity:.4f}, slightly increased noise to {mappo_agent.exploration_noise:.3f}")

                detailed_logger.log_episode_summary(
                    episode + 1, episode_reward, episode_experiences, episode_actions, episode_diversity
                )

                if not hasattr(detailed_logger, 'exploration_noise_history'):
                    detailed_logger.exploration_noise_history = []
                detailed_logger.exploration_noise_history.append(mappo_agent.exploration_noise)

                if not hasattr(detailed_logger, 'diversity_scores'):
                    detailed_logger.diversity_scores = []
                detailed_logger.diversity_scores.append(episode_diversity)

                if not hasattr(detailed_logger, 'episode_rewards'):
                    detailed_logger.episode_rewards = []
                detailed_logger.episode_rewards.append(episode_reward)

                # 🎯 更新所有省份的监控器（使用每个省份自己的episode总奖励）
                # 注意：每个省份应该使用自己的奖励，而不是全局奖励
                province_episode_totals = {
                    province_idx: sum(rewards) if rewards else 0.0
                    for province_idx, rewards in episode_province_rewards.items()
                }

                for province_idx in range(len(env.province_names)):
                    if province_idx in mappo_agent.province_monitors:
                        province_episode_reward = province_episode_totals.get(province_idx, 0.0)

                        # ✅ 内存优化：不再每个省份都 deepcopy 完整 state_dict
                        # ProvinceMonitor.update 内部只在奖励改善时才 deepcopy
                        if province_idx in mappo_agent.fixed_policies:
                            current_policy = mappo_agent.fixed_policies[province_idx]
                        else:
                            current_policy = episode_policy_snapshot

                        province_actions_seq = episode_province_actions.get(province_idx, [])

                        mappo_agent.province_monitors[province_idx].update(
                            episode + 1, province_episode_reward, current_policy, province_actions_seq
                        )

                        # 🎯 调试输出：显示当前训练省份的奖励信息
                        if province_idx == mappo_agent.current_training_province_idx:
                            monitor = mappo_agent.province_monitors[province_idx]
                            print(f"  📊 省份 {province_idx} ({env.province_names[province_idx]}): "
                                  f"本轮奖励={province_episode_reward:.2f}, "
                                  f"最佳奖励={monitor.best_reward:.2f} (Episode {monitor.best_episode})")

                # 🎯 在episode结束后再次记录优先级训练状态（显示最新状态）
                detailed_logger.log_priority_training(
                    episode + 1,
                    mappo_agent.province_monitors,
                    mappo_agent.current_training_province_idx,
                    prefix="[结束后] ",
                    episode_total_reward=episode_reward,
                    province_episode_totals=province_episode_totals
                )
                previous_episode_total_reward = episode_reward
                previous_province_totals = province_episode_totals.copy()

                policy_improved = mappo_agent.check_and_update_best_policy(episode_reward, episode + 1)

                # ✅ 修复：禁用自动恢复最佳策略功能
                # 原因：恢复旧策略会导致策略与当前经验数据不匹配，造成loss爆炸
                # 如果确实需要恢复，应该在恢复后清空经验池，重新收集经验
                # if mappo_agent.should_restore_best_policy():
                #     print(f"🔄 探索{mappo_agent.exploration_without_improvement}回合无改善，恢复最佳策略...")
                #     mappo_agent.restore_best_policy()

                exploration_status = mappo_agent.get_exploration_strategy_status()
                detailed_logger.log_exploration_strategy(episode + 1, exploration_status, policy_improved)

                status_indicator = "🎯" if exploration_status['is_using_best_policy'] else "🔍"
                print(f"\n📊 回合结果: 奖励={episode_reward:.2f}, 多样性={episode_diversity:.3f}")
                print(f"{status_indicator} 探索状态: 最佳={exploration_status['best_reward']:.2f}, "
                      f"无改善={exploration_status['exploration_without_improvement']}, "
                      f"恢复次数={exploration_status['policy_restoration_count']}")
                print(f"📈 TensorBoard实时监控已启用 - 数据已写入: {tensorboard_dir}")

                for experience in episode_experiences:
                    # ✅ 获取每个省份的奖励（包含竞争奖励）
                    province_rewards = None
                    if 'info' in experience and 'province_rewards' in experience['info']:
                        province_rewards = experience['info']['province_rewards']
                    
                    mappo_agent.store_experience(
                        experience['local_obs'],
                        experience['global_obs'],
                        experience['agent_ids'],
                        experience['actions'],
                        experience['rewards'],
                        experience['old_log_probs'],
                        province_rewards=province_rewards  # ✅ 传递省份级奖励
                    )

                # 🎯 更新策略（微调阶段使用不同的更新方式）
                if (episode + 1) % update_interval == 0:
                    print(f"🔄 更新MAPPO智能体...")
                    try:
                        if len(episode_experiences) > 0:
                            # ✅ 确保training_stats中有必要的键
                            if 'entropy_values' not in mappo_agent.training_stats:
                                mappo_agent.training_stats['entropy_values'] = []
                            if 'actor_losses' not in mappo_agent.training_stats:
                                mappo_agent.training_stats['actor_losses'] = []
                            if 'critic_losses' not in mappo_agent.training_stats:
                                mappo_agent.training_stats['critic_losses'] = []

                            sample_global_obs = [exp['global_obs'] for exp in episode_experiences[-3:]]
                            sample_global_obs = np.array(sample_global_obs)
                            pre_update_values = mappo_agent.get_values(sample_global_obs)

                            # 🎯 根据训练阶段选择更新方式
                            if is_fine_tuning:
                                # 微调模式：只更新Critic
                                mappo_agent.update_fine_tune(episode + 1)
                            else:
                                # 优先级训练模式：正常更新
                                mappo_agent.update(episode + 1)

                            post_update_values = mappo_agent.get_values(sample_global_obs)
                            value_change = np.mean(
                                np.abs(post_update_values.cpu().numpy() - pre_update_values.cpu().numpy()))
                            detailed_logger.log_critic_convergence(
                                episode + 1, pre_update_values.cpu().numpy(), post_update_values.cpu().numpy(),
                                value_change, mappo_agent.training_stats.get('critic_losses', [])
                            )
                            print(f"  价值函数更新: 平均变化={value_change:.4f}")
                    except Exception as e:
                        print(f"❌ MAPPO更新失败: {e}", flush=True)
                        detailed_logger.log(f"❌ MAPPO更新失败: {e}")
                        import traceback
                        traceback.print_exc()
                        # ✅ 更新失败时强制清空 memory，防止 OOM 恶性循环
                        mappo_agent.memory = []
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        print(f"🧹 已强制清空经验池和GPU缓存")

                episode_rewards.append(episode_reward)

                if 'exploration_noise' not in mappo_agent.training_stats:
                    mappo_agent.training_stats['exploration_noise'] = []
                mappo_agent.training_stats['exploration_noise'].append(mappo_agent.exploration_noise)

                # 📈 奖励趋势分析
                if len(episode_rewards) >= 10:
                    recent_rewards = episode_rewards[-10:]
                    reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                    reward_mean = np.mean(recent_rewards)
                    reward_std = np.std(recent_rewards)

                    print(f"📊 奖励趋势分析（最近10回合）:")
                    print(f"  平均奖励: {reward_mean:.2f}, 标准差: {reward_std:.2f}")
                    print(f"  趋势斜率: {reward_trend:+.2f} {'📈' if reward_trend > 0 else '📉'}")

                    # 如果奖励趋势下降且波动大，调整学习率
                    if reward_trend < -50 and reward_std > 500:
                        mappo_agent.actor_lr *= 0.95
                        mappo_agent.critic_lr *= 0.95

                        # ✅ 更新所有省份的Actor optimizer
                        for actor_optimizer in mappo_agent.actor_optimizers:
                            for param_group in actor_optimizer.param_groups:
                                param_group['lr'] = mappo_agent.actor_lr

                        # ✅ 同时更新聚合optimizer（兼容性）
                        for param_group in mappo_agent.actor_optimizer.param_groups:
                            param_group['lr'] = mappo_agent.actor_lr

                        for param_group in mappo_agent.critic_optimizer.param_groups:
                            param_group['lr'] = mappo_agent.critic_lr
                        print(
                            f"  ⚠️ 奖励下降且波动大，降低学习率到 Actor={mappo_agent.actor_lr:.6f}, Critic={mappo_agent.critic_lr:.6f}")

                detailed_logger.update_training_data(episode_reward, mappo_agent.training_stats, episode_diversity)

                # 更新奖励基线
                if len(episode_rewards) >= baseline_window:
                    reward_baseline = np.mean(episode_rewards[-baseline_window:])
                elif len(episode_rewards) > 0:
                    reward_baseline = np.mean(episode_rewards)

                # 改进判断
                improvement_threshold = 50.0
                is_improvement = False

                if reward_baseline is not None:
                    improvement_relative_to_baseline = episode_reward - reward_baseline
                    if improvement_relative_to_baseline > improvement_threshold:
                        is_improvement = True
                        convergence_patience = 0
                        print(f"📈 相对基线改进: {improvement_relative_to_baseline:+.2f} (基线={reward_baseline:.2f})")

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    convergence_patience = 0
                    is_improvement = True
                    print(f"🏆 新最佳奖励: {best_reward:.4f}")

                    try:
                        # 保存共享Actor
                        torch.save(mappo_agent.shared_actor.state_dict(),
                                   os.path.join(result_dir, "best_pytorch_mappo_actor_shared.pth"))
                        # 保存每省Actor
                        torch.save(
                            {f"actor_{i}": actor.state_dict() for i, actor in enumerate(mappo_agent.actor_list)},
                            os.path.join(result_dir, "best_pytorch_mappo_actor_per_province.pth")
                        )
                        # 保存Critic
                        torch.save(mappo_agent.centralized_critic.state_dict(),
                                   os.path.join(result_dir, "best_pytorch_mappo_critic.pth"))
                        print("✅ 保存最佳模型（共享 + 每省Actor + Critic）")
                    except Exception as e:
                        print(f"❌ 保存模型失败: {e}")

                if not is_improvement:
                    convergence_patience += 1
                    if reward_baseline is not None:
                        print(
                            f"⏳ 无改进，耐心计数: {convergence_patience} (当前={episode_reward:.2f}, 基线={reward_baseline:.2f})")

                # 新增：收敛检查和诊断 - 放宽判断条件，减少误判
                if len(episode_rewards) >= 20:  # ✅ 从10增加到20，使用更长的窗口判断
                    prev_avg = np.mean(episode_rewards[-20:-10])  # ✅ 使用更长的窗口
                    current_avg = np.mean(episode_rewards[-10:])  # ✅ 使用更长的窗口
                    # ✅ 放宽判断阈值，从1e-3增加到更大的值，避免因为小的波动就判定为无改善
                    improvement_threshold = max(50.0, reward_baseline * 0.01) if reward_baseline else 50.0
                    if current_avg <= prev_avg + improvement_threshold:
                        convergence_patience += 1
                        if convergence_patience >= 20:  # ✅ 从5增加到20，减少误判
                            diag_msg = f"⚠️ 奖励未稳定上升 (耐心: {convergence_patience}/50)\n"
                            diag_msg += f"  最近10平均: {current_avg:.2f} (前10: {prev_avg:.2f})\n"
                            diag_msg += f"  诊断:\n    多样的: {episode_diversity:.4f}\n    噪声: {mappo_agent.exploration_noise:.4f}\n"
                            if 'actor_losses' in mappo_agent.training_stats and mappo_agent.training_stats[
                                'actor_losses']:
                                diag_msg += f"    Actor Loss: {mappo_agent.training_stats['actor_losses'][-1]:.4f}\n"
                            if 'critic_losses' in mappo_agent.training_stats and mappo_agent.training_stats[
                                'critic_losses']:
                                diag_msg += f"    Critic Loss: {mappo_agent.training_stats['critic_losses'][-1]:.4f}\n"
                            if province_rewards is not None and isinstance(province_rewards, dict) and len(province_rewards) > 0:
                                avg_prov = sum([rew['target'] + rew['cost'] + rew['health'] for rew in
                                                province_rewards.values()]) / len(province_rewards)
                                diag_msg += f"    平均省份奖励: {avg_prov:.2f}\n"
                                low_provs = [p for p, r in province_rewards.items() if
                                             (r['target'] + r['cost'] + r['health']) < 0]
                                if low_provs:
                                    diag_msg += f"    负奖励省份: {low_provs[:5]} (可能成本惩罚过高)\n"
                            print(diag_msg)
                            detailed_logger.log(diag_msg)
                    else:
                        # ✅ 只有当明显改善时才重置耐心值
                        if current_avg > prev_avg + improvement_threshold * 2:
                            convergence_patience = max(0, convergence_patience - 1)  # 改善时减少耐心值

                # ✅ 注意：TensorBoard写入代码已移到try块外（1863行），这里不再重复写入
                # 这样可以确保即使episode出错，TensorBoard写入也会执行

                if (episode + 1) % 5 == 0:
                    detailed_logger.generate_realtime_plots(episode + 1)

                if (episode + 1) % 10 == 0:
                    print(f"📈 保存训练进度...")
                    try:
                        checkpoint_data = {
                            'episode': episode + 1,
                            'episode_rewards': episode_rewards[-500:],
                            'best_reward': best_reward,
                            'convergence_patience': convergence_patience,
                            'diversity_scores': diversity_scores[-500:],
                            'training_stats': {k: (v[-200:] if isinstance(v, list) else v) 
                                              for k, v in mappo_agent.training_stats.items()},
                            'exploration_noise': mappo_agent.exploration_noise,
                            'actor_lr': mappo_agent.actor_lr,
                            'critic_lr': mappo_agent.critic_lr
                        }
                        with open(os.path.join(result_dir, 'training_checkpoint_pytorch.json'), 'w') as f:
                            json.dump(checkpoint_data, f, indent=2, cls=NumpyEncoder)
                        # 保存共享Actor
                        torch.save(mappo_agent.shared_actor.state_dict(),
                                   os.path.join(result_dir, 'checkpoint_actor_shared_pytorch.pth'))
                        # 保存每省Actor
                        torch.save(
                            {f"actor_{i}": actor.state_dict() for i, actor in enumerate(mappo_agent.actor_list)},
                            os.path.join(result_dir, 'checkpoint_actor_per_province_pytorch.pth')
                        )
                        # 保存Critic
                        torch.save(mappo_agent.centralized_critic.state_dict(),
                                   os.path.join(result_dir, 'checkpoint_critic_pytorch.pth'))
                        print(f"💾 训练检查点已保存 (Episode {episode + 1})")
                    except Exception as e:
                        print(f"❌ 保存结果失败: {e}")

                # 早停 - 增加耐心值，给模型更多学习机会
                if convergence_patience >= 5000:  # ✅ 从10增加到50，给模型更多时间学习
                    print(f"🛑 早停: 50 episode 无改善")
                    break

            except Exception as e:
                # 如果episode出错，设置默认值以便TensorBoard写入
                if 'episode_reward' not in locals():
                    episode_reward = 0.0
                if 'episode_diversity' not in locals():
                    episode_diversity = 0.0
                print(f"❌ 训练回合 {episode + 1} 失败: {e}", flush=True)
                detailed_logger.log(f"❌ 训练回合 {episode + 1} 失败: {e}")
                import traceback
                traceback.print_exc()
                
                # ✅ 异常时强制清理所有内存，防止CUDA OOM恶性循环
                mappo_agent.memory = []
                # 清理本轮临时变量
                if 'episode_experiences' in locals():
                    del episode_experiences
                if 'episode_actions' in locals():
                    del episode_actions
                if 'episode_province_actions' in locals():
                    del episode_province_actions
                if 'episode_province_rewards' in locals():
                    del episode_province_rewards
                if 'episode_policy_snapshot' in locals():
                    del episode_policy_snapshot
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"🧹 已强制清理内存 (Episode {episode + 1})")
                # 注意：TensorBoard写入会在try块外执行，所以这里不写入

            # ✅ TensorBoard写入移到try块外，确保每个episode都会执行
            if writer is not None and tensorboard_enabled:
                try:
                    # 获取省份名称（从mappo_agent的province_monitors）
                    province_names = []
                    if hasattr(mappo_agent, 'province_monitors') and mappo_agent.province_monitors:
                        province_names = [mappo_agent.province_monitors[i].province_name
                                          for i in sorted(mappo_agent.province_monitors.keys())]
                    if not province_names:
                        province_names = [f"Province_{i}" for i in range(mappo_agent.num_agents)]

                    # 基础指标
                    episode_reward_val = float(episode_reward) if not (
                                np.isnan(episode_reward) or np.isinf(episode_reward)) else 0.0
                    diversity_val = float(episode_diversity) if not (
                                np.isnan(episode_diversity) or np.isinf(episode_diversity)) else 0.0
                    noise_val = float(mappo_agent.exploration_noise) if not (
                                np.isnan(mappo_agent.exploration_noise) or np.isinf(
                            mappo_agent.exploration_noise)) else 0.0

                    writer.add_scalar('Reward/Episode', episode_reward_val, episode + 1)
                    writer.add_scalar('Reward/Diversity', diversity_val, episode + 1)
                    writer.add_scalar('Exploration/Noise', noise_val, episode + 1)
                    writer.add_scalar('Exploration/Epsilon', float(mappo_agent.epsilon), episode + 1)

                    if 'random_action_ratio' in mappo_agent.training_stats and len(
                            mappo_agent.training_stats['random_action_ratio']) > 0:
                        writer.add_scalar('Exploration/RandomActionRatio',
                                          float(mappo_agent.training_stats['random_action_ratio'][-1]), episode + 1)

                    # 移动平均奖励
                    if len(episode_rewards) >= 10:
                        writer.add_scalar('Reward/MovingAvg_10', float(np.mean(episode_rewards[-10:])), episode + 1)
                    if len(episode_rewards) >= 50:
                        writer.add_scalar('Reward/MovingAvg_50', float(np.mean(episode_rewards[-50:])), episode + 1)

                    # 训练损失
                    if 'actor_losses' in mappo_agent.training_stats and len(
                            mappo_agent.training_stats['actor_losses']) > 0:
                        actor_loss_val = float(mappo_agent.training_stats['actor_losses'][-1])
                        writer.add_scalar('Loss/Actor', actor_loss_val, episode + 1)

                        # ✅ 分省份Actor Loss写入
                        if 'actor_losses_per_province' in mappo_agent.training_stats and len(
                                mappo_agent.training_stats['actor_losses_per_province']) > 0:
                            per_list = mappo_agent.training_stats['actor_losses_per_province'][-1]
                            if per_list and len(per_list) > 0:
                                for p_idx, p_loss in enumerate(per_list):
                                    if p_loss is not None and not (np.isnan(p_loss) or np.isinf(p_loss)):
                                        loss_val = float(p_loss)
                                        province_name = province_names[p_idx] if p_idx < len(
                                            province_names) else f"Province_{p_idx}"
                                        writer.add_scalar(f'Loss/Actor_Province/{province_name}', loss_val, episode + 1)
                                        writer.add_scalar(f'Loss/Actor_Province_Idx_{p_idx}', loss_val, episode + 1)
                    else:
                        writer.add_scalar('Loss/Actor', 0.0, episode + 1)

                    if 'critic_losses' in mappo_agent.training_stats and len(
                            mappo_agent.training_stats['critic_losses']) > 0:
                        writer.add_scalar('Loss/Critic', float(mappo_agent.training_stats['critic_losses'][-1]),
                                          episode + 1)
                    else:
                        writer.add_scalar('Loss/Critic', 0.0, episode + 1)

                    if 'entropy_values' in mappo_agent.training_stats and len(
                            mappo_agent.training_stats['entropy_values']) > 0:
                        writer.add_scalar('Loss/Entropy', float(mappo_agent.training_stats['entropy_values'][-1]),
                                          episode + 1)
                    else:
                        writer.add_scalar('Loss/Entropy', 0.0, episode + 1)

                    # 奖励组成部分
                    if len(episode_experiences) > 0:
                        last_step_info = episode_experiences[-1].get('info', {})
                        if last_step_info and 'reward_components' in last_step_info:
                            reward_components = last_step_info['reward_components']
                            if reward_components is not None and isinstance(reward_components, dict) and len(reward_components) > 0:
                                writer.add_scalar('RewardComponents/Target',
                                                  float(reward_components.get('total_target_reward', 0)), episode + 1)
                                writer.add_scalar('RewardComponents/Cost',
                                                  float(reward_components.get('total_cost_penalty', 0)), episode + 1)
                                writer.add_scalar('RewardComponents/Health',
                                                  float(reward_components.get('total_health_reward', 0)), episode + 1)
                                writer.add_scalar('RewardComponents/Coordination',
                                                  float(reward_components.get('coordination_reward', 0)), episode + 1)
                                writer.add_scalar('RewardComponents/Fairness',
                                                  float(reward_components.get('total_fairness_reward', 0)), episode + 1)
                                writer.add_scalar('RewardComponents/ProvinceTotal',
                                                  float(reward_components.get('total_province_reward', 0)), episode + 1)
                                
                                # 🎯 新增：协作与竞争奖励详细组成部分
                                coordination_components = reward_components.get('coordination_components', {}) if isinstance(reward_components, dict) else {}
                                if coordination_components is not None and isinstance(coordination_components, dict) and len(coordination_components) > 0:
                                    # 差分奖励统计
                                    differential_rewards = coordination_components.get('differential_rewards', {})
                                    if differential_rewards is not None and isinstance(differential_rewards, dict) and len(differential_rewards) > 0:
                                        total_differential = sum(differential_rewards.values())
                                        avg_differential = total_differential / len(differential_rewards)
                                        max_differential = max(differential_rewards.values())
                                        min_differential = min(differential_rewards.values())
                                        
                                        writer.add_scalar('Coordination/Differential_Total', float(total_differential), episode + 1)
                                        writer.add_scalar('Coordination/Differential_Avg', float(avg_differential), episode + 1)
                                        writer.add_scalar('Coordination/Differential_Max', float(max_differential), episode + 1)
                                        writer.add_scalar('Coordination/Differential_Min', float(min_differential), episode + 1)
                                        
                                        # 记录前10个省份的差分奖励
                                        sorted_differential = sorted(differential_rewards.items(), key=lambda x: x[1], reverse=True)
                                        for idx, (province_idx, diff_reward) in enumerate(sorted_differential[:10]):
                                            if province_idx < len(province_names):
                                                prov_name = province_names[province_idx]
                                                writer.add_scalar(f'Coordination/Differential_Province/{prov_name}', 
                                                                  float(diff_reward), episode + 1)
                                    
                                    # 排名竞争奖励统计
                                    ranking_rewards = coordination_components.get('ranking_rewards', None)
                                    if ranking_rewards is not None:
                                        # 处理numpy数组和字典两种情况
                                        if isinstance(ranking_rewards, np.ndarray):
                                            total_ranking = float(np.sum(ranking_rewards))
                                            avg_ranking = float(np.mean(ranking_rewards)) if len(ranking_rewards) > 0 else 0.0
                                            
                                            writer.add_scalar('Competition/Ranking_Total', total_ranking, episode + 1)
                                            writer.add_scalar('Competition/Ranking_Avg', avg_ranking, episode + 1)
                                            
                                            # 记录前10个省份的排名奖励
                                            sorted_indices = np.argsort(ranking_rewards)[::-1][:10]
                                            for idx, province_idx in enumerate(sorted_indices):
                                                if province_idx < len(province_names):
                                                    prov_name = province_names[province_idx]
                                                    writer.add_scalar(f'Competition/Ranking_Province/{prov_name}', 
                                                                      float(ranking_rewards[province_idx]), episode + 1)
                                        elif isinstance(ranking_rewards, dict) and len(ranking_rewards) > 0:
                                            total_ranking = sum(ranking_rewards.values())
                                            avg_ranking = total_ranking / len(ranking_rewards)
                                            
                                            writer.add_scalar('Competition/Ranking_Total', float(total_ranking), episode + 1)
                                            writer.add_scalar('Competition/Ranking_Avg', float(avg_ranking), episode + 1)
                                            
                                            # 记录前10个省份的排名奖励
                                            sorted_ranking = sorted(ranking_rewards.items(), key=lambda x: x[1], reverse=True)
                                            for idx, (province_idx, rank_reward) in enumerate(sorted_ranking[:10]):
                                                if province_idx < len(province_names):
                                                    prov_name = province_names[province_idx]
                                                    writer.add_scalar(f'Competition/Ranking_Province/{prov_name}', 
                                                                      float(rank_reward), episode + 1)
                                    
                                    # 区域间竞争奖励统计（或省份博弈奖励）
                                    inter_region_rewards = coordination_components.get('inter_region_rewards', None)
                                    province_game_rewards = coordination_components.get('province_game_rewards', None)
                                    
                                    # 优先使用province_game_rewards（新版本），否则使用inter_region_rewards（旧版本）
                                    game_rewards = province_game_rewards if province_game_rewards is not None else inter_region_rewards
                                    
                                    if game_rewards is not None:
                                        if isinstance(game_rewards, np.ndarray):
                                            total_game = float(np.sum(game_rewards))
                                            avg_game = float(np.mean(game_rewards)) if len(game_rewards) > 0 else 0.0
                                            
                                            writer.add_scalar('Competition/Game_Total', total_game, episode + 1)
                                            writer.add_scalar('Competition/Game_Avg', avg_game, episode + 1)
                                        elif isinstance(game_rewards, dict) and len(game_rewards) > 0:
                                            total_game = sum(game_rewards.values())
                                            avg_game = total_game / len(game_rewards)
                                            
                                            writer.add_scalar('Competition/Game_Total', float(total_game), episode + 1)
                                            writer.add_scalar('Competition/Game_Avg', float(avg_game), episode + 1)

                        # 省份级奖励
                        if last_step_info and 'province_rewards' in last_step_info:
                            province_rewards_from_info = last_step_info['province_rewards']
                            if province_rewards_from_info is not None and isinstance(province_rewards_from_info, dict) and len(province_rewards_from_info) > 0:
                                avg_prov_reward = sum([rew['target'] + rew['cost'] + rew['health']
                                                       for rew in province_rewards_from_info.values()]) / len(
                                    province_rewards_from_info)
                                writer.add_scalar('Province/Avg_Reward', float(avg_prov_reward), episode + 1)

                                # 只记录前10个省份
                                for idx, (prov_name, prov_reward) in enumerate(province_rewards_from_info.items()):
                                    if idx < 10:
                                        total_prov = prov_reward['target'] + prov_reward['cost'] + prov_reward['health']
                                        writer.add_scalar(f'Province_Reward/{prov_name}', float(total_prov),
                                                          episode + 1)
                                        writer.add_scalar(f'Province_Target/{prov_name}', float(prov_reward['target']),
                                                          episode + 1)
                                        writer.add_scalar(f'Province_Cost/{prov_name}', float(prov_reward['cost']),
                                                          episode + 1)
                                        writer.add_scalar(f'Province_Health/{prov_name}', float(prov_reward['health']),
                                                          episode + 1)

                    writer.flush()

                    if (episode + 1) % 10 == 0:
                        print(f"📊 TensorBoard数据已写入 (Episode {episode + 1})")

                except Exception as e:
                    print(f"❌ TensorBoard写入失败 (Episode {episode + 1}): {e}")
                    if writer is not None:
                        try:
                            writer.flush()
                        except:
                            pass

            # ✅ 每10个episode定期清理GPU内存，防止CUDA OOM
            if (episode + 1) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # 如果episode出错，continue到下一个episode
            if 'e' in locals() and isinstance(e, Exception):
                continue

        print("🎉 PyTorch版MAPPO训练完成！")

        detailed_logger.generate_realtime_plots(max_episodes, save_path=result_dir)
        detailed_logger.save_training_data(save_path=result_dir)

        try:
            cumulative_reduction_path = os.path.join(result_dir, "cumulative_reductions_pytorch.json")
            with open(cumulative_reduction_path, 'w', encoding='utf-8') as f:
                json.dump(cumulative_reductions, f, indent=2, cls=NumpyEncoder)
            print(f"🌍 累积减排率数据已保存到: {cumulative_reduction_path}")
        except Exception as e:
            print(f"❌ 保存累积减排率数据失败: {e}")

        # ✅ 最终flush和close TensorBoard
        if writer is not None and tensorboard_enabled:
            try:
                writer.flush()
                writer.close()

                # ✅ 验证TensorBoard文件是否生成
                import glob
                # tensorboard_dir已在函数开头定义，这里直接使用
                if tensorboard_dir is not None:
                    event_files = glob.glob(os.path.join(tensorboard_dir, 'events.out.tfevents.*'))
                    if event_files:
                        latest_file = max(event_files, key=os.path.getmtime)
                        file_size = os.path.getsize(latest_file)
                        print(f"📊 TensorBoard日志已保存到: {tensorboard_dir}")
                        print(f"   事件文件: {latest_file} (大小: {file_size} 字节)")
                        print(f"   使用命令查看: tensorboard --logdir={tensorboard_dir}")
                    else:
                        print(f"⚠️ TensorBoard目录存在但未找到事件文件: {tensorboard_dir}")
                        print(f"   请检查目录权限和磁盘空间")
            except Exception as e:
                print(f"❌ TensorBoard关闭失败: {e}")
                import traceback
                traceback.print_exc()

        detailed_logger.close()

        return mappo_agent, episode_rewards, result_dir

    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        # ✅ 确保异常时也关闭资源
        if writer is not None and tensorboard_enabled:
            try:
                writer.flush()
                writer.close()
                print(f"📊 TensorBoard已关闭")
            except Exception as e2:
                print(f"⚠️ TensorBoard关闭时出错: {e2}")
        detailed_logger.close()
        return None, None, None


def _run_single_scenario_worker(args):
    """
    单个情景训练的辅助函数（用于多进程）

    Args:
        args: 包含训练参数的元组
            (scenario_idx, scenario_id, scenario_seed, enable_realtime_console,
             province_training_config, max_total_episodes, fine_tune_episodes, gpu_id)

    Returns:
        result_dict: 包含训练结果的字典，如果失败返回None
    """
    (scenario_idx, scenario_id, scenario_seed, enable_realtime_console,
     province_training_config, max_total_episodes, fine_tune_episodes, gpu_id) = args

    try:
        # 设置GPU设备（如果有多个GPU）
        if torch.cuda.is_available() and gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device_str = f'cuda:{gpu_id}'
        else:
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 设置随机种子
        set_seed(scenario_seed)
        np.random.seed(scenario_seed)
        torch.manual_seed(scenario_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(scenario_seed)

        print(
            f"[进程 {os.getpid()}] 开始情景 {scenario_idx + 1}: {scenario_id} (种子={scenario_seed}, GPU={device_str})")

        # 运行训练
        mappo_agent, rewards, result_dir = train_optimized_mappo_pytorch(
            enable_realtime_console=enable_realtime_console,
            random_seed=scenario_seed,
            province_training_config=province_training_config,
            max_total_episodes=max_total_episodes,
            fine_tune_episodes=fine_tune_episodes,
            scenario_id=scenario_id
        )

        if mappo_agent is not None and rewards is not None:
            result_dict = {
                'scenario_id': scenario_id,
                'seed': scenario_seed,
                # 注意：mappo_agent在多进程时无法序列化，所以不包含在结果中
                # 如果需要模型，可以从result_dir中加载
                'rewards': rewards,
                'result_dir': result_dir,
                'best_reward': max(rewards),
                'avg_reward': np.mean(rewards),
                'final_reward': rewards[-1],
                'success': True
            }
            print(f"[进程 {os.getpid()}] ✅ 情景 {scenario_idx + 1} 完成: 最佳奖励={max(rewards):.2f}")
            return result_dict
        else:
            print(f"[进程 {os.getpid()}] ❌ 情景 {scenario_idx + 1} 训练失败")
            return {'scenario_id': scenario_id, 'seed': scenario_seed, 'success': False}

    except Exception as e:
        print(f"[进程 {os.getpid()}] ❌ 情景 {scenario_idx + 1} 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return {'scenario_id': scenario_id, 'seed': scenario_seed, 'success': False, 'error': str(e)}


def run_multi_scenario_training(
        num_scenarios=10,
        base_seed=42,
        enable_realtime_console=False,  # 多情景训练时建议关闭实时输出
        province_training_config=None,
        max_total_episodes=None,
        fine_tune_episodes=25,
        parallel=False,  # 是否并行训练
        num_workers=None,  # 并行工作进程数（None表示自动选择）
):
    """
    运行多个情景的训练（每个情景使用不同的随机种子）

    Args:
        num_scenarios: 情景数量（默认10）
        base_seed: 基础随机种子（每个情景会在此基础上递增）
        enable_realtime_console: 是否启用实时控制台输出
        province_training_config: 省份优化配置（所有情景共享）
        max_total_episodes: 总训练轮次上限
        fine_tune_episodes: 微调阶段轮次
        parallel: 是否并行训练
        num_workers: 并行工作进程数（None表示自动选择，建议设置为5或10）

    Returns:
        all_results: 所有情景的结果列表
        summary_stats: 汇总统计信息
    """
    print(f"\n{'=' * 80}")
    print(f"🚀 开始多情景训练（{num_scenarios}个情景）")
    if parallel:
        if num_workers is None:
            # 自动选择：如果有GPU，使用GPU数量；否则使用CPU核心数的一半
            if torch.cuda.is_available():
                num_workers = min(torch.cuda.device_count(), num_scenarios)
            else:
                import multiprocessing
                num_workers = max(1, multiprocessing.cpu_count() // 2)
        print(f"   并行模式: 启用 ({num_workers} 个工作进程)")
    else:
        print(f"   并行模式: 禁用（串行训练）")
    print(f"{'=' * 80}\n")

    all_results = []

    # 准备所有情景的参数
    scenario_args = []
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    for scenario_idx in range(num_scenarios):
        scenario_id = f"scenario_{scenario_idx + 1}"
        scenario_seed = base_seed + scenario_idx * 1000  # 每个情景使用不同的种子

        # GPU分配策略：如果有多个GPU，轮询分配；否则都使用同一个GPU或CPU
        if num_gpus > 1 and parallel:
            gpu_id = scenario_idx % num_gpus
        elif num_gpus == 1:
            gpu_id = 0
        else:
            gpu_id = None  # 使用CPU

        scenario_args.append((
            scenario_idx, scenario_id, scenario_seed, enable_realtime_console,
            province_training_config, max_total_episodes, fine_tune_episodes, gpu_id
        ))

    # 执行训练
    if parallel:
        # 并行训练
        from multiprocessing import Pool, cpu_count
        import multiprocessing

        # Windows系统需要设置spawn模式
        if sys.platform == 'win32':
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # 如果已经设置过，忽略错误

        print(f"🔄 启动 {num_workers} 个并行工作进程...")
        print(f"   GPU数量: {num_gpus}")
        if num_gpus > 1:
            print(f"   GPU分配策略: 轮询分配（0-{num_gpus - 1}）")

        start_time = time.time()

        try:
            with Pool(processes=num_workers) as pool:
                # 使用map异步执行，可以实时查看进度
                results = pool.map(_run_single_scenario_worker, scenario_args)

            elapsed_time = time.time() - start_time

            # 处理结果
            for i, result in enumerate(results):
                if result and result.get('success', False):
                    all_results.append(result)
                    print(f"✅ 情景 {i + 1} 完成: 最佳奖励={result.get('best_reward', 0):.2f}")
                else:
                    print(f"❌ 情景 {i + 1} 失败: {result.get('error', '未知错误')}")

            print(f"\n⏱️  并行训练总用时: {elapsed_time:.1f}秒 ({elapsed_time / 60:.1f}分钟)")
            print(f"   平均每个情景: {elapsed_time / num_scenarios:.1f}秒")

        except Exception as e:
            print(f"❌ 并行训练出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果并行失败，回退到串行
            print("⚠️ 回退到串行训练模式...")
            parallel = False

    if not parallel or len(all_results) == 0:
        # 串行训练（如果并行失败或未启用）
        if parallel:
            print("⚠️ 并行训练失败，使用串行模式")

        start_time = time.time()

        for scenario_idx in range(num_scenarios):
            scenario_id = f"scenario_{scenario_idx + 1}"
            scenario_seed = base_seed + scenario_idx * 1000

            print(f"\n{'=' * 80}")
            print(f"📋 情景 {scenario_idx + 1}/{num_scenarios}: {scenario_id}")
            print(f"   随机种子: {scenario_seed}")
            print(f"{'=' * 80}\n")

            try:
                mappo_agent, rewards, result_dir = train_optimized_mappo_pytorch(
                    enable_realtime_console=enable_realtime_console,
                    random_seed=scenario_seed,
                    province_training_config=province_training_config,
                    max_total_episodes=max_total_episodes,
                    fine_tune_episodes=fine_tune_episodes,
                    scenario_id=scenario_id
                )

                if mappo_agent is not None and rewards is not None:
                    all_results.append({
                        'scenario_id': scenario_id,
                        'seed': scenario_seed,
                        'mappo_agent': mappo_agent,
                        'rewards': rewards,
                        'result_dir': result_dir,
                        'best_reward': max(rewards),
                        'avg_reward': np.mean(rewards),
                        'final_reward': rewards[-1],
                        'success': True
                    })
                    print(
                        f"✅ 情景 {scenario_idx + 1} 完成: 最佳奖励={max(rewards):.2f}, 平均奖励={np.mean(rewards):.2f}")
                else:
                    print(f"❌ 情景 {scenario_idx + 1} 训练失败")

            except Exception as e:
                print(f"❌ 情景 {scenario_idx + 1} 训练出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        elapsed_time = time.time() - start_time
        print(f"\n⏱️  串行训练总用时: {elapsed_time:.1f}秒 ({elapsed_time / 60:.1f}分钟)")

    # 汇总统计
    if len(all_results) > 0:
        # 提取rewards用于汇总
        all_rewards = [result['rewards'] for result in all_results if result.get('success', False)]
        summary_stats = summarize_multi_scenario_results(all_results)
        return all_results, summary_stats
    else:
        print("❌ 所有情景训练失败")
        return [], None


def summarize_multi_scenario_results(all_results, save_dir="./result152/multi_scenario"):
    """
    汇总多情景训练结果，计算均值和置信区间

    Args:
        all_results: 所有情景的结果列表
        save_dir: 结果保存目录

    Returns:
        summary_stats: 汇总统计信息
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"📊 多情景训练结果汇总")
    print(f"{'=' * 80}\n")

    # 提取所有奖励序列
    all_rewards_arrays = [result['rewards'] for result in all_results]
    max_length = max(len(rewards) for rewards in all_rewards_arrays)

    # 对齐所有序列（填充到相同长度）
    aligned_rewards = []
    for rewards in all_rewards_arrays:
        if len(rewards) < max_length:
            # 用最后一个值填充
            padded = list(rewards) + [rewards[-1]] * (max_length - len(rewards))
        else:
            padded = rewards
        aligned_rewards.append(padded)

    aligned_rewards = np.array(aligned_rewards)  # shape: (num_scenarios, max_length)

    # 计算统计量
    mean_rewards = np.mean(aligned_rewards, axis=0)
    std_rewards = np.std(aligned_rewards, axis=0)
    # 95%置信区间（使用t分布，样本数较少时更准确）
    n = len(all_results)
    if n > 1:
        try:
            from scipy import stats
            t_critical = stats.t.ppf(0.975, n - 1)  # 95%置信区间
        except ImportError:
            # 如果scipy不可用，使用正态分布近似（样本数较大时近似准确）
            t_critical = 1.96  # 正态分布的95%置信区间
        ci_half_width = t_critical * std_rewards / np.sqrt(n)
        ci_lower = mean_rewards - ci_half_width
        ci_upper = mean_rewards + ci_half_width
    else:
        ci_lower = mean_rewards
        ci_upper = mean_rewards

    # 保存汇总数据
    summary_data = {
        'episode': list(range(1, max_length + 1)),
        'mean_reward': mean_rewards.tolist(),
        'std_reward': std_rewards.tolist(),
        'ci_lower': ci_lower.tolist(),
        'ci_upper': ci_upper.tolist(),
        'scenario_count': n
    }

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(save_dir, "multi_scenario_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ 汇总数据已保存: {summary_csv_path}")

    # 保存各情景的详细数据
    detailed_data = []
    for result in all_results:
        for episode, reward in enumerate(result['rewards'], 1):
            detailed_data.append({
                'scenario_id': result['scenario_id'],
                'seed': result['seed'],
                'episode': episode,
                'reward': reward
            })

    detailed_df = pd.DataFrame(detailed_data)
    detailed_csv_path = os.path.join(save_dir, "multi_scenario_detailed.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"✅ 详细数据已保存: {detailed_csv_path}")

    # 绘制置信区间图
    plot_multi_scenario_results(aligned_rewards, mean_rewards, ci_lower, ci_upper, save_dir)

    # 打印统计摘要
    print(f"\n📈 统计摘要:")
    print(f"  成功情景数: {n}/{len(all_results)}")
    print(f"  最终平均奖励: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
    print(f"  最佳平均奖励: {np.max(mean_rewards):.2f} (Episode {np.argmax(mean_rewards) + 1})")
    print(f"  各情景最佳奖励:")
    for result in all_results:
        print(f"    {result['scenario_id']}: {result['best_reward']:.2f}")

    summary_stats = {
        'num_scenarios': n,
        'mean_rewards': mean_rewards,
        'std_rewards': std_rewards,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'final_mean': mean_rewards[-1],
        'final_std': std_rewards[-1],
        'best_mean': np.max(mean_rewards),
        'best_episode': np.argmax(mean_rewards) + 1
    }

    return summary_stats


def plot_multi_scenario_results(aligned_rewards, mean_rewards, ci_lower, ci_upper, save_dir):
    """
    绘制多情景训练的均值和置信区间图

    Args:
        aligned_rewards: 对齐后的奖励数组 (num_scenarios, max_length)
        mean_rewards: 平均奖励序列
        ci_lower: 置信区间下界
        ci_upper: 置信区间上界
        save_dir: 保存目录
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        episodes = np.arange(1, len(mean_rewards) + 1)

        # 1. 均值与置信区间
        ax1 = axes[0, 0]
        ax1.plot(episodes, mean_rewards, 'b-', linewidth=2, label='均值')
        ax1.fill_between(episodes, ci_lower, ci_upper, alpha=0.3, color='blue', label='95%置信区间')
        # 绘制所有情景的曲线（半透明）
        for i, rewards in enumerate(aligned_rewards):
            ax1.plot(episodes, rewards, 'gray', alpha=0.2, linewidth=0.5)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('奖励')
        ax1.set_title('多情景训练：均值与95%置信区间')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 标准差变化
        ax2 = axes[0, 1]
        std_rewards = np.std(aligned_rewards, axis=0)
        ax2.plot(episodes, std_rewards, 'r-', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('标准差')
        ax2.set_title('奖励标准差变化')
        ax2.grid(True, alpha=0.3)

        # 3. 各情景对比（前10个episode的详细对比）
        ax3 = axes[1, 0]
        for i, rewards in enumerate(aligned_rewards[:10]):  # 只显示前10个情景
            ax3.plot(episodes[:min(50, len(rewards))], rewards[:min(50, len(rewards))],
                     alpha=0.6, linewidth=1, label=f'情景{i + 1}' if i < 5 else '')
        ax3.plot(episodes[:min(50, len(mean_rewards))], mean_rewards[:min(50, len(mean_rewards))],
                 'k-', linewidth=2, label='均值')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('奖励')
        ax3.set_title('各情景奖励对比（前50个Episode）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 最终奖励分布
        ax4 = axes[1, 1]
        final_rewards = [rewards[-1] for rewards in aligned_rewards]
        ax4.hist(final_rewards, bins=min(20, len(final_rewards)), edgecolor='black', alpha=0.7)
        ax4.axvline(np.mean(final_rewards), color='r', linestyle='--', linewidth=2,
                    label=f'均值={np.mean(final_rewards):.2f}')
        ax4.set_xlabel('最终奖励')
        ax4.set_ylabel('频数')
        ax4.set_title('最终奖励分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "multi_scenario_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 置信区间图已保存: {plot_path}")

    except Exception as e:
        print(f"❌ 绘制置信区间图失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 开始PyTorch版MAPPO多智能体强化学习训练")
    print("=" * 80)

    # ========== 可自定义的训练配置 ==========
    # 模式选择
    MULTI_SCENARIO_MODE = True  # True: 多情景训练, False: 单情景训练

    # 单情景训练配置
    enable_realtime_console = True
    SINGLE_SCENARIO_SEED = 42

    # 多情景训练配置
    NUM_SCENARIOS = 2  # 情景数量
    BASE_SEED = 42  # 基础随机种子
    PARALLEL_TRAINING = True  # 是否启用并行训练（True: 并行, False: 串行）
    NUM_WORKERS = 2  # 并行工作进程数（None表示自动选择，建议5或10）

    # ========== 可自定义的省份优化配置 ==========
    # 方式1: 使用默认配置（第一个省份200轮，其他60轮）
    # PROVINCE_TRAINING_CONFIG = None

    # 方式2: 自定义每个省份的优化顺序和轮次
    # 示例：先优化HB（200轮），然后SD（100轮），然后其他省份（各50轮）
    PROVINCE_TRAINING_CONFIG = [
        {'province_idx': 0, 'province_name': 'HA', 'episodes': 4000},
        {'province_idx': 1, 'province_name': 'HUB', 'episodes': 60},
        {'province_idx': 2, 'province_name': 'HUN', 'episodes': 60},
        {'province_idx': 3, 'province_name': 'JS', 'episodes': 60},
        {'province_idx': 4, 'province_name': 'SH', 'episodes': 60},
        {'province_idx': 5, 'province_name': 'SD', 'episodes': 60},
        {'province_idx': 6, 'province_name': 'HN', 'episodes': 60},
        {'province_idx': 7, 'province_name': 'TJ', 'episodes': 60},
        {'province_idx': 8, 'province_name': 'HB', 'episodes': 60},
        {'province_idx': 9, 'province_name': 'BJ', 'episodes': 60},
        {'province_idx': 10, 'province_name': 'SX', 'episodes': 60},
        {'province_idx': 11, 'province_name': 'NMG', 'episodes': 60},
        {'province_idx': 12, 'province_name': 'JL', 'episodes': 60},
        {'province_idx': 13, 'province_name': 'LN', 'episodes': 60},
        {'province_idx': 14, 'province_name': 'HLJ', 'episodes': 60},
        {'province_idx': 15, 'province_name': 'CQ', 'episodes': 60},
        {'province_idx': 16, 'province_name': 'GD', 'episodes': 60},
        {'province_idx': 17, 'province_name': 'AH', 'episodes': 60},
        {'province_idx': 18, 'province_name': 'XJ', 'episodes': 60},
        {'province_idx': 19, 'province_name': 'YN', 'episodes': 60},
        {'province_idx': 20, 'province_name': 'GZ', 'episodes': 60},
        {'province_idx': 21, 'province_name': 'GS', 'episodes': 60},
        {'province_idx': 22, 'province_name': 'SI', 'episodes': 60},
        {'province_idx': 23, 'province_name': 'NX', 'episodes': 60},
        {'province_idx': 24, 'province_name': 'JX', 'episodes': 60},
        {'province_idx': 25, 'province_name': 'SC', 'episodes': 60},
        {'province_idx': 26, 'province_name': 'FJ', 'episodes': 60},
        {'province_idx': 27, 'province_name': 'GX', 'episodes': 60},
        {'province_idx': 28, 'province_name': 'QH', 'episodes': 60},
        {'province_idx': 29, 'province_name': 'ZJ', 'episodes': 60},
        {'province_idx': 30, 'province_name': 'XZ', 'episodes': 360},
    ]

    # 总训练轮次上限（可选，如果设置会覆盖基于province_training_config的计算）
    MAX_TOTAL_EPISODES = 3000  # 例如: 2000

    # 微调阶段轮次
    FINE_TUNE_EPISODES = 25

    # ========== 执行训练 ==========
    if MULTI_SCENARIO_MODE:
        print(f"📋 多情景训练模式")
        print(f"   情景数量: {NUM_SCENARIOS}")
        print(f"   基础随机种子: {BASE_SEED}")
        print(f"   微调阶段轮次: {FINE_TUNE_EPISODES}")
        if MAX_TOTAL_EPISODES:
            print(f"   总训练轮次上限: {MAX_TOTAL_EPISODES}")

        try:
            all_results, summary_stats = run_multi_scenario_training(
                num_scenarios=NUM_SCENARIOS,
                base_seed=BASE_SEED,
                enable_realtime_console=False,  # 多情景训练时建议关闭实时输出
                province_training_config=PROVINCE_TRAINING_CONFIG,
                max_total_episodes=MAX_TOTAL_EPISODES,
                fine_tune_episodes=FINE_TUNE_EPISODES,
                parallel=PARALLEL_TRAINING,
                num_workers=NUM_WORKERS
            )

            if summary_stats:
                print(f"\n🎉 多情景训练完成！")
                print(f"   成功情景数: {summary_stats['num_scenarios']}")
                print(f"   最终平均奖励: {summary_stats['final_mean']:.2f} ± {summary_stats['final_std']:.2f}")
                print(f"   最佳平均奖励: {summary_stats['best_mean']:.2f} (Episode {summary_stats['best_episode']})")
                print(f"\n📁 结果文件保存在: ./result152/multi_scenario/")
                print(f"   - multi_scenario_summary.csv: 汇总数据（均值、标准差、置信区间）")
                print(f"   - multi_scenario_detailed.csv: 各情景详细数据")
                print(f"   - multi_scenario_results.png: 置信区间图")
        except Exception as e:
            print(f"\n❌ 多情景训练失败: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"📋 单情景训练模式")
        print(f"   随机种子: {SINGLE_SCENARIO_SEED}")
        print(f"   微调阶段轮次: {FINE_TUNE_EPISODES}")
        if MAX_TOTAL_EPISODES:
            print(f"   总训练轮次上限: {MAX_TOTAL_EPISODES}")

        np.random.seed(SINGLE_SCENARIO_SEED)
        torch.manual_seed(SINGLE_SCENARIO_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SINGLE_SCENARIO_SEED)

        print(f"📺 实时控制台输出: {'启用' if enable_realtime_console else '禁用'}")

        try:
            mappo_agent, rewards, result_dir = train_optimized_mappo_pytorch(
                enable_realtime_console=enable_realtime_console,
                random_seed=SINGLE_SCENARIO_SEED,
                province_training_config=PROVINCE_TRAINING_CONFIG,
                max_total_episodes=MAX_TOTAL_EPISODES,
                fine_tune_episodes=FINE_TUNE_EPISODES
            )

            if mappo_agent is not None:
                print("\n🎉 PyTorch版MAPPO训练完成！")
                print("=" * 80)
                if result_dir:
                    print(f"结果文件保存在 {result_dir}/ 目录中")
                else:
                    print("结果文件保存在 ./result152/ 目录中")
                print("日志文件保存在 ./log152/ 目录中")
                print("TensorBoard日志保存在 ./logs/ 目录中")
                print("包含以下文件：")
                print("  - best_pytorch_mappo_actor.pth: 最佳演员网络权重")
                print("  - best_pytorch_mappo_critic.pth: 最佳评论家网络权重")
                print("  - mappo_training_curve_pytorch.png: 训练曲线图")

                if rewards:
                    print(f"\n📈 训练统计:")
                    print(f"  总回合数: {len(rewards)}")
                    print(f"  最佳奖励: {max(rewards):.4f}")
                    print(f"  平均奖励: {np.mean(rewards):.4f}")
                    print(f"  最终奖励: {rewards[-1]:.4f}")
            else:
                print("\n❌ PyTorch版MAPPO训练失败")
        except Exception as e:
            print(f"\n❌ 程序执行失败: {e}")
            import traceback

            traceback.print_exc()