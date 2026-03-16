# ga_optimization.py - Genetic algorithm baseline for emission control
"""
Use a genetic algorithm to optimize an 8-step emission-reduction strategy for
31 provinces.

Reward components are aligned with the MAPPO setting whenever possible:
- PM2.5 target reward
- cost penalty
- cumulative health benefit
- coordination reward
- ranking/game-style competition reward
- fairness reward

Each chromosome encodes one full multi-step strategy.
"""

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import json
import copy
from typing import List, Tuple, Dict, Optional
import random
import time
import sys
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial

# Windows requires the spawn start method
if sys.platform == 'win32':
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Ignore the error if the start method is already set.

# Reuse the shared environment utilities and fairness-aware environment
from rsm_env import (
    RunningMeanStd, set_seed, huber_loss, relative_loss, r2_metric, rmse_metric,
    RSMEmissionEnv
)
from experiment_configs import GA2_CONFIG, get_required_files

# Set a default random seed
set_seed(42)

# Set TensorFlow logging flags before worker processes are spawned
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow warnings.


# ==================== Parallel evaluation helpers ====================

# Global worker configuration for multiprocessing
_global_env_config = None

def _init_worker(env_config):
    """Initialize a worker process with its own environment config."""
    import os
    global _global_env_config
    _global_env_config = env_config
    
    # Set the environment variable before TensorFlow is imported in the worker.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Each worker is independent, so TensorFlow can manage its own threads.

def _evaluate_chromosome_worker(chromosome_data: Tuple) -> Tuple[int, float, Dict]:
    """
    Evaluate a single chromosome inside a worker process.

    Args:
        chromosome_data: Tuple of `(index, genes, weights)`.

    Returns:
        Tuple of `(index, fitness, fitness_components)`.
    """
    import os
    import sys
    
    # Silence TensorFlow logs in multiprocessing mode.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Standard output can be redirected here if worker logs become too noisy.
    # sys.stdout = open(os.devnull, 'w')
    
    index, genes, weights = chromosome_data
    
    # Create an isolated environment instance inside each worker.
    try:
        # Delay the import to avoid serialization issues.
        from rsm_env import RSMEmissionEnv
        
        env = RSMEmissionEnv(
            model_path=_global_env_config['model_path'],
            scaler_path=_global_env_config['scaler_path'],
            base_conc_path=_global_env_config['base_conc_path'],
            clean_conc_path=_global_env_config['clean_conc_path'],
            province_map_path=_global_env_config['province_map_path'],
            base_emission_path=_global_env_config['base_emission_path'],
            cost_data_path=_global_env_config['cost_data_path'],
            transport_matrix_path=_global_env_config.get('transport_matrix_path', "./other data/transport.csv"),
            fairness_weight=_global_env_config.get('fairness_weight', 10000),
            fairness_metric=_global_env_config.get('fairness_metric', 'l2'),
            fairness_mode=_global_env_config.get('fairness_mode', 'penalty'),
            fairness_external_only=_global_env_config.get('fairness_external_only', False),
            max_steps=_global_env_config['max_steps']
        )
    except Exception as e:
        # 如果环境创建失败，返回默认值
        print(f"⚠️ 进程 {os.getpid()} 环境创建失败: {e}")
        return index, float('-inf'), {}
    
    # 重置环境
    env.reset()
    
    # ✅ 基础奖励
    total_target_reward = 0.0
    total_cost_penalty = 0.0
    total_health_reward = 0.0  # 累积健康效益
    
    # ✅ 协作奖励（区域人口加权PM2.5改善）
    total_coordination_reward = 0.0
    
    # ✅ 竞争奖励
    total_ranking_reward = 0.0  # 排名竞争奖励（零和）
    total_game_reward = 0.0  # 省份博弈奖励（中央拨款）

    # ✅ 公平性奖励（责任份额 vs 成本份额）
    total_fairness_reward = 0.0
    
    all_pm25 = []
    all_reduction_rates = []
    
    # 执行8步减排
    for step in range(env.max_steps):
        actions = genes[step]  # 直接使用基因数据
        
        # 执行环境步骤
        _, reward, done, info = env.step(actions)
        
        # 提取奖励组成部分
        reward_components = info.get('reward_components', {})
        
        # ✅ 基础奖励
        total_target_reward += reward_components.get('total_target_reward', 0)
        total_cost_penalty += reward_components.get('total_cost_penalty', 0)
        total_health_reward += reward_components.get('total_health_reward', 0)  # 累积健康效益
        
        # ✅ 协作奖励
        total_coordination_reward += reward_components.get('coordination_reward', 0)

        # ✅ 公平性奖励
        total_fairness_reward += reward_components.get('total_fairness_reward', 0)
        
        # ✅ 竞争奖励（从coordination_components中提取）
        coordination_components = reward_components.get('coordination_components', {})
        if coordination_components and isinstance(coordination_components, dict):
            # 排名竞争奖励
            ranking_rewards = coordination_components.get('ranking_rewards', None)
            if ranking_rewards is not None:
                if isinstance(ranking_rewards, np.ndarray):
                    total_ranking_reward += float(np.sum(ranking_rewards))
                elif isinstance(ranking_rewards, dict):
                    total_ranking_reward += sum(ranking_rewards.values())
            
            # 省份博弈奖励（中央拨款）
            game_rewards = coordination_components.get('province_game_rewards', None)
            if game_rewards is not None:
                if isinstance(game_rewards, np.ndarray):
                    total_game_reward += float(np.sum(game_rewards))
                elif isinstance(game_rewards, dict):
                    total_game_reward += sum(game_rewards.values())
        
        # 记录PM2.5浓度
        predicted_pm25 = info.get('predicted_pm25', [])
        if len(predicted_pm25) > 0:
            all_pm25.append(np.mean(predicted_pm25))
        
        # 记录减排率
        reduction_rate = 1.0 - np.mean(env.cumulative_factors)
        all_reduction_rates.append(reduction_rate)
        
        if done:
            break
    
    # ✅ 计算综合适应度（包含所有奖励组成部分）
    # 适应度 = 基础奖励（target + cost + health）+ 协作奖励 + 竞争奖励（ranking + game）
    fitness = (
        weights['pm25_target'] * total_target_reward +
        weights['cost'] * total_cost_penalty +
        weights['health'] * total_health_reward +
        weights['coordination'] * total_coordination_reward +
        weights.get('ranking', 1.0) * total_ranking_reward +
        weights.get('game', 1.0) * total_game_reward +
        weights.get('fairness', 1.0) * total_fairness_reward
    )
    
    # ✅ 构建适应度分量（包含所有奖励组成部分）
    fitness_components = {
        # 基础奖励
        'pm25_target': total_target_reward,
        'cost': total_cost_penalty,
        'health': total_health_reward,  # 累积健康效益
        # 协作奖励
        'coordination': total_coordination_reward,
        # 竞争奖励
        'ranking': total_ranking_reward,  # 排名竞争奖励（零和）
        'game': total_game_reward,  # 省份博弈奖励（中央拨款）
        # 公平性奖励
        'fairness': total_fairness_reward,
        # 统计信息
        'final_pm25': all_pm25[-1] if all_pm25 else 0,
        'final_reduction': all_reduction_rates[-1] if all_reduction_rates else 0
    }
    
    return index, fitness, fitness_components


# ==================== 日志记录器 ====================

class GALogger:
    """遗传算法日志记录器"""

    def __init__(self, log_dir: str = "./log_GA"):
        """
        初始化日志记录器

        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建日志文件
        self.main_log_path = os.path.join(log_dir, f"ga_main_{self.timestamp}.log")
        self.generation_log_path = os.path.join(log_dir, f"ga_generations_{self.timestamp}.log")
        self.progress_log_path = os.path.join(log_dir, f"ga_progress_{self.timestamp}.log")
        self.best_log_path = os.path.join(log_dir, f"ga_best_solutions_{self.timestamp}.log")

        # 打开日志文件（使用无缓冲模式，确保实时写入）
        self.main_log = open(self.main_log_path, 'w', encoding='utf-8', buffering=1)  # 行缓冲
        self.generation_log = open(self.generation_log_path, 'w', encoding='utf-8', buffering=1)
        self.progress_log = open(self.progress_log_path, 'w', encoding='utf-8', buffering=1)
        self.best_log = open(self.best_log_path, 'w', encoding='utf-8', buffering=1)

        # 写入CSV头（包含所有奖励组成部分）
        self.generation_csv_path = os.path.join(log_dir, f"ga_generations_{self.timestamp}.csv")
        self.generation_csv = open(self.generation_csv_path, 'w', encoding='utf-8', buffering=1)
        self.generation_csv.write("generation,best_fitness,avg_fitness,worst_fitness,std_fitness,"
                                  "best_pm25,best_cost,best_health,best_coordination,"
                                  "best_ranking,best_game,best_fairness,"
                                  "diversity,elapsed_time,total_time\n")
        self.generation_csv.flush()

        # 记录开始时间
        self.start_time = time.time()
        self.generation_start_time = None
        self.last_progress_update_time = 0  # 上次更新进度文件的时间（初始化为0，确保第一次更新）
        self.progress_update_interval = 60  # 进度更新间隔（秒），1分钟

        # 写入初始信息
        self._log_header()

        # 初始化进度文件
        self._init_progress_file()

    def _log_header(self):
        """写入日志头"""
        header = f"""
{'=' * 80}
🧬 遗传算法优化日志
{'=' * 80}
开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
日志目录: {self.log_dir}
{'=' * 80}

日志文件说明:
- ga_main_*.log: 主要运行日志
- ga_generations_*.log: 每代详细信息
- ga_generations_*.csv: 每代数据（可用Excel打开）
- ga_progress_*.log: 进度更新（每分钟更新一次）
- ga_best_solutions_*.log: 最优解记录

"""
        self.main_log.write(header)
        self.main_log.flush()
        import os
        os.fsync(self.main_log.fileno())

    def _init_progress_file(self):
        """初始化进度文件"""
        progress_msg = f"""
{'=' * 60}
🧬 遗传算法优化进度
{'=' * 60}
状态: 初始化中...
开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
更新时间: {datetime.now().strftime("%H:%M:%S")}
说明: 进度文件每分钟更新一次
{'=' * 60}
"""
        self.progress_log.write(progress_msg)
        self.progress_log.flush()
        import os
        os.fsync(self.progress_log.fileno())

    def log_config(self, config: Dict):
        """记录配置信息"""
        msg = f"""
{'=' * 60}
📋 优化配置
{'=' * 60}
种群大小: {config.get('population_size', 'N/A')}
进化代数: {config.get('generations', 'N/A')}
交叉概率: {config.get('crossover_rate', 'N/A')}
变异概率: {config.get('mutation_rate', 'N/A')}
精英比例: {config.get('elite_ratio', 'N/A')}
锦标赛大小: {config.get('tournament_size', 'N/A')}
染色体维度: {config.get('chromosome_dim', 'N/A')}
权重配置: {config.get('weights', 'N/A')}
{'=' * 60}

"""
        self.main_log.write(msg)
        self.main_log.flush()
        import os
        os.fsync(self.main_log.fileno())
        print(msg)

    def log_generation_start(self, generation: int, total_generations: int):
        """记录代开始"""
        self.generation_start_time = time.time()
        elapsed = time.time() - self.start_time
        current_time = time.time()

        # 估计剩余时间
        if generation > 0:
            avg_time_per_gen = elapsed / generation
            remaining_gens = total_generations - generation
            eta = avg_time_per_gen * remaining_gens
            eta_str = self._format_time(eta)
        else:
            eta_str = "计算中..."

        progress = (generation / total_generations) * 100

        # 更新进度文件（每分钟更新一次，但第一代总是更新）
        should_update = (
                generation == 0 or  # 第一代总是更新
                current_time - self.last_progress_update_time >= self.progress_update_interval
        )

        if should_update:
            progress_msg = f"""
{'=' * 60}
🧬 遗传算法优化进度
{'=' * 60}
当前代数: {generation + 1} / {total_generations}
进度: {progress:.1f}% [{'█' * int(progress / 2)}{' ' * (50 - int(progress / 2))}]
已用时间: {self._format_time(elapsed)}
预计剩余: {eta_str}
更新时间: {datetime.now().strftime("%H:%M:%S")}
状态: 正在评估种群...
{'=' * 60}
"""
            self.progress_log.seek(0)  # 回到文件开头
            self.progress_log.truncate()  # 清空文件
            self.progress_log.write(progress_msg)
            self.progress_log.flush()
            import os
            os.fsync(self.progress_log.fileno())  # 强制同步到磁盘
            self.last_progress_update_time = current_time

        # 控制台输出（实时显示）
        if generation == 0:
            print(f"\n⏳ 开始第 {generation + 1}/{total_generations} 代...")
        else:
            print(f"\r⏳ 进度: {progress:.1f}% | 代数: {generation + 1}/{total_generations} | "
                  f"已用: {self._format_time(elapsed)} | 剩余: {eta_str}", end='', flush=True)

    def log_generation_end(self, generation: int, total_generations: int,
                           fitnesses: List[float], best_chromosome, diversity: float):
        """记录代结束"""
        gen_time = time.time() - self.generation_start_time
        total_time = time.time() - self.start_time

        # 计算统计信息
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        worst_fitness = min(fitnesses)
        std_fitness = np.std(fitnesses)

        # 获取最优解的组成部分（包含所有奖励）
        if best_chromosome and best_chromosome.fitness_components:
            comp = best_chromosome.fitness_components
            best_pm25 = comp.get('final_pm25', 0)
            best_cost = comp.get('cost', 0)
            best_health = comp.get('health', 0)
            best_coordination = comp.get('coordination', 0)
            best_ranking = comp.get('ranking', 0)  # 排名竞争奖励
            best_game = comp.get('game', 0)  # 省份博弈奖励（中央拨款）
            best_fairness = comp.get('fairness', 0)  # 公平性奖励
        else:
            best_pm25 = best_cost = best_health = best_coordination = best_ranking = best_game = best_fairness = 0

        # 写入详细日志
        gen_msg = f"""
{'=' * 60}
📊 第 {generation + 1}/{total_generations} 代 完成
{'=' * 60}
⏱️  本代用时: {gen_time:.2f}秒 | 总用时: {self._format_time(total_time)}

📈 适应度统计:
   最优: {best_fitness:>12.2f}
   平均: {avg_fitness:>12.2f}
   最差: {worst_fitness:>12.2f}
   标准差: {std_fitness:>10.2f}

🎯 最优解详情（基础奖励）:
   PM2.5目标奖励: {comp.get('pm25_target', 0):>10.2f}
   成本惩罚: {best_cost:>10.2f}
   累积健康效益: {best_health:>10.2f}

🤝 协作奖励:
   区域协作奖励: {best_coordination:>10.2f}

🏆 竞争奖励:
   排名竞争奖励: {best_ranking:>10.2f} (零和)
   省份博弈奖励: {best_game:>10.2f} (中央拨款)

⚖️ 公平性奖励:
   公平性奖励: {best_fairness:>10.2f}

📊 最终结果:
   最终PM2.5: {best_pm25:>10.2f} μg/m³
   最终减排率: {comp.get('final_reduction', 0) * 100:>8.1f}%

🌈 种群多样性: {diversity:.6f}
{'=' * 60}
"""
        self.generation_log.write(gen_msg)
        self.generation_log.flush()
        import os
        os.fsync(self.generation_log.fileno())

        # 写入CSV（包含所有奖励组成部分）
        csv_line = f"{generation + 1},{best_fitness:.4f},{avg_fitness:.4f}," \
                   f"{worst_fitness:.4f},{std_fitness:.4f}," \
                   f"{best_pm25:.4f},{best_cost:.4f},{best_health:.4f}," \
                   f"{best_coordination:.4f},{best_ranking:.4f},{best_game:.4f},{best_fairness:.4f}," \
                   f"{diversity:.6f}," \
                   f"{gen_time:.2f},{total_time:.2f}\n"
        self.generation_csv.write(csv_line)
        self.generation_csv.flush()
        import os
        os.fsync(self.generation_csv.fileno())

        # 更新进度文件（代完成，每分钟更新一次）
        current_time = time.time()
        progress = ((generation + 1) / total_generations) * 100

        # 更新条件：第一代、超过更新间隔、或最后一代
        should_update_progress = (
                generation == 0 or  # 第一代总是更新
                current_time - self.last_progress_update_time >= self.progress_update_interval or
                generation + 1 == total_generations  # 最后一代总是更新
        )

        if should_update_progress:
            if generation + 1 > 0:
                avg_time_per_gen = total_time / (generation + 1)
                remaining_gens = total_generations - (generation + 1)
                eta = avg_time_per_gen * remaining_gens
                eta_str = self._format_time(eta)
            else:
                eta_str = "计算中..."

            progress_msg = f"""
{'=' * 60}
🧬 遗传算法优化进度
{'=' * 60}
当前代数: {generation + 1} / {total_generations}
进度: {progress:.1f}% [{'█' * int(progress / 2)}{' ' * (50 - int(progress / 2))}]
已用时间: {self._format_time(total_time)}
预计剩余: {eta_str}
更新时间: {datetime.now().strftime("%H:%M:%S")}
状态: 第{generation + 1}代完成 | 最优适应度: {best_fitness:.2f} | 平均PM2.5: {best_pm25:.2f}μg/m³
{'=' * 60}
"""
            self.progress_log.seek(0)
            self.progress_log.truncate()
            self.progress_log.write(progress_msg)
            self.progress_log.flush()
            import os
            os.fsync(self.progress_log.fileno())
            self.last_progress_update_time = current_time

        # 每代都在控制台输出简要信息
        print(
            f"\r✅ 第{generation + 1}代完成 | 最优={best_fitness:.2f} | 平均={avg_fitness:.2f} | PM2.5={best_pm25:.2f}μg/m³ | 用时={gen_time:.1f}秒",
            end='', flush=True)

        # 每10代在主日志中记录摘要
        if (generation + 1) % 10 == 0 or generation == 0:
            summary = f"""
[{datetime.now().strftime("%H:%M:%S")}] 第{generation + 1}代: 最优={best_fitness:.2f}, 平均={avg_fitness:.2f}, PM2.5={best_pm25:.2f}μg/m³
"""
            self.main_log.write(summary)
            self.main_log.flush()
            os.fsync(self.main_log.fileno())
            print()  # 换行
            print(summary.strip())

    def log_new_best(self, generation: int, fitness: float, chromosome):
        """记录新的最优解"""
        msg = f"""
{'=' * 60}
🏆 发现新最优解! (第 {generation + 1} 代)
{'=' * 60}
适应度: {fitness:.4f}
时间: {datetime.now().strftime("%H:%M:%S")}
"""
        if chromosome.fitness_components:
            comp = chromosome.fitness_components
            msg += f"""
详细分解（基础奖励）:
  PM2.5目标奖励: {comp.get('pm25_target', 0):.2f}
  成本惩罚: {comp.get('cost', 0):.2f}
  累积健康效益: {comp.get('health', 0):.2f}

协作奖励:
  区域协作奖励: {comp.get('coordination', 0):.2f}

竞争奖励:
  排名竞争奖励: {comp.get('ranking', 0):.2f} (零和)
  省份博弈奖励: {comp.get('game', 0):.2f} (中央拨款)

最终结果:
  最终PM2.5: {comp.get('final_pm25', 0):.2f} μg/m³
  最终减排率: {comp.get('final_reduction', 0) * 100:.1f}%
"""
        msg += f"{'=' * 60}\n"

        self.best_log.write(msg)
        self.best_log.flush()
        import os
        os.fsync(self.best_log.fileno())

        self.main_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] 🏆 新最优解: {fitness:.4f}\n")
        self.main_log.flush()
        os.fsync(self.main_log.fileno())

    def log_evaluation_progress(self, current: int, total: int):
        """记录评估进度（仅控制台输出，不更新文件）"""
        if current % 10 == 0 or current == total:
            progress_pct = current / total * 100
            print(f"\r   评估进度: {current}/{total} ({progress_pct:.1f}%)", end='', flush=True)

    def log_parallel_progress(self, completed: int, total: int, elapsed: float):
        """记录并行评估进度（实时更新进度文件）"""
        current_time = time.time()
        
        # 更新进度文件
        progress_pct = completed / total * 100
        avg_time = elapsed / completed if completed > 0 else 0
        remaining = avg_time * (total - completed)
        
        progress_msg = f"""
{'=' * 60}
🔄 并行评估进度
{'=' * 60}
评估进度: {completed} / {total} ({progress_pct:.1f}%)
进度条: [{'█' * int(progress_pct / 2)}{' ' * (50 - int(progress_pct / 2))}]
已用时间: {self._format_time(elapsed)}
预计剩余: {self._format_time(remaining)}
平均每个: {avg_time:.2f}秒
更新时间: {datetime.now().strftime("%H:%M:%S")}
{'=' * 60}
"""
        self.progress_log.seek(0)
        self.progress_log.truncate()
        self.progress_log.write(progress_msg)
        self.progress_log.flush()
        os.fsync(self.progress_log.fileno())

    def log_message(self, message: str, level: str = "INFO"):
        """记录一般消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}\n"
        self.main_log.write(formatted)
        self.main_log.flush()
        import os
        os.fsync(self.main_log.fileno())
        if level in ["ERROR", "WARNING"]:
            print(formatted.strip())

    def log_final_summary(self, best_fitness: float, best_chromosome,
                          total_generations: int, history: Dict):
        """记录最终摘要"""
        total_time = time.time() - self.start_time

        summary = f"""

{'=' * 80}
🎉 遗传算法优化完成!
{'=' * 80}

📊 优化结果摘要:
{'─' * 40}
总进化代数: {total_generations}
总用时: {self._format_time(total_time)}
平均每代用时: {total_time / total_generations:.2f}秒

🏆 最终最优解:
{'─' * 40}
适应度: {best_fitness:.4f}
"""
        if best_chromosome and best_chromosome.fitness_components:
            comp = best_chromosome.fitness_components
            summary += f"""
基础奖励:
  PM2.5目标奖励: {comp.get('pm25_target', 0):.2f}
  成本惩罚: {comp.get('cost', 0):.2f}
  累积健康效益: {comp.get('health', 0):.2f}

协作奖励:
  区域协作奖励: {comp.get('coordination', 0):.2f}

竞争奖励:
  排名竞争奖励: {comp.get('ranking', 0):.2f} (零和)
  省份博弈奖励: {comp.get('game', 0):.2f} (中央拨款)

最终结果:
  最终PM2.5: {comp.get('final_pm25', 0):.2f} μg/m³
  最终减排率: {comp.get('final_reduction', 0) * 100:.1f}%
"""

        # 计算改进情况
        if len(history['best_fitness']) > 1:
            initial_fitness = history['best_fitness'][0]
            improvement = best_fitness - initial_fitness
            improvement_pct = (improvement / abs(initial_fitness)) * 100 if initial_fitness != 0 else 0
            summary += f"""
📈 优化改进:
{'─' * 40}
初始最优适应度: {initial_fitness:.4f}
最终最优适应度: {best_fitness:.4f}
绝对改进: {improvement:.4f}
相对改进: {improvement_pct:.2f}%
"""

        summary += f"""
{'=' * 80}
结束时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'=' * 80}
"""

        self.main_log.write(summary)
        self.main_log.flush()
        print(summary)

        # 更新最终进度
        final_progress_msg = f"""
{'=' * 60}
✅ 遗传算法优化完成!
{'=' * 60}
完成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
总用时: {self._format_time(total_time)}
最优适应度: {best_fitness:.4f}
{'=' * 60}
"""
        self.progress_log.seek(0)
        self.progress_log.truncate()
        self.progress_log.write(final_progress_msg)
        self.progress_log.flush()
        import os
        os.fsync(self.progress_log.fileno())

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"

    def close(self):
        """关闭所有日志文件"""
        self.main_log.close()
        self.generation_log.close()
        self.progress_log.close()
        self.best_log.close()
        self.generation_csv.close()
        print(f"\n📁 日志已保存到: {self.log_dir}")


class GAChromosome:
    """遗传算法染色体：表示一个完整的多步减排方案"""

    def __init__(self, num_provinces: int, action_dim: int, max_steps: int):
        """
        初始化染色体

        Args:
            num_provinces: 省份数量 (31)
            action_dim: 每个省份的动作维度 (25 = 5污染物 × 5行业)
            max_steps: 最大步数 (8)
        """
        self.num_provinces = num_provinces
        self.action_dim = action_dim
        self.max_steps = max_steps

        # 基因：每一步每个省份的减排率 [0, 0.2]
        # 形状：(max_steps, num_provinces, action_dim)
        self.genes = np.random.uniform(0.01, 0.15, (max_steps, num_provinces, action_dim))

        # 适应度值
        self.fitness = None
        self.fitness_components = None  # 存储各个目标的分值

    def copy(self) -> 'GAChromosome':
        """深拷贝染色体"""
        new_chromosome = GAChromosome(self.num_provinces, self.action_dim, self.max_steps)
        new_chromosome.genes = self.genes.copy()
        new_chromosome.fitness = self.fitness
        new_chromosome.fitness_components = copy.deepcopy(self.fitness_components) if self.fitness_components else None
        return new_chromosome

    def get_actions_for_step(self, step: int) -> np.ndarray:
        """获取某一步的所有省份动作"""
        return self.genes[step]


class GAOptimizer:
    """遗传算法优化器"""

    def __init__(self,
                 env: RSMEmissionEnv,
                 population_size: int = 200,
                 generations: int = 500,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_ratio: float = 0.1,
                 tournament_size: int = 5,
                 log_dir: str = "./log_GA",
                 n_jobs: int = -1,
                 env_config: Dict = None):
        """
        初始化GA优化器

        Args:
            env: 减排环境
            population_size: 种群大小
            generations: 进化代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_ratio: 精英保留比例
            tournament_size: 锦标赛选择的参与个体数
            log_dir: 日志保存目录
            n_jobs: 并行工作进程数，-1表示使用所有CPU核心
        """
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size
        
        # 并行设置
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count()))
        
        # 保存环境配置（用于多进程）
        if env_config is None:
            # 默认配置（从run_ga_optimization传入）
            self.env_config = {
                'model_path': "./models_unet",
                'scaler_path': "./models_unet",
                'base_conc_path': "./conc/base/base.csv",
                'clean_conc_path': "./conc/clean/clean.csv",
                'province_map_path': "./prov_grid_map/36kmprov.csv",
                'base_emission_path': "./input_emi/base",
                'cost_data_path': "./other data/cost.csv",
                'transport_matrix_path': "./other data/transport.csv",
                'fairness_weight': 10000,
                'fairness_metric': 'l2',
                'fairness_mode': 'penalty',
                'fairness_external_only': False,
                'max_steps': env.max_steps
            }
        else:
            self.env_config = env_config

        # 问题维度
        self.num_provinces = env.num_provinces
        self.action_dim = env.action_dim
        self.max_steps = env.max_steps

        # 初始化日志记录器
        self.logger = GALogger(log_dir)
        
        # 性能统计
        self.evaluation_times = []
        self.parallel_evaluation_count = 0
        self.serial_evaluation_count = 0
        
        print(f"🚀 并行设置: 使用 {self.n_jobs} 个工作进程")
        if self.n_jobs > 1:
            print(f"   预期加速比: ~{self.n_jobs}x (理想情况)")
            print(f"   注意: 实际加速比取决于环境评估时间和进程开销")

        # 种群
        self.population: List[GAChromosome] = []

        # 进化历史（包含所有奖励组成部分）
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_pm25': [],
            'best_cost': [],
            'best_health': [],
            'best_coordination': [],
            'best_ranking': [],  # 排名竞争奖励
            'best_game': [],  # 省份博弈奖励（中央拨款）
            'best_fairness': [],  # 公平性奖励
            'diversity': []
        }

        # 最优解
        self.best_chromosome: Optional[GAChromosome] = None
        self.best_fitness = float('-inf')

        # 权重配置（可调整各目标的重要性）
        self.weights = {
            'pm25_target': 1.0,  # PM2.5达标权重
            'cost': 1.0,  # 成本权重（负向）
            'health': 1.0,  # 累积健康效益权重
            'coordination': 1.0,  # 区域协作权重
            'ranking': 1.0,  # 排名竞争奖励权重（零和）
            'game': 1.0,  # 省份博弈奖励权重（中央拨款）
            'fairness': 1.0  # 公平性奖励权重（责任份额 vs 成本份额）
        }

        # 记录配置
        self.logger.log_config({
            'population_size': population_size,
            'generations': generations,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'elite_ratio': elite_ratio,
            'tournament_size': tournament_size,
            'chromosome_dim': f"{self.max_steps} × {self.num_provinces} × {self.action_dim} = {self.max_steps * self.num_provinces * self.action_dim}",
            'weights': self.weights
        })

    def initialize_population(self):
        """初始化种群"""
        print(f"\n🌱 初始化种群 (大小: {self.population_size})...")

        self.population = []

        for i in range(self.population_size):
            chromosome = GAChromosome(self.num_provinces, self.action_dim, self.max_steps)

            # 使用不同的初始化策略增加多样性
            if i < self.population_size // 4:
                # 保守策略：低减排率
                chromosome.genes = np.random.uniform(0.01, 0.08, chromosome.genes.shape)
            elif i < self.population_size // 2:
                # 激进策略：高减排率
                chromosome.genes = np.random.uniform(0.10, 0.20, chromosome.genes.shape)
            elif i < 3 * self.population_size // 4:
                # 渐进策略：逐步增加减排率
                for step in range(self.max_steps):
                    progress = (step + 1) / self.max_steps
                    low = 0.01 + progress * 0.05
                    high = 0.05 + progress * 0.15
                    chromosome.genes[step] = np.random.uniform(low, high, (self.num_provinces, self.action_dim))
            # 其余使用默认随机初始化

            self.population.append(chromosome)

        print(f"   ✅ 种群初始化完成")

    def evaluate_fitness(self, chromosome: GAChromosome) -> float:
        """
        评估染色体的适应度

        执行完整的8步减排模拟，计算综合适应度
        包含：基础奖励（target + cost + health）+ 协作奖励 + 竞争奖励（ranking + game）
        """
        # 重置环境
        self.env.reset()

        # ✅ 基础奖励
        total_target_reward = 0.0
        total_cost_penalty = 0.0
        total_health_reward = 0.0  # 累积健康效益
        
        # ✅ 协作奖励
        total_coordination_reward = 0.0
        
        # ✅ 竞争奖励
        total_ranking_reward = 0.0  # 排名竞争奖励（零和）
        total_game_reward = 0.0  # 省份博弈奖励（中央拨款）

        all_pm25 = []
        all_reduction_rates = []

        # 执行8步减排
        for step in range(self.max_steps):
            actions = chromosome.get_actions_for_step(step)

            # 执行环境步骤
            _, reward, done, info = self.env.step(actions)

            # 提取奖励组成部分
            reward_components = info.get('reward_components', {})
            
            # ✅ 基础奖励
            total_target_reward += reward_components.get('total_target_reward', 0)
            total_cost_penalty += reward_components.get('total_cost_penalty', 0)
            total_health_reward += reward_components.get('total_health_reward', 0)  # 累积健康效益
            
            # ✅ 协作奖励
            total_coordination_reward += reward_components.get('coordination_reward', 0)
            
            # ✅ 竞争奖励（从coordination_components中提取）
            coordination_components = reward_components.get('coordination_components', {})
            if coordination_components and isinstance(coordination_components, dict):
                # 排名竞争奖励
                ranking_rewards = coordination_components.get('ranking_rewards', None)
                if ranking_rewards is not None:
                    if isinstance(ranking_rewards, np.ndarray):
                        total_ranking_reward += float(np.sum(ranking_rewards))
                    elif isinstance(ranking_rewards, dict):
                        total_ranking_reward += sum(ranking_rewards.values())
                
                # 省份博弈奖励（中央拨款）
                game_rewards = coordination_components.get('province_game_rewards', None)
                if game_rewards is not None:
                    if isinstance(game_rewards, np.ndarray):
                        total_game_reward += float(np.sum(game_rewards))
                    elif isinstance(game_rewards, dict):
                        total_game_reward += sum(game_rewards.values())

            # 记录PM2.5浓度
            predicted_pm25 = info.get('predicted_pm25', [])
            if len(predicted_pm25) > 0:
                all_pm25.append(np.mean(predicted_pm25))

            # 记录减排率
            reduction_rate = 1.0 - np.mean(self.env.cumulative_factors)
            all_reduction_rates.append(reduction_rate)

            if done:
                break

        # ✅ 计算综合适应度（包含所有奖励组成部分）
        fitness = (
                self.weights['pm25_target'] * total_target_reward +
                self.weights['cost'] * total_cost_penalty +  # 成本是负值，所以直接加
                self.weights['health'] * total_health_reward +
                self.weights['coordination'] * total_coordination_reward +
                self.weights['ranking'] * total_ranking_reward +
                self.weights['game'] * total_game_reward
        )

        # ✅ 存储适应度分量（包含所有奖励组成部分）
        chromosome.fitness = fitness
        chromosome.fitness_components = {
            # 基础奖励
            'pm25_target': total_target_reward,
            'cost': total_cost_penalty,
            'health': total_health_reward,  # 累积健康效益
            # 协作奖励
            'coordination': total_coordination_reward,
            # 竞争奖励
            'ranking': total_ranking_reward,  # 排名竞争奖励（零和）
            'game': total_game_reward,  # 省份博弈奖励（中央拨款）
            # 统计信息
            'final_pm25': all_pm25[-1] if all_pm25 else 0,
            'final_reduction': all_reduction_rates[-1] if all_reduction_rates else 0
        }

        return fitness

    def evaluate_population(self):
        """评估整个种群的适应度（并行版本）"""
        # 找出需要评估的染色体
        to_evaluate = []
        for i, chromosome in enumerate(self.population):
            if chromosome.fitness is None:
                to_evaluate.append((i, chromosome))
        
        if not to_evaluate:
            print()  # 换行
            return
        
        # 准备并行评估数据
        chromosome_data_list = []
        for idx, chromosome in to_evaluate:
            chromosome_data_list.append((
                idx,
                chromosome.genes.copy(),  # 只传递基因数据（可序列化）
                self.weights.copy()
            ))
        
        # 并行评估
        if self.n_jobs > 1 and len(to_evaluate) > 1:
            # 使用多进程并行评估
            print(f"\n   🔄 并行评估 {len(to_evaluate)} 个个体 (使用 {self.n_jobs} 个进程)...")
            start_time = time.time()
            
            try:
                with Pool(processes=self.n_jobs, initializer=_init_worker, initargs=(self.env_config,)) as pool:
                    # 使用 imap_unordered 实现实时进度更新
                    completed = 0
                    total = len(chromosome_data_list)
                    results = []
                    
                    for result in pool.imap_unordered(_evaluate_chromosome_worker, chromosome_data_list):
                        results.append(result)
                        completed += 1
                        
                        # 实时更新进度（控制台和日志）
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed
                        remaining = avg_time * (total - completed)
                        
                        # 控制台进度条
                        progress_pct = completed / total * 100
                        bar_len = 30
                        filled = int(bar_len * completed / total)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        print(f"\r   [{bar}] {progress_pct:5.1f}% ({completed}/{total}) | "
                              f"用时: {elapsed:.0f}s | 剩余: {remaining:.0f}s", end='', flush=True)
                        
                        # 每10个个体更新一次日志
                        if completed % 10 == 0 or completed == total:
                            self.logger.log_parallel_progress(completed, total, elapsed)
                
                # 更新染色体适应度
                for idx, fitness, fitness_components in results:
                    if idx < len(self.population):
                        self.population[idx].fitness = fitness
                        self.population[idx].fitness_components = fitness_components
                
                elapsed = time.time() - start_time
                self.evaluation_times.append(elapsed)
                self.parallel_evaluation_count += len(to_evaluate)
                avg_time_per_individual = elapsed / len(to_evaluate)
                print(f"\n   ✅ 并行评估完成 (总用时 {elapsed:.1f}秒, 平均 {avg_time_per_individual:.2f}秒/个体)")
                
            except Exception as e:
                print(f"\n   ❌ 并行失败: {e}")
                print("   回退到串行评估...")
                import traceback
                traceback.print_exc()
                # 回退到串行评估
                start_time = time.time()
                for i, chromosome in to_evaluate:
                    self.evaluate_fitness(chromosome)
                    self.logger.log_evaluation_progress(i + 1, self.population_size)
                elapsed = time.time() - start_time
                self.evaluation_times.append(elapsed)
                self.serial_evaluation_count += len(to_evaluate)
        else:
            # 串行评估（单进程或只有一个需要评估）
            start_time = time.time()
            for i, chromosome in to_evaluate:
                self.evaluate_fitness(chromosome)
                self.logger.log_evaluation_progress(i + 1, self.population_size)
            elapsed = time.time() - start_time
            self.evaluation_times.append(elapsed)
            self.serial_evaluation_count += len(to_evaluate)
        
        print()  # 换行

    def tournament_selection(self) -> GAChromosome:
        """锦标赛选择"""
        tournament = random.sample(self.population, self.tournament_size)
        winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
        return winner.copy()

    def crossover(self, parent1: GAChromosome, parent2: GAChromosome) -> Tuple[GAChromosome, GAChromosome]:
        """
        交叉操作：多点交叉

        在步骤维度和省份维度进行交叉
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        # 随机选择交叉方式
        crossover_type = random.choice(['step', 'province', 'uniform'])

        if crossover_type == 'step':
            # 按步骤交叉
            crossover_point = random.randint(1, self.max_steps - 1)
            child1.genes[crossover_point:] = parent2.genes[crossover_point:].copy()
            child2.genes[crossover_point:] = parent1.genes[crossover_point:].copy()

        elif crossover_type == 'province':
            # 按省份交叉
            crossover_point = random.randint(1, self.num_provinces - 1)
            child1.genes[:, crossover_point:, :] = parent2.genes[:, crossover_point:, :].copy()
            child2.genes[:, crossover_point:, :] = parent1.genes[:, crossover_point:, :].copy()

        else:  # uniform
            # 均匀交叉
            mask = np.random.random(child1.genes.shape) < 0.5
            child1.genes = np.where(mask, parent1.genes, parent2.genes)
            child2.genes = np.where(mask, parent2.genes, parent1.genes)

        # 重置适应度
        child1.fitness = None
        child2.fitness = None

        return child1, child2

    def mutate(self, chromosome: GAChromosome) -> GAChromosome:
        """
        变异操作：多种变异策略
        """
        if random.random() > self.mutation_rate:
            return chromosome

        mutated = chromosome.copy()

        # 随机选择变异方式
        mutation_type = random.choice(['gaussian', 'uniform', 'swap', 'creep'])

        if mutation_type == 'gaussian':
            # 高斯变异：添加高斯噪声
            noise = np.random.normal(0, 0.02, mutated.genes.shape)
            mutated.genes += noise

        elif mutation_type == 'uniform':
            # 均匀变异：随机重置部分基因
            mask = np.random.random(mutated.genes.shape) < 0.1
            random_values = np.random.uniform(0.01, 0.20, mutated.genes.shape)
            mutated.genes = np.where(mask, random_values, mutated.genes)

        elif mutation_type == 'swap':
            # 交换变异：交换两个步骤的策略
            step1, step2 = random.sample(range(self.max_steps), 2)
            mutated.genes[step1], mutated.genes[step2] = mutated.genes[step2].copy(), mutated.genes[step1].copy()

        else:  # creep
            # 蠕变变异：小幅度渐变
            step = random.randint(0, self.max_steps - 1)
            province = random.randint(0, self.num_provinces - 1)
            delta = np.random.uniform(-0.03, 0.03, self.action_dim)
            mutated.genes[step, province] += delta

        # 确保基因值在有效范围内 [0.01, 0.20]
        mutated.genes = np.clip(mutated.genes, 0.01, 0.20)

        # 重置适应度
        mutated.fitness = None

        return mutated

    def calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) < 2:
            return 0.0

        # 计算所有个体基因的标准差
        all_genes = np.array([c.genes.flatten() for c in self.population])
        diversity = np.mean(np.std(all_genes, axis=0))

        return diversity

    def evolve_one_generation(self, generation: int):
        """进化一代"""
        # 记录代开始
        self.logger.log_generation_start(generation, self.generations)

        # 评估当前种群
        self.evaluate_population()

        # 按适应度排序
        self.population.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=True)

        # 更新最优解
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_chromosome = self.population[0].copy()
            # 记录新最优解
            self.logger.log_new_best(generation, self.best_fitness, self.best_chromosome)

        # 记录历史
        fitnesses = [c.fitness for c in self.population if c.fitness is not None]
        diversity = self.calculate_diversity()

        self.history['best_fitness'].append(max(fitnesses))
        self.history['avg_fitness'].append(np.mean(fitnesses))
        self.history['diversity'].append(diversity)

        if self.best_chromosome and self.best_chromosome.fitness_components:
            comp = self.best_chromosome.fitness_components
            self.history['best_pm25'].append(comp.get('final_pm25', 0))
            self.history['best_cost'].append(comp.get('cost', 0))
            self.history['best_health'].append(comp.get('health', 0))
            self.history['best_coordination'].append(comp.get('coordination', 0))
            self.history['best_ranking'].append(comp.get('ranking', 0))  # 排名竞争奖励
            self.history['best_game'].append(comp.get('game', 0))  # 省份博弈奖励（中央拨款）
            self.history['best_fairness'].append(comp.get('fairness', 0))  # 公平性奖励

        # 记录代结束
        self.logger.log_generation_end(generation, self.generations, fitnesses,
                                       self.best_chromosome, diversity)

        # 精英保留
        elite_count = max(1, int(self.population_size * self.elite_ratio))
        new_population = [c.copy() for c in self.population[:elite_count]]

        # 生成新个体
        while len(new_population) < self.population_size:
            # 选择父代
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # 交叉
            child1, child2 = self.crossover(parent1, parent2)

            # 变异
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population

    def run(self) -> GAChromosome:
        """运行遗传算法优化"""
        self.logger.log_message("开始遗传算法优化", "INFO")

        print(f"\n{'=' * 80}")
        print(f"🧬 开始遗传算法优化")
        print(f"{'=' * 80}")

        # 初始化种群
        self.logger.log_message("初始化种群...", "INFO")
        self.initialize_population()
        self.logger.log_message(f"种群初始化完成，共{self.population_size}个个体", "INFO")

        try:
            # 进化循环
            for generation in range(self.generations):
                self.evolve_one_generation(generation)

        except KeyboardInterrupt:
            self.logger.log_message("用户中断优化过程", "WARNING")
            print("\n\n⚠️ 用户中断，保存当前结果...")
        except Exception as e:
            self.logger.log_message(f"优化过程出错: {str(e)}", "ERROR")
            raise

        # 记录最终摘要
        self.logger.log_final_summary(self.best_fitness, self.best_chromosome,
                                      len(self.history['best_fitness']), self.history)

        return self.best_chromosome

    def save_results(self, save_dir: str = "./ga_results2"):
        """保存优化结果"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = self.logger.timestamp  # 使用日志的时间戳保持一致

        self.logger.log_message(f"保存结果到 {save_dir}", "INFO")

        # 保存最优策略
        if self.best_chromosome:
            strategy_path = os.path.join(save_dir, f"best_strategy_{timestamp}.npy")
            np.save(strategy_path, self.best_chromosome.genes)
            self.logger.log_message(f"最优策略已保存: {strategy_path}", "INFO")

            # 保存适应度分量
            components_path = os.path.join(save_dir, f"best_fitness_components_{timestamp}.json")
            with open(components_path, 'w') as f:
                # 🔧 修复：将numpy类型转换为Python原生类型
                def convert_numpy_types(obj):
                    """递归转换numpy类型为Python原生类型"""
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj
                
                serializable_components = convert_numpy_types(self.best_chromosome.fitness_components)
                json.dump(serializable_components, f, indent=2)
            self.logger.log_message(f"适应度分量已保存: {components_path}", "INFO")

        # 保存进化历史
        history_path = os.path.join(save_dir, f"evolution_history_{timestamp}.csv")
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(history_path, index=False)
        self.logger.log_message(f"进化历史已保存: {history_path}", "INFO")

        # 绘制进化曲线
        self.plot_evolution(save_dir, timestamp)
        
        # 输出性能统计
        if self.evaluation_times:
            total_eval_time = sum(self.evaluation_times)
            avg_eval_time = np.mean(self.evaluation_times)
            print(f"\n📊 评估性能统计:")
            print(f"   并行评估次数: {self.parallel_evaluation_count}")
            print(f"   串行评估次数: {self.serial_evaluation_count}")
            print(f"   总评估时间: {total_eval_time:.1f} 秒")
            print(f"   平均每次评估: {avg_eval_time:.2f} 秒")
            if self.parallel_evaluation_count > 0:
                parallel_avg = total_eval_time / len(self.evaluation_times) if self.evaluation_times else 0
                estimated_serial_time = parallel_avg * self.parallel_evaluation_count / self.n_jobs
                speedup = estimated_serial_time / total_eval_time if total_eval_time > 0 else 1.0
                print(f"   估算加速比: ~{speedup:.2f}x")

        # 关闭日志
        self.logger.close()

        print(f"✅ 结果已保存到 {save_dir}")

    def plot_evolution(self, save_dir: str, timestamp: str):
        """绘制进化曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. 适应度曲线
        ax1 = axes[0, 0]
        ax1.plot(self.history['best_fitness'], 'b-', linewidth=2, label='最优适应度')
        ax1.plot(self.history['avg_fitness'], 'g--', linewidth=1, alpha=0.7, label='平均适应度')
        ax1.set_xlabel('代数')
        ax1.set_ylabel('适应度')
        ax1.set_title('适应度进化曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. PM2.5浓度变化
        ax2 = axes[0, 1]
        if self.history['best_pm25']:
            ax2.plot(self.history['best_pm25'], 'r-', linewidth=2)
        ax2.set_xlabel('代数')
        ax2.set_ylabel('PM2.5 (μg/m³)')
        ax2.set_title('最优解PM2.5浓度变化')
        ax2.grid(True, alpha=0.3)

        # 3. 成本变化
        ax3 = axes[0, 2]
        if self.history['best_cost']:
            ax3.plot(self.history['best_cost'], 'm-', linewidth=2)
        ax3.set_xlabel('代数')
        ax3.set_ylabel('成本惩罚')
        ax3.set_title('最优解成本变化')
        ax3.grid(True, alpha=0.3)

        # 4. 累积健康效益变化
        ax4 = axes[1, 0]
        if self.history['best_health']:
            ax4.plot(self.history['best_health'], 'c-', linewidth=2)
        ax4.set_xlabel('代数')
        ax4.set_ylabel('累积健康效益奖励')
        ax4.set_title('最优解累积健康效益变化')
        ax4.grid(True, alpha=0.3)

        # 5. 协作奖励变化
        ax5 = axes[1, 1]
        if self.history['best_coordination']:
            ax5.plot(self.history['best_coordination'], 'orange', linewidth=2)
        ax5.set_xlabel('代数')
        ax5.set_ylabel('区域协作奖励')
        ax5.set_title('最优解区域协作奖励变化')
        ax5.grid(True, alpha=0.3)

        # 6. 种群多样性
        ax6 = axes[1, 2]
        ax6.plot(self.history['diversity'], 'purple', linewidth=2)
        ax6.set_xlabel('代数')
        ax6.set_ylabel('多样性')
        ax6.set_title('种群多样性变化')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"evolution_curves_{timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ✅ 新增：绘制竞争奖励图
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        
        # 排名竞争奖励
        ax_ranking = axes2[0]
        if self.history['best_ranking']:
            ax_ranking.plot(self.history['best_ranking'], 'green', linewidth=2)
        ax_ranking.set_xlabel('代数')
        ax_ranking.set_ylabel('排名竞争奖励（零和）')
        ax_ranking.set_title('最优解排名竞争奖励变化')
        ax_ranking.grid(True, alpha=0.3)
        
        # 省份博弈奖励
        ax_game = axes2[1]
        if self.history['best_game']:
            ax_game.plot(self.history['best_game'], 'blue', linewidth=2)
        ax_game.set_xlabel('代数')
        ax_game.set_ylabel('省份博弈奖励（中央拨款）')
        ax_game.set_title('最优解省份博弈奖励变化')
        ax_game.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"competition_curves_{timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📊 进化曲线已保存")


def analyze_best_strategy(env: RSMEmissionEnv, chromosome: GAChromosome, save_dir: str = "./ga_results2"):
    """详细分析最优策略"""
    print(f"\n{'=' * 80}")
    print(f"📊 最优策略详细分析")
    print(f"{'=' * 80}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 重置环境并执行最优策略
    env.reset()

    step_results = []
    province_trajectories = {prov: [] for prov in env.province_names}

    for step in range(env.max_steps):
        actions = chromosome.get_actions_for_step(step)
        _, reward, done, info = env.step(actions)

        predicted_pm25 = info.get('predicted_pm25', [])
        reward_components = info.get('reward_components', {})
        province_rewards = info.get('province_rewards', {})

        # ✅ 提取竞争奖励
        coordination_components = reward_components.get('coordination_components', {})
        ranking_reward = 0.0
        game_reward = 0.0
        if coordination_components and isinstance(coordination_components, dict):
            ranking_rewards = coordination_components.get('ranking_rewards', None)
            if ranking_rewards is not None:
                if isinstance(ranking_rewards, np.ndarray):
                    ranking_reward = float(np.sum(ranking_rewards))
                elif isinstance(ranking_rewards, dict):
                    ranking_reward = sum(ranking_rewards.values())
            
            game_rewards = coordination_components.get('province_game_rewards', None)
            if game_rewards is not None:
                if isinstance(game_rewards, np.ndarray):
                    game_reward = float(np.sum(game_rewards))
                elif isinstance(game_rewards, dict):
                    game_reward = sum(game_rewards.values())
        
        step_result = {
            'step': step + 1,
            'reward': reward,
            'avg_pm25': np.mean(predicted_pm25) if len(predicted_pm25) > 0 else 0,
            'avg_reduction': (1 - np.mean(env.cumulative_factors)) * 100,
            'target_reward': reward_components.get('total_target_reward', 0),
            'cost_penalty': reward_components.get('total_cost_penalty', 0),
            'health_reward': reward_components.get('total_health_reward', 0),  # 累积健康效益
            'coordination_reward': reward_components.get('coordination_reward', 0),
            'fairness_reward': reward_components.get('total_fairness_reward', 0),
            'ranking_reward': ranking_reward,  # 排名竞争奖励
            'game_reward': game_reward  # 省份博弈奖励（中央拨款）
        }
        step_results.append(step_result)

        # 记录各省份轨迹
        for i, prov in enumerate(env.province_names):
            if i < len(predicted_pm25):
                province_trajectories[prov].append({
                    'step': step + 1,
                    'pm25': predicted_pm25[i],
                    'reduction_rate': (1 - np.mean(env.cumulative_factors[i])) * 100,
                    'reward': province_rewards.get(prov, {}).get('total', 0)
                })

        print(f"\n步骤 {step + 1}:")
        print(f"  平均PM2.5: {step_result['avg_pm25']:.2f} μg/m³")
        print(f"  平均减排率: {step_result['avg_reduction']:.1f}%")
        print(f"  步骤奖励: {reward:.2f}")

        if done:
            break

    # 保存步骤结果
    step_df = pd.DataFrame(step_results)
    step_df.to_csv(os.path.join(save_dir, f"step_analysis_{timestamp}.csv"), index=False)

    # 保存省份轨迹
    for prov, trajectory in province_trajectories.items():
        if trajectory:
            traj_df = pd.DataFrame(trajectory)
            traj_df.to_csv(os.path.join(save_dir, f"province_{prov}_trajectory_{timestamp}.csv"), index=False)

    # 绘制省份PM2.5变化图
    plt.figure(figsize=(14, 8))

    for prov in env.province_names[:10]:  # 只绘制前10个省份
        if province_trajectories[prov]:
            steps = [t['step'] for t in province_trajectories[prov]]
            pm25_values = [t['pm25'] for t in province_trajectories[prov]]
            plt.plot(steps, pm25_values, '-o', label=prov, linewidth=2, markersize=4)

    plt.xlabel('步骤')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.title('各省份PM2.5浓度变化轨迹（遗传算法最优策略）')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"province_pm25_trajectory_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制减排率热力图
    reduction_matrix = np.zeros((len(env.province_names), env.max_steps))
    for i, prov in enumerate(env.province_names):
        if province_trajectories[prov]:
            for j, t in enumerate(province_trajectories[prov]):
                reduction_matrix[i, j] = t['reduction_rate']

    plt.figure(figsize=(12, 16))
    plt.imshow(reduction_matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='减排率 (%)')
    plt.xlabel('步骤')
    plt.ylabel('省份')
    plt.yticks(range(len(env.province_names)), env.province_names)
    plt.xticks(range(env.max_steps), [f'Step {i + 1}' for i in range(env.max_steps)])
    plt.title('各省份各步骤减排率热力图（遗传算法最优策略）')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reduction_heatmap_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 详细分析结果已保存到 {save_dir}")

    return step_results, province_trajectories


def run_ga_optimization(run_config=None):
    """Run the genetic algorithm optimization baseline."""
    print("Starting genetic algorithm optimization")
    print("=" * 80)

    run_config = copy.deepcopy(run_config or GA2_CONFIG)
    env_config = copy.deepcopy(run_config["env_config"])
    required_files = get_required_files(env_config)

    for file_path, description in required_files:
        if not os.path.exists(file_path):
            print(f"Missing required input: {description}: {file_path}")
            return None

    try:
        print("\nInitializing the environment...")
        env = RSMEmissionEnv(**env_config)
        env_config["max_steps"] = env.max_steps

        num_cores = cpu_count()
        print("\nSystem summary:")
        print(f"  CPU cores: {num_cores}")
        print(f"  Recommended worker count: {num_cores}")

        ga_optimizer = GAOptimizer(
            env=env,
            population_size=run_config["population_size"],
            generations=run_config["generations"],
            crossover_rate=run_config["crossover_rate"],
            mutation_rate=run_config["mutation_rate"],
            elite_ratio=run_config["elite_ratio"],
            tournament_size=run_config["tournament_size"],
            n_jobs=run_config["n_jobs"],
            env_config=env_config,
        )

        best_chromosome = ga_optimizer.run()
        save_dir = run_config["save_dir"]
        ga_optimizer.save_results(save_dir)

        if best_chromosome:
            analyze_best_strategy(env, best_chromosome, save_dir)

        return ga_optimizer, best_chromosome

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Genetic algorithm optimization")
    print("=" * 80)
    print("Goal: search for an 8-step emission-reduction policy across 31 provinces")
    print("Objectives: PM2.5 attainment + cost + health + coordination + fairness")
    print("=" * 80)

    np.random.seed(42)
    random.seed(42)

    result = run_ga_optimization(GA2_CONFIG)

    if result:
        ga_optimizer, best_chromosome = result
        print("\nGenetic algorithm optimization finished")
        print("Results are saved in ./ga_results2/")
        print("Generated files include the best strategy, fitness breakdowns, and evolution logs")
    else:
        print("\nGenetic algorithm optimization failed")
