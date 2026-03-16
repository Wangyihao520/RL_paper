#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepRSM training script for the end-to-end U-Net surrogate model.

The model follows the general idea of predicting air-quality responses to
emission changes by learning a direct mapping from emission inputs and
background chemistry indicators to gridded PM2.5 responses.

Key design choices:
1. Include the baseline scenario (0% reduction) and the clean scenario
   (100% reduction) in the training set.
2. Repeat these boundary scenarios as high-value samples during training.
3. Learn the full emission-to-concentration response surface more robustly.
4. Use ratio-based targets for better generalization across scenarios.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D, \
    Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import argparse
import joblib
import pickle
import json
from datetime import datetime
import warnings
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# Matplotlib display settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Reduce TensorFlow logging noise
tf.get_logger().setLevel('ERROR')

# GPU configuration
print("=== GPU configuration check ===")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid reserving the full GPU upfront.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Detected {len(gpus)} GPU device(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        # Use the first visible GPU by default.
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("GPU training is enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected. Training will run on CPU.")

# Enable mixed precision when supported.
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision is enabled")
except:
    print("Mixed precision is unavailable. Using default precision.")

print("=" * 50)


class DeepRSMTrainer:
    """Trainer for the end-to-end DeepRSM U-Net surrogate model."""

    def __init__(self, data_path='./', output_path='./models/', pollutant='PM25_TOT'):
        """Initialize the trainer."""
        self.data_path = data_path
        self.output_path = output_path
        self.model_save_path = output_path  # 添加model_save_path属性
        self.pollutant = pollutant

        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)

        # 前体物（5种）
        self.precursors = ['NOx', 'SO2', 'NH3', 'VOC', 'PM25']
        # 行业（5个）
        self.sectors = ['AG', 'AR', 'IN', 'PP', 'TR']  # 假设这是你的5个行业缩写

        # 化学指标列表（18种）
        self.chemical_indicators = [
            'HNO3', 'N2O5', 'NO2', 'HONO', 'NO', 'H2O2', 'O3', 'OH',
            'FORM', 'ISOP', 'TERP', 'SO2', 'NH3',
            'PM25_SO4', 'PM25_NO3', 'PM25_NH4', 'PM25_OC', 'PM25_TOT'
        ]

        # 网格布局参数
        self.grid_cols = 144
        self.grid_rows = 120
        self.padded_grid_rows = 128  # 120 向上填充到 2的4次方倍数 (16的倍数)
        self.padded_grid_cols = 144  # 144 已经是 16 的倍数

        # 填充的行数 (用于在数据预处理时进行填充)
        self.pad_top = (self.padded_grid_rows - self.grid_rows) // 2
        self.pad_bottom = self.padded_grid_rows - self.grid_rows - self.pad_top
        # 列不需要填充，所以pad_left/right为0
        self.pad_left = 0
        self.pad_right = 0

        # 排放物种映射 (用于从通用前体物名映射到文件列名)
        self.emission_species_map = {
            'NOx': 'no2', 'SO2': 'so2', 'NH3': 'nh3',
            'VOC': 'voc', 'PM25': 'pm25'
        }

        # 计算输入和输出通道数
        self.num_emission_channels = len(self.precursors) * len(self.sectors)  # 5 * 5 = 25
        self.num_chemical_channels = len(self.chemical_indicators) * 2  # 18 * 2 = 36 (base + clean)
        self.total_input_channels = self.num_emission_channels + self.num_chemical_channels  # 25 + 36 = 61
        self.output_channels = 1  # 预测PM25_TOT浓度

        print(f"DeepRSM训练器初始化完成")
        print(f"目标污染物: {self.pollutant}")
        print(
            f"网格尺寸: {self.grid_rows}x{self.grid_cols} (原始), {self.padded_grid_rows}x{self.padded_grid_cols} (填充后)")
        print(f"填充量: 上={self.pad_top}, 下={self.pad_bottom}, 左={self.pad_left}, 右={self.pad_right}")
        print(f"排放输入通道数 (行业*物种): {self.num_emission_channels}")
        print(f"化学指标输入通道数 (基准+清洁): {self.num_chemical_channels}")
        print(f"总输入通道数: {self.total_input_channels}")
        print(f"输出通道数: {self.output_channels}")

    def load_training_data(self, num_scenarios_to_load=150, normal_repeat=3, extreme_repeat=5):
        """加载训练数据 - 为U-Net准备数据，支持可配置的重复策略，使用比例值数据"""
        print(f"加载训练数据（加载前 {num_scenarios_to_load} 个情景）...")
        print(f"数据处理策略: 排放和浓度使用比例值，化学指标保持原始值")
        print(f"数据增强策略: 一般情景重复{normal_repeat}次，极端情景重复{extreme_repeat}次")
        print(f"特殊处理: 基准情景(0%减排)和清洁情景(100%减排)作为极端重要的边界情景重复{extreme_repeat}次")
        print(f"特殊处理: 情景0(基准)和情景149(极端)将获得额外的重复次数以改善模型性能")

        # 加载省份与网格映射关系
        self.prov_grid = pd.read_csv(os.path.join(self.data_path, 'prov_grid_map/36kmprov.csv'))
        self.fid_to_rc_map = {
            row['FID']: (row['FID'] // self.grid_cols, row['FID'] % self.grid_cols)
            for _, row in self.prov_grid.iterrows()
        }
        # 优化：提前创建FID到省份名称的映射 Series，用于向量化查找
        self.fid_to_province_name = self.prov_grid.set_index('FID')['name'].reindex(
            range(self.grid_rows * self.grid_cols), fill_value='UNKNOWN_PROVINCE'
        )
        print(f"省份网格映射: {len(self.prov_grid)} 个网格")

        # 加载基准和清洁情景浓度数据
        base_conc_df = pd.read_csv(os.path.join(self.data_path, 'conc/base/base.csv'))
        try:
            clean_conc_df = pd.read_csv(os.path.join(self.data_path, 'conc/clean/clean.csv'))
        except FileNotFoundError:
            print("未找到清洁情景数据，使用基准数据的10%作为替代")
            clean_conc_df = base_conc_df.copy()
            for col in self.chemical_indicators:
                if col in clean_conc_df.columns:
                    clean_conc_df[col] = clean_conc_df[col] * 0.1

        print(f"基准情景数据: {len(base_conc_df)} 个网格")
        print(f"清洁情景数据: {len(clean_conc_df)} 个网格")

        # 转换为空间网格
        self.base_conc_grid = self._df_to_spatial_grid(base_conc_df, [self.pollutant])
        self.base_chemical_grid = self._df_to_spatial_grid(base_conc_df, self.chemical_indicators)
        self.clean_chemical_grid = self._df_to_spatial_grid(clean_conc_df, self.chemical_indicators)

        # 转换清洁情景的浓度网格
        self.clean_conc_grid = self._df_to_spatial_grid(clean_conc_df, [self.pollutant])

        # 预加载基准排放数据
        self.base_emissions_dfs = {}
        for sector in self.sectors:
            emission_file_path = os.path.join(self.data_path, 'input_emi/base', f'{sector}.csv')
            if os.path.exists(emission_file_path):
                df = pd.read_csv(emission_file_path)
                if 'FID' in df.columns:
                    df = df.set_index('FID')
                self.base_emissions_dfs[sector] = df

        print(f"已加载 {len(self.base_emissions_dfs)} 个行业的基准排放数据")

        # 加载情景数据并构建训练数据
        X_data_list = []
        Y_data_list = []
        scenario_id_list = []
        processed_scenario_count = 0

        # === 首先添加基准情景（0%减排）===
        print("=== 添加基准情景（0%减排）作为训练数据 ===")
        
        # 1. 构建基准情景的排放比例输入通道（全为1.0，表示无减排）
        base_emission_channels = np.ones(
            (self.grid_rows, self.grid_cols, self.num_emission_channels), dtype=np.float32
        )
        
        # 2. 合并排放比例和化学指标作为模型输入X
        X_base_scenario = np.concatenate([
            base_emission_channels,
            self.base_chemical_grid,
            self.clean_chemical_grid
        ], axis=-1)
        
        # 3. 基准情景的浓度比例为1.0（基准浓度 / 基准浓度 = 1.0）
        Y_base_scenario = np.ones((self.grid_rows, self.grid_cols, 1), dtype=np.float32)
        
        # 4. 添加基准情景的训练样本（重复extreme_repeat次）
        for _ in range(extreme_repeat):
            X_data_list.append(X_base_scenario)
            Y_data_list.append(Y_base_scenario)
            scenario_id_list.append(-1)  # 使用-1标识基准情景
            
        print(f"基准情景（0%减排）已添加 {extreme_repeat} 次")
        
        # === 然后添加清洁情景（100%减排）===
        print("=== 添加清洁情景（100%减排）作为训练数据 ===")
        
        # 1. 构建清洁情景的排放比例输入通道（全为0.0，表示完全减排）
        clean_emission_channels = np.zeros(
            (self.grid_rows, self.grid_cols, self.num_emission_channels), dtype=np.float32
        )
        
        # 2. 合并排放比例和化学指标作为模型输入X
        X_clean_scenario = np.concatenate([
            clean_emission_channels,
            self.base_chemical_grid,
            self.clean_chemical_grid
        ], axis=-1)
        
        # 3. 清洁情景的浓度比例（清洁浓度 / 基准浓度）
        base_conc_values = self.base_conc_grid[:, :, 0]
        clean_conc_values = self.clean_conc_grid[:, :, 0]
        
        # 避免除零错误，确保输出数据类型为float32
        clean_conc_ratio = np.divide(clean_conc_values, base_conc_values,
                                   out=np.ones_like(clean_conc_values, dtype=np.float32),
                                   where=base_conc_values!=0)
        
        # 对于基准浓度为0的网格，如果清洁浓度也为0，比例设为1；否则设为一个大值
        zero_base_mask = (base_conc_values == 0)
        clean_conc_ratio[zero_base_mask & (clean_conc_values == 0)] = 1.0
        clean_conc_ratio[zero_base_mask & (clean_conc_values > 0)] = 10.0
        
        Y_clean_scenario = clean_conc_ratio.reshape(self.grid_rows, self.grid_cols, 1)
        
        # 4. 添加清洁情景的训练样本（重复extreme_repeat次）
        for _ in range(extreme_repeat):
            X_data_list.append(X_clean_scenario)
            Y_data_list.append(Y_clean_scenario)
            scenario_id_list.append(-2)  # 使用-2标识清洁情景
            
        print(f"清洁情景（100%减排）已添加 {extreme_repeat} 次")
        
        # 打印基准和清洁情景的统计信息
        print(f"基准情景浓度比例统计: 最小值={np.min(Y_base_scenario):.4f}, 最大值={np.max(Y_base_scenario):.4f}, 平均值={np.mean(Y_base_scenario):.4f}")
        print(f"清洁情景浓度比例统计: 最小值={np.min(Y_clean_scenario):.4f}, 最大值={np.max(Y_clean_scenario):.4f}, 平均值={np.mean(Y_clean_scenario):.4f}")

        # === 然后加载常规情景数据 ===
        print("=== 加载常规情景数据 ===")

        for scenario_id in range(num_scenarios_to_load):
            print(f"处理情景 {scenario_id}...")

            # 加载减排系数
            factor_file_name = f'scenario_{scenario_id:03d}_provincial_reduction.csv'
            factor_path = os.path.join(self.data_path, 'factor/prov', factor_file_name)

            if not os.path.exists(factor_path):
                print(f"  警告: 未找到情景 {scenario_id} 的减排系数文件 ({factor_path})，跳过该情景。")
                continue

            reduction_factors_df = pd.read_csv(factor_path)
            # 创建省份到各前体物减排系数的映射
            prov_red_map_precursor_wise = {
                prec: reduction_factors_df.set_index('PROV_NAME')[prec].to_dict()
                for prec in self.precursors if prec in reduction_factors_df.columns
            }

            # --- 1. 构建排放比例输入通道 ---
            current_emission_channels = np.zeros(
                (self.grid_rows, self.grid_cols, self.num_emission_channels), dtype=np.float32
            )
            channel_idx = 0

            for sector_name in self.sectors:
                if sector_name not in self.base_emissions_dfs:
                    channel_idx += len(self.precursors)
                    continue

                base_sector_df = self.base_emissions_dfs[sector_name]

                for precursor_name in self.precursors:
                    emission_col_name = self.emission_species_map.get(precursor_name)
                    if emission_col_name is None or emission_col_name not in base_sector_df.columns:
                        channel_idx += 1
                        continue

                    base_emission_flat = base_sector_df[emission_col_name].reindex(
                        range(self.grid_rows * self.grid_cols), fill_value=0
                    ).values.astype(np.float32)

                    # 应用减排系数
                    reduction_factor_flat = self.fid_to_province_name.map(
                        prov_red_map_precursor_wise.get(precursor_name, {})
                    ).fillna(0.0).values.astype(np.float32)

                    # 特殊处理情景0：确保其作为基准情景的特殊性
                    if scenario_id == 0:
                        scenario_emission_flat = base_emission_flat
                    else:
                        scenario_emission_flat = base_emission_flat * (1 - reduction_factor_flat)

                    scenario_emission_flat[scenario_emission_flat < 0] = 0

                    # 计算排放比例：情景排放 / 基准排放
                    base_emission_grid = base_emission_flat.reshape(self.grid_rows, self.grid_cols)
                    scenario_emission_grid = scenario_emission_flat.reshape(self.grid_rows, self.grid_cols)
                    
                    # 避免除零错误，确保输出数据类型为float32
                    emission_ratio = np.divide(scenario_emission_grid, base_emission_grid, 
                                             out=np.ones_like(scenario_emission_grid, dtype=np.float32), 
                                             where=base_emission_grid!=0)
                    
                    # 对于基准排放为0的网格，如果情景排放也为0，比例设为1；否则设为一个大值
                    zero_base_mask = (base_emission_grid == 0)
                    emission_ratio[zero_base_mask & (scenario_emission_grid == 0)] = 1.0
                    emission_ratio[zero_base_mask & (scenario_emission_grid > 0)] = 10.0  # 设置一个合理的上限

                    current_emission_channels[:, :, channel_idx] = emission_ratio
                    channel_idx += 1

            # --- 2. 合并排放比例和化学指标作为模型输入X ---
            X_current_scenario = np.concatenate([
                current_emission_channels,
                self.base_chemical_grid,
                self.clean_chemical_grid
            ], axis=-1)

            # --- 3. 构建PM2.5浓度比例作为模型输出Y ---
            scenario_conc_file_name = f'scenario_{scenario_id:03d}_monthly_136_855_all_species.csv'
            scenario_conc_path = os.path.join(self.data_path, 'conc', scenario_conc_file_name)

            if not os.path.exists(scenario_conc_path):
                print(f"  警告: 未找到情景 {scenario_id} 的浓度数据 ({scenario_conc_path})，跳过该情景。")
                continue

            scenario_conc_df = pd.read_csv(scenario_conc_path)
            scenario_conc_grid = self._df_to_spatial_grid(scenario_conc_df, [self.pollutant])
            
            # 计算浓度比例：情景浓度 / 基准浓度
            base_conc_values = self.base_conc_grid[:, :, 0]
            scenario_conc_values = scenario_conc_grid[:, :, 0]
            
            # 避免除零错误，确保输出数据类型为float32
            conc_ratio = np.divide(scenario_conc_values, base_conc_values,
                                 out=np.ones_like(scenario_conc_values, dtype=np.float32),
                                 where=base_conc_values!=0)
            
            # 对于基准浓度为0的网格，如果情景浓度也为0，比例设为1；否则设为一个大值
            zero_base_mask = (base_conc_values == 0)
            conc_ratio[zero_base_mask & (scenario_conc_values == 0)] = 1.0
            conc_ratio[zero_base_mask & (scenario_conc_values > 0)] = 10.0  # 设置一个合理的上限
            
            Y_current_scenario = conc_ratio.reshape(self.grid_rows, self.grid_cols, 1)

            # --- 4. 智能数据增强策略 ---
            # 根据情景特征确定重复次数
            if scenario_id == 0:
                # 情景0是基准情景，非常重要，增加重复次数
                repeat_times = max(normal_repeat, extreme_repeat) + 2  # 额外增加2次
                print(f"  情景0(基准情景)重复{repeat_times}次")
            elif scenario_id == 149:
                # 情景149是极端情景，也很重要，增加重复次数
                repeat_times = max(normal_repeat, extreme_repeat) + 1  # 额外增加1次
                print(f"  情景149(极端情景)重复{repeat_times}次")
            elif scenario_id < 100:
                # 一般情景
                repeat_times = normal_repeat
            else:
                # 其他极端情景
                repeat_times = extreme_repeat

            # 添加重复的训练样本
            for _ in range(repeat_times):
                X_data_list.append(X_current_scenario)
                Y_data_list.append(Y_current_scenario)
                scenario_id_list.append(scenario_id)

            if processed_scenario_count % 10 == 0:  # 每10个情景打印一次进度
                print(f"  ✓ 情景 {scenario_id} 数据处理完成。")
            processed_scenario_count += 1

        # 转换为numpy数组
        self.X_data = np.array(X_data_list, dtype=np.float32)
        self.Y_data = np.array(Y_data_list, dtype=np.float32)
        self.scenario_ids = np.array(scenario_id_list)

        print(f"=== 数据加载完成 ===")
        print(f"训练数据形状: X={self.X_data.shape}, Y={self.Y_data.shape}")
        print(f"处理的常规情景数量: {processed_scenario_count}")
        print(f"基准情景（ID=-1）样本数: {np.sum(self.scenario_ids == -1)}")
        print(f"清洁情景（ID=-2）样本数: {np.sum(self.scenario_ids == -2)}")
        print(f"总训练样本数: {len(self.X_data)}")
        
        # 统计重复后的情景分布
        unique_scenarios, counts = np.unique(self.scenario_ids, return_counts=True)
        print(f"重复后的情景分布:")
        for scenario_id, count in zip(unique_scenarios, counts):
            if scenario_id == -1:
                print(f"  基准情景(0%减排): {count}次")
            elif scenario_id == -2:
                print(f"  清洁情景(100%减排): {count}次")
            elif scenario_id in [0, 149] or count != normal_repeat:
                print(f"  情景{scenario_id}: {count}次")
        
        # 数据质量分析
        self._analyze_data_quality()

        return True

    def shuffle_data_before_training(self):
        """在训练前打乱数据，确保每次训练的数据顺序不同"""
        print("在训练前打乱数据...")
        indices = np.random.permutation(self.X_data.shape[0])
        self.X_data = self.X_data[indices]
        self.Y_data = self.Y_data[indices]
        
        # 如果有样本类型信息，也要相应地打乱
        if hasattr(self, 'sample_types'):
            self.sample_types = self.sample_types[indices]
        
        print(f"数据打乱完成，总样本数: {len(self.X_data)}")
        print(f"打乱后X数据形状: {self.X_data.shape}")
        print(f"打乱后Y数据形状: {self.Y_data.shape}")
        
        # 验证数据质量
        print(f"打乱后X数据范围: [{np.min(self.X_data):.4f}, {np.max(self.X_data):.4f}]")
        print(f"打乱后Y数据范围: [{np.min(self.Y_data):.4f}, {np.max(self.Y_data):.4f}]")

    def _analyze_data_quality(self):
        """分析训练数据质量，重点关注边界情景的特殊性"""
        print("\n=== 训练数据质量分析 ===")
        
        # 基本统计信息
        print(f"数据形状: X={self.X_data.shape}, Y={self.Y_data.shape}")
        print(f"总样本数: {len(self.X_data)}")
        
        # 排放比例分析
        emission_data = self.X_data[:, :, :, :self.num_emission_channels]
        print(f"排放比例统计: 最小值={np.min(emission_data):.4f}, 最大值={np.max(emission_data):.4f}, 平均值={np.mean(emission_data):.4f}")
        
        # 浓度比例分析
        conc_data = self.Y_data[:, :, :, 0]
        print(f"浓度比例统计: 最小值={np.min(conc_data):.4f}, 最大值={np.max(conc_data):.4f}, 平均值={np.mean(conc_data):.4f}")
        
        # 情景分布分析
        unique_scenarios, counts = np.unique(self.scenario_ids, return_counts=True)
        print(f"\n情景分布:")
        print(f"  总情景数: {len(unique_scenarios)}")
        print(f"  样本数最多的情景: {unique_scenarios[np.argmax(counts)]} ({np.max(counts)}次)")
        print(f"  样本数最少的情景: {unique_scenarios[np.argmin(counts)]} ({np.min(counts)}次)")
        
        # 重点分析边界情景
        print(f"\n=== 边界情景特殊性分析 ===")
        
        # 分析基准情景（ID=-1）
        if -1 in unique_scenarios:
            base_scenario_mask = self.scenario_ids == -1
            base_scenario_emission = emission_data[base_scenario_mask]
            base_scenario_conc = conc_data[base_scenario_mask]
            
            print(f"基准情景（ID=-1, 0%减排）:")
            print(f"  样本数: {np.sum(base_scenario_mask)}")
            print(f"  排放比例: 最小={np.min(base_scenario_emission):.4f}, 最大={np.max(base_scenario_emission):.4f}, 平均={np.mean(base_scenario_emission):.4f}")
            print(f"  浓度比例: 最小={np.min(base_scenario_conc):.4f}, 最大={np.max(base_scenario_conc):.4f}, 平均={np.mean(base_scenario_conc):.4f}")
            
            # 检查基准情景是否正确（排放比例应该为1.0，浓度比例应该为1.0）
            if np.allclose(base_scenario_emission, 1.0, rtol=1e-5):
                print(f"  ✓ 基准情景排放比例正确（全为1.0）")
            else:
                print(f"  ⚠ 基准情景排放比例异常，应该全为1.0")
            
            if np.allclose(base_scenario_conc, 1.0, rtol=1e-5):
                print(f"  ✓ 基准情景浓度比例正确（全为1.0）")
            else:
                print(f"  ⚠ 基准情景浓度比例异常，应该全为1.0")
        
        # 分析清洁情景（ID=-2）
        if -2 in unique_scenarios:
            clean_scenario_mask = self.scenario_ids == -2
            clean_scenario_emission = emission_data[clean_scenario_mask]
            clean_scenario_conc = conc_data[clean_scenario_mask]
            
            print(f"清洁情景（ID=-2, 100%减排）:")
            print(f"  样本数: {np.sum(clean_scenario_mask)}")
            print(f"  排放比例: 最小={np.min(clean_scenario_emission):.4f}, 最大={np.max(clean_scenario_emission):.4f}, 平均={np.mean(clean_scenario_emission):.4f}")
            print(f"  浓度比例: 最小={np.min(clean_scenario_conc):.4f}, 最大={np.max(clean_scenario_conc):.4f}, 平均={np.mean(clean_scenario_conc):.4f}")
            
            # 检查清洁情景是否正确（排放比例应该为0.0）
            if np.allclose(clean_scenario_emission, 0.0, rtol=1e-5):
                print(f"  ✓ 清洁情景排放比例正确（全为0.0）")
            else:
                print(f"  ⚠ 清洁情景排放比例异常，应该全为0.0")
            
            # 清洁情景的浓度比例应该小于1.0（通常在0.1-0.8之间）
            if np.mean(clean_scenario_conc) < 1.0:
                print(f"  ✓ 清洁情景浓度比例合理（小于1.0，平均={np.mean(clean_scenario_conc):.3f}）")
            else:
                print(f"  ⚠ 清洁情景浓度比例异常，应该小于1.0")
        
        # 分析情景0
        if 0 in unique_scenarios:
            scenario_0_mask = self.scenario_ids == 0
            scenario_0_emission = emission_data[scenario_0_mask]
            scenario_0_conc = conc_data[scenario_0_mask]
            
            print(f"情景0 (基准情景):")
            print(f"  样本数: {np.sum(scenario_0_mask)}")
            print(f"  排放比例: 最小={np.min(scenario_0_emission):.4f}, 最大={np.max(scenario_0_emission):.4f}, 平均={np.mean(scenario_0_emission):.4f}")
            print(f"  浓度比例: 最小={np.min(scenario_0_conc):.4f}, 最大={np.max(scenario_0_conc):.4f}, 平均={np.mean(scenario_0_conc):.4f}")
            
            # 检查情景0是否真的是基准情景（排放比例应该接近1.0）
            if np.mean(scenario_0_emission) > 0.9:
                print(f"  ✓ 情景0确实是基准情景（排放比例接近1.0）")
            else:
                print(f"  ⚠ 情景0的排放比例偏低，可能不是真正的基准情景")
        
        # 分析情景149
        if 149 in unique_scenarios:
            scenario_149_mask = self.scenario_ids == 149
            scenario_149_emission = emission_data[scenario_149_mask]
            scenario_149_conc = conc_data[scenario_149_mask]
            
            print(f"情景149 (极端情景):")
            print(f"  样本数: {np.sum(scenario_149_mask)}")
            print(f"  排放比例: 最小={np.min(scenario_149_emission):.4f}, 最大={np.max(scenario_149_emission):.4f}, 平均={np.mean(scenario_149_emission):.4f}")
            print(f"  浓度比例: 最小={np.min(scenario_149_conc):.4f}, 最大={np.max(scenario_149_conc):.4f}, 平均={np.mean(scenario_149_conc):.4f}")
            
            # 检查情景149是否是极端减排情景（排放比例应该很低）
            if np.mean(scenario_149_emission) < 0.3:
                print(f"  ✓ 情景149确实是极端减排情景（排放比例很低）")
            else:
                print(f"  ⚠ 情景149的排放比例偏高，可能不是极端减排情景")
        
        # 分析中等情景作为对比
        middle_scenarios = [20, 40, 60, 80, 100, 120]
        middle_performance = []
        
        print(f"\n=== 中等情景对比分析 ===")
        for scenario_id in middle_scenarios:
            if scenario_id in unique_scenarios:
                scenario_mask = self.scenario_ids == scenario_id
                scenario_emission = emission_data[scenario_mask]
                scenario_conc = conc_data[scenario_mask]
                
                avg_emission = np.mean(scenario_emission)
                avg_conc = np.mean(scenario_conc)
                middle_performance.append((scenario_id, avg_emission, avg_conc))
                
                print(f"  情景{scenario_id}: 排放比例={avg_emission:.3f}, 浓度比例={avg_conc:.3f}")
        
        # 数据异常检测
        print(f"\n=== 数据异常检测 ===")
        
        # 检查负值
        negative_emission = np.sum(emission_data < 0)
        negative_conc = np.sum(conc_data < 0)
        print(f"负值检测: 排放比例负值={negative_emission}个, 浓度比例负值={negative_conc}个")
        
        # 检查极端值
        extreme_emission = np.sum(emission_data > 5.0)
        extreme_conc = np.sum(conc_data > 5.0)
        print(f"极端值检测: 排放比例>5.0有{extreme_emission}个, 浓度比例>5.0有{extreme_conc}个")
        
        # 检查NaN值
        nan_emission = np.sum(np.isnan(emission_data))
        nan_conc = np.sum(np.isnan(conc_data))
        print(f"NaN值检测: 排放比例NaN={nan_emission}个, 浓度比例NaN={nan_conc}个")
        
        # 数据分布建议
        print(f"\n=== 数据分布优化建议 ===")
        
        # 检查边界情景的样本数
        base_count = np.sum(self.scenario_ids == -1) if -1 in unique_scenarios else 0
        clean_count = np.sum(self.scenario_ids == -2) if -2 in unique_scenarios else 0
        scenario_0_count = np.sum(self.scenario_ids == 0) if 0 in unique_scenarios else 0
        scenario_149_count = np.sum(self.scenario_ids == 149) if 149 in unique_scenarios else 0
        middle_count = np.sum([np.sum(self.scenario_ids == s) for s in middle_scenarios if s in unique_scenarios])
        
        print(f"当前边界情景分布:")
        print(f"  基准情景(ID=-1, 0%减排): {base_count} 样本")
        print(f"  清洁情景(ID=-2, 100%减排): {clean_count} 样本")
        print(f"  情景0 (基准): {scenario_0_count} 样本")
        print(f"  情景149 (极端): {scenario_149_count} 样本")
        print(f"  中等情景: {middle_count} 样本")
        
        # 建议
        total_boundary_samples = base_count + clean_count + scenario_0_count + scenario_149_count
        if total_boundary_samples > 0:
            boundary_ratio = total_boundary_samples / len(self.X_data)
            print(f"边界情景占比: {boundary_ratio:.2%}")
            if boundary_ratio < 0.15:
                print(f"  建议: 边界情景占比偏低，建议增加到15%以上")
            elif boundary_ratio > 0.4:
                print(f"  建议: 边界情景占比偏高，可能影响中等情景的学习")
            else:
                print(f"  ✓ 边界情景占比合理")
        
        print(f"=== 数据质量分析完成 ===\n")

    def _df_to_spatial_grid(self, data_df, target_columns):
        """
        将DataFrame的特定列转换为空间网格 (H, W, C)。
        data_df: 包含'FID'和目标列的DataFrame。
        target_columns: 一个列表，指定要转换为通道的列名。
        """
        num_channels = len(target_columns)
        spatial_grid = np.zeros((self.grid_rows, self.grid_cols, num_channels), dtype=np.float32)

        full_df = data_df.set_index('FID').reindex(range(self.grid_rows * self.grid_cols), fill_value=0)

        for i, col_name in enumerate(target_columns):
            if col_name in full_df.columns:
                flat_data = full_df[col_name].values
                spatial_grid[:, :, i] = flat_data.reshape(self.grid_rows, self.grid_cols)
            else:
                print(f"  警告: DataFrame中未找到列 '{col_name}'，该通道填充为0。")

        return spatial_grid

    def _standardize_data(self, X_data, Y_data):
        """
        数据标准化函数
        - 排放比例和浓度比例：不进行标准化，保持原始物理意义
        - 化学指标：进行对数标准化处理
        """
        print("开始数据标准化...")
        
        # 检查数据中的NaN和Inf值
        if np.any(np.isnan(X_data)) or np.any(np.isinf(X_data)):
            print("警告: X_data中存在NaN或Inf值")
        if np.any(np.isnan(Y_data)) or np.any(np.isinf(Y_data)):
            print("警告: Y_data中存在NaN或Inf值")
        
        # 分析数据分布
        print("\n=== 数据分布分析 ===")
        
        # 排放比例数据分析 (前19个特征)
        emission_ratios = X_data[:, :, :, :19]
        print(f"排放比例统计:")
        print(f"  形状: {emission_ratios.shape}")
        print(f"  范围: [{emission_ratios.min():.6f}, {emission_ratios.max():.6f}]")
        print(f"  均值: {emission_ratios.mean():.6f}")
        print(f"  标准差: {emission_ratios.std():.6f}")
        print(f"  中位数: {np.median(emission_ratios):.6f}")
        print(f"  95%分位数: {np.percentile(emission_ratios, 95):.6f}")
        
        # 浓度比例数据分析
        conc_ratios = Y_data
        print(f"\n浓度比例统计:")
        print(f"  形状: {conc_ratios.shape}")
        print(f"  范围: [{conc_ratios.min():.6f}, {conc_ratios.max():.6f}]")
        print(f"  均值: {conc_ratios.mean():.6f}")
        print(f"  标准差: {conc_ratios.std():.6f}")
        print(f"  中位数: {np.median(conc_ratios):.6f}")
        print(f"  95%分位数: {np.percentile(conc_ratios, 95):.6f}")
        
        # 化学指标数据分析 (后面的特征)
        if X_data.shape[3] > 19:
            chemical_indicators = X_data[:, :, :, 19:]
            print(f"\n化学指标统计:")
            print(f"  形状: {chemical_indicators.shape}")
            print(f"  范围: [{chemical_indicators.min():.6f}, {chemical_indicators.max():.6f}]")
            print(f"  均值: {chemical_indicators.mean():.6f}")
            print(f"  标准差: {chemical_indicators.std():.6f}")
        
        print("\n=== 标准化策略 ===")
        print("排放比例: 不进行标准化，保持原始物理意义")
        print("浓度比例: 不进行标准化，保持原始物理意义")
        print("化学指标: 进行对数标准化处理")
        
        # 初始化标准化器
        self.scalers = {}
        
        # 复制数据以避免修改原始数据
        X_standardized = X_data.copy()
        Y_standardized = Y_data.copy()
        
        # 1. 排放比例 (前19个特征) - 不进行标准化
        print(f"\n处理排放比例数据 (特征 0-18)...")
        print(f"  保持原始值，不进行标准化")
        # X_standardized[:, :, :, :19] 保持不变
        
        # 2. 浓度比例 - 不进行标准化
        print(f"\n处理浓度比例数据...")
        print(f"  保持原始值，不进行标准化")
        # Y_standardized 保持不变
        
        # 3. 化学指标 (如果存在) - 进行对数标准化
        if X_data.shape[3] > 19:
            print(f"\n处理化学指标数据 (特征 19+)...")
            
            chemical_data = X_data[:, :, :, 19:]
            n_samples, height, width, n_chemical_features = chemical_data.shape
            
            # 重塑为2D数组进行标准化
            chemical_data_reshaped = chemical_data.reshape(-1, n_chemical_features)
            
            # 对数变换 + 标准化
            chemical_data_log = np.log1p(np.maximum(chemical_data_reshaped, 0))
            
            # 使用StandardScaler进行标准化
            from sklearn.preprocessing import StandardScaler
            chemical_scaler = StandardScaler()
            chemical_data_standardized = chemical_scaler.fit_transform(chemical_data_log)
            
            # 重塑回原始形状
            chemical_data_standardized = chemical_data_standardized.reshape(n_samples, height, width, n_chemical_features)
            
            # 更新X_standardized
            X_standardized[:, :, :, 19:] = chemical_data_standardized
            
            # 保存标准化器
            self.scalers['chemical'] = chemical_scaler
            
            print(f"  化学指标标准化完成")
            print(f"  标准化后范围: [{chemical_data_standardized.min():.6f}, {chemical_data_standardized.max():.6f}]")

        # 最终检查
        print(f"\n=== 标准化结果摘要 ===")
        print(f"X_data 最终范围: [{X_standardized.min():.6f}, {X_standardized.max():.6f}]")
        print(f"Y_data 最终范围: [{Y_standardized.min():.6f}, {Y_standardized.max():.6f}]")
        
        # 检查是否有异常值
        if np.any(np.isnan(X_standardized)) or np.any(np.isinf(X_standardized)):
            print("警告: 标准化后的X_data中存在NaN或Inf值")
        if np.any(np.isnan(Y_standardized)) or np.any(np.isinf(Y_standardized)):
            print("警告: 标准化后的Y_data中存在NaN或Inf值")
        
        print("数据标准化完成！")
        return X_standardized, Y_standardized

    def data_generator(self, batch_size, validation_split=0.2, stratified_split=True):
        """数据生成器 - 为U-Net模型提供批次数据，支持分层抽样"""
        n_samples = self.X_data.shape[0]  # 情景数量

        if stratified_split and n_samples >= 100 and hasattr(self, 'sample_types'):
            # 使用准确的样本类型信息进行分层抽样
            normal_indices = np.where(self.sample_types == 'normal')[0]
            extreme_indices = np.where(self.sample_types == 'extreme')[0]

            print(f"=== 分层抽样数据划分（重复增强后）===")
            print(f"总样本数: {n_samples}")
            print(f"一般情景样本数: {len(normal_indices)}")
            print(f"极端情景样本数: {len(extreme_indices)}")

            # 分别对正常和极端情景样本进行训练/验证划分
            np.random.shuffle(normal_indices)
            np.random.shuffle(extreme_indices)

            n_val_normal = max(1, int(len(normal_indices) * validation_split))
            n_val_extreme = max(1, int(len(extreme_indices) * validation_split))

            # 训练集：包含正常和极端情景样本
            train_normal = normal_indices[n_val_normal:]
            train_extreme = extreme_indices[n_val_extreme:]
            train_indices = np.concatenate([train_normal, train_extreme])

            # 验证集：包含正常和极端情景样本
            val_normal = normal_indices[:n_val_normal]
            val_extreme = extreme_indices[:n_val_extreme]
            val_indices = np.concatenate([val_normal, val_extreme])

            print(
                f"训练集组成: 一般样本{len(train_normal)}个 + 极端样本{len(train_extreme)}个 = {len(train_indices)}个")
            print(f"验证集组成: 一般样本{len(val_normal)}个 + 极端样本{len(val_extreme)}个 = {len(val_indices)}个")
            print(f"训练集一般样本比例: {len(train_normal) / len(train_indices) * 100:.1f}%")
            print(f"验证集一般样本比例: {len(val_normal) / len(val_indices) * 100:.1f}%")

        else:
            # 传统随机划分
            print("使用传统随机划分方式")
            indices = np.random.permutation(n_samples)
            n_val = int(n_samples * validation_split)
            train_indices = indices[n_val:]
            val_indices = indices[:n_val]

        n_train = len(train_indices)
        n_val = len(val_indices)

        def generate_batches(sample_indices, is_training=True):
            while True:
                if is_training:
                    np.random.shuffle(sample_indices)

                for i in range(0, len(sample_indices), batch_size):
                    batch_idx = sample_indices[i:i + batch_size]

                    # 直接从标准化后的数据中取批次
                    batch_X = self.X_data[batch_idx]
                    batch_Y = self.Y_data[batch_idx]
                    
                    # 进行空间填充以适应模型输入要求
                    batch_X_padded = np.pad(batch_X,
                                          ((0, 0), (self.pad_top, self.pad_bottom), 
                                           (self.pad_left, self.pad_right), (0, 0)),
                                          mode='constant', constant_values=0)
                    batch_Y_padded = np.pad(batch_Y,
                                          ((0, 0), (self.pad_top, self.pad_bottom), 
                                           (self.pad_left, self.pad_right), (0, 0)),
                                          mode='constant', constant_values=0)

                    # 减少打印频率，避免输出过多信息
                    if is_training and i == 0:  # 只在每个epoch的第一个batch打印
                        print(f"--- DataGenerator Batch Info (Is Training: {is_training}) ---")
                        print(f"  Batch X Shape (padded): {batch_X_padded.shape}, Batch Y Shape (padded): {batch_Y_padded.shape}")
                        print(f"  Batch samples: {batch_idx}")
                        print(
                            f"  Batch X Min: {np.min(batch_X_padded):.4f}, Max: {np.max(batch_X_padded):.4f}, Mean: {np.mean(batch_X_padded):.4f}")
                        print(
                            f"  Batch Y Min: {np.min(batch_Y_padded):.4f}, Max: {np.max(batch_Y_padded):.4f}, Mean: {np.mean(batch_Y_padded):.4f}")
                        print("---------------------------------------------------")

                    yield batch_X_padded, batch_Y_padded

        return (generate_batches(train_indices, True),
                generate_batches(val_indices, False),
                n_train, n_val)

    def _unet_block(self, input_tensor, filters, name, dropout_rate=0):
        """U-Net中的一个编码/解码块"""
        x = Conv2D(filters, 3, padding='same', name=f'{name}_conv1')(input_tensor)
        x = BatchNormalization(name=f'{name}_bn1')(x)
        x = Activation('relu', name=f'{name}_relu1')(x)

        x = Conv2D(filters, 3, padding='same', name=f'{name}_conv2')(x)
        x = BatchNormalization(name=f'{name}_bn2')(x)
        x = Activation('relu', name=f'{name}_relu2')(x)

        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f'{name}_dropout')(x)
        return x

    def build_unet_model(self):
        """构建U-Net模型架构"""
        print("构建U-Net模型...")

        # 输入形状现在是填充后的尺寸
        input_shape = (self.padded_grid_rows, self.padded_grid_cols, self.total_input_channels)
        inputs = Input(shape=input_shape, name='input_image')

        # --- 编码器 (下采样路径) ---
        conv1 = self._unet_block(inputs, 64, 'conv1')
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)  # 128x144 -> 64x72

        conv2 = self._unet_block(pool1, 128, 'conv2')
        pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)  # 64x72 -> 32x36

        conv3 = self._unet_block(pool2, 256, 'conv3')
        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)  # 32x36 -> 16x18

        conv4 = self._unet_block(pool3, 512, 'conv4', dropout_rate=0.3)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)  # 16x18 -> 8x9

        # --- 瓶颈层 ---
        conv_bottle = self._unet_block(pool4, 1024, 'conv_bottle', dropout_rate=0.3)

        # --- 解码器 (上采样路径) ---
        up6 = UpSampling2D(size=(2, 2), name='up6')(conv_bottle)  # 8x9 -> 16x18
        up6 = Conv2D(512, 2, activation='relu', padding='same', name='conv_up6')(up6)
        merge6 = Concatenate(axis=-1, name='merge6')([conv4, up6])  # 16x18 匹配
        conv6 = self._unet_block(merge6, 512, 'conv6')

        up7 = UpSampling2D(size=(2, 2), name='up7')(conv6)  # 16x18 -> 32x36
        up7 = Conv2D(256, 2, activation='relu', padding='same', name='conv_up7')(up7)
        merge7 = Concatenate(axis=-1, name='merge7')([conv3, up7])  # 32x36 匹配
        conv7 = self._unet_block(merge7, 256, 'conv7')

        up8 = UpSampling2D(size=(2, 2), name='up8')(conv7)  # 32x36 -> 64x72
        up8 = Conv2D(128, 2, activation='relu', padding='same', name='conv_up8')(up8)
        merge8 = Concatenate(axis=-1, name='merge8')([conv2, up8])  # 64x72 匹配
        conv8 = self._unet_block(merge8, 128, 'conv8')

        up9 = UpSampling2D(size=(2, 2), name='up9')(conv8)  # 64x72 -> 128x144
        up9 = Conv2D(64, 2, activation='relu', padding='same', name='conv_up9')(up9)
        merge9 = Concatenate(axis=-1, name='merge9')([conv1, up9])  # 128x144 匹配
        conv9 = self._unet_block(merge9, 64, 'conv9')

        # 最后一层卷积，输出PM2.5浓度 (单通道)
        # 此时输出已经是填充后的尺寸 (128, 144, 1)
        outputs = Conv2D(self.output_channels, 1, activation='linear', name='output_pm25')(conv9)

        # 构建模型
        model = Model(inputs=inputs, outputs=outputs, name='U_Net_DeepRSM')

        print("U-Net模型架构:")
        model.summary()

        return model

    def relative_loss(self, y_true, y_pred):
        """
        自定义相对损失函数，基于论文公式：
        L(ŷ, y) = (1/NHWC) * Σ(|ŷ - y| / y)

        其中：
        - ŷ：DeepRSM预测的污染物浓度
        - y：CTM模拟的污染物浓度
        - N、H、W、C：分别表示样本数量、高度、宽度和通道数
        """
        epsilon = tf.constant(1e-6, dtype=tf.float32)  # 避免除以零

        # 确保数据类型为float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 裁剪到原始网格尺寸 (去除填充)
        cropped_y_true = y_true[:,
                         self.pad_top: self.pad_top + self.grid_rows,
                         self.pad_left: self.pad_left + self.grid_cols,
                         :]
        cropped_y_pred = y_pred[:,
                         self.pad_top: self.pad_top + self.grid_rows,
                         self.pad_left: self.pad_left + self.grid_cols,
                         :]

        # 获取维度信息
        N = tf.cast(tf.shape(cropped_y_true)[0], tf.float32)  # 批次大小
        H = tf.cast(tf.shape(cropped_y_true)[1], tf.float32)  # 高度
        W = tf.cast(tf.shape(cropped_y_true)[2], tf.float32)  # 宽度
        C = tf.cast(tf.shape(cropped_y_true)[3], tf.float32)  # 通道数

        # 计算总元素数量
        total_elements = N * H * W * C

        # 计算绝对误差 |ŷ - y|
        abs_diff = tf.abs(cropped_y_pred - cropped_y_true)

        # 计算分母，使用最大值确保数值稳定性
        # 对于接近零的真实值，使用epsilon作为最小分母
        denominator = tf.maximum(tf.abs(cropped_y_true), epsilon)

        # 计算相对误差 |ŷ - y| / y
        relative_error = abs_diff / denominator

        # 限制相对误差范围，防止极端值影响训练
        relative_error = tf.clip_by_value(relative_error, 0.0, 10.0)

        # 按照公式计算损失：(1/NHWC) * Σ(相对误差)
        total_relative_error = tf.reduce_sum(relative_error)
        final_loss = total_relative_error / total_elements

        # 确保损失值为有限值
        final_loss = tf.where(tf.math.is_finite(final_loss),
                              final_loss,
                              tf.constant(1.0, dtype=tf.float32))

        return final_loss

    def r2_metric(self, y_true, y_pred):
        """R2评估指标 - 在裁剪后的区域计算，增强数值稳定性"""
        # 确保数据类型
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 裁剪到原始网格尺寸
        cropped_y_true = y_true[:,
                         self.pad_top: self.pad_top + self.grid_rows,
                         self.pad_left: self.pad_left + self.grid_cols,
                         :]
        cropped_y_pred = y_pred[:,
                         self.pad_top: self.pad_top + self.grid_rows,
                         self.pad_left: self.pad_left + self.grid_cols,
                         :]

        # 展平数据
        y_true_flat = tf.reshape(cropped_y_true, [-1])
        y_pred_flat = tf.reshape(cropped_y_pred, [-1])

        # 计算均值
        y_true_mean = tf.reduce_mean(y_true_flat)

        # 计算总平方和和残差平方和
        ss_tot = tf.reduce_sum(tf.square(y_true_flat - y_true_mean))
        ss_res = tf.reduce_sum(tf.square(y_true_flat - y_pred_flat))

        # 添加数值稳定性检查
        epsilon = tf.constant(1e-8, dtype=tf.float32)
        ss_tot = tf.maximum(ss_tot, epsilon)  # 防止除零

        # 计算R2
        r2 = 1.0 - (ss_res / ss_tot)

        # 限制R2的范围，防止异常值
        r2 = tf.clip_by_value(r2, -10.0, 1.0)

        # 检查是否为NaN或Inf
        r2 = tf.where(tf.math.is_finite(r2), r2, tf.constant(0.0, dtype=tf.float32))

        return r2

    def rmse_metric(self, y_true, y_pred):
        """RMSE评估指标 - 在裁剪后的区域计算，增强数值稳定性"""
        # 确保数据类型
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 裁剪到原始网格尺寸
        cropped_y_true = y_true[:,
                         self.pad_top: self.pad_top + self.grid_rows,
                         self.pad_left: self.pad_left + self.grid_cols,
                         :]
        cropped_y_pred = y_pred[:,
                         self.pad_top: self.pad_top + self.grid_rows,
                         self.pad_left: self.pad_left + self.grid_cols,
                         :]

        # 计算MSE
        mse = tf.reduce_mean(tf.square(cropped_y_true - cropped_y_pred))

        # 添加数值稳定性检查
        epsilon = tf.constant(1e-8, dtype=tf.float32)
        mse = tf.maximum(mse, epsilon)  # 确保MSE非负且不为零

        # 计算RMSE
        rmse = tf.sqrt(mse)

        # 检查是否为NaN或Inf
        rmse = tf.where(tf.math.is_finite(rmse), rmse, tf.constant(1.0, dtype=tf.float32))

        return rmse

    def train_model(self, epochs=1000, batch_size=8, stratified=False, k_folds=5):
        """
        训练U-Net模型，支持K折交叉验证

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            stratified: 是否使用分层采样
            k_folds: K折交叉验证的折数
        """
        print(f"\n开始训练U-Net模型...")
        print(f"训练参数: epochs={epochs}, batch_size={batch_size}, k_folds={k_folds}")

        # 在训练开始前完全打乱数据
        self.shuffle_data_before_training()

        # 创建模型保存目录
        os.makedirs(self.model_save_path, exist_ok=True)

        if k_folds > 1:
            # K折交叉验证
            return self._train_with_kfold(epochs, batch_size, k_folds)
        else:
            # 单次训练
            return self._train_single(epochs, batch_size, stratified)

    def _train_with_kfold(self, epochs, batch_size, k_folds):
        """
        执行K折交叉验证训练
        """
        print(f"\n=== 开始{k_folds}折交叉验证训练 ===")

        from sklearn.model_selection import KFold
        import numpy as np

        # 准备数据索引
        n_samples = len(self.X_data)
        indices = np.arange(n_samples)

        # 创建K折分割器
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # 存储每折的结果
        fold_results = []
        fold_r2_scores = []
        fold_rmse_scores = []

        # 执行K折交叉验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            print(f"\n--- 第 {fold + 1}/{k_folds} 折训练 ---")
            print(f"训练样本数: {len(train_idx)}, 验证样本数: {len(val_idx)}")

            # 分割数据
            X_train_fold = self.X_data[train_idx]
            Y_train_fold = self.Y_data[train_idx]
            X_val_fold = self.X_data[val_idx]
            Y_val_fold = self.Y_data[val_idx]
            
            # 对数据进行空间填充以适应模型输入要求
            X_train_fold_padded = np.pad(X_train_fold,
                                       ((0, 0), (self.pad_top, self.pad_bottom), 
                                        (self.pad_left, self.pad_right), (0, 0)),
                                       mode='constant', constant_values=0)
            Y_train_fold_padded = np.pad(Y_train_fold,
                                       ((0, 0), (self.pad_top, self.pad_bottom), 
                                        (self.pad_left, self.pad_right), (0, 0)),
                                       mode='constant', constant_values=0)
            X_val_fold_padded = np.pad(X_val_fold,
                                     ((0, 0), (self.pad_top, self.pad_bottom), 
                                      (self.pad_left, self.pad_right), (0, 0)),
                                     mode='constant', constant_values=0)
            Y_val_fold_padded = np.pad(Y_val_fold,
                                     ((0, 0), (self.pad_top, self.pad_bottom), 
                                      (self.pad_left, self.pad_right), (0, 0)),
                                     mode='constant', constant_values=0)

            # 创建新的模型实例
            model = self.build_unet_model()

            # 设置初始学习率
            initial_lr = 0.001
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                loss=self.relative_loss,
                metrics=['mse', 'mae', self.r2_metric, self.rmse_metric]
            )

            # 设置回调函数
            fold_model_path = os.path.join(self.model_save_path, f'{self.pollutant}_fold_{fold + 1}.h5')
            fold_log_path = os.path.join(self.model_save_path, f'{self.pollutant}_fold_{fold + 1}_log.csv')

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    fold_model_path,
                    monitor='val_r2_metric',
                    mode='max',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1,
                    cooldown=5
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_r2_metric',
                    mode='max',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.CSVLogger(fold_log_path, append=False)
            ]

            # 训练模型
            print(f"开始第 {fold + 1} 折训练...")
            history = model.fit(
                X_train_fold_padded, Y_train_fold_padded,
                validation_data=(X_val_fold_padded, Y_val_fold_padded),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # 评估模型
            print(f"第 {fold + 1} 折训练完成，正在评估...")
            val_results = model.evaluate(X_val_fold_padded, Y_val_fold_padded, verbose=0)

            # 提取R²和RMSE值
            val_loss = val_results[0]
            val_mse = val_results[1]
            val_mae = val_results[2]
            val_r2 = val_results[3]
            val_rmse = val_results[4]

            # 存储结果
            fold_result = {
                'fold': fold + 1,
                'val_loss': val_loss,
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'history': history.history
            }
            fold_results.append(fold_result)
            fold_r2_scores.append(val_r2)
            fold_rmse_scores.append(val_rmse)

            print(f"第 {fold + 1} 折结果:")
            print(f"  验证损失: {val_loss:.6f}")
            print(f"  验证R²: {val_r2:.6f}")
            print(f"  验证RMSE: {val_rmse:.6f}")
            print(f"  验证MSE: {val_mse:.6f}")
            print(f"  验证MAE: {val_mae:.6f}")

        # 计算总体结果
        mean_r2 = np.mean(fold_r2_scores)
        std_r2 = np.std(fold_r2_scores)
        mean_rmse = np.mean(fold_rmse_scores)
        std_rmse = np.std(fold_rmse_scores)

        print(f"\n=== K折交叉验证总结 ===")
        print(f"各折R²值: {[f'{r2:.6f}' for r2 in fold_r2_scores]}")
        print(f"各折RMSE值: {[f'{rmse:.6f}' for rmse in fold_rmse_scores]}")
        print(f"平均R²: {mean_r2:.6f} ± {std_r2:.6f}")
        print(f"平均RMSE: {mean_rmse:.6f} ± {std_rmse:.6f}")

        # 找到最佳折
        best_fold_idx = np.argmax(fold_r2_scores)
        best_fold = fold_results[best_fold_idx]
        print(f"最佳折: 第 {best_fold['fold']} 折 (R² = {best_fold['val_r2']:.6f})")

        # 保存交叉验证结果
        cv_results = {
            'fold_results': fold_results,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'best_fold': best_fold_idx + 1,
            'fold_r2_scores': fold_r2_scores,
            'fold_rmse_scores': fold_rmse_scores
        }

        # 绘制交叉验证结果
        self._plot_cv_results(cv_results)

        return cv_results

    def _train_single(self, epochs, batch_size, stratified):
        """
        执行单次训练（原有的训练逻辑）
        """
        # 数据生成器
        train_gen, val_gen, n_train, n_val = self.data_generator(batch_size, validation_split=0.2,
                                                                 stratified_split=stratified)

        # 创建模型
        model = self.build_unet_model()

        # 设置初始学习率
        initial_lr = 0.001
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
            loss=self.relative_loss,
            metrics=['mse', 'mae', self.r2_metric, self.rmse_metric]
        )

        # 设置回调函数
        model_path = os.path.join(self.model_save_path, f'{self.pollutant}_best.h5')
        log_path = os.path.join(self.model_save_path, f'{self.pollutant}_training_log.csv')

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_r2_metric',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=20,
                min_lr=1e-7,
                verbose=1,
                cooldown=5
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_r2_metric',
                mode='max',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(log_path, append=False)
        ]

        # 训练模型
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # 绘制训练历史
        self.save_training_history(history)

        return history

    def _plot_cv_results(self, cv_results):
        """
        绘制交叉验证结果
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.pollutant} K折交叉验证结果', fontsize=16)

        # 各折R²对比
        axes[0, 0].bar(range(1, len(cv_results['fold_r2_scores']) + 1),
                       cv_results['fold_r2_scores'],
                       color='skyblue', alpha=0.7)
        axes[0, 0].axhline(y=cv_results['mean_r2'], color='red', linestyle='--',
                           label=f'平均R² = {cv_results["mean_r2"]:.4f}')
        axes[0, 0].set_title('各折R²值对比')
        axes[0, 0].set_xlabel('折数')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 各折RMSE对比
        axes[0, 1].bar(range(1, len(cv_results['fold_rmse_scores']) + 1),
                       cv_results['fold_rmse_scores'],
                       color='lightcoral', alpha=0.7)
        axes[0, 1].axhline(y=cv_results['mean_rmse'], color='red', linestyle='--',
                           label=f'平均RMSE = {cv_results["mean_rmse"]:.4f}')
        axes[0, 1].set_title('各折RMSE值对比')
        axes[0, 1].set_xlabel('折数')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 训练历史（最佳折）
        best_fold_idx = cv_results['best_fold'] - 1
        best_history = cv_results['fold_results'][best_fold_idx]['history']

        # 损失曲线
        axes[1, 0].plot(best_history['loss'], label='训练损失', color='blue')
        axes[1, 0].plot(best_history['val_loss'], label='验证损失', color='red')
        axes[1, 0].set_title(f'最佳折(第{cv_results["best_fold"]}折)损失曲线')
        axes[1, 0].set_xlabel('轮数')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # R²曲线
        axes[1, 1].plot(best_history['r2_metric'], label='训练R²', color='blue')
        axes[1, 1].plot(best_history['val_r2_metric'], label='验证R²', color='red')
        axes[1, 1].set_title(f'最佳折(第{cv_results["best_fold"]}折)R²曲线')
        axes[1, 1].set_xlabel('轮数')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(self.model_save_path, f'{self.pollutant}_cv_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"交叉验证结果图已保存: {plot_path}")
        plt.show()

        # 保存详细结果到文件
        results_path = os.path.join(self.model_save_path, f'{self.pollutant}_cv_summary.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"{self.pollutant} K折交叉验证结果摘要\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"折数: {len(cv_results['fold_r2_scores'])}\n")
            f.write(f"各折R²值: {cv_results['fold_r2_scores']}\n")
            f.write(f"各折RMSE值: {cv_results['fold_rmse_scores']}\n")
            f.write(f"平均R²: {cv_results['mean_r2']:.6f} ± {cv_results['std_r2']:.6f}\n")
            f.write(f"平均RMSE: {cv_results['mean_rmse']:.6f} ± {cv_results['std_rmse']:.6f}\n")
            f.write(f"最佳折: 第 {cv_results['best_fold']} 折\n")
            f.write(f"最佳R²: {cv_results['fold_results'][cv_results['best_fold'] - 1]['val_r2']:.6f}\n")

        print(f"交叉验证结果摘要已保存: {results_path}")

    def save_training_history(self, history):
        """保存训练历史和绘制曲线"""
        print("保存训练历史...")

        # 保存训练历史到CSV
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(self.output_path, f'{self.pollutant}_unet_training_history.csv'), index=False)

        # 绘制训练曲线
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 损失曲线
        axes[0, 0].plot(history.history['loss'], label='训练损失')
        axes[0, 0].plot(history.history['val_loss'], label='验证损失')
        axes[0, 0].set_title('模型损失 (Relative Loss)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # R2曲线
        if 'r2_metric' in history.history:
            axes[0, 1].plot(history.history['r2_metric'], label='训练R²')
            axes[0, 1].plot(history.history['val_r2_metric'], label='验证R²')
            axes[0, 1].set_title('R²决定系数')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # 显示最佳R2值
            best_r2 = max(history.history['val_r2_metric'])
            axes[0, 1].axhline(y=best_r2, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].text(0.02, 0.98, f'最佳验证R²: {best_r2:.4f}',
                            transform=axes[0, 1].transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # RMSE曲线
        if 'rmse_metric' in history.history:
            axes[0, 2].plot(history.history['rmse_metric'], label='训练RMSE')
            axes[0, 2].plot(history.history['val_rmse_metric'], label='验证RMSE')
            axes[0, 2].set_title('均方根误差')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('RMSE')
            axes[0, 2].legend()
            axes[0, 2].grid(True)

            # 显示最佳RMSE值
            best_rmse = min(history.history['val_rmse_metric'])
            axes[0, 2].axhline(y=best_rmse, color='red', linestyle='--', alpha=0.7)
            axes[0, 2].text(0.02, 0.98, f'最佳验证RMSE: {best_rmse:.4f}',
                            transform=axes[0, 2].transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # MSE曲线
        axes[1, 0].plot(history.history['mse'], label='训练MSE')
        axes[1, 0].plot(history.history['val_mse'], label='验证MSE')
        axes[1, 0].set_title('均方误差')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # MAE曲线
        axes[1, 1].plot(history.history['mae'], label='训练MAE')
        axes[1, 1].plot(history.history['val_mae'], label='验证MAE')
        axes[1, 1].set_title('平均绝对误差')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 学习率曲线
        if 'lr' in history.history:
            axes[1, 2].plot(history.history['lr'])
            axes[1, 2].set_title('学习率')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].axis('off')  # 如果没有lr信息，隐藏子图

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{self.pollutant}_unet_training_curves.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 打印最终训练结果摘要
        print("\n=== 训练结果摘要 ===")
        if 'val_r2_metric' in history.history:
            print(f"最佳验证R²: {max(history.history['val_r2_metric']):.4f}")
        if 'val_rmse_metric' in history.history:
            print(f"最佳验证RMSE: {min(history.history['val_rmse_metric']):.4f}")
        print(f"最佳验证损失: {min(history.history['val_loss']):.6f}")
        print(f"最终验证MSE: {history.history['val_mse'][-1]:.6f}")
        print(f"最终验证MAE: {history.history['val_mae'][-1]:.6f}")

    def save_model_config(self):
        """Save model metadata and optional scaler objects."""
        try:
            # Save model configuration.
            config = {
                'model_type': 'unet',
                'input_shape': self.input_shape,
                'num_emission_channels': self.num_emission_channels,
                'num_chemical_channels': self.num_chemical_channels,
                'pollutant': self.pollutant,
                'data_path': self.data_path,
                'output_path': self.output_path,
                'model_version': '2.0_ratio_based',
                'standardization_method': 'no_standardization_for_ratios'
            }
            
            config_path = os.path.join(self.output_path, 'model_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"模型配置已保存到: {config_path}")
            
            # Save scalers when they exist.
            if hasattr(self, 'scalers') and self.scalers:
                scalers_path = os.path.join(self.output_path, 'scalers.pkl')
                with open(scalers_path, 'wb') as f:
                    pickle.dump(self.scalers, f)
                print(f"标准化器已保存到: {scalers_path}")
                print(f"保存的标准化器: {list(self.scalers.keys())}")
            else:
                print("无需保存标准化器（比例值不进行标准化）")
                
        except Exception as e:
            print(f"保存模型配置时出错: {e}")
            # 创建一个空的标准化器文件以避免加载时出错
            try:
                scalers_path = os.path.join(self.output_path, 'scalers.pkl')
                with open(scalers_path, 'wb') as f:
                    pickle.dump({}, f)
                print("已创建空的标准化器文件")
            except:
                pass

    def run_training(self, epochs=200, batch_size=32, normal_repeat=3, extreme_repeat=5):
        """运行完整的训练流程，支持可配置的重复策略"""
        print("=== DeepRSM U-Net模型训练开始（数据重复增强版本）===")
        print("训练目标：建立排放和背景化学指标到PM2.5浓度的端到端响应关系")
        print(f"数据增强策略：一般情景重复{normal_repeat}次，极端情景重复{extreme_repeat}次，训练前随机打乱")
        print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 加载和准备数据
        # load_training_data 现在负责加载并处理所有数据到X_data和Y_data
        if not self.load_training_data(num_scenarios_to_load=150, 
                                     normal_repeat=normal_repeat, 
                                     extreme_repeat=extreme_repeat):  # 可配置重复次数
            print("训练数据加载失败，退出训练")
            return

        # 2. 训练模型
        history = self.train_model(epochs=epochs, batch_size=batch_size)

        if history is None:
            print("模型训练失败")
            return

        # 3. 输出训练总结
        print("\n=== 训练完成 ===")
        if hasattr(history, 'history') and 'val_r2_metric' in history.history:
            print(f"最佳验证R²: {max(history.history['val_r2_metric']):.4f}")
        if hasattr(history, 'history') and 'val_rmse_metric' in history.history:
            print(f"最佳验证RMSE: {min(history.history['val_rmse_metric']):.4f}")
        print(f"最佳验证损失: {min(history.history['val_loss']):.6f}")
        print(f"模型已保存到: {self.output_path}")
        print(f"成功建立排放-浓度响应关系（一般×{normal_repeat}, 极端×{extreme_repeat}）!")

        return history

    def k_fold_cross_validation(self, k_folds=5, epochs=200, batch_size=8, stratified=True):
        """K折交叉验证训练"""
        print(f"=== {k_folds}折交叉验证开始 ===")

        n_samples = self.X_data.shape[0]

        if stratified and n_samples >= 100:
            # 分层K折：确保每折都包含正常和极端情景
            normal_scenarios = list(range(min(100, n_samples)))
            extreme_scenarios = list(range(100, n_samples))

            np.random.shuffle(normal_scenarios)
            np.random.shuffle(extreme_scenarios)

            # 将正常和极端情景分别分成k折
            normal_folds = np.array_split(normal_scenarios, k_folds)
            extreme_folds = np.array_split(extreme_scenarios, k_folds)

            print(f"分层交叉验证: 正常情景{len(normal_scenarios)}个，极端情景{len(extreme_scenarios)}个")

        else:
            # 传统K折
            indices = np.random.permutation(n_samples)
            folds = np.array_split(indices, k_folds)
            print(f"传统交叉验证: 总计{n_samples}个情景")

        cv_results = []

        for fold in range(k_folds):
            print(f"\n--- 第 {fold + 1}/{k_folds} 折训练 ---")

            if stratified and n_samples >= 100:
                # 当前折作为验证集
                val_normal = normal_folds[fold].tolist()
                val_extreme = extreme_folds[fold].tolist()
                val_indices = np.array(val_normal + val_extreme)

                # 其他折作为训练集
                train_normal = []
                train_extreme = []
                for i in range(k_folds):
                    if i != fold:
                        train_normal.extend(normal_folds[i].tolist())
                        train_extreme.extend(extreme_folds[i].tolist())
                train_indices = np.array(train_normal + train_extreme)

                print(f"训练集: 正常{len(train_normal)}个 + 极端{len(train_extreme)}个 = {len(train_indices)}个")
                print(f"验证集: 正常{len(val_normal)}个 + 极端{len(val_extreme)}个 = {len(val_indices)}个")

            else:
                # 传统划分
                val_indices = folds[fold]
                train_indices = np.concatenate([folds[i] for i in range(k_folds) if i != fold])
                print(f"训练集: {len(train_indices)}个情景，验证集: {len(val_indices)}个情景")

            # 构建新模型
            self.model = self.build_unet_model()

            # 编译模型
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=self.relative_loss,
                metrics=[self.r2_metric, self.rmse_metric, 'mse', 'mae']
            )

            # 设置回调（简化版，避免保存太多模型）
            callbacks = [
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.8,  # 更温和的衰减
                    patience=25,  # 增加patience
                    min_lr=1e-9,
                    verbose=1,
                    cooldown=10  # 添加冷却期
                ),
                ReduceLROnPlateau(
                    monitor='val_r2_metric',
                    factor=0.9,  # 基于R2的更温和衰减
                    patience=25,
                    min_lr=1e-9,
                    mode='max',  # R²越大越好
                    verbose=1,

                    cooldown=8
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=300,  # 大幅增加patience，给模型更多时间学习
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=1e-6  # 添加最小改善阈值
                ),
                # 添加学习率调度器
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: 0.001 * 0.95 ** epoch,  # 指数衰减
                    verbose=0
                )
            ]

            # 创建当前折的数据生成器
            def generate_fold_batches(sample_indices, is_training=True):
                while True:
                    if is_training:
                        np.random.shuffle(sample_indices)

                    for i in range(0, len(sample_indices), batch_size):
                        batch_idx = sample_indices[i:i + batch_size]
                        batch_X = self.X_data[batch_idx]
                        batch_Y = self.Y_data[batch_idx]
                        
                        # 进行空间填充以适应模型输入要求
                        batch_X_padded = np.pad(batch_X,
                                              ((0, 0), (self.pad_top, self.pad_bottom), 
                                               (self.pad_left, self.pad_right), (0, 0)),
                                              mode='constant', constant_values=0)
                        batch_Y_padded = np.pad(batch_Y,
                                              ((0, 0), (self.pad_top, self.pad_bottom), 
                                               (self.pad_left, self.pad_right), (0, 0)),
                                              mode='constant', constant_values=0)
                        
                        yield batch_X_padded, batch_Y_padded

            train_gen = generate_fold_batches(train_indices, True)
            val_gen = generate_fold_batches(val_indices, False)

            # 训练当前折
            history = self.model.fit(
                train_gen,
                steps_per_epoch=max(1, len(train_indices) // batch_size),
                validation_data=val_gen,
                validation_steps=max(1, len(val_indices) // batch_size),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )

            # 记录当前折的最佳结果
            fold_result = {
                'fold': fold + 1,
                'best_val_loss': min(history.history['val_loss']),
                'best_val_r2': max(history.history['val_r2_metric']) if 'val_r2_metric' in history.history else 0,
                'best_val_rmse': min(history.history['val_rmse_metric']) if 'val_rmse_metric' in history.history else 0,
                'final_val_mse': history.history['val_mse'][-1],
                'final_val_mae': history.history['val_mae'][-1],
                'train_samples': len(train_indices),
                'val_samples': len(val_indices)
            }

            cv_results.append(fold_result)

            print(f"第{fold + 1}折结果:")
            print(f"  最佳验证损失: {fold_result['best_val_loss']:.6f}")
            print(f"  最佳验证R²: {fold_result['best_val_r2']:.4f}")
            print(f"  最佳验证RMSE: {fold_result['best_val_rmse']:.4f}")

        # 计算交叉验证统计结果
        self._summarize_cv_results(cv_results)

        return cv_results

    def _summarize_cv_results(self, cv_results):
        """总结交叉验证结果"""
        print(f"\n=== {len(cv_results)}折交叉验证结果总结 ===")

        # 首先显示每一折的详细结果
        print("\n各折详细结果:")
        print("折数\t验证损失\t验证R²\t\t验证RMSE\t验证MSE\t\t验证MAE")
        print("-" * 80)
        for result in cv_results:
            print(
                f"{result['fold']}\t{result['best_val_loss']:.6f}\t{result['best_val_r2']:.4f}\t\t{result['best_val_rmse']:.4f}\t\t{result['final_val_mse']:.6f}\t{result['final_val_mae']:.4f}")

        # 提取各指标
        val_losses = [r['best_val_loss'] for r in cv_results]
        val_r2s = [r['best_val_r2'] for r in cv_results]
        val_rmses = [r['best_val_rmse'] for r in cv_results]
        val_mses = [r['final_val_mse'] for r in cv_results]
        val_maes = [r['final_val_mae'] for r in cv_results]

        # 计算统计量
        print(f"\n=== 交叉验证统计结果 ===")
        print("验证损失 (Relative Loss):")
        print(f"  均值: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
        print(f"  范围: [{np.min(val_losses):.6f}, {np.max(val_losses):.6f}]")
        print(f"  变异系数: {np.std(val_losses) / np.mean(val_losses) * 100:.2f}%")

        print("验证R²:")
        print(f"  均值: {np.mean(val_r2s):.4f} ± {np.std(val_r2s):.4f}")
        print(f"  范围: [{np.min(val_r2s):.4f}, {np.max(val_r2s):.4f}]")
        print(f"  变异系数: {np.std(val_r2s) / np.mean(val_r2s) * 100:.2f}%" if np.mean(
            val_r2s) > 0 else "  变异系数: N/A")

        print("验证RMSE:")
        print(f"  均值: {np.mean(val_rmses):.4f} ± {np.std(val_rmses):.4f}")
        print(f"  范围: [{np.min(val_rmses):.4f}, {np.max(val_rmses):.4f}]")
        print(f"  变异系数: {np.std(val_rmses) / np.mean(val_rmses) * 100:.2f}%")

        print("验证MSE:")
        print(f"  均值: {np.mean(val_mses):.6f} ± {np.std(val_mses):.6f}")

        print("验证MAE:")
        print(f"  均值: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")

        # 找出最佳折
        best_fold_idx = np.argmax(val_r2s)  # R²最高的折
        print(f"\n最佳折数: 第{cv_results[best_fold_idx]['fold']}折")
        print(f"  验证R²: {cv_results[best_fold_idx]['best_val_r2']:.4f}")
        print(f"  验证损失: {cv_results[best_fold_idx]['best_val_loss']:.6f}")
        print(f"  验证RMSE: {cv_results[best_fold_idx]['best_val_rmse']:.4f}")

        # 保存交叉验证结果
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(os.path.join(self.output_path, f'{self.pollutant}_cv_results.csv'), index=False)

        # 绘制交叉验证结果图
        self._plot_cv_results_simple(cv_results)

        print(f"交叉验证结果已保存到: {self.output_path}")

        # 计算模型稳定性指标
        r2_stability = 1 - (np.std(val_r2s) / np.mean(val_r2s)) if np.mean(val_r2s) > 0 else 0
        loss_stability = 1 - (np.std(val_losses) / np.mean(val_losses))
        print(f"\n模型稳定性评估:")
        print(f"  R²稳定性: {r2_stability:.4f} (越接近1越稳定)")
        print(f"  损失稳定性: {loss_stability:.4f} (越接近1越稳定)")

    def _plot_cv_results_simple(self, cv_results):
        """绘制交叉验证结果图（简化版本）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        folds = [r['fold'] for r in cv_results]
        val_losses = [r['best_val_loss'] for r in cv_results]
        val_r2s = [r['best_val_r2'] for r in cv_results]
        val_rmses = [r['best_val_rmse'] for r in cv_results]
        val_mses = [r['final_val_mse'] for r in cv_results]

        # 验证损失
        axes[0, 0].bar(folds, val_losses, alpha=0.7, color='skyblue')
        axes[0, 0].axhline(y=np.mean(val_losses), color='red', linestyle='--', label=f'均值: {np.mean(val_losses):.6f}')
        axes[0, 0].set_title('各折验证损失')
        axes[0, 0].set_xlabel('折数')
        axes[0, 0].set_ylabel('验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 验证R²
        axes[0, 1].bar(folds, val_r2s, alpha=0.7, color='lightgreen')
        axes[0, 1].axhline(y=np.mean(val_r2s), color='red', linestyle='--', label=f'均值: {np.mean(val_r2s):.4f}')
        axes[0, 1].set_title('各折验证R²')
        axes[0, 1].set_xlabel('折数')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 验证RMSE
        axes[1, 0].bar(folds, val_rmses, alpha=0.7, color='orange')
        axes[1, 0].axhline(y=np.mean(val_rmses), color='red', linestyle='--', label=f'均值: {np.mean(val_rmses):.4f}')
        axes[1, 0].set_title('各折验证RMSE')
        axes[1, 0].set_xlabel('折数')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 验证MSE
        axes[1, 1].bar(folds, val_mses, alpha=0.7, color='pink')
        axes[1, 1].axhline(y=np.mean(val_mses), color='red', linestyle='--', label=f'均值: {np.mean(val_mses):.6f}')
        axes[1, 1].set_title('各折验证MSE')
        axes[1, 1].set_xlabel('折数')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{self.pollutant}_cv_results.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description='DeepRSM U-Net training script for emission-response learning')
    parser.add_argument('--data_path', type=str, default='./', help='Root directory of the dataset')
    parser.add_argument('--output_path', type=str, default='./models_unet/', help='Directory for model outputs')
    parser.add_argument('--pollutant', type=str, default='PM25_TOT', help='Target pollutant to predict')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Mini-batch size')

    # Cross-validation settings
    parser.add_argument('--k_folds', type=int, default=5, help='Number of CV folds; use 1 to disable cross-validation')
    parser.add_argument('--stratified', action='store_true', default=True, help='Use stratified sampling for regular and extreme scenarios')
    parser.add_argument('--no_stratified', dest='stratified', action='store_false', help='Disable stratified sampling')
    parser.add_argument('--load_scenarios', type=int, default=150, help='Number of scenarios to load')

    # Scenario repetition settings
    parser.add_argument('--normal_repeat', type=int, default=6, help='Repeat factor for regular scenarios')
    parser.add_argument('--extreme_repeat', type=int, default=10, help='Repeat factor for extreme boundary scenarios')

    args = parser.parse_args()

    print(f"=== DeepRSM U-Net模型训练开始（数据重复增强版本）===")
    print("训练目标：建立排放和背景化学指标到PM2.5浓度的端到端响应关系")
    print(f"数据增强策略：一般情景重复{args.normal_repeat}次，极端情景重复{args.extreme_repeat}次，训练前随机打乱")
    print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练参数:")
    print(f"  污染物: {args.pollutant}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  分层采样: {args.stratified}")
    print(f"  K折交叉验证: {args.k_folds}折" if args.k_folds > 1 else "  交叉验证: 不使用")
    print(f"  加载情景数: {args.load_scenarios}")
    print(f"  数据增强: 一般情景×{args.normal_repeat}, 极端情景×{args.extreme_repeat}")
    print(f"  数据打乱: 训练开始前完全随机打乱")

    # 创建训练器
    trainer = DeepRSMTrainer(
        data_path=args.data_path,
        output_path=args.output_path,
        pollutant=args.pollutant
    )

    # 加载训练数据
    print("\n开始加载训练数据（包含数据重复增强）...")
    if not trainer.load_training_data(num_scenarios_to_load=args.load_scenarios,
                                    normal_repeat=args.normal_repeat,
                                    extreme_repeat=args.extreme_repeat):
        print("训练数据加载失败，退出训练")
        exit(1)

    # 开始训练
    if args.k_folds > 1:
        print(f"\n使用{args.k_folds}折交叉验证训练（重复增强数据）...")
        results = trainer.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            stratified=args.stratified,
            k_folds=args.k_folds
        )

        # 输出最终结果
        print(f"\n=== 最终训练结果（数据重复增强 一般×{args.normal_repeat}, 极端×{args.extreme_repeat}）===")
        print(f"平均R²: {results['mean_r2']:.6f} ± {results['std_r2']:.6f}")
        print(f"平均RMSE: {results['mean_rmse']:.6f} ± {results['std_rmse']:.6f}")
        print(f"最佳折: 第 {results['best_fold']} 折")
        print(f"各折R²: {[f'{r2:.4f}' for r2 in results['fold_r2_scores']]}")
        print("=== 交叉验证完成（数据重复增强）===")
        print("详细结果已保存到CSV文件和图表中")

    else:
        print(f"\n使用单次训练（重复增强数据）...")
        history = trainer.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            stratified=args.stratified,
            k_folds=1
        )

        print(f"\n=== 单次训练完成（数据重复增强 一般×{args.normal_repeat}, 极端×{args.extreme_repeat}）===")
        if hasattr(history, 'history') and 'val_r2_metric' in history.history:
            print(f"最佳验证R²: {max(history.history['val_r2_metric']):.4f}")
        if hasattr(history, 'history') and 'val_rmse_metric' in history.history:
            print(f"最佳验证RMSE: {min(history.history['val_rmse_metric']):.4f}")
        if hasattr(history, 'history') and 'val_loss' in history.history:
            print(f"最佳验证损失: {min(history.history['val_loss']):.6f}")
        print(f"模型已保存到: {trainer.output_path}")
        print(f"成功建立排放-浓度响应关系（使用重复增强数据 一般×{args.normal_repeat}, 极端×{args.extreme_repeat}）!")

    # 保存模型配置
    trainer.save_model_config()
    print("训练完成！")