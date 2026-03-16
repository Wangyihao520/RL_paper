# MARL_Envoronment.py - TensorFlow version
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


class RunningMeanStd:
    """Running mean and variance tracker for reward normalization."""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


# Set random seeds for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)


class HealthBenefitCalculator:
    """Estimate avoided premature deaths using the GEMM formulation."""

    def __init__(self, health_data_dir="./health", province_map_path="./prov_grid_map/36kmprov.csv"):
        """
        Initialize the health-benefit calculator.

        Args:
            health_data_dir: Directory containing health-related input files.
            province_map_path: Path to the province-grid mapping file.
        """
        self.health_data_dir = health_data_dir
        self.province_map_path = province_map_path
        self.population_data = None
        self.incidence_data = None
        self.province_map = None  # FID与省份的映射关系
        self.province_vsl = None  # 省份生命统计价值

        # GEMM模型参数
        self.theta_mean = 0.1430
        self.theta_sigma = 0.01807
        self.alpha = 1.6
        self.mu = 15.5
        self.nu = 36.8

        # 加载数据
        self._load_health_data()
        self._load_province_map()
        self._setup_province_vsl()

        # ✅ 性能优化：预计算并缓存省份级数据
        self._cache_province_health_data()

    def _load_health_data(self):
        """加载健康相关数据"""
        try:
            # 加载人口数据
            pop_file = os.path.join(self.health_data_dir, "36kmpop.csv")
            if os.path.exists(pop_file):
                self.population_data = pd.read_csv(pop_file)
                print(f"✅ 已加载人口数据: {len(self.population_data)} 条记录")
            else:
                print(f"⚠️ 人口数据文件不存在: {pop_file}")

            # 加载发病率/死亡率数据
            incidence_file = os.path.join(self.health_data_dir, "Incidence.csv")
            if os.path.exists(incidence_file):
                self.incidence_data = pd.read_csv(incidence_file)
                print(f"✅ 已加载发病率数据: {len(self.incidence_data)} 条记录")
            else:
                print(f"⚠️ 发病率数据文件不存在: {incidence_file}")

        except Exception as e:
            print(f"❌ 加载健康数据失败: {e}")
            self.population_data = None
            self.incidence_data = None

    def _load_province_map(self):
        """加载FID与省份的映射关系"""
        try:
            import pandas as pd
            self.province_map = pd.read_csv(self.province_map_path)
            print(f"✅ 省份映射数据加载成功，共{len(self.province_map)}个网格")

            # 创建FID到省份名称的映射字典，提高查询效率
            self.fid_to_province = dict(zip(self.province_map['FID'], self.province_map['name']))

            # 统计各省份的网格数量
            province_counts = self.province_map['name'].value_counts()
            print(f"✅ 省份网格分布: {dict(province_counts.head())}")

        except Exception as e:
            print(f"❌ 加载省份映射数据失败: {e}")
            self.province_map = None
            self.fid_to_province = {}

    def _setup_province_vsl(self):
        """设置各省份的VSL(生命统计价值) - 单位：万元/人"""
        # 基于各省份经济发展水平设置不同的VSL值
        self.province_vsl = {
            # 发达省份 (高VSL)
            'BJ': 494.0, 'SH': 524.0, 'GD': 396.0, 'JS': 391.0, 'ZJ': 388.0,
            # 中等发达省份 (中等VSL)
            'SD': 375.0, 'FJ': 380.0, 'TJ': 481.0, 'HUB': 315.0, 'HN': 291.0,
            'LN': 414.0, 'SC': 287.0, 'HB': 306.0, 'AH': 286.0, 'JX': 291.0,
            'HLJ': 346.0, 'JL': 345.0, 'SX': 278.0, 'HUN': 302.0,
            # 欠发达省份 (较低VSL)
            'NMG': 390.0, 'YN': 243.0, 'GX': 261.0, 'GZ': 212.0, 'GS': 241.0,
            'QH': 275.0, 'NX': 257.0, 'XJ': 270.0, 'XZ': 237.0,
            # 直辖市/特别行政区
            'CQ': 315.0, 'SI': 290.0, 'HA': 278.0
        }
        print(f"✅ 已设置 {len(self.province_vsl)} 个省份的VSL值")

    def _cache_province_health_data(self):
        """
        ✅ 性能优化：预计算并缓存省份级健康数据

        将每个省份的网格FID、人口、发病率等数据预先提取并缓存，
        避免每次计算时重复查询DataFrame
        """
        self.province_health_cache = {}

        if self.population_data is None or self.incidence_data is None or self.province_map is None:
            print("⚠️ 健康数据不完整，无法缓存省份健康数据")
            return

        try:
            # 创建FID到人口和发病率的快速查找字典
            pop_dict = dict(zip(self.population_data['FID'], self.population_data['pop']))
            inc_dict = dict(zip(self.incidence_data['FID'], self.incidence_data['value']))

            # 获取所有省份
            provinces = self.province_map['name'].unique()

            for province_name in provinces:
                province_fids = self.province_map[self.province_map['name'] == province_name]['FID'].values

                # 提取该省份所有网格的人口和发病率
                populations = []
                incidences = []
                valid_fids = []

                for fid in province_fids:
                    pop = pop_dict.get(fid, 0.0)
                    inc = inc_dict.get(fid, 0.0)

                    # 只保留有效数据
                    if pop > 0 and inc >= 0:
                        populations.append(pop)
                        incidences.append(inc)
                        valid_fids.append(fid)

                # 缓存为numpy数组
                self.province_health_cache[province_name] = {
                    'fids': np.array(valid_fids),
                    'populations': np.array(populations, dtype=np.float32),
                    'incidences': np.array(incidences, dtype=np.float32),
                    'vsl': self.province_vsl.get(province_name, 100.0)
                }

            print(f"✅ 已缓存 {len(self.province_health_cache)} 个省份的健康数据")

        except Exception as e:
            print(f"❌ 缓存省份健康数据失败: {e}")
            self.province_health_cache = {}

    def calculate_premature_deaths(self, pm25_concentrations, fid_list=None):
        """
        基于GEMM模型计算过早死亡人数

        参数:
        pm25_concentrations: PM2.5浓度数组 (μg/m³)
        fid_list: FID列表，用于匹配人口和发病率数据

        返回:
        premature_deaths: 过早死亡人数数组
        """
        if self.population_data is None or self.incidence_data is None:
            print("⚠️ 健康数据未加载，返回零值")
            return np.zeros_like(pm25_concentrations)

        try:
            # 如果没有提供FID列表，假设按顺序对应
            if fid_list is None:
                fid_list = list(range(len(pm25_concentrations)))

            # 获取对应的人口和发病率数据
            population = []
            incidence = []

            for fid in fid_list:
                # 查找对应的人口数据
                pop_row = self.population_data[self.population_data['FID'] == fid]
                if not pop_row.empty:
                    population.append(pop_row['pop'].iloc[0])
                else:
                    population.append(0)

                # 查找对应的发病率数据
                inc_row = self.incidence_data[self.incidence_data['FID'] == fid]
                if not inc_row.empty:
                    incidence.append(inc_row['value'].iloc[0])
                else:
                    incidence.append(0)

            population = np.array(population)
            incidence = np.array(incidence)

            # GEMM模型计算
            # z = max(0, PM2.5 - 2.4)
            z_values = np.maximum(0, pm25_concentrations - 2.4)

            # 使用均值进行计算（不考虑不确定性）
            theta = self.theta_mean

            # GEMM相对风险计算
            relative_risk = np.exp(
                theta * np.log(z_values / self.alpha + 1) /
                (1 + np.exp((self.mu - z_values) / self.nu))
            )

            # 过早死亡人数 = (RR - 1) / RR * 基线死亡率 * 人口 * 系数
            premature_deaths = ((relative_risk - 1) / relative_risk) * incidence * population * 0.72969255

            # 确保非负值
            premature_deaths = np.maximum(0, premature_deaths)

            return premature_deaths

        except Exception as e:
            print(f"❌ 计算过早死亡人数失败: {e}")
            return np.zeros_like(pm25_concentrations)

    def calculate_health_benefit_by_province(self, baseline_pm25, improved_pm25, province_name):
        """
        按省份计算健康效益 - 使用缓存和向量化计算（性能优化版）

        参数:
        baseline_pm25: 基准PM2.5浓度 (μg/m³) - 标量值，应用于该省份所有网格
        improved_pm25: 改善后PM2.5浓度 (μg/m³) - 标量值，应用于该省份所有网格
        province_name: 省份名称

        返回:
        health_benefit: 健康效益 (万元)
        avoided_deaths: 避免的过早死亡人数
        """
        try:
            # ✅ 优先使用缓存的省份健康数据（快速路径）
            if hasattr(self, 'province_health_cache') and province_name in self.province_health_cache:
                cache = self.province_health_cache[province_name]
                populations = cache['populations']
                incidences = cache['incidences']
                vsl = cache['vsl']

                if len(populations) == 0:
                    return 0.0, 0.0

                # ✅ 向量化GEMM模型计算
                # z = max(0, PM2.5 - 2.4)
                z_baseline = max(0.0, baseline_pm25 - 2.4)
                z_improved = max(0.0, improved_pm25 - 2.4)

                if z_baseline <= 0:
                    return 0.0, 0.0

                # 计算基准相对风险
                rr_baseline = np.exp(
                    self.theta_mean * np.log(z_baseline / self.alpha + 1) /
                    (1 + np.exp((self.mu - z_baseline) / self.nu))
                )

                # 计算改善后相对风险
                if z_improved > 0:
                    rr_improved = np.exp(
                        self.theta_mean * np.log(z_improved / self.alpha + 1) /
                        (1 + np.exp((self.mu - z_improved) / self.nu))
                    )
                else:
                    rr_improved = 1.0  # 无风险

                # 向量化计算过早死亡人数
                baseline_deaths = ((rr_baseline - 1) / rr_baseline) * incidences * populations * 0.72969255
                improved_deaths = ((rr_improved - 1) / rr_improved) * incidences * populations * 0.72969255

                # 计算避免的死亡人数
                avoided_deaths = np.maximum(0, baseline_deaths - improved_deaths)
                total_avoided_deaths = np.sum(avoided_deaths)

                # 计算健康效益
                total_health_benefit = total_avoided_deaths * vsl

                return total_health_benefit, total_avoided_deaths

            # ⚠️ 备用路径：如果没有缓存，使用原始方法（较慢）
            if (self.population_data is None or self.incidence_data is None or
                    self.province_map is None or not hasattr(self, 'fid_to_province')):
                return 0.0, 0.0

            province_fids = self.province_map[self.province_map['name'] == province_name]['FID'].values
            if len(province_fids) == 0:
                return 0.0, 0.0

            # 使用字典快速查找
            pop_dict = dict(zip(self.population_data['FID'], self.population_data['pop']))
            inc_dict = dict(zip(self.incidence_data['FID'], self.incidence_data['value']))

            populations = []
            incidences = []
            for fid in province_fids:
                pop = pop_dict.get(fid, 0.0)
                inc = inc_dict.get(fid, 0.0)
                if pop > 0 and inc >= 0:
                    populations.append(pop)
                    incidences.append(inc)

            if not populations:
                return 0.0, 0.0

            populations = np.array(populations, dtype=np.float32)
            incidences = np.array(incidences, dtype=np.float32)
            vsl = self.province_vsl.get(province_name, 100.0)

            # 向量化计算
            z_baseline = max(0.0, baseline_pm25 - 2.4)
            z_improved = max(0.0, improved_pm25 - 2.4)

            if z_baseline <= 0:
                return 0.0, 0.0

            rr_baseline = np.exp(
                self.theta_mean * np.log(z_baseline / self.alpha + 1) /
                (1 + np.exp((self.mu - z_baseline) / self.nu))
            )

            if z_improved > 0:
                rr_improved = np.exp(
                    self.theta_mean * np.log(z_improved / self.alpha + 1) /
                    (1 + np.exp((self.mu - z_improved) / self.nu))
                )
            else:
                rr_improved = 1.0

            baseline_deaths = ((rr_baseline - 1) / rr_baseline) * incidences * populations * 0.72969255
            improved_deaths = ((rr_improved - 1) / rr_improved) * incidences * populations * 0.72969255

            avoided_deaths = np.maximum(0, baseline_deaths - improved_deaths)
            total_avoided_deaths = float(np.sum(avoided_deaths))
            total_health_benefit = total_avoided_deaths * vsl

            # 输出计算摘要（仅在第一次调用或调试时）
            if hasattr(self, '_debug_health_calc') and self._debug_health_calc:
                print(f"🏥 {province_name} 健康效益计算摘要:")
                print(f"  总网格数: {len(province_fids)}")
                print(f"  PM2.5变化: {baseline_pm25:.1f} → {improved_pm25:.1f} μg/m³")
                print(f"  避免死亡: {total_avoided_deaths:.2f} 人")
                print(f"  VSL: {vsl} 万元/人")
                print(f"  健康效益: {total_health_benefit:.1f} 万元")

            return total_health_benefit, total_avoided_deaths

        except Exception as e:
            print(f"❌ 计算省份健康效益失败 ({province_name}): {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0

    def calculate_total_health_benefit(self, baseline_pm25_dict, improved_pm25_dict):
        """
        计算所有省份的总健康效益

        参数:
        baseline_pm25_dict: 基准PM2.5浓度字典 {省份名: 浓度}
        improved_pm25_dict: 改善后PM2.5浓度字典 {省份名: 浓度}

        返回:
        total_benefit: 总健康效益 (万元)
        province_benefits: 各省份健康效益详情
        """
        total_benefit = 0.0
        province_benefits = {}

        for province in baseline_pm25_dict.keys():
            if province in improved_pm25_dict:
                baseline = baseline_pm25_dict[province]
                improved = improved_pm25_dict[province]

                benefit, avoided = self.calculate_health_benefit_by_province(
                    baseline, improved, province
                )

                province_benefits[province] = {
                    'health_benefit': benefit,
                    'avoided_deaths': avoided,
                    'baseline_pm25': baseline,
                    'improved_pm25': improved,
                    'vsl': self.province_vsl.get(province, 100.0)
                }

                total_benefit += benefit

        return total_benefit, province_benefits


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置TensorFlow
print(f"TensorFlow版本: {tf.__version__}")
print(f"GPU可用性: {len(tf.config.list_physical_devices('GPU'))} 个GPU")

# 🚀 启用GPU训练配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 启用GPU内存增长（避免占用全部GPU内存）
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # 设置GPU内存限制（可选，如果需要限制内存使用）
        # tf.config.experimental.set_memory_limit(gpus[0], 4096)  # 限制为4GB

        print(f"✅ 已启用GPU训练，使用 {len(gpus)} 个GPU")
        print(f"GPU设备列表: {[gpu.name for gpu in gpus]}")

        # 🔧 GPU性能优化配置（简化版，避免兼容性问题）
        # 1. 禁用XLA编译优化（避免libdevice问题）
        tf.config.optimizer.set_jit(False)
        print("✅ 已禁用XLA JIT编译（避免libdevice问题）")

        # 2. 设置GPU执行策略
        tf.config.experimental.set_synchronous_execution(False)  # 异步执行
        print("✅ 已启用异步GPU执行")

        # 3. 禁用混合精度训练以避免类型不匹配问题
        # 确保使用标准精度训练
        print("ℹ️  使用标准精度训练（FP32）以确保稳定性")

    except RuntimeError as e:
        print(f"⚠️ GPU配置失败: {e}")
        print("将回退到CPU训练")
        # 设置CPU线程数
        tf.config.threading.set_intra_op_parallelism_threads(16)
        tf.config.threading.set_inter_op_parallelism_threads(16)
else:
    print("❌ 未检测到GPU，使用CPU训练")
    # 设置CPU线程数以优化CPU训练
    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.threading.set_inter_op_parallelism_threads(16)
    print("已配置CPU训练，线程数：16")

# 打印设备信息
print(f"可用设备: {tf.config.list_logical_devices()}")


# # 设置TensorFlow使用CPU（避免GPU内存问题）
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.config.threading.set_intra_op_parallelism_threads(16)
# tf.config.threading.set_inter_op_parallelism_threads(16)
# print("已配置TensorFlow使用CPU，线程数：16")


# 定义自定义损失函数
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    small_error_loss = 0.5 * tf.square(error)
    big_error_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, small_error_loss, big_error_loss)


def relative_loss(y_true, y_pred):
    """相对损失函数"""
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    abs_diff = tf.abs(y_pred - y_true)
    denominator = tf.maximum(tf.abs(y_true), epsilon)
    relative_error = tf.clip_by_value(abs_diff / denominator, 0.0, 10.0)

    return tf.reduce_mean(relative_error)


def r2_metric(y_true, y_pred):
    """R²指标"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    y_true_mean = tf.reduce_mean(y_true_flat)
    ss_tot = tf.reduce_sum(tf.square(y_true_flat - y_true_mean))
    ss_res = tf.reduce_sum(tf.square(y_true_flat - y_pred_flat))

    epsilon = tf.constant(1e-8, dtype=tf.float32)
    ss_tot = tf.maximum(ss_tot, epsilon)
    r2 = tf.clip_by_value(1.0 - (ss_res / ss_tot), -10.0, 1.0)

    return tf.where(tf.math.is_finite(r2), r2, tf.constant(0.0, dtype=tf.float32))


def rmse_metric(y_true, y_pred):
    """RMSE指标"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    epsilon = tf.constant(1e-8, dtype=tf.float32)
    mse = tf.maximum(mse, epsilon)
    rmse = tf.sqrt(mse)

    return tf.where(tf.math.is_finite(rmse), rmse, tf.constant(1.0, dtype=tf.float32))


class RSMEmissionEnv:
    """Emission-control environment driven by the U-Net DeepRSM surrogate."""

    def __init__(self,
                 model_path,
                 scaler_path,
                 base_conc_path,
                 clean_conc_path,
                 province_map_path,
                 base_emission_path,
                 cost_data_path,
                 transport_matrix_path=None,
                 fairness_weight=500,
                 fairness_metric='l1',
                 fairness_mode='penalty',
                 fairness_external_only=False,
                 max_steps=8):

        print("Initializing the U-Net DeepRSM environment...")

        # Grid settings
        self.grid_cols = 144
        self.grid_rows = 120
        self.padded_grid_rows = 128
        self.padded_grid_cols = 144

        self.pad_top = (self.padded_grid_rows - self.grid_rows) // 2
        self.pad_bottom = self.padded_grid_rows - self.grid_rows - self.pad_top
        self.pad_left = 0
        self.pad_right = 0

        # Model configuration
        self.precursors = ['NOx', 'SO2', 'NH3', 'VOC', 'PM25']
        self.sectors = ['AG', 'AR', 'IN', 'PP', 'TR']
        self.chemical_indicators = [
            'HNO3', 'N2O5', 'NO2', 'HONO', 'NO', 'H2O2', 'O3', 'OH',
            'FORM', 'ISOP', 'TERP', 'SO2', 'NH3',
            'PM25_SO4', 'PM25_NO3', 'PM25_NH4', 'PM25_OC', 'PM25_TOT'
        ]

        # Map species names in `cost.csv` to internal precursor names.
        self.species_mapping = {
            'NOx': 'NOx',
            'SO2': 'SO2',
            'NH3': 'NH3',
            'VOC': 'VOC',
            'PPM2.5': 'PM25'  # cost.csv中使用PPM2.5，我们的precursors中使用PM25
        }

        self.num_emission_channels = len(self.precursors) * len(self.sectors)
        self.num_chemical_channels = len(self.chemical_indicators) * 2
        self.total_input_channels = self.num_emission_channels + self.num_chemical_channels

        # Core environment settings
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
        self.action_dim = 25  # 5污染物 × 5行业 = 25个减排因子
        self.max_step_reduction = 0.20  # 单步最大减排率20%

        # Optional performance statistics controlled by the environment variable
        # `ENABLE_PERF_STATS=true`.
        self.enable_perf_stats = os.environ.get('ENABLE_PERF_STATS', 'False').lower() == 'true'
        self.perf_stats = {
            'build_emission_time': [],
            'standardize_time': [],
            'inference_time': [],
            'aggregate_time': [],
            'total_time': []
        }

        # Optional external logger for performance reporting.
        self.logger = None

        # Load cost coefficients and baseline emissions.
        self._load_cost_data(cost_data_path)
        self._load_base_emissions(base_emission_path)

        # Fairness-reward configuration.
        self.fairness_weight = float(fairness_weight)
        self.fairness_metric = str(fairness_metric).lower().strip()
        self.fairness_mode = str(fairness_mode).lower().strip()
        self.fairness_external_only = bool(fairness_external_only)
        self.transport_matrix_path = transport_matrix_path  # 延后到拿到province_names后再加载

        # Initialize the health-benefit calculator.
        try:
            self.health_calculator = HealthBenefitCalculator(health_data_dir="./health")
            print("Health-benefit calculator initialized successfully")
        except Exception as e:
            print(f"Health-benefit calculator initialization failed: {e}")
            self.health_calculator = None

        # Load the trained U-Net surrogate.
        self._load_unet_model(model_path, scaler_path)

        # Warm up the model once to reduce the first inference overhead.
        self._warmup_gpu_model()

        # Load concentration and mapping data.
        self._load_data(base_conc_path, clean_conc_path, province_map_path)

        # 计算基准浓度（不再计算固定目标）
        self.province_base_conc = self._calculate_province_concentration(self.base_conc_df)

        # === 新增：用U-Net模型预测减排0%时的PM2.5作为基准浓度 ===
        try:
            print("正在用U-Net模型预测减排0%时的省份基准PM2.5...")
            zero_reduction_factors = np.ones((self.num_provinces, self.action_dim))
            model_base_pm25 = self._predict_pm25(zero_reduction_factors)
            self.province_base_conc = {prov: float(val) for prov, val in zip(self.province_names, model_base_pm25)}
            print("已用模型预测的基准PM2.5替换原始基准浓度！")
            for i, prov in enumerate(self.province_names):
                print(f"  {prov}: {model_base_pm25[i]:.2f} μg/m³")
        except Exception as e:
            print(f"⚠️ 用模型基准替换失败: {e}")

        # 定义渐进式目标参数（移动到province_features计算之前）
        self.initial_target_ratio = 0.95  # 第1步目标：基准的95%
        self.final_target_ratio = 0.50  # 第8步目标：基准的50%

        # 🎯 计算省份特定特征（移动到province_base_conc计算之后）
        self._compute_province_features()

        # 🌍 加载区域数据用于协作奖励
        region_file = os.path.join(os.path.dirname(cost_data_path), 'region.csv')
        self.region_data = self._load_region_data(region_file)

        # ✅ 计算省份人口（用于人口加权PM2.5改善计算）
        self.province_population = self._compute_province_population()
        # ✅ 缓存人口数组（按province_names顺序）
        self.province_population_array = np.array(
            [self.province_population.get(p, 0.0) for p in self.province_names],
            dtype=np.float32
        )

        # ✅ 加载并缓存传输矩阵（源=行，受体=列）
        if self.transport_matrix_path is None:
            self.transport_matrix_path = os.path.join(os.path.dirname(cost_data_path), 'transport.csv')
        self._load_transport_matrix(self.transport_matrix_path)

        # 💰 加载GDP数据并缓存（用于省份博弈奖励）
        gdp_file = os.path.join(os.path.dirname(cost_data_path), 'GDP.csv')
        self._load_gdp_data(gdp_file)

        # ✅ 预计算基准PM2.5数组（用于快速计算改善率）
        self.province_base_conc_array = np.array([
            self.province_base_conc.get(prov_name, 50.0)
            for prov_name in self.province_names
        ], dtype=np.float32)

        # 现在可以设置状态维度（基于实际省份数量和新增特征）
        self.province_feature_dim = 8  # 新增的省份特征维度
        self.agent_state_dim = self.action_dim + self.num_provinces + self.province_feature_dim  # 25 + 32 + 8 = 65
        self.state_dim = self.agent_state_dim

        # 初始化状态
        self.current_state = None
        self.cumulative_factors = None
        self.last_step_pm25 = None

        # ✅ 性能优化：缓存基准排放数据
        self._cache_base_emissions_by_province()

        print(f"环境初始化完成，使用 {self.num_provinces} 个省份")
        print(f"状态空间维度: {self.state_dim}, 动作空间维度: {self.action_dim}")
        print(
            f"基准浓度范围: {min(self.province_base_conc.values()):.2f} ~ {max(self.province_base_conc.values()):.2f} μg/m³")
        print(f"目标设置: 第1步{self.initial_target_ratio:.0%} → 第{self.max_steps}步{self.final_target_ratio:.0%}")
        print(f"💰 真实成本函数已加载")
        print(f"🏭 基准排放数据已加载")
        print(f"🎯 省份特征维度: {self.province_feature_dim}")
        if getattr(self, 'transport_matrix', None) is not None:
            print(f"🧭 传输矩阵已加载: {self.transport_matrix_path}")
        print(f"⚖️ 公平性奖励: weight={self.fairness_weight}, metric={self.fairness_metric}, external_only={self.fairness_external_only}")
        print(f"⚖️ 公平性模式: {self.fairness_mode} (penalty=惩罚偏离, reward=奖励匹配)")

    def _load_transport_matrix(self, transport_matrix_path):
        """
        加载省际传输矩阵（源=行，受体=列），并按self.province_names重排为numpy矩阵。

        transport.csv格式：
          - 第一列 'prov'：源省（行索引）
          - 第一行其余列名：受体省（列索引）
        """
        self.transport_matrix = None
        self.transport_provinces = None

        if transport_matrix_path is None:
            return

        try:
            if not os.path.exists(transport_matrix_path):
                print(f"⚠️ 未找到传输矩阵文件: {transport_matrix_path}，公平性奖励将不可用")
                return

            df = pd.read_csv(transport_matrix_path)
            if 'prov' not in df.columns:
                print(f"⚠️ 传输矩阵缺少prov列: {transport_matrix_path}，公平性奖励将不可用")
                return

            df = df.copy()
            df['prov'] = df['prov'].astype(str)
            df = df.set_index('prov')

            # 仅保留与省份列表相关的行/列，并按province_names顺序重排
            provinces = list(self.province_names)
            missing_rows = [p for p in provinces if p not in df.index]
            missing_cols = [p for p in provinces if p not in df.columns]
            if missing_rows or missing_cols:
                print(f"⚠️ 传输矩阵与province_names不完全一致：缺少行{missing_rows}，缺少列{missing_cols}")

            df = df.reindex(index=provinces, columns=provinces, fill_value=0.0)
            mat = df.values.astype(np.float32, copy=False)

            # 可选：只算外部性（不把本省作为受体）
            if self.fairness_external_only:
                np.fill_diagonal(mat, 0.0)

            # 非负截断（避免异常负值影响）
            mat = np.maximum(mat, 0.0)

            self.transport_matrix = mat
            self.transport_provinces = provinces
            print(f"✅ 传输矩阵加载成功，形状: {self.transport_matrix.shape}")

        except Exception as e:
            print(f"❌ 加载传输矩阵失败: {e}")
            self.transport_matrix = None
            self.transport_provinces = None

    def _compute_fairness_reward(self, predicted_pm25, costs_array):
        """
        基于“责任份额 vs 成本份额”的匹配，计算公平性奖励（惩罚偏离）。

        责任（源头贡献）：
          b_j = Pop_j * max(0, base_pm25_j - current_pm25_j)
          r_i = sum_j T_{i->j} * b_j   (T: 源=行，受体=列)

        成本份额：
          s_i = cost_i / sum(cost)

        Returns:
          fairness_total_reward: float（总公平性项；penalty模式<=0，reward模式>=0）
          fairness_per_province: np.ndarray shape=(num_provinces,)
          diagnostics: dict（份额与中间量，便于记录/调试）
        """
        eps = 1e-8
        n = len(self.province_names)

        if self.fairness_weight <= 0.0:
            return 0.0, np.zeros(n, dtype=np.float32), {}
        if self.transport_matrix is None:
            return 0.0, np.zeros(n, dtype=np.float32), {'warning': 'transport_matrix_missing'}
        if costs_array is None or not isinstance(costs_array, np.ndarray) or costs_array.shape[0] != n:
            return 0.0, np.zeros(n, dtype=np.float32), {'warning': 'costs_array_invalid'}

        # 受体侧人口加权“改善”（用改善，而不是当前浓度）
        improvement = (self.province_base_conc_array - predicted_pm25).astype(np.float32, copy=False)
        improvement = np.maximum(improvement, 0.0)
        b = (self.province_population_array * improvement).astype(np.float32, copy=False)  # (n,)

        # 源头责任/贡献：r = T @ b
        r = self.transport_matrix.dot(b).astype(np.float32, copy=False)
        r = np.maximum(r, 0.0)

        sum_r = float(np.sum(r))
        sum_c = float(np.sum(np.maximum(costs_array, 0.0)))
        if sum_r <= eps or sum_c <= eps:
            return 0.0, np.zeros(n, dtype=np.float32), {
                'sum_r': sum_r,
                'sum_cost': sum_c,
                'warning': 'degenerate_sum'
            }

        responsibility_share = r / (sum_r + eps)
        cost_share = np.maximum(costs_array, 0.0) / (sum_c + eps)

        diff = cost_share - responsibility_share
        # 计算偏离度（penalty）：L1或L2
        if self.fairness_metric == 'l2':
            per_component = (diff ** 2).astype(np.float32)
            penalty = float(np.sum(per_component))
        else:
            # 默认L1
            per_component = np.abs(diff).astype(np.float32)
            penalty = float(np.sum(per_component))

        # 两种模式：
        # - penalty: 负惩罚（越偏离越扣分）
        # - reward: 相似度奖励（越匹配越加分），范围约[0, fairness_weight]
        if self.fairness_mode == 'reward':
            # 对于两个“份额分布”，L1与L2的最大值均<=2（极端两点质量分布）。
            # 因此用 score = 1 - penalty/2 映射到[0,1]（截断到>=0）
            score = float(max(0.0, 1.0 - penalty / 2.0))
            fairness_total_reward = float(self.fairness_weight * score)

            # 把总奖励分配到各省：越“贴合”的省分得越多，但不出现负值
            if penalty <= eps:
                per = np.full(n, fairness_total_reward / n, dtype=np.float32)
            else:
                weights = 1.0 - (per_component / (penalty + eps))  # in [0,1]
                weights = np.maximum(weights, 0.0)
                wsum = float(np.sum(weights))
                if wsum <= eps:
                    per = np.full(n, fairness_total_reward / n, dtype=np.float32)
                else:
                    per = (fairness_total_reward * (weights / (wsum + eps))).astype(np.float32)
        else:
            # penalty模式（原实现）：每省按偏离度扣分
            per = -(self.fairness_weight * per_component).astype(np.float32)
            fairness_total_reward = float(np.sum(per))
            score = None

        diagnostics = {
            'penalty': penalty,
            'score': score,
            'mode': self.fairness_mode,
            'sum_r': sum_r,
            'sum_cost': sum_c,
            'responsibility_share': responsibility_share,
            'cost_share': cost_share,
        }
        return fairness_total_reward, per, diagnostics

    def _load_cost_data(self, cost_data_path):
        """加载成本系数数据"""
        print("加载成本系数数据...")

        try:
            self.cost_data = pd.read_csv(cost_data_path)
            print(f"成本数据加载成功，形状: {self.cost_data.shape}")
            print(f"成本数据列名: {list(self.cost_data.columns)}")

            # 构建成本系数字典：{省份: {污染物: {'a': a系数, 'b': b系数}}}
            self.cost_coefficients = {}

            for _, row in self.cost_data.iterrows():
                province = row['sheng']
                species = row['species']
                a_coeff = row['a']
                b_coeff = row['b']

                if province not in self.cost_coefficients:
                    self.cost_coefficients[province] = {}

                # 映射species名称到我们的precursors
                if species in self.species_mapping:
                    mapped_species = self.species_mapping[species]
                    self.cost_coefficients[province][mapped_species] = {
                        'a': a_coeff,
                        'b': b_coeff
                    }

            print(f"成本系数数据处理完成，包含 {len(self.cost_coefficients)} 个省份")

            # 显示一些示例数据
            sample_province = list(self.cost_coefficients.keys())[0]
            print(f"示例省份 {sample_province} 的成本系数:")
            for species, coeffs in self.cost_coefficients[sample_province].items():
                print(f"  {species}: a={coeffs['a']:.3f}, b={coeffs['b']:.3f}")

        except Exception as e:
            print(f"加载成本数据失败: {e}")
            # 使用默认成本系数
            self._setup_default_cost_parameters()

    def _load_region_data(self, region_file_path):
        """加载省份区域数据用于协作奖励"""
        print("🌍 加载省份区域数据...")

        try:
            region_df = pd.read_csv(region_file_path)
            self.province_regions = {}

            for _, row in region_df.iterrows():
                province = row['prov']
                region = row['region']
                if province in self.province_names:
                    self.province_regions[province] = int(region)

            # 创建区域到省份的映射
            self.region_provinces = {}
            for province, region in self.province_regions.items():
                if region not in self.region_provinces:
                    self.region_provinces[region] = []
                self.region_provinces[region].append(province)

            print(f"✅ 区域数据加载成功，共 {len(self.province_regions)} 个省份，{len(self.region_provinces)} 个区域")

            # 显示各区域的省份分布
            for region, provinces in self.region_provinces.items():
                print(f"  区域{region}: {len(provinces)}个省份 {provinces}")

            return True

        except Exception as e:
            print(f"❌ 加载区域数据失败: {e}")
            # 如果加载失败，使用默认的单区域设置
            self.province_regions = {prov: 1 for prov in self.province_names}
            self.region_provinces = {1: self.province_names.copy()}
            print("⚠️ 使用默认单区域设置")
            return False

    def _compute_province_population(self):
        """
        计算每个省份的人口（用于人口加权PM2.5改善计算）

        Returns:
            province_population: 字典 {province_name: population}
        """
        province_population = {}

        if self.health_calculator is not None and self.health_calculator.population_data is not None:
            # 从health_calculator中获取人口数据
            for province_name in self.province_names:
                if province_name in self.province_to_grids:
                    province_grids = self.province_to_grids[province_name]
                    total_population = 0.0

                    for grid_fid in province_grids:
                        pop_row = self.health_calculator.population_data[
                            self.health_calculator.population_data['FID'] == grid_fid
                            ]
                        if not pop_row.empty and 'pop' in pop_row.columns:
                            total_population += pop_row['pop'].iloc[0]

                    province_population[province_name] = total_population
                else:
                    province_population[province_name] = 0.0
        else:
            # 如果没有人口数据，使用默认值（每个省份1000万人口）
            print("⚠️ 未找到人口数据，使用默认值（每个省份1000万人口）")
            for province_name in self.province_names:
                province_population[province_name] = 10000000.0

        print(f"✅ 省份人口计算完成，总人口: {sum(province_population.values()) / 1e8:.2f}亿")
        return province_population

    def _load_gdp_data(self, gdp_file_path):
        """
        加载GDP数据并缓存为数组（用于快速计算省份博弈奖励）

        Args:
            gdp_file_path: GDP数据文件路径
        """
        print("💰 加载GDP数据...")
        try:
            gdp_df = pd.read_csv(gdp_file_path)
            print(f"GDP数据加载成功，形状: {gdp_df.shape}")

            # 构建省份GDP字典
            self.province_gdp_dict = {}
            for _, row in gdp_df.iterrows():
                province = row['prov']
                gdp = row['GDP']
                if province in self.province_names:
                    self.province_gdp_dict[province] = float(gdp)

            # ✅ 预计算GDP数组（按province_names顺序，用于快速矩阵计算）
            self.province_gdp_array = np.array([
                self.province_gdp_dict.get(prov_name, 1e8)  # 默认1亿，避免除零
                for prov_name in self.province_names
            ], dtype=np.float32)

            print(f"✅ GDP数据缓存完成，共 {len(self.province_gdp_dict)} 个省份")
            print(
                f"   GDP范围: {np.min(self.province_gdp_array) / 1e8:.2f} ~ {np.max(self.province_gdp_array) / 1e8:.2f} 亿元")

        except Exception as e:
            print(f"❌ 加载GDP数据失败: {e}")
            # 使用默认GDP值（每个省份1000亿元）
            self.province_gdp_dict = {prov: 1e9 for prov in self.province_names}
            self.province_gdp_array = np.full(len(self.province_names), 1e9, dtype=np.float32)
            print("⚠️ 使用默认GDP值（每个省份1000亿元）")

    def _load_base_emissions(self, base_emission_path):
        """加载基准排放数据"""
        print("加载基准排放数据...")

        try:
            self.base_emissions = {}

            # 加载各个行业的排放数据
            for sector in self.sectors:
                sector_file = os.path.join(base_emission_path, f"{sector}.csv")
                if os.path.exists(sector_file):
                    sector_data = pd.read_csv(sector_file)
                    self.base_emissions[sector] = sector_data
                    print(f"  {sector} 排放数据加载成功，形状: {sector_data.shape}")
                else:
                    print(f"  警告: 未找到 {sector} 排放数据文件: {sector_file}")

            print(f"基准排放数据加载完成，包含 {len(self.base_emissions)} 个行业")

            # 显示排放数据的列名（以第一个行业为例）
            if self.base_emissions:
                sample_sector = list(self.base_emissions.keys())[0]
                sample_data = self.base_emissions[sample_sector]
                print(f"排放数据列名（{sample_sector}）: {list(sample_data.columns)}")

                # 显示一些统计信息
                for precursor in self.precursors:
                    # 在排放数据中查找对应的列名
                    emission_cols = [col for col in sample_data.columns if precursor.lower() in col.lower()]
                    if emission_cols:
                        col_name = emission_cols[0]  # 取第一个匹配的列
                        total_emission = sample_data[col_name].sum()
                        print(f"  {sample_sector}-{precursor}: 总排放量 {total_emission:.2f} t")

        except Exception as e:
            print(f"加载基准排放数据失败: {e}")
            self.base_emissions = {}

    def _cache_base_emissions_by_province(self):
        """
        ✅ 性能优化：预计算并缓存各省份的基准排放量

        将每个省份每个行业每个污染物的基准排放量预先计算并缓存，
        避免每次调用_get_base_emission时重复查询DataFrame
        """
        self.base_emission_cache = {}

        if not hasattr(self, 'base_emissions') or not self.base_emissions:
            print("⚠️ 基准排放数据未加载，无法缓存")
            return

        if not hasattr(self, 'province_to_grids') or not self.province_to_grids:
            print("⚠️ 省份网格映射未加载，无法缓存基准排放")
            return

        try:
            for province_name in self.province_names:
                self.base_emission_cache[province_name] = {}
                province_grids = self.province_to_grids.get(province_name, np.array([]))

                for sector in self.sectors:
                    self.base_emission_cache[province_name][sector] = {}

                    if sector not in self.base_emissions:
                        continue

                    sector_data = self.base_emissions[sector]

                    for precursor in self.precursors:
                        # 查找对应的污染物列
                        precursor_cols = []
                        for col in sector_data.columns:
                            if precursor.lower() == 'pm25' and 'pm25' in col.lower():
                                precursor_cols.append(col)
                            elif precursor.lower() in col.lower():
                                precursor_cols.append(col)

                        if not precursor_cols:
                            self.base_emission_cache[province_name][sector][precursor] = 0.0
                            continue

                        emission_col = precursor_cols[0]

                        # 向量化计算该省份的总排放量
                        valid_grids = province_grids[(province_grids >= 0) & (province_grids < len(sector_data))]
                        if len(valid_grids) > 0:
                            try:
                                emissions = sector_data.iloc[valid_grids][emission_col].fillna(0).values
                                total_emission = float(np.sum(emissions))
                            except:
                                total_emission = 0.0
                        else:
                            total_emission = 0.0

                        self.base_emission_cache[province_name][sector][precursor] = max(total_emission, 0.0)

            print(f"✅ 已缓存 {len(self.base_emission_cache)} 个省份的基准排放数据")

        except Exception as e:
            print(f"❌ 缓存基准排放数据失败: {e}")
            self.base_emission_cache = {}

    def _setup_default_cost_parameters(self):
        """设置默认成本参数（当无法加载真实数据时使用）"""
        print("设置默认成本参数...")

        # 默认成本系数
        default_coeffs = {
            'NOx': {'a': 8.0, 'b': 10.0},
            'SO2': {'a': 5.0, 'b': 6.0},
            'NH3': {'a': 4.0, 'b': 5.0},
            'VOC': {'a': 12.0, 'b': 15.0},
            'PM25': {'a': 6.0, 'b': 8.0}
        }

        # 为所有省份设置相同的默认系数
        self.cost_coefficients = {}
        province_names = ['AH', 'BJ', 'CQ', 'FJ', 'GD', 'GS', 'GX', 'GZ', 'HA', 'HB',
                          'HLJ', 'HN', 'HUB', 'HUN', 'JL', 'JS', 'JX', 'LN', 'NMG', 'NX',
                          'QH', 'SC', 'SD', 'SH', 'SI', 'SX', 'TJ', 'XJ', 'XZ', 'YN', 'ZJ']

        for province in province_names:
            self.cost_coefficients[province] = default_coeffs.copy()

        print("默认成本参数设置完成")

    def calculate_real_emission_cost(self, province_name, reduction_factors):
        """
        计算真实的排放减排成本

        使用成本函数：y = a * 基准排放 * 减排率² + b * 基准排放 * 减排率
        其中：
        - y: 成本（万元）
        - a, b: 省份和污染物特定的成本系数
        - 基准排放: 该省份该污染物的基准排放量（吨）
        - 减排率: 减排百分比（如0.36表示减排36%）

        Args:
            province_name: 省份名称
            reduction_factors: 25个减排因子 (5污染物 × 5行业)

        Returns:
            tuple: (总成本(万元), 成本详情字典)
        """
        total_cost = 0
        cost_details = {}

        # 获取该省份的成本系数
        if province_name not in self.cost_coefficients:
            print(f"警告: 省份 {province_name} 的成本系数不存在，跳过成本计算")
            return 0.0, {}

        province_cost_coeffs = self.cost_coefficients[province_name]

        # 遍历5个污染物和5个行业
        for precursor_idx, precursor in enumerate(self.precursors):
            if precursor not in province_cost_coeffs:
                continue

            precursor_cost = 0
            precursor_details = {}

            for sector_idx, sector in enumerate(self.sectors):
                # 计算减排因子索引
                factor_idx = precursor_idx * len(self.sectors) + sector_idx
                reduction_factor = reduction_factors[factor_idx]

                # 计算减排率 (1 - 减排因子)
                reduction_rate = 1 - reduction_factor

                # 获取该省份该污染物的基准排放量
                base_emission = self._get_base_emission(province_name, sector, precursor)

                # 计算减排量 (t)
                reduction_amount = base_emission * reduction_rate

                if reduction_amount > 0:
                    # 使用成本函数 y = a * 基准排放 * 减排率² + b * 基准排放 * 减排率
                    a = province_cost_coeffs[precursor]['a']
                    b = province_cost_coeffs[precursor]['b']

                    # 成本函数：y = a * 基准排放 * 减排率² + b * 基准排放 * 减排率
                    # 其中减排率是百分比形式（如0.36表示减排36%）
                    sector_cost = a * base_emission * (reduction_rate ** 2) + b * base_emission * reduction_rate

                    precursor_cost += sector_cost
                    precursor_details[sector] = {
                        'base_emission': base_emission,
                        'reduction_amount': reduction_amount,
                        'reduction_rate': reduction_rate,
                        'cost': sector_cost
                    }

            total_cost += precursor_cost
            cost_details[precursor] = {
                'total_cost': precursor_cost,
                'sectors': precursor_details
            }

        return total_cost, cost_details

    def _get_base_emission(self, province_name, sector, precursor):
        """
        获取指定省份、行业、污染物的基准排放量（使用缓存加速）

        Args:
            province_name: 省份名称
            sector: 行业名称
            precursor: 污染物名称

        Returns:
            基准排放量 (t)
        """
        # ✅ 优先使用缓存（快速路径）
        if hasattr(self, 'base_emission_cache') and province_name in self.base_emission_cache:
            province_cache = self.base_emission_cache[province_name]
            if sector in province_cache and precursor in province_cache[sector]:
                return province_cache[sector][precursor]

        # ⚠️ 备用路径：如果没有缓存，使用原始方法
        if sector not in self.base_emissions:
            return 0.0

        sector_data = self.base_emissions[sector]

        if not hasattr(self, 'province_to_grids') or province_name not in self.province_to_grids:
            return 0.0

        province_grids = self.province_to_grids[province_name]

        # 在排放数据中查找对应的污染物列
        precursor_cols = []
        for col in sector_data.columns:
            if precursor.lower() == 'pm25' and 'pm25' in col.lower():
                precursor_cols.append(col)
            elif precursor.lower() in col.lower():
                precursor_cols.append(col)

        if not precursor_cols:
            return 0.0

        emission_col = precursor_cols[0]

        # 向量化计算
        valid_grids = province_grids[(province_grids >= 0) & (province_grids < len(sector_data))]
        if len(valid_grids) > 0:
            try:
                emissions = sector_data.iloc[valid_grids][emission_col].fillna(0).values
                total_emission = float(np.sum(emissions))
            except:
                total_emission = 0.0
        else:
            total_emission = 0.0

        return max(total_emission, 0.0)

    def _load_unet_model(self, model_path, scaler_path):
        """加载U-Net模型（改进版 - 更好的错误处理和数据类型兼容性）"""
        unet_model_path = os.path.join(model_path, "PM25_TOT_fold_3.h5")
        unet_scaler_path = os.path.join(model_path, "scalers.pkl")

        if not os.path.exists(unet_model_path):
            raise FileNotFoundError(f"未找到U-Net模型文件: {unet_model_path}")

        try:
            print("🔄 加载U-Net模型...")

            # 定义自定义对象
            custom_objects = {
                'huber_loss': huber_loss,
                'relative_loss': relative_loss,
                'r2_metric': r2_metric,
                'rmse_metric': rmse_metric
            }

            # 🔧 尝试加载模型（带编译）
            try:
                self.unet_model = load_model(unet_model_path, custom_objects=custom_objects)
                print("✅ U-Net模型加载成功（带编译）")
            except Exception as e1:
                print(f"⚠️ 带编译加载失败: {e1}")
                print("🔄 尝试不带编译加载...")

                # 🔧 尝试不带编译加载
                try:
                    self.unet_model = load_model(unet_model_path, custom_objects=custom_objects, compile=False)
                    print("✅ U-Net模型加载成功（不带编译）")
                except Exception as e2:
                    print(f"⚠️ 不带编译加载也失败: {e2}")
                    raise RuntimeError(f"U-Net模型加载完全失败: {e2}")

            # 🔧 检查模型信息
            print(f"模型输入形状: {self.unet_model.input_shape}")
            print(f"模型输出形状: {self.unet_model.output_shape}")

            # 检查数据类型策略
            if hasattr(self.unet_model, 'dtype_policy'):
                policy = self.unet_model.dtype_policy
                print(f"模型数据类型策略: {policy.name}")

                # 如果是混合精度，给出警告
                if policy.name == 'mixed_float16':
                    print("⚠️ 检测到混合精度模型，可能需要特殊的数据类型处理")

            # 🔧 加载标准化器
            if os.path.exists(unet_scaler_path):
                try:
                    with open(unet_scaler_path, 'rb') as f:
                        self.unet_scalers = pickle.load(f)
                    print("✅ U-Net标准化器加载成功")

                    # 显示标准化器信息
                    if isinstance(self.unet_scalers, dict):
                        print(f"标准化器包含: {list(self.unet_scalers.keys())}")

                except Exception as e:
                    print(f"⚠️ 标准化器加载失败: {e}")
                    print("将使用默认的标准化处理")
                    self.unet_scalers = None
            else:
                print("⚠️ 未找到标准化器文件，将使用默认处理")
                self.unet_scalers = None

        except Exception as e:
            print(f"❌ U-Net模型加载失败: {e}")
            raise RuntimeError(f"U-Net模型加载完全失败: {e}")

    def set_logger(self, logger):
        """设置日志记录器（用于性能统计输出到日志文件）"""
        self.logger = logger

    def _warmup_gpu_model(self):
        """GPU模型预热，提高首次预测速度（改进版）"""
        if len(tf.config.list_physical_devices('GPU')) == 0:
            print("CPU模式，跳过GPU预热")
            return

        try:
            print("🔥 开始GPU模型预热...")

            # 🔧 检测模型的数据类型策略
            model_dtype = tf.float32  # 默认
            if hasattr(self.unet_model, 'dtype_policy'):
                policy = self.unet_model.dtype_policy
                if policy.name == 'mixed_float16':
                    model_dtype = tf.float16
                    print(f"检测到混合精度模型，使用 {model_dtype}")
                else:
                    print(f"检测到数据类型策略: {policy.name}")

            # 创建虚拟输入数据
            dummy_input_shape = (1, self.padded_grid_rows, self.padded_grid_cols, self.total_input_channels)
            dummy_input = np.random.random(dummy_input_shape).astype(np.float32)

            # 🚀 性能优化：在GPU上进行预热
            with tf.device('/GPU:0'):
                dummy_tensor = tf.constant(dummy_input, dtype=model_dtype)

                # 预热模型：进行多次前向传播
                for i in range(3):
                    try:
                        _ = self.unet_model(dummy_tensor, training=False)
                        print(f"  预热轮次 {i + 1}/3 完成")
                    except Exception as e:
                        print(f"  预热轮次 {i + 1}/3 失败: {e}")
                        # 如果GPU预热失败，尝试CPU预热
                        if i == 0:  # 只在第一次失败时尝试CPU
                            print("  尝试CPU预热...")
                            with tf.device('/CPU:0'):
                                cpu_tensor = tf.constant(dummy_input, dtype=tf.float32)
                                _ = self.unet_model(cpu_tensor, training=False)
                                print("  CPU预热成功")
                        break

                print("✅ GPU模型预热完成")

        except Exception as e:
            print(f"⚠️ GPU预热失败: {e}")
            print("将继续使用未预热的模型")

            # 🔧 最后尝试：简单的CPU预热
            try:
                print("尝试简单的CPU预热...")
                dummy_input_cpu = np.random.random(
                    (1, self.padded_grid_rows, self.padded_grid_cols, self.total_input_channels)).astype(np.float32)
                _ = self.unet_model.predict(dummy_input_cpu, verbose=0)
                print("✅ CPU预热成功")
            except Exception as e2:
                print(f"⚠️ CPU预热也失败: {e2}")
                print("继续使用未预热的模型")

    def _load_data(self, base_conc_path, clean_conc_path, province_map_path):
        """加载数据文件"""
        # 加载基准浓度
        self.base_conc_df = pd.read_csv(base_conc_path)
        print(f"基准浓度数据加载成功，形状: {self.base_conc_df.shape}")

        # 加载清洁情景浓度
        try:
            self.clean_conc_df = pd.read_csv(clean_conc_path)
            print(f"清洁情景数据加载成功，形状: {self.clean_conc_df.shape}")
        except FileNotFoundError:
            print("未找到清洁情景数据，使用基准数据的10%作为替代")
            self.clean_conc_df = self.base_conc_df.copy()
            for col in self.chemical_indicators:
                if col in self.clean_conc_df.columns:
                    self.clean_conc_df[col] = self.clean_conc_df[col] * 0.1

        # 加载省份映射
        self.province_map = pd.read_csv(province_map_path)
        print(f"省份映射数据加载成功，形状: {self.province_map.shape}")

        # 转换为空间网格
        self.base_chemical_grid = self._df_to_spatial_grid(self.base_conc_df, self.chemical_indicators)
        self.clean_chemical_grid = self._df_to_spatial_grid(self.clean_conc_df, self.chemical_indicators)
        self.base_conc_grid = self._df_to_spatial_grid(self.base_conc_df, ['PM25_TOT'])

        # ========================================
        # 🚀 推理加速缓存（CPU/GPU通用）
        # ========================================
        # 说明：
        # - 预测时最重的开销不仅是U-Net前向，还包括每次构造输入/np.pad/拼接/聚合到省份
        # - 这里把“与动作无关”的化学通道提前拼接并一次性pad好，避免每次重复做
        # - 同时缓存base_conc_values，减少重复切片/类型转换
        self._chem_concat = np.concatenate(
            [self.base_chemical_grid, self.clean_chemical_grid], axis=-1
        ).astype(np.float32, copy=False)  # (H, W, C_chem)

        self._chem_padded = np.zeros(
            (self.padded_grid_rows, self.padded_grid_cols, self._chem_concat.shape[-1]),
            dtype=np.float32,
        )
        self._chem_padded[
        self.pad_top:self.pad_top + self.grid_rows,
        self.pad_left:self.pad_left + self.grid_cols,
        :
        ] = self._chem_concat

        # base_conc_values: (H, W)
        self._base_conc_values = self.base_conc_grid[:, :, 0].astype(np.float32, copy=False)

        # 设置省份映射
        self._setup_provinces()

    def _df_to_spatial_grid(self, data_df, target_columns):
        """将DataFrame转换为空间网格"""
        num_channels = len(target_columns)
        spatial_grid = np.zeros((self.grid_rows, self.grid_cols, num_channels), dtype=np.float32)

        full_df = data_df.set_index('FID').reindex(range(self.grid_rows * self.grid_cols), fill_value=0)

        for i, col_name in enumerate(target_columns):
            if col_name in full_df.columns:
                flat_data = full_df[col_name].values
                spatial_grid[:, :, i] = flat_data.reshape(self.grid_rows, self.grid_cols)

        return spatial_grid

    def _setup_provinces(self):
        """设置省份信息"""
        print("设置省份信息...")

        # 获取所有省份名称
        all_province_names = self.province_map['name'].unique()

        # 定义中国境内的省份列表（过滤掉境外地区如QT等）
        china_provinces = ['AH', 'BJ', 'CQ', 'FJ', 'GD', 'GS', 'GX', 'GZ', 'HA', 'HB',
                           'HLJ', 'HN', 'HUB', 'HUN', 'JL', 'JS', 'JX', 'LN', 'NMG', 'NX',
                           'QH', 'SC', 'SD', 'SH', 'SI', 'SX', 'TJ', 'XJ', 'XZ', 'YN', 'ZJ']

        # 只保留中国境内的省份
        self.province_names = [name for name in all_province_names if name in china_provinces]
        self.num_provinces = len(self.province_names)

        print(f"发现 {len(all_province_names)} 个地区，过滤后保留 {self.num_provinces} 个中国境内省份:")
        print(f"保留的省份: {self.province_names}")

        # 建立省份名称到网格的映射
        self.province_to_grids = {}
        # ✅ 预计算：省份网格行列索引（用于快速构建排放通道/快速聚合）
        # 这样可以把“逐grid循环”变成“按省份一次性广播赋值/索引”，显著提速
        self.province_grid_rc = {}  # {prov_name: (rows:int32[], cols:int32[])}
        self._grid_size = int(self.grid_rows * self.grid_cols)
        for province_name in self.province_names:
            province_grids = self.province_map[self.province_map['name'] == province_name]['FID'].values
            # 过滤非法FID，避免越界
            province_grids = province_grids[(province_grids >= 0) & (province_grids < self._grid_size)]
            self.province_to_grids[province_name] = province_grids
            print(f"省份 {province_name}: {len(province_grids)} 个网格")

            # 预计算行列索引
            if len(province_grids) > 0:
                rows = (province_grids // self.grid_cols).astype(np.int32, copy=False)
                cols = (province_grids % self.grid_cols).astype(np.int32, copy=False)
                self.province_grid_rc[province_name] = (rows, cols)
            else:
                self.province_grid_rc[province_name] = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    def _calculate_province_concentration(self, conc_data):
        """计算各省份的平均浓度"""
        print("计算各省份平均浓度...")
        province_conc = {}
        pm25_col = 'PM25_TOT'

        for province_name in self.province_names:
            if province_name in self.province_to_grids:
                province_grids = self.province_to_grids[province_name]
                grid_concs = []

                for grid_fid in province_grids:
                    # 查找对应的浓度值
                    conc_row = conc_data[conc_data['FID'] == grid_fid]
                    if len(conc_row) > 0:
                        grid_concs.append(conc_row[pm25_col].values[0])

                if grid_concs:
                    province_conc[province_name] = np.mean(grid_concs)
                    print(f"省份 {province_name}: {province_conc[province_name]:.2f} μg/m³")
                else:
                    province_conc[province_name] = 50.0
                    print(f"省份 {province_name}: 使用默认值 50.0 μg/m³")
            else:
                province_conc[province_name] = 50.0
                print(f"省份 {province_name}: 未找到网格，使用默认值 50.0 μg/m³")

        return province_conc

    def reset(self):
        """重置环境"""
        print(f"\n🔄 重置环境...")
        self.current_step = 0
        self.done = False

        initial_actions = np.ones((self.num_provinces, self.action_dim))  # 32个省份，每个25个动作
        self.cumulative_factors = np.ones((self.num_provinces, self.action_dim))  # 32×25的累积因子矩阵

        # 重置健康效益历史记录
        self.previous_health_benefits = {}
        self.last_step_pm25 = np.array([
            self.province_base_conc.get(prov_name, 50.0) for prov_name in self.province_names
        ])

        print(f"初始PM2.5浓度范围: [{np.min(self.last_step_pm25):.2f}, {np.max(self.last_step_pm25):.2f}] μg/m³")

        # 获取第1步的目标
        step_targets, target_ratio = self.get_step_target(0)
        print(
            f"第1步目标PM2.5浓度范围: [{np.min(list(step_targets.values())):.2f}, {np.max(list(step_targets.values())):.2f}] μg/m³ (基准的{target_ratio:.0%})")

        # 构建状态
        self.current_state = []
        for i in range(self.num_provinces):
            province_one_hot = np.zeros(self.num_provinces)
            province_one_hot[i] = 1.0

            agent_state = np.concatenate([
                initial_actions[i],
                province_one_hot
            ])
            self.current_state.append(agent_state)

        self.current_state = np.array(self.current_state)
        print(f"环境重置完成，状态形状: {self.current_state.shape}")
        return self.current_state

    def step(self, actions):
        """执行环境步骤"""
        # 🔧 修复：智能体输出的是减排率，需要转换为保留率
        # 智能体输出范围：0-1（减排率）
        # 环境需要范围：0-1（保留率）
        # 转换公式：保留率 = 1 - 减排率
        retention_actions = 1.0 - actions

        # 每步最多减排20%，最少不减排（转换为保留率：0.8-1.0）
        retention_actions = np.clip(retention_actions, 0.8, 1.0)

        # 💾 保存前一步的累积因子（用于快速减排惩罚计算）
        self.previous_cumulative_factors = self.cumulative_factors.copy() if self.cumulative_factors is not None else None

        # 💾 保存当前actions（保留用于兼容性）
        self.current_actions = actions.copy()

        # 使用保留率更新累积因子
        self.cumulative_factors = self.cumulative_factors * retention_actions
        self.cumulative_factors = np.clip(self.cumulative_factors, 0.0, 1.0)

        # 预测PM2.5浓度
        predicted_pm25 = self._predict_pm25(self.cumulative_factors)

        # ✅ 计算奖励（返回奖励值、组成部分和省份级奖励）
        reward, reward_components, province_rewards_dict = self._calculate_reward(predicted_pm25, actions=actions)

        # 更新状态
        province_encoding = np.eye(self.num_provinces)
        self.current_state = np.hstack([self.cumulative_factors, province_encoding])

        self.last_step_pm25 = predicted_pm25.copy()
        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.done = True

        # ✅ 使用_calculate_reward返回的province_rewards，确保数据一致性
        info = {
            "predicted_pm25": predicted_pm25,
            "reward_components": reward_components,
            "province_rewards": province_rewards_dict  # 直接从_calculate_reward获取
        }

        return self.current_state, reward, self.done, info

    def _predict_pm25(self, actions):
        """使用U-Net预测PM2.5浓度（优化版 - 支持性能监控和日志输出）"""
        import time
        total_start = time.time() if self.enable_perf_stats else None

        try:
            # ✅ 关键修复：不再强制CPU
            # 你的环境里 TensorFlow 是 CPU build（is_cuda_build=False）时，本质上也用不了GPU；
            # 但这里仍然改成"按可用设备选择"，保证未来换到CUDA/WSL2后能直接吃到GPU。
            try:
                device = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
            except Exception:
                device = '/CPU:0'

            with tf.device(device):
                # ✅ 性能监控：记录构建emission_channels的时间
                t0 = time.time() if self.enable_perf_stats else None
                emission_channels = self._build_emission_channels(actions)  # (H, W, 25)
                if self.enable_perf_stats and t0:
                    self.perf_stats['build_emission_time'].append(time.time() - t0)

                # ✅ 预分配padded输入，避免np.pad/np.concatenate的大开销
                X_input_padded = np.zeros(
                    (1, self.padded_grid_rows, self.padded_grid_cols, self.total_input_channels),
                    dtype=np.float32
                )
                # 排放通道放入中心区域
                X_input_padded[
                0,
                self.pad_top:self.pad_top + self.grid_rows,
                self.pad_left:self.pad_left + self.grid_cols,
                :self.num_emission_channels
                ] = emission_channels
                # 化学通道直接拷贝缓存好的padded结果
                X_input_padded[0, :, :, self.num_emission_channels:] = self._chem_padded

                # ✅ 性能监控：记录标准化的时间
                t1 = time.time() if self.enable_perf_stats else None
                X_input_standardized = self._standardize_input(X_input_padded)
                if self.enable_perf_stats and t1:
                    self.perf_stats['standardize_time'].append(time.time() - t1)

                # ✅ 性能监控：记录模型推理的时间
                t2 = time.time() if self.enable_perf_stats else None
                # 使用predict（保持兼容性）
                Y_pred_padded = self.unet_model.predict(X_input_standardized, verbose=0)
                if self.enable_perf_stats and t2:
                    self.perf_stats['inference_time'].append(time.time() - t2)

                Y_pred = Y_pred_padded[
                         0,
                         self.pad_top:self.pad_top + self.grid_rows,
                         self.pad_left:self.pad_left + self.grid_cols,
                         0
                         ]

                predicted_concentrations = Y_pred * self._base_conc_values
                predicted_concentrations = np.clip(predicted_concentrations, 0.1, 500.0)

                # ✅ 性能监控：记录聚合到省份的时间
                t3 = time.time() if self.enable_perf_stats else None
                # ✅ 快速聚合（向量化索引）
                province_concentrations = self._aggregate_to_provinces(predicted_concentrations)
                result = np.array([province_concentrations.get(prov_name, 35.0) for prov_name in self.province_names])
                if self.enable_perf_stats and t3:
                    self.perf_stats['aggregate_time'].append(time.time() - t3)

                # ✅ 性能监控：记录总时间并输出到日志（每100次输出一次）
                if self.enable_perf_stats and total_start:
                    total_time = time.time() - total_start
                    self.perf_stats['total_time'].append(total_time)
                    if len(self.perf_stats['total_time']) % 100 == 0:
                        self._print_perf_stats()

                return result

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            print("回退到基准浓度")
            return np.array([self.province_base_conc.get(prov_name, 50.0) for prov_name in self.province_names])

    def _print_perf_stats(self):
        """输出性能统计信息到日志文件（而不是控制台）"""
        if not self.enable_perf_stats:
            return

        stats = self.perf_stats
        n = len(stats['total_time'])
        if n == 0:
            return

        # 构建日志消息
        log_lines = [
            f"📊 性能统计（最近{n}次预测）:",
            f"  构建emission_channels: 平均 {np.mean(stats['build_emission_time']):.4f}s, 最大 {np.max(stats['build_emission_time']):.4f}s",
            f"  标准化: 平均 {np.mean(stats['standardize_time']):.4f}s, 最大 {np.max(stats['standardize_time']):.4f}s",
            f"  模型推理: 平均 {np.mean(stats['inference_time']):.4f}s, 最大 {np.max(stats['inference_time']):.4f}s",
            f"  聚合到省份: 平均 {np.mean(stats['aggregate_time']):.4f}s, 最大 {np.max(stats['aggregate_time']):.4f}s",
            f"  总时间: 平均 {np.mean(stats['total_time']):.4f}s, 最大 {np.max(stats['total_time']):.4f}s"
        ]

        # 计算各部分占比
        total_avg = np.mean(stats['total_time'])
        if total_avg > 0:
            log_lines.append(f"  时间占比:")
            log_lines.append(f"    推理: {np.mean(stats['inference_time']) / total_avg * 100:.1f}%")
            log_lines.append(f"    构建: {np.mean(stats['build_emission_time']) / total_avg * 100:.1f}%")
            log_lines.append(f"    标准化: {np.mean(stats['standardize_time']) / total_avg * 100:.1f}%")
            log_lines.append(f"    聚合: {np.mean(stats['aggregate_time']) / total_avg * 100:.1f}%")

        # 输出到日志（如果设置了logger）
        if self.logger:
            # 支持DetailedLogger的log方法
            if hasattr(self.logger, 'log'):
                for line in log_lines:
                    self.logger.log(line)
            # 支持标准logging模块
            elif hasattr(self.logger, 'info'):
                for line in log_lines:
                    self.logger.info(line)
            # 如果logger是文件对象
            elif hasattr(self.logger, 'write'):
                for line in log_lines:
                    self.logger.write(line + '\n')
                    self.logger.flush()

    def _build_emission_channels_batch(self, actions_batch):
        """
        批量构建排放通道（优化版：向量化并行计算）

        ========================================
        🚀 性能优化说明
        ========================================

        优化策略：
        1. 预分配内存（避免动态扩展）
        2. 向量化操作（避免嵌套循环）
        3. 批量处理所有省份和所有batch

        性能提升：
        - 原方法：31省份 × batch_size次循环
        - 优化方法：31省份次循环（向量化处理batch）
        - 加速比：约batch_size倍（对于31个省份，约31倍）

        ========================================

        Args:
            actions_batch: 批量cumulative_factors，shape = (batch_size, num_provinces, action_dim)

        Returns:
            emission_channels_batch: shape = (batch_size, grid_rows, grid_cols, num_emission_channels)
        """
        batch_size = actions_batch.shape[0]
        emission_channels_batch = np.zeros((batch_size, self.grid_rows, self.grid_cols, self.num_emission_channels),
                                           dtype=np.float32)

        # ✅ 向量化处理：对所有batch同时构建emission_channels
        # 逻辑与_build_emission_channels一致：将25个减排因子映射到25个排放通道
        for prov_idx, prov_name in enumerate(self.province_names):
            if prov_name not in getattr(self, 'province_grid_rc', {}):
                continue

            rows, cols = self.province_grid_rc[prov_name]
            if rows.size == 0:
                continue

            # 提取所有batch中该省份的actions（向量化）
            province_actions_batch = actions_batch[:, prov_idx, :].astype(np.float32, copy=False)  # (B, 25)

            # ✅ 核心加速：一次性给该省份所有网格赋值（广播到网格维度）
            # 左侧: (B, n_cells, 25), 右侧: (B, 1, 25) -> broadcast
            emission_channels_batch[:, rows, cols, :] = province_actions_batch[:, None, :]

        return emission_channels_batch

    def _predict_pm25_batch(self, actions_batch):
        """
        批量预测PM2.5浓度（优化版：支持GPU并行计算，向量化处理）

        ========================================
        🚀 性能优化说明
        ========================================

        优化策略：
        1. 批量构建emission_channels（向量化，避免循环）
        2. 使用GPU加速（如果可用）
        3. 批量预测（一次性处理所有省份）
        4. 向量化后处理（避免循环）

        性能提升：
        - 串行处理：31个省份 × 0.1秒/省份 = 3.1秒
        - 批量处理：31个省份一次性处理 = 0.3-0.5秒（GPU）或 1-2秒（CPU）
        - 加速比：6-10倍（GPU）或 1.5-3倍（CPU）

        ========================================

        Args:
            actions_batch: 批量cumulative_factors，shape = (batch_size, num_provinces, action_dim)
                          或 list of arrays，每个array shape = (num_provinces, action_dim)

        Returns:
            predicted_pm25_batch: 批量预测结果，shape = (batch_size, num_provinces)
        """
        try:
            batch_size = len(actions_batch) if isinstance(actions_batch, list) else actions_batch.shape[0]

            # ✅ 尝试使用GPU加速（如果可用）
            try:
                import tensorflow as tf
                # 方法1：使用list_physical_devices（推荐）
                gpus = tf.config.list_physical_devices('GPU')
                if len(gpus) > 0:
                    device = '/GPU:0'
                    use_gpu = True
                    # 确保GPU内存增长已启用（避免内存不足）
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except:
                        pass
                else:
                    # 方法2：尝试使用list_logical_devices（备用）
                    try:
                        logical_gpus = tf.config.list_logical_devices('GPU')
                        if len(logical_gpus) > 0:
                            device = '/GPU:0'
                            use_gpu = True
                        else:
                            device = '/CPU:0'
                            use_gpu = False
                            # 诊断信息
                            print(f"  ⚠️ GPU不可用诊断:")
                            print(f"     - 物理GPU数量: {len(gpus)}")
                            print(f"     - 逻辑GPU数量: {len(logical_gpus)}")
                            print(f"     - 使用CPU模式")
                    except Exception as e2:
                        device = '/CPU:0'
                        use_gpu = False
                        print(f"  ⚠️ GPU检测失败: {e2}，使用CPU")
            except Exception as e:
                device = '/CPU:0'
                use_gpu = False
                print(f"  ⚠️ GPU检测异常: {e}，使用CPU")

            with tf.device(device):
                # ✅ 优化：批量构建排放通道（向量化，避免循环）
                if isinstance(actions_batch, list):
                    # 转换为numpy数组
                    actions_batch_array = np.array(actions_batch)  # (batch_size, num_provinces, action_dim)
                else:
                    actions_batch_array = actions_batch

                # 批量构建emission_channels
                emission_channels_batch = self._build_emission_channels_batch(actions_batch_array)

                # ✅ 预分配padded输入，避免np.pad/np.concatenate的大开销
                X_input_padded_batch = np.zeros(
                    (batch_size, self.padded_grid_rows, self.padded_grid_cols, self.total_input_channels),
                    dtype=np.float32
                )
                # 排放通道放入中心区域
                X_input_padded_batch[
                :,
                self.pad_top:self.pad_top + self.grid_rows,
                self.pad_left:self.pad_left + self.grid_cols,
                :self.num_emission_channels
                ] = emission_channels_batch
                # 化学通道：直接广播写入
                X_input_padded_batch[:, :, :, self.num_emission_channels:] = self._chem_padded[None, :, :, :]

                # 批量标准化
                X_input_standardized = self._standardize_input_batch(X_input_padded_batch)

                # ✅ 批量预测（一次性处理所有batch，GPU加速）
                Y_pred_padded_batch = self.unet_model.predict(
                    X_input_standardized,
                    verbose=0,
                    batch_size=batch_size  # 使用batch_size确保一次性处理
                )  # (batch_size, H_padded, W_padded, 1)

                # ✅ 向量化后处理（避免循环）
                # 提取有效区域
                Y_pred_batch = Y_pred_padded_batch[:,
                               self.pad_top:self.pad_top + self.grid_rows,
                               self.pad_left:self.pad_left + self.grid_cols,
                               0]  # (batch_size, H, W)

                # 向量化计算predicted_concentrations
                predicted_concentrations_batch = (
                            Y_pred_batch * self._base_conc_values[None, :, :])  # (batch_size, H, W)
                predicted_concentrations_batch = np.clip(predicted_concentrations_batch, 0.1, 500.0)

                # ✅ 批量聚合到省份：避免“对batch逐个循环”，改为按省份向量化求均值
                predicted_pm25_batch = self._aggregate_to_provinces_batch(predicted_concentrations_batch)
                return predicted_pm25_batch  # (batch_size, num_provinces)

        except Exception as e:
            print(f"❌ 批量预测失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到逐个预测
            print("⚠️ 回退到逐个预测模式")
            predicted_pm25_batch = []
            for actions_i in actions_batch:
                pm25 = self._predict_pm25(actions_i)
                predicted_pm25_batch.append(pm25)
            return np.array(predicted_pm25_batch)

    def _standardize_input_batch(self, X_input_batch):
        """
        批量标准化输入数据（优化版：直接批量处理）

        Args:
            X_input_batch: shape = (batch_size, H, W, C)

        Returns:
            standardized_batch: shape = (batch_size, H, W, C)
        """
        if self.unet_scalers is None:
            return X_input_batch

        # ✅ _standardize_input已经支持批量处理（n_samples维度）
        # 直接调用即可，无需逐个处理
        return self._standardize_input(X_input_batch)

    def _build_emission_channels(self, actions):
        """构建排放通道（GPU优化版）"""
        emission_channels = np.zeros((self.grid_rows, self.grid_cols, self.num_emission_channels), dtype=np.float32)

        # ✅ 关键加速：使用预计算的(rows, cols)一次性赋值，避免逐grid循环
        for prov_idx, prov_name in enumerate(self.province_names):
            province_actions = np.asarray(actions[prov_idx], dtype=np.float32)
            if prov_name not in self.province_grid_rc:
                continue
            rows, cols = self.province_grid_rc[prov_name]
            if rows.size == 0:
                continue
            emission_channels[rows, cols, :] = province_actions

        return emission_channels

    def _standardize_input(self, X_input):
        """标准化输入数据（GPU优化版 - 修复数据类型问题）"""
        if self.unet_scalers is None:
            return X_input

        # 🚀 性能优化：在GPU上进行标准化操作
        X_standardized = X_input.copy()

        if 'chemical' in self.unet_scalers and X_input.shape[3] > 25:
            chemical_data = X_input[:, :, :, 25:]
            n_samples, height, width, n_chemical_features = chemical_data.shape

            # 🚀 性能优化：使用更高效的reshape操作
            chemical_data_reshaped = chemical_data.reshape(-1, n_chemical_features)

            # 🔧 修复数据类型：确保使用正确的数据类型
            try:
                # 🚀 性能优化：使用TensorFlow的log1p函数
                device_name = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
                with tf.device(device_name):
                    chemical_data_tensor = tf.constant(chemical_data_reshaped, dtype=tf.float32)
                    chemical_data_log = tf.math.log1p(tf.maximum(chemical_data_tensor, 0.0))
                    chemical_data_log_np = chemical_data_log.numpy()
            except Exception as e:
                print(f"GPU标准化失败: {e}，回退到CPU")
                # 回退到CPU numpy操作
                chemical_data_log_np = np.log1p(np.maximum(chemical_data_reshaped, 0.0))

            chemical_scaler = self.unet_scalers['chemical']
            chemical_data_standardized = chemical_scaler.transform(chemical_data_log_np)
            chemical_data_standardized = chemical_data_standardized.reshape(n_samples, height, width,
                                                                            n_chemical_features)

            X_standardized[:, :, :, 25:] = chemical_data_standardized

        return X_standardized

    def _aggregate_to_provinces(self, concentration_grid):
        """将网格级浓度聚合到省份级别"""
        province_concentrations = {}

        # ✅ 加速：使用预计算的(rows, cols)一次性索引取均值，避免逐grid循环
        grid_rc = getattr(self, 'province_grid_rc', {})
        for prov_name in self.province_names:
            if prov_name not in grid_rc:
                province_concentrations[prov_name] = 35.0
                continue
            rows, cols = grid_rc[prov_name]
            if rows.size == 0:
                province_concentrations[prov_name] = 35.0
                continue
            province_concentrations[prov_name] = float(concentration_grid[rows, cols].mean())

        return province_concentrations

    def _aggregate_to_provinces_batch(self, concentration_grid_batch):
        """
        批量将网格级浓度聚合到省份级别（用于批量反事实预测）

        Args:
            concentration_grid_batch: shape = (batch_size, H, W)

        Returns:
            predicted_pm25_batch: shape = (batch_size, num_provinces)
        """
        batch_size = concentration_grid_batch.shape[0]
        out = np.full((batch_size, self.num_provinces), 35.0, dtype=np.float32)

        grid_rc = getattr(self, 'province_grid_rc', {})
        for p_idx, prov_name in enumerate(self.province_names):
            if prov_name not in grid_rc:
                continue
            rows, cols = grid_rc[prov_name]
            if rows.size == 0:
                continue
            # (B, n_cells) -> mean over cells => (B,)
            out[:, p_idx] = concentration_grid_batch[:, rows, cols].mean(axis=1)

        return out

    def _calculate_reward(self, predicted_pm25, actions=None):
        """
        计算奖励（包含健康效益、协作奖励、竞争奖励）

        ========================================
        🎯 新的奖励机制设计说明
        ========================================

        奖励由以下部分组成：

        1. **省份基础奖励**（每个省份独立计算）：
           - PM2.5目标奖励：根据PM2.5浓度是否达标给予奖励/惩罚
           - 成本惩罚：根据减排成本给予惩罚
           - 健康效益奖励：根据健康效益给予奖励

        2. **协作奖励（区域人口加权PM2.5改善）**：
           - 对每个区域，计算区域人口加权PM2.5改善
           - 公式：区域人口加权PM2.5改善 = Σ(省份人口 × (基准PM2.5 - 当前PM2.5)) / Σ(省份人口)
           - 区域协作奖励 = 改善值 × 奖励系数（改善越大奖励越大）
           - 每个省份的协作奖励 = 区域协作奖励 / 该区域省份数量
           - 优点：不需要前向传播计算，节省计算开销；鼓励区域整体改善

        3. **排名竞争奖励（区域内排名）**：
           - 对每个区域内的省份，根据其基础效用进行排名
           - 排名越高（效用越大），奖励越高
           - 使用零和设计：区域内排名奖励总和为0（避免改变总奖励尺度）
           - 目的：鼓励省份在区域内竞争，提高整体表现

        4. **区域间博弈竞争奖励**：
           - 对各区域，根据其团队目标进行排名
           - 排名越高（团队目标越大），奖励越高
           - 使用零和设计：区域间排名奖励总和为0（避免改变总奖励尺度）
           - 区域内所有省份共享该区域的排名奖励
           - 目的：鼓励区域间竞争，提高整体表现

        ========================================

        Args:
            predicted_pm25: 预测的PM2.5浓度数组
            actions: 当前步骤的所有省份动作（保留用于兼容性，新方法不使用）

        Returns:
            final_total_reward: 最终总奖励 = 省份基础奖励总和 + 协作奖励（区域人口加权PM2.5改善）
            reward_components: 奖励组成部分字典
            province_rewards_dict: 每个省份的奖励组成部分字典
        """
        try:
            # 分别记录各个奖励组成部分
            total_target_reward = 0.0  # PM2.5目标奖励总和
            total_cost_penalty = 0.0  # 成本惩罚总和
            total_health_reward = 0.0  # 健康效益奖励总和
            total_province_reward = 0.0  # 省份总奖励（包含前三项）
            total_fairness_reward = 0.0  # 公平性奖励总和（<=0）

            all_reduction_rates = []
            total_health_benefit = 0.0

            # ✅ 新增：存储每个省份的奖励组成部分
            province_rewards_dict = {}

            # ✅ 新增：缓存健康效益和成本数据（用于省份博弈奖励，矩阵计算）
            health_benefits_array = np.zeros(len(self.province_names), dtype=np.float32)
            costs_array = np.zeros(len(self.province_names), dtype=np.float32)

            # 获取当前步骤的目标
            step_targets, target_ratio = self.get_step_target(self.current_step)

            for i, province_name in enumerate(self.province_names):
                current = predicted_pm25[i]
                base = self.province_base_conc[province_name]
                target = step_targets[province_name]

                # 📈 PM2.5目标奖励：改进的三段式奖励（更宽松、更合理）
                # 1. 未达标：轻微惩罚（鼓励达标）
                # 2. 达标区间：给予奖励，越接近目标奖励越高
                # 3. 过度减排：温和惩罚，避免过度减排但不至于太严厉
                if current > target:
                    # 未达标：轻微惩罚，惩罚程度与偏离程度相关
                    deviation_ratio = (current - target) / target
                    target_reward = -10.0 * deviation_ratio  # 轻微惩罚
                elif current >= target * 0.8:
                    # 达标区间（目标值的80%-100%）：给予奖励
                    # 扩大达标区间，从90%降到80%，给予更多探索空间
                    if current >= target * 0.9:
                        # 接近目标（90%-100%）：奖励更高
                        progress = (current - target * 0.9) / (target - target * 0.9)  # 0-1之间
                        target_reward = 50.0 * (1.0 - progress)  # 提高奖励强度，在目标处奖励最高
                    else:
                        # 一般达标（80%-90%）：给予基础奖励
                        progress = (current - target * 0.8) / (target * 0.1)  # 0-1之间
                        target_reward = 25.0 * progress  # 提高基础奖励强度
                else:
                    # 过度减排（低于目标的80%）：温和惩罚
                    # 降低惩罚系数，从-50降到-20，避免过于严厉
                    over_reduction = (target * 0.8 - current) / target
                    target_reward = -20.0 * over_reduction  # 过度减排惩罚

                # 💰 成本惩罚（降低惩罚强度）
                real_cost, cost_details = self.calculate_real_emission_cost(province_name, self.cumulative_factors[i])
                cost_penalty = -real_cost * 0.5 / 20000  # 成本惩罚（万元），从5降到0.5，大幅降低惩罚强度

                # ✅ 缓存成本数据（用于省份博弈奖励）
                costs_array[i] = real_cost  # 成本（万元）

                # 🏥 健康效益奖励
                health_benefit = 0.0
                avoided_deaths = 0.0
                if self.health_calculator is not None:
                    try:
                        health_benefit, avoided_deaths = self.health_calculator.calculate_health_benefit_by_province(
                            baseline_pm25=base,
                            improved_pm25=current,
                            province_name=province_name
                        )

                        # 使用累积健康效益作为奖励（直接按当前累计值计入）
                        health_reward = health_benefit / 100000
                        total_health_benefit += health_benefit

                        # ✅ 缓存健康效益数据（用于省份博弈奖励）
                        health_benefits_array[i] = health_benefit  # 健康效益（万元）
                    except Exception as e:
                        print(f"⚠️ 计算{province_name}健康效益失败: {e}")
                        health_reward = 0.0
                        health_benefits_array[i] = 0.0
                else:
                    health_reward = 0.0
                    health_benefits_array[i] = 0.0

                # 🏆 省份总奖励 = PM2.5目标奖励 + 成本惩罚 + 健康效益奖励
                # 现在健康奖励使用累积奖励形式
                province_reward = target_reward + cost_penalty + health_reward

                # ✅ 存储该省份的奖励组成部分
                province_rewards_dict[province_name] = {
                    'target': target_reward,
                    'cost': cost_penalty,
                    'health': health_reward,
                    'fairness': 0.0,  # 先占位，后面统一计算并写入
                    'total': province_reward
                }

                # 分别累加各个组成部分
                total_target_reward += target_reward
                total_cost_penalty += cost_penalty
                total_health_reward += health_reward
                total_province_reward += province_reward

                # 记录减排率（用于统计）
                pm25_improvement_rate = (base - current) / base
                all_reduction_rates.append(pm25_improvement_rate)

                # 📋 显示详细信息（精简版）
                print(f"\n{province_name:>4}:")
                print(
                    f"  📈 PM2.5: {base:6.1f} → {current:6.1f} → 目标{target:6.1f} | 改善{pm25_improvement_rate * 100:5.1f}%")
                print(f"  💰 真实减排成本: {real_cost:8.1f} 万元")
                if self.health_calculator is not None and health_benefit > 0:
                    print(f"  🏥 健康效益: {health_benefit:8.1f} 万元 (避免死亡 {avoided_deaths:.2f} 人)")
                print(f"  🎯 奖励分解:")
                if current > target:
                    print(f"    PM2.5目标奖励: {target_reward:6.2f} (未达标，无惩罚)")
                elif current >= target * 0.9:
                    print(f"    PM2.5目标奖励: {target_reward:6.2f} (达标奖励)")
                else:
                    print(f"    PM2.5目标奖励: {target_reward:6.2f} (过度减排惩罚)")
                print(f"    成本惩罚: {cost_penalty:6.2f}")
                if self.health_calculator is not None:
                    print(f"    健康效益奖励: {health_reward:6.2f}")
                print(f"    省份总奖励: {province_reward:+.1f}")

                # 🔬 添加该省份的物种减排信息
                if hasattr(self, 'cumulative_factors') and self.cumulative_factors is not None:
                    # 计算当前步骤的减排动作（相对于上一步的变化）
                    if hasattr(self, 'previous_cumulative_factors') and self.previous_cumulative_factors is not None:
                        # 计算单步减排率
                        step_reduction_rates = (self.previous_cumulative_factors[i] - self.cumulative_factors[i]) / \
                                               self.previous_cumulative_factors[i]
                        # 计算累积减排率
                        cumulative_reduction_rates = 1.0 - self.cumulative_factors[i]
                    else:
                        # 如果没有上一步数据，说明这是第一步
                        if hasattr(self, 'current_step') and self.current_step == 0:
                            # 第一步：单步减排率等于累积减排率
                            step_reduction_rates = 1.0 - self.cumulative_factors[i]
                            cumulative_reduction_rates = 1.0 - self.cumulative_factors[i]
                        else:
                            # 其他情况：使用累积减排率（作为近似）
                            step_reduction_rates = 1.0 - self.cumulative_factors[i]
                            cumulative_reduction_rates = 1.0 - self.cumulative_factors[i]

                    # 分析该省份的物种减排情况
                    province_analysis = self._analyze_province_species_reduction(
                        step_reduction_rates, province_name, i, cumulative_reduction_rates
                    )
                    print(province_analysis)

            # 🌍 计算区域协作与竞争奖励（新方法：区域人口加权PM2.5改善 + 排名竞争 + 省份博弈奖励）
            coordination_reward, coordination_components = self._compute_region_coordination_reward(
                predicted_pm25, all_reduction_rates,
                actions=None,  # 不再需要actions
                step_targets=step_targets,
                health_benefits_array=health_benefits_array,  # ✅ 传递健康效益数据
                costs_array=costs_array  # ✅ 传递成本数据
            )

            # ⚖️ 计算公平性奖励（责任份额 vs 成本份额）
            fairness_total_reward, fairness_per_province, fairness_diag = self._compute_fairness_reward(
                predicted_pm25=predicted_pm25,
                costs_array=costs_array
            )
            total_fairness_reward = fairness_total_reward

            # ✅ 将协作和竞争奖励加入到每个省份的奖励中
            # 协作奖励：每个省份的协作奖励 = 区域协作奖励 / 该区域省份数量
            # 竞争奖励：排名奖励（零和设计）和省份博弈奖励（中央拨款100）

            # ✅ 安全获取并转换奖励数组
            # ranking_rewards 和 province_game_rewards 已经是 numpy 数组
            # province_coordination_rewards 是字典

            # 1. 获取 ranking_rewards（已经是 numpy 数组）
            ranking_rewards = coordination_components.get('ranking_rewards', None)
            if ranking_rewards is None:
                ranking_rewards_array = np.zeros(len(self.province_names), dtype=np.float32)
            elif isinstance(ranking_rewards, np.ndarray):
                ranking_rewards_array = ranking_rewards
            elif isinstance(ranking_rewards, dict):
                ranking_rewards_array = np.array([ranking_rewards.get(i, 0.0) for i in range(len(self.province_names))],
                                                 dtype=np.float32)
            else:
                ranking_rewards_array = np.zeros(len(self.province_names), dtype=np.float32)

            # 2. 获取 province_game_rewards（已经是 numpy 数组）
            province_game_rewards = coordination_components.get('province_game_rewards', None)
            if province_game_rewards is None:
                province_game_rewards_array = np.zeros(len(self.province_names), dtype=np.float32)
            elif isinstance(province_game_rewards, np.ndarray):
                province_game_rewards_array = province_game_rewards
            elif isinstance(province_game_rewards, dict):
                province_game_rewards_array = np.array(
                    [province_game_rewards.get(i, 0.0) for i in range(len(self.province_names))], dtype=np.float32)
            else:
                province_game_rewards_array = np.zeros(len(self.province_names), dtype=np.float32)

            # 3. 获取 province_coordination_rewards（是字典）
            province_coordination_rewards = coordination_components.get('province_coordination_rewards', None)
            if province_coordination_rewards is None:
                coordination_rewards_array = np.zeros(len(self.province_names), dtype=np.float32)
            elif isinstance(province_coordination_rewards, np.ndarray):
                coordination_rewards_array = province_coordination_rewards
            elif isinstance(province_coordination_rewards, dict):
                coordination_rewards_array = np.array(
                    [province_coordination_rewards.get(i, 0.0) for i in range(len(self.province_names))],
                    dtype=np.float32)
            else:
                coordination_rewards_array = np.zeros(len(self.province_names), dtype=np.float32)

            # ✅ 向量化计算总协作竞争奖励
            total_coordination_competition_reward_array = (
                    coordination_rewards_array +
                    ranking_rewards_array +
                    province_game_rewards_array
            )
            total_coordination_competition_reward = np.sum(total_coordination_competition_reward_array)

            # ✅ 更新省份奖励字典
            for province_idx, province_name in enumerate(self.province_names):
                if province_name in province_rewards_dict:
                    coordination_reward_province = coordination_rewards_array[province_idx]
                    ranking_reward = ranking_rewards_array[province_idx]
                    game_reward = province_game_rewards_array[province_idx]
                    fairness_reward_province = float(fairness_per_province[province_idx]) if fairness_per_province is not None else 0.0

                    # 记录到省份奖励字典
                    province_rewards_dict[province_name]['coordination'] = float(coordination_reward_province)
                    province_rewards_dict[province_name]['ranking'] = float(ranking_reward)
                    province_rewards_dict[province_name]['game'] = float(game_reward)
                    province_rewards_dict[province_name]['competition'] = float(ranking_reward + game_reward)  # 总竞争奖励
                    province_rewards_dict[province_name]['fairness'] = fairness_reward_province

                    # 更新省份总奖励（包含协作和竞争奖励）
                    province_rewards_dict[province_name]['total'] += float(total_coordination_competition_reward_array[province_idx])
                    province_rewards_dict[province_name]['total'] += fairness_reward_province

            # 📊 总体统计（清晰版）
            avg_improvement = np.mean(all_reduction_rates)

            # 总奖励 = 省份基础奖励 + 协作奖励（区域人口加权PM2.5改善）+ 竞争奖励（排名+区域间）
            # 注意：竞争奖励总和为0（零和），所以总奖励不变，但每个省份的奖励不同
            final_total_reward = total_province_reward + total_coordination_competition_reward + total_fairness_reward
            # 由于竞争奖励是零和的，total_coordination_competition_reward - coordination_reward应该接近0
            # 所以 final_total_reward ≈ total_province_reward + coordination_reward

            print(f"\n📊 总体统计:")
            print(f"  📈 平均PM2.5改善率: {avg_improvement:.1%}")
            print(f"\n  🎯 奖励组成:")
            print(f"    PM2.5目标奖励总和: {total_target_reward:+.2f}")
            print(f"    成本惩罚总和: {total_cost_penalty:+.2f}")
            if self.health_calculator is not None:
                print(f"    健康效益奖励总和: {total_health_reward:+.2f}")
            else:
                print(f"    健康效益奖励总和: 0.00 (未启用)")
            print(f"    区域协作奖励（人口加权PM2.5改善）: {coordination_reward:+.2f}")
            print(f"    公平性奖励（责任份额 vs 成本份额）: {total_fairness_reward:+.2f}")
            if coordination_components is not None and isinstance(coordination_components, dict):
                ranking_sum = np.sum(ranking_rewards_array) if 'ranking_rewards' in coordination_components else 0.0
                game_sum = np.sum(
                    province_game_rewards_array) if 'province_game_rewards' in coordination_components else 0.0
                print(f"    排名竞争奖励总和: {ranking_sum:+.2f} (零和)")
                print(f"    省份博弈奖励总和: {game_sum:+.2f} (中央拨款50)")
            print(f"\n  🏆 省份基础奖励总和: {total_province_reward:+.2f}")
            print(f"  🏆 最终总奖励: {final_total_reward:+.2f}")

            # ✅ 在info中返回奖励组成部分，方便日志记录
            reward_components = {
                'total_target_reward': total_target_reward,
                'total_cost_penalty': total_cost_penalty,
                'total_health_reward': total_health_reward,
                'coordination_reward': coordination_reward,  # 总协作奖励（区域人口加权PM2.5改善之和）
                'total_fairness_reward': total_fairness_reward,
                'fairness_diagnostics': fairness_diag,
                'total_province_reward': total_province_reward,
                'final_total_reward': final_total_reward,
                # ✅ 新增：协作与竞争奖励的详细组成部分
                'coordination_components': coordination_components if 'coordination_components' in locals() else {}
            }

            # ✅ 返回奖励值、组成部分和省份级奖励
            return final_total_reward, reward_components, province_rewards_dict

        except Exception as e:
            print(f"❌ _calculate_reward函数执行失败: {e}")
            import traceback
            print("详细错误堆栈:")
            traceback.print_exc()
            print(f"错误类型: {type(e).__name__}")
            if hasattr(e, '__traceback__') and e.__traceback__ is not None:
                print(f"错误位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")

            # 返回默认值，避免程序崩溃
            default_reward_components = {
                'total_target_reward': 0.0,
                'total_cost_penalty': 0.0,
                'total_health_reward': 0.0,
                'coordination_reward': 0.0,
                'total_province_reward': 0.0,
                'final_total_reward': 0.0,
                'coordination_components': {}
            }
            default_province_rewards = {prov: {'target': 0.0, 'cost': 0.0, 'health': 0.0, 'total': 0.0}
                                        for prov in self.province_names}
            return 0.0, default_reward_components, default_province_rewards

    def _compute_province_base_utility(self, predicted_pm25, province_idx, step_targets):
        """
        计算单个省份的基础效用（不含协作/竞争项）

        基础效用 = PM2.5目标奖励 + 成本惩罚 + 健康效益奖励

        Args:
            predicted_pm25: 预测的PM2.5浓度数组
            province_idx: 省份索引
            step_targets: 当前步骤的目标浓度字典

        Returns:
            base_utility: 省份基础效用值
        """
        province_name = self.province_names[province_idx]
        current = predicted_pm25[province_idx]
        base = self.province_base_conc[province_name]
        target = step_targets[province_name]

        # 📈 PM2.5目标奖励（与_calculate_reward中的逻辑一致）
        if current > target:
            deviation_ratio = (current - target) / target
            target_reward = -10.0 * deviation_ratio
        elif current >= target * 0.8:
            if current >= target * 0.9:
                progress = (current - target * 0.9) / (target - target * 0.9)
                target_reward = 50.0 * (1.0 - progress)
            else:
                progress = (current - target * 0.8) / (target * 0.1)
                target_reward = 25.0 * progress
        else:
            over_reduction = (target * 0.8 - current) / target
            target_reward = -20.0 * over_reduction

        # 💰 成本惩罚
        real_cost, _ = self.calculate_real_emission_cost(province_name, self.cumulative_factors[province_idx])
        cost_penalty = -real_cost * 0.5 / 100000

        # 🏥 健康效益奖励
        health_reward = 0.0
        if self.health_calculator is not None:
            try:
                health_benefit, _ = self.health_calculator.calculate_health_benefit_by_province(
                    baseline_pm25=base,
                    improved_pm25=current,
                    province_name=province_name
                )
                health_reward = health_benefit / 100000
            except Exception as e:
                health_reward = 0.0

        # 基础效用 = 目标奖励 + 成本惩罚 + 健康效益
        base_utility = target_reward + cost_penalty + health_reward
        return base_utility

    def _compute_region_team_goal(self, predicted_pm25, region_id, step_targets):
        """
        计算区域团队目标 G_r

        区域团队目标 = 区域内所有省份的基础效用之和

        Args:
            predicted_pm25: 预测的PM2.5浓度数组
            region_id: 区域ID
            step_targets: 当前步骤的目标浓度字典

        Returns:
            team_goal: 区域团队目标值
        """
        if not hasattr(self, 'region_provinces') or region_id not in self.region_provinces:
            return 0.0

        region_provinces = self.region_provinces[region_id]
        team_goal = 0.0

        for province_name in region_provinces:
            if province_name in self.province_names:
                province_idx = self.province_names.index(province_name)
                base_utility = self._compute_province_base_utility(predicted_pm25, province_idx, step_targets)
                team_goal += base_utility

        return team_goal

    def _compute_ranking_competition_reward(self, predicted_pm25, step_targets):
        """
        计算排名竞争奖励（区域内排名）- 优化版：使用PM2.5改善率，矩阵计算

        对每个区域内的省份，根据PM2.5浓度改善率进行排名：
        - 改善率 = (基准PM2.5 - 当前PM2.5) / 基准PM2.5
        - 排名越高（改善率越大），奖励越高
        - 使用零和设计：区域内排名奖励总和为0（避免改变总奖励尺度）

        Args:
            predicted_pm25: 预测的PM2.5浓度数组，shape=(num_provinces,)
            step_targets: 当前步骤的目标浓度字典（保留用于兼容性）

        Returns:
            ranking_rewards: 每个省份的排名奖励数组，shape=(num_provinces,)
        """
        if not hasattr(self, 'region_provinces') or not self.region_provinces:
            return np.zeros(len(self.province_names), dtype=np.float32)

        # ✅ 矩阵计算：一次性计算所有省份的PM2.5改善率
        # 改善率 = (基准PM2.5 - 当前PM2.5) / 基准PM2.5
        improvement_rates = (self.province_base_conc_array - predicted_pm25) / (self.province_base_conc_array + 1e-8)
        improvement_rates = np.clip(improvement_rates, -1.0, 1.0)  # 限制范围

        # ✅ 初始化排名奖励数组
        ranking_rewards = np.zeros(len(self.province_names), dtype=np.float32)

        print(f"\n🏆 排名竞争分析（区域内排名，基于PM2.5改善率）:")

        # ✅ 使用numpy加速：对每个区域进行向量化处理
        for region_id, region_provinces in self.region_provinces.items():
            if len(region_provinces) < 2:
                continue  # 单个省份的区域不计算排名

            # ✅ 快速获取区域内省份索引（向量化）
            region_province_indices = np.array([
                self.province_names.index(prov_name)
                for prov_name in region_provinces
                if prov_name in self.province_names
            ], dtype=np.int32)

            if len(region_province_indices) < 2:
                continue

            # ✅ 提取区域内省份的改善率（向量化索引）
            region_improvement_rates = improvement_rates[region_province_indices]

            # ✅ 使用numpy的argsort进行快速排序（从高到低）
            sorted_indices = np.argsort(region_improvement_rates)[::-1]  # 降序排列

            # ✅ 计算排名奖励（零和设计）- 向量化
            num_provinces = len(region_province_indices)
            ranking_scores = np.linspace(1.0, -1.0, num_provinces, dtype=np.float32)
            ranking_scores = ranking_scores - np.mean(ranking_scores)  # 确保总和为0
            ranking_coefficient = 5.0  # 可调整：控制排名奖励的强度
            ranking_scores = ranking_scores * ranking_coefficient

            # ✅ 向量化分配排名奖励
            ranking_rewards[region_province_indices[sorted_indices]] = ranking_scores

            # 输出详细信息（可选，可以注释掉以提高性能）
            for rank, original_idx in enumerate(sorted_indices):
                province_idx = region_province_indices[original_idx]
                province_name = self.province_names[province_idx]
                improvement_rate = region_improvement_rates[original_idx]
                ranking_reward = ranking_scores[rank]
                print(f"  区域{region_id} - 排名{rank + 1}: {province_name:4s} "
                      f"(改善率={improvement_rate * 100:6.2f}%, "
                      f"排名奖励={ranking_reward:+.2f})")

        return ranking_rewards

    def _compute_province_game_reward(self, health_benefits_array, costs_array):
        """
        计算省份博弈奖励（优化版：矩阵计算，快速排序）

        中央每步拨款50亿元（50奖励），根据三项指标竞争分配：
        1. 绩效：健康效益（万元）
        2. 减排效果：健康效益/成本
        3. 公平性：减排成本/本省GDP

        每项指标排名：第1名31分，最后1名1分
        总分排序后分配奖励（总和=50）：
        - 1-5名：5（每个省份）
        - 6-15名：1.25（每个省份）
        - 16-25名：0.95（每个省份）
        - 26-31名：0.5（每个省份）

        Args:
            health_benefits_array: 健康效益数组（万元），shape=(num_provinces,)
            costs_array: 减排成本数组（万元），shape=(num_provinces,)

        Returns:
            province_game_rewards: 每个省份的博弈奖励数组，shape=(num_provinces,)
        """
        num_provinces = len(self.province_names)

        # ✅ 矩阵计算：三项指标（向量化）
        # 1. 绩效：健康效益（直接使用）
        performance_scores = health_benefits_array.copy()

        # 2. 减排效果：健康效益/成本（避免除零）
        cost_effectiveness = np.divide(
            health_benefits_array,
            costs_array + 1e-6,  # 避免除零
            out=np.zeros_like(health_benefits_array),
            where=(costs_array > 1e-6)
        )

        # 3. 公平性：减排成本/GDP（避免除零）
        fairness_scores = np.divide(
            costs_array,
            self.province_gdp_array + 1e-6,  # 避免除零
            out=np.zeros_like(costs_array),
            where=(self.province_gdp_array > 1e-6)
        )

        # ✅ 快速排名：使用numpy的argsort（从高到低）
        # 每项指标排名：第1名31分，最后1名1分
        performance_ranks = np.argsort(performance_scores)[::-1]  # 降序
        cost_effectiveness_ranks = np.argsort(cost_effectiveness)[::-1]
        fairness_ranks = np.argsort(fairness_scores)[::-1]

        # ✅ 向量化分配排名分数（第1名31分，最后1名1分）
        performance_points = np.zeros(num_provinces, dtype=np.float32)
        cost_effectiveness_points = np.zeros(num_provinces, dtype=np.float32)
        fairness_points = np.zeros(num_provinces, dtype=np.float32)

        performance_points[performance_ranks] = np.linspace(31, 1, num_provinces, dtype=np.float32)
        cost_effectiveness_points[cost_effectiveness_ranks] = np.linspace(31, 1, num_provinces, dtype=np.float32)
        fairness_points[fairness_ranks] = np.linspace(31, 1, num_provinces, dtype=np.float32)

        # ✅ 计算总分（向量化）
        total_scores = performance_points + cost_effectiveness_points + fairness_points

        # ✅ 快速排序：根据总分排名（从高到低）
        sorted_indices = np.argsort(total_scores)[::-1]

        # ✅ 向量化分配奖励
        province_game_rewards = np.zeros(num_provinces, dtype=np.float32)

        # 奖励分配规则（总和正好=50，无需归一化）
        # 1-5名：每个省份5，总和=25
        # 6-15名：每个省份1.25，总和=12.5
        # 16-25名：每个省份0.95，总和=9.5
        # 26-31名：每个省份0.5，总和=3
        # 总计：25+12.5+9.5+3=50
        province_game_rewards[sorted_indices[0:5]] = 5.0  # 1-5名：5
        province_game_rewards[sorted_indices[5:15]] = 1.25  # 6-15名：1.25
        province_game_rewards[sorted_indices[15:25]] = 0.95  # 16-25名：0.95
        province_game_rewards[sorted_indices[25:]] = 0.5  # 26-31名：0.5

        # ✅ 验证总和是否为50（应该正好是50，无需归一化）
        total_reward = np.sum(province_game_rewards)
        if abs(total_reward - 50.0) > 0.01:  # 允许小的浮点误差
            print(f"⚠️ 警告：省份博弈奖励总和={total_reward:.2f}，不等于50，进行归一化")
            if total_reward > 0:
                province_game_rewards = province_game_rewards * (50.0 / total_reward)
            else:
                print(f"❌ 错误：省份博弈奖励总和为0，无法归一化")
                province_game_rewards = np.zeros(num_provinces, dtype=np.float32)

        # 输出详细信息（可选，可以注释掉以提高性能）
        print(f"\n🎮 省份博弈奖励分析（中央拨款50亿元）:")
        print(f"  总奖励分配: {np.sum(province_game_rewards):.2f} (应正好=50)")
        print(f"  前5名省份:")
        for rank in range(min(5, num_provinces)):
            province_idx = sorted_indices[rank]
            province_name = self.province_names[province_idx]
            total_score = total_scores[province_idx]
            reward = province_game_rewards[province_idx]
            print(f"    排名{rank + 1}: {province_name:4s} "
                  f"(总分={total_score:6.1f}, "
                  f"绩效={performance_points[province_idx]:5.1f}, "
                  f"减排效果={cost_effectiveness_points[province_idx]:5.1f}, "
                  f"公平性={fairness_points[province_idx]:5.1f}, "
                  f"奖励={reward:.2f})")

        return province_game_rewards

    def _compute_region_coordination_reward(self, predicted_pm25, reduction_rates, actions, step_targets,
                                            health_benefits_array=None, costs_array=None):
        """
        计算区域协作奖励（使用区域人口加权PM2.5改善方法）

        新的协作奖励设计：
        1. 区域人口加权PM2.5改善：计算每个区域的人口加权PM2.5改善，改善越大协作奖励越大
        2. 排名竞争：区域内省份根据PM2.5改善率排名，排名越高奖励越高（零和设计）
        3. 省份博弈奖励：根据三项指标（绩效、减排效果、公平性）竞争分配中央拨款

        Args:
            predicted_pm25: 预测的PM2.5浓度数组
            reduction_rates: 减排率数组（保留用于兼容性，新方法不使用）
            actions: 当前步骤的所有省份动作（保留用于兼容性，新方法不使用）
            step_targets: 当前步骤的目标浓度字典
            health_benefits_array: 健康效益数组（万元），shape=(num_provinces,)，用于省份博弈奖励
            costs_array: 减排成本数组（万元），shape=(num_provinces,)，用于省份博弈奖励

        Returns:
            coordination_reward: 总协作奖励（所有区域的协作奖励之和）
            reward_components: 奖励组成部分字典（包含区域协作奖励、排名奖励、省份博弈奖励）
        """
        if not hasattr(self, 'region_provinces') or not self.region_provinces:
            return 0.0, {
                'region_coordination_rewards': {},
                'province_coordination_rewards': {},
                'ranking_rewards': {},
                'province_game_rewards': {},
                'total_coordination_reward': 0.0
            }

        # 1. 计算区域人口加权PM2.5改善协作奖励
        region_coordination_rewards, province_coordination_rewards = self._compute_region_population_weighted_improvement(
            predicted_pm25, step_targets
        )

        # 2. ✅ 计算排名竞争奖励（区域内排名，基于PM2.5改善率）
        ranking_rewards = self._compute_ranking_competition_reward(predicted_pm25, step_targets)

        # 3. ✅ 计算省份博弈奖励（根据三项指标竞争分配中央拨款）
        # 确保health_benefits_array和costs_array是有效的数组
        try:
            if (health_benefits_array is not None and
                    costs_array is not None and
                    isinstance(health_benefits_array, np.ndarray) and
                    isinstance(costs_array, np.ndarray) and
                    len(health_benefits_array) == len(self.province_names) and
                    len(costs_array) == len(self.province_names)):
                province_game_rewards = self._compute_province_game_reward(health_benefits_array, costs_array)
            else:
                # 如果没有提供有效数据，返回零奖励
                province_game_rewards = np.zeros(len(self.province_names), dtype=np.float32)
                print(f"⚠️ 未提供有效健康效益和成本数据，省份博弈奖励为0")
                print(
                    f"   health_benefits_array类型: {type(health_benefits_array)}, 长度: {len(health_benefits_array) if hasattr(health_benefits_array, '__len__') else 'N/A'}")
                print(
                    f"   costs_array类型: {type(costs_array)}, 长度: {len(costs_array) if hasattr(costs_array, '__len__') else 'N/A'}")
        except Exception as e:
            print(f"❌ 计算省份博弈奖励时出错: {e}")
            import traceback
            traceback.print_exc()
            province_game_rewards = np.zeros(len(self.province_names), dtype=np.float32)

        # 4. 计算总协作奖励
        total_coordination_reward = sum(region_coordination_rewards.values())

        print(f"\n📊 协作与竞争奖励汇总:")
        print(f"  区域协作奖励总和: {total_coordination_reward:+.2f}")
        print(f"  排名竞争奖励总和: {np.sum(ranking_rewards):+.2f} (零和)")
        print(f"  省份博弈奖励总和: {np.sum(province_game_rewards):+.2f} (中央拨款50)")

        reward_components = {
            'region_coordination_rewards': region_coordination_rewards,
            'province_coordination_rewards': province_coordination_rewards,
            'ranking_rewards': ranking_rewards,
            'province_game_rewards': province_game_rewards,
            'total_coordination_reward': total_coordination_reward
        }

        return total_coordination_reward, reward_components

    def _compute_region_population_weighted_improvement(self, predicted_pm25, step_targets):
        """
        计算区域人口加权PM2.5改善协作奖励

        对每个区域，计算：
        1. 区域人口加权PM2.5改善 = Σ(省份人口 × (基准PM2.5 - 当前PM2.5)) / Σ(省份人口)
        2. 区域协作奖励 = 改善值 × 奖励系数（改善越大奖励越大）
        3. 每个省份的协作奖励 = 区域协作奖励 / 该区域省份数量

        Args:
            predicted_pm25: 预测的PM2.5浓度数组
            step_targets: 当前步骤的目标浓度字典

        Returns:
            region_coordination_rewards: 每个区域的协作奖励字典 {region_id: reward}
            province_coordination_rewards: 每个省份的协作奖励字典 {province_idx: reward}
        """
        region_coordination_rewards = {}
        province_coordination_rewards = {i: 0.0 for i in range(len(self.province_names))}

        # 奖励系数：每1 μg/m³的改善给予多少奖励
        improvement_reward_coefficient = 2.0  # 可以根据需要调整

        for region_id, region_provinces in self.region_provinces.items():
            total_population = 0.0
            weighted_improvement = 0.0

            # 计算区域人口加权PM2.5改善
            for province_name in region_provinces:
                if province_name not in self.province_names:
                    continue

                province_idx = self.province_names.index(province_name)
                population = self.province_population.get(province_name, 0.0)

                if population > 0:
                    # 计算该省份的PM2.5改善（基准浓度 - 当前浓度）
                    base_pm25 = self.province_base_conc.get(province_name, 50.0)
                    current_pm25 = predicted_pm25[province_idx]
                    improvement = base_pm25 - current_pm25  # 改善值（正值表示改善）

                    # 累加人口加权改善
                    weighted_improvement += population * improvement
                    total_population += population

            # 计算区域平均人口加权改善
            if total_population > 0:
                avg_weighted_improvement = weighted_improvement / total_population
            else:
                avg_weighted_improvement = 0.0

            # 计算区域协作奖励（改善越大奖励越大）
            region_reward = avg_weighted_improvement * improvement_reward_coefficient
            region_coordination_rewards[region_id] = region_reward

            # 将区域协作奖励平均分配给该区域的每个省份
            num_provinces_in_region = len([p for p in region_provinces if p in self.province_names])
            if num_provinces_in_region > 0:
                province_reward = region_reward / num_provinces_in_region
                for province_name in region_provinces:
                    if province_name in self.province_names:
                        province_idx = self.province_names.index(province_name)
                        province_coordination_rewards[province_idx] = province_reward

            print(f"  区域{region_id}: 人口加权PM2.5改善={avg_weighted_improvement:.2f} μg/m³, "
                  f"协作奖励={region_reward:+.2f}, "
                  f"省份数={num_provinces_in_region}, "
                  f"每省份协作奖励={province_reward if num_provinces_in_region > 0 else 0.0:+.2f}")

        return region_coordination_rewards, province_coordination_rewards

    def get_step_target(self, step):
        """获取当前步骤的目标浓度"""
        if step == 0:
            target_ratio = self.initial_target_ratio
        else:
            # 线性插值计算目标比例
            progress = step / (self.max_steps - 1)
            target_ratio = self.initial_target_ratio - progress * (self.initial_target_ratio - self.final_target_ratio)

        step_targets = {}
        for prov_name, base_conc in self.province_base_conc.items():
            step_targets[prov_name] = base_conc * target_ratio

        return step_targets, target_ratio

    def get_local_observations(self):
        """获取所有智能体的局部观察（增强版）"""
        local_observations = []

        for i in range(self.num_provinces):
            province_name = self.province_names[i]

            # 局部观察包括：
            # 1. 当前省份的25个累积减排因子
            # 2. 省份one-hot编码
            # 3. 省份特定特征（新增）
            province_one_hot = np.zeros(self.num_provinces)
            province_one_hot[i] = 1.0

            province_features = self.province_features.get(province_name, np.zeros(8))

            local_obs = np.concatenate([
                self.cumulative_factors[i],  # 25个累积减排因子
                province_one_hot,  # 32个省份的one-hot编码
                province_features  # 8个省份特定特征
            ])
            local_observations.append(local_obs)

        return np.array(local_observations)

    def get_global_observation(self):
        """获取全局观察（包含所有智能体的信息）"""
        # 全局观察包括：
        # 1. 所有省份的累积减排因子 (32 * 25 = 800)
        # 2. 所有省份的当前PM2.5浓度 (32)
        # 3. 所有省份的基准PM2.5浓度 (32)
        # 4. 当前步骤信息 (1)

        all_cumulative_factors = self.cumulative_factors.flatten()  # 800维
        all_current_pm25 = self.last_step_pm25  # 32维
        all_base_pm25 = np.array([self.province_base_conc[prov] for prov in self.province_names])  # 32维
        step_info = np.array([self.current_step / self.max_steps])  # 1维

        global_obs = np.concatenate([
            all_cumulative_factors,
            all_current_pm25,
            all_base_pm25,
            step_info
        ])

        return global_obs

    def get_agent_ids(self):
        """获取智能体ID的one-hot编码"""
        return np.eye(self.num_provinces)

    def _compute_province_features(self):
        """计算省份特定特征，用于增强智能体的差异化决策"""
        print("计算省份特定特征...")

        self.province_features = {}

        for i, province_name in enumerate(self.province_names):
            features = []

            # 1. 基准PM2.5浓度（标准化）
            base_conc = self.province_base_conc.get(province_name, 50.0)
            features.append(base_conc / 100.0)  # 标准化到0-1范围

            # 2. 省份经济发展水平代理指标（基于排放总量）
            total_emission = 0
            for sector in self.sectors:
                for precursor in self.precursors:
                    total_emission += self._get_base_emission(province_name, sector, precursor)
            features.append(min(total_emission / 1000000, 1.0))  # 标准化

            # 3. 成本敏感性（基于成本系数）
            if province_name in self.cost_coefficients:
                avg_cost_a = np.mean([coeffs['a'] for coeffs in self.cost_coefficients[province_name].values()])
                avg_cost_b = np.mean([coeffs['b'] for coeffs in self.cost_coefficients[province_name].values()])
                features.append(min(avg_cost_a / 20.0, 1.0))  # 标准化
                features.append(min(avg_cost_b / 30.0, 1.0))  # 标准化
            else:
                features.extend([0.5, 0.5])  # 默认值

            # 4. 地理位置特征（基于省份索引的简单编码）
            features.append(i / self.num_provinces)  # 省份索引标准化

            # 5. 污染物结构特征（不同污染物的相对比例）
            if province_name in self.cost_coefficients:
                nox_ratio = self.cost_coefficients[province_name].get('NOx', {'a': 8}).get('a', 8) / 20.0
                so2_ratio = self.cost_coefficients[province_name].get('SO2', {'a': 5}).get('a', 5) / 20.0
                features.append(min(nox_ratio, 1.0))
                features.append(min(so2_ratio, 1.0))
            else:
                features.extend([0.4, 0.25])  # 默认值

            # 6. 减排潜力指标（基于基准浓度与目标的差距）
            final_target = base_conc * self.final_target_ratio
            reduction_potential = (base_conc - final_target) / base_conc
            features.append(reduction_potential)

            # 确保特征向量长度为8
            while len(features) < 8:
                features.append(0.0)
            features = features[:8]

            self.province_features[province_name] = np.array(features, dtype=np.float32)

            print(
                f"  {province_name}: 基准浓度{base_conc:.1f}, 排放总量{total_emission:.0f}t, 减排潜力{reduction_potential:.2f}")

        print(f"省份特征计算完成，每个省份{len(features)}个特征")

    def _analyze_province_species_reduction(self, step_reduction_rates, province_name, province_index,
                                            cumulative_reduction_rates):
        """分析特定省份的物种减排情况"""
        if len(step_reduction_rates) != 25:  # 25个物种
            return f"  🔬 {province_name} 物种减排详情: 数据维度错误"

        # 物种名称映射
        species_names = [
            'TR-PM25', 'TR-SO2', 'TR-NOx', 'TR-VOC', 'TR-NH3',
            'PP-PM25', 'PP-SO2', 'PP-NOx', 'PP-VOC', 'PP-NH3',
            'IN-PM25', 'IN-SO2', 'IN-NOx', 'IN-VOC', 'IN-NH3',
            'AG-PM25', 'AG-SO2', 'AG-NOx', 'AG-VOC', 'AG-NH3',
            'RE-PM25', 'RE-SO2', 'RE-NOx', 'RE-VOC', 'RE-NH3'
        ]

        # 计算统计信息
        step_avg = np.mean(step_reduction_rates)
        step_std = np.std(step_reduction_rates)
        step_max = np.max(step_reduction_rates)
        step_min = np.min(step_reduction_rates)

        cumulative_avg = np.mean(cumulative_reduction_rates)
        cumulative_std = np.std(cumulative_reduction_rates)

        # 找出减排率最高的5个物种
        top_indices = np.argsort(step_reduction_rates)[::-1][:5]

        # 统计减排模式
        high_reduction = np.sum(step_reduction_rates > 0.15)  # 高减排（>15%）
        medium_reduction = np.sum((step_reduction_rates >= 0.05) & (step_reduction_rates <= 0.15))  # 中减排（5-15%）
        low_reduction = np.sum(step_reduction_rates < 0.05)  # 低减排（<5%）

        # 生成分析报告
        analysis = f"  🔬 {province_name} 物种减排详情:\n"
        analysis += f"    单步减排: 平均{step_avg:.1%}, 标准差{step_std:.1%}, 最大{step_max:.1%}, 最小{step_min:.1%}\n"
        analysis += f"    累积减排: 平均{cumulative_avg:.1%}, 标准差{cumulative_std:.1%}\n"
        analysis += f"    🏆 单步减排率排名 (前5个):\n"

        for rank, idx in enumerate(top_indices, 1):
            species_name = species_names[idx]
            reduction_rate = step_reduction_rates[idx]
            analysis += f"      {rank}. {species_name}: {reduction_rate:.1%}\n"

        analysis += f"    📊 减排模式: 高({high_reduction}个) 中({medium_reduction}个) 低({low_reduction}个)"

        return analysis

    def log_reduction_rates(self, episode, step, logger=None):
        """记录每一步所有省份的累积减排率和单步减排率"""
        if not hasattr(self, 'cumulative_factors') or self.cumulative_factors is None:
            return

        # 计算单步减排率和累积减排率
        step_reduction_rates = []
        cumulative_reduction_rates = []

        for i in range(self.num_provinces):
            if hasattr(self, 'previous_cumulative_factors') and self.previous_cumulative_factors is not None:
                # 计算单步减排率
                step_reduction = (self.previous_cumulative_factors[i] - self.cumulative_factors[i]) / \
                                 self.previous_cumulative_factors[i]
                step_reduction_rates.append(step_reduction)
            else:
                # 第一步：单步减排率等于累积减排率
                step_reduction = 1.0 - self.cumulative_factors[i]
                step_reduction_rates.append(step_reduction)

            # 计算累积减排率
            cumulative_reduction = 1.0 - self.cumulative_factors[i]
            cumulative_reduction_rates.append(cumulative_reduction)

        # 转换为numpy数组
        step_reduction_rates = np.array(step_reduction_rates)
        cumulative_reduction_rates = np.array(cumulative_reduction_rates)

        # 计算每个省份的平均单步减排率和累积减排率
        step_avg_by_province = np.mean(step_reduction_rates, axis=1)
        cumulative_avg_by_province = np.mean(cumulative_reduction_rates, axis=1)

        # 生成日志内容
        log_content = f"\n📊 第{episode + 1}轮第{step + 1}步减排率统计:\n"
        log_content += "省份    单步减排率    累积减排率\n"
        log_content += "-" * 40 + "\n"

        for i, province_name in enumerate(self.province_names):
            step_avg = step_avg_by_province[i] * 100  # 转换为百分比
            cumulative_avg = cumulative_avg_by_province[i] * 100  # 转换为百分比
            log_content += f"{province_name:>4}    {step_avg:6.1f}%      {cumulative_avg:6.1f}%\n"

        # 计算总体统计
        overall_step_avg = np.mean(step_avg_by_province) * 100
        overall_cumulative_avg = np.mean(cumulative_avg_by_province) * 100
        log_content += "-" * 40 + "\n"
        log_content += f"总体    {overall_step_avg:6.1f}%      {overall_cumulative_avg:6.1f}%\n"

        # 输出到控制台
        print(log_content)

        # 如果提供了logger，也写入日志文件
        if logger is not None:
            logger.log(log_content)

        return step_reduction_rates, cumulative_reduction_rates


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
