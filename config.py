# config.py
from pathlib import Path
from stable_baselines3 import PPO, DQN, A2C # 需要导入算法类
import numpy as np # 用于为新归一化器提供示例参考值

ALGO_CLASS_MAP = {"PPO": PPO, "DQN": DQN, "A2C": A2C}
# --- 实验常量 ---
ALL_AVAILABLE_ALGORITHMS = ["PPO", "A2C", "DQN"]
ALL_DISTRIBUTION_CONFIGS = [
    {"name": "标准正态(μ10,σ2)", "dist_type": "normal", "params": {"mu": 10.0, "sigma": 2.0}, "threshold_ratio": 0.85},
    {"name": "正态-小幅均值改变(μ11,σ2)", "dist_type": "normal", "params": {"mu": 11.0, "sigma": 2.0}, "threshold_ratio": 0.85},
    {"name": "正态-大幅均值改变(μ14,σ2)", "dist_type": "normal", "params": {"mu": 14.0, "sigma": 2.0}, "threshold_ratio": 0.70},
    {"name": "正态-均值标准差均改变(μ12,σ4)", "dist_type": "normal", "params": {"mu": 12.0, "sigma": 4.0}, "threshold_ratio": 0.75},
    {"name": "正态-标准差极大(μ10,σ8)", "dist_type": "normal", "params": {"mu": 10.0, "sigma": 8.0}, "threshold_ratio": 0.65},
    {"name": "均匀分布(宽,0-20)", "dist_type": "uniform", "params": {"low": 0.0, "high": 20.0}, "threshold_ratio": 0.70},
    {"name": "泊松分布(高均值,λ15)", "dist_type": "poisson", "params": {"lam": 15}, "threshold_ratio": 0.60},
    {"name": "指数分布(小尺度,s5)", "dist_type": "exponential", "params": {"scale": 5.0}, "threshold_ratio": 0.70},
    {"name": "伽马分布(尖峰,shape5,scale2)", "dist_type": "gamma", "params": {"shape": 5.0, "scale": 2.0}, "threshold_ratio": 0.70},
    {"name": "拉普拉斯分布(窄,μ10,b1)", "dist_type": "laplace", "params": {"mu": 10.0, "b": 1.0}, "threshold_ratio": 0.75},
    {"name": "Beta分布(U形,0-20)", "dist_type": "beta", "params": {"alpha": 0.5, "beta": 0.5, "min_val": 0.0, "max_val": 20.0}, "threshold_ratio": 0.60},
    {"name": "Beta分布(偏左,0-20)", "dist_type": "beta", "params": {"alpha": 2, "beta": 5, "min_val": 0.0, "max_val": 20.0}, "threshold_ratio": 0.65},
]
TIMESTEPS_INITIAL_TRAIN = 60000
TIMESTEPS_RETRAIN = 40000
TIMESTEPS_FINETUNE = 15000
HPO_TIMESTEPS_CONFIG = {
    'initial': 10000, 'retrain': 8000, 'finetune': 3000, 'eval_episodes': 20
}
N_HPO_TRIALS = 50 # 调试时可以改小，例如 3-5

# --- 归一化配置 ---
DEFAULT_NORMALIZER_TYPE = "min_max" # 可选: "min_max", "z_score", "sigmoid", "robust_scaler", "quantile", "power_transform"

# 示例：假设我们有一些关于奖励、斜率、标准差指标的经验数据或期望范围
# 这些值需要根据实际数据或预跑实验来仔细设定
# 对于 QuantileNormalizer 的 reference_values，这里用随机数示意，实际应来自真实数据
np.random.seed(42) # for reproducibility of reference_values examples
sample_rewards = np.concatenate([np.random.normal(25, 10, 70), np.random.normal(5, 5, 30)]) # 假设双峰分布
sample_slopes = np.random.normal(0, 0.5, 100)
sample_std_devs = np.random.gamma(2, 3, 100)


# Min-Max 归一化器的参数配置
MIN_MAX_NORMALIZER_PARAMS = {
    "reward": {"min": -10.0, "max": 50.0}, # 扩展奖励范围
    "slope": {"min": -2.0, "max": 2.0},
    "std_dev": {"min": 0.0, "max": 25.0}
}

# Z-Score 归一化器的参数配置
Z_SCORE_NORMALIZER_PARAMS = {
    "reward": {"mean": np.mean(sample_rewards), "std": np.std(sample_rewards) if np.std(sample_rewards) > 1e-6 else 1.0},
    "slope": {"mean": np.mean(sample_slopes), "std": np.std(sample_slopes) if np.std(sample_slopes) > 1e-6 else 1.0},
    "std_dev": {"mean": np.mean(sample_std_devs), "std": np.std(sample_std_devs) if np.std(sample_std_devs) > 1e-6 else 1.0}
}

# Sigmoid 归一化器的参数配置
SIGMOID_NORMALIZER_PARAMS = {
    "reward": {"center": np.median(sample_rewards), "scale": np.std(sample_rewards) if np.std(sample_rewards) > 1e-6 else 1.0},
    "slope": {"center": 0.0, "scale": 0.5},
    "std_dev": {"center": np.median(sample_std_devs), "scale": np.std(sample_std_devs) if np.std(sample_std_devs) > 1e-6 else 1.0}
}

# 新增：RobustScaler 归一化器的参数配置
ROBUST_SCALER_NORMALIZER_PARAMS = {
    "reward": {"median": np.median(sample_rewards), "iqr": np.percentile(sample_rewards, 75) - np.percentile(sample_rewards, 25) or 1.0},
    "slope": {"median": np.median(sample_slopes), "iqr": np.percentile(sample_slopes, 75) - np.percentile(sample_slopes, 25) or 1.0},
    "std_dev": {"median": np.median(sample_std_devs), "iqr": np.percentile(sample_std_devs, 75) - np.percentile(sample_std_devs, 25) or 1.0}
}
# 确保iqr不为0，如果为0，则用1.0代替，避免除零

# 新增：QuantileNormalizer 归一化器的参数配置
QUANTILE_NORMALIZER_PARAMS = {
    "reward": {"reference_values": list(sample_rewards), "output_distribution": "uniform"}, # 或 "normal"
    "slope": {"reference_values": list(sample_slopes), "output_distribution": "uniform"},
    "std_dev": {"reference_values": list(sample_std_devs), "output_distribution": "uniform"}
}
# QuantileNormalizer 的另一个例子，输出为正态
QUANTILE_NORMALIZER_NORMAL_OUTPUT_PARAMS = {
    "reward": {"reference_values": list(sample_rewards), "output_distribution": "normal"},
    "slope": {"reference_values": list(sample_slopes), "output_distribution": "normal"},
    "std_dev": {"reference_values": list(sample_std_devs), "output_distribution": "normal"}
}


# 新增：PowerTransformNormalizer (Yeo-Johnson) 归一化器的参数配置
# Lambda值通常需要从数据中估计得到。这里用示例值。
# 例如: stats.yeojohnson_normmax(sample_rewards + 1e-6) # 加一个小数避免0，如果需要
# 对于Yeo-Johnson，不需要严格正数，但lambda的估计可能依赖于具体数据。
# 假设我们已经为每个指标预先计算/选择好了lambda值
POWER_TRANSFORM_NORMALIZER_PARAMS = {
    # 实际的lambda值应该通过对代表性数据进行拟合来确定，例如使用sklearn的PowerTransformer
    # from sklearn.preprocessing import PowerTransformer
    # pt_reward = PowerTransformer(method='yeo-johnson', standardize=False)
    # pt_reward.fit(sample_rewards.reshape(-1, 1))
    # reward_lambda = pt_reward.lambdas_[0]
    # 这里使用硬编码的示例lambda值
    "reward": {"lambda_": 0.5}, # 示例 lambda
    "slope": {"lambda_": 1.0},  # 示例 lambda (lambda=1 通常接近无变换)
    "std_dev": {"lambda_": 0.0} # 示例 lambda (lambda=0 接近对数变换)
}


ALL_NORMALIZER_CONFIGS = {
    "min_max": MIN_MAX_NORMALIZER_PARAMS,
    "z_score": Z_SCORE_NORMALIZER_PARAMS,
    "sigmoid": SIGMOID_NORMALIZER_PARAMS,
    "robust_scaler": ROBUST_SCALER_NORMALIZER_PARAMS,
    "quantile_uniform": QUANTILE_NORMALIZER_PARAMS, # 重命名以区分
    "quantile_normal": QUANTILE_NORMALIZER_NORMAL_OUTPUT_PARAMS, # 新增
    "power_transform": POWER_TRANSFORM_NORMALIZER_PARAMS,
}
# 为了命令行方便，可以只用 "quantile"，然后在代码内部决定是 uniform 还是 normal
# 或者让用户通过更复杂的配置来指定 Quantile 的输出类型。
# 为简单起见，我们这里定义了两种 quantile 类型。


FIXED_SLOPE_Y_LIMITS = (-0.3, 0.3)
FIXED_STD_DEV_Y_LIMITS = (0, 5.0)

NORMALIZED_PENALTY_SLOPE = 0.2
NORMALIZED_PENALTY_STD = 0.1

DEFAULT_HYPERPARAMS = {
    "DQN": {"learning_starts": 1000, "buffer_size": 100000, "exploration_fraction": 0.1,
            "exploration_final_eps": 0.02, "target_update_interval": 500, "learning_rate": 1e-3},
    "PPO": {"n_steps": 512, "batch_size": 64, "learning_rate": 3e-4, "gamma": 0.99},
    "A2C": {"n_steps": 5, "learning_rate": 7e-4, "ent_coef": 0.0, "gamma": 0.99}
}
FINETUNE_LEARNING_RATE = 1e-5
LOGS_DIR_NAME = "logs"
PLOTS_DIR_NAME = "plots"
HPO_STUDY_DIR_NAME = "hpo_studies"
CONVERGENCE_SLOPE_THRESHOLD_NEG = -0.05
CONVERGENCE_STD_THRESHOLD_HIGH = 15.0 # 这个原始阈值可能需要根据指标的典型范围调整
CONVERGENCE_METRICS_LAST_N_EPISODES = 30
PROJECT_ROOT_DIR = Path(__file__).resolve().parent
FIXED_THRESHOLD_FOR_COMPLEXITY = 10.0
SAMPLES_FOR_COMPLEXITY_CALCULATION = 100000
LOGS_DIR = PROJECT_ROOT_DIR / LOGS_DIR_NAME
PLOTS_DIR = PROJECT_ROOT_DIR / PLOTS_DIR_NAME
HPO_STUDY_DIR = PROJECT_ROOT_DIR / HPO_STUDY_DIR_NAME