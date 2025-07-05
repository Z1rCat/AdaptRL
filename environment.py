# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any  # <--- 确保 Tuple 已导入


class MultiDistributionEnv(gym.Env):
    """
    自定义Gym环境，核心任务是比较一个固定阈值与从不同概率分布中采样的随机值。
    支持动态改变内部概率分布。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}  # Gymnasium规范

    DIST_GENERATORS = {
        'normal': lambda params: np.random.normal(params['mu'], params['sigma']),
        'uniform': lambda params: np.random.uniform(params['low'], params['high']),
        'exponential': lambda params: np.random.exponential(params['scale']),
        'poisson': lambda params: np.random.poisson(params['lam']),
        'gamma': lambda params: np.random.gamma(params['shape'], params['scale']),
        'beta': lambda params: params.get('min_val', 0) + \
                               (params.get('max_val', 1) - params.get('min_val', 0)) * \
                               np.random.beta(params['alpha'], params['beta']),
        'laplace': lambda params: np.random.laplace(params['mu'], params['b'])
    }

    def __init__(self, dist_config: Dict, verbose_level: int = 0):
        super().__init__()

        self.observation_space = spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.current_fixed_value = 10.0
        self.verbose = verbose_level

        self.current_dist_config = None
        self.dist_type = None
        self.dist_params = None

        self.set_distribution_from_config(dist_config)

        self.step_count = 0
        self.correct_guesses = 0
        self.episode_rewards = []

    def _validate_distribution(self):
        if self.dist_type not in self.DIST_GENERATORS:
            raise ValueError(f"不支持的分布类型: {self.dist_type}")

        required_params_map = {
            'normal': ['mu', 'sigma'], 'uniform': ['low', 'high'], 'exponential': ['scale'],
            'poisson': ['lam'], 'gamma': ['shape', 'scale'],
            'beta': ['alpha', 'beta'],
            'laplace': ['mu', 'b']
        }
        required_params = required_params_map.get(self.dist_type, [])
        for param_key in required_params:
            if param_key not in self.dist_params:
                raise ValueError(f"分布 '{self.dist_type}' 缺少必需参数: '{param_key}'。当前参数: {self.dist_params}")

    def set_distribution_from_config(self, dist_config: Dict):
        new_dist_type = dist_config.get('dist_type')
        new_dist_params = dist_config.get('params')

        if not new_dist_type or not isinstance(new_dist_params, dict):
            raise ValueError("分布配置无效，必须包含 'dist_type' 和 'params' (字典)。")

        self.dist_type = new_dist_type
        self.dist_params = new_dist_params.copy()

        if self.dist_type == 'beta':
            self.dist_params.setdefault('min_val', 0.0)
            self.dist_params.setdefault('max_val', 1.0)

        self._validate_distribution()
        self.current_dist_config = dist_config

        if self.verbose > 0:
            print(
                f"[环境日志] 分布已设置为: {self.current_dist_config.get('name', self.dist_type)}，参数: {self.dist_params}")

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.step_count = 0
        self.correct_guesses = 0
        self.episode_rewards = []

        observation = np.array([self.current_fixed_value], dtype=np.float32)
        info = {"current_distribution_name": self.current_dist_config.get('name',
                                                                          self.dist_type)} if self.current_dist_config else {}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.dist_type is None or self.dist_params is None:
            raise RuntimeError("环境分布未正确初始化。")

        target_sample = self.DIST_GENERATORS[self.dist_type](self.dist_params)

        correct = (action == 1 and self.current_fixed_value > target_sample) or \
                  (action == 0 and self.current_fixed_value <= target_sample)

        reward = 1.0 if correct else -0.1

        self.step_count += 1
        if correct:
            self.correct_guesses += 1
        self.episode_rewards.append(reward)

        terminated = self.step_count >= 50
        truncated = False

        observation = np.array([self.current_fixed_value], dtype=np.float32)
        info = {}
        # In environment.py, MultiDistributionEnv.step
        if terminated:
            current_episode_stats = {
                "total_steps_in_episode": self.step_count,
                "correct_rate": self.correct_guesses / self.step_count if self.step_count > 0 else 0,
                "total_reward": sum(self.episode_rewards)
            }
            info['episode_stats'] = current_episode_stats
            print(f"DEBUG ENV: Episode terminated. Stats: {current_episode_stats}")  # 打印环境生成的stats

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def set_env_verbose(self, verbose_level: int):
        self.verbose = verbose_level