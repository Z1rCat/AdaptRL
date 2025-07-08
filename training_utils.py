# training_utils.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any, Tuple, List, Optional
import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DQN, A2C

import optuna

from environment import MultiDistributionEnv
from config import ALGO_CLASS_MAP, DEFAULT_HYPERPARAMS  # DEFAULT_HYPERPARAMS is used here
from utils import sanitize_filename


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 0, trial: Optional[optuna.Trial] = None, eval_freq_pruning: int = 1000):
        super().__init__(verbose)
        self.episode_data: List[Dict[str, Any]] = []
        self.trial = trial
        self.eval_freq_pruning = eval_freq_pruning
        self.next_eval_step = self.eval_freq_pruning
        self.optuna_report_step_counter = 0  # Use a separate counter for optuna reports
        self.last_heartbeat_time = time.time()

    def _on_step(self) -> bool:
        # HPO Keep-alive logging (Time-based)
        current_time = time.time()
        if self.trial and (current_time - self.last_heartbeat_time) > 15:
            self.last_heartbeat_time = current_time
            print(f"HPO Trial #{self.trial.number}: Training in progress... Timestep: {self.num_timesteps}/{self.model._total_timesteps}", flush=True)

        # Check for 'dones' in locals, which is typical for VecEnvs
        # For single envs, it might be 'done'
        dones = self.locals.get('dones', np.array([self.locals.get('done', False)]))
        infos = self.locals.get('infos', [self.locals.get('info', {})])

        for env_idx in range(len(dones)):
            if dones[env_idx]:  # Episode ended for this environment
                # Monitor wrapper puts episode stats under 'episode' key in info
                # Our custom env might put it under 'episode_stats'
                monitor_stats = infos[env_idx].get('episode')
                custom_stats = infos[env_idx].get('episode_stats', {})

                if monitor_stats and 'r' in monitor_stats and 'l' in monitor_stats:
                    self.episode_data.append({
                        "episode": len(self.episode_data) + 1,
                        # Use custom stats if available, fallback to monitor stats
                        "total_steps_in_episode": custom_stats.get("total_steps_in_episode", monitor_stats['l']),
                        "correct_rate": custom_stats.get("correct_rate", np.nan),  # Monitor doesn't provide this
                        "total_reward": custom_stats.get("total_reward", monitor_stats['r'])
                    })
                # else: # If no monitor_stats, and custom_stats were also not found or incomplete
                # if self.verbose > 0 and not custom_stats: # Only log if custom_stats was also missing
                # print(f"Callback Log: Env {env_idx} episode ended. Monitor 'episode' key not found or incomplete. Info: {infos[env_idx]}")

        # Optuna Pruning Logic
        if self.trial is not None and self.num_timesteps >= self.next_eval_step:
            self.next_eval_step += self.eval_freq_pruning  # Schedule next check

            if len(self.episode_data) >= 5:  # Require at least 5 episodes for a somewhat stable mean
                # Safely extract rewards, handling potential NaNs or missing keys
                recent_rewards = [
                    d['total_reward'] for d in self.episode_data[-10:]  # Use last 10 or fewer if not enough
                    if isinstance(d.get('total_reward'), (int, float)) and not np.isnan(d['total_reward'])
                ]

                if recent_rewards:  # Only report if we have valid rewards
                    intermediate_value = float(np.mean(recent_rewards))
                    self.optuna_report_step_counter += 1  # Increment report step
                    self.trial.report(intermediate_value,
                                      self.optuna_report_step_counter)  # Use report_step_counter as step

                    if self.trial.should_prune():
                        # print(f"Trial {self.trial.number} pruned at report step {self.optuna_report_step_counter} (env steps ~{self.num_timesteps}) with value {intermediate_value:.2f}.")
                        raise optuna.TrialPruned()
                # elif self.verbose > 0:
                # print(f"Trial {self.trial.number}, EnvSteps ~{self.num_timesteps}: Not enough valid reward data for Optuna report {self.optuna_report_step_counter + 1}.")
            # elif self.verbose > 0:
            # print(f"Trial {self.trial.number}, EnvSteps ~{self.num_timesteps}: Not enough episode data ({len(self.episode_data)}) for Optuna report {self.optuna_report_step_counter + 1}.")
        return True


def train_agent(
        env_config: Dict,
        algorithm: str,
        total_timesteps: int,
        policy: str = "MlpPolicy",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        hyperparams_override: Optional[Dict[str, Any]] = None,  # This will contain 'finetune_learning_rate'
        sb3_verbose: int = 0,
        log_file_suffix: str = "training",
        trial: Optional[optuna.Trial] = None
) -> Tuple[Union[PPO, DQN, A2C, None], TrainingLoggerCallback, float]:
    # Start with default hyperparameters for the algorithm
    current_default_params = DEFAULT_HYPERPARAMS.get(algorithm.upper(), {}).copy()

    # Prepare parameters for model creation:
    # These are parameters that the SB3 model constructor actually accepts.
    model_constructor_params = current_default_params.copy()
    if hyperparams_override:
        # Only update model_constructor_params with keys that are NOT 'finetune_learning_rate'
        # or other custom keys not meant for the SB3 constructor.
        for key, value in hyperparams_override.items():
            if key != "finetune_learning_rate":  # Explicitly exclude it
                model_constructor_params[key] = value

    # For logging purposes, show all params that were considered (including custom ones)
    # print_params_for_logging = current_default_params.copy()
    # if hyperparams_override: print_params_for_logging.update(hyperparams_override)

    def _make_env():
        env = MultiDistributionEnv(dist_config=env_config, verbose_level=0)  # Keep verbose low for HPO
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([_make_env])
    model_class = ALGO_CLASS_MAP.get(algorithm.upper())

    training_duration = 0.0
    model = None

    # Initialize logger_callback early in case of early exit
    pruning_report_interval = max(500, total_timesteps // 20) if trial and total_timesteps > 0 else 1000
    logger_callback = TrainingLoggerCallback(
        verbose=0 if trial else 1,
        trial=trial,
        eval_freq_pruning=pruning_report_interval
    )

    if not model_class:
        print(f"错误: 不支持的算法 '{algorithm}'")
        return None, logger_callback, training_duration

    try:
        # Use model_constructor_params which excludes 'finetune_learning_rate'
        model = model_class(policy=policy, env=vec_env, verbose=sb3_verbose,
                            policy_kwargs=policy_kwargs or {}, **model_constructor_params)
    except Exception as e_model_create:
        print(f"创建模型 {algorithm} 失败，参数: {model_constructor_params}。错误: {e_model_create}")
        return None, logger_callback, training_duration

    # print(f"开始训练 {algorithm} (分布: {env_config.get('name', 'N/A')}), 使用参数: {print_params_for_logging}, 总步数: {total_timesteps}.")
    # if trial: print(f"  Optuna Pruning: Report interval approx every {pruning_report_interval} environment steps.")

    train_start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=logger_callback, reset_num_timesteps=True)
        training_duration = time.time() - train_start_time
        # print(f"  模型 {algorithm} 训练完成，耗时: {training_duration:.2f} 秒。") # Less verbose for HPO
    except optuna.TrialPruned:
        training_duration = time.time() - train_start_time
        # print(f"  模型 {algorithm} 训练被Optuna剪枝，耗时: {training_duration:.2f} 秒。") # Callback already prints
        return None, logger_callback, training_duration  # Model is None, return logger and duration
    except Exception as e_learn:
        training_duration = time.time() - train_start_time
        print(
            f"模型 {algorithm} 在分布 {env_config.get('name')} 上训练时发生错误: {e_learn}，耗时: {training_duration:.2f} 秒。")
        return None, logger_callback, training_duration  # Model is None or partially trained, return logger and duration

    # Save Excel log if not an HPO trial and data exists
    if logger_callback.episode_data and not trial:
        log_df = pd.DataFrame(logger_callback.episode_data)
        # Ensure critical columns for analysis exist and are not all NaN before saving
        log_df_cleaned = log_df.dropna(subset=['total_reward', 'total_steps_in_episode'], how='all')

        if not log_df_cleaned.empty:
            clean_log_suffix = sanitize_filename(log_file_suffix)
            # Use LOGS_DIR from config if available, otherwise default to "logs"
            try:
                from config import LOGS_DIR as configured_logs_dir; logs_path = configured_logs_dir
            except ImportError:
                logs_path = Path("logs")

            logs_path.mkdir(parents=True, exist_ok=True)
            excel_log_path = logs_path / f"{algorithm}_{clean_log_suffix}_episode_log.xlsx"
            try:
                log_df_cleaned.to_excel(excel_log_path, index=False)
                # print(f"训练日志已保存至: {excel_log_path}") # Less verbose for HPO context, main_experiment can log this
            except Exception as e_save_log:
                print(f"保存训练日志到 {excel_log_path} 失败: {e_save_log}")
        # elif logger_callback.verbose > 0 : # logger_callback.verbose is not set based on `self`
        # print(f"训练日志为空或所有关键行都包含NaN，不保存Excel文件。原始数据行数: {len(log_df)}")

    return model, logger_callback, training_duration


def evaluate_agent(
        env_config: Dict,
        model: Union[PPO, DQN, A2C, None],
        episodes: int = 100,
        deterministic: bool = True  # Usually True for evaluation
) -> Dict[str, Any]:
    if model is None:
        return {
            "avg_reward": -float('inf'), "std_reward": np.nan,
            "avg_correct_rate": np.nan, "rewards_raw": [], "correct_rates_raw": []
        }

    rewards_per_episode: List[float] = []
    correct_rates_per_episode: List[float] = []
    # Create a fresh environment instance for evaluation
    eval_env_instance = MultiDistributionEnv(dist_config=env_config, verbose_level=0)

    last_heartbeat_time = time.time()

    for i_episode in range(episodes):
        # Keep-alive heartbeat for long evaluations
        current_time = time.time()
        if (current_time - last_heartbeat_time) > 15:
            last_heartbeat_time = current_time
            print(f"Evaluation in progress... Episode {i_episode + 1}/{episodes}", flush=True)

        obs, _ = eval_env_instance.reset()  # Reset returns obs, info
        terminated = False
        truncated = False
        # No need to accumulate total_episode_reward within the loop if using info at the end

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env_instance.step(action.item())

            if terminated or truncated:  # Episode has ended
                # Our custom environment MultiDistributionEnv puts stats in 'episode_stats' at the end of an episode
                episode_final_stats = info.get('episode_stats')
                if episode_final_stats:
                    rewards_per_episode.append(episode_final_stats.get("total_reward", np.nan))
                    correct_rates_per_episode.append(episode_final_stats.get("correct_rate", np.nan))
                else:  # Fallback if 'episode_stats' is missing for some reason
                    # This path should ideally not be taken if MultiDistributionEnv is consistent
                    # print(f"警告 (evaluate_agent, ep {i_episode+1}): 回合结束但未在info中找到'episode_stats'. Info: {info}")
                    rewards_per_episode.append(np.nan)  # Log NaN to indicate missing data
                    correct_rates_per_episode.append(np.nan)

    # Filter out NaNs before calculating mean/std, in case some episodes had issues
    valid_rewards = [r for r in rewards_per_episode if not np.isnan(r)]
    valid_correct_rates = [cr for cr in correct_rates_per_episode if not np.isnan(cr)]

    avg_reward = np.mean(valid_rewards) if valid_rewards else -float('inf')  # Return -inf if no valid rewards
    std_reward = np.std(valid_rewards) if len(valid_rewards) > 1 else np.nan  # Std requires at least 2 points
    avg_correct_rate = np.mean(valid_correct_rates) if valid_correct_rates else np.nan

    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_correct_rate": avg_correct_rate,
        "rewards_raw": valid_rewards,  # Return only valid rewards/rates
        "correct_rates_raw": valid_correct_rates
    }