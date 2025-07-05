# hpo_definitions.py
import numpy as np
from scipy import stats
import optuna
from pathlib import Path
from typing import Dict, Tuple, Callable, List, Any, Union
import os
import pandas as pd
import time  # Ensure time is imported for duration calculation within finetuning

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 从其他模块导入
from config import (
    ALGO_CLASS_MAP,
    CONVERGENCE_METRICS_LAST_N_EPISODES,
    LOGS_DIR, PLOTS_DIR, HPO_STUDY_DIR, FINETUNE_LEARNING_RATE  # Added FINETUNE_LEARNING_RATE
)
from training_utils import train_agent, evaluate_agent, TrainingLoggerCallback
from environment import MultiDistributionEnv
from utils import sanitize_filename

# 确保HPO结果目录存在
HPO_STUDY_DIR.mkdir(parents=True, exist_ok=True)


def calculate_convergence_metrics(
        episode_data: List[Dict[str, Any]],
        last_n_episodes_for_slope: int = CONVERGENCE_METRICS_LAST_N_EPISODES,
        last_n_episodes_for_std: int = CONVERGENCE_METRICS_LAST_N_EPISODES
) -> Dict[str, float]:
    # 修复: 初始化值确保为 float 类型，避免 np.nan 引入类型问题
    metrics = {"slope": 0.0, "std_dev": 0.0, "max_reward_achieved": -float('inf')}

    if not episode_data or len(episode_data) < 5:
        return metrics

    rewards = [d['total_reward'] for d in episode_data if
               isinstance(d.get('total_reward'), (int, float))]  # Ensure rewards are numbers
    if not rewards: return metrics

    # 修复: 将 np.max 的结果显式转换为 float
    metrics["max_reward_achieved"] = float(np.max(rewards)) if rewards else -float(
        'inf')  # Handle empty rewards after filtering

    if len(rewards) >= last_n_episodes_for_std and len(rewards) >= 2:
        # 修复: 将 np.std 的结果显式转换为 float
        metrics["std_dev"] = float(np.std(rewards[-last_n_episodes_for_std:]))

    if len(rewards) >= last_n_episodes_for_slope and len(rewards) >= 2:
        y = np.array(rewards[-last_n_episodes_for_slope:])
        x = np.arange(len(y))
        try:
            slope_val, _, _, _, _ = stats.linregress(x, y)
            # 修复: 添加更严格的类型检查，以处理 linregress 可能的复杂返回值类型
            if isinstance(slope_val, (int, float)) and not np.isnan(slope_val):
                metrics["slope"] = float(slope_val)
            else:
                metrics["slope"] = 0.0
        except ValueError:
            metrics["slope"] = 0.0
        except Exception:
            metrics["slope"] = 0.0
    return metrics


# --- 特定算法的超参数建议函数 ---
def suggest_dqn_params(trial: optuna.Trial) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000, 200000])
    learning_starts = trial.suggest_categorical("learning_starts", [1000, 5000, 10000, 20000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.9, 0.999)  # Keep gamma tight
    tau = trial.suggest_float("tau", 0.001, 0.1, log=True)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    gradient_steps = trial.suggest_categorical("gradient_steps", [-1, 1, 4, 8])  # -1 means = train_freq
    target_update_interval = trial.suggest_categorical("target_update_interval", [250, 500, 1000, 2000, 4000])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.3)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.001, 0.05, log=True)

    # It's often better to suggest a separate learning rate for finetuning if that's a distinct phase.
    finetune_lr = trial.suggest_float("finetune_learning_rate", 1e-6, 5e-4, log=True)

    hyperparams = {
        "learning_rate": lr, "buffer_size": buffer_size, "learning_starts": learning_starts,
        "batch_size": batch_size, "gamma": gamma, "tau": tau, "train_freq": train_freq,
        "gradient_steps": gradient_steps, "target_update_interval": target_update_interval,
        "exploration_fraction": exploration_fraction, "exploration_final_eps": exploration_final_eps,
        "finetune_learning_rate": finetune_lr  # Add to hyperparams dict
    }
    policy_kwargs = {}
    return hyperparams, policy_kwargs


def suggest_ppo_params(trial: optuna.Trial) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 20),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),  # Keep gamma tight
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),  # Clip range can be constant or scheduled
        "vf_coef": trial.suggest_float("vf_coef", 0.2, 0.8),
        "finetune_learning_rate": trial.suggest_float("finetune_learning_rate", 1e-6, 5e-4, log=True)
        # Add to hyperparams dict
    }
    policy_kwargs = {}
    return hyperparams, policy_kwargs


def suggest_a2c_params(trial: optuna.Trial) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),  # A2C can tolerate higher LR
        "n_steps": trial.suggest_categorical("n_steps", [5, 10, 16, 20, 32]),  # A2C typically uses small n_steps
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.2, 0.8),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 5.0),
        "finetune_learning_rate": trial.suggest_float("finetune_learning_rate", 1e-6, 5e-4, log=True)
        # Add to hyperparams dict
    }
    policy_kwargs = {}
    return hyperparams, policy_kwargs


# --- HPO 基础目标函数 ---
def _objective_base(
        trial: optuna.Trial,
        algo_name: str,
        hyperparam_suggestion_func: Callable[[optuna.Trial], Tuple[Dict[str, Any], Dict[str, Any]]],
        fixed_initial_dist_config: Dict[str, Any],
        fixed_target_scenario_config: Dict[str, Any],
        hpo_timesteps: Dict[str, int]
) -> float:
    """
    Optuna的基础目标函数。执行简化的训练-评估-适应-评估流程。
    收敛性惩罚已移除。训练和适应时长被记录为用户属性。
    """
    hyperparams, policy_kwargs = hyperparam_suggestion_func(trial)
    initial_dist_name_clean = sanitize_filename(fixed_initial_dist_config['name'])
    target_scenario_name_clean = sanitize_filename(fixed_target_scenario_config['name'])

    # 阶段1: 初始训练
    initial_train_log_suffix_hpo = f"hpo_trial{trial.number}_initial_{initial_dist_name_clean}_{algo_name}"
    main_trained_model, initial_logger, initial_train_duration = train_agent(  # Unpack 3 values
        env_config=fixed_initial_dist_config, algorithm=algo_name,
        total_timesteps=hpo_timesteps['initial'], hyperparams_override=hyperparams,
        policy_kwargs=policy_kwargs, log_file_suffix=initial_train_log_suffix_hpo,
        sb3_verbose=0, trial=trial
    )
    trial.set_user_attr("initial_train_duration_s", initial_train_duration if main_trained_model else None)

    if main_trained_model is None:  # Pruned or failed during initial training
        return -float('inf')

    # 阶段2: 基线评估
    baseline_perf = evaluate_agent(env_config=fixed_initial_dist_config, model=main_trained_model,
                                   episodes=hpo_timesteps['eval_episodes'])

    # 阶段3: 适应性测试
    eval_before_adapt = evaluate_agent(env_config=fixed_target_scenario_config, model=main_trained_model,
                                       episodes=hpo_timesteps['eval_episodes'])

    adaptation_trigger_threshold = baseline_perf['avg_reward'] * fixed_target_scenario_config['threshold_ratio']
    adapted_model_for_scenario = None
    adaptation_logger = None
    adaptation_duration = 0.0  # Initialize adaptation duration
    adapt_type_msg = "无操作"

    # Temporary model save path for finetuning
    temp_model_save_path_hpo = LOGS_DIR / f"hpo_trial{trial.number}_{algo_name}_temp_model_ft.zip"
    main_trained_model.save(temp_model_save_path_hpo)

    if eval_before_adapt['avg_reward'] < adaptation_trigger_threshold:
        adapt_type_msg = "重训练"
        retrain_log_suffix_hpo = f"hpo_trial{trial.number}_retrain_{target_scenario_name_clean}_{algo_name}"
        adapted_model_for_scenario, adaptation_logger, adaptation_duration = train_agent(  # Unpack 3 values
            env_config=fixed_target_scenario_config, algorithm=algo_name,
            total_timesteps=hpo_timesteps['retrain'], hyperparams_override=hyperparams,
            policy_kwargs=policy_kwargs, log_file_suffix=retrain_log_suffix_hpo,
            sb3_verbose=0, trial=trial
        )
    else:
        adapt_type_msg = "微调"
        vec_env_for_finetune = None
        finetune_start_time = time.time()  # Start timer for finetuning
        try:
            def _make_finetune_env():
                env = MultiDistributionEnv(dist_config=fixed_target_scenario_config)
                return Monitor(env)

            vec_env_for_finetune = DummyVecEnv([_make_finetune_env])

            adapted_model_for_scenario = ALGO_CLASS_MAP[algo_name].load(temp_model_save_path_hpo,
                                                                        env=vec_env_for_finetune)

            # Use specific finetune_learning_rate from HPO if available, else main LR, else default from config
            finetune_lr = hyperparams.get("finetune_learning_rate",
                                          hyperparams.get("learning_rate", FINETUNE_LEARNING_RATE))

            if hasattr(adapted_model_for_scenario, 'policy') and \
                    hasattr(adapted_model_for_scenario.policy, 'optimizer') and \
                    adapted_model_for_scenario.policy.optimizer is not None:
                try:
                    # print(f"  Trial {trial.number}: Setting finetuning LR to {finetune_lr:.2e}")
                    adapted_model_for_scenario.policy.optimizer.param_groups[0]['lr'] = finetune_lr
                except Exception as e_lr:
                    print(f"  Trial {trial.number}: Warning - Could not set finetuning LR for {algo_name}: {e_lr}")

            # Use a new logger for finetuning phase; TrainingLoggerCallback will handle pruning reports
            adaptation_logger = TrainingLoggerCallback(verbose=0, trial=trial,
                                                       eval_freq_pruning=max(1, hpo_timesteps['finetune'] // 10))

            adapted_model_for_scenario.learn(
                total_timesteps=hpo_timesteps['finetune'],
                callback=adaptation_logger,
                reset_num_timesteps=False  # IMPORTANT: Do not reset timesteps for finetuning
            )
            adaptation_duration = time.time() - finetune_start_time
        except optuna.TrialPruned:
            adaptation_duration = time.time() - finetune_start_time
            # print(f"Trial {trial.number} pruned during finetune learn().")
            # Ensure cleanup even on pruning
            if temp_model_save_path_hpo.exists(): os.remove(temp_model_save_path_hpo)
            if vec_env_for_finetune: vec_env_for_finetune.close()
            raise
        except Exception as e_finetune:
            adaptation_duration = time.time() - finetune_start_time
            print(f"Trial {trial.number} error during finetune for {algo_name}: {e_finetune}")
            adapted_model_for_scenario = None  # Mark as failed
        finally:
            if vec_env_for_finetune: vec_env_for_finetune.close()

    trial.set_user_attr("adaptation_type", adapt_type_msg)
    trial.set_user_attr("adaptation_duration_s", adaptation_duration if adapted_model_for_scenario else None)

    if temp_model_save_path_hpo.exists():
        try:
            os.remove(temp_model_save_path_hpo)
        except OSError as e:
            print(f"Warning: Could not remove temp model {temp_model_save_path_hpo}: {e}")

    if adapted_model_for_scenario is None:  # If adaptation (retrain or finetune) failed or pruned
        return -float('inf')

    final_eval_results = evaluate_agent(env_config=fixed_target_scenario_config, model=adapted_model_for_scenario,
                                        episodes=hpo_timesteps['eval_episodes'])

    # Calculate convergence metrics for logging
    initial_episode_data = initial_logger.episode_data if initial_logger and hasattr(initial_logger,
                                                                                     'episode_data') else []
    adapt_episode_data = adaptation_logger.episode_data if adaptation_logger and hasattr(adaptation_logger,
                                                                                         'episode_data') else []
    initial_conv_metrics = calculate_convergence_metrics(initial_episode_data)
    adapt_conv_metrics = calculate_convergence_metrics(adapt_episode_data)

    base_obj_initial_reward = baseline_perf['avg_reward']
    base_obj_final_reward = final_eval_results['avg_reward']

    if base_obj_initial_reward == -float('inf') or base_obj_final_reward == -float('inf'):
        objective_value = -float('inf')
    else:
        objective_value = (base_obj_initial_reward + base_obj_final_reward) / 2.0

    if np.isnan(objective_value):
        objective_value = -float('inf')

    final_objective_value = objective_value  # No penalty

    # Set user attributes for Optuna trial (can be viewed in Optuna dashboard or study.trials_dataframe())
    trial.set_user_attr("initial_reward", base_obj_initial_reward if base_obj_initial_reward != -float('inf') else None)
    trial.set_user_attr("final_adapted_reward",
                        base_obj_final_reward if base_obj_final_reward != -float('inf') else None)
    trial.set_user_attr("initial_slope",
                        initial_conv_metrics['slope'] if not np.isnan(initial_conv_metrics['slope']) else None)
    trial.set_user_attr("initial_std",
                        initial_conv_metrics['std_dev'] if not np.isnan(initial_conv_metrics['std_dev']) else None)
    trial.set_user_attr("adapt_slope",
                        adapt_conv_metrics['slope'] if not np.isnan(adapt_conv_metrics['slope']) else None)
    trial.set_user_attr("adapt_std",
                        adapt_conv_metrics['std_dev'] if not np.isnan(adapt_conv_metrics['std_dev']) else None)

    # Verbose logging for each trial can be helpful for debugging HPO but might clutter logs.
    # print(
    #     f"T{trial.number} End: Algo:{algo_name}, InitD:{initial_dist_name_clean}, TargetD:{target_scenario_name_clean}."
    #     f" IR:{base_obj_initial_reward:.2f}(S:{initial_conv_metrics['slope']:.2f},SD:{initial_conv_metrics['std_dev']:.2f}),"
    #     f" FAR:{base_obj_final_reward:.2f}(S:{adapt_conv_metrics['slope']:.2f},SD:{adapt_conv_metrics['std_dev']:.2f})."
    #     f" ObjV:{final_objective_value:.2f}"
    # )

    return final_objective_value


# --- HPO Study 执行函数 ---
def run_hpo_for_algorithm(
        algo_name: str,
        initial_dist_cfg: Dict[str, Any],
        target_dist_cfg: Dict[str, Any],
        hpo_timesteps_cfg: Dict[str, int],
        n_hpo_trials: int
) -> Dict[str, Any]:
    print(f"\n--- 开始为算法 {algo_name} 进行HPO ---")
    print(f"HPO初始分布: {initial_dist_cfg['name']}")
    print(f"HPO目标适应场景: {target_dist_cfg['name']}")

    suggestion_func_map = {"DQN": suggest_dqn_params, "PPO": suggest_ppo_params, "A2C": suggest_a2c_params}
    if algo_name not in suggestion_func_map:
        raise ValueError(f"不支持为算法 {algo_name} 进行HPO (未定义参数建议函数)。")
    hyperparam_suggestion_func = suggestion_func_map[algo_name]

    study_name_suffix = f"mainobj_{sanitize_filename(initial_dist_cfg['name'])}_to_{sanitize_filename(target_dist_cfg['name'])}"
    study_name = f"{algo_name.lower()}_hpo_{study_name_suffix}"  # Consistent study name

    storage_path = HPO_STUDY_DIR / f"{study_name}.db"
    storage_name = f"sqlite:///{storage_path.resolve()}"

    # Pruner configuration: n_warmup_steps is the number of intermediate values reported before pruning can occur.
    # This should align with how often TrainingLoggerCallback calls trial.report().
    # If eval_freq_pruning in TrainingLoggerCallback is e.g., total_timesteps / 10,
    # then n_warmup_steps = 3 would mean pruning starts after the 3rd report.
    pruner_n_warmup_reports = 3  # Number of reports before pruning is active

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=max(5, n_hpo_trials // 10),  # Run some trials before pruner activates across trials
            n_warmup_steps=pruner_n_warmup_reports,  # Number of *reports* within a trial before it's eligible
            interval_steps=1  # Prune after each report (step) post-warmup if median is bad
        )
    )

    objective_with_args = lambda trial: _objective_base(
        trial, algo_name, hyperparam_suggestion_func,
        initial_dist_cfg, target_dist_cfg, hpo_timesteps_cfg
    )

    completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Study '{study_name}': {completed_trials_count} trials已完成。")
    remaining_trials = n_hpo_trials - completed_trials_count

    if remaining_trials > 0:
        print(f"开始优化 {remaining_trials} 个新 trials...")
        try:
            # We now pass the keep-alive callback here
            study.optimize(
                objective_with_args,
                n_trials=remaining_trials,
                callbacks=[hpo_keep_alive_callback],  # <-- 核心修复：添加回调
                gc_after_trial=True,
                n_jobs=1  # Keep n_jobs=1 for simplicity and stable logging
            )
        except KeyboardInterrupt:
            print("\nHPO process manually interrupted by user.")
    else:
        print("已达到目标HPO trials数量。")

    print(f"\n--- {algo_name} 参数寻优完成 (初始: {initial_dist_cfg['name']}, 目标: {target_dist_cfg['name']}) ---")
    if not study.trials:
        print("警告: Optuna study中没有任何trial记录。")
        return {}

    try:
        successful_trials = [t for t in study.trials if
                             t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > -float(
                                 'inf')]
        if not successful_trials:
            print("警告: Optuna study中没有成功的trial可供选择最佳参数。")
            return {}

        best_trial = study.best_trial  # Optuna should pick the best completed trial
        if best_trial.value is None or best_trial.value == -float('inf'):
            print("警告: Optuna study 的 best_trial 值无效。没有找到有效参数。")
            # Try to find best from successful_trials manually if Optuna's is bad
            if successful_trials:
                # 修复: 为 key 函数添加默认值，处理 t.value 可能为 None 的情况
                best_trial = max(successful_trials, key=lambda t: t.value if t.value is not None else -float('inf'))
                if best_trial.value is None or best_trial.value == -float('inf'):
                    return {}  # Still no good trial
            else:  # Should not be reached if successful_trials was checked
                return {}

        print(f"最佳Trial #{best_trial.number}: 目标值: {best_trial.value:.4f}")
        print("  最佳超参数:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        best_params_df = pd.DataFrame([best_trial.params])
        best_params_filename = f"best_{algo_name.lower()}_params_{study_name_suffix}.csv"  # Use same suffix as study name
        best_params_path = HPO_STUDY_DIR / best_params_filename
        best_params_df.to_csv(best_params_path, index=False)
        print(f"最佳参数已保存至: {best_params_path}")
        return best_trial.params
    except ValueError:
        print("警告: Optuna study中没有trial或无法确定最佳trial。")
        return {}
    except Exception as e:
        print(f"选择最佳参数时发生未知错误: {e}")
        return {}


# --- HPO Callback for Keep-Alive ---
def hpo_keep_alive_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """
    一个简单的回调函数，在每次HPO trial结束后打印进度。
    这可以作为一个“心-跳”信息，防止Streamlit的WebSocket连接因超时而关闭。
    """
    print(f"HPO Trial #{trial.number} 完成. 状态: {trial.state}. 目标值: {trial.value if trial.value is not None else 'N/A'}")