# main_experiment.py (最终优化版)
import sys
from pathlib import Path
import pandas as pd
import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import traceback  # For detailed error logging
import matplotlib.pyplot as plt
import argparse # <-- 导入 argparse

# 从我们创建的模块中导入必要的函数和类
from config import (
    LOGS_DIR, PLOTS_DIR, HPO_STUDY_DIR,
    ALL_AVAILABLE_ALGORITHMS, ALL_DISTRIBUTION_CONFIGS,
    TIMESTEPS_INITIAL_TRAIN, TIMESTEPS_RETRAIN, TIMESTEPS_FINETUNE,
    HPO_TIMESTEPS_CONFIG, N_HPO_TRIALS,
    DEFAULT_HYPERPARAMS, FINETUNE_LEARNING_RATE,
    FIXED_THRESHOLD_FOR_COMPLEXITY, SAMPLES_FOR_COMPLEXITY_CALCULATION,
    ALGO_CLASS_MAP
)
from utils import (
    set_chinese_font_for_matplotlib, Tee, sanitize_filename,
    select_items_interactively,
    calculate_distribution_complexity
)
from environment import MultiDistributionEnv
from training_utils import train_agent, evaluate_agent, TrainingLoggerCallback
from plotting_utils import (
    plot_evaluation_reward_curves,
    plot_adaptation_performance_boxplots,
    plot_overall_summary_barchart,
    plot_learning_curves_from_logs,
    plot_combined_learning_curve,
    plot_complexity_vs_performance,
    plot_reward_distribution_over_training_stages,
    plot_adaptation_delta_rewards,
    plot_convergence_metrics_over_training,
    plot_aggregated_performance_radar,  # 确保这个是您修改后的新版雷达图函数
    plot_overall_summary_barchart_comparison,
    plot_hpo_reward_distribution_panels  # 确保这个也从plotting_utils导入
)
from hpo_definitions import run_hpo_for_algorithm


def define_plot_options(all_results):
    """
    定义所有可用的绘图选项。
    这是一个辅助函数，用于在实验开始前和结束后都能获取绘图列表。
    """
    plot_options = [
        {"id": "eval_curves", "name": "评估奖励曲线 (适应场景, 按初始分布分组)",
         "func": plot_evaluation_reward_curves, "args": [all_results, PLOTS_DIR]},
        {"id": "boxplots", "name": "自适应性能箱线图 (适应场景, 按初始分布分组)",
         "func": plot_adaptation_performance_boxplots, "args": [all_results, PLOTS_DIR]},
        {"id": "summary_bar", "name": "总体奖励柱状图 (主要运行, 按初始分布分组)",
         "func": plot_overall_summary_barchart, "args": [all_results, PLOTS_DIR]},
        {"id": "learning_curves", "name": "详细学习曲线 (各训练阶段)", "type": "loop"},
        {"id": "reward_stages", "name": "训练阶段奖励分布 (各训练阶段)", "type": "loop"},
        {"id": "combined_curves", "name": "组合学习曲线 (初始+适应)", "type": "loop"},
        {"id": "complexity_perf", "name": "复杂度 vs 性能散点图 (多种类型)", "type": "multi_call"},
        {"id": "delta_rewards", "name": "自适应奖励变化量柱状图 (可按初始分布过滤)", "type": "multi_call_filter"},
        {"id": "convergence_dynamics", "name": "训练收敛性指标动态图 (各训练阶段)", "type": "loop"},
    ]

    hpo_runs_exist = any(r.get('run_type') == 'hpo_params' for r in all_results) if all_results else True
    default_runs_exist = any(r.get('run_type') == 'default_params' for r in all_results) if all_results else True

    # 仅当可能进行HPO对比时，才显示这些选项
    if hpo_runs_exist and default_runs_exist:
        plot_options.extend([
            # 这是您新添加的高级雷达图
            {"id": "hpo_profile_radar", "name": "高级HPO性能对比雷达图 (6维度剖析, 按算法聚合)",
             "func": plot_aggregated_performance_radar, "args": [all_results, PLOTS_DIR, 'algorithm']},
            {"id": "hpo_barchart", "name": "HPO对比总体摘要柱状图 (按初始分布分组)",
             "func": plot_overall_summary_barchart_comparison, "args": [all_results, PLOTS_DIR, True]},
            {"id": "hpo_distrib_panels_final", "name": "HPO参数奖励分布对比 (最终评估, 多面板)",
             "func": plot_hpo_reward_distribution_panels, "args": [all_results, PLOTS_DIR, 'final_eval']},
        ])
        if any(r.get('scenario_type') == 'adaptation' for r in all_results) if all_results else True:
            plot_options.append(
                {"id": "hpo_distrib_panels_initial", "name": "HPO参数奖励分布对比 (适应前评估, 多面板)",
                 "func": plot_hpo_reward_distribution_panels, "args": [all_results, PLOTS_DIR, 'initial_eval']}
            )
    return plot_options


def run_complete_experiment():
    """
    执行完整的实验流程，所有配置在开始时一次性完成，实现全自动运行。
    """
    # --- -1: 解析命令行参数 (用于非交互式运行) ---
    parser = argparse.ArgumentParser(description="强化学习适应性实验框架")
    parser.add_argument("--non-interactive", action="store_true", help="启用非交互模式，通过命令行参数运行")
    parser.add_argument("--algos", nargs='+', help="要运行的算法列表")
    parser.add_argument("--initial-dists", nargs='+', help="初始训练分布列表")
    parser.add_argument("--target-dists", nargs='+', help="目标适应场景列表")
    parser.add_argument("--run-hpo", action="store_true", help="运行HPO对比")
    parser.add_argument("--plots", nargs='+', help="要生成的图表ID列表")
    args = parser.parse_args()
    
    # --- 0. 初始化环境和日志 ---
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    HPO_STUDY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    main_run_log_file_path = LOGS_DIR / f"main_run_log_{timestamp}.txt"
    log_file_handle = None
    tee_output_redirect = None

    try:
        log_file_handle = open(main_run_log_file_path, "w", encoding='utf-8')
        tee_output_redirect = Tee(sys.stdout, log_file_handle, sys.stderr)
        sys.stdout = tee_output_redirect
        sys.stderr = tee_output_redirect

        set_chinese_font_for_matplotlib()
        print(f"主运行日志将保存至: {main_run_log_file_path}")
        print("--- 强化学习适应性实验 (全自动流程) ---")

        # --- 0.2: 创建本次运行的专属结果目录 ---
        run_results_dir = PLOTS_DIR / f"run_{timestamp}"
        run_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"所有图表和结果将保存至: {run_results_dir}")

        # --- 0.5 计算并排序分布配置 (基于复杂度) ---
        print("\n--- 正在计算和排序分布复杂度 ---")
        distributions_with_complexity = []
        for config_item in ALL_DISTRIBUTION_CONFIGS:
            complexity = calculate_distribution_complexity(
                config_item,
                FIXED_THRESHOLD_FOR_COMPLEXITY,
                SAMPLES_FOR_COMPLEXITY_CALCULATION
            )
            distributions_with_complexity.append({**config_item, "complexity": complexity})
        sorted_distributions = sorted(distributions_with_complexity, key=lambda x: x.get("complexity", -1.0),
                                      reverse=True)
        print("分布复杂度计算和排序完成。")

        # --- 1. 一次性完成所有用户配置 ---
        selected_algo_list = []
        selected_initial_dist_list = []
        selected_target_scenarios_list = []
        run_hpo_comparison = False
        selected_plots_ids = []

        if args.non_interactive:
            print("\n" + "=" * 25 + " 步骤1: 从命令行参数加载配置 " + "=" * 25)
            # 参数验证和映射
            if not all([args.algos, args.initial_dists, args.target_dists, args.plots]):
                 print("错误：在非交互模式下，--algos, --initial-dists, --target-dists, --plots 均为必需参数。", file=sys.stderr)
                 return

            selected_algo_list = [a for a in args.algos if a in ALL_AVAILABLE_ALGORITHMS]
            
            all_dist_names = [d['name'] for d in sorted_distributions]
            selected_initial_dist_list = [d for d in sorted_distributions if d['name'] in args.initial_dists]
            selected_target_scenarios_list = [d for d in sorted_distributions if d['name'] in args.target_dists]

            run_hpo_comparison = args.run_hpo
            
            all_possible_plots = define_plot_options([])
            plot_id_map = {p['id']: p for p in all_possible_plots}
            selected_plots_desc = [plot_id_map[p_id] for p_id in args.plots if p_id in plot_id_map]
            
            # 检查是否有无效的输入
            if len(selected_algo_list) != len(args.algos):
                print(f"警告: 无效的算法名称被忽略: {set(args.algos) - set(ALL_AVAILABLE_ALGORITHMS)}")
            if len(selected_initial_dist_list) != len(args.initial_dists):
                print(f"警告: 无效的初始分布名称被忽略: {set(args.initial_dists) - set(d['name'] for d in selected_initial_dist_list)}")
            if len(selected_target_scenarios_list) != len(args.target_dists):
                print(f"警告: 无效的目标场景名称被忽略: {set(args.target_dists) - set(d['name'] for d in selected_target_scenarios_list)}")

            # 新增：关键配置验证
            if not all([selected_algo_list, selected_initial_dist_list, selected_target_scenarios_list]):
                print("错误：一个或多个关键配置列表（算法、初始分布、目标场景）在过滤无效名称后为空。实验中止。", file=sys.stderr)
                return
            
            representative_target_for_hpo = selected_target_scenarios_list[0] if selected_target_scenarios_list else None
            
        else:
            print("\n" + "=" * 25 + " 步骤1: 一次性实验配置 (交互模式) " + "=" * 25)
            dist_display_func = lambda item: f"{item['name']} (C: {item.get('complexity', -1.0):.2f})"

            selected_algo_list = select_items_interactively(ALL_AVAILABLE_ALGORITHMS, "要运行的算法", allow_multiple=True)
            if not selected_algo_list: print("未选择算法，实验中止。"); return

            selected_initial_dist_list = select_items_interactively(sorted_distributions, "初始训练分布",
                                                                    display_func=dist_display_func, allow_multiple=True)
            if not selected_initial_dist_list: print("未选择初始训练分布，实验中止。"); return

            selected_target_scenarios_list = select_items_interactively(sorted_distributions,
                                                                        "目标适应场景 (HPO将针对选中的第一个)",
                                                                        display_func=dist_display_func, allow_multiple=True)
            if not selected_target_scenarios_list: print("未选择目标适应场景，实验中止。"); return

            representative_target_for_hpo = selected_target_scenarios_list[0] if selected_target_scenarios_list else None

            run_hpo_comparison_choice = input("\n是否为选定算法运行HPO并与默认参数对比? (yes/no, 默认no): ").strip().lower()
            run_hpo_comparison = run_hpo_comparison_choice == 'yes'

            # --- 新增：预先选择所有要生成的图表 ---
            print("\n--- 请选择实验结束后希望自动生成的图表 ---")
            # 传入一个空列表来获取所有可能的绘图选项
            all_possible_plots = define_plot_options([])
            selected_plots_desc = select_items_interactively(
                items=all_possible_plots, item_type_name="图表类型",
                display_func=lambda item: item['name'], allow_multiple=True
            )

        print("\n" + "=" * 25 + " 配置完成，实验将全自动运行 " + "=" * 25)
        print(f"\n--- 选定配置概览 ---")
        print(f"  算法: {', '.join(selected_algo_list)}")
        print(f"  初始训练分布: {', '.join([d['name'] for d in selected_initial_dist_list])}")
        print(f"  目标适应场景: {', '.join([s['name'] for s in selected_target_scenarios_list])}")
        if run_hpo_comparison:
            if representative_target_for_hpo:
                print(f"  将运行HPO对比。HPO代表性目标场景: {representative_target_for_hpo['name']}")
            else:
                print("  将尝试运行HPO对比，但缺少有效的代表性目标场景。")
        else:
            print("  将仅使用默认参数运行。")
        if selected_plots_desc:
            print(f"  将生成图表: {', '.join([p['name'] for p in selected_plots_desc])}")
        else:
            print("  将不生成任何图表。")
        print("=" * 70 + "\n")

        all_runs_master_results = []

        # --- 2. 主实验循环 (全自动) ---
        for selected_algo in selected_algo_list:
            for selected_initial_dist in selected_initial_dist_list:
                print(f"\n\n{'=' * 80}")
                print(f"--- 开始处理: 算法='{selected_algo}', 初始分布='{selected_initial_dist['name']}' ---")
                print(f"{'=' * 80}\n")

                algo_name_clean = sanitize_filename(selected_algo)
                initial_dist_name_clean = sanitize_filename(selected_initial_dist['name'])

                run_configurations = [
                    {"run_type": "default_params", "hyperparams": DEFAULT_HYPERPARAMS.get(selected_algo, {}).copy()}]

                if run_hpo_comparison:
                    # 修复：仅当存在有效的HPO目标时才执行HPO流程
                    if not representative_target_for_hpo:
                        print(f"警告: HPO对比已启用，但未找到有效的代表性目标场景。跳过对 {selected_algo} on {selected_initial_dist['name']} 的HPO流程。")
                    else:
                        print(f"\n--- {selected_algo} on {selected_initial_dist['name']}: HPO流程 ---")
                        hpo_target_name_clean = sanitize_filename(representative_target_for_hpo['name'])
                        best_params_filename = f"best_{algo_name_clean.lower()}_params_mainobj_{initial_dist_name_clean}_to_{hpo_target_name_clean}.csv"
                        best_params_path = HPO_STUDY_DIR / best_params_filename

                        best_hyperparams_hpo = None
                        if best_params_path.exists():
                            print(f"检测到已存在的优化参数文件: {best_params_path}")
                            try:
                                df_best_params = pd.read_csv(best_params_path)
                                if not df_best_params.empty:
                                    best_hyperparams_hpo = df_best_params.iloc[0].to_dict()
                                    print(f"成功加载已优化的参数: {best_hyperparams_hpo}")
                            except Exception as e_load_params:
                                print(f"警告: 加载优化参数文件 {best_params_path} 失败: {e_load_params}。将重新运行HPO。")

                        if best_hyperparams_hpo is None:
                            print(
                                f"为 '{selected_algo}' (初始: {selected_initial_dist['name']} -> HPO目标: {representative_target_for_hpo['name']}) 执行参数寻优...")
                            best_hyperparams_hpo = run_hpo_for_algorithm(
                                algo_name=selected_algo, initial_dist_cfg=selected_initial_dist,
                                target_dist_cfg=representative_target_for_hpo, hpo_timesteps_cfg=HPO_TIMESTEPS_CONFIG,
                                n_hpo_trials=N_HPO_TRIALS
                            )
                            if not best_hyperparams_hpo:
                                print(f"警告: {selected_algo} 的参数寻优未能找到有效参数。此组合的HPO对比将跳过。")
                            else:
                                print(f"参数寻优完成！最优参数: {best_hyperparams_hpo}")

                        if best_hyperparams_hpo:
                            # 参数类型转换逻辑... (保持不变)
                            params_to_int_map = {"PPO": ["n_steps", "batch_size", "n_epochs"],
                                                 "DQN": ["buffer_size", "learning_starts", "batch_size", "train_freq",
                                                         "gradient_steps", "target_update_interval"], "A2C": ["n_steps"]}
                            params_to_int = params_to_int_map.get(selected_algo, [])
                            for param_name, current_val in list(best_hyperparams_hpo.items()):
                                if current_val is None: continue
                                if param_name in params_to_int:
                                    try:
                                        if not isinstance(current_val, int): best_hyperparams_hpo[param_name] = int(
                                            float(current_val))
                                    except ValueError:
                                        print(f"警告: 转换参数 '{param_name}' (值: {current_val}) 为整数时出错。")
                                elif "learning_rate" in param_name:
                                    try:
                                        if not isinstance(current_val, float): best_hyperparams_hpo[param_name] = float(
                                            current_val)
                                    except ValueError:
                                        print(f"警告: 转换学习率 '{param_name}' (值: {current_val}) 为浮点数时出错。")
                            run_configurations.append({"run_type": "hpo_params", "hyperparams": best_hyperparams_hpo})

                for config_run in run_configurations:
                    # ... 核心训练和评估循环 (这部分逻辑保持不变) ...
                    current_run_type = config_run["run_type"]
                    current_hyperparams = config_run["hyperparams"]

                    print(
                        f"\n---== 执行运行: 类型='{current_run_type}', 算法='{selected_algo}', 初始='{selected_initial_dist['name']}' ==---")
                    print(f"使用参数: {current_hyperparams}")

                    experiment_results_this_config_run = []
                    empty_eval_placeholder = {'rewards_raw': [], 'avg_reward': np.nan, 'std_reward': np.nan,
                                              'avg_correct_rate': np.nan}

                    # 阶段 3.1: 初始训练
                    print(
                        f"\n--- 阶段 3.1 ({current_run_type}): 在 '{selected_initial_dist['name']}' 上训练 {selected_algo} ---")
                    initial_train_log_suffix = f"{current_run_type}_{initial_dist_name_clean}_{algo_name_clean}_initial"
                    main_trained_model, _, initial_training_duration = train_agent(
                        env_config=selected_initial_dist, algorithm=selected_algo,
                        total_timesteps=TIMESTEPS_INITIAL_TRAIN,
                        hyperparams_override=current_hyperparams, log_file_suffix=initial_train_log_suffix,
                        sb3_verbose=0
                    )

                    initial_training_log_path_str = str(
                        LOGS_DIR / f"{selected_algo}_{sanitize_filename(initial_train_log_suffix)}_episode_log.xlsx")
                    current_model_save_path = LOGS_DIR / f"{algo_name_clean}_{current_run_type}_model_on_{initial_dist_name_clean}_{timestamp}.zip"

                    if not main_trained_model:
                        print(
                            f"错误：初始训练未能生成模型 ({current_run_type}, {selected_algo} on {selected_initial_dist['name']})。跳过此配置。")
                        all_runs_master_results.append({
                            "run_type": current_run_type, "algorithm": selected_algo,
                            "initial_training_dist_name": selected_initial_dist['name'],
                            "scenario_name": f"基线 ({selected_initial_dist['name']})",
                            "scenario_type": "baseline_FAIL",
                            "target_dist_type": selected_initial_dist['dist_type'],
                            "target_dist_params": str(selected_initial_dist['params']),
                            "complexity": selected_initial_dist.get('complexity', -1.0), "adapt_type": "训练失败",
                            "initial_eval": empty_eval_placeholder, "final_eval": empty_eval_placeholder,
                            "initial_training_log": "N/A",
                            "adaptation_log": "N/A", "initial_training_duration": initial_training_duration,
                            "adaptation_duration": 0.0
                        })
                        continue

                    main_trained_model.save(current_model_save_path)
                    print(f"初始训练模型 ({current_run_type}) 已保存至: {current_model_save_path}")

                    # 阶段 3.2: 基线评估
                    print(f"\n--- 阶段 3.2 ({current_run_type}): 评估 '{selected_initial_dist['name']}' 基线性能 ---")
                    baseline_perf = evaluate_agent(env_config=selected_initial_dist, model=main_trained_model,
                                                   episodes=100)
                    print(
                        f"基线性能 ({current_run_type}, {selected_initial_dist['name']}) - 平均奖励: {baseline_perf['avg_reward']:.2f} ± {baseline_perf['std_reward']:.2f}")

                    experiment_results_this_config_run.append({
                        "run_type": current_run_type, "algorithm": selected_algo,
                        "initial_training_dist_name": selected_initial_dist['name'],
                        "scenario_name": f"基线 ({selected_initial_dist['name']})", "scenario_type": "baseline",
                        "target_dist_type": selected_initial_dist['dist_type'],
                        "target_dist_params": str(selected_initial_dist['params']),
                        "complexity": selected_initial_dist.get('complexity', -1.0), "adapt_type": "无适应",
                        "initial_eval": empty_eval_placeholder, "final_eval": baseline_perf,
                        "initial_training_log": initial_training_log_path_str if Path(
                            initial_training_log_path_str).exists() else "N/A",
                        "adaptation_log": "N/A", "initial_training_duration": initial_training_duration,
                        "adaptation_duration": 0.0
                    })

                    # 阶段 3.3: 适应性测试
                    for target_scenario_config in selected_target_scenarios_list:
                        target_scenario_name = target_scenario_config['name']
                        adaptation_duration_for_this_scenario = 0.0
                        if selected_initial_dist['name'] == target_scenario_name: continue

                        target_scenario_name_clean = sanitize_filename(target_scenario_name)
                        print(f"\n--- 阶段 3.3 ({current_run_type}): 目标 '{target_scenario_name}' ---")

                        eval_before_adapt = evaluate_agent(env_config=target_scenario_config, model=main_trained_model,
                                                           episodes=50)
                        print(
                            f"  适应前性能 ({current_run_type}, {target_scenario_name}) - 平均奖励: {eval_before_adapt['avg_reward']:.2f}")

                        adaptation_trigger_threshold = baseline_perf['avg_reward'] * target_scenario_config[
                            'threshold_ratio']
                        adapted_model = None
                        adapt_type = "无操作"
                        adaptation_log_path_str = "N/A"
                        log_suffix_base = f"{current_run_type}_from_{initial_dist_name_clean}_to_{target_scenario_name_clean}_{algo_name_clean}"

                        if eval_before_adapt['avg_reward'] < adaptation_trigger_threshold:
                            adapt_type = "重训练"
                            print(
                                f"  性能低于阈值 ({eval_before_adapt['avg_reward']:.2f} < {adaptation_trigger_threshold:.2f})。在 '{target_scenario_name}' 上重训练...")
                            retrain_log_suffix = f"{log_suffix_base}_retrain"
                            adapted_model, _, adaptation_duration_for_this_scenario = train_agent(
                                env_config=target_scenario_config, algorithm=selected_algo,
                                total_timesteps=TIMESTEPS_RETRAIN,
                                hyperparams_override=current_hyperparams, log_file_suffix=retrain_log_suffix,
                                sb3_verbose=0
                            )
                            if adapted_model:
                                adaptation_log_path_str = str(
                                    LOGS_DIR / f"{selected_algo}_{sanitize_filename(retrain_log_suffix)}_episode_log.xlsx")
                        else:
                            adapt_type = "微调"
                            print(
                                f"  性能可接受 ({eval_before_adapt['avg_reward']:.2f} >= {adaptation_trigger_threshold:.2f})。在 '{target_scenario_name}' 上微调...")
                            vec_env_for_finetune = None
                            finetune_actual_learn_start_time = time.time()
                            try:
                                def _make_finetune_env():
                                    return Monitor(MultiDistributionEnv(dist_config=target_scenario_config))

                                vec_env_for_finetune = DummyVecEnv([_make_finetune_env])
                                adapted_model = ALGO_CLASS_MAP[selected_algo].load(current_model_save_path,
                                                                                   env=vec_env_for_finetune)
                                finetune_lr_to_use = current_hyperparams.get("finetune_learning_rate",
                                                                             current_hyperparams.get("learning_rate",
                                                                                                     FINETUNE_LEARNING_RATE))
                                if hasattr(adapted_model, 'policy') and hasattr(adapted_model.policy,
                                                                                'optimizer') and adapted_model.policy.optimizer is not None:
                                    try:
                                        original_lr = adapted_model.policy.optimizer.param_groups[0]['lr']
                                        adapted_model.policy.optimizer.param_groups[0]['lr'] = finetune_lr_to_use
                                        print(f"    微调学习率从 {original_lr:.2e} 调整为 {finetune_lr_to_use:.2e}")
                                    except Exception as e_lr:
                                        print(f"    警告：无法动态修改学习率: {e_lr}。")
                                finetune_logger_callback = TrainingLoggerCallback(verbose=0)
                                adapted_model.learn(total_timesteps=TIMESTEPS_FINETUNE,
                                                    callback=finetune_logger_callback, reset_num_timesteps=False)
                                if finetune_logger_callback.episode_data:
                                    finetune_log_suffix = f"{log_suffix_base}_finetune"
                                    adaptation_log_path_str = str(
                                        LOGS_DIR / f"{selected_algo}_{sanitize_filename(finetune_log_suffix)}_episode_log.xlsx")
                                    pd.DataFrame(finetune_logger_callback.episode_data).to_excel(
                                        Path(adaptation_log_path_str), index=False)
                            finally:
                                if vec_env_for_finetune: vec_env_for_finetune.close()
                            adaptation_duration_for_this_scenario = time.time() - finetune_actual_learn_start_time

                        model_to_eval_final = adapted_model if adapted_model is not None else main_trained_model
                        print(
                            f"  评估 {selected_algo} 在 '{target_scenario_name}' 上最终性能 (适应: {adapt_type}, 运行: {current_run_type})...")
                        final_eval_results = evaluate_agent(env_config=target_scenario_config,
                                                            model=model_to_eval_final, episodes=100)
                        print(
                            f"  适应后性能 ({current_run_type}, {target_scenario_name}) - 平均奖励: {final_eval_results['avg_reward']:.2f} ± {final_eval_results['std_reward']:.2f}")

                        experiment_results_this_config_run.append({
                            "run_type": current_run_type, "algorithm": selected_algo,
                            "initial_training_dist_name": selected_initial_dist['name'],
                            "scenario_name": target_scenario_name, "scenario_type": "adaptation",
                            "target_dist_type": target_scenario_config['dist_type'],
                            "target_dist_params": str(target_scenario_config['params']),
                            "complexity": target_scenario_config.get('complexity', -1.0),
                            "adapt_type": adapt_type, "initial_eval": eval_before_adapt,
                            "final_eval": final_eval_results,
                            "initial_training_log": initial_training_log_path_str if Path(
                                initial_training_log_path_str).exists() else "N/A",
                            "adaptation_log": adaptation_log_path_str if Path(
                                adaptation_log_path_str).exists() else "N/A",
                            "initial_training_duration": initial_training_duration,
                            "adaptation_duration": adaptation_duration_for_this_scenario
                        })

                    if os.path.exists(current_model_save_path):
                        try:
                            os.remove(current_model_save_path)
                        except Exception as e_del:
                            print(f"删除模型 {current_model_save_path} 失败: {e_del}")

                    all_runs_master_results.extend(experiment_results_this_config_run)

        print(f"\n{'=' * 80}")
        print("--- 所有实验运行处理完毕 ---")
        print(f"{'=' * 80}\n")

        # --- 3. 结果汇总与自动绘图 ---
        if not all_runs_master_results:
            print("实验未产生任何结果，无法进行总结和绘图。")
            return

        print("\n--- 步骤3: 生成实验结果总结 ---")
        summary_list_for_df = []
        for item in all_runs_master_results:
            initial_avg_reward = item.get('initial_eval', {}).get('avg_reward', np.nan)
            final_avg_reward = item.get('final_eval', {}).get('avg_reward', np.nan)
            summary_list_for_df.append({
                '运行类型': item['run_type'], '算法': item['algorithm'],
                '初始训练分布': item['initial_training_dist_name'],
                '目标场景/基线': item['scenario_name'], '场景类型': item['scenario_type'],
                '自适应类型': item['adapt_type'],
                '复杂度': f"{item.get('complexity', -1.0):.3f}",
                '自适应前均奖': f"{initial_avg_reward:.2f}" if not np.isnan(initial_avg_reward) else 'N/A',
                '自适应后均奖': f"{final_avg_reward:.2f}" if not np.isnan(final_avg_reward) else 'N/A',
                '初始训练时长(s)': f"{item.get('initial_training_duration', 0.0):.2f}",
                '适应阶段时长(s)': f"{item.get('adaptation_duration', 0.0):.2f}",
                '初始训练日志': Path(item.get('initial_training_log', "N/A")).name,
                '自适应阶段日志': Path(item.get('adaptation_log', "N/A")).name
            })
        results_df_final = pd.DataFrame(summary_list_for_df)
        csv_filename_final = f"ALL_RESULTS_SUMMARY_{timestamp}.csv"
        csv_path_final = run_results_dir / csv_filename_final #<-- 使用本次运行的专属目录
        results_df_final.to_csv(csv_path_final, index=False, encoding='utf-8-sig')
        print(f"\n所有运行的实验结果摘要已保存至: {csv_path_final}")

        if not selected_plots_desc:
            print("\n未选择任何图表进行生成，实验结束。")
            # 确保在返回前打印最终结果目录
            print(f"\nFINAL_RESULTS_DIR:{run_results_dir.resolve()}")
            return

        print("\n--- 步骤4: 根据预选配置自动生成图表 ---")

        # --------------------------------------------------------------------
        # --- 最终修复：重构绘图逻辑，确保执行路径清晰 ---
        # --------------------------------------------------------------------
        results_df = pd.DataFrame(all_runs_master_results)
        
        # 1. 准备一个包含最新结果的绘图选项查找表
        plot_options_to_run = define_plot_options(all_runs_master_results)
        plot_options_map = {p['id']: p for p in plot_options_to_run}
        
        # 2. 准备一个已绘制的、针对每个初始训练只画一次的图表的集合
        plotted_initial_training_graphs = set()

        # 3. 直接遍历用户在UI上选择的图表ID列表
        selected_plot_ids = [p['id'] for p in selected_plots_desc]

        for plot_id in selected_plot_ids:
            plot_desc = plot_options_map.get(plot_id)
            if not plot_desc:
                print(f"警告: 在绘图阶段，无法找到预选图表ID '{plot_id}' 的执行定义，跳过。")
                continue

            print(f"\n>>> 开始生成图表: {plot_desc.get('name', '未知图表')}")
            
            try:
                plot_type = plot_desc.get("type")
                plot_func = plot_desc.get("func")

                # A. 处理全局图表 (直接调用)
                if plot_type is None and callable(plot_func):
                     # 将args中的 PLOTS_DIR 替换为本次运行的专属目录
                    new_args = [arg if arg is not PLOTS_DIR else run_results_dir for arg in plot_desc.get("args", [])]
                    plot_func(*new_args)

                # B. 处理需要循环所有结果的图表
                elif plot_type == "loop":
                    for index, res_item in results_df.iterrows():
                        # --- 提取和验证循环内变量 ---
                        initial_dist_name = res_item.get('initial_training_dist_name')
                        if not isinstance(initial_dist_name, str): continue
                        
                        s_run_type = res_item.get('run_type')
                        if not isinstance(s_run_type, str): continue
                        
                        s_algo = res_item.get('algorithm')
                        if not isinstance(s_algo, str): continue
                        
                        # 为每个初始分布创建子目录
                        plot_target_dir = run_results_dir / sanitize_filename(initial_dist_name)
                        plot_target_dir.mkdir(parents=True, exist_ok=True)
                        
                        initial_dist_cfg = next((d for d in sorted_distributions if d['name'] == initial_dist_name), {})
                        initial_complexity = initial_dist_cfg.get('complexity', -1.0)
                        
                        # --- 绘制与 "初始训练" 相关的、只应画一次的图 ---
                        initial_log_path = res_item.get("initial_training_log")
                        initial_log = Path(initial_log_path) if isinstance(initial_log_path, str) and initial_log_path != "N/A" else None
                        
                        initial_graph_key = (s_run_type, s_algo, initial_dist_name)
                        if initial_log and initial_log.exists() and initial_graph_key not in plotted_initial_training_graphs:
                            title_init_base = f"{s_run_type} - {s_algo} - 初始训练 ({initial_dist_name})"
                            suffix_init = f"{s_run_type}_{sanitize_filename(s_algo)}_{sanitize_filename(initial_dist_name)}_initial"
                            
                            if plot_id == "learning_curves":
                                plot_learning_curves_from_logs(initial_log, title_init_base, plot_target_dir, suffix_init, initial_complexity)
                            elif plot_id == "reward_stages":
                                plot_reward_distribution_over_training_stages(initial_log, title_init_base, plot_target_dir, f"{suffix_init}_stages", initial_complexity)
                            elif plot_id == "convergence_dynamics":
                                plot_convergence_metrics_over_training(initial_log, title_init_base, plot_target_dir, f"{suffix_init}_conv", initial_complexity)
                            
                            plotted_initial_training_graphs.add(initial_graph_key)
                        
                        # --- 绘制与 "适应阶段" 相关的图 ---
                        if res_item.get("scenario_type") == "adaptation":
                            target_scenario_name = res_item.get('scenario_name')
                            if not isinstance(target_scenario_name, str): continue
                            
                            adapt_type = res_item.get('adapt_type')
                            if not isinstance(adapt_type, str): continue

                            adapt_log_path = res_item.get("adaptation_log")
                            adapt_log = Path(adapt_log_path) if isinstance(adapt_log_path, str) and adapt_log_path != "N/A" else None

                            if plot_id == "combined_curves" and initial_log and initial_log.exists() and adapt_log and adapt_log.exists():
                                target_dist_cfg = next((d for d in sorted_distributions if d['name'] == target_scenario_name), {})
                                target_complexity = target_dist_cfg.get('complexity', -1.0)
                                plot_combined_learning_curve(
                                    initial_log, adapt_log, f"{s_run_type} - {s_algo}", initial_dist_name, 
                                    target_scenario_name, adapt_type, plot_target_dir, 
                                    f"{s_run_type}_{sanitize_filename(s_algo)}_{sanitize_filename(initial_dist_name)}_to_{sanitize_filename(target_scenario_name)}_comb",
                                    initial_complexity, target_complexity
                                )
                
                # C. 处理具有特殊调用逻辑的图表 (如复杂度vs性能)
                elif plot_type == "multi_call":
                    if plot_id == "complexity_perf":
                        plot_complexity_vs_performance(all_runs_master_results, run_results_dir, 'avg_reward', 'baseline')
                        plot_complexity_vs_performance(all_runs_master_results, run_results_dir, 'avg_reward', 'adaptation_initial')
                        plot_complexity_vs_performance(all_runs_master_results, run_results_dir, 'avg_reward', 'adaptation_final')
                        plot_complexity_vs_performance(all_runs_master_results, run_results_dir, 'avg_correct_rate', 'adaptation_final')
                
                # D. 处理需要按初始分布过滤的特殊图表
                elif plot_type == "multi_call_filter":
                    if plot_id == "delta_rewards":
                        unique_initial_dist_names = results_df['initial_training_dist_name'].unique()
                        plot_adaptation_delta_rewards(all_runs_master_results, run_results_dir) # 全局图
                        for init_dist_name_filter in unique_initial_dist_names:
                             if isinstance(init_dist_name_filter, str):
                                plot_adaptation_delta_rewards(all_runs_master_results, run_results_dir, init_dist_name_filter) # 过滤后的图

                print(f"<<< 完成生成图表: {plot_desc.get('name', '未知图表')}")

            except Exception as e_plot:
                print(f"!!! 生成图表 '{plot_desc.get('name', plot_id)}' 时发生严重错误: {e_plot}")
                traceback.print_exc() # 打印详细的错误追溯信息
        
        # --------------------------------------------------------------------
        # --- 新绘图逻辑结束 ---
        # --------------------------------------------------------------------
        
        print("\n所有预选图表生成完毕。")
        
        # --- 5. 打印最终结果目录路径，供 app.py 捕获 ---
        print(f"\nFINAL_RESULTS_DIR:{run_results_dir.resolve()}")

    except KeyboardInterrupt:
        print("\n用户手动中断实验。")
    except Exception as e:
        print(f"\n程序运行时发生严重错误: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        if tee_output_redirect is not None:
            print("\n关闭日志文件并恢复标准输出/错误流。")
            if isinstance(sys.stdout, Tee):
                tee_output_redirect.close()
            elif log_file_handle and not log_file_handle.closed:
                log_file_handle.close()
        elif log_file_handle and not log_file_handle.closed:
            log_file_handle.close()
        plt.close('all')


if __name__ == "__main__":
    run_complete_experiment()