import sys
from pathlib import Path
import pandas as pd
import time
import numpy as np
import os
import traceback
import re
from math import pi

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import seaborn as sns


# ==================================================================
# 阶段一: 模拟核心功能 (确保脚本可以独立运行)
# ==================================================================

class DummyModel:
    """一个虚拟的模型类，用于模拟真实的SB3模型。"""
    pass


def train_agent_mock(env_config, algorithm, total_timesteps, hyperparams_override, **kwargs):
    """模拟的训练函数。"""
    print(f"  (模拟训练) 正在为算法 '{algorithm}' 在 '{env_config['name']}' 上模拟训练 {total_timesteps} 步...")
    run_type = config_run.get('run_type', 'default_params')
    # HPO参数应该训练得更快
    base_duration = total_timesteps / (2000 if run_type == 'default_params' else 2500)
    mock_duration = base_duration + np.random.uniform(0.5, 1.5)
    time.sleep(0.01)  # 最小化等待
    print(f"  (模拟训练) 完成，耗时: {mock_duration:.2f} 秒。")
    return DummyModel(), None, mock_duration


def evaluate_agent_mock(env_config, model, episodes, run_type='default_params', **kwargs):
    """模拟的评估函数，根据运行类型生成有区分度的结果。"""
    print(f"  (模拟评估) 正在 '{env_config['name']}' 上评估 {episodes} 个回合...")
    if run_type == 'hpo_params':
        avg_reward = np.random.normal(loc=42, scale=4)
    else:
        avg_reward = np.random.normal(loc=35, scale=5)
    std_reward = np.random.normal(loc=4, scale=1)
    return {"avg_reward": avg_reward, "std_reward": std_reward}


# ==================================================================
# 阶段二: 绘图函数 (包括高级雷达图)
# ==================================================================

COLOR_DEFAULT = '#4C566A'
COLOR_HPO = '#5E81AC'
COLOR_IMPROVEMENT = '#A3BE8C'
COLOR_DECLINE = '#BF616A'


def sanitize_filename(filename: str) -> str:
    if not isinstance(filename, str): filename = str(filename)
    sanitized = re.sub(r'[^\w\s\.\-\(\)一-龥]', '', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized.strip('._- ') or "untitled"


def plot_aggregated_performance_radar(results_data: list, plots_dir: Path, aggregation_level: str = 'algorithm',
                                      **kwargs):
    """绘制高级性能雷达图。"""
    print("--- 正在生成高级性能雷达图 ---")
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_data)
    processed_metrics = []
    grouped = df.groupby([aggregation_level, 'run_type'])
    for name, group in grouped:
        algo_name, run_type = name
        baseline_df = group[group['scenario_type'] == 'baseline']
        adaptation_df = group[group['scenario_type'] == 'adaptation']
        m1 = baseline_df['final_eval'].apply(lambda x: x.get('avg_reward', np.nan)).mean()
        m2 = baseline_df['initial_training_duration'].astype(float).mean()
        m3 = adaptation_df['final_eval'].apply(lambda x: x.get('avg_reward', np.nan)).mean()
        m4 = adaptation_df['adaptation_duration'].astype(float).mean()
        m5 = (m1 + m3) / 2 if pd.notna(m1) and pd.notna(m3) else np.nan
        m6 = m2 + m4 if pd.notna(m2) and pd.notna(m4) else np.nan
        processed_metrics.append({
            aggregation_level: algo_name, 'run_type': run_type, 'M1_InitialAvgReward': m1,
            'M2_InitialAvgTime': m2, 'M3_AdaptedAvgReward': m3, 'M4_AdaptedAvgTime': m4,
            'M5_OverallAvgReward': m5, 'M6_TotalAvgTime': m6,
        })
    aggregated_df = pd.DataFrame(processed_metrics)
    metric_display_info = {
        'M1_InitialAvgReward': {"label": "初始环境\n平均奖励", "higher_is_better": True},
        'M2_InitialAvgTime': {"label": "初始训练\n平均时长(s)", "higher_is_better": False},
        'M3_AdaptedAvgReward': {"label": "适应后\n平均奖励", "higher_is_better": True},
        'M4_AdaptedAvgTime': {"label": "适应阶段\n平均时长(s)", "higher_is_better": False},
        'M5_OverallAvgReward': {"label": "总体\n平均奖励", "higher_is_better": True},
        'M6_TotalAvgTime': {"label": "总平均\n训练时长(s)", "higher_is_better": False},
    }
    metrics_to_plot = list(metric_display_info.keys())
    for group_val in aggregated_df[aggregation_level].unique():
        # ... (此处省略完整的绘图代码以保持简洁，但它与我们之前完善的版本相同)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title(f"高级雷达图: {group_val}")
        plot_path = plots_dir / f"radar_perf_sci-style_{sanitize_filename(str(group_val))}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"高级雷达图已保存至: {plot_path}")


# (此处省略其他绘图函数的模拟版本)
def plot_evaluation_reward_curves(*args, **kwargs): print("正在生成: 评估奖励曲线...")


def plot_adaptation_performance_boxplots(*args, **kwargs): print("正在生成: 自适应性能箱线图...")


def plot_overall_summary_barchart(*args, **kwargs): print("正在生成: 总体奖励柱状图...")


def plot_overall_summary_barchart_comparison(*args, **kwargs): print("正在生成: HPO对比总体摘要柱状图...")


def plot_hpo_reward_distribution_panels(*args, **kwargs): print("正在生成: HPO参数奖励分布对比...")


def plot_adaptation_delta_rewards(*args, **kwargs): print("正在生成: 自适应奖励变化量柱状图...")


# ==================================================================
# 阶段三: 主实验流程
# ==================================================================
def select_items_interactively(items, item_type_name, **kwargs):
    """模拟的交互式选择函数。"""
    print(f"\n--- 为 '{item_type_name}' 进行自动选择 ---")
    if not items: return []
    if "算法" in item_type_name:
        selection = [items[0]]  # 自动选择第一个算法
    elif "分布" in item_type_name or "场景" in item_type_name:
        selection = items[:2]  # 自动选择前两个
    else:  # 图表
        selection = items
    print(f"自动选择: {[(s['name'] if isinstance(s, dict) else s) for s in selection]}")
    return selection


def run_complete_experiment():
    PLOTS_DIR = Path.cwd() / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    print("--- 强化学习适应性实验 (模拟版本) ---")

    # --- 1. 前置所有交互式选择 ---
    ALL_AVAILABLE_ALGORITHMS = ["PPO", "DQN", "A2C"]
    ALL_DISTRIBUTION_CONFIGS = [
        {"name": "标准正态", "dist_type": "normal", "params": {"mu": 10.0, "sigma": 2.0}},
        {"name": "变动正态", "dist_type": "normal", "params": {"mu": 11.0, "sigma": 2.0}},
        {"name": "均匀分布", "dist_type": "uniform", "params": {"low": 0.0, "high": 20.0}},
    ]

    selected_algo_list = select_items_interactively(ALL_AVAILABLE_ALGORITHMS, "要运行的算法")
    run_hpo_comparison = True  # 自动启用HPO对比
    selected_initial_dist_list = select_items_interactively(ALL_DISTRIBUTION_CONFIGS, "初始训练分布")
    selected_target_scenarios_list = select_items_interactively(ALL_DISTRIBUTION_CONFIGS, "目标适应场景")

    plot_options = [
        {"id": "boxplots", "name": "自适应性能箱线图", "func": plot_adaptation_performance_boxplots,
         "args": [None, PLOTS_DIR]},
        {"id": "hpo_radar", "name": "高级HPO性能对比雷达图", "func": plot_aggregated_performance_radar,
         "args": [None, PLOTS_DIR, 'algorithm']},
        {"id": "hpo_barchart", "name": "HPO对比总体摘要柱状图", "func": plot_overall_summary_barchart_comparison,
         "args": [None, PLOTS_DIR, True]},
    ]
    selected_plots_desc = select_items_interactively(plot_options, "图表类型")

    print("\n" + "=" * 50)
    print("--- 配置完成，即将开始全自动模拟运行 ---")
    print(f"  算法: {', '.join(selected_algo_list)}")
    print(f"  将生成图表: {', '.join([p['name'] for p in selected_plots_desc])}")
    print("=" * 50 + "\n")

    all_runs_master_results = []

    # --- 2. 全自动实验循环 (使用模拟函数) ---
    for selected_algo in selected_algo_list:
        for selected_initial_dist in selected_initial_dist_list:
            global config_run  # 模拟全局变量
            for run_type in (["default_params", "hpo_params"] if run_hpo_comparison else ["default_params"]):
                config_run = {'run_type': run_type}
                print(f"\n--- 处理: {selected_algo} on {selected_initial_dist['name']} (类型: {run_type}) ---")

                model, _, train_time = train_agent_mock(selected_initial_dist, selected_algo, 4000, {})
                eval_base = evaluate_agent_mock(selected_initial_dist, model, 50, run_type=run_type)
                all_runs_master_results.append(
                    {'run_type': run_type, 'algorithm': selected_algo, 'scenario_type': 'baseline',
                     'final_eval': eval_base, 'initial_training_dist_name': selected_initial_dist['name'],
                     'initial_training_duration': train_time, 'adaptation_duration': 0.0})

                for target_scenario in selected_target_scenarios_list:
                    if target_scenario['name'] == selected_initial_dist['name']: continue
                    model_adapt, _, adapt_time = train_agent_mock(target_scenario, selected_algo, 2000, {})
                    eval_adapt = evaluate_agent_mock(target_scenario, model_adapt, 50, run_type=run_type)
                    all_runs_master_results.append(
                        {'run_type': run_type, 'algorithm': selected_algo, 'scenario_type': 'adaptation',
                         'final_eval': eval_adapt, 'initial_training_dist_name': selected_initial_dist['name'],
                         'initial_training_duration': train_time, 'adaptation_duration': adapt_time})

    print(f"\n{'=' * 80}\n--- 所有模拟实验运行完毕 ---\n{'=' * 80}\n")

    # --- 3. 结果汇总与自动化绘图 ---
    if all_runs_master_results:
        if selected_plots_desc:
            print("\n--- 根据您的预设选择，开始自动生成图表 ---")
            for plot_desc in selected_plots_desc:
                print(f"\n---\n正在生成: {plot_desc['name']}\n---")
                plot_desc["args"][0] = all_runs_master_results
                plot_desc["func"](*plot_desc["args"])
            print("\n--- 所有选定图表生成完毕 ---")


if __name__ == "__main__":
    run_complete_experiment()