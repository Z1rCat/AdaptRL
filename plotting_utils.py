# plotting_utils.py
import pandas as pd
from pandas import DataFrame as pd_DataFrame
from pandas import Series as pd_Series
import numpy as np
import re
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.container import BarContainer
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, cast
from scipy import stats
from collections import deque, defaultdict
import traceback
import colorsys

# 从 utils 导入，避免重复定义
from utils import sanitize_filename

# --- 核心改进：全局字体管理 ---
DEFAULT_FONT = 'Microsoft YaHei'
# 定义一个全局、统一的中文字体属性对象，强制应用到所有文本元素
try:
    CHINESE_FONT = fm.FontProperties(family=DEFAULT_FONT)
    # 启动时检查一次字体是否存在
    if DEFAULT_FONT not in [font.name for font in fm.fontManager.ttflist]:
        print(f"!!! 字体警告: 系统中未找到核心中文字体 '{DEFAULT_FONT}'。图表中的中文可能无法显示。 !!!")
except Exception as e:
    print(f"字体初始化失败: {e}。将使用Matplotlib默认字体。")
    CHINESE_FONT = fm.FontProperties() # Fallback

def plot_decorator(title_prefix: str = "", default_figsize: Tuple[int, int] = (12, 7), default_dpi: int = 120):
    def decorator(func):
        def wrapper(*args, **kwargs):
            fig, ax = plt.subplots(figsize=default_figsize, dpi=default_dpi)
            sns.set_style("whitegrid")
            plt.rcParams['axes.unicode_minus'] = False
            try:
                plot_info = func(fig, ax, *args, **kwargs)
                if not isinstance(plot_info, dict) or 'save_path' not in plot_info or 'title' not in plot_info:
                    plt.close(fig)
                    return
                save_path = plot_info['save_path']
                title = plot_info['title']
                complexity = plot_info.get('complexity')
                dist_name_for_title = plot_info.get('dist_name_for_title', '') # <-- 修复：恢复默认空字符串
                full_title = _get_title_with_complexity(
                    base_title=f"{title_prefix}{title}", 
                    complexity=complexity,
                    dist_name_for_title=dist_name_for_title
                )
                fig.suptitle(full_title, fontsize=TITLE_FONTSIZE, fontproperties=CHINESE_FONT, weight='bold')
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                if isinstance(ax, Axes):
                    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"图表已保存至: {save_path}")
            except Exception as e:
                print(f"!!!!!! 在执行绘图函数 '{func.__name__}' 时发生严重错误 !!!!!!")
                traceback.print_exc()
                plt.close(fig)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator

# --- Matplotlib全局美化和配置 ---
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10
GRID_STYLE = '--'
GRID_ALPHA = 0.6

def _get_title_with_complexity(base_title: str, complexity: Optional[float] = None, dist_name_for_title: str = "") -> str:
    if complexity is not None and complexity >= 0:
        if dist_name_for_title:
            return f"{base_title}\n(环境: {dist_name_for_title} | 复杂度: {complexity:.2f})"
        return f"{base_title} (复杂度: {complexity:.2f})"
    return base_title

@plot_decorator(default_figsize=(12, 7))
def plot_hpo_reward_distribution_panels(fig: Figure, ax: Axes, plots_dir: Path, results_data: List[Dict[str, Any]], plot_target: str = 'final_eval', plot_format: str = 'png'):
    if not results_data: return None
    df_results = pd.DataFrame(results_data)
    if df_results.empty: return None
    unique_combos = df_results[['algorithm', 'initial_training_dist_name']].drop_duplicates().to_dict(orient='records') # type: ignore
    for combo in unique_combos:
        df_combo_filtered = df_results[(df_results['algorithm'] == combo['algorithm']) & (df_results['initial_training_dist_name'] == combo['initial_training_dist_name'])]
        df_hpo, df_default = df_combo_filtered[df_combo_filtered['run_type'] == 'hpo_params'], df_combo_filtered[df_combo_filtered['run_type'] == 'default_params']
        if df_hpo.empty or df_default.empty: continue # type: ignore
        reward_col = 'final_eval' if plot_target == 'final_eval' else 'initial_eval'
        title_part = '最终评估' if plot_target == 'final_eval' else '适应前评估'
        hpo_rewards = [
            r for _, row in df_hpo.iterrows() # type: ignore
            if isinstance(row.get('hpo_trials_df'), pd_DataFrame)
            for r in row['hpo_trials_df']['user_attrs_' + ('final_adapted_reward' if plot_target == 'final_eval' else 'initial_reward')].dropna().tolist() # type: ignore
        ]
        default_rewards = [r for _, row in df_default.iterrows() for r in (row.get(reward_col).get('rewards_raw', []) if isinstance(row.get(reward_col), dict) else [])] # type: ignore
        if not hpo_rewards or not default_rewards: continue
        plot_data = pd.DataFrame([{'type': 'HPO Trials', 'reward': r} for r in hpo_rewards] + [{'type': 'Default Params', 'reward': r} for r in default_rewards])
        sns.boxplot(x='type', y='reward', data=plot_data, ax=ax, palette=['#4c72b0', '#55a868'], width=0.5)
        sns.stripplot(x='type', y='reward', data=plot_data, ax=ax, color=".25", size=3, alpha=0.6)
        try:
            _, p_val = stats.mannwhitneyu(hpo_rewards, default_rewards, alternative='two-sided')
            sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            y_max, y_min = plot_data['reward'].max(), plot_data['reward'].min()
            bar_h, tip_h = y_max + (y_max - y_min) * 0.07, y_max + (y_max - y_min) * 0.09
            ax.plot([0, 0, 1, 1], [bar_h, tip_h, tip_h, bar_h], lw=1.2, c='black')
            ax.text(0.5, tip_h, sig_text, ha='center', va='bottom', color='black', fontsize=12, weight='bold')
        except ValueError: pass
        stats_text = (f"HPO: N={len(hpo_rewards)}, Mean={np.mean(hpo_rewards):.2f}, Std={np.std(hpo_rewards):.2f}\n"
                      f"Default: N={len(default_rewards)}, Mean={np.mean(default_rewards):.2f}, Std={np.std(default_rewards):.2f}")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5), fontproperties=CHINESE_FONT)
        ax.set_xlabel('参数类型', fontproperties=CHINESE_FONT); ax.set_ylabel('单轮奖励', fontproperties=CHINESE_FONT)
        for label in ax.get_xticklabels(): label.set_fontproperties(CHINESE_FONT)
        title = f'HPO vs 默认参数奖励分布对比 ({title_part})'
        suffix = sanitize_filename(f"hpo_distrib_compare_{combo['algorithm']}_{combo['initial_training_dist_name']}_{plot_target}")
        save_path = plots_dir / f"{suffix}.{plot_format}"
        return {"title": title, "save_path": save_path, "dist_name_for_title": combo['initial_training_dist_name']}
    return None

@plot_decorator(default_figsize=(10, 10), title_prefix="综合性能雷达图")
def plot_aggregated_performance_radar(fig: Figure, ax: Axes, plots_dir: Path, results_data: List[Dict[str, Any]], metrics_to_plot: Optional[List[str]] = None, aggregation_level: str = 'algorithm', plot_format: str = 'png'):
    if not results_data: return None
    metrics = metrics_to_plot or ['m1_final_reward', 'm2_reward_improvement', 'm3_total_time', 'm4_final_correct_rate', 'm5_initial_stability', 'm6_adaptation_stability']
    processed = [{'algorithm': r.get('algorithm', 'N/A'), 'run_type': r.get('run_type', 'N/A'),
                  'initial_training_dist_name': r.get('initial_training_dist_name', 'N/A'),
                  'm1_final_reward': (r.get('final_eval') or {}).get('avg_reward', np.nan),
                  'm2_reward_improvement': ((r.get('final_eval') or {}).get('avg_reward', np.nan) - (r.get('initial_eval') or {}).get('avg_reward', np.nan)) if r.get('scenario_type') == 'adaptation' else np.nan,
                  'm3_total_time': -((r.get('initial_training_duration') or 0) + (r.get('adaptation_duration') or 0)),
                  'm4_final_correct_rate': ((r.get('final_eval') or {}).get('avg_correct_rate', np.nan)) * 100,
                  'm5_initial_stability': 1 / (1 + (r.get('initial_eval') or {}).get('std_reward', np.inf)),
                  'm6_adaptation_stability': 1 / (1 + (r.get('final_eval') or {}).get('std_reward', np.inf))} for r in results_data]
    if not processed: return None
    df = pd.DataFrame(processed)
    aggregated_df = df.groupby(['run_type', aggregation_level]).mean(numeric_only=True).reset_index()
    normalizer_map = {'m1_final_reward': (0, 50), 'm2_reward_improvement': (0, 40), 'm3_total_time': (-500, 0),
                      'm4_final_correct_rate': (0, 100), 'm5_initial_stability': (0, 1), 'm6_adaptation_stability': (0, 1)}
    normalized_df = aggregated_df.copy()
    for metric in metrics:
        if metric in normalized_df.columns and bool(normalized_df[metric].notna().any()):
            series = normalized_df[metric].fillna(normalized_df[metric].median())
            min_val, max_val = normalizer_map.get(metric, (series.min(), series.max()))
            normalized_df[metric] = ((series - min_val) / (max_val - min_val)) if (max_val - min_val) > 1e-6 else 0.5
        else: normalized_df[metric] = 0.0
        normalized_df[metric] = normalized_df[metric].clip(0, 1)
    metric_labels = {'m1_final_reward': '最终奖励', 'm2_reward_improvement': '奖励提升', 'm3_total_time': '执行效率\n(时间越短越好)',
                     'm4_final_correct_rate': '正确率', 'm5_initial_stability': '初始稳定性', 'm6_adaptation_stability': '适应后稳定性'}
    labels = [metric_labels.get(m, m) for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_closed = np.concatenate((angles, [angles[0]]))
    fig.clear(); ax = fig.add_subplot(111, polar=True)
    ax.set_xticks(angles); ax.set_xticklabels(labels, fontproperties=CHINESE_FONT, size=12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["25%", "50%", "75%", "100%"]); ax.set_ylim(0, 1)
    
    # SCI风格配色方案：为HPO使用Set2的鲜艳对比色，为Default统一使用中性灰色作为基准
    groups = np.unique(normalized_df[aggregation_level])
    hpo_colors = plt.cm.get_cmap('Set2', len(groups))
    default_color = '#808080'  # 统一的灰色基准

    for i, group in enumerate(groups):
        hpo_color = hpo_colors(i) # type: ignore

        # HPO 参数 (鲜艳颜色, 实线, 较粗)
        data_hpo = normalized_df[(normalized_df[aggregation_level] == group) & (normalized_df['run_type'] == 'hpo_params')]
        if not data_hpo.empty:
            values = data_hpo[metrics].fillna(0.5).values.flatten().tolist() # type: ignore
            values_closed = np.concatenate((values, [values[0]]))
            ax.plot(angles_closed, values_closed, color=hpo_color, linewidth=2.5, linestyle='-', label=f"{group} (HPO优化)")
            ax.fill(angles_closed, values_closed, color=hpo_color, alpha=0.25)

        # Default 参数 (统一灰色, 虚线, 较细)
        data_default = normalized_df[(normalized_df[aggregation_level] == group) & (normalized_df['run_type'] == 'default_params')]
        if not data_default.empty:
            values = data_default[metrics].fillna(0.5).values.flatten().tolist() # type: ignore
            values_closed = np.concatenate((values, [values[0]]))
            ax.plot(angles_closed, values_closed, color=default_color, linewidth=1.5, linestyle='--', label=f"{group} (默认参数)")
            ax.fill(angles_closed, values_closed, color=default_color, alpha=0.10)

    # 优化图例，确保顺序正确，更清晰
    handles, labels = ax.get_legend_handles_labels()
    # 通过标签中的 '(HPO优化)' 和 '(默认参数)' 进行排序，确保HPO在前，Default在后
    sorted_legend = sorted(zip(handles, labels), key=lambda x: (x[1].split(' ')[0], 'HPO' not in x[1]))
    sorted_handles, sorted_labels = zip(*sorted_legend) if sorted_legend else ([], [])
    ax.legend(handles=sorted_handles, labels=sorted_labels, loc='upper right', bbox_to_anchor=(1.35, 1.1), prop=CHINESE_FONT)

    title = f'(按{aggregation_level}聚合)'
    save_path = plots_dir / f"radar_perf_profile_{aggregation_level}.{plot_format}"
    return {"title": title, "save_path": save_path}

@plot_decorator(default_figsize=(14, 8))
def plot_overall_summary_barchart_comparison(fig: Figure, ax: Axes, plots_dir: Path, results_data: List[Dict[str, Any]], group_by_initial_dist: bool = True, plot_format: str = 'png'):
    if not results_data: return None
    df = pd.DataFrame(results_data)
    if df.empty: return None
    plot_data = [{'algorithm': r['algorithm'], 'initial_training_dist_name': r['initial_training_dist_name'],
                  'run_type': r['run_type'], 'scenario': f"{r['scenario_name']}\n(C: {r.get('complexity', -1):.2f})" if r.get('scenario_type') != 'baseline' else r['scenario_name'],
                  'reward': (r.get('final_eval') or {}).get('avg_reward')} for _, r in df.iterrows() if isinstance((r.get('final_eval') or {}).get('avg_reward'), (int, float))]
    if not plot_data: return None
    df_plot = pd.DataFrame(plot_data)
    grouping_col = 'initial_training_dist_name' if group_by_initial_dist else 'algorithm'
    if grouping_col not in df_plot.columns or df_plot[grouping_col].empty: return None
    group_val = df_plot[grouping_col].unique()[0]
    df_group = df_plot[df_plot[grouping_col] == group_val]
    if df_group.empty: return None
    
    # 使用安全的walrus操作符来提取复杂度，避免了对re.search结果的两次调用和潜在的None错误
    sorted_scenarios = sorted(
        df_group['scenario'].unique(), # type: ignore
        key=lambda s: ('基线' in str(s), float(match.group(1)) if (match := re.search(r'\(C: ([\d.-]+)\)', str(s))) else -1.0)
    )
    df_group['scenario_sorted'] = pd.Categorical(df_group['scenario'], categories=sorted_scenarios, ordered=True)
    
    sns.barplot(data=df_group, x='scenario_sorted', y='reward', hue='run_type', palette={'default_params': '#55a868', 'hpo_params': '#4c72b0'}, ax=ax, hue_order=['default_params', 'hpo_params']) # type: ignore
    for container in ax.containers:
        if isinstance(container, BarContainer): ax.bar_label(container, fmt='%.1f', label_type='edge', size=9, fontproperties=CHINESE_FONT)
    ax.set_xlabel('目标场景 / 基线 (按复杂度排序)', fontproperties=CHINESE_FONT, size=LABEL_FONTSIZE)
    ax.set_ylabel('平均奖励', fontproperties=CHINESE_FONT, size=LABEL_FONTSIZE)
    ax.tick_params(axis='x', labelrotation=15, labelsize=TICK_FONTSIZE)
    for label in ax.get_xticklabels(): label.set_fontproperties(CHINESE_FONT)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['默认参数' if lbl == 'default_params' else 'HPO优化' if lbl == 'hpo_params' else lbl for lbl in labels]
    legend = ax.legend(handles=handles, labels=new_labels, title='参数类型', prop=CHINESE_FONT)
    if legend.get_title(): legend.get_title().set_fontproperties(CHINESE_FONT)
    title = '默认参数 vs. HPO 性能对比'
    save_path = plots_dir / f"overall_summary_barchart_{group_val}_comparison.{plot_format}"
    return {"title": title, "save_path": save_path, "dist_name_for_title": group_val}

@plot_decorator(default_figsize=(12, 7))
def plot_learning_curves_from_logs(fig: Figure, ax: Axes, plots_dir: Path, log_file_path: Path, title: str, suffix: str, complexity: Optional[float] = None, plot_format: str = 'png'):
    try: log_data = pd.read_excel(log_file_path)
    except Exception: return None
    if log_data.empty: return None
    log_data['total_reward_smooth'] = log_data['total_reward'].rolling(window=20, min_periods=1).mean()
    ax.set_xlabel('训练轮数 (Episodes)', fontproperties=CHINESE_FONT)
    ax.set_ylabel('总奖励 (平滑后)', color='tab:blue', fontproperties=CHINESE_FONT)
    ax.plot(log_data.index, log_data['total_reward_smooth'], color='tab:blue', label='平滑奖励')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    if 'total_steps' in log_data.columns:
        ax2 = ax.twinx()
        ax2.set_ylabel('每轮步数 (Steps)', color='tab:green', fontproperties=CHINESE_FONT)
        ax2.plot(log_data.index, log_data['total_steps'], color='tab:green', alpha=0.6, linestyle='--', label='每轮步数')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', prop=CHINESE_FONT)
    else: ax.legend(loc='lower right', prop=CHINESE_FONT)
    save_path = plots_dir / f"learning_curve_{sanitize_filename(suffix)}.{plot_format}"
    dist_name_match = re.search(r'\((.*?)\)', title)
    dist_name = dist_name_match.group(1) if dist_name_match else ''
    return {"title": title, "save_path": save_path, "complexity": complexity, "dist_name_for_title": dist_name}

@plot_decorator(default_figsize=(14, 7))
def plot_combined_learning_curve(fig: Figure, ax: Axes, plots_dir: Path, initial_log_path: Path, adaptation_log_path: Path, algorithm_name: str, initial_dist_label: str, scenario_name_label: str, adapt_type: str, combined_suffix: str, initial_complexity: Optional[float] = None, adapt_complexity: Optional[float] = None, plot_format: str = 'png'):
    try:
        initial_log, adapt_log = pd.read_excel(initial_log_path), pd.read_excel(adaptation_log_path)
    except Exception: return None
    if initial_log.empty or adapt_log.empty: return None
    initial_log['phase'], adapt_log['phase'] = 'Initial Training', adapt_type
    initial_ep_count = len(initial_log)
    adapt_log['episode_global'] = adapt_log.index + initial_ep_count
    combined_log = pd.concat([initial_log, adapt_log], ignore_index=True)
    combined_log['reward_smooth'] = combined_log['total_reward'].rolling(window=20, min_periods=1).mean()
    sns.lineplot(data=combined_log, x=combined_log.index, y='reward_smooth', hue='phase', palette={'Initial Training': 'blue', adapt_type: 'red'}, ax=ax)
    ax.axvline(x=initial_ep_count - 1, color='black', linestyle='--', linewidth=2, label='环境改变')
    ax.set_xlabel('全局训练轮数 (Episodes)', fontproperties=CHINESE_FONT)
    ax.set_ylabel('平滑奖励', fontproperties=CHINESE_FONT)
    legend = ax.legend(title='训练阶段', prop=CHINESE_FONT)
    if legend.get_title(): legend.get_title().set_fontproperties(CHINESE_FONT)
    full_title = f'组合学习曲线: {algorithm_name}\n从 "{initial_dist_label}" (C:{initial_complexity:.2f}) 到 "{scenario_name_label}" (C:{adapt_complexity:.2f})'
    save_path = plots_dir / f"combined_lrn_crv_{sanitize_filename(combined_suffix)}.{plot_format}"
    return {"title": full_title, "save_path": save_path, "complexity": None, "dist_name_for_title": ""}

@plot_decorator(default_figsize=(10, 7))
def plot_complexity_vs_performance(fig: Figure, ax: Axes, plots_dir: Path, results_data: List[Dict[str, Any]], performance_metric_key: str = 'avg_reward', plot_type: str = 'adaptation_final', plot_format: str = 'png'):
    if not results_data: return None
    plot_data = []
    for r in results_data:
        eval_data = None
        if plot_type == 'baseline' and r.get('scenario_type') == 'baseline': eval_data = r.get('final_eval')
        elif plot_type == 'adaptation_initial' and r.get('scenario_type') == 'adaptation': eval_data = r.get('initial_eval')
        elif plot_type == 'adaptation_final' and r.get('scenario_type') == 'adaptation': eval_data = r.get('final_eval')
        if isinstance(eval_data, dict) and isinstance(eval_data.get(performance_metric_key), (int, float)):
            plot_data.append({'complexity': r.get('complexity', -1.0), 'performance': eval_data[performance_metric_key], 'algorithm': r['algorithm']})
    if not plot_data: return None
    df_plot = pd.DataFrame(plot_data)
    sns.scatterplot(data=df_plot, x='complexity', y='performance', hue='algorithm', style='algorithm', s=100, ax=ax)
    ax.set_xlabel('环境复杂度', fontproperties=CHINESE_FONT)
    ax.set_ylabel(performance_metric_key, fontproperties=CHINESE_FONT)
    legend = ax.legend(title='算法', prop=CHINESE_FONT)
    if legend.get_title(): legend.get_title().set_fontproperties(CHINESE_FONT)
    title = f'环境复杂度 vs. {performance_metric_key} ({plot_type})'
    save_path = plots_dir / f"cmplx_vs_{performance_metric_key}_{plot_type}.{plot_format}"
    return {"title": title, "save_path": save_path, "complexity": None}

@plot_decorator(default_figsize=(12, 7))
def plot_reward_distribution_over_training_stages(fig: Figure, ax: Axes, plots_dir: Path, log_file_path: Path, title_prefix: str, suffix: str, complexity: Optional[float] = None, num_stages: int = 4, plot_format: str = 'png'):
    try: log_data = pd.read_excel(log_file_path)
    except Exception: return None
    if log_data.empty: return None
    stage_len = len(log_data) // num_stages
    if stage_len == 0: return None
    stages_data = [{'stage': f'Stage {i+1}', 'reward': r} for i in range(num_stages) for r in log_data['total_reward'][i*stage_len:(i+1)*stage_len if i < num_stages-1 else len(log_data)]]
    sns.violinplot(x='stage', y='reward', data=pd.DataFrame(stages_data), ax=ax, inner='quartile')
    ax.set_xlabel('训练阶段', fontproperties=CHINESE_FONT); ax.set_ylabel('奖励', fontproperties=CHINESE_FONT)
    title = f'训练阶段奖励分布: {title_prefix}'
    match = re.search(r'\((.*?)\)', title_prefix)
    dist_name = match.group(1) if match else ''
    save_path = plots_dir / f"reward_dist_stages_{sanitize_filename(suffix)}.{plot_format}"
    return {"title": title, "save_path": save_path, "complexity": complexity, "dist_name_for_title": dist_name}

@plot_decorator(default_figsize=(12, 10))
def plot_convergence_metrics_over_training(fig: Figure, ax: Axes, plots_dir: Path, log_file_path: Path, title_prefix: str, suffix: str, complexity: Optional[float] = None, smoothing_window_reward: int = 20, metrics_calculation_window: int = 30, metrics_calculation_step: int = 10, fixed_slope_ylim: Optional[Tuple[float, float]] = (-0.25, 0.5), fixed_std_ylim: Optional[Tuple[float, float]] = (0, 20), plot_format: str = 'png'):
    try: log_data = pd.read_excel(log_file_path)
    except Exception: return None
    if log_data.empty or len(log_data['total_reward'].dropna()) < metrics_calculation_window: return None
    indices, slopes, std_devs = [], [], []
    window = deque(maxlen=metrics_calculation_window)
    for i, r in enumerate(log_data['total_reward'].dropna()):
        window.append(r)
        if i % metrics_calculation_step == 0 and len(window) == metrics_calculation_window:
            indices.append(i)
            std_devs.append(np.std(window))
            try: slopes.append(stats.linregress(np.arange(len(window)), list(window)).slope) # type: ignore
            except (ValueError, TypeError): slopes.append(0.0)
    if not indices: return None
    ax.remove()
    axes = fig.subplots(2, 1, sharex=True)
    axes[0].plot(indices, slopes, color='tab:green', label='奖励变化趋势 (斜率)')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[0].set_ylabel('线性回归斜率', fontproperties=CHINESE_FONT)
    axes[0].legend(prop=CHINESE_FONT); axes[0].grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    if fixed_slope_ylim: axes[0].set_ylim(fixed_slope_ylim)
    axes[1].plot(indices, std_devs, color='tab:red', label='奖励稳定性 (标准差)')
    axes[1].set_ylabel('奖励标准差', fontproperties=CHINESE_FONT)
    axes[1].set_xlabel('训练轮数', fontproperties=CHINESE_FONT)
    axes[1].legend(prop=CHINESE_FONT); axes[1].grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    if fixed_std_ylim: axes[1].set_ylim(fixed_std_ylim)
    title = f"收敛性指标动态图: {title_prefix}"
    match = re.search(r'\((.*?)\)', title_prefix)
    dist_name = match.group(1) if match else ''
    save_path = plots_dir / f"conv_metrics_dyn_{sanitize_filename(suffix)}.{plot_format}"
    return {"title": title, "save_path": save_path, "complexity": complexity, "dist_name_for_title": dist_name}

@plot_decorator(default_figsize=(12, 8))
def plot_adaptation_dumbbell(fig: Figure, ax: Axes, plots_dir: Path, results_data: List[Dict[str, Any]], initial_training_filter: str, plot_format: str = 'png'):
    if not results_data: return None
    df_full = pd.DataFrame(results_data)
    df_filtered = df_full[(df_full['initial_training_dist_name'] == initial_training_filter) & (df_full['scenario_type'] == 'adaptation')].copy()
    if df_filtered.empty: return None
    plot_data = [{'scenario': f"{r.get('scenario_name', 'N/A')}\n(类型: {r.get('adapt_type', 'N/A')})", 'algorithm': r.get('algorithm', 'N/A'),
                  'before': r['initial_eval']['avg_reward'], 'after': r['final_eval']['avg_reward']}
                 for _, r in df_filtered.iterrows() if isinstance(r.get('initial_eval'), dict) and isinstance(r.get('final_eval'), dict) and
                 isinstance(r['initial_eval'].get('avg_reward'), (float, int)) and isinstance(r['final_eval'].get('avg_reward'), (float, int))] # type: ignore
    if not plot_data: return None
    df_plot = pd.DataFrame(plot_data)
    if df_plot.empty: return None
    df_plot['before'] = pd.to_numeric(df_plot['before'])
    df_plot['after'] = pd.to_numeric(df_plot['after'])
    df_plot['delta'] = df_plot['after'] - df_plot['before']
    df_plot = df_plot.sort_values(by='delta', ascending=True).reset_index(drop=True)
    fig.set_figheight(max(6, len(df_plot) * 0.7))
    for i, row in df_plot.iterrows():
        before_val, after_val, delta = float(row['before']), float(row['after']), float(row['delta'])
        line_color = '#5cb85c' if delta >= 0 else '#d9534f'
        ax.plot([before_val, after_val], [i, i], color=line_color, alpha=0.7, lw=2, zorder=1) # type: ignore
        delta_text = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        text_color = '#3a7d3a' if delta >= 0 else '#9c4240'
        ax.text((before_val + after_val) / 2, i + 0.15, delta_text, ha='center', va='bottom', fontsize=8, color=text_color, weight='semibold', fontproperties=CHINESE_FONT) # type: ignore
    ax.scatter(df_plot['before'], df_plot.index, color='#d9534f', s=120, zorder=2, label='适应前', alpha=0.9)
    ax.scatter(df_plot['after'], df_plot.index, color='#5cb85c', s=120, zorder=2, label='适应后', alpha=0.9)
    x_min, x_max = float(df_plot['before'].min()), float(df_plot['after'].max())
    padding = (x_max - x_min) * 0.03 if (x_max - x_min) > 0 else 0.5
    for i in df_plot.index:
        before_val, after_val = float(df_plot.loc[i, 'before']), float(df_plot.loc[i, 'after'])
        ha_before, ha_after = ('right', 'left') if after_val >= before_val else ('left', 'right')
        ax.text(before_val - padding if ha_before == 'right' else before_val + padding, i, f'{before_val:.1f}', ha=ha_before, va='center', color='#d9534f', fontsize=9, weight='bold', fontproperties=CHINESE_FONT)
        ax.text(after_val + padding if ha_after == 'left' else after_val - padding, i, f'{after_val:.1f}', ha=ha_after, va='center', color='#5cb85c', fontsize=9, weight='bold', fontproperties=CHINESE_FONT)
    y_labels = [f"{str(r['scenario']).splitlines()[0]} - {str(r['algorithm'])}" for _, r in df_plot.iterrows()]
    ax.set_yticks(df_plot.index); ax.set_yticklabels(y_labels, fontproperties=CHINESE_FONT, fontsize=11)
    ax.set_xlabel('平均奖励', fontproperties=CHINESE_FONT, fontsize=LABEL_FONTSIZE); ax.set_ylabel('')
    
    # 优化图例：移动到图表顶部中心，水平排列，移除边框
    legend = ax.legend(
        prop=CHINESE_FONT,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),  # 将图例放置在绘图区域的上方
        ncol=len(ax.get_legend_handles_labels()[0]),  # 自动设置列数以实现水平布局
        frameon=False,  # 移除边框
        fontsize=LEGEND_FONTSIZE
    )
    
    if legend.get_title(): legend.get_title().set_fontproperties(CHINESE_FONT)
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    title = '适应性表现哑铃图'
    save_path = plots_dir / f"adaptation_dumbbell_{sanitize_filename(initial_training_filter)}.{plot_format}"
    return {"title": title, "save_path": save_path, "dist_name_for_title": initial_training_filter}

@plot_decorator(default_figsize=(12, 10))
def plot_correlation_heatmap(fig: Figure, ax: Axes, plots_dir: Path, results_data: List[Dict[str, Any]], plot_format: str = 'png'):
    if not results_data: return None
    df_adapt = pd.DataFrame(results_data)[lambda df: df['scenario_type'] == 'adaptation'].copy()
    if df_adapt.empty: return None
    metrics_data = [{'环境复杂度': r.get('complexity', np.nan), '初始奖励': (r.get('initial_eval') or {}).get('avg_reward', np.nan),
                     '最终奖励': (r.get('final_eval') or {}).get('avg_reward', np.nan),
                     '奖励提升量': ((r.get('final_eval') or {}).get('avg_reward', np.nan) - (r.get('initial_eval') or {}).get('avg_reward', np.nan)),
                     '最终正确率': (r.get('final_eval') or {}).get('avg_correct_rate', np.nan),
                     '初始稳定性': 1 / (1 + (r.get('initial_eval') or {}).get('std_reward', np.inf)),
                     '最终稳定性': 1 / (1 + (r.get('final_eval') or {}).get('std_reward', np.inf)),
                     '训练时长': r.get('initial_training_duration', np.nan), '适应时长': r.get('adaptation_duration', np.nan)}
                    for _, r in df_adapt.iterrows()]
    if not metrics_data: return None
    df_metrics = pd.DataFrame(metrics_data).dropna()
    if len(df_metrics) < 2: return None
    sns.heatmap(df_metrics.corr(), ax=ax, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, annot_kws={"size": 10})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontproperties=CHINESE_FONT)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontproperties=CHINESE_FONT)
    title = '性能指标相关性热力图'
    save_path = plots_dir / f"performance_correlation_heatmap.{plot_format}"
    return {"title": title, "save_path": save_path, "dist_name_for_title": "全局数据"}


@plot_decorator(default_figsize=(14, 8), title_prefix="复杂度 vs. 适应性表现")
def plot_complexity_vs_adaptation_performance(fig: Figure, ax: Axes, plots_dir: Path, results_data: List[Dict[str, Any]], performance_metric: str, plot_format: str = 'png'):
    """
    全新整合的图表，用箭头展示适应性表现的变化，并与基线进行对比。
    """
    if not results_data: return None

    # 1. 数据处理和准备
    plot_data = defaultdict(lambda: {'baseline': np.nan, 'initial': np.nan, 'final': np.nan, 'algorithm': ''})
    for r in results_data:
        complexity = r.get('complexity', -1.0)
        if complexity < 0: continue
        
        eval_data = r.get('final_eval')
        if not isinstance(eval_data, dict): continue
        perf_value = eval_data.get(performance_metric)
        if not isinstance(perf_value, (int, float)): continue

        key = (r['algorithm'], r['scenario_name'], complexity)
        plot_data[key]['algorithm'] = r['algorithm']
        
        if r.get('scenario_type') == 'baseline':
            plot_data[key]['baseline'] = perf_value
        elif r.get('scenario_type') == 'adaptation':
            # 这里的逻辑需要确保我们能同时获取到 initial 和 final 的数据
            # 假设一个 adaptation run 同时包含 initial_eval 和 final_eval
            initial_eval_data = r.get('initial_eval', {})
            initial_perf = initial_eval_data.get(performance_metric)
            if isinstance(initial_perf, (int, float)):
                 plot_data[key]['initial'] = initial_perf
            plot_data[key]['final'] = perf_value

    df_plot = pd.DataFrame([{
        'algorithm': v['algorithm'], 
        'scenario': k[1], 
        'complexity': k[2], 
        **v
    } for k, v in plot_data.items()]).dropna(subset=['initial', 'final'])
    
    if df_plot.empty:
        print(f"警告: 没有足够的数据来生成 '复杂度 vs. 适应性表现' 图表 (指标: {performance_metric})。")
        return None

    # 2. 绘图逻辑
    algorithms = df_plot['algorithm'].unique()
    colors = plt.cm.get_cmap('Set2', len(algorithms))
    algo_color_map = {algo: colors(i) for i, algo in enumerate(algorithms)} # type: ignore

    # 绘制箭头
    for _, row in df_plot.iterrows():
        color = '#5cb85c' if row['final'] >= row['initial'] else '#d9534f'
        ax.annotate(
            text='', 
            xy=(float(row['complexity']), float(row['final'])), # <-- 修复：显式转换为float
            xytext=(float(row['complexity']), float(row['initial'])), # <-- 修复：显式转换为float
            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.8", lw=1.5, color=color, shrinkA=3, shrinkB=3),
            zorder=2
        )

    # 绘制起点、终点和基准点
    ax.scatter(df_plot['complexity'], df_plot['initial'], c=[algo_color_map[a] for a in df_plot['algorithm']], marker='o', s=40, alpha=0.7, label='适应前', zorder=3)
    ax.scatter(df_plot['complexity'], df_plot['final'], c=[algo_color_map[a] for a in df_plot['algorithm']], marker='o', s=80, facecolors='none', edgecolors=[algo_color_map[a] for a in df_plot['algorithm']], lw=1.5, label='适应后', zorder=3)
    
    df_baseline = df_plot.dropna(subset=['baseline'])
    if not df_baseline.empty:
        ax.scatter(df_baseline['complexity'], df_baseline['baseline'], c=[algo_color_map[a] for a in df_baseline['algorithm']], marker='*', s=150, ec='black', lw=0.5, label='基线 (理想性能)', zorder=4)

    # 3. 图表美化
    ax.set_xlabel('环境复杂度', fontproperties=CHINESE_FONT, fontsize=LABEL_FONTSIZE)
    metric_label = '平均正确率' if performance_metric == 'avg_correct_rate' else '平均奖励'
    ax.set_ylabel(metric_label, fontproperties=CHINESE_FONT, fontsize=LABEL_FONTSIZE)

    # 创建自定义图例
    legend_elements = [
        Line2D([0], [0], color='gray', marker='o', markersize=6, alpha=0.7, linestyle='None', label='适应前'),
        Line2D([0], [0], color='gray', marker='o', markersize=8, markerfacecolor='none', linestyle='None', label='适应后'),
        Line2D([0], [0], color='gray', marker='*', markersize=12, markeredgecolor='black', linestyle='None', label='基线 (理想)'),
        Patch(facecolor='#5cb85c', edgecolor='#5cb85c', label='性能提升', alpha=0.7),
        Patch(facecolor='#d9534f', edgecolor='#d9534f', label='性能下降', alpha=0.7)
    ]
    # 为每个算法创建一个图例条目
    for algo, color in algo_color_map.items():
        legend_elements.append(Patch(facecolor=color, label=f'算法: {algo}', alpha=0.8)) # type: ignore

    legend = ax.legend(handles=legend_elements, prop=CHINESE_FONT, loc='best')
    if legend.get_title():
        legend.get_title().set_fontproperties(CHINESE_FONT)

    title = f'({metric_label})'
    save_path = plots_dir / f"cmplx_vs_adapt_perf_{performance_metric}.{plot_format}"
    return {"title": title, "save_path": save_path}


@plot_decorator()
def plot_adaptation_delta_rewards(fig: Figure, ax: Axes, plots_dir: Path, results_data: List[Dict[str, Any]], initial_training_filter: Optional[str] = None, plot_format: str = 'png'):
    if not results_data: return None
    df_full = pd.DataFrame(results_data)
    df_filtered = df_full[df_full['initial_training_dist_name'] == initial_training_filter] if initial_training_filter else df_full
    delta_data = [{'场景': f"{r['scenario_name']} ({r['adapt_type']})", '算法': r['algorithm'],
                   '奖励变化量': r['final_eval']['avg_reward'] - r['initial_eval']['avg_reward']}
                  for _, r in df_filtered.iterrows() if r.get('scenario_type') == 'adaptation' and isinstance(r.get('initial_eval'), dict) and
                  isinstance(r.get('final_eval'), dict) and isinstance(r['initial_eval'].get('avg_reward'), (float, int)) and # type: ignore
                  isinstance(r['final_eval'].get('avg_reward'), (float, int))] # type: ignore
    if not delta_data: return None
    df_delta = pd.DataFrame(delta_data)
    fig.set_figheight(max(6, len(df_delta['场景'].unique()) * 0.5))
    sns.barplot(y='场景', x='奖励变化量', hue='算法', data=df_delta, ax=ax, orient='h')
    ax.set_xlabel('平均奖励变化量', fontproperties=CHINESE_FONT)
    ax.set_ylabel('目标场景 (适应类型)', fontproperties=CHINESE_FONT)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.8)
    ax.grid(True, axis='x', linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    legend = ax.legend(title='算法', prop=CHINESE_FONT)
    if legend.get_title(): legend.get_title().set_fontproperties(CHINESE_FONT)
    title = '适应后平均奖励变化量'
    filename_suffix = f"_init_{sanitize_filename(initial_training_filter)}" if initial_training_filter else "_all"
    save_path = plots_dir / f"adapt_delta_rewards{filename_suffix}.{plot_format}"
    return {"title": title, "save_path": save_path, "dist_name_for_title": initial_training_filter or "所有环境"}