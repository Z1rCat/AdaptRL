# plotting_utils.py
import matplotlib.pyplot as plt
from math import pi
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
from scipy import stats
import scipy.stats as stats # For statistical tests
import matplotlib.gridspec as gridspec
try:
    from utils import sanitize_filename, \
        _get_title_with_complexity  # Assuming _get_title_with_complexity is in utils now
except ImportError:
    # Fallback sanitize_filename, consistent with your new provided code and original
    def sanitize_filename(filename: str) -> str:
        sanitized = re.sub(r'[^\w\s\.\-\(\)一-龥]', '', filename)  # Chinese characters included
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('._- ')
        return sanitized


    def _get_title_with_complexity(base_title: str, complexity: float = None, dist_name_for_title: str = "") -> str:
        comp_str = ""
        if complexity is not None and complexity >= 0:
            comp_str = f" (C: {complexity:.2f})"  # Changed from Cmplx for consistency
        # Avoid duplicating dist_name if already in base_title
        if dist_name_for_title and dist_name_for_title.lower() in base_title.lower():
            return f"{base_title}{comp_str}"
        elif dist_name_for_title:
            return f"{base_title} - {dist_name_for_title}{comp_str}"
        return f"{base_title}{comp_str}"

# --- Style constants ---
COLOR_INITIAL_TRAIN = 'dodgerblue'
COLOR_ADAPTATION_PHASE = 'orangered'
COLOR_REWARD_RAW = 'lightcoral'  # Used for raw reward line
COLOR_REWARD_SMOOTHED = 'crimson'  # Used for smoothed reward line on primary axis
COLOR_CORRECT_RATE_SMOOTHED = 'darkcyan'  # Changed for better contrast
COLOR_BOXPLOT_INITIAL = '#a9d1f7'  # Light blue
COLOR_BOXPLOT_FINAL = '#f7c8a9'  # Light orange
SEABORN_PALETTE_BARCHART = "viridis"
SEABORN_PALETTE_SCATTER = "deep"
GRID_STYLE = '--'  # Changed from ':' for better visibility
GRID_ALPHA = 0.6  # Slightly more visible grid
RAW_LINE_ALPHA = 0.35  # Slightly less opaque raw line
SMOOTH_LINEWIDTH = 2.2
MARKER_SIZE = 4
TITLE_FONTSIZE = 15
SUBPLOT_TITLE_FONTSIZE = 12
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 9
SUPTITLE_FONTSIZE = 16

# Colors for Radar Chart
COLOR_DEFAULT_RADAR = 'dimgrey'        # Dark grey for default
COLOR_HPO_RADAR = 'deepskyblue'    # Bright blue for HPO
COLOR_HPO_IMPROVEMENT = 'forestgreen' # Green for HPO improvement
COLOR_HPO_DECLINE = 'orangered'     # Orange/Red for HPO decline
import scipy.stats as stats  # For statistical tests
import matplotlib.gridspec as gridspec  # For more complex subplot layouts if needed


def plot_hpo_reward_distribution_panels(
        results_data: List[Dict[str, Any]],
        plots_dir: Path,
        # group_by_keys: List[str] = ['algorithm', 'initial_training_dist_name'], # Keys to create a new figure for
        # panel_key: str = 'scenario_name' # Key to create panels within a figure
        plot_target: str = 'final_eval'  # 'final_eval' or 'initial_eval' (for adaptation scenarios)
):
    """
    为每个 (算法, 初始训练分布) 组合创建一个图。
    图中的每个面板展示一个特定场景下，使用默认参数与HPO参数时的奖励分布对比。
    使用小提琴图结合箱线图和散点图（类似雨云图）。
    """
    if not results_data:
        print("分布对比图警告: 无数据可用于绘制。")
        return

    df_results = pd.DataFrame(results_data)

    # Ensure necessary columns exist
    required_cols = ['algorithm', 'initial_training_dist_name', 'scenario_name', 'run_type', plot_target]
    if not all(col in df_results.columns for col in required_cols):
        print(f"分布对比图警告: 结果数据缺少必要列 (如 {', '.join(required_cols)})。")
        return

    # Get unique combinations of algorithm and initial_training_dist_name
    unique_algo_initial_dist_combos = df_results[['algorithm', 'initial_training_dist_name']].drop_duplicates().to_dict(
        'records')

    for combo in unique_algo_initial_dist_combos:
        algo = combo['algorithm']
        initial_dist = combo['initial_training_dist_name']

        # Filter data for the current algorithm and initial distribution
        current_combo_df = df_results[
            (df_results['algorithm'] == algo) &
            (df_results['initial_training_dist_name'] == initial_dist)
            ]

        if current_combo_df.empty:
            continue

        # Get unique scenarios for this combination
        scenarios = current_combo_df['scenario_name'].unique()
        num_scenarios = len(scenarios)
        if num_scenarios == 0:
            continue

        # Determine subplot layout (e.g., try to make it somewhat square or 2-3 columns)
        # cols = min(3, num_scenarios)
        cols = min(2, num_scenarios) if num_scenarios > 1 else 1  # Max 2 columns for better readability per panel
        rows = (num_scenarios + cols - 1) // cols

        fig_height = max(6, rows * 5)  # Adjust height based on number of rows
        fig_width = max(8, cols * 7)  # Adjust width based on number of columns

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()  # Flatten to 1D array for easy iteration

        fig.suptitle(f"HPO参数效果对比: {algo} - 初始于 {initial_dist}\n({plot_target} 奖励分布)",
                     fontsize=SUPTITLE_FONTSIZE, fontweight='bold', y=1.0 if rows * cols > 1 else 1.05)

        plot_idx = 0
        for i_ax, scenario_name in enumerate(scenarios):
            ax = axes[i_ax]

            scenario_data_default = current_combo_df[
                (current_combo_df['scenario_name'] == scenario_name) &
                (current_combo_df['run_type'] == 'default_params')
                ]
            scenario_data_hpo = current_combo_df[
                (current_combo_df['scenario_name'] == scenario_name) &
                (current_combo_df['run_type'] == 'hpo_params')
                ]

            rewards_default_raw = []
            if not scenario_data_default.empty and isinstance(scenario_data_default.iloc[0][plot_target], dict):
                rewards_default_raw = scenario_data_default.iloc[0][plot_target].get('rewards_raw', [])

            rewards_hpo_raw = []
            if not scenario_data_hpo.empty and isinstance(scenario_data_hpo.iloc[0][plot_target], dict):
                rewards_hpo_raw = scenario_data_hpo.iloc[0][plot_target].get('rewards_raw', [])

            # Create a DataFrame for plotting this panel
            plot_panel_data = []
            if rewards_default_raw:
                for r in rewards_default_raw: plot_panel_data.append({'参数类型': '默认参数', '奖励': r})
            if rewards_hpo_raw:
                for r in rewards_hpo_raw: plot_panel_data.append({'参数类型': 'HPO参数', '奖励': r})

            if not plot_panel_data:
                ax.text(0.5, 0.5, "无足够数据", horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                ax.set_title(f"{scenario_name}\n(数据不足)", fontsize=SUBPLOT_TITLE_FONTSIZE - 1)
                plot_idx += 1
                continue

            df_panel = pd.DataFrame(plot_panel_data)

            # --- Raincloud components ---
            # 1. Violin plot for density
            sns.violinplot(x='参数类型', y='奖励', data=df_panel, ax=ax,
                           palette={"默认参数": COLOR_DEFAULT_RADAR, "HPO参数": COLOR_HPO_RADAR},
                           inner=None,  # We'll draw box and points separately for more control
                           cut=0, linewidth=1.5, scale='width')  # scale='width' makes violins have same width

            # 2. Box plot (narrower, inside or beside violin)
            # For placing it nicely with violin, often need to adjust positions or use a library that combines them.
            # Here, we'll overlay a narrower boxplot.
            sns.boxplot(x='参数类型', y='奖励', data=df_panel, ax=ax,
                        palette={"默认参数": "#FFFFFF", "HPO参数": "#FFFFFF"},  # White boxes to see violin underneath
                        width=0.3, showfliers=False,  # No fliers, stripplot will show them
                        boxprops=dict(alpha=0.7, zorder=2, edgecolor='black'),
                        medianprops=dict(color='black', linewidth=2, zorder=3),
                        whiskerprops=dict(color='black', linewidth=1.5, zorder=2),
                        capprops=dict(color='black', linewidth=1.5, zorder=2))

            # 3. Strip plot for individual data points (jittered)
            sns.stripplot(x='参数类型', y='奖励', data=df_panel, ax=ax,
                          palette={"默认参数": COLOR_DEFAULT_RADAR, "HPO参数": COLOR_HPO_RADAR},
                          dodge=True, alpha=0.5, jitter=0.2, size=3.5, zorder=1)

            # --- Statistical Annotation ---
            mean_default = df_panel[df_panel['参数类型'] == '默认参数']['奖励'].mean()
            mean_hpo = df_panel[df_panel['参数类型'] == 'HPO参数']['奖励'].mean()

            ax.text(0.25, 0.02, f"均值: {mean_default:.1f}", ha='center', transform=ax.transAxes,
                    fontsize=TICK_FONTSIZE - 1)
            ax.text(0.75, 0.02, f"均值: {mean_hpo:.1f}", ha='center', transform=ax.transAxes,
                    fontsize=TICK_FONTSIZE - 1)

            # Perform statistical test (e.g., Mann-Whitney U for non-parametric)
            p_value_text = ""
            if rewards_default_raw and rewards_hpo_raw and len(rewards_default_raw) > 1 and len(rewards_hpo_raw) > 1:
                try:
                    # Ensure no NaNs if they can occur
                    stat, p_value = stats.mannwhitneyu([r for r in rewards_default_raw if not np.isnan(r)],
                                                       [r for r in rewards_hpo_raw if not np.isnan(r)],
                                                       alternative='two-sided')  # or 'less'/'greater' if directional hypothesis
                    if p_value < 0.001:
                        p_value_text = "p < 0.001"
                    elif p_value < 0.01:
                        p_value_text = "p < 0.01"
                    elif p_value < 0.05:
                        p_value_text = "p < 0.05"
                    else:
                        p_value_text = f"p = {p_value:.2f}"
                except ValueError as e:  # e.g. all values are identical
                    p_value_text = "检验无法进行"
                    # print(f"Stat test error for {scenario_name}: {e}")

            complexity = current_combo_df[current_combo_df['scenario_name'] == scenario_name]['complexity'].iloc[0]
            title_suffix = f" (C: {complexity:.2f})" if complexity >= 0 else ""
            ax.set_title(f"{scenario_name}{title_suffix}\n{p_value_text}", fontsize=SUBPLOT_TITLE_FONTSIZE)

            ax.set_ylabel("最终评估奖励", fontsize=LABEL_FONTSIZE)
            ax.set_xlabel("")  # X-axis label is clear from categories
            ax.tick_params(axis='x', labelsize=LABEL_FONTSIZE - 1)
            ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
            ax.set_ylim(max(-5, df_panel['奖励'].min() - 5 if not df_panel.empty else -5),
                        min(60, df_panel['奖励'].max() + 5 if not df_panel.empty else 60))  # Dynamic Y Lim
            ax.grid(axis='y', linestyle=GRID_STYLE, alpha=GRID_ALPHA - 0.1)
            plot_idx += 1

        # Remove any unused subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95 if rows * cols > 1 else 0.92])  # Adjust for suptitle and legend

        # Save the figure
        s_algo = sanitize_filename(algo)
        s_initial_dist = sanitize_filename(initial_dist)
        s_plot_target = sanitize_filename(plot_target)
        plot_filename = f"hpo_distrib_compare_{s_algo}_init_{s_initial_dist}_{s_plot_target}.png"
        plot_path = plots_dir / plot_filename
        try:
            plt.savefig(plot_path, dpi=300)
            print(f"HPO奖励分布对比图已保存至: {plot_path}")
        except Exception as e_save:
            print(f"保存图表 {plot_path} 失败: {e_save}")
        plt.close(fig)
def plot_aggregated_performance_radar(
        results_data: List[Dict[str, Any]],
        plots_dir: Path,
        metrics_to_plot: Optional[List[str]] = None,
        aggregation_level: str = 'algorithm',
        theta_offset_degrees: float = 0  # New: Angle offset in degrees
):
    """
    绘制聚合性能指标雷达图，比较默认参数和HPO参数在多个指标上的表现。
    - 不进行数据归一化，直接显示聚合后的原始（或均值）指标。
    - 包含轴标签。
    - 改进的Y轴刻度和网格。
    - 增强的数据系列区分度。
    - 优化的调色板和美学。
    - 可选的角度偏移。
    """
    if not results_data:
        print("雷达图警告: 无数据可用于绘制。")
        return

    df = pd.DataFrame(results_data)
    required_radar_cols = [aggregation_level, 'run_type', 'final_eval', 'initial_eval', 'scenario_type']
    if not all(col in df.columns for col in required_radar_cols):
        print(f"雷达图警告: 结果数据缺少必要字段 for radar plot (e.g., {aggregation_level}, run_type).")
        # return # Allow to proceed if some metrics can still be derived

    raw_metrics_data = []
    for _, row in df.iterrows():
        if not all(k in row for k in [aggregation_level, 'run_type', 'scenario_type']) or \
                not isinstance(row.get('final_eval'), dict) or \
                not isinstance(row.get('initial_eval'), dict):
            continue

        m1_final_reward = row['final_eval'].get('avg_reward', np.nan)
        m2_reward_improvement = np.nan
        if row['scenario_type'] == 'adaptation':  # Improvement only for adaptation scenarios
            initial_reward_on_target = row['initial_eval'].get('avg_reward', np.nan)
            if not np.isnan(m1_final_reward) and not np.isnan(initial_reward_on_target):
                m2_reward_improvement = m1_final_reward - initial_reward_on_target

        initial_duration = float(row.get('initial_training_duration', 0.0) or 0.0)
        adaptation_duration = float(row.get('adaptation_duration', 0.0) or 0.0) if row[
                                                                                       'scenario_type'] == 'adaptation' else 0.0
        m3_total_time = initial_duration + adaptation_duration
        if m3_total_time == 0 and not (np.isnan(initial_duration) and np.isnan(adaptation_duration)):
            m3_total_time = np.nan

        m4_final_correct_rate_raw = row['final_eval'].get('avg_correct_rate', np.nan)
        m4_final_correct_rate_perc = m4_final_correct_rate_raw * 100 if not np.isnan(
            m4_final_correct_rate_raw) else np.nan

        raw_metrics_data.append({
            aggregation_level: row[aggregation_level], 'run_type': row['run_type'],
            'M1_AvgFinalReward': m1_final_reward, 'M2_AvgRewardImprovement': m2_reward_improvement,
            'M3_AvgTotalTime_s': m3_total_time, 'M4_AvgFinalCorrectRate_perc': m4_final_correct_rate_perc,
        })

    if not raw_metrics_data: print("雷达图警告: 提取指标后无有效数据。"); return
    metrics_df = pd.DataFrame(raw_metrics_data)
    metrics_df = metrics_df[metrics_df['run_type'].isin(['hpo_params', 'default_params'])]
    if metrics_df.empty: print(f"雷达图警告: 筛选 run_type 后无数据。"); return

    agg_funcs = {
        'M1_AvgFinalReward': 'mean', 'M2_AvgRewardImprovement': lambda x: np.nanmean(x.astype(float)),
        'M3_AvgTotalTime_s': 'mean', 'M4_AvgFinalCorrectRate_perc': 'mean'
    }
    aggregated_df = metrics_df.groupby([aggregation_level, 'run_type']).agg(agg_funcs).reset_index()

    metric_display_info = {
        'M1_AvgFinalReward': {"label": "平均最终奖励", "higher_is_better": True},
        'M2_AvgRewardImprovement': {"label": "平均奖励提升", "higher_is_better": True},
        'M3_AvgTotalTime_s': {"label": "平均总训练时长 (s)\n(越低越好)", "higher_is_better": False},
        # Indicate direction
        'M4_AvgFinalCorrectRate_perc': {"label": "平均最终正确率 (%)", "higher_is_better": True}
    }

    if metrics_to_plot is None: metrics_to_plot = list(metric_display_info.keys())
    metrics_to_plot = [m for m in metrics_to_plot if m in aggregated_df.columns and m in metric_display_info]
    if not metrics_to_plot or len(metrics_to_plot) < 3:
        print(f"雷达图警告: 有效指标少于3个 ({len(metrics_to_plot)})，无法绘制。");
        return

    for metric in metrics_to_plot:  # Fill NaNs for plotting
        aggregated_df[metric] = aggregated_df[metric].fillna(
            aggregated_df[metric].median() if not aggregated_df[metric].isnull().all() else 0)

    unique_groups = aggregated_df[aggregation_level].unique()

    for group_val in unique_groups:
        group_data = aggregated_df[aggregated_df[aggregation_level] == group_val]
        default_data_row = group_data[group_data['run_type'] == 'default_params']
        hpo_data_row = group_data[group_data['run_type'] == 'hpo_params']

        if default_data_row.empty and hpo_data_row.empty: continue

        # Use only metrics available in this group's data (should be all from metrics_to_plot due to fillna)
        current_metrics_for_plot = [m for m in metrics_to_plot if m in group_data.columns]
        if len(current_metrics_for_plot) < 3: continue

        axis_labels = [metric_display_info[m]["label"] for m in current_metrics_for_plot]
        num_vars = len(axis_labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(11, 10), subplot_kw=dict(polar=True))  # Slightly wider for labels
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f9f9f9')  # Light grey background for plot area

        # Determine overall Y-axis range for this specific plot
        all_vals_for_scale = []
        if not default_data_row.empty: all_vals_for_scale.extend(
            default_data_row[current_metrics_for_plot].iloc[0].values)
        if not hpo_data_row.empty: all_vals_for_scale.extend(hpo_data_row[current_metrics_for_plot].iloc[0].values)
        all_vals_for_scale = [v for v in all_vals_for_scale if not np.isnan(v)]

        if not all_vals_for_scale:
            y_min, y_max = 0, 10  # Fallback
        else:
            y_min_data = min(all_vals_for_scale)
            y_max_data = max(all_vals_for_scale)
            y_min = 0 if y_min_data >= 0 else y_min_data - abs(
                y_min_data * 0.1) - 0.1  # Ensure 0 is included or padding
            y_max = y_max_data + abs(y_max_data * 0.1) + 0.1  # Add padding
            if y_min == y_max: y_min -= 0.5; y_max += 0.5

        # Use MaxNLocator for "nice" tick values on the radial axis
        y_locator = mticker.MaxNLocator(nbins=5, prune='both', min_n_ticks=4)
        y_ticks = y_locator.tick_values(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_ticks[0], y_ticks[-1])  # Set ylim to actual tick range

        # Plotting function
        def add_series_to_radar(data_series, color, line_label, metrics_list, angle_list_closed, is_hpo=False,
                                default_series=None):
            if data_series.empty: return

            values = data_series[metrics_list].iloc[0].values.flatten().tolist()
            values_closed = values + values[:1]

            ax.plot(angle_list_closed, values_closed, color=color, linewidth=2.2, linestyle='-', label=line_label,
                    zorder=3)
            ax.fill(angle_list_closed, values_closed, color=color, alpha=0.2, zorder=2)

            # Add markers, color them if HPO and highlighting gaps
            for i, value in enumerate(values):
                marker_color = color
                if is_hpo and default_series is not None and not default_series.empty:
                    default_value = default_series[metrics_list].iloc[0, i]
                    higher_is_better = metric_display_info[metrics_list[i]]["higher_is_better"]
                    if not np.isnan(value) and not np.isnan(default_value):
                        if (higher_is_better and value > default_value) or \
                                (not higher_is_better and value < default_value):
                            marker_color = COLOR_HPO_IMPROVEMENT
                        elif (higher_is_better and value < default_value) or \
                                (not higher_is_better and value > default_value):
                            marker_color = COLOR_HPO_DECLINE

                ax.plot(angle_list_closed[i], value, marker='o', markersize=7, color=marker_color,
                        markeredgecolor='white', markeredgewidth=0.5, zorder=4)

        add_series_to_radar(default_data_row, COLOR_DEFAULT_RADAR, 'Default Params', current_metrics_for_plot,
                            angles_closed)
        add_series_to_radar(hpo_data_row, COLOR_HPO_RADAR, 'HPO Params', current_metrics_for_plot, angles_closed,
                            is_hpo=True, default_series=default_data_row)

        ax.set_theta_offset(pi / 2 + np.deg2rad(theta_offset_degrees))  # Apply offset
        ax.set_theta_direction(-1)

        ax.set_xticks(angles)
        ax.set_xticklabels(axis_labels, fontsize=LABEL_FONTSIZE - 2, color="#333333", ha='center', va='center')

        # Adjust spoke label positions to be further out and avoid overlap
        for i, label in enumerate(ax.get_xticklabels()):
            angle_rad = angles[i] + np.deg2rad(theta_offset_degrees) + pi / 2
            if angle_rad > pi / 2 and angle_rad < 3 * pi / 2:  # Labels on the left
                label.set_horizontalalignment('right')
            elif angle_rad < pi / 2 or angle_rad > 3 * pi / 2:  # Labels on the right
                label.set_horizontalalignment('left')
            # Add a radial offset to push labels further from the plot if needed
            label.set_position((label.get_position()[0], y_ticks[-1] * 1.15))  # Relative to max Y tick

        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        for yt_label in ax.get_yticklabels():
            yt_label.set_fontsize(TICK_FONTSIZE - 1)
            yt_label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0))
        ax.tick_params(axis='y', pad=10)  # Pad y-tick labels from axis

        title_str = f'HPO性能对比雷达图 ({aggregation_level}: {group_val})\n(原始均值, 未归一化)'
        plt.title(title_str, size=TITLE_FONTSIZE, y=1.12, color="navy", fontweight='bold')

        # Legend with improved markers and placement
        handles, labels = ax.get_legend_handles_labels()
        # Create proxy artists for legend if markers are dynamically colored
        proxy_default = plt.Line2D([0], [0], linestyle='-', color=COLOR_DEFAULT_RADAR, marker='o', markersize=7,
                                   label='Default Params')
        proxy_hpo = plt.Line2D([0], [0], linestyle='-', color=COLOR_HPO_RADAR, marker='o', markersize=7,
                               label='HPO Params')
        # Proxy for HPO improvement/decline markers
        proxy_improve = plt.Line2D([0], [0], linestyle='none', marker='o', markersize=7, color=COLOR_HPO_IMPROVEMENT,
                                   label='HPO 优于 Default')
        proxy_decline = plt.Line2D([0], [0], linestyle='none', marker='o', markersize=7, color=COLOR_HPO_DECLINE,
                                   label='HPO 劣于 Default')

        ax.legend(handles=[proxy_default, proxy_hpo, proxy_improve, proxy_decline],
                  loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2,
                  fontsize=LEGEND_FONTSIZE, frameon=True, facecolor='white', edgecolor='silver')

        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA, color='silver', linewidth=0.7)
        ax.spines['polar'].set_color('black')
        ax.spines['polar'].set_linewidth(1.2)

        plt.tight_layout(pad=3.0)  # Increased padding

        plot_filename = f"radar_perf_RAW_{sanitize_filename(str(group_val))}_{sanitize_filename(aggregation_level)}.png"
        plot_path = plots_dir / plot_filename
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"雷达图已保存至: {plot_path}")
        except Exception as e_save:
            print(f"保存雷达图 {plot_path} 失败: {e_save}")
        plt.close(fig)


def plot_overall_summary_barchart_comparison(
        results_data: List[Dict[str, Any]],
        plots_dir: Path,
        group_by_initial_dist: bool = True
):
    if not results_data:
        print("绘图警告(对比柱状图): 无数据可用于绘制。")
        return

    df_results = pd.DataFrame(results_data)
    required_cols = ['initial_training_dist_name', 'algorithm', 'scenario_name', 'final_eval', 'run_type']
    if not all(col in df_results.columns for col in required_cols):
        print("绘图错误(对比柱状图): 结果数据缺少必要的列 (例如 'run_type')。")
        return

    plot_data_list = []
    for _, row in df_results.iterrows():
        final_eval = row['final_eval']
        if isinstance(final_eval, dict) and isinstance(final_eval.get('avg_reward'), (int, float)) and not np.isnan(
                final_eval.get('avg_reward')):
            complexity = row.get('complexity', -1.0)
            scenario_label = f"{row['scenario_name']}"
            if complexity >= 0 and row.get('scenario_type') != 'baseline':
                scenario_label += f"\n(C: {complexity:.2f})"

            plot_data_list.append({
                '算法': row['algorithm'],
                '初始训练分布': row['initial_training_dist_name'],
                '目标场景/基线': scenario_label,
                '最终平均奖励': final_eval['avg_reward'],
                '运行类型': row.get('run_type', '未知运行')
            })

    if not plot_data_list:
        print("绘图警告(对比柱状图): 没有有效的最终评估数据。")
        return

    summary_df = pd.DataFrame(plot_data_list)
    initial_dist_groups = summary_df['初始训练分布'].unique() if group_by_initial_dist else [None]

    for initial_dist_name_filter in initial_dist_groups:
        current_summary_df = summary_df.copy()  # Start with full df
        plot_suffix = "overall_comparison"
        title_suffix_detail = "所有初始分布"

        if initial_dist_name_filter:
            current_summary_df = summary_df[summary_df['初始训练分布'] == initial_dist_name_filter]
            if current_summary_df.empty: continue
            plot_suffix = f"{sanitize_filename(initial_dist_name_filter)}_comparison"
            title_suffix_detail = f"初始于: {initial_dist_name_filter}"

        if current_summary_df.empty: continue

        current_summary_df['图例标签'] = current_summary_df['算法'] + " [" + current_summary_df['运行类型'] + "]"

        def sort_key_scenario(scenario_str):
            parts = scenario_str.split('\n(C: ')
            name_part = parts[0]
            complexity_part = -1.0
            if len(parts) > 1:
                try:
                    complexity_part = float(parts[1].rstrip(')'))
                except ValueError:
                    pass
            return (name_part, complexity_part)

        ordered_x_labels = sorted(current_summary_df['目标场景/基线'].unique(), key=sort_key_scenario)

        num_scenarios = len(ordered_x_labels)
        num_hues = len(current_summary_df['图例标签'].unique())
        # Dynamic width: base + per_scenario + per_hue_within_scenario + legend_space
        fig_width = max(10, num_scenarios * (0.5 + num_hues * 0.15) + 3.5)

        plt.figure(figsize=(fig_width, 8))
        palette = sns.color_palette("Paired", n_colors=num_hues)  # "Paired" is good for categorical hues

        sns.barplot(
            x='目标场景/基线', y='最终平均奖励', hue='图例标签',
            data=current_summary_df, palette=palette, dodge=True, order=ordered_x_labels,
            edgecolor='black', linewidth=0.7  # Add edge color for bars
        )

        plt.title(f'最终平均奖励对比 ({title_suffix_detail})', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=20)
        plt.ylabel('最终平均奖励', fontsize=LABEL_FONTSIZE)
        plt.xlabel('目标场景 或 基线 (复杂度)', fontsize=LABEL_FONTSIZE)
        plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE if num_scenarios < 12 else TICK_FONTSIZE - 1)
        plt.yticks(fontsize=TICK_FONTSIZE)

        y_max_val = current_summary_df['最终平均奖励'].max() if not current_summary_df.empty else 50
        plt.ylim(min(0, current_summary_df['最终平均奖励'].min() * 1.1 if not current_summary_df.empty and
                                                                          current_summary_df[
                                                                              '最终平均奖励'].min() < 0 else 0),
                 max(55, y_max_val * 1.15))

        plt.legend(title='算法 [运行类型]', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=LEGEND_FONTSIZE,
                   frameon=True, facecolor='whitesmoke')
        plt.grid(axis='y', linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        sns.despine(trim=True)
        # Adjust layout to prevent legend cutoff, rect might need tuning
        plt.tight_layout(rect=[0, 0, 0.80 if fig_width > 15 else 0.85, 0.95])

        plot_filename = f"overall_summary_barchart_{plot_suffix}.png"
        plot_path = plots_dir / plot_filename
        try:
            plt.savefig(plot_path, dpi=300)
            print(f"总体奖励摘要对比图 ({title_suffix_detail}) 已保存至: {plot_path}")
        except Exception as e_save:
            print(f"保存图表 {plot_path} 失败: {e_save}")
        plt.close()


def plot_evaluation_reward_curves(results_data: List[Dict], plots_dir: Path):
    if not results_data: print("绘图警告(评估曲线): 无数据。"); return
    df_results = pd.DataFrame(results_data)
    if 'initial_training_dist_name' not in df_results.columns:
        print("绘图警告(评估曲线): 缺少 'initial_training_dist_name'。");
        return

    unique_initial_dists = df_results['initial_training_dist_name'].unique()

    for initial_dist_name in unique_initial_dists:
        # Filter for adaptation scenarios only for this plot type
        current_dist_results_all = [r for r in results_data if
                                    r.get('initial_training_dist_name') == initial_dist_name and r.get(
                                        'scenario_type') == 'adaptation']

        # Further filter by run_type if present, prefer HPO if both default and HPO exist for the same scenario
        # This logic can be complex if you have many run_types. For now, let's assume we plot for each run_type found.
        # Or, if you want one plot per initial_dist, you might need to decide how to combine/select run_types.
        # For simplicity, this version will create plots for each scenario, implicitly handling different run_types if they exist as separate entries.

        if not current_dist_results_all: continue

        num_results = len(current_dist_results_all)
        cols = min(3, num_results)  # Allow up to 3 columns
        rows = (num_results + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False, sharey=True)  # Share Y axis
        axes = axes.flatten()

        fig.suptitle(f"评估奖励曲线 (初始训练: {initial_dist_name})", fontsize=SUPTITLE_FONTSIZE,
                     y=1.0 if rows > 1 else 0.98, fontweight='bold')

        for idx, res_item in enumerate(current_dist_results_all):
            ax = axes[idx]
            complexity = res_item.get('complexity', -1.0)
            title_scenario_part = f"{res_item['scenario_name']}"
            if complexity >= 0: title_scenario_part += f" (C: {complexity:.2f})"

            run_type_label = f" [{res_item.get('run_type', 'N/A')}]" if 'run_type' in res_item else ""
            title = f"{res_item['algorithm']}{run_type_label} - {title_scenario_part}\n({res_item['adapt_type']})"

            initial_rewards = res_item.get('initial_eval', {}).get('rewards_raw', [])
            final_rewards = res_item.get('final_eval', {}).get('rewards_raw', [])

            if initial_rewards: ax.plot(initial_rewards, label='适应前', linestyle='-', marker='.', alpha=0.8,
                                        markersize=MARKER_SIZE, color='salmon', linewidth=1.5)
            if final_rewards: ax.plot(final_rewards, label='适应后', linestyle='--', marker='.', alpha=0.8,
                                      markersize=MARKER_SIZE, color='mediumseagreen', linewidth=1.5)

            ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
            if idx >= (rows - 1) * cols: ax.set_xlabel("评估回合",
                                                       fontsize=LABEL_FONTSIZE)  # X-label only for bottom row
            if idx % cols == 0: ax.set_ylabel("回合总奖励", fontsize=LABEL_FONTSIZE)  # Y-label only for left column

            ax.set_ylim(-5, 55)
            ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
            if initial_rewards or final_rewards: ax.legend(loc='best', fontsize=LEGEND_FONTSIZE - 1)
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

        for i in range(num_results, len(axes)): fig.delaxes(axes[i])
        plt.tight_layout(rect=[0, 0.02, 1, 0.95 if rows > 1 else 0.92])

        clean_initial_dist_name = sanitize_filename(initial_dist_name)
        plot_path = plots_dir / f"evaluation_reward_curves_{clean_initial_dist_name}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"评估奖励曲线图 ({initial_dist_name}) 已保存至: {plot_path}")
        plt.close(fig)


def plot_adaptation_performance_boxplots(results_data: List[Dict], plots_dir: Path):
    if not results_data: print("绘图警告(箱线图): 无数据。"); return
    df_results = pd.DataFrame(results_data)
    if 'initial_training_dist_name' not in df_results.columns:
        print("绘图警告(箱线图): 缺少 'initial_training_dist_name'。");
        return

    unique_initial_dists = df_results['initial_training_dist_name'].unique()

    for initial_dist_name in unique_initial_dists:
        current_dist_results = [r for r in results_data if
                                r.get('initial_training_dist_name') == initial_dist_name and r.get(
                                    'scenario_type') == 'adaptation']
        if not current_dist_results: continue

        num_results = len(current_dist_results)
        cols = min(3, num_results)  # Allow up to 3 columns
        rows = (num_results + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False, sharey=True)  # Share Y axis
        axes = axes.flatten()
        fig.suptitle(f"自适应性能箱线图 (初始训练: {initial_dist_name})", fontsize=SUPTITLE_FONTSIZE,
                     y=1.0 if rows > 1 else 0.98, fontweight='bold')

        for idx, res_item in enumerate(current_dist_results):
            ax = axes[idx]
            plot_df_data = []
            # Ensure 'rewards_raw' exists and is a list
            if isinstance(res_item.get('initial_eval'), dict) and isinstance(
                    res_item['initial_eval'].get('rewards_raw'), list):
                for r_val in res_item['initial_eval']['rewards_raw']: plot_df_data.append(
                    {'评估阶段': '适应前', '奖励': r_val})
            if isinstance(res_item.get('final_eval'), dict) and isinstance(res_item['final_eval'].get('rewards_raw'),
                                                                           list):
                for r_val in res_item['final_eval']['rewards_raw']: plot_df_data.append(
                    {'评估阶段': '适应后', '奖励': r_val})

            if not plot_df_data:
                ax.set_title(f"{res_item['algorithm']} - {res_item['scenario_name']}\n(无数据)",
                             fontsize=SUBPLOT_TITLE_FONTSIZE - 1)
                continue

            plot_df = pd.DataFrame(plot_df_data)
            palette_boxplot = {"适应前": COLOR_BOXPLOT_INITIAL, "适应后": COLOR_BOXPLOT_FINAL}
            sns.boxplot(x='评估阶段', y='奖励', data=plot_df, ax=ax, palette=palette_boxplot, showmeans=True,
                        order=['适应前', '适应后'], width=0.5, fliersize=3)  # Smaller fliers

            complexity = res_item.get('complexity', -1.0)
            title_scenario_part = f"{res_item['scenario_name']}"
            if complexity >= 0: title_scenario_part += f" (C: {complexity:.2f})"

            run_type_label = f" [{res_item.get('run_type', 'N/A')}]" if 'run_type' in res_item else ""
            title = f"{res_item['algorithm']}{run_type_label} - {title_scenario_part}\n({res_item['adapt_type']}) - 奖励分布"

            ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
            if idx >= (rows - 1) * cols: ax.set_xlabel("评估阶段", fontsize=LABEL_FONTSIZE)
            if idx % cols == 0: ax.set_ylabel("每回合总奖励", fontsize=LABEL_FONTSIZE)

            ax.set_ylim(-5, 55)
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA, axis='y')  # Grid only on y-axis for boxplot

        for i in range(num_results, len(axes)): fig.delaxes(axes[i])
        plt.tight_layout(rect=[0, 0.02, 1, 0.95 if rows > 1 else 0.92])
        clean_initial_dist_name = sanitize_filename(initial_dist_name)
        plot_path = plots_dir / f"adaptation_performance_boxplots_{clean_initial_dist_name}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"自适应性能箱线图 ({initial_dist_name}) 已保存至: {plot_path}")
        plt.close(fig)


def plot_overall_summary_barchart(results_data: List[Dict], plots_dir: Path):  # For individual (non-comparison) runs
    if not results_data: print("绘图警告(总体柱状图): 无数据。"); return
    df_results = pd.DataFrame(results_data)
    if 'initial_training_dist_name' not in df_results.columns:
        print("绘图警告(总体柱状图): 缺少 'initial_training_dist_name'。");
        return

    # This plot is usually for a single run_type (e.g. 'default_params' or one specific 'hpo_params' run)
    # If multiple run_types are present, it might become too busy or misrepresent.
    # Consider filtering by run_type or using plot_overall_summary_barchart_comparison

    # Let's assume this plot is for results of a single "main" run, or we pick one run_type
    # For this version, it will plot all data, hue by algo. If run_type varies, it will be mixed.
    # To make it specific, one might filter results_data before calling this.

    unique_initial_dists = df_results['initial_training_dist_name'].unique()

    for initial_dist_name in unique_initial_dists:
        current_dist_summary_data = []
        # Filter results for the current initial_dist_name
        relevant_results = [res for res in results_data if res.get('initial_training_dist_name') == initial_dist_name]

        for res_item in relevant_results:
            if res_item.get('final_eval') and isinstance(res_item['final_eval'].get('avg_reward'), (int, float)):
                complexity = res_item.get('complexity', -1.0)
                label = f"{res_item['scenario_name']}"
                if res_item.get('scenario_type') != 'baseline' and complexity >= 0:
                    label += f"\n(C: {complexity:.2f})"

                run_type_info = f" [{res_item['run_type']}]" if 'run_type' in res_item and res_item[
                    'run_type'] != 'default_params' else ""  # Add run_type if not default

                current_dist_summary_data.append(
                    {'算法': res_item['algorithm'] + run_type_info,  # Include run_type in algo label if not default
                     '场景': label,
                     '自适应类型': res_item['adapt_type'],  # Not directly used in this plot but good for df
                     '最终平均奖励': res_item['final_eval']['avg_reward']})

        if not current_dist_summary_data: continue  # Skip if no data for this initial_dist
        summary_df = pd.DataFrame(current_dist_summary_data)
        if summary_df.empty: continue

        def sort_key_scenario(scenario_str):
            parts = scenario_str.split('\n(C: ')
            name_part = parts[0]
            complexity_part = -1.0
            if len(parts) > 1:
                try:
                    complexity_part = float(parts[1].rstrip(')'))
                except ValueError:
                    pass
            return (name_part, complexity_part)

        ordered_x_labels = sorted(summary_df['场景'].unique(), key=sort_key_scenario)

        num_scenarios = len(ordered_x_labels)
        num_algos = len(summary_df['算法'].unique())  # This now includes run_type info
        fig_width = max(12, num_scenarios * num_algos * 0.18 + 4)

        plt.figure(figsize=(fig_width, 8))
        # Palette might need more colors if "algo + run_type" creates many unique hues
        palette = sns.color_palette(SEABORN_PALETTE_BARCHART, n_colors=num_algos)

        sns.barplot(x='场景', y='最终平均奖励', hue='算法', data=summary_df, palette=palette,
                    dodge=True, order=ordered_x_labels, edgecolor='black', linewidth=0.7)
        plt.title(f'最终平均奖励 (初始训练: {initial_dist_name})', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=20)
        plt.ylabel('最终平均奖励', fontsize=LABEL_FONTSIZE)
        plt.xlabel('测试场景 (适应目标 或 基线)', fontsize=LABEL_FONTSIZE)
        plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE if num_scenarios < 12 else TICK_FONTSIZE - 1)
        plt.yticks(fontsize=TICK_FONTSIZE)

        y_max_val = summary_df['最终平均奖励'].max() if not summary_df.empty else 50
        plt.ylim(min(0, summary_df['最终平均奖励'].min() * 1.1 if not summary_df.empty and summary_df[
            '最终平均奖励'].min() < 0 else 0),
                 max(55, y_max_val * 1.15))

        plt.legend(title='算法 [运行类型]', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=LEGEND_FONTSIZE,
                   frameon=True)
        plt.grid(axis='y', linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        sns.despine(trim=True)
        plt.tight_layout(rect=[0, 0, 0.82 if fig_width > 15 else 0.85, 0.95])

        clean_initial_dist_name = sanitize_filename(initial_dist_name)
        plot_path = plots_dir / f"summary_rewards_barchart_{clean_initial_dist_name}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"总体奖励摘要图 ({initial_dist_name}) 已保存至: {plot_path}")
        plt.close()


def plot_learning_curves_from_logs(log_file_path: Path, title: str, plots_dir: Path, suffix: str,
                                   complexity: float = None):
    if not log_file_path or not log_file_path.exists():
        # print(f"学习曲线日志文件未找到: {log_file_path}");
        return
    try:
        df = pd.read_excel(log_file_path)
        if df.empty or not all(col in df.columns for col in ['episode', 'total_reward', 'correct_rate']):
            # print(f"学习曲线日志 {log_file_path} 为空或缺少列。")
            return

        # Adaptive window size based on number of episodes
        window_size = max(5, min(30, len(df) // 15 if len(df) > 15 else 5))
        df['total_reward_smoothed'] = df['total_reward'].rolling(window=window_size, min_periods=1, center=True).mean()
        df['correct_rate_smoothed'] = df['correct_rate'].rolling(window=window_size, min_periods=1, center=True).mean()

        fig, ax1 = plt.subplots(figsize=(12, 6.5))  # Adjusted size

        full_title = _get_title_with_complexity(base_title=title, complexity=complexity)  # Use helper for title
        fig.suptitle(full_title, fontsize=SUPTITLE_FONTSIZE, fontweight='bold', y=1.02)

        # Plot raw total_reward (lighter)
        ax1.plot(df['episode'], df['total_reward'], color=COLOR_REWARD_RAW, alpha=RAW_LINE_ALPHA, linewidth=1.2,
                 label='原始总奖励')
        # Plot smoothed total_reward (darker)
        ax1.plot(df['episode'], df['total_reward_smoothed'], color=COLOR_REWARD_SMOOTHED, linewidth=SMOOTH_LINEWIDTH,
                 label=f'平滑总奖励 (窗={window_size})')
        ax1.set_xlabel('回合', fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('总奖励', color=COLOR_REWARD_SMOOTHED, fontsize=LABEL_FONTSIZE)
        ax1.tick_params(axis='y', labelcolor=COLOR_REWARD_SMOOTHED, labelsize=TICK_FONTSIZE)
        ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE)
        ax1.set_ylim(-5, 55)

        ax2 = ax1.twinx()
        ax2.plot(df['episode'], df['correct_rate_smoothed'], color=COLOR_CORRECT_RATE_SMOOTHED, linestyle='--',
                 linewidth=SMOOTH_LINEWIDTH, label=f'平滑正确率 (窗={window_size})')
        ax2.set_ylabel('正确率', color=COLOR_CORRECT_RATE_SMOOTHED, fontsize=LABEL_FONTSIZE)
        ax2.tick_params(axis='y', labelcolor=COLOR_CORRECT_RATE_SMOOTHED, labelsize=TICK_FONTSIZE)
        ax2.set_ylim(-0.05, 1.05)  # Y-limit for correct rate 0-1

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='center right', fontsize=LEGEND_FONTSIZE, frameon=True,
                   facecolor='white', framealpha=0.8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        ax1.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)

        clean_suffix = sanitize_filename(suffix)
        plot_path = plots_dir / f"learning_curve_{clean_suffix}.png"
        plt.savefig(plot_path, dpi=300)
        # print(f"学习曲线图 ({title}) 已保存至: {plot_path}") # Title can be long
        plt.close(fig)
    except Exception as e:
        print(f"从日志 {log_file_path} 绘制学习曲线时发生错误: {e}")


def plot_combined_learning_curve(
        initial_log_path: Path,
        adaptation_log_path: Path,
        algorithm_name: str,  # Includes run_type
        initial_dist_label: str,
        scenario_name_label: str,
        adapt_type: str,
        plots_dir: Path,
        combined_suffix: str,
        initial_complexity: float = None,
        adapt_complexity: float = None
):
    if not initial_log_path or not initial_log_path.exists(): return
    if not adaptation_log_path or not adaptation_log_path.exists(): return
    try:
        df_initial = pd.read_excel(initial_log_path)
        df_adaptation = pd.read_excel(adaptation_log_path)
        if df_initial.empty or not all(k in df_initial.columns for k in ['episode', 'total_reward']): return
        if df_adaptation.empty or not all(k in df_adaptation.columns for k in ['episode', 'total_reward']): return

        win_init = max(5, min(30, len(df_initial) // 15 if len(df_initial) > 15 else 5))
        df_initial['total_reward_smoothed'] = df_initial['total_reward'].rolling(window=win_init, min_periods=1,
                                                                                 center=True).mean()

        win_adapt = max(5, min(30, len(df_adaptation) // 15 if len(df_adaptation) > 15 else 5))
        df_adaptation['total_reward_smoothed'] = df_adaptation['total_reward'].rolling(window=win_adapt, min_periods=1,
                                                                                       center=True).mean()

        max_initial_episode = df_initial['episode'].max() if not df_initial.empty else 0
        # Shift adaptation phase episode numbers
        df_adaptation['episode_shifted'] = df_adaptation['episode'] + max_initial_episode

        plt.figure(figsize=(14, 7))

        # Plot raw data with more transparency
        plt.plot(df_initial['episode'], df_initial['total_reward'], color=COLOR_INITIAL_TRAIN,
                 alpha=RAW_LINE_ALPHA * 0.6, linewidth=1.0)
        plt.plot(df_adaptation['episode_shifted'], df_adaptation['total_reward'], color=COLOR_ADAPTATION_PHASE,
                 alpha=RAW_LINE_ALPHA * 0.6, linewidth=1.0)

        # Plot smoothed data
        plt.plot(df_initial['episode'], df_initial['total_reward_smoothed'], color=COLOR_INITIAL_TRAIN,
                 linewidth=SMOOTH_LINEWIDTH,
                 label=f'初始训练 ({initial_dist_label}, 窗={win_init})')
        plt.plot(df_adaptation['episode_shifted'], df_adaptation['total_reward_smoothed'], color=COLOR_ADAPTATION_PHASE,
                 linewidth=SMOOTH_LINEWIDTH,
                 label=f'适应阶段 ({scenario_name_label} - {adapt_type}, 窗={win_adapt})')

        plt.xlabel('总回合数 (初始 + 适应)', fontsize=LABEL_FONTSIZE)
        plt.ylabel('每回合总奖励', fontsize=LABEL_FONTSIZE)
        plt.ylim(-5, 55)

        title_init_part = f"{initial_dist_label}"
        if initial_complexity is not None and initial_complexity >= 0: title_init_part += f" (C:{initial_complexity:.2f})"
        title_adapt_part = f"{scenario_name_label}"
        if adapt_complexity is not None and adapt_complexity >= 0: title_adapt_part += f" (C:{adapt_complexity:.2f})"

        title = f"{algorithm_name}: {title_init_part} → {title_adapt_part} ({adapt_type})"  # Arrow for transition
        plt.title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)

        plt.legend(loc='best', fontsize=LEGEND_FONTSIZE, frameon=True, facecolor='white', framealpha=0.85)
        plt.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        plt.tight_layout()

        clean_suffix = sanitize_filename(combined_suffix)
        plot_path = plots_dir / f"combined_lrn_crv_{clean_suffix}.png"  # Shortened filename
        plt.savefig(plot_path, dpi=300)
        # print(f"组合学习曲线图 ({combined_suffix}) 已保存至: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"绘制组合学习曲线时发生错误 (Files: {initial_log_path.name}, {adaptation_log_path.name}): {e}")
        if 'plt' in locals() and plt.gcf().number > 0: plt.close()


def plot_complexity_vs_performance(results_data: List[Dict[str, Any]], plots_dir: Path,
                                   performance_metric_key: str = 'avg_reward', plot_type: str = 'adaptation_final'):
    if not results_data: print("绘图警告(复杂度vs性能): 无数据。"); return
    plot_data = []
    for item in results_data:
        complexity = item.get('complexity', -1.0)
        if complexity < 0: continue  # Skip invalid complexity

        perf_value = np.nan
        eval_dict_key = None
        if plot_type == 'baseline' and item.get('scenario_type') == 'baseline':
            eval_dict_key = 'final_eval'
        elif plot_type == 'adaptation_initial' and item.get('scenario_type') == 'adaptation':
            eval_dict_key = 'initial_eval'
        elif plot_type == 'adaptation_final' and item.get('scenario_type') == 'adaptation':
            eval_dict_key = 'final_eval'

        if eval_dict_key and isinstance(item.get(eval_dict_key), dict):
            perf_value = item[eval_dict_key].get(performance_metric_key, np.nan)

        if not np.isnan(perf_value):
            run_type_info = f"_{item['run_type']}" if 'run_type' in item and item[
                'run_type'] != 'default_params' else ""
            plot_data.append({
                '复杂度': complexity,
                '性能指标': perf_value,
                '算法': item['algorithm'] + run_type_info,  # Include run_type in hue
                '初始训练分布': item['initial_training_dist_name'],
                '目标场景': item['scenario_name'] if item.get('scenario_type') == 'adaptation' else item[
                    'initial_training_dist_name']
            })

    if not plot_data: print(f"绘图警告(复杂度vs性能): 无有效数据 for {plot_type} / {performance_metric_key}。"); return

    df_plot = pd.DataFrame(plot_data)
    df_plot['算法'] = pd.Categorical(df_plot['算法'])
    df_plot['初始训练分布'] = pd.Categorical(df_plot['初始训练分布'])

    # Determine number of unique styles for style legend; if too many, don't use style for '初始训练分布'
    style_legend = '初始训练分布' if len(df_plot['初始训练分布'].unique()) <= 7 else None  # Limit styles
    size_legend = '目标场景' if len(df_plot['目标场景'].unique()) <= 10 else None  # Limit size categories

    plt.figure(figsize=(11, 7.5))  # Adjusted size
    # Use a specific palette if number of algos (hue) is known and not too large
    num_hues = len(df_plot['算法'].unique())
    palette_scatter = sns.color_palette(SEABORN_PALETTE_SCATTER,
                                        n_colors=num_hues) if num_hues <= 10 else SEABORN_PALETTE_SCATTER

    scatter_plot = sns.scatterplot(
        data=df_plot, x='复杂度', y='性能指标', hue='算法',
        style=style_legend,
        size=size_legend,
        legend='auto', palette=palette_scatter,
        sizes=(60, 220) if size_legend else (80, 80),  # Adjust sizes
        alpha=0.8
    )

    y_label = performance_metric_key.replace('avg_', '').replace('_', ' ').title()
    plot_type_label_map = {'baseline': '基线性能', 'adaptation_initial': '适应前性能', 'adaptation_final': '适应后性能'}
    plot_type_readable = plot_type_label_map.get(plot_type, plot_type.replace('_', ' ').title())

    plt.title(f'分布复杂度 vs {y_label} ({plot_type_readable})', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    plt.xlabel('分布复杂度 (越高越难区分)', fontsize=LABEL_FONTSIZE)
    plt.ylabel(y_label, fontsize=LABEL_FONTSIZE)
    plt.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)

    handles, labels = scatter_plot.get_legend_handles_labels()
    if len(handles) > 20:  # If legend is too crowded
        # Simplify legend: only show hue
        hue_handles = [h for h, l in zip(handles, labels) if l in df_plot['算法'].unique().tolist()]
        hue_labels = [l for l in df_plot['算法'].unique().tolist()]
        plt.legend(hue_handles, hue_labels, title='算法', bbox_to_anchor=(1.02, 1), loc='upper left',
                   fontsize=LEGEND_FONTSIZE - 1)
    else:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=LEGEND_FONTSIZE - 1,
                   title="图例")

    plt.tight_layout(rect=[0, 0, 0.80 if len(handles) > 8 else 0.85, 0.95])
    sns.despine(trim=True)

    plot_filename = f"cmplx_vs_{sanitize_filename(performance_metric_key)}_{sanitize_filename(plot_type)}.png"  # Shortened
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300)
    # print(f"复杂度vs性能图 ({plot_type_readable} - {y_label}) 已保存至: {plot_path}")
    plt.close()


def plot_reward_distribution_over_training_stages(log_file_path: Path, title_prefix: str, plots_dir: Path, suffix: str,
                                                  complexity: float = None, num_stages: int = 4):
    if not log_file_path or not log_file_path.exists(): return
    try:
        df = pd.read_excel(log_file_path)
        if df.empty or not all(k in df.columns for k in ['episode', 'total_reward']): return

        min_data_points = num_stages * 5
        if len(df) < min_data_points:
            # print(f"阶段奖励分布图日志 {log_file_path.name} 数据点 ({len(df)}) 过少 (需 {min_data_points})。")
            return

        df = df.sort_values(by='episode').reset_index(drop=True)
        total_episodes_in_log = len(df)  # Use actual number of episodes in the log

        # Ensure episodes_per_stage is at least 1
        episodes_per_stage = max(1, total_episodes_in_log // num_stages)
        stage_labels = []
        df['训练阶段'] = pd.NA  # Initialize with pandas NA for categorical

        for i in range(num_stages):
            start_idx = i * episodes_per_stage
            end_idx = (
                                  i + 1) * episodes_per_stage if i < num_stages - 1 else total_episodes_in_log  # Last stage takes all remaining

            if start_idx >= total_episodes_in_log: continue  # No more data for this stage

            # Use actual episode numbers from log for labels
            ep_start_label = df['episode'].iloc[start_idx]
            ep_end_label = df['episode'].iloc[
                min(end_idx - 1, total_episodes_in_log - 1)]  # Ensure end_idx is within bounds for iloc

            stage_label = f"阶段 {i + 1}\n(Ep {ep_start_label}-{ep_end_label})"
            df.loc[df.index[start_idx:end_idx], '训练阶段'] = stage_label  # Use df.index for loc after sorting
            if not df.iloc[start_idx:end_idx].empty:  # Only add if stage has data
                stage_labels.append(stage_label)

        df_plot = df.dropna(subset=['训练阶段'])  # Remove rows where stage was not assigned
        if df_plot.empty: return

        plt.figure(figsize=(max(7, len(stage_labels) * 2.3), 6))
        sns.violinplot(x='训练阶段', y='total_reward', data=df_plot, palette="coolwarm", order=stage_labels,
                       inner="quartile", cut=0, linewidth=1.2, scale="width",
                       bw_adjust=0.75)  # bw_adjust for smoothness

        dist_name_for_title = Path(log_file_path).stem.replace('_episode_log', '').replace(suffix, '').strip(' _-')
        # dist_name_for_title = ' '.join(s.capitalize() for s in dist_name_for_title.split('_')) # Nicer formatting

        plot_title_full = _get_title_with_complexity(
            base_title=f"{title_prefix}\n训练阶段奖励分布",
            complexity=complexity,
            dist_name_for_title=""  # Keep title prefix as is, dist_name_for_title is for specific cases
        )
        plt.title(plot_title_full, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=12)
        plt.xlabel("训练阶段 (按实际回合数)", fontsize=LABEL_FONTSIZE)
        plt.ylabel("每回合总奖励", fontsize=LABEL_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE - 1, rotation=20, ha='right')
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.ylim(-5, 55)
        plt.grid(axis='y', linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        sns.despine(trim=True)
        plt.tight_layout()

        clean_suffix = sanitize_filename(suffix)
        plot_path = plots_dir / f"reward_dist_stages_{clean_suffix}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # print(f"训练阶段奖励分布图 ({suffix}) 已保存至: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"从日志 {log_file_path} 绘制阶段奖励分布图时发生错误: {e}")


def plot_convergence_metrics_over_training(
        log_file_path: Path,
        title_prefix: str,
        plots_dir: Path,
        suffix: str,
        complexity: float = None,
        smoothing_window_reward: int = 20,  # Increased default smoothing
        metrics_calculation_window: int = 30,
        metrics_calculation_step: int = 10,  # Calculate less frequently
        fixed_slope_ylim: Optional[Tuple[float, float]] = (-0.25, 0.5),
        fixed_std_ylim: Optional[Tuple[float, float]] = (0, 20)
):
    if not log_file_path or not log_file_path.exists(): return
    try:
        df = pd.read_excel(log_file_path)
        if df.empty or not all(col in df.columns for col in ['episode', 'total_reward']): return

        actual_calc_window = min(metrics_calculation_window, len(df) - 1 if len(df) > 1 else 1)
        if len(df) < actual_calc_window or actual_calc_window < 5:  # Need more points for meaningful slope/std
            # print(f"收敛指标图日志 {log_file_path.name} 数据点 ({len(df)}) 过少 (需 {actual_calc_window}, 至少5)。")
            return

        actual_smoothing_window = min(smoothing_window_reward, len(df))

        df = df.sort_values(by='episode').reset_index(drop=True)
        # Ensure 'total_reward_smoothed' is calculated before being used by other parts
        df['total_reward_smoothed'] = df['total_reward'].rolling(window=actual_smoothing_window, min_periods=1,
                                                                 center=True).mean()

        episodes_col = df['episode'].values
        rewards_col = df['total_reward'].values  # Use raw rewards for metrics

        calculated_at_episode_step = []
        slopes = []
        std_devs = []

        for i in range(0, len(rewards_col) - actual_calc_window + 1, metrics_calculation_step):
            window_rewards = rewards_col[i: i + actual_calc_window]
            current_episode_point = episodes_col[i + actual_calc_window - 1]

            if len(window_rewards) >= 5:  # More robust slope with more points
                x_axis = np.arange(len(window_rewards))
                try:
                    slope, intercept, r_val, p_val, std_err = stats.linregress(x_axis, window_rewards)
                    slopes.append(slope if not np.isnan(slope) else 0.0)
                except ValueError:
                    slopes.append(0.0)
            else:
                slopes.append(np.nan)

            if len(window_rewards) >= 2:  # Std dev needs at least 2 points to be meaningful
                std_devs.append(np.std(window_rewards))
            else:
                std_devs.append(np.nan)

            if not (np.isnan(slopes[-1] if slopes else np.nan) and np.isnan(
                    std_devs[-1] if std_devs else np.nan)):  # Only add if at least one metric is valid
                calculated_at_episode_step.append(current_episode_point)
            else:  # If both are NaN, remove the last added slope/std to keep lists aligned with valid_episodes
                if slopes: slopes.pop()
                if std_devs: std_devs.pop()

        fig, ax1 = plt.subplots(figsize=(12, 7))
        dist_name_for_title = ""  # title_prefix should contain this info

        plot_title_full = _get_title_with_complexity(
            base_title=f"{title_prefix}\n训练收敛性指标", complexity=complexity, dist_name_for_title=dist_name_for_title
        )
        fig.suptitle(plot_title_full, fontsize=SUPTITLE_FONTSIZE, fontweight='bold', y=1.0)

        color_reward_ax = COLOR_REWARD_SMOOTHED
        ax1.set_xlabel('回合', fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel(f'平滑总奖励 (窗={actual_smoothing_window})', color=color_reward_ax, fontsize=LABEL_FONTSIZE)
        ax1.plot(df['episode'], df['total_reward_smoothed'], color=color_reward_ax, linewidth=SMOOTH_LINEWIDTH,
                 label='平滑总奖励')
        ax1.tick_params(axis='y', labelcolor=color_reward_ax, labelsize=TICK_FONTSIZE)
        ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE)
        ax1.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        ax1.set_ylim(-5, 55)

        # Ensure calculated_at_episode_step is not empty before proceeding
        if calculated_at_episode_step:
            ax2 = ax1.twinx()
            color_slope_ax = 'darkorange'
            ax2.set_ylabel(f'奖励斜率 (窗={actual_calc_window})', color=color_slope_ax, fontsize=LABEL_FONTSIZE)
            ax2.plot(calculated_at_episode_step, slopes, color=color_slope_ax, linestyle='--', linewidth=1.8,
                     label='奖励斜率', marker='.', markersize=4, alpha=0.8)
            ax2.tick_params(axis='y', labelcolor=color_slope_ax, labelsize=TICK_FONTSIZE)
            if fixed_slope_ylim:
                ax2.set_ylim(fixed_slope_ylim)
            else:
                valid_slopes = [s for s in slopes if not np.isnan(s)]
                if valid_slopes:
                    s_min, s_max = np.percentile(valid_slopes, [2, 98])  # Wider percentile
                    s_pad = (s_max - s_min) * 0.2 if (s_max - s_min) > 1e-3 else 0.2
                    ax2.set_ylim(s_min - s_pad, s_max + s_pad)
                else:
                    ax2.set_ylim(-0.5, 0.5)

            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 65))
            color_std_ax = 'forestgreen'  # Changed color
            ax3.set_ylabel(f'奖励标准差 (窗={actual_calc_window})', color=color_std_ax, fontsize=LABEL_FONTSIZE)
            ax3.plot(calculated_at_episode_step, std_devs, color=color_std_ax, linestyle=':', linewidth=1.8,
                     label='奖励标准差', marker='x', markersize=4, alpha=0.8)
            ax3.tick_params(axis='y', labelcolor=color_std_ax, labelsize=TICK_FONTSIZE)
            if fixed_std_ylim:
                ax3.set_ylim(fixed_std_ylim)
            else:
                valid_stds = [s for s in std_devs if not np.isnan(s)]
                if valid_stds:
                    std_min, std_max = np.percentile(valid_stds, [2, 98])
                    std_pad = (std_max - std_min) * 0.2 if (std_max - std_min) > 1e-2 else 1.0
                    ax3.set_ylim(max(0, std_min - std_pad), std_max + std_pad)
                else:
                    ax3.set_ylim(0, 15)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='lower center', ncol=3,
                       fontsize=LEGEND_FONTSIZE, frameon=True, facecolor='white', framealpha=0.85,
                       bbox_to_anchor=(0.5, -0.25))
        else:  # Only ax1 legend if no data for ax2/ax3
            ax1.legend(loc='best', fontsize=LEGEND_FONTSIZE)

        fig.tight_layout(rect=[0, 0.05, 1, 0.93])  # Adjust for suptitle and potential bottom legend
        sns.despine(ax=ax1, right=False, trim=True)
        if 'ax2' in locals(): sns.despine(ax=ax2, left=False, right=False, trim=True)
        if 'ax3' in locals(): sns.despine(ax=ax3, left=False, trim=True)

        clean_suffix = sanitize_filename(suffix)
        plot_path = plots_dir / f"conv_metrics_dyn_{clean_suffix}.png"  # Shortened
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # bbox_inches for multi-axis legend
        # print(f"收敛指标动态图 ({suffix}) 已保存至: {plot_path}")
        plt.close(fig)

    except Exception as e:
        print(f"从日志 {log_file_path} 绘制收敛指标动态图时发生错误: {e}")
        # import traceback; traceback.print_exc() # For debugging
        if 'fig' in locals() and hasattr(fig, 'number') and plt.fignum_exists(fig.number): plt.close(fig)


def plot_adaptation_delta_rewards(results_data: List[Dict[str, Any]], plots_dir: Path,
                                  initial_training_filter: str = None):
    if not results_data: print("绘图警告(Delta奖励): 无数据。"); return
    delta_data = []
    for item in results_data:
        if item.get('scenario_type') == 'adaptation' and \
                isinstance(item.get('initial_eval'), dict) and \
                isinstance(item['initial_eval'].get('avg_reward'), (float, int)) and \
                isinstance(item.get('final_eval'), dict) and \
                isinstance(item['final_eval'].get('avg_reward'), (float, int)):

            if initial_training_filter and item.get('initial_training_dist_name') != initial_training_filter:
                continue

            delta_reward = item['final_eval']['avg_reward'] - item['initial_eval']['avg_reward']
            complexity_target = item.get('complexity', -1.0)

            # Create a more concise Y-axis label
            target_short_name = item['scenario_name'].split('(')[0].strip()  # Get name before parenthesis
            if len(target_short_name) > 20: target_short_name = target_short_name[:18] + "..."

            label = f"{target_short_name} ({item['adapt_type'][:3]})"  # Target (AdaptType)
            if complexity_target >= 0: label += f" (C:{complexity_target:.2f})"

            run_type_info = f" [{item['run_type']}]" if 'run_type' in item and item[
                'run_type'] != 'default_params' else ""

            delta_data.append({
                '场景 (类型) [目标C]': label,
                '算法': item['algorithm'] + run_type_info,
                '奖励变化量': delta_reward,
                '初始训练分布': item['initial_training_dist_name']  # For potential sub-grouping if needed
            })

    if not delta_data:
        filter_msg = f" (过滤器: {initial_training_filter})" if initial_training_filter else ""
        print(f"绘图警告(Delta奖励): 无有效适应性数据{filter_msg}。");
        return

    df_delta = pd.DataFrame(delta_data)
    df_delta = df_delta.sort_values(by='奖励变化量', ascending=False)

    num_transitions = len(df_delta['场景 (类型) [目标C]'].unique())
    fig_height = max(6, num_transitions * 0.35 + 2 if num_transitions > 5 else 7)  # Dynamic height
    fig_width = 10

    plt.figure(figsize=(fig_width, fig_height))

    num_hues = len(df_delta['算法'].unique())
    algo_palette = sns.color_palette("Spectral", n_colors=num_hues) if num_hues > 6 else sns.color_palette("muted",
                                                                                                           n_colors=num_hues)

    sns.barplot(y='场景 (类型) [目标C]', x='奖励变化量', hue='算法', data=df_delta, palette=algo_palette, dodge=True,
                orient='h', edgecolor='grey', linewidth=0.5)

    title_str = f'适应后平均奖励变化量'
    if initial_training_filter:
        title_str += f"\n(初始训练于: {initial_training_filter})"
    else:
        title_str += "\n(所有初始分布汇总)"

    plt.title(title_str, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
    plt.xlabel('平均奖励变化量 (适应后 - 适应前)', fontsize=LABEL_FONTSIZE)
    plt.ylabel('目标场景 (适应类型) [目标复杂度]', fontsize=LABEL_FONTSIZE)  # Y-label

    ytick_fontsize = TICK_FONTSIZE - 1
    if num_transitions > 20: ytick_fontsize -= 1
    plt.yticks(fontsize=max(6.5, ytick_fontsize))
    plt.xticks(fontsize=TICK_FONTSIZE)

    plt.axvline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.8)
    plt.grid(axis='x', linestyle=GRID_STYLE, alpha=GRID_ALPHA)

    plt.legend(title='算法 [运行类型]', loc='best', fontsize=LEGEND_FONTSIZE - 1, frameon=True)
    sns.despine(trim=True, left=True, bottom=True)  # Keep axis lines clean
    plt.tight_layout(pad=0.5)

    filename_suffix = f"_init_{sanitize_filename(initial_training_filter)}" if initial_training_filter else "_allInits"
    plot_path = plots_dir / f"adapt_delta_rewards{filename_suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # print(f"适应性奖励变化图 ({initial_training_filter if initial_training_filter else 'All'}) 已保存至: {plot_path}")
    plt.close()


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from math import pi
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# --- Style and Color Definitions for the Sci-Style Radar ---
COLOR_DEFAULT = '#4C566A'
COLOR_HPO = '#5E81AC'
COLOR_IMPROVEMENT = '#A3BE8C'
COLOR_DECLINE = '#BF616A'
GRID_COLOR = '#D8DEE9'
SPINE_COLOR = '#434C5E'


def sanitize_filename(filename: str) -> str:
    if not isinstance(filename, str):
        filename = str(filename)
    sanitized = re.sub(r'[^\w\s\.\-\(\)一-龥]', '', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = sanitized.strip('._- ')
    return sanitized if sanitized else "untitled"


def plot_aggregated_performance_radar(
        results_data: List[Dict[str, Any]],
        plots_dir: Path,
        aggregation_level: str = 'algorithm',  # 这个参数保留，但内部逻辑已改变
        theta_offset_degrees: float = 0
):
    """
    为每个独立的实验运行（按算法和初始环境分组）绘制一个详细的、带数值标注的性能雷达图。
    该图对比了默认参数与HPO参数在六个关键维度上的表现。
    """
    if not results_data:
        print("雷达图警告: 无数据可用于绘制。")
        return

    df = pd.DataFrame(results_data)
    required_cols = [aggregation_level, 'initial_training_dist_name', 'run_type', 'scenario_type', 'final_eval',
                     'initial_training_duration', 'adaptation_duration']
    if not all(col in df.columns for col in required_cols):
        print(f"雷达图警告: 结果数据缺少必要字段。")
        return

    # 新的分组逻辑：按算法和初始环境来创建独立的图表
    unique_experiments = df[[aggregation_level, 'initial_training_dist_name']].drop_duplicates()

    for _, experiment_row in unique_experiments.iterrows():
        algo_name = experiment_row[aggregation_level]
        initial_dist_name = experiment_row['initial_training_dist_name']

        # 筛选出当前独立实验的所有相关数据
        experiment_df = df[
            (df[aggregation_level] == algo_name) &
            (df['initial_training_dist_name'] == initial_dist_name)
            ]

        processed_metrics = []
        # 在独立实验数据内，再按运行类型（default vs hpo）分组
        for run_type, group in experiment_df.groupby('run_type'):
            baseline_df = group[group['scenario_type'] == 'baseline']
            adaptation_df = group[group['scenario_type'] == 'adaptation']

            # --- 计算六个维度的指标 ---
            initial_rewards = baseline_df['final_eval'].apply(
                lambda x: x.get('avg_reward', np.nan) if isinstance(x, dict) else np.nan).dropna()
            m1_initial_avg_reward = initial_rewards.mean() if not initial_rewards.empty else np.nan

            initial_times = baseline_df['initial_training_duration'].astype(float).dropna()
            m2_initial_avg_time = initial_times.mean() if not initial_times.empty else np.nan

            # 注意：这里是对一个初始环境的所有适应场景取平均，得到适应能力的综合表现
            adapted_rewards = adaptation_df['final_eval'].apply(
                lambda x: x.get('avg_reward', np.nan) if isinstance(x, dict) else np.nan).dropna()
            m3_adapted_avg_reward = adapted_rewards.mean() if not adapted_rewards.empty else np.nan

            adapted_times = adaptation_df['adaptation_duration'].astype(float).dropna()
            m4_adapted_avg_time = adapted_times.mean() if not adapted_times.empty else np.nan

            valid_rewards_for_overall = [r for r in [m1_initial_avg_reward, m3_adapted_avg_reward] if not np.isnan(r)]
            m5_overall_avg_reward = np.mean(valid_rewards_for_overall) if len(valid_rewards_for_overall) > 0 else np.nan

            valid_times_for_total = [t for t in [m2_initial_avg_time, m4_adapted_avg_time] if not np.isnan(t)]
            m6_total_avg_time = np.sum(valid_times_for_total) if len(valid_times_for_total) > 0 else np.nan

            processed_metrics.append({
                'run_type': run_type,
                'M1_InitialAvgReward': m1_initial_avg_reward, 'M2_InitialAvgTime': m2_initial_avg_time,
                'M3_AdaptedAvgReward': m3_adapted_avg_reward, 'M4_AdaptedAvgTime': m4_adapted_avg_time,
                'M5_OverallAvgReward': m5_overall_avg_reward, 'M6_TotalAvgTime': m6_total_avg_time,
            })

        if len(processed_metrics) < 2:
            print(f"雷达图警告: 实验 '{algo_name} - {initial_dist_name}' 缺少默认或HPO数据，跳过绘图。")
            continue

        metrics_df = pd.DataFrame(processed_metrics)
        metric_display_info = {
            'M1_InitialAvgReward': {"label": "初始环境\n平均奖励", "higher_is_better": True},
            'M2_InitialAvgTime': {"label": "初始训练\n平均时间 (s)", "higher_is_better": False},
            'M3_AdaptedAvgReward': {"label": "适应环境\n平均奖励", "higher_is_better": True},
            'M4_AdaptedAvgTime': {"label": "适应阶段\n平均时间 (s)", "higher_is_better": False},
            'M5_OverallAvgReward': {"label": "总体\n平均奖励", "higher_is_better": True},
            'M6_TotalAvgTime': {"label": "总体平均\n训练时间 (s)", "higher_is_better": False},
        }
        metrics_to_plot = list(metric_display_info.keys())

        default_data_row = metrics_df[metrics_df['run_type'] == 'default_params']
        hpo_data_row = metrics_df[metrics_df['run_type'] == 'hpo_params']

        if default_data_row.empty or hpo_data_row.empty:
            continue

        values_default = default_data_row[metrics_to_plot].fillna(0).iloc[0].values.flatten().tolist()
        values_hpo = hpo_data_row[metrics_to_plot].fillna(0).iloc[0].values.flatten().tolist()

        num_vars = len(metrics_to_plot)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        values_default_closed, values_hpo_closed = values_default + [values_default[0]], values_hpo + [values_hpo[0]]
        angles_closed = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(12, 11), subplot_kw=dict(polar=True))
        ax.set_facecolor('white')

        # --- 填充差异区域 ---
        for i in range(num_vars):
            is_better = metric_display_info[metrics_to_plot[i]]['higher_is_better']
            if (is_better and values_hpo[i] > values_default[i]) or \
                    (not is_better and values_hpo[i] < values_default[i]):
                fill_color = COLOR_HPO_IMPROVEMENT
            else:
                fill_color = COLOR_HPO_DECLINE
            ax.fill_between(angles_closed[i:i + 2], values_default_closed[i:i + 2], values_hpo_closed[i:i + 2],
                            color=fill_color, alpha=0.3, zorder=2)

        # --- 绘制数据线和点 ---
        ax.plot(angles_closed, values_default_closed, color=COLOR_DEFAULT_RADAR, linewidth=2.5, linestyle='-',
                label='默认参数', zorder=3)
        ax.plot(angles_closed, values_hpo_closed, color=COLOR_HPO_RADAR, linewidth=2.5, linestyle='-', label='HPO参数',
                zorder=3)
        ax.scatter(angles, values_default, color=COLOR_DEFAULT_RADAR, s=60, zorder=4, edgecolor='white', linewidth=1)
        ax.scatter(angles, values_hpo, color=COLOR_HPO_RADAR, s=60, zorder=4, edgecolor='white', linewidth=1)

        # --- 新增：添加数值标签 ---
        def add_value_labels(angles, values, color):
            for i, value in enumerate(values):
                angle_rad = angles[i]
                # 根据角度调整文本对齐方式，避免与数据点重叠
                ha = 'center'
                if np.cos(angle_rad) > 0.1: ha = 'left'
                if np.cos(angle_rad) < -0.1: ha = 'right'

                ax.text(angle_rad, value, f" {value:.1f} ", color='white', ha=ha, va='center',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor='none', alpha=0.9))

        add_value_labels(angles, values_default, COLOR_DEFAULT_RADAR)
        add_value_labels(angles, values_hpo, COLOR_HPO_RADAR)
        # --- 数值标签结束 ---

        ax.set_theta_offset(pi / 6 + np.deg2rad(theta_offset_degrees))  # 平顶布局
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        axis_labels = [f"{info['label']}\n(越高越好)" if info['higher_is_better'] else f"{info['label']}\n(越低越好)"
                       for m, info in metric_display_info.items()]
        ax.set_xticklabels(axis_labels, fontsize=11, color='#333333', y=0.03)

        for i, label in enumerate(ax.get_xticklabels()):
            angle_rad = angles[i] + ax.get_theta_offset()
            if np.isclose(angle_rad % np.pi, 0):
                label.set_horizontalalignment('center')
            elif np.cos(angle_rad) > 0:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')
            label.set_position((label.get_position()[0], label.get_position()[1] * 1.30))

        all_vals = np.concatenate([values_default, values_hpo])
        y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)
        y_pad = (y_max - y_min) * 0.20 if (y_max - y_min) > 1e-6 else 1.0
        ax.set_ylim(min(0, y_min) - y_pad, y_max + y_pad)
        ax.tick_params(axis='y', labelsize=9, colors=GRID_COLOR)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

        ax.grid(color=GRID_COLOR, linestyle='--', linewidth=1)
        ax.spines['polar'].set_color(SPINE_COLOR)
        ax.spines['polar'].set_linewidth(1.5)

        fig.suptitle(f'算法性能对比: {algo_name}', fontsize=18, fontweight='bold', color=SPINE_COLOR, y=0.99)
        ax.set_title(f'初始环境: {initial_dist_name}\nHPO优化参数 vs. 默认参数', fontsize=14, color=COLOR_DEFAULT_RADAR,
                     pad=50)

        legend_elements = [
            Line2D([0], [0], color=COLOR_DEFAULT_RADAR, lw=3, label='默认参数'),
            Line2D([0], [0], color=COLOR_HPO_RADAR, lw=3, label='HPO参数'),
            Polygon([[0, 0]], facecolor=COLOR_HPO_IMPROVEMENT, alpha=0.5, label='HPO 表现更优'),
            Polygon([[0, 0]], facecolor=COLOR_HPO_DECLINE, alpha=0.5, label='HPO 表现更差')
        ]
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=11,
                  frameon=False)

        plt.tight_layout(pad=4)

        # 创建唯一的文件名
        s_algo = sanitize_filename(algo_name)
        s_initial_dist = sanitize_filename(initial_dist_name)
        plot_filename = f"radar_perf_profile_{s_algo}_init_{s_initial_dist}.png"
        plot_path = plots_dir / plot_filename
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"带数值的雷达图已保存至: {plot_path}")
        except Exception as e_save:
            print(f"保存带数值的雷达图 {plot_path} 失败: {e_save}")
        plt.close(fig)