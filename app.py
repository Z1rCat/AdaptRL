import streamlit as st
from pathlib import Path
import subprocess
import sys
import pandas as pd
import os

# --- 从项目中导入必要的配置和函数 ---
# 假设 main_experiment.py 和其他工具脚本在同一目录下或在Python路径中
try:
    from config import ALL_AVAILABLE_ALGORITHMS, ALL_DISTRIBUTION_CONFIGS
    from main_experiment import define_plot_options
    from utils import calculate_distribution_complexity
except ImportError as e:
    st.error(f"导入模块失败: {e}。请确保 app.py 与您的项目文件位于同一目录。")
    st.stop()

# --- 应用配置和状态管理 ---
st.set_page_config(page_title="RL 适应性实验框架", layout="wide")

if 'experiment_running' not in st.session_state:
    st.session_state.experiment_running = False
if 'process' not in st.session_state:
    st.session_state.process = None
if 'final_results_dir' not in st.session_state:
    st.session_state.final_results_dir = None
if 'selected_plots_ids' not in st.session_state:
    st.session_state.selected_plots_ids = []

# --- 辅助函数 ---
@st.cache_data
def get_sorted_distributions():
    """计算分布的复杂度并排序，利用Streamlit缓存以避免重复计算。"""
    distributions_with_complexity = []
    for config_item in ALL_DISTRIBUTION_CONFIGS:
        # 注意：这里的复杂度计算参数是写死的，如果config.py变化需要同步
        complexity = calculate_distribution_complexity(config_item, 10.0, 100000)
        distributions_with_complexity.append({**config_item, "complexity": complexity})
    return sorted(distributions_with_complexity, key=lambda x: x.get("complexity", -1.0), reverse=True)

# --- UI 界面 ---
st.title("🚀 强化学习智能体适应性分析框架")
st.markdown("通过此界面配置并启动您的强化学习实验。实验将在后台运行，您可以在下方实时查看日志输出。")

# 获取数据
sorted_distributions = get_sorted_distributions()
dist_names = [f"{d['name']} (C: {d.get('complexity', -1.0):.2f})" for d in sorted_distributions]
dist_map = {f"{d['name']} (C: {d.get('complexity', -1.0):.2f})": d['name'] for d in sorted_distributions}

all_plot_options = define_plot_options([])
plot_names = [p['name'] for p in all_plot_options]
plot_map = {p['name']: p['id'] for p in all_plot_options}


# --- 参数配置侧边栏 ---
with st.sidebar:
    st.header("⚙️ 实验参数配置")
    
    selected_algo_list = st.multiselect(
        "1. 选择算法",
        options=ALL_AVAILABLE_ALGORITHMS,
        default=ALL_AVAILABLE_ALGORITHMS[0] if ALL_AVAILABLE_ALGORITHMS else None,
        help="选择一个或多个您想要测试的强化学习算法。"
    )

    selected_initial_dist_names_display = st.multiselect(
        "2. 选择初始训练分布",
        options=dist_names,
        default=dist_names[0] if dist_names else None,
        help="选择智能体进行初始训练时所处的环境。"
    )

    selected_target_scenarios_names_display = st.multiselect(
        "3. 选择目标适应场景",
        options=dist_names,
        default=dist_names[1] if len(dist_names) > 1 else None,
        help="选择智能体在初始训练后需要去适应的新环境。"
    )
    
    run_hpo_comparison = st.checkbox(
        "运行超参数优化 (HPO)",
        value=False,
        help="勾选此项后，将为每个算法组合运行HPO，并与默认参数进行对比。耗时较长。"
    )

    selected_plots_names = st.multiselect(
        "4. 选择要生成的图表",
        options=plot_names,
        default=[p_name for p_name, p_id in plot_map.items() if p_id in ['hpo_barchart', 'combined_curves']],
        help="选择在实验结束后自动生成的可视化图表。"
    )

    # 将显示名称映射回配置中的实际名称
    selected_initial_dist_list = [dist_map[name] for name in selected_initial_dist_names_display]
    selected_target_scenarios_list = [dist_map[name] for name in selected_target_scenarios_names_display]
    selected_plots_ids = [plot_map[name] for name in selected_plots_names]


# --- 主内容区 ---
col1, col2 = st.columns([1, 1])

with col1:
    start_button_pressed = st.button("🏁 开始运行实验", type="primary", use_container_width=True, disabled=st.session_state.experiment_running)

with col2:
    stop_button_pressed = False
    if st.session_state.experiment_running:
        if st.button("🛑 停止实验", use_container_width=True):
            if st.session_state.process:
                st.session_state.process.terminate()
                st.session_state.process = None
                st.session_state.experiment_running = False
                st.warning("实验已手动终止。")
                st.rerun()

# --- 日志输出区 ---
log_placeholder = st.empty()
log_container = log_placeholder.container(border=True)
with log_container:
    st.markdown("📋 **实验日志输出**")
    log_output_area = st.empty()
    if not st.session_state.experiment_running:
        log_output_area.info("请配置参数并点击“开始运行实验”。")


# --- 结果展示区 (仅在实验结束后显示) ---
if st.session_state.final_results_dir:
    results_path = Path(st.session_state.final_results_dir)
    st.subheader("📊 实验结果")
    st.markdown(f"所有生成的图表和日志都保存在以下目录中:\n`{results_path}`")

    # 尝试查找并显示结果摘要CSV
    try:
        summary_file = next(results_path.glob("ALL_RESULTS_SUMMARY_*.csv"))
        if summary_file.exists():
            st.markdown("### 最终结果摘要:")
            latest_summary = pd.read_csv(summary_file)
            st.dataframe(latest_summary)
    except (StopIteration, FileNotFoundError):
        st.warning("未找到最终结果摘要CSV文件。")
    except Exception as e:
        st.error(f"加载结果摘要时出错: {e}")

    # --- 新的、基于选择的绘图逻辑 ---
    st.markdown("### 您选择的图表:")

    # 最终修复：此映射中的关键字必须与 plotting_utils.py 中 savefig 生成的文件名严格对应
    PLOT_ID_TO_FILENAME_PATTERNS = {
        'eval_curves': {'contains': ['evaluation_reward_curves']},
        'boxplots': {'contains': ['adaptation_performance_boxplots']},
        'summary_bar': {'contains': ['overall_summary_barchart'], 'not_contains': ['hpo_comparison']},
        'learning_curves': {'contains': ['_learning_curve.png']},
        'reward_stages': {'contains': ['_stages_distribution.png']},
        'combined_curves': {'contains': ['_comb.png']},  # 修正: 对应 plot_combined_learning_curve
        'complexity_perf': {'contains': ['complexity_vs_performance']},
        'delta_rewards': {'contains': ['adaptation_delta_rewards']},
        'convergence_dynamics': {'contains': ['_convergence_metrics.png']},
        'hpo_profile_radar': {'contains': ['hpo_aggregated_performance_radar']},
        'hpo_barchart': {'contains': ['hpo_comparison_overall_summary_barchart']}, # 修正: 对应 plot_overall_summary_barchart_comparison
        'hpo_distrib_panels_final': {'contains': ['hpo_distrib_compare', '_final_eval.png']}, # 修正: 对应 plot_hpo_reward_distribution_panels
        'hpo_distrib_panels_initial': {'contains': ['hpo_distrib_compare', '_initial_eval.png']},# 修正: 对应 plot_hpo_reward_distribution_panels
    }

    all_plot_files = list(results_path.glob("**/*.png"))
    filtered_plot_files = []
    
    selected_ids = st.session_state.get('selected_plots_ids', [])
    
    if selected_ids:
        # 使用集合来避免重复添加文件
        matched_files = set()
        for plot_id in selected_ids:
            if plot_id in PLOT_ID_TO_FILENAME_PATTERNS:
                patterns = PLOT_ID_TO_FILENAME_PATTERNS[plot_id]
                must_contain = patterns.get('contains', [])
                must_not_contain = patterns.get('not_contains', [])
                
                for plot_file in all_plot_files:
                    name = plot_file.name
                    contains_all = all(c in name for c in must_contain)
                    contains_none = not any(nc in name for nc in must_not_contain)
                    
                    if contains_all and contains_none:
                        matched_files.add(plot_file)
        
        filtered_plot_files = sorted(list(matched_files)) # 从集合转换回排序列表
    else:
        # 如果没有记录选择（例如直接加载了有结果的页面），则显示全部
        filtered_plot_files = sorted(all_plot_files)

    # --- 分类并展示过滤后的图表 ---
    if not filtered_plot_files and selected_ids:
        st.warning("根据您的选择，没有找到对应的图表文件。请检查实验是否成功生成了图表。")
    else:
        # 全局图表 (在根目录下)
        global_plots = [p for p in filtered_plot_files if p.parent == results_path]
        
        # 按初始环境分类的图表 (在子目录下)
        from collections import defaultdict
        plots_by_subdir = defaultdict(list)
        for p in filtered_plot_files:
            if p.parent != results_path and p.parent.is_dir():
                plots_by_subdir[p.parent].append(p)

        if global_plots:
            with st.expander("全局汇总图表", expanded=True):
                for plot_file in sorted(global_plots):
                    st.image(str(plot_file), caption=plot_file.name)

        if plots_by_subdir:
            for sub_dir, plots_in_dir in sorted(plots_by_subdir.items()):
                with st.expander(f"初始环境: {sub_dir.name}", expanded=True):
                    for plot_file in sorted(plots_in_dir):
                        st.image(str(plot_file), caption=plot_file.name)

# --- 执行逻辑 ---
if start_button_pressed:
    # --- 输入验证 ---
    if not all([selected_algo_list, selected_initial_dist_list, selected_target_scenarios_list, selected_plots_ids]):
        st.error("错误：算法、初始分布、目标场景和要生成的图表均为必填项。")
    else:
        st.session_state.experiment_running = True
        st.session_state.final_results_dir = None # 重置上一次的运行结果
        st.session_state.selected_plots_ids = selected_plots_ids # 记录当前选择的图表
        
        # --- 构建命令行参数 ---
        py_executable = sys.executable  # 获取当前环境的Python解释器路径
        command = [
            py_executable, "main_experiment.py",
            "--non-interactive",
            "--algos", *selected_algo_list,
            "--initial-dists", *selected_initial_dist_list,
            "--target-dists", *selected_target_scenarios_list,
            "--plots", *selected_plots_ids
        ]
        if run_hpo_comparison:
            command.append("--run-hpo")

        st.info("实验已启动！正在执行命令: \n" + " ".join(f'"{c}"' if " " in c else c for c in command))

        # --- 运行子进程 ---
        proc_env = os.environ.copy()
        proc_env["PYTHONIOENCODING"] = "utf-8"
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            env=proc_env
        )
        st.session_state.process = process
        
        # --- 实时显示日志 ---
        log_content = ""
        log_output_area.code(log_content, language="log")

        process = st.session_state.process
        # 修复：在访问 process.stdout 之前，确保其存在
        while process and process.stdout and process.poll() is None:
            line = process.stdout.readline()
            if not line:
                break
            log_content += line
            log_output_area.code(log_content, language="log")
            
            # 捕获最终结果目录
            if "FINAL_RESULTS_DIR:" in line:
                try:
                    # 从日志行中提取路径
                    results_dir_str = line.split("FINAL_RESULTS_DIR:")[1].strip()
                    st.session_state.final_results_dir = results_dir_str
                except IndexError:
                    pass # 如果分割失败则忽略

        # 确保读取所有剩余输出
        if process:
            # 修复：同样在这里检查 stdout
            if process.stdout:
                remaining_output = process.stdout.read()
                if remaining_output:
                    log_content += remaining_output
                    log_output_area.code(log_content, language="log")
                    if "FINAL_RESULTS_DIR:" in remaining_output:
                        try:
                            # 改进的解析逻辑，更稳健地处理最后一行
                            results_dir_str = remaining_output.split("FINAL_RESULTS_DIR:")[-1].strip().splitlines()[0]
                            st.session_state.final_results_dir = results_dir_str
                        except IndexError:
                            pass
            
            return_code = process.wait()
            st.session_state.process = None
        
        st.session_state.experiment_running = False

        if return_code == 0:
            st.success("✅ 实验成功完成！")
        else:
            st.error(f"❌ 实验失败，返回码: {return_code}。请检查上面的日志以获取详细信息。")
        
        st.rerun()