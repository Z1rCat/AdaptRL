# utils.py
import sys
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Any, Callable, Union, Dict  # Added Union, Dict
import numpy as np


# NOTE: _get_title_with_complexity was here, but it's more of a plotting helper.
# It's defined in plotting_utils.py and imported from there if needed by other modules,
# or just kept local to plotting_utils.py if only used there.

# --- 中文字体设置 ---
def set_chinese_font_for_matplotlib():
    # This print statement will appear in logs due to Tee
    # print("\n--- 正在配置Matplotlib中文字体支持 ---")
    try:
        plt.rcParams['font.family'] = 'sans-serif'  # Ensure sans-serif is the base family
        preferred_fonts = [
            'Microsoft YaHei', 'SimHei', 'Heiti SC', 'PingFang SC',
            'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
            'DengXian', 'FangSong', 'KaiTi', 'Arial Unicode MS'  # Added Arial Unicode MS as a common fallback
        ]
        available_font_names = {font.name for font in fm.fontManager.ttflist}
        found_font_name = next((font for font in preferred_fonts if font in available_font_names), None)

        current_sans_serif_list = plt.rcParams.get('font.sans-serif', [])
        # Remove duplicates while preserving order and potentially existing good fonts
        current_sans_serif_list = list(dict.fromkeys(current_sans_serif_list))

        if found_font_name:
            # Ensure the found font is at the beginning of the list
            if found_font_name in current_sans_serif_list:
                current_sans_serif_list.remove(found_font_name)
            new_sans_serif_list = [found_font_name] + current_sans_serif_list
            plt.rcParams['font.sans-serif'] = new_sans_serif_list
            # print(f"成功：已找到并优先设置字体 '{found_font_name}' 用于中文显示。当前sans-serif列表: {plt.rcParams['font.sans-serif']}")
        else:
            # print("警告：在Matplotlib的已知字体中未找到推荐的特定中文字体。")
            # Attempt to add some common fallbacks if not already present
            fallback_fonts_to_add = [f for f in preferred_fonts if
                                     f not in current_sans_serif_list]  # Add those not present

            # Ensure 'sans-serif' itself is at the very end as the ultimate fallback
            if 'sans-serif' in current_sans_serif_list:
                current_sans_serif_list.remove('sans-serif')

            new_sans_serif_list = current_sans_serif_list + fallback_fonts_to_add + ['sans-serif']
            plt.rcParams['font.sans-serif'] = list(dict.fromkeys(new_sans_serif_list))  # Remove duplicates
            # print(f"     将尝试使用后备字体列表: {plt.rcParams['font.sans-serif']}")
            # print("     如果图片中的中文显示为方块或乱码，请确保操作系统已安装支持中文的字体并清理Matplotlib缓存。")
    except Exception as e_font:
        print(f"错误：在配置中文字体时发生异常: {e_font}")
        # Fallback to a very basic sans-serif list
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans',
                                           'sans-serif']  # Common cross-platform fallbacks

    plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign displays correctly
    # print("--- Matplotlib中文字体支持配置结束 ---\n")


# --- 文件名清理 ---
def sanitize_filename(filename: str) -> str:
    """通过移除或替换非法字符来清理字符串，使其成为有效的文件名。"""
    if not isinstance(filename, str):
        filename = str(filename)
    sanitized = re.sub(r'[^\w\s\.\-\(\)一-龥]', '', filename)  # Allow CJK, alphanumeric, basic punctuation
    sanitized = re.sub(r'\s+', '_', sanitized)  # Replace whitespace with underscore
    sanitized = sanitized.strip('._- ')  # Strip leading/trailing junk
    return sanitized if sanitized else "untitled"  # Ensure not empty


# --- 输出重定向 Tee ---
class Tee:
    """
    将print输出同时重定向到多个文件流 (例如控制台和文件)
    并正确处理原始标准输出/错误的保存和恢复。
    """

    def __init__(self, stdout_original, file_obj, stderr_original=None):
        self.stdout_original = stdout_original
        self.stderr_original = stderr_original if stderr_original else stdout_original  # Default stderr to stdout if not given
        self.file_obj = file_obj
        self.is_closed = False

    def write(self, text):
        if self.is_closed:
            # If Tee is closed, try writing to original stdout to avoid losing output
            # This case should ideally not be hit if close() is managed properly.
            try:
                self.stdout_original.write(text)
                self.stdout_original.flush()
            except:  # Ignore errors if original stdout is also problematic
                pass
            return

        try:
            self.stdout_original.write(text)
            self.stdout_original.flush()  # Flush original stdout immediately
        except Exception as e_stdout:
            # print(f"Tee: Error writing to original stdout: {e_stdout}", file=self.stderr_original) # Avoid recursion
            pass  # Avoid printing error messages that might themselves cause issues

        if self.file_obj and not self.file_obj.closed:
            try:
                self.file_obj.write(text)
                self.file_obj.flush()  # Flush file immediately
            except Exception as e_file:
                # print(f"Tee: Error writing to file {self.file_obj.name}: {e_file}", file=self.stderr_original)
                pass

    def flush(self):
        if self.is_closed: return
        try:
            self.stdout_original.flush()
        except Exception:
            pass
        if self.file_obj and not self.file_obj.closed:
            try:
                self.file_obj.flush()
            except Exception:
                pass

    def close(self):
        if self.is_closed:
            return
        self.is_closed = True  # Mark as closed first to prevent recursive writes in case of errors during close

        # Restore original stdout and stderr
        # Important: Check if sys.stdout/stderr are still this Tee instance
        # This prevents issues if close is called multiple times or if streams were restored by other means.
        if sys.stdout is self:
            sys.stdout = self.stdout_original
        if self.stderr_original and sys.stderr is self:  # Only restore stderr if it was also redirected to this Tee
            sys.stderr = self.stderr_original

        # Close the file object if it's not one of the original streams
        if self.file_obj and self.file_obj not in (self.stdout_original, self.stderr_original):
            if hasattr(self.file_obj, 'close') and not self.file_obj.closed:
                try:
                    self.file_obj.close()
                except Exception as e_close_file:
                    # If an error occurs while closing the file, print it to the (now restored) original stderr.
                    # This should not go through the Tee's write method.
                    if self.stderr_original:
                        print(f"Tee: Error closing file object: {e_close_file}", file=self.stderr_original)
                    else:  # Fallback if stderr_original was None (should not happen with current main_experiment)
                        print(f"Tee: Error closing file object: {e_close_file}")


def get_samples_for_dist(dist_config: Dict[str, Any], n_samples: int) -> np.ndarray:
    """根据分布配置生成样本"""
    dist_type = dist_config["dist_type"]
    params = dist_config["params"]
    # This local generator dict is fine, or it could be part of a class or global if used more widely
    _DIST_GENERATORS_FOR_SAMPLING = {
        'normal': lambda p, n: np.random.normal(p['mu'], p['sigma'], size=n),
        'uniform': lambda p, n: np.random.uniform(p['low'], p['high'], size=n),
        'exponential': lambda p, n: np.random.exponential(p['scale'], size=n),
        'poisson': lambda p, n: np.random.poisson(p['lam'], size=n),
        'gamma': lambda p, n: np.random.gamma(p['shape'], p['scale'], size=n),
        'beta': lambda p, n: p.get('min_val', 0) + \
                             (p.get('max_val', 1) - p.get('min_val', 0)) * \
                             np.random.beta(p['alpha'], p['beta'], size=n),
        'laplace': lambda p, n: np.random.laplace(p['mu'], p['b'], size=n)
    }
    if dist_type not in _DIST_GENERATORS_FOR_SAMPLING:
        raise ValueError(f"未知分布类型用于采样: {dist_type}")
    return _DIST_GENERATORS_FOR_SAMPLING[dist_type](params, n_samples)


def calculate_distribution_complexity(
        dist_config: Dict[str, Any],
        threshold: float,
        n_samples_for_cdf_estimation: int = 100000  # Default from config
) -> float:
    """
    通过采样估算分布的复杂度。
    复杂度定义为 P_smaller / P_larger。
    P_smaller = min(P(X <= threshold), P(X > threshold))
    P_larger = max(P(X <= threshold), P(X > threshold))
    值域 [0, 1]，越接近1表示越难区分，越复杂。越接近0表示越容易区分。
    """
    try:
        # print(f"Calculating complexity for {dist_config.get('name','Unknown')} with threshold {threshold}") # Debug
        samples = get_samples_for_dist(dist_config, n_samples_for_cdf_estimation)

        p_less_equal = np.sum(samples <= threshold) / n_samples_for_cdf_estimation
        p_greater = 1.0 - p_less_equal

        # Handle edge cases where probabilities are 0 or 1 to avoid division by zero or NaN
        if p_greater == 0.0 and p_less_equal == 0.0:  # Should not happen if n_samples > 0 unless distribution is ill-defined for sampling
            complexity = 0.0
        elif p_greater == 0.0:  # All samples <= threshold (p_less_equal is 1.0)
            complexity = 0.0  # Very easy to distinguish, p_smaller is 0 (p_greater)
        elif p_less_equal == 0.0:  # All samples > threshold (p_greater is 1.0)
            complexity = 0.0  # Very easy to distinguish, p_smaller is 0 (p_less_equal)
        elif p_less_equal == p_greater:  # Probabilities are equal (0.5 each)
            complexity = 1.0  # Most complex
        else:
            p_smaller = min(p_less_equal, p_greater)
            p_larger = max(p_less_equal, p_greater)
            complexity = p_smaller / p_larger

        # print(f"  P(X<={threshold})={p_less_equal:.3f}, P(X>{threshold})={p_greater:.3f}, Complexity={complexity:.3f}") # Debug
        return complexity
    except Exception as e:
        print(f"计算分布 '{dist_config.get('name', 'Unknown')}' 的复杂度时出错: {e}")
        return -1.0  # Return an invalid/error indicator


# --- 交互式选择 ---
def select_items_interactively(
        items: List[Any],
        item_type_name: str,
        display_func: Callable[[Any], str] = lambda x: str(x),
        allow_multiple: bool = True
) -> Union[List[Any], Any, None]:
    """
    提供交互式命令行界面，让用户从列表中选择一个或多个项目。
    """
    if not items:
        print(f"没有可用的{item_type_name}。")
        return [] if allow_multiple else None

    prompt_suffix = "(输入数字，多个用逗号隔开，或输入 'all' 选择全部)" if allow_multiple else "(输入数字选择一项)"
    print(f"\n请选择 {item_type_name} {prompt_suffix}:")  # Removed "要测试的" for generality
    for i, item in enumerate(items):
        print(f"  {i + 1}. {display_func(item)}")

    while True:
        try:
            choice_str = input(f"输入您的选择 (或输入 'q' 退出选择): ").strip().lower()
            if not choice_str:
                print("输入为空，请重新选择。")
                continue
            if choice_str == 'q':
                print(f"已取消选择 {item_type_name}。")
                return [] if allow_multiple else None

            if allow_multiple and choice_str == 'all':
                return list(items)

            chosen_indices_str = choice_str.split(',')
            chosen_indices = []
            for x_str in chosen_indices_str:
                x_str = x_str.strip()
                if not x_str: continue  # Skip empty strings if user enters ",,"
                chosen_indices.append(int(x_str) - 1)

            if not allow_multiple and len(chosen_indices) > 1:
                print("此选项只允许选择一项。请重新输入。")
                continue

            selected_items = []
            valid_choice = True
            for idx in chosen_indices:
                if 0 <= idx < len(items):
                    selected_items.append(items[idx])
                else:
                    print(f"无效的选项索引: {idx + 1}。范围应在 1 到 {len(items)} 之间。")
                    valid_choice = False
                    break  # Stop processing this set of choices

            if valid_choice and selected_items:
                if not allow_multiple:
                    return selected_items[0]

                    # For multiple selections, ensure unique items are returned, preserving order.
                # If items are dicts with 'name', use 'name' for uniqueness. Otherwise, direct uniqueness.
                if all(isinstance(s, dict) and 'name' in s for s in selected_items):
                    unique_selected_map = {}
                    for s_item in selected_items:
                        if s_item['name'] not in unique_selected_map:
                            unique_selected_map[s_item['name']] = s_item
                    return list(unique_selected_map.values())
                else:
                    # General unique item selection, preserving order (Python 3.7+)
                    return list(dict.fromkeys(selected_items))

            elif not selected_items and valid_choice:
                print("未选择任何有效项。请重新选择。")
            # If not valid_choice, error message already printed.

        except ValueError:
            print("输入格式错误。请使用数字" + ("和逗号（如果允许多选）。" if allow_multiple else "。"))
        except Exception as e:  # Catch any other unexpected errors during selection
            print(f"{item_type_name} 选择过程中发生错误: {e}")
            return [] if allow_multiple else None