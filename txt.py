import os


def convert_py_to_txt_in_same_directory():
    """
    自动读取与此脚本同一目录下的所有.py文件，并将其内容输出为同名的.txt文档。
    """
    try:
        # 获取当前脚本的绝对路径
        script_path = os.path.abspath(__file__)
        # 获取当前脚本所在的目录
        script_dir = os.path.dirname(script_path)
    except NameError:
        # 如果在交互式环境（如Jupyter）中运行 __file__ 可能未定义
        # 这种情况下，我们假设目标目录是当前工作目录
        script_dir = os.getcwd()
        print(f"警告: 无法通过 __file__ 确定脚本目录，将使用当前工作目录: {script_dir}")

    print(f"将在目录 '{script_dir}' 中查找 .py 文件...")

    found_py_files = False
    converted_count = 0

    # 遍历目录中的所有文件和文件夹
    for item_name in os.listdir(script_dir):
        # 构建完整的文件路径
        item_path = os.path.join(script_dir, item_name)

        # 检查是否是文件并且以 .py 结尾
        if os.path.isfile(item_path) and item_name.endswith(".py"):
            found_py_files = True

            # 如果这个 .py 文件就是当前运行的脚本本身，可以选择跳过或处理
            # if item_path == script_path:
            #     print(f"跳过脚本本身: {item_name}")
            #     continue

            # 获取不带扩展名的文件名
            base_name = os.path.splitext(item_name)[0]
            # 构建新的 .txt 文件名和路径
            txt_file_name = base_name + ".txt"
            txt_file_path = os.path.join(script_dir, txt_file_name)

            print(f"正在处理: {item_name} -> {txt_file_name}")

            try:
                # 读取 .py 文件内容
                with open(item_path, 'r', encoding='utf-8') as py_file:
                    content = py_file.read()

                # 将内容写入 .txt 文件
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(content)

                print(f"成功创建: {txt_file_name}")
                converted_count += 1
            except Exception as e:
                print(f"处理文件 {item_name} 时发生错误: {e}")

    if not found_py_files:
        print("目录中未找到 .py 文件。")
    else:
        print(f"\n处理完成。共转换了 {converted_count} 个文件。")


if __name__ == "__main__":
    convert_py_to_txt_in_same_directory()