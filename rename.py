import os


def replace_filenames(folder_path, old_str, new_str):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建原始文件的完整路径
        old_file_path = os.path.join(folder_path, filename)
        # 检查文件路径是否是文件而不是文件夹
        if os.path.isfile(old_file_path):
            # 构建新文件名
            new_filename = filename.replace(old_str, new_str)
            # 构建新文件的完整路径
            new_file_path = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)


# 调用示例
folder_path = 'DA/Ours/veis_to_bdd100k'  # 替换为你的文件夹路径
old_str = 'city'  # 要替换的字符串
new_str = 'bdd100k'  # 新的字符串
replace_filenames(folder_path, old_str, new_str)
