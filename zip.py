import concurrent.futures
import os
import zipfile


def compress_folders(directory):
    folders_to_compress = []

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            zip_file = folder_path + '.zip'
            if not os.path.exists(zip_file):
                folders_to_compress.append(folder_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(compress_folder, folders_to_compress)


def compress_folder(folder_path):
    with zipfile.ZipFile(folder_path + '.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))
    print(f"Compressed folder: {folder_path}")


# 指定要压缩的目录
root = 'work_dirs_all'

for dir in os.listdir(root):
    path = os.path.join(root, dir)
    print(path)
    compress_folders(path)
