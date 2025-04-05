"""
usage: 需新建文件夹将所有需要组织的pkl文件放入，然后再进行处理
"""

import os
import shutil

import re


def organize_files_by_id(source_dir, pattern=r"_(\d{5})_"):
    """
    根据文件名中的ID（如12728）将文件分类到对应文件夹

    参数：
        source_dir: 源文件夹路径
        pattern: 匹配ID的正则表达式模式
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"源文件夹不存在: {source_dir}")

    for filename in os.listdir(source_dir):
        # 匹配文件名中的ID
        match = re.search(pattern, filename)
        if match:
            folder_id = match.group(1)
            target_dir = os.path.join(source_dir, folder_id)

            # 创建目标文件夹（如果不存在）
            os.makedirs(target_dir, exist_ok=True)

            # 移动文件
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved: {filename} -> {folder_id}/")


if __name__ == "__main__":
    source_directory = (
        r"G:\master\pyaw\scripts\results\aw_cases\temp"  # modify
    )
    organize_files_by_id(source_directory)
