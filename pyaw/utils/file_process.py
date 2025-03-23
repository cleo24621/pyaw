# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/21/2025 10:59
@Project     : pyaw
@Description : 和文件处理相关。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
from pathlib import Path

import h5py


def get_root_dir_path():
    """

    Returns:
        str: root dir path of the current project

    """
    # Get the current script directory
    current_dir = Path(__file__).resolve()

    # Traverse up to the root, looking for a specific file or folder such as '.git'.
    while not (current_dir / ".git").exists():
        current_dir = current_dir.parent
    print("Project Root Directory:", current_dir)
    return current_dir


def print_hdf5_variable_name(file_path: str) -> None:
    """
    打印HDF5文件中所有数据集的变量名

    Args:
        file_path: HDF5文件路径

    Returns:

    """
    with h5py.File(file_path, 'r') as h5file:
        h5file.visit(lambda name: print(name))


def main():
    pass


if __name__ == "__main__":
    main()
