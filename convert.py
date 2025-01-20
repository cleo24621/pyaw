"""将某文件夹下的pkl文件转换为csv文件"""

import os
from pathlib import Path

import pandas as pd


def get_pkl_filenames(folder_path):
    """
    Get all filenames with the .pkl extension in the specified folder.

    :param folder_path: Path to the folder to search for .pkl files.
    :return: List of filenames with the .pkl extension.
    """
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    return pkl_files


folder_path = Path(r'\\Diskstation1\file_three\aw\substorm\pkl030')
pkl_files = get_pkl_filenames(folder_path)

for pkl_file in pkl_files:
    pkl_file = Path(pkl_file)
    df = pd.read_pickle(folder_path / pkl_file)
    df.to_csv(f'{pkl_file.stem}.csv')
    print(f'convert {pkl_file.stem}.pkl to {pkl_file.stem}.csv')

# print("Pickle files:", pkl_files)
