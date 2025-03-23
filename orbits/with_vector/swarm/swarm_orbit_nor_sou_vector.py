# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/20/2025 10:39
@Project     : pyaw
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import utils.coordinate
import utils.data
from pyaw import utils


def get_split_indices(array):
    """
    Split array into northern and southern indices
    """
    neg_indices = np.where(array < 0)[0]
    if not neg_indices.size:
        return (0, len(array)), (len(array), len(array))

    start_south = neg_indices[0]
    pos_indices = np.where(array[start_south:] >= 0)[0]
    end_south = start_south + pos_indices[0] if pos_indices.size else len(array)

    return (0, start_south), (start_south, end_south)


def get_df_e_tct16(file_path):
    """
    get the data of tct16 file
    :param file_path:
    :return:
    """
    # 提取文件名和扩展名
    file_name = os.path.basename(file_path)
    assert 'TCT16' in file_name
    # 读取数据（增加异常处理）
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df_e = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit()
    return df_e


def get_df_b_mag_lr(file_path,file_path_igrf):
    """
    get the data of mag_lr file and the data of mag_lr_igrf file
    :param file_path:
    :return:
    """
    # 提取文件名和扩展名
    file_name = os.path.basename(file_path)
    file_name_igrf = os.path.basename(file_path_igrf)
    assert 'LR' in file_name and 'LR' in file_name_igrf
    parts = file_name.split('_')
    parts_igrf = file_name_igrf.split('_')
    orbit_number = parts[5]
    orbit_number_igrf = parts_igrf[6]
    assert orbit_number == orbit_number_igrf
    # 读取数据（增加异常处理）
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df_b = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit()
    try:
        # modify (same orbit number)
        if not os.path.exists(file_path_igrf):
            raise FileNotFoundError(f"File not found: {file_path_igrf}")
        df_b_igrf = pd.read_pickle(file_path_igrf)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit()
    return df_b, df_b_igrf


def process_data_tct16(df_e,hemisphere='north'):
    assert hemisphere in ['north', 'south']
    # 提取数据
    lats = df_e['Latitude'].values
    lons = df_e['Longitude'].values
    Ehx = df_e['Ehx'].values
    Ehy = df_e['Ehy'].values
    VsatN = df_e['VsatN'].values
    VsatE = df_e['VsatE'].values
    # 计算旋转矩阵
    rotmat_nec2sc, rotmat_sc2nec = utils.coordinate.get_rotmat_nec2sc_sc2nec(VsatN, VsatE)
    en, ee = utils.coordinate.do_rotation(-Ehx, -Ehy, rotmat_sc2nec)  # en: e north; ee: e east
    # nor and south
    indices = get_split_indices(lats)
    northern_slice = slice(*indices[0])
    southern_slice = slice(*indices[1])
    if hemisphere == 'north':
        lons_hemi = lons[northern_slice]
        lats_hemi = lats[northern_slice]
        ee_hemi = ee[northern_slice]
        en_hemi = en[northern_slice]
    else:
        lons_hemi = lons[southern_slice]
        lats_hemi = lats[southern_slice]
        ee_hemi = ee[southern_slice]
        en_hemi = en[southern_slice]
    # 正则化矢量
    magnitudes = np.sqrt(ee_hemi ** 2 + en_hemi ** 2)
    min_mag = 1e-8
    max_mag = np.max(np.clip(magnitudes, a_min=min_mag, a_max=None))
    scale_factor = 1.8
    ee_hemi_norm = (ee_hemi / max_mag) * scale_factor
    en_hemi_norm = (en_hemi / max_mag) * scale_factor
    return lons_hemi,lats_hemi,ee_hemi_norm, en_hemi_norm


def process_data(df_b,df_b_igrf):
    # 提取数据
    lats = df_b['Latitude'].values
    lons = df_b['Longitude'].values

    Bn, Be, _ = utils.data.get_3arrs(df_b['B_NEC'].values)
    bn_igrf, be_igrf, _ = utils.data.get_3arrs(df_b_igrf['B_NEC_IGRF'].values)
    bn = Bn - bn_igrf
    be = Be - be_igrf

    # 北半球筛选（使用布尔索引）
    mask_northern = lats >= 0
    orbit_lats = lats[mask_northern]
    orbit_lons = lons[mask_northern]
    be_nor = be[mask_northern]
    bn_nor = bn[mask_northern]

    # 正则化矢量
    magnitudes = np.sqrt(be_nor**2 + bn_nor**2)
    min_mag = 1e-8
    max_mag = np.max(np.clip(magnitudes, a_min=min_mag, a_max=None))
    scale_factor = 1.8
    be_nor_norm = (be_nor / max_mag) * scale_factor
    bn_nor_norm = (bn_nor / max_mag) * scale_factor

    return orbit_lons, orbit_lats, be_nor_norm, bn_nor_norm


def test_magnetic():
    """

    :return:
    """


def test_electric():
    """

    :return:
    """


def main():
    pass


if __name__ == "__main__":
    # magnetic field
    # dir = Path(r"V:\aw\swarm\vires\measurements\SW_OPER_MAGA_LR_1B")
    # dir_igrf = Path(r"V:\aw\swarm\vires\igrf\SW_OPER_MAGA_LR_1B")
    # file_name = Path("SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl")
    # file_name_igrf = Path("IGRF_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl")
    # file_path,file_path_igrf = str(dir / file_name), str(dir_igrf / file_name_igrf)
    # df_b, df_b_igrf = get_df_b_mag_lr(file_path,file_path_igrf)

    # electric field
