# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/19/2025 9:34
@Project     : pyaw
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
from pathlib import Path

import numpy as np

from zh1 import zh1




def get_orbit_num_indicator_st_et(file_name):
    """
    1: ascending (south to north)
    0: descending (north to south)
    :return:
    """
    parts = file_name.split("_")
    part = parts[6]
    assert part[-1] in ["0","1"]
    start_time = parts[7] + "_" + parts[8]
    end_time = parts[9] + "_" + parts[10]
    return parts[6][:-1],parts[6][-1],start_time,end_time

def get_split_indices(lats,indicator):
    assert indicator in ["1", "0"]
    if all(lats>0):
        return (0, len(lats)), (len(lats), len(lats))
    elif all(lats<0):
        return (len(lats), len(lats)), (0, len(lats))
    elif indicator == "1":
        start_north = np.where(lats>0)[0][0]
        return (start_north,None),(0,start_north)  # north, south slice
    else:
        start_south = np.where(lats<0)[0][0]
        return (0,start_south),(start_south,None)  # north, south slice


def test_get_split_indices():
    """
    make sure the indices (split north and south data) are correct.
    :return:
    """
    # 0 descending (north to south)
    file_name = "CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5"
    df_path = Path(
        r"V:\aw\zh1\efd\ulf\2a\20210401_20210630\CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5")
    indicator = get_orbit_num_indicator_st_et(file_name)[1]
    efd = zh1.EFDULF(df_path)
    dfs = efd.dfs
    lats = dfs['GEO_LAT'].squeeze().values
    indices = get_split_indices(lats,indicator)
    northern_slice = slice(*indices[0])
    southern_slice = slice(*indices[1])
    orbit_lats_north = lats[northern_slice]
    orbit_lats_south = lats[southern_slice]
    assert all(orbit_lats_north>0)
    assert all(orbit_lats_south<0)

    # 1 ascending (south to north)
    file_name = "CSES_01_EFD_1_L2A_A1_175381_20210401_012158_20210401_015642_000.h5"
    df_path = Path(
        r"V:\aw\zh1\efd\ulf\2a\20210401_20210630\CSES_01_EFD_1_L2A_A1_175381_20210401_012158_20210401_015642_000.h5")
    indicator = get_orbit_num_indicator_st_et(file_name)[1]
    efd = zh1.EFDULF(df_path)
    dfs = efd.dfs
    lats = dfs['GEO_LAT'].squeeze().values
    indices = get_split_indices(lats, indicator)
    northern_slice = slice(*indices[0])
    southern_slice = slice(*indices[1])
    orbit_lats_north = lats[northern_slice]
    orbit_lats_south = lats[southern_slice]
    assert all(orbit_lats_north > 0)
    assert all(orbit_lats_south < 0)



def main():
    pass


if __name__ == "__main__":
    main()
