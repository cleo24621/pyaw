# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/23/2025 1:14
@Project     : pyaw
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
from pathlib import Path

import pandas as pd
from pyaw.core.process_data import SwarmProcess
from pyaw.core import plot_satellite_orbit as pso

def main():

    df_path = Path(
        r"V:\aw\swarm\vires\auxiliaries\SW_OPER_MAGA_LR_1B") / "aux_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl"
    df = pd.read_pickle(df_path)

    lats = df['Latitude'].values
    indices = SwarmProcess.get_split_indices(lats)
    northern_slice = slice(*indices[0])
    orbit_lats = lats[northern_slice]
    orbit_lons = df['Longitude'].values[northern_slice]

    pso.plot_northern_hemisphere_satellite_orbit(orbit_lons, orbit_lats)


if __name__ == "__main__":
    main()
