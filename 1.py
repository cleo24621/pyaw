# -*- coding: utf-8 -*-
"""
@File        : 1.py
@Author      : cleo
@Date        : 2024/9/24 14:48
@Project     : PyWave
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2024, cleo
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : cleo
@Last Modified Date: 2024/9/24 14:48
"""
import os
from datetime import datetime, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from viresclient import SwarmRequest


def save_SW_EFIx_TCT16(start, end, satellite='A'):
    """
    start,end: like "20240812T000000", other formats are ok.
    """
    tct_vars = [
        # Satellite velocity in NEC frame
        "VsatC", "VsatE", "VsatN",
        # Geomagnetic field components derived from 1Hz product
        #  (in satellite-track coordinates)
        "Bx", "By", "Bz",
        # Electric field components derived from -VxB with along-track ion drift
        #  (in satellite-track coordinates)
        # Eh: derived from horizontal sensor
        # Ev: derived from vertical sensor
        "Ehx", "Ehy", "Ehz",
        "Evx", "Evy", "Evz",
        # Ion drift corotation signal, removed from ion drift & electric field
        #  (in satellite-track coordinates)
        "Vicrx", "Vicry", "Vicrz",
        # Ion drifts along-track from vertical (..v) and horizontal (..h) TII sensor
        "Vixv", "Vixh",
        # Ion drifts cross-track (y from horizontal sensor, z from vertical sensor)
        #  (in satellite-track coordinates)
        "Viy", "Viz",
        # Random error estimates for the above
        #  (Negative value indicates no estimate available)
        "Vixv_error", "Vixh_error", "Viy_error", "Viz_error",
        # Quasi-dipole magnetic latitude and local time
        #  redundant with VirES auxiliaries, QDLat & MLT
        "Latitude_QD", "MLT_QD",
        # Refer to release notes link above for details:
        "Calibration_flags", "Quality_flags",
    ]
    request = SwarmRequest()
    request.set_collection(f"SW_EXPT_EFI{satellite}_TCT16")
    request.set_products(measurements=tct_vars)
    data = request.get_between(start, end)
    df = data.as_dataframe()
    dir = f'data/Swarm/SW_EFIx_TCT16/{satellite}'
    fn = f'SW_EFI{satellite}_TCT16_{start}_{end}_1.csv'
    # 如果目录不存在则创建
    os.makedirs(dir, exist_ok=True)
    fp = os.path.join(dir, fn)
    # 检查文件是否存在
    if not os.path.exists(fp):
        # 文件不存在时保存
        df.to_csv(fp)
        print(f"File saved to: {fp}")
    else:
        # 文件已存在时提示
        print(f"File already exists at: {fp}, skipping save.")

def save_SW_MAGx_HR_1B(start, end, satellite='A'):
    request = SwarmRequest()
    request.set_collection(f"SW_OPER_MAG{satellite}_HR_1B")
    request.set_products(
        measurements=["B_NEC"],
    )
    data = request.get_between(
        start_time=start,
        end_time=end,
        asynchronous=False
    )
    df = data.as_dataframe()
    dir = f'data/Swarm/SW_MAGx_HR_1B/{satellite}'
    fn = f'SW_MAG{satellite}_HR_1B_{start}_{end}_1.csv'
    # 如果目录不存在则创建
    os.makedirs(dir, exist_ok=True)
    fp = os.path.join(dir, fn)
    # 检查文件是否存在
    if not os.path.exists(fp):
        # 文件不存在时保存
        df.to_csv(fp)
        print(f"File saved to: {fp}")
    else:
        # 文件已存在时提示
        print(f"File already exists at: {fp}, skipping save.")


def get_time_strs_forB(start, num_elements):
    """
        start: like datetime(2024, 8, 12, 0, 0, 0)
        """
    # 创建时间字符串列表
    time_strs = [
        (start + timedelta(hours=i)).strftime("%Y%m%dT%H%M%S")
        for i in range(num_elements)
    ]
    return time_strs

# 1 day
# time_strs = get_time_strs_forB(datetime(2024, 8, 12, 0, 0, 0),25)
# for i in range(24):
#     save_SW_MAGx_HR_1B(time_strs[i],time_strs[i+1], satellite='A')

# tl = ["20240812T000000","20240813T000000"]






def main():
    pass


if __name__ == "__main__":
    main()
