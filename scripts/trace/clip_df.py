import os
import re

import numpy as np
import pandas as pd
from datetime import datetime

import glob


def get_pkl_filenames(directory):
    """使用 glob 获取所有 .pkl 文件的完整路径或文件名"""
    # 获取完整路径
    pkl_paths = glob.glob(os.path.join(directory, "*.pkl"))

    # 仅提取文件名（可选）
    pkl_files = [os.path.basename(path) for path in pkl_paths]
    return pkl_files


def datetime_converter(value, *, to='str'):
    """
    在 numpy.datetime64 和 字符串之间双向转换

    参数：
        value: 输入值（字符串或np.datetime64）
        to: 转换目标类型，可选 'str' 或 'datetime64'

    返回：
        转换后的值
    """
    if to == 'str':
        if isinstance(value, np.datetime64):
            # 将 np.datetime64 转为字符串
            dt64 = np.datetime64(value, 'us')  # 统一转为微秒精度
            dt_obj = dt64.astype(datetime)  # 转为Python datetime对象
            return dt_obj.strftime("%Y%m%dT%H%M%S")  # 输出无分隔符格式
        raise ValueError("输入必须是 np.datetime64 类型")

    elif to == 'datetime64':
        if isinstance(value, str):
            # 将字符串转为 np.datetime64
            try:
                dt_obj = datetime.strptime(value, "%Y%m%dT%H%M%S")
                return np.datetime64(dt_obj)
            except ValueError:
                raise ValueError("字符串格式应为 YYYYMMDDTHHMMSS")
        raise ValueError("输入必须是字符串类型")

    raise ValueError("参数 'to' 必须是 'str' 或 'datetime64'")

trace_dir = r"G:\master\pyaw\scripts\results\aw_cases\archive\trace_points\pkl\12728"
trace_fns = get_pkl_filenames(trace_dir)

path = r"V:\aw\swarm\vires\measurements\SW_OPER_MAGA_HR_1B\SW_OPER_MAGA_HR_1B_12728_20160301T012924_20160301T030258.pkl"
df = pd.read_pickle(path)
df_pos = df[['Radius','Latitude','Longitude']].copy()
save_dir = r"G:\master\pyaw\scripts\results\aw_cases"
for fn in trace_fns:
    # 正则表达式匹配
    pattern = r"Swarm[A-Za-z]+_(\d+)_case_(\d{8}T\d{6})_(\d{8}T\d{6})\.pkl"
    match = re.search(pattern, fn)
    
    if match:
        orbit_num = match.group(1)
        start_time = match.group(2)
        end_time = match.group(3)
        print(orbit_num, start_time, end_time)
        save_fn = f"SW_OPER_MAGA_HR_1B_{orbit_num}_{start_time}_{end_time}.pkl"
        save_path = os.path.join(save_dir,save_fn)
        df_clip = df_pos[start_time:end_time].copy()
        df_clip.to_pickle(save_path)
    else:
        print("未匹配到有效信息")



# st_str = "20160301T014840"
# st = np.datetime64(datetime.strptime(st_str, "%Y%m%dT%H%M%S"))
# et_str = "20160301T014900"
# et = np.datetime64(datetime.strptime(et_str, "%Y%m%dT%H%M%S"))
#
# df_clip = df[st:et]
# save_dir = r"G:\master\pyaw\scripts\results\aw_cases"
# save_name = f"SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258"

print("---")