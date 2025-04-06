"""
获取事件对应的卫星轨迹
usage: 2种模式，一种处理一个轨道的，一种批量处理的（将未处理的trace_points的文件夹放在temp里面然后进行批量处理，得到其对应的多个轨道号的各自的多个轨迹。之后再移回来，然后删除temp）。

"""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime

import glob


def get_pkl_filenames(directory):
    """使用 glob 获取所有 .pkl 文件的完整路径或文件名"""
    pkl_paths = glob.glob(os.path.join(directory, "*.pkl"))
    pkl_fns = [os.path.basename(path) for path in pkl_paths]
    return pkl_fns


def datetime_converter(value, *, to="str"):
    """
    在 numpy.datetime64 和 字符串之间双向转换

    参数：
        value: 输入值（字符串或np.datetime64）
        to: 转换目标类型，可选 'str' 或 'datetime64'

    返回：
        转换后的值
    """
    if to == "str":
        if isinstance(value, np.datetime64):
            # 将 np.datetime64 转为字符串
            dt64 = np.datetime64(value, "us")  # 统一转为微秒精度
            dt_obj = dt64.astype(datetime)  # 转为Python datetime对象
            return dt_obj.strftime("%Y%m%dT%H%M%S")  # 输出无分隔符格式
        raise ValueError("输入必须是 np.datetime64 类型")

    elif to == "datetime64":
        if isinstance(value, str):
            # 将字符串转为 np.datetime64
            try:
                dt_obj = datetime.strptime(value, "%Y%m%dT%H%M%S")
                return np.datetime64(dt_obj)
            except ValueError:
                raise ValueError("字符串格式应为 YYYYMMDDTHHMMSS")
        raise ValueError("输入必须是字符串类型")

    raise ValueError("参数 'to' 必须是 'str' 或 'datetime64'")


BACH_MODE = False


if not BACH_MODE:
    trace_dir = r"G:\master\pyaw\scripts\results\aw_cases\archive\trace_points\pkl\12834"  # modify
    trace_fns = get_pkl_filenames(trace_dir)

    path = r"V:\aw\swarm\vires\auxiliaries\SW_OPER_MAGA_HR_1B\aux_SW_OPER_MAGA_HR_1B_12834_20160307T224621_20160308T001954.pkl"  # modify
    df = pd.read_pickle(path)
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
            save_fn = f"aux_SW_OPER_MAGA_HR_1B_{orbit_num}_{start_time}_{end_time}.pkl"
            save_path = os.path.join(save_dir, save_fn)
            if not Path(save_path).exists():
                df_clip = df[start_time:end_time].copy()
                df_clip.to_pickle(save_path)
                print(f"文件已保存: {save_path}")
            else:
                print(f"文件已存在，跳过: {save_path}")

        else:
            print("未匹配到有效信息")
else:
    trace_dir_root = r"G:\master\pyaw\scripts\results\aw_cases\archive\trace_points\pkl"
    trace_dir_name_pairs = [
        (entry.name, str(entry))
        for entry in Path(trace_dir_root).iterdir()
        if entry.is_dir()
    ]
    df_path_root = r"V:\aw\swarm\vires\auxiliaries\SW_OPER_MAGA_HR_1B"
    save_dir = r"G:\master\pyaw\scripts\results\aw_cases"
    pkl_paths = glob.glob(os.path.join(df_path_root, "*.pkl"))
    for trace_dir_name_pair in trace_dir_name_pairs:
        trace_dir_name = trace_dir_name_pair[0]
        trace_dir = trace_dir_name_pair[1]
        trace_fns = get_pkl_filenames(trace_dir)
        for pkl_path in pkl_paths:
            if trace_dir_name not in pkl_path:  # todo: 也许需要更加严格的判定
                continue
            df = pd.read_pickle(pkl_path)
            for fn in trace_fns:
                # 正则表达式匹配
                pattern = r"Swarm[A-Za-z]+_(\d+)_case_(\d{8}T\d{6})_(\d{8}T\d{6})\.pkl"
                match = re.search(pattern, fn)
                if match:
                    orbit_num = match.group(1)
                    start_time = match.group(2)
                    end_time = match.group(3)
                    print(orbit_num, start_time, end_time)
                    save_fn = (
                        f"aux_SW_OPER_MAGA_HR_1B_{orbit_num}_{start_time}_{end_time}.pkl"
                    )
                    save_path = os.path.join(save_dir, save_fn)
                    if not Path(save_path).exists():
                        df_clip = df[start_time:end_time].copy()
                        df_clip.to_pickle(save_path)
                        print(f"文件已保存: {save_path}")
                        time.sleep(0.5)
                    else:
                        print(f"文件已存在，跳过: {save_path}")

                else:
                    print("未匹配到有效信息")


print("---")
