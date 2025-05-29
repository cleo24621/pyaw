import numpy as np
import pandas as pd
import utils
from matplotlib import pyplot as plt

import pyaw.utils

fp = r"\\Diskstation1\file_three\aw\swarm\vires\SW_OPER_MAGA_HR_1B\SW_OPER_MAGA_HR_1B_11803_20151231T225422_20160101T002758.pkl"
df = pd.read_pickle(fp)
df_aux = pd.read_pickle(
    r"\\Diskstation1\file_three\aw\swarm\vires\AHY9U3~9\SW_OPER_MAGA_HR_1B\aux_SW_OPER_MAGA_HR_1B_11803_20151231T225422_20160101T002758.pkl"
)
bn, be, bc = get_3arrs(df["B_NEC"])
bn_mov_ave = pyaw.utils.move_average(bn, window=50 * 20, center=True, min_periods=1)
bn_disturb = bn - bn_mov_ave


def plot(datetimes, values, latitudes, qdlats, mlts, step=20000):
    # 创建图像
    fig, ax = plt.subplots(figsize=(18, 8))
    # 绘制数据
    ax.plot(datetimes, values, label="disturb magnetic field")
    datetime_ls = [
        np.datetime64("2015-12-31T23:06:00"),
        np.datetime64("2015-12-31T23:08:30"),
        np.datetime64("2015-12-31T23:20:00"),
        np.datetime64("2015-12-31T23:27:00"),
        np.datetime64("2015-12-31T23:53:00"),
        np.datetime64("2015-12-31T23:57:00"),
        np.datetime64("2016-01-01T00:04:00"),
        np.datetime64("2016-01-01T00:13:00"),
    ]
    for datetime_ in datetime_ls:
        plt.axvline(datetime_, color="r", linestyle="--")
    # plt.text(np.datetime64('2015-12-31T23:06'), max(values) * 0.9, f"{np.datetime64('2015-12-31T23:06')}", rotation=90, color='r', ha='right', va='top')
    # ax.set_ylabel('Value', color='b')
    # ax.tick_params(axis='y', labelcolor='b')

    # 设置时间轴标签
    datetime_ticks = datetimes[::step]
    latitude_ticks = latitudes[::step]
    qdlat_ticks = qdlats[::step]
    mlt_ticks = mlts[::step]
    ax.set_xticks(datetime_ticks)
    datetime_ticks_formatted = [
        t[11:19] for t in np.datetime_as_string(datetime_ticks, unit="s")
    ]
    ax.set_xticklabels(
        [
            (
                f"time: {t}\nlat: {lat:.2f}°\nqdlat: {qdlat:.2f}\nmlt: {mlt:.2f}"
                if i == 0
                else f"{t}\n{lat:.2f}\n{qdlat:.2f}\n{mlt:.2f}°"
            )
            for i, t, lat, qdlat, mlt in zip(
            range(len(datetime_ticks_formatted)),
            datetime_ticks_formatted,
            latitude_ticks,
            qdlat_ticks,
            mlt_ticks,
        )
        ]
    )

    plt.show()


plot(
    df.index.values,
    bn_disturb,
    df["Latitude"].values,
    df_aux["QDLat"].values,
    df_aux["MLT"].values,
)
