# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/21/2025 20:12
@Project     : pyaw
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from nptyping import NDArray, Datetime64, Float64


def plot_xticks_of_times_lats_qdlats_mlts(
    datetimes: NDArray[Datetime64],
    values: NDArray[Float64],
    latitudes: NDArray[Float64],
    qdlats: NDArray[Float64],
    mlts: NDArray[Float64],
    step: int = 20000,
) -> tuple[plt.Figure, plt.Axes]:
    """
    e.g., plot_with_x_dt_lat_qdlat_mlt(df.index.values,bn_disturb,df['Latitude'].values,df_aux['QDLat'].values,df_aux['MLT'].values)
    :param datetimes:
    :param values:
    :param latitudes:
    :param qdlats: 地磁纬度
    :param mlts: 磁地方时
    :param step:
    :return:
    """
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

    return fig, ax


def main():
    pass


if __name__ == "__main__":
    main()
