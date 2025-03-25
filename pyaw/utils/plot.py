import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nptyping import NDArray, Datetime64, Float64
from plotly import express as px


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


def plot_where_is(series):
    """
    :param series: pd.Series
    :return: None
    """
    plt.plot(series.isna(), marker='.', linestyle='None', color='red')
    plt.title('NaN Positions in the Series')
    plt.show()
    return None


def compare_before_after_interpolate(series_: pd.Series, method_='linear', figsize=(10, 6)):
    """
    :param series_: the type of index is pd.datetime.
    :param method_:
    :param figsize:
    :return:
    """
    print(f'The number of NaN values: {series_.isna().sum()}')
    series_interpolate = series_.interpolate(method=method_)
    x = series_.index
    fig, axs = plt.subplots(3, figsize=figsize)
    axs[0].plot(x, series_, )
    axs[1].plot(x, series_interpolate, )
    axs[2].plot(x, series_, x)
    plt.show()


def plt_1f_2curve(x, y1, y2, title='', xlabel='', ylabel='', y1lable='', y2lable=''):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, label=y1lable)
    plt.plot(x, y2, label=y2lable)
    plt.legend()
    plt.show()


def plt_subplots(x, y1, y2, y3, y4) -> None:
    """
    plot a figure including 5 sub figures. the 1st sub figure on a line.
    :param x:
    :param y1:
    :param y2:
    :param y3:
    :param y4:
    :return:
    """
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)  # 3 行 2 列的网格
    # 0
    ax1 = fig.add_subplot(gs[0, :])  # 第 0 行，跨越所有列
    ax1.plot(x, y1)
    ax1.plot(x, y2)
    ax1.plot(x, y3)
    ax1.plot(x, y4)
    ax1.grid(which='both', linestyle='--', linewidth=0.5)

    # 1,0
    ax10 = fig.add_subplot(gs[1, 0])  # 第 1 行，第 0 列
    ax10.plot(x, y1)
    ax10.set_title('10 title')
    # 1,1
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(x, y2)
    # 2,0
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.plot(x, y3)
    # 2,1
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.plot(x, y4)
    # show figure
    plt.tight_layout()
    plt.show()
    return None


def plot_2signals_baselined(signal1: pd.Series, signal2: pd.Series,figsize=(10,4)):
    assert all(signal1.index == signal2.index), "signal1 and signal2 must have the same index"
    pass


def plt_mark_nan(series):
    plt.plot(series.isna(), marker='.', linestyle='None', color='red')
    plt.title('NaN Positions in b_baselined')
    plt.show()


def px_1f_2curve(x, y1, y2, title=''):
    # todo:: add code for real-time display of adjustment parameters
    df = pd.DataFrame({'x': x, 'y1': y1, 'y2': y2})
    fig = px.line(df, x='x', y=['y1', 'y2'], title=title)
    fig.show()


def px_beta(beta, me, mi):
    df = pd.DataFrame({'x': beta.index, 'y': beta.values})
    fig = px.line(df, x='x', y='y', title='Plasma beta')
    # add specific value horizontal line
    specific_v = me / mi
    fig.add_shape(type="line", x0=df['x'].min(), y0=specific_v, x1=df['x'].max(), y1=specific_v,
                  line=dict(color="Red", width=2, dash="dash"),  # type of line
                  )
    # add annotation
    fig.add_annotation(x=df['x'].max(), y=specific_v, text=f"$m_e/m_i$={specific_v}", showarrow=False, yshift=10)
    # title
    fig.update_layout(xaxis_title="Time (UT)", yaxis_title="$\\Beta$", )
    fig.show()
