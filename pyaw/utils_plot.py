# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/31
@DESCRIPTION: 
"""
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike


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


def plt_subplots(x: ArrayLike, y1: ArrayLike, y2: ArrayLike, y3: ArrayLike, y4: ArrayLike) -> None:
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


# plotly
# interactive

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
