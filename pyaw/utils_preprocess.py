# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/31
@DESCRIPTION: 
"""

from typing import Optional

import numpy as np
import pandas as pd
import pywt
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt


def set_bursts_nan_diff(series, threshold, print_: bool = True):
    """
    other method: series_copy[(np.abs(series_copy) / np.abs(series_copy).mean()) > 10] = np.nan
    :param series:
    :param threshold:
    :param print_:
    :return:
    """
    # todo: may need improve
    series_copy = series.copy()
    diff = series_copy.diff()
    # 设置一个突变检测阈值
    bursts = diff[diff.abs() > threshold]
    if print_:
        print(len(bursts))
        print(bursts)
    series_copy.loc[diff.abs() > threshold] = np.nan
    return series_copy  # return series_copy  # series_scores = (series - series.mean()) / series.std()  # # 设置阈值，通常 Z 分数大于 3 或小于 -3 的点可以认为是异常点，超过的设置为nan  # threshold = 2  # print(series[np.abs(series_scores) > threshold])  # series[np.abs(series_scores) > threshold] = np.nan  # return series


def set_bursts_nan_std(signal: pd.Series, std_times: float = 1.0, print_: bool = True):
    """
    :param signal:
    :param threshold:
    :param print_:
    :return:
    """
    signal_copy = signal.copy()
    threshold = std_times * signal_copy.std()
    bursts = np.abs(signal_copy - signal_copy.mean()) > threshold
    if print_:
        print(len(signal_copy[bursts]))
        print(signal_copy[bursts])
    signal_copy[bursts] = np.nan
    return signal_copy


def move_average(series_: pd.Series, window: int, min_periods: int = 1, draw: bool = False,
                 savefig: bool = False) -> pd.Series:
    """
    :param series_:
    :param window:
    :param draw:
    :param savefig:
    :return:
    """
    # series_mov_ave = series_.rolling(window=window).mean()
    # series_mov_ave = series_.rolling(window=window, min_periods=min_periods,center=True).mean()
    series_mov_ave = series_.rolling(window=window,center=True).mean()  # 'center=True' 得到的结果等于‘结果.mean()=0’
    # figure: before and after moving average comparison
    if draw:
        plt.figure()
        plt.plot(series_.index, series_)
        plt.plot(series_mov_ave.index, series_mov_ave)
        plt.xlabel('Time (UTC)')
        if savefig:
            plt.savefig(f'before and after moving average comparison')
    return series_mov_ave


def wavelet_smooth(series_: pd.Series, method='linear', wavelet='db4', level=6, threshold=0.2,
                   mode='soft') -> pd.Series:
    # process nan
    print(f'The number of NaN values: {series_.isna().sum()}')
    series_ = series_.interpolate(method=method)
    # 使用小波变换进行多尺度分解
    wavelet = wavelet  # 选择小波函数，例如 'db4' (Daubechies)
    coeffs = pywt.wavedec(series_, wavelet, level=level)  # 进行离散小波分解，设定分解层数
    # 处理高频细节系数，设置某些高频部分为零，以达到平滑效果
    threshold = threshold  # 设置阈值
    coeffs[1:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
    # 使用处理后的系数重构信号
    smoothed_signal = pywt.waverec(coeffs, wavelet)
    return smoothed_signal


# todo: savgol_filter()

class LHBFilter:
    def __init__(self, signal, fs, lowcut: Optional[float] = None, highcut: Optional[float] = None,
                 order: Optional[int] = 5):
        """

        :param signal: array_like
        :param fs:
        :param lowcut:
        :param highcut:
        :param order:
        """
        assert lowcut is not None or highcut is not None, "lowcut and highcut cannot be both None"
        self.signal = signal
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def get_filter(self):
        nyquist = 0.5 * self.fs  # 计算 Nyquist 频率
        if self.lowcut is not None and self.highcut is not None:
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            b, a = butter(self.order, [low, high], btype="band")
        elif self.lowcut is not None:
            low = self.lowcut / nyquist
            b, a = butter(self.order, low, btype="low")
        else:
            high = self.highcut / nyquist
            b, a = butter(self.order, high, btype="high")
        return b, a

    def apply_filter(self):
        b, a = self.get_filter()
        return filtfilt(b, a, self.signal)


def align_high2low(signal_high: pd.Series, signal_low: pd.Series) -> pd.Series:
    """
    signal_high aligned to signal_low (for 2 signal cross analysis)
    :param signal_high: index is pd.Timestamps type. with high sample rate, so long data length, like swarm vfm50 magnetic data
    :param signal_low: index is pd.Timestamps type. with low sample rate, so short data length, like swarm efi16 electric data
    :return: index is pd.Timestamps type. signal_high aligned to signal_low.
    """
    from scipy.interpolate import interp1d
    interp_func = interp1d(signal_high.index.astype('int64'), signal_high.values, kind='linear',
                           fill_value="extrapolate")
    return pd.Series(data=interp_func(signal_low.index.astype('int64')), index=signal_low.index)
