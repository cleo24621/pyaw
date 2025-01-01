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
from scipy.interpolate import interpolate
from scipy.signal import butter, filtfilt


def get_3arrays(array):
    """
    :param array: like np.array([[a1,b1,c1],[a2,b2,c2],...]). like B_NEC column of the df_b get from MAGx_HR_1B file.
    :return: 3 arrays. np.array([a1,a2,...]), np.array([b1,b2,...]), np.array([c1,c2,...]).
    """
    bn = []
    be = []
    bc = []
    for ndarray_ in array:
        bn.append(ndarray_[0])
        be.append(ndarray_[1])
        bc.append(ndarray_[2])
    bn = np.array(bn)
    be = np.array(be)
    bc = np.array(bc)
    return bn, be, bc

def get_rotation_matrices_nec2sc_sc2nec(VsatN,VsatE):
    """
    :param VsatN: velocity of satellite in the north direction
    :param VsatE: velocity of satellite in the east direction
    :return: rotation_matrix_2d_nec2sc, rotation_matrix_2d_sc2nec
    """
    theta = np.arctan(VsatE / VsatN)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # Stack components to construct the rotation matrices
    rotation_matrix = np.array([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])
    # Transpose axes to create a (3, 2, 2) array
    rotation_matrix_2d_nec2sc = np.transpose(rotation_matrix, (2, 0, 1))
    rotation_matrix_2d_sc2nec = rotation_matrix_2d_nec2sc.transpose(0, 2, 1)
    return rotation_matrix_2d_nec2sc, rotation_matrix_2d_sc2nec


def do_rotation(coordinates1, coordinates2, rotation_matrix):
    """
    :param coordinates1: N of NEC or X of S/C
    :param coordinates2: E of NEC or Y of S/C
    :param rotation_matrix: one of the rotation matrices returned by get_rotation_matrices_nec2sc_sc2nec
    :return: rotation of the coordinates
    """
    vectors12 = np.stack((coordinates1, coordinates2), axis=1)
    vectors12_rotated = np.einsum('nij,nj->ni', rotation_matrix, vectors12)
    return vectors12_rotated[:,0],vectors12_rotated[:,1]


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


def set_outliers_nan_std(array, std_times: float = 1.0, print_: bool = True):
    """
    :param array: the array to process
    :param std_times: standard deviation times
    :param print_: print the outliers or not
    :return: the array with outliers set to nan
    """
    array_copy = array.copy()
    threshold = std_times * np.std(array_copy)
    bursts = np.abs(array_copy - np.mean(array_copy)) > threshold
    if print_:
        print(len(array_copy[bursts]))
        print(array_copy[bursts])
    array_copy[bursts] = np.nan
    return array_copy


def get_array_interpolated(x,y):
    """
    :param x: ndarray consisting of np.datetime64.
    :param y:
    :return:
    """
    y_copy = y
    # Mask for missing values
    mask = np.isnan(y_copy)
    # Interpolate
    y_copy[mask] = interpolate.interp1d(x[~mask].astype('int'), y_copy[~mask], kind='linear')(
        x[mask].astype('int'))  # note:: 当x是时间类型事，该方法也支持
    return y_copy

def move_average(array,window, center:bool=True,min_periods: int|None = None):
    """
    :param min_periods: the 'min_periods' parameter of the series.rolling() function
    :param center: the 'center' parameter of the series.rolling() function
    :param window: the window of the moving average. equal to fs * (the seconds of the window), and the windows must be an integer.
    :param array: the array to process
    :return:
    """
    assert type(window) == int, "window must be an integer"
    # todo:: use the plot of the later part to verify the 'center', 'min_periods' parameters
    array_series = pd.Series(array)
    array_series_mov_ave = array_series.rolling(window=window, center=center,min_periods=min_periods).mean()  # 'center=True' 得到的结果等于‘结果.mean()=0’，即经过b-b.mean()（baselined）
    return array_series_mov_ave.values

def transform_time_string_to_datetime64ns(time_string):
    """
    :param time_string: str. like "20160311T064700".
    :return: np.datetime64[ns]
    """
    # Insert delimiters to make it ISO 8601 compliant
    formatted_string = time_string[:4] + "-" + time_string[4:6] + "-" + time_string[6:8] + "T" + time_string[9:11] + ":" + time_string[11:13] + ":" + time_string[13:]
    # Convert to numpy.datetime64 with nanosecond precision
    return np.datetime64(formatted_string, 'ns')

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

def get_butter_filter(array,fs,lowcut,highcut,order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b,a,array)





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
