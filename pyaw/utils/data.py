# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/21/2025 20:55
@Project     : pyaw
@Description : process data.

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
import numpy as np
import pandas as pd
import pywt
from nptyping import NDArray
from typing import List

from numpy import ndarray
from scipy.interpolate import interpolate, interp1d
from scipy.signal import butter, filtfilt


class DataStructure:
    @staticmethod
    def get_3arrs(
        array: NDArray,
    ):
        """
        get 3 NDArray from 1 NDArray.
        the former format is np.array([[a1,b1,c1],[a2,b2,c2],...]),
        and the latter formats are np.array([a1,a2,...]), np.array([b1,b2,...]), np.array([c1,c2,...])

        Args:
            array: the former NDArray. such as B_NEC column of the df_b get from MAGx_HR_1B file.

        Returns:
            tuple[NDArray[Float64], NDArray[Float64], NDArray[Float64]]: the latter NDArray
        """
        array1 = []
        array2 = []
        array3 = []
        for array_ in array:
            array1.append(array_[0])
            array2.append(array_[1])
            array3.append(array_[2])
        array1 = np.array(array1)
        array2 = np.array(array2)
        array3 = np.array(array3)
        return array1, array2, array3

    @staticmethod
    def split_array(array: NDArray, step: int=11) -> List[NDArray]:
        """
        按照step分割array
        Args:
            array:
            step:

        Returns:
            List[NDArray]: 元素是按顺序分割的arrays

        """
        if len(array.shape) > 2:
            raise "Cannot handle arrays of shapes with a length greater than 2"
        if len(array.shape) == 1:
            # Split the array
            result = [array[i: i + step] for i in range(0, len(array) - step, step)]
            # Add the remaining columns to the last segment
            remainder = array[step * len(result):]
            if remainder.size > 0:
                if len(result) > 0:
                    # Append remaining columns to the last split
                    result[-1] = np.hstack((result[-1], remainder))
                else:
                    # If there's no initial split, the remainder is the only result
                    result.append(remainder)
        else:
            result = [
                array[:, i: i + step] for i in range(0, array.shape[1] - step, step)
            ]  # todo:: 更少的组成一部分；时间对应
            remainder = array[:, step * len(result):]
            if remainder.size > 0:
                if len(result) > 0:
                    result[-1] = np.hstack((result[-1], remainder))
                else:
                    result.append(remainder)
        return result


class OutlierData:
    @staticmethod
    def set_bursts_nan_diff(series, threshold, print_: bool = True):  # todo: 备用（磁场）；转化为array使用
        """
        set the value of bursts to nan
        other method: series_copy[(np.abs(series_copy) / np.abs(series_copy).mean()) > 10] = np.nan
        :param series:
        :param threshold:
        :param print_:
        :return:
        """
        series_copy = series.copy()
        diff = series_copy.diff()
        # 设置一个突变检测阈值
        bursts = diff[diff.abs() > threshold]
        if print_:
            print(len(bursts))
            print(bursts)
        series_copy.loc[diff.abs() > threshold] = np.nan
        return series_copy  # return series_copy  # series_scores = (series - series.mean()) / series.std()  # # 设置阈值，通常 Z 分数大于 3 或小于 -3 的点可以认为是异常点，超过的设置为nan  # threshold = 2  # print(series[np.abs(series_scores) > threshold])  # series[np.abs(series_scores) > threshold] = np.nan  # return series

    @staticmethod
    def set_outliers_nan_std(
        array: NDArray, std_times: float = 1.0, print_: bool = True
    ) -> NDArray:
        """
        note: use copy
        Args:
            array: the array to process
            std_times: standard deviation times
            print_: print the outliers or not

        Returns:
            the array with outliers set to nan
        """
        array_copy = array.copy()
        threshold = std_times * np.std(array_copy)
        bursts = np.abs(array_copy - np.mean(array_copy)) > threshold
        if print_:
            print(len(array_copy[bursts]))
            print(array_copy[bursts])
        array_copy[bursts] = np.nan
        return array_copy


class Butter:
    """
    a class to process array using butterworth filter
    非抗混叠
    """

    def __init__(self, arr: NDArray, fs: float):
        """

        Args:
            arr: the array to be processed
            fs:
        """
        self.arr = arr
        self.fs = fs
        self.nyquist = 0.5 * fs

    def apply_bandpass_filter(
        self, lowcut: float, highcut: float, order: int
    ) -> NDArray:
        """

        Args:
            lowcut: lower cut-off frequency
            highcut: upper cut-off frequency
            order:

        Returns:
            array after filter
        """
        low = lowcut / self.nyquist
        high = highcut / self.nyquist
        b, a = butter(order, [low, high], btype="bandpass")
        return filtfilt(b, a, self.arr)

    def apply_lowpass_filter(self, lowcut: float, order: int) -> NDArray:
        """

        Args:
            lowcut: lower cut-off frequency
            order:

        Returns:
            array after filter
        """
        low = lowcut / self.nyquist
        b, a = butter(order, low, btype="lowpass")
        return filtfilt(b, a, self.arr)

    def apply_highpass_filter(self, highcut: float, order: int) -> NDArray:
        """

        Args:
            highcut: upper cut-off frequency
            order:

        Returns:
            array after filter
        """
        high = highcut / self.nyquist
        b, a = butter(order, high, btype="highpass")
        return filtfilt(b, a, self.arr)


class Histogram2d:
    @staticmethod
    def get_phase_histogram2d(
            frequencies: NDArray, phase_diffs: NDArray, num_bins: int
    ):
        """
        :param frequencies: 1d
        :param phase_diffs: 2d
        :param num_bins:
        :return: 1d ndarray, 2d ndarray
        """
        phase_bins = np.linspace(-180, 180, num_bins + 1)
        hist_counts = np.zeros((len(frequencies), num_bins))  # 2个轴分别为相位差和频率
        for i, _ in enumerate(frequencies):
            hist_counts[i], _ = np.histogram(
                phase_diffs[i], bins=phase_bins
            )  # note: 返回的2个变量，一个是次数，一个是phase_bins，前者的长度比后者小1，2点组成一个线段
        return phase_bins, hist_counts

    @staticmethod
    def get_ratio_histogram2d(frequencies: NDArray, ratio_bins: NDArray, bins: NDArray):
        """
        :param frequencies: 1d
        :param ratio_bins: 2d. different from phase_bins that is in [-180, 180], ratio_bins is in [0, max(ratio)] or [0,percentile95(ratio)] or other reasonable value.
        :param bins: 1d
        :return: 2d ndarray
        """
        hist_counts = np.zeros((len(frequencies), len(bins) - 1))
        for i, _ in enumerate(frequencies):
            hist_counts[i], _ = np.histogram(ratio_bins[i], bins=bins)
        return hist_counts

    @staticmethod
    def get_phase_histogram_f_ave(
            phase_bins: ndarray, phase_histogram2d: ndarray
    ):
        """
        :param phase_bins: shape should be (n,)
        :param phase_histogram2d: shape should be (m,n-1)
        :return:
        """
        phases = []
        for i in range(len(phase_bins) - 1):
            phases.append((phase_bins[i + 1] + phase_bins[i]) / 2)

        phases_ave = []
        for histogram in phase_histogram2d:
            if np.sum(histogram) == 0:
                phases_ave.append(0)
            else:
                phases_ave.append(np.sum(histogram * phases) / np.sum(histogram))
        return np.array(phases_ave)


def normalize_to_01(arr: NDArray) -> NDArray:
    """
    Normalizes a NumPy array to the range [0, 1] using min-max scaling.
    :param arr: NumPy array.
    :return: NumPy array with normalized values in the range [0, 1].
    """
    min_val = np.min(arr)  # Find the minimum value in the entire array
    max_val = np.max(arr)  # Find the maximum value in the entire array
    # Handle the case where all values are equal to avoid division by zero
    if max_val == min_val:
        return np.zeros_like(arr)  # Or handle as needed, return an array of 0's
    normalized_data = (arr - min_val) / (max_val - min_val)
    return normalized_data


def get_array_interpolated(x, y):
    """
    interpolate the missing values of the array using 1d interpolation
    :param x: ndarray consisting of np.datetime64.
    :param y: the array to process
    :return:the array with missing values interpolated
    """
    y_copy = y
    # Mask for missing values
    mask = np.isnan(y_copy)
    # Interpolate
    y_copy[mask] = interpolate.interp1d(
        x[~mask].astype("int"), y_copy[~mask], kind="linear"
    )(
        x[mask].astype("int")
    )  # note:: 当x是时间类型时，该方法也支持
    return y_copy


def move_average(
    array: NDArray, window: int, center: bool = True, min_periods: int | None = None
):
    """
    在用于电场和磁场信号时，窗口长度一般对应于事件的长度，如20s。
    :param min_periods: the 'min_periods' parameter of the series.rolling() function
    :param center: the 'center' parameter of the series.rolling() function
    :param window: the window of the moving average. equal to fs * (the seconds of the window), and the windows must be an integer.
    :param array: the array to process
    :return: the array with moving average
    """
    assert type(window) == int, "window must be an integer"
    array_series = pd.Series(array)
    array_series_mov_ave = array_series.rolling(
        window=window, center=center, min_periods=min_periods
    ).mean()  # 'center=True' 得到的结果等于‘结果.mean()=0’，即经过b-b.mean()（baseline correction）
    return array_series_mov_ave.values


def wavelet_smooth(
    series_: pd.Series, wavelet="db4", level=6, threshold=0.2, mode="soft"
) -> pd.Series:
    # process nan
    print(f"The number of NaN values: {series_.isna().sum()}")
    series_ = series_.interpolate(method="linear")
    # 使用小波变换进行多尺度分解
    wavelet = wavelet  # 选择小波函数，例如 'db4' (Daubechies)
    coefficients = pywt.wavedec(
        series_, wavelet, level=level
    )  # 进行离散小波分解，设定分解层数
    # 处理高频细节系数，设置某些高频部分为零，以达到平滑效果
    threshold = threshold  # 设置阈值
    coefficients[1:] = [pywt.threshold(c, threshold, mode=mode) for c in coefficients[1:]]
    # 使用处理后的系数重构信号
    smoothed_signal = pywt.waverec(coefficients, wavelet)
    return smoothed_signal


def align_high2low(
    arr_high: NDArray, arr_high_index: NDArray, arr_low_index: NDArray
) -> NDArray:
    """
    signal_high aligned to signal_low using linear interpolation.
    :param arr_low_index: index of arr_low
    :param arr_high_index: index of arr_high
    :param arr_high: index is pd.Timestamps type. with high sample rate, so long data length, like swarm vfm50 magnetic data
    :return: index is pd.Timestamps type. signal_high aligned to signal_low.
    """
    interp_func = interp1d(
        arr_high_index.astype("int64"),
        arr_high,
        kind="linear",
        fill_value="extrapolate",
    )
    return interp_func(arr_low_index.astype("int64"))


def get_middle_element(lst):
    """
    :param lst: list or NDArray
    :return:
    """
    n = len(lst)
    if n == 0:
        return None  # Handle the case of an empty list
    mid = n // 2
    if n % 2 == 0:  # Even number of elements
        return lst[mid - 1]  # Return the former one of the two middle elements
    else:  # Odd number of elements
        return lst[mid]  # Return the single middle element


def main():
    pass


if __name__ == "__main__":
    main()
