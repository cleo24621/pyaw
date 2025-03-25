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
from typing import List, Optional

from scipy.interpolate import interpolate, interp1d


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


def normalize_array(arr: NDArray,
        target_min: float = 0.0,
        target_max: float = 1.0,
        clip: bool = False) -> NDArray:
    """Normalizes a  NumPy array to specified range [target_min, target_max].

    Args:
        arr: Input array to be normalized
        target_min: Minimum value of target range (default=0.0)
        target_max: Maximum value of target range (default=1.0)
        clip: Whether to clip values to target range (default=False)
    Returns:
        Normalized array with values in [target_min, target_max]. For example:

        normalize_array(np.array([1, 2, 3]), 0, 1)
        output: [0.  0.5 1. ]
        normalize_array(np.array([10, 20, 30]), 0, 100)
        output: [  0.  50. 100.]
        normalize_array(np.array([1,2,3]),-1,1)
        output: [-1.  0.  1.]
    """
    if target_max <= target_min:
        raise ValueError("target_max must be greater than target_min")

    arr_min = np.min(arr)
    arr_max = np.max(arr)

    # Handle constant array case
    if arr_max == arr_min:
        return np.full_like(arr, (target_min + target_max) / 2)

    # Normalize to [0, 1] first
    normalized = (arr - arr_min) / (arr_max - arr_min)

    # Scale to target range
    scaled = normalized * (target_max - target_min) + target_min

    if clip:
        scaled = np.clip(scaled, target_min, target_max)

    return scaled


def interpolate_missing(
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        kind: str = 'linear'
) -> np.ndarray:
    """
    Interpolate missing values (NaNs) in 1D arrays, supporting both timestamp-based
    and index-based interpolation.

    Parameters:
    -----------
    data : np.ndarray
        1D array with NaNs representing missing values
    timestamps : Optional[np.ndarray[datetime64]], default=None
        1D array of timestamps for temporal interpolation. If None, uses array indices.
    kind : str, default='linear'
        Interpolation method (see scipy.interpolate.interp1d for options)

    Returns:
    --------
    np.ndarray
        Interpolated array with NaNs replaced

    Raises:
    -------
    ValueError:
        - If data is not 1D
        - If timestamps shape doesn't match data
        - If fewer than 2 non-NaN values

    Examples:
    ---------
    # Index-based interpolation
    >>> data = np.array([1, np.nan, 3])
    >>> interpolate_missing(data)
    array([1., 2., 3.])

    # Temporal interpolation
    >>> timestamps = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[D]')
    >>> values = np.array([10, np.nan, 30])
    >>> interpolate_missing(values, timestamps)
    array([10., 20., 30.])
    """
    # Validate input array
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    # Create working copy
    data_interp = data.copy()
    valid_mask = ~np.isnan(data_interp)
    valid_count = np.count_nonzero(valid_mask)

    # Check sufficient data points
    if valid_count < 2:
        raise ValueError(f"Require ≥2 non-NaN values, got {valid_count}")

    # Create x-values based on input type
    if timestamps is not None:
        if timestamps.shape != data.shape:
            raise ValueError("Timestamps must match data shape")
        x = timestamps.astype(np.int64)  # Convert to numeric
    else:
        x = np.arange(data.size)  # Use indices

    # Setup interpolation function
    interp_func = interpolate.interp1d(
        x[valid_mask],
        data_interp[valid_mask],
        kind=kind,
        bounds_error=False,
        fill_value='extrapolate'
    )

    # Fill missing values
    missing_mask = ~valid_mask
    data_interp[missing_mask] = interp_func(x[missing_mask])

    return data_interp


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
