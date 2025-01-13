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
from numpy import ndarray
from scipy.interpolate import interpolate, interp1d
from scipy.signal import butter, filtfilt, welch


def convert_tstr2dt64(time_string: str) -> np.datetime64:
    """
    convert "20160311T064700" type string to np.datetime64[ns] type
    :param time_string: "20160311T064700" type
    :return: np.datetime64[ns]
    """
    # Insert delimiters to make it ISO 8601 compliant
    formatted_string = time_string[:4] + "-" + time_string[4:6] + "-" + time_string[6:8] + "T" + time_string[
                                                                                                 9:11] + ":" + time_string[
                                                                                                               11:13] + ":" + time_string[
                                                                                                                              13:]
    # Convert to numpy.datetime64 with nanosecond precision
    return np.datetime64(formatted_string, 'ns')


def get_3arrs(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    get 3 ndarrays from 1 ndarray. the former format is np.array([[a1,b1,c1],[a2,b2,c2],...]),
    and the latter formats are np.array([a1,a2,...]), np.array([b1,b2,...]), np.array([c1,c2,...])
    :param array: the former ndarray. such as B_NEC column of the df_b get from MAGx_HR_1B file.
    :return: the latter ndarray
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


def get_rotmat_nec2sc_sc2nec(vsat_n: np.ndarray, vsat_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    :param vsat_n: velocity of satellite in the north direction
    :param vsat_e: velocity of satellite in the east direction
    :return: rotation_matrix_2d_nec2sc, rotation_matrix_2d_sc2nec
    """
    theta = np.arctan(vsat_e / vsat_n)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # Stack components to construct the rotation matrices
    rotation_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    # Transpose axes to create a (3, 2, 2) array
    rotation_matrix_2d_nec2sc = np.transpose(rotation_matrix, (2, 0, 1))
    rotation_matrix_2d_sc2nec = rotation_matrix_2d_nec2sc.transpose(0, 2, 1)
    return rotation_matrix_2d_nec2sc, rotation_matrix_2d_sc2nec


def do_rotation(coordinates1: np.ndarray, coordinates2: np.ndarray, rotation_matrix: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
    """
    :param coordinates1: N of NEC or X of S/C
    :param coordinates2: E of NEC or Y of S/C
    :param rotation_matrix: one of the rotation matrices returned by get_rotation_matrices_nec2sc_sc2nec
    :return: the coordinates after rotated
    """
    vectors12 = np.stack((coordinates1, coordinates2), axis=1)
    vectors12_rotated = np.einsum('nij,nj->ni', rotation_matrix, vectors12)
    return vectors12_rotated[:, 0], vectors12_rotated[:, 1]


def set_bursts_nan_diff(series, threshold, print_: bool = True):
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


def set_outliers_nan_std(array: np.ndarray, std_times: float = 1.0, print_: bool = True):
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
    y_copy[mask] = interpolate.interp1d(x[~mask].astype('int'), y_copy[~mask], kind='linear')(
        x[mask].astype('int'))  # note:: 当x是时间类型时，该方法也支持
    return y_copy


def move_average(array: np.ndarray, window: int, center: bool = True, min_periods: int | None = None):
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
    array_series_mov_ave = array_series.rolling(window=window, center=center,
                                                min_periods=min_periods).mean()  # 'center=True' 得到的结果等于‘结果.mean()=0’，即经过b-b.mean()（baseline correction）
    return array_series_mov_ave.values


def wavelet_smooth(series_: pd.Series, wavelet='db4', level=6, threshold=0.2, mode='soft') -> pd.Series:
    # process nan
    print(f'The number of NaN values: {series_.isna().sum()}')
    series_ = series_.interpolate(method='linear')
    # 使用小波变换进行多尺度分解
    wavelet = wavelet  # 选择小波函数，例如 'db4' (Daubechies)
    coeffs = pywt.wavedec(series_, wavelet, level=level)  # 进行离散小波分解，设定分解层数
    # 处理高频细节系数，设置某些高频部分为零，以达到平滑效果
    threshold = threshold  # 设置阈值
    coeffs[1:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
    # 使用处理后的系数重构信号
    smoothed_signal = pywt.waverec(coeffs, wavelet)
    return smoothed_signal


class Butter:
    """
    a class to process array using butterworth filter
    非抗混叠
    """

    def __init__(self, arr: np.ndarray, fs: float):
        """
        :param arr: the array to process
        :param fs:
        """
        self.arr = arr
        self.fs = fs
        self.nyquist = 0.5 * fs

    def apply_bandpass_filter(self, lowcut: float, highcut: float, order: int) -> np.ndarray:
        """
        :param lowcut:
        bandpass filter
        :param highcut:
        :param order:
        :return: the array after bandpass filter
        """
        low = lowcut / self.nyquist
        high = highcut / self.nyquist
        b, a = butter(order, [low, high], btype="bandpass")
        return filtfilt(b, a, self.arr)  # todo:: inplace or not?

    def apply_lowpass_filter(self, lowcut: float, order: int) -> np.ndarray:
        """
        :param lowcut:
        :param order:
        :return: the array after lowpass filter
        """
        low = lowcut / self.nyquist
        b, a = butter(order, low, btype="lowpass")
        return filtfilt(b, a, self.arr)

    def apply_highpass_filter(self, highcut: float, order: int) -> np.ndarray:
        """
        :param highcut:
        :param order:
        :return: the array after highpass filter
        """
        high = highcut / self.nyquist
        b, a = butter(order, high, btype="highpass")
        return filtfilt(b, a, self.arr)


def threshold_and_set(data, threshold, set_value):
    """Sets elements in a 2D NumPy array exceeding a threshold to a specific value.
    Args:
    data: A 2D NumPy array.
    threshold: The value to compare against. Elements exceeding this will be changed.
    set_value: The new value to assign to elements exceeding the threshold.
    Returns:
    The modified 2D NumPy array (changes are made in-place).
    """
    data[data > threshold] = set_value
    return data


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
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


def get_phase_histogram2d(freqs: np.ndarray, phase_diffs: np.ndarray, num_bins: int):
    """
    :param freqs: 1d
    :param phase_diffs: 2d
    :param num_bins:
    :return: 1d ndarray, 2d ndarray
    """
    phase_bins = np.linspace(-180, 180, num_bins + 1)
    hist_counts = np.zeros((len(freqs), num_bins))  # 2个轴分别为相位差和频率
    for i, _ in enumerate(freqs):
        hist_counts[i], _ = np.histogram(phase_diffs[i],
                                         bins=phase_bins)  # note: 返回的2个变量，一个是次数，一个是phase_bins，前者的长度比后者小1，2点组成一个线段
    return phase_bins, hist_counts


def get_ratio_histogram2d(freqs: np.ndarray, ratio_bins: np.ndarray, bins: np.ndarray):
    """
    :param freqs: 1d
    :param ratio_bins: 2d. different from phase_bins that is in [-180, 180], ratio_bins is in [0, max(ratio)] or [0,percentile95(ratio)] or other reasonable value.
    :param bins: 1d
    :return: 2d ndarray
    """
    hist_counts = np.zeros((len(freqs), len(bins) - 1))
    for i, _ in enumerate(freqs):
        hist_counts[i], _ = np.histogram(ratio_bins[i], bins=bins)
    return hist_counts


def align_high2low(arr_high: np.ndarray, arr_high_index: np.ndarray, arr_low_index: np.ndarray) -> np.ndarray:
    """
    signal_high aligned to signal_low using linear interpolation.
    :param arr_low_index: index of arr_low
    :param arr_high_index: index of arr_high
    :param arr_high: index is pd.Timestamps type. with high sample rate, so long data length, like swarm vfm50 magnetic data
    :return: index is pd.Timestamps type. signal_high aligned to signal_low.
    """
    interp_func = interp1d(arr_high_index.astype('int64'), arr_high, kind='linear', fill_value="extrapolate")
    return interp_func(arr_low_index.astype('int64'))

def split_array(data, step=11):  # todo:: step
    if len(data.shape) > 2:
        raise "Cannot handle arrays of shapes with a length greater than 2"
    if len(data.shape) == 1:
        # Split the array
        result = [data[i:i + step] for i in range(0, len(data) - step, step)]
        # Add the remaining columns to the last segment
        remainder = data[step * len(result):]
        if remainder.size > 0:
            if len(result) > 0:
                # Append remaining columns to the last split
                result[-1] = np.hstack((result[-1], remainder))
            else:
                # If there's no initial split, the remainder is the only result
                result.append(remainder)
    else:
        result = [data[:, i:i + step] for i in range(0, data.shape[1] - step, step)]  # todo:: 更少的组成一部分；时间对应
        remainder = data[:, step * len(result):]
        if remainder.size > 0:
            if len(result) > 0:
                result[-1] = np.hstack((result[-1], remainder))
            else:
                result.append(remainder)
    return result


def get_coherences(Zxx1,Zxx2,cpsd_12,step=11) -> np.ndarray:
    """
    the times and freqs corresponding to Zxx1 and Zxx2 should be the same.
    Zxx1 can be Zxx_e or Zxx_b, the order doesn't affect the result.
    :param Zxx1:
    :param Zxx2:
    :param cpsd_12:
    :return:
    """
    cpsd12_split = split_array(cpsd_12,step=step)  # ls
    denominator1ls = split_array(np.abs(Zxx1 ** 2),step=step)
    denominator2ls = split_array(np.abs(Zxx2 ** 2),step=step)

    coherences_f = []
    for i in range(len(cpsd12_split)):
        nominator = cpsd12_split[i].mean(axis=1)  # along axis1, not all elements.
        denominator = np.sqrt(denominator1ls[i].mean(axis=1)) * np.sqrt(denominator2ls[i].mean(axis=1))
        coherences_f.append(nominator / denominator)

    coherences = []
    for c_f in coherences_f:
        coherences.append(np.abs(c_f).mean())

    return np.array(coherences)


def get_middle_element(lst):
    """
    :param lst: list or np.ndarray
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


def get_phase_histogram_f_ave(phase_bins:ndarray,phase_histogram2d:ndarray):
    """
    :param phase_bins: shape should be (n,)
    :param phase_histogram2d: shape should be (m,n-1)
    :return:
    """
    phases = []
    for i in range(len(phase_bins)-1):
        phases.append((phase_bins[i+1]+phase_bins[i])/2)

    phases_ave = []
    for histogram in phase_histogram2d:
        if np.sum(histogram) == 0:
            phases_ave.append(0)
        else:
            phases_ave.append(np.sum(histogram * phases)/np.sum(histogram))
    return np.array(phases_ave)


class FFT:
    """
    a class to get fft of a signal of sampling frequency fs
    """

    def __init__(self, array: np.ndarray, fs: float):
        self.array = array
        self.fs = fs

    def get_fft(self):
        """
        because the returned frequencies are determined by the length of the signal and the sampling rate,
        the previous frequencies returned by the high sampling rate signal are exactly the same as the frequencies of
        the low sampling rate signal.
        :return: frequencies, amplitude_spectrum, phase
        """
        n = len(self.array)
        fft_values = np.fft.fft(self.array)  # fft
        amplitude_spectrum = np.abs(fft_values)  # magnitude of fft
        phase = np.angle(fft_values)  # phase of fft
        f_values = np.fft.fftfreq(n, d=1 / self.fs)  # frequencies
        # only positive frequencies
        return f_values[:n // 2], amplitude_spectrum[:n // 2], phase[:n // 2]

    def plot_fft(self, figsize=(10, 6), title='fft'):
        freqs, amps, _ = self.get_fft()
        fig = plt.figure(figsize=figsize)
        plt.plot(freqs, amps, color='red')
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude Spectra')
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.title(f'{title}: (fs={self.fs})')
        plt.show()
        return fig


class PSD:
    """
    a class to get psd of a signal of sampling frequency fs using welch method.
    default 'density'
    返回的频率由更多参数决定，也许可以通过更改参数来得到相同的频率 todo
    """

    def __init__(self, array, fs, nperseg: Optional[int] = None, noverlap: Optional[int] = None, window='hann',
                 scaling='density'):
        self.array = array
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window
        self.scaling = scaling

    def get_psd(self):
        freqs, Pxx = welch(self.array, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, window=self.window,
                           scaling=self.scaling)
        return freqs, Pxx


class CWT:
    def __init__(self, arr1, arr2, scales=np.arange(1, 128), wavelet='cmor1.5-1.0', fs=16):
        """
        signal1 and signal2 are aligned in time.
        :param arr1:
        :param arr2:
        :param scales:
        :param wavelet:
        :param fs:
        """
        self.arr1 = arr1
        self.arr2 = arr2
        self.scales = scales
        self.wavelet = wavelet
        self.sampling_period = 1 / fs

    def get_cross_spectral(self):
        coefficients_f, freqs = pywt.cwt(self.arr1, self.scales, self.wavelet,
                                         sampling_period=self.sampling_period)  # CWT for signal1
        coefficients_g, _ = pywt.cwt(self.arr2, self.scales, self.wavelet,
                                     sampling_period=self.sampling_period)  # CWT for signal2
        cross_spectrum = coefficients_f * np.conj(coefficients_g)
        cross_spectrum_modulus = np.abs(cross_spectrum)
        cross_phase = np.degrees(np.angle(cross_spectrum))
        return cross_spectrum_modulus, cross_phase, freqs

    # def plot_module(self, figsize=(10, 4)):  #     cross_spectrum_modulus, _, freqs = self.get_cross_spectral()  #     plt.figure(figsize=figsize)  #     plt.imshow(cross_spectrum_modulus, extent=[self.arr1.index[0], self.arr1.index[-1], freqs[-1], freqs[0]],  #                aspect='auto', cmap='jet')  #     plt.colorbar(label='Module')  #     plt.xlabel('UT Time [s]')  #     plt.ylabel('Frequency [Hz]')  #     plt.title('One dimensional Continuous Wavelet Transform Modulus')  #     plt.show()  #     return figure  #  # def plot_phase(self, figsize=(10, 4)):  #     _, cross_phase, freqs = self.get_cross_spectral()  #     plt.figure(figsize=figsize)  #     plt.imshow(cross_phase, extent=[self.arr1.index[0], self.arr1.index[-1], freqs[-1], freqs[0]],  #                aspect='auto', cmap='jet')  #     plt.colorbar(label='Phase [degree]')  #     plt.xlabel('UT Time [s]')  #     plt.ylabel('Frequency [Hz]')  #     plt.title('Cross-Phase')  #     plt.show()  #     return figure  #  # def get_phase_hist_counts(self, num_bins=50):  #     """  #     :param num_bins: Number of bin edges for phase histogram  #     :return:  #     """  #     cross_spectrum_modulus, cross_phase, freqs = self.get_cross_spectral()  #     hist_counts = np.zeros((len(freqs), num_bins - 1))  # num_bins-1 bins  #     phase_bins = np.linspace(-180, 180, num_bins)  #     # Loop over each frequency and calculate histogram for phases  #     for i, freq in enumerate(freqs):  #         # # 仅考虑模值大于阈值的相位  #         # valid_phases = cross_phase[i][cross_spectrum_modulus[i] > 100]  #         # if len(valid_phases) > 0:  #         #     hist_counts[i], _ = np.histogram(valid_phases, bins=phase_bins)  #         hist_counts[i], _ = np.histogram(cross_phase[i], bins=phase_bins)  #     # Normalize counts for better visualization  #     hist_counts = hist_counts / np.max(hist_counts, axis=1, keepdims=True)  #     return hist_counts, freqs  #  # def plot_phase_hist_counts(self, figsize=(10, 4)):  #     hist_counts, freqs = self.get_phase_hist_counts()  #     plt.figure(figsize=figsize)  #     plt.imshow(hist_counts, extent=[-180, 180, freqs[-1], freqs[0]], aspect='auto', cmap='jet')  #     plt.colorbar(label='Normalized Counts')  #     plt.xlabel('Phase [degree]')  #     plt.ylabel('Frequency [Hz]')  #     plt.title('Phase Histogram')  #     plt.show()  #     return figure
