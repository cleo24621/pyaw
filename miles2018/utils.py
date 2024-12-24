# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/12/24
@DESCRIPTION: 
"""
import numpy as np
import pandas as pd
from numpy import ndarray
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


def set_outliers_nan(array,std_times: float = 1.0, print_: bool = True):
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
        x[mask].astype('int'))
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
    array_series_mov_ave = array_series.rolling(window=window, center=center,min_periods=min_periods).mean()  # 'center=True' 得到的结果等于‘结果.mean()=0’
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

def get_butter_filter(array,fs,lowcut,highcut,order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b,a,array)

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

def normalize_to_01(data):
  """Normalizes a NumPy array to the range [0, 1] using min-max scaling.

  Args:
    data: A 2D NumPy array.

  Returns:
    A new 2D NumPy array with normalized values in the range [0, 1].
  """

  min_val = np.min(data)  # Find the minimum value in the entire array
  max_val = np.max(data)  # Find the maximum value in the entire array

  # Handle the case where all values are equal to avoid division by zero
  if max_val == min_val:
    return np.zeros_like(data)  # Or handle as needed, return an array of 0's

  normalized_data = (data - min_val) / (max_val - min_val)
  return normalized_data

def get_phase_diff_hist_counts(freqs:ndarray, phase_diffs:ndarray, num_bins:int):
    """
    :param freqs: 1d
    :param phase_diffs: 2d
    :param num_bins:
    :return: 1d ndarray, 2d ndarray
    """
    phase_bins = np.linspace(-180, 180, num_bins+1)
    hist_counts = np.zeros((len(freqs), num_bins))  # 2个轴分别为相位差和频率
    for i, _ in enumerate(freqs):
        hist_counts[i], _ = np.histogram(phase_diffs[i], bins=phase_bins)
        # note:: 返回的2个变量，一个是次数，一个是phase_bins，前者的长度比后者小1，2点组成一个线段
    return phase_bins, hist_counts

def get_ratio_hist_counts(freqs:ndarray, ratios: ndarray,bins:ndarray):
    """
    :param freqs: 1d
    :param ratios: 2d
    :param bins: 1d
    :return: 2d ndarray
    """
    hist_counts = np.zeros((len(freqs), len(bins) - 1))
    for i, _ in enumerate(freqs):
        hist_counts[i], _ = np.histogram(ratios[i], bins=bins)
    return hist_counts