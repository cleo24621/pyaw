import numpy as np
from nptyping import NDArray
from numpy import ndarray



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
