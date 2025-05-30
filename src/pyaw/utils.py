import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import pywt
from numpy import ndarray
from numpy.typing import NDArray
from scipy import signal
from scipy.interpolate import interpolate, interp1d
from scipy.signal import buttord, butter, welch


def split_array(array: NDArray, step: int = 11) -> List[NDArray]:
    """Split a 1D or 2D NumPy array into segments of a specified step size along the last axis."""
    if len(array.shape) > 2:
        raise "Cannot handle arrays of shapes with a length greater than 2"
    if len(array.shape) == 1:
        # Split the array
        result = [array[i : i + step] for i in range(0, len(array) - step, step)]
        # Add the remaining columns to the last segment
        remainder = array[step * len(result) :]
        if remainder.size > 0:
            if len(result) > 0:
                # Append remaining columns to the last split
                result[-1] = np.hstack((result[-1], remainder))
            else:
                # If there's no initial split, the remainder is the only result
                result.append(remainder)
    else:
        # Split the 2D array. I haven't encountered this situation yet.
        result = [array[:, i : i + step] for i in range(0, array.shape[1] - step, step)]
        remainder = array[:, step * len(result) :]
        if remainder.size > 0:
            if len(result) > 0:
                result[-1] = np.hstack((result[-1], remainder))
            else:
                result.append(remainder)
    return result


def get_3arrays(
    array: NDArray,
):
    """
    Get 3 NDArray from 1 NDArray.
    The former format is np.array([[a1,b1,c1],[a2,b2,c2],...]),
    and the latter formats are np.array([a1,a2,...]), np.array([b1,b2,...]), np.array([c1,c2,...]).

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


def nan_outliers_by_std_dev(
    _array: NDArray, std_times: float = 1.0, print_: bool = True
) -> NDArray:
    """
    Improved outlier detection with NaN handling and zero-std protection

    Args:
        _array: Input array to process (NaN values will be ignored in calculations)
        std_times: Threshold-- multiplier for standard deviation.
        print_: Whether to print outliers information

    Returns:
        Array with outliers set to NaN (preserves original NaNs)
    """
    array_copy = _array.copy()
    valid_values = array_copy[~np.isnan(array_copy)]

    # Handle empty array case
    if len(valid_values) == 0:
        if print_:
            print("Warning: Input array contains only NaN values")
        return array_copy

    mean_val = np.nanmean(array_copy)  # the type of 'mean_val' is float64
    std_val = np.nanstd(array_copy)  # the type of 'std_val' is float64
    mean_val = np.float64(mean_val)
    std_val = np.float64(std_val)

    # Handle zero standard deviation case
    if np.isclose(std_val, 0):
        if print_:
            print(
                f"No outliers detected - standard deviation is zero (mean: {mean_val:.2f})"
            )
        return array_copy

    threshold = std_times * std_val
    deviations = np.abs(array_copy - mean_val)
    bursts = deviations > threshold

    if print_:
        outliers = array_copy[bursts & ~np.isnan(array_copy)]
        print(f"Outliers detected: {len(outliers)}")
        print(f"Outlier values: {outliers}")

    array_copy[bursts] = np.nan
    return array_copy


def normalize_array(
    arr: NDArray, target_min: float = 0.0, target_max: float = 1.0, clip: bool = False
) -> NDArray:
    """Normalizes a  NumPy array to specified range [target_min, target_max].

    Args:
        arr: Input array to be normalized
        target_min: Minimum value of target range (default=0.0)
        target_max: Maximum value of target range (default=1.0)
        clip: Whether to clip values to target range (default=False)

    Examples:
        >>> normalize_array(np.array([1, 2, 3]), 0, 1)
        [0.  0.5 1.]
        >>> normalize_array(np.array([10, 20, 30]), 0, 100)
        [  0.  50. 100.]
        >>> normalize_array(np.array([1,2,3]),-1,1)
        [-1.  0.  1.]
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
    data: np.ndarray, timestamps: Optional[np.ndarray] = None, kind: str = "linear"
) -> np.ndarray:
    """Interpolate NaNs in a 1D arrays, supporting both consecutive timestamp-based and number-based interpolation.

    Args:
        data: 1D array with NaNs representing missing values.
        timestamps: 1D array of timestamps for temporal interpolation. If None, uses array indices.
        kind: Interpolation method (see scipy.interpolate.interp1d for options).

    Returns:
        Interpolated array with NaNs replaced.

    Raises:
        ValueError:
            - If data is not 1D
            - If timestamps shape doesn't match data
            - If fewer than 2 non-NaN values

    Examples:
        >>> # number-based interpolation
        >>> _array = np.array([1, np.nan, 3])
        >>> interpolate_missing(_array)
        array([1., 2., 3.])
        >>> # Temporal interpolation
        >>> _timestamps = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[D]')
        >>> values = np.array([10, np.nan, 30])
        >>> interpolate_missing(values, _timestamps)
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
        fill_value="extrapolate",
    )

    # Fill missing values
    missing_mask = ~valid_mask
    data_interp[missing_mask] = interp_func(x[missing_mask])

    return data_interp


def move_average(
    _array: NDArray, window: int, center: bool = True, min_periods: int | None = 1
) -> NDArray:
    """
    Args:
        _array: the original array.

    Returns:
        The array with moving average.
    """
    assert type(window) == int, "window must be an integer"
    array_series = pd.Series(_array)
    array_series_mov_ave = array_series.rolling(
        window=window, center=center, min_periods=min_periods
    ).mean()  # 'center=True' 得到的结果等于‘结果.mean()=0’，即经过b-b.mean()（baseline correction）
    return array_series_mov_ave.values


def set_bursts_nan_diff(series, threshold, print_: bool = True):
    """Set the value of bursts to NaN."""
    series_copy = series.copy()
    diff = series_copy.diff()
    # 设置一个突变检测阈值
    bursts = diff[diff.abs() > threshold]
    if print_:
        print(len(bursts))
        print(bursts)
    series_copy.loc[diff.abs() > threshold] = np.nan
    return series_copy


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
    coefficients[1:] = [
        pywt.threshold(c, threshold, mode=mode) for c in coefficients[1:]
    ]

    # 使用处理后的系数重构信号
    smoothed_signal = pywt.waverec(coefficients, wavelet)

    return smoothed_signal


def time_align_high2low(
    arr_high: NDArray, arr_high_index: NDArray, arr_low_index: NDArray
) -> NDArray:
    """Signal_high aligned to signal_low using linear interpolation.

    Args:
        arr_high: 1D array of high sample rate signal values.
        arr_high_index: 1D array of timestamps corresponding to arr_high (the timestamps should be consecutive)
        (the type of timestamp should be pd.Timestamp).
        arr_low_index: 1D array of timestamps corresponding to arr_low (the timestamps should be consecutive)
        (the type of timestamp should be pd.Timestamp).

    Returns:
        1D array of signal_high values aligned to signal_low timestamps.
    """
    interp_func = interp1d(
        arr_high_index.astype("int64"),
        arr_high,
        kind="linear",
        fill_value="extrapolate",
    )

    return interp_func(arr_low_index.astype("int64"))


def get_middle_element(_list):
    """Get the middle element of a list."""
    n = len(_list)

    if n == 0:
        return None  # Handle the case of an empty list

    mid = n // 2
    if n % 2 == 0:  # Even number of elements
        return _list[mid - 1]  # Return the former one of the two middle elements
    else:  # Odd number of elements
        return _list[mid]  # Return the single middle element


def customize_butter(fs, f_t, f_z, _type="lowpass"):
    """customize 'scipy.signal.butter()'.

    Args:
        fs: 采样率 (Hz)
        f_t: 通带截止频率 (Hz)。低，例如100
        f_z: 阻带截止频率 (Hz)。高，例如200
        _type: ‘lowpass’, ‘highpass’, ‘bandpass’

    Returns:
        The customized butterworth filter coefficients b, a.
    """
    # 归一化频率
    wp = f_t / (fs / 2)
    ws = f_z / (fs / 2)

    # 计算阶数（默认 gpass=3dB, gstop=40dB）
    order, wn = buttord(wp, ws, 3, 40)
    return butter(order, wn, _type)


def get_phase_histogram2d(frequencies: NDArray, phase_diffs: NDArray, num_bins: int):
    """Get the histogram of phase differences for each frequency.

    Args:
        frequencies: 1D array of frequencies.
        phase_diffs: 2D array of phase differences, where one row corresponds to the input frequencies,
        the another corresponds to the times?
        num_bins: number of bins for the histogram?
    """
    phase_bins = np.linspace(-180, 180, num_bins + 1)
    hist_counts = np.zeros((len(frequencies), num_bins))  # 2个轴分别为相位差和频率
    for i, _ in enumerate(frequencies):
        hist_counts[i], _ = np.histogram(
            phase_diffs[i], bins=phase_bins
        )  # note: 返回的2个变量，一个是次数，一个是phase_bins，前者的长度比后者小1，2点组成一个线段
    return phase_bins, hist_counts


def get_ratio_histogram2d(frequencies: NDArray, ratio_bins: NDArray, bins: NDArray):
    """Refer to 'get_phase_histogram2d'."""
    hist_counts = np.zeros((len(frequencies), len(bins) - 1))
    for i, _ in enumerate(frequencies):
        hist_counts[i], _ = np.histogram(ratio_bins[i], bins=bins)
    return hist_counts


def get_phase_histogram_f_ave(phase_bins: ndarray, phase_histogram2d: ndarray):
    """Refer to 'get_phase_histogram2d'."""
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


class CustomizedFFT:
    """Get the fft of a signal."""

    def __init__(self, array: NDArray, fs: float):
        """

        Args:
            array: 1D array of the signal values.
            fs: the sampling rate of the signal (Hz).
        """
        self.array = array
        self.fs = fs

    def get_fft(self) -> tuple[NDArray, NDArray, NDArray]:
        """Only return the data of positive frequencies.

        Note:
            Because the returned frequencies are determined by the length of the signal and the sampling rate,
            the previous frequencies returned by the high sampling rate signal are exactly the same as
            the frequencies of the low sampling rate signal.
        """
        n = len(self.array)
        fft_values = np.fft.fft(self.array)  # fft
        magnitudes = np.abs(fft_values)  # magnitude of fft
        phases = np.angle(fft_values)  # phase of fft
        frequencies = np.fft.fftfreq(n, d=1 / self.fs)  # frequencies

        # only positive frequencies
        return frequencies[: n // 2], magnitudes[: n // 2], phases[: n // 2]


class CustomizedWelch:
    """Get the customized 'scipy.signal.welch()' of a signal."""

    def __init__(
        self,
        array: NDArray,
        fs: float,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        window: Optional[str] = "hann",
        scaling: Optional[str] = "density",
        nfft: Optional[int] = None,
    ):
        """Initialize the CustomizedWelch class.

        Args:
            array: 1D array of the signal values.
            fs: the sampling rate of the signal (Hz).
        """
        self.array = array
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window
        self.scaling = scaling
        self.nfft = nfft

    def get_psd(self) -> tuple[NDArray, NDArray]:
        """

        Returns:
            The 1D array of sample frequencies and the 1D array of power spectral density or power spectrum.
        """
        frequencies, pxx = welch(
            self.array,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            window=self.window,
            scaling=self.scaling,
            nfft=self.nfft,
        )
        return frequencies, pxx


class CustomizedCWT:
    """Customized a Continuous Wavelet Transform."""

    def __init__(
        self,
        arr1: NDArray,
        arr2: NDArray,
        scales: NDArray = np.arange(1, 128),
        wavelet: str = "cmor1.5-1.0",
        fs: float = 16.0,
    ) -> None:
        """

        Args:
            arr1: The 1D array of the first signal values.
            arr2: The 1D array of the second signal values.
            fs: the sampling rate of the signals (Hz).

        Notes:
            Signal1 and signal2 are aligned in time.
        """
        self.arr1 = arr1
        self.arr2 = arr2
        self.scales = scales
        self.wavelet = wavelet
        self.sampling_period = 1 / fs

    def get_cross_spectral(self):
        """Get the cross spectral density of the two signals.

        Returns:
            2D array of the modulus of the cross spectral density.
            2D array of the phase of the cross spectral density.
            1D array of frequencies corresponding to the scales.
        """
        coefficients_f, frequencies = pywt.cwt(
            self.arr1, self.scales, self.wavelet, sampling_period=self.sampling_period
        )  # CWT for signal1
        coefficients_g, _ = pywt.cwt(
            self.arr2, self.scales, self.wavelet, sampling_period=self.sampling_period
        )  # CWT for signal2
        cross_spectrum = coefficients_f * np.conj(coefficients_g)  # 2 维
        cross_spectrum_modulus = np.abs(cross_spectrum)
        cross_phase = np.degrees(np.angle(cross_spectrum))
        return cross_spectrum_modulus, cross_phase, frequencies


def get_coherence(
    zxx1: NDArray, zxx2: NDArray, cpsd_12: NDArray, step: int = 11
) -> NDArray:
    """Get the average coherence magnitude over frequency for time segments.

    Args:
        zxx1: Spectrogram or stft of x (refer to scipy doc).
        zxx2: ~
        cpsd_12: the cross power spectral density of signal1 (array1) and signal2 (array2).
        step: Zxx1, Zxx2, cpsd_12 的拆分间隔

    Notes:
        The times and frequencies corresponding to Zxx1 and Zxx2 should be the same.
        Zxx1 can be Zxx_e or Zxx_b, the order doesn't affect the result.
    """
    cpsd12_split = split_array(cpsd_12, step=step)  # ls
    denominator1ls = split_array(np.abs(zxx1**2), step=step)
    denominator2ls = split_array(np.abs(zxx2**2), step=step)

    coherence_f = []
    for i in range(len(cpsd12_split)):
        nominator = cpsd12_split[i].mean(axis=1)  # along axis1, not all elements.
        denominator = np.sqrt(denominator1ls[i].mean(axis=1)) * np.sqrt(
            denominator2ls[i].mean(axis=1)
        )
        # Perform division, avoiding division by zero by setting the result to a specified value (e.g., 0)
        result = np.divide(
            nominator,
            denominator,
            out=np.full_like(nominator, fill_value=0, dtype=complex),
            where=denominator != 0,
        )
        coherence_f.append(result)
        # if denominator == 0:
        #     coherence_f.append(0)
        # else:
        #     coherence_f.append(nominator / denominator)

    coherence = []
    for c_f in coherence_f:
        coherence.append(np.abs(c_f).mean())

    return np.array(coherence)


def calculate_segmented_complex_coherency(  # Renamed function
    datetimes: NDArray[np.datetime64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    fs: float,
    segment_length_sec: float = None,
    nperseg: int = None,
    nfft_coh: int = None,
    window_coh: str = "hann",
    noverlap_coh: int = None,
    freq_range: tuple[float, float] = None,
) -> tuple[NDArray[np.datetime64], NDArray[np.complex128]]:  # Changed return type
    """
    Calculates average COMPLEX COHERENCY over frequency for distinct time segments.

    Segments the input signals x and y. For each segment, computes the
    Complex Coherency using Welch's method for PSD/CPSD estimation.
    Averages the *complex* coherency values over a specified frequency range
    (or all frequencies) to get a single complex coherency value for that segment.

    The magnitude of the result represents the average sqrt(MSC), and the phase
    represents the average phase difference in the frequency band.

    Args:
        datetimes (NDArray): Datetime array corresponding to x and y.
        x (NDArray): First time series signal.
        y (NDArray): Second time series signal.
        fs (float): Sampling frequency.
        segment_length_sec (float, optional): Segment length in seconds.
        nperseg (int, optional): Segment length in samples.
        nfft_coh (int, optional): FFT length for Welch/CSD within segments. Defaults to nperseg.
        window_coh (str, optional): Window for Welch/CSD. Defaults to 'hann'.
        noverlap_coh (int, optional): Overlap for Welch/CSD. Defaults to nfft_coh // 2.
        freq_range (tuple[float, float], optional): (fmin, fmax) to average over. Defaults to all frequencies.

    Returns:
        tuple[NDArray[np.datetime64], NDArray[np.complex128]]: # Changed return type description
            - mid_times: Array of datetime objects representing segment midpoints.
            - avg_complex_coherency: Array of the average complex coherency values for each segment.

    Raises:
        ValueError: If inputs/parameters are invalid.
    """
    # --- Input validation (same as before) ---
    if not (len(datetimes) == len(x) == len(y)):
        raise ValueError("Input arrays datetimes, x, and y must have the same length.")
    if segment_length_sec is None and nperseg is None:
        raise ValueError("One of 'segment_length_sec' or 'nperseg' must be provided.")
    if segment_length_sec is not None and nperseg is not None:
        warnings.warn(
            "Both 'segment_length_sec' and 'nperseg' provided. Using 'nperseg'.",
            stacklevel=2,
        )
    if nperseg is None:
        nperseg = int(segment_length_sec * fs)
        if nperseg <= 0:
            raise ValueError(
                "Resulting nperseg from segment_length_sec must be positive."
            )
    elif nperseg <= 0:
        raise ValueError("nperseg must be positive.")

    n_times = len(datetimes)
    if n_times < nperseg:
        warnings.warn(
            f"Signal length ({n_times}) is shorter than segment length ({nperseg}). Calculating over the entire signal.",
            stacklevel=2,
        )
        nperseg = n_times

    # --- Parameters for Welch/CSD (same as before) ---
    nfft_internal = nfft_coh if nfft_coh is not None else nperseg
    if noverlap_coh is None:
        # If nfft_internal is used for Welch/CSD, overlap should be relative to it
        noverlap_internal = nfft_internal // 2
    else:
        noverlap_internal = noverlap_coh
    if noverlap_internal >= nfft_internal:
        warnings.warn(
            f"noverlap_coh ({noverlap_internal}) is greater or equal to internal nfft ({nfft_internal}). Setting overlap to nfft // 2.",
            stacklevel=2,
        )
        noverlap_internal = nfft_internal // 2

    mid_times = []
    avg_complex_coherency_list = []  # Renamed list

    # Iterate through segments
    for i_start in range(0, n_times, nperseg):
        i_end = min(i_start + nperseg, n_times)
        segment_len = i_end - i_start

        if segment_len < nfft_internal:
            warnings.warn(
                f"Skipping segment {i_start}-{i_end} (length {segment_len}) as it's shorter than internal nfft ({nfft_internal}).",
                stacklevel=2,
            )
            continue
        if segment_len <= 0:
            continue

        dt_segment = datetimes[i_start:i_end]
        x_segment = x[i_start:i_end]
        y_segment = y[i_start:i_end]

        try:
            # Calculate PSDs and CPSD (same as before)
            f_seg, pxx_seg = signal.welch(
                x_segment,
                fs=fs,
                window=window_coh,
                nperseg=nfft_internal,
                noverlap=noverlap_internal,
                nfft=nfft_internal,
                scaling="density",
            )
            _, pyy_seg = signal.welch(
                y_segment,
                fs=fs,
                window=window_coh,
                nperseg=nfft_internal,
                noverlap=noverlap_internal,
                nfft=nfft_internal,
                scaling="density",
            )
            _, pxy_seg = signal.csd(
                x_segment,
                y_segment,
                fs=fs,
                window=window_coh,
                nperseg=nfft_internal,
                noverlap=noverlap_internal,
                nfft=nfft_internal,
                scaling="density",
            )
        except ValueError as e:
            warnings.warn(
                f"Welch/CSD failed for segment {i_start}-{i_end}: {e}. Skipping segment.",
                stacklevel=2,
            )
            continue

        # --- Calculate Complex Coherency (frequency-dependent) ---
        # Denominator term: sqrt(Pxx * Pyy)
        # Use np.abs for safety inside sqrt, though PSDs should be non-negative
        denominator_sqrt = np.sqrt(np.abs(pxx_seg * pyy_seg))

        # Calculate complex coherency cxy(f) = Pxy / sqrt(Pxx * Pyy)
        complex_coherency_f = np.divide(
            pxy_seg,  # Numerator is complex CPSD
            denominator_sqrt,
            out=np.zeros_like(pxy_seg, dtype=np.complex128),  # Output is complex
            where=denominator_sqrt > 1e-15,  # Tolerance for denominator
        )
        # Note: Clipping magnitude to 1 isn't strictly necessary but can handle numerical noise
        # mag = np.abs(complex_coherency_f)
        # mask_mag_gt_1 = mag > 1.0
        # if np.any(mask_mag_gt_1):
        #      complex_coherency_f[mask_mag_gt_1] /= mag[mask_mag_gt_1]

        # Average COMPLEX coherency over the specified frequency range
        if freq_range:
            fmin, fmax = freq_range
            freq_mask = (f_seg >= fmin) & (f_seg <= fmax)
            if np.any(freq_mask):
                # Average the COMPLEX values in the specified range
                segment_avg_value = np.mean(complex_coherency_f[freq_mask])
            else:
                warnings.warn(
                    f"No frequencies found in range {freq_range} for segment {i_start}-{i_end}. Setting coherency to NaN.",
                    stacklevel=2,
                )
                segment_avg_value = np.nan + 0j  # Complex NaN
        else:
            # Average the COMPLEX values over all frequencies
            segment_avg_value = np.mean(complex_coherency_f)

        # Calculate midpoint time (same as before)
        mid_time_ns = (
            dt_segment[0].astype("datetime64[ns]").astype(np.int64)
            + (
                dt_segment[-1].astype("datetime64[ns]").astype(np.int64)
                - dt_segment[0].astype("datetime64[ns]").astype(np.int64)
            )
            // 2
        )
        mid_time = mid_time_ns.astype("datetime64[ns]")

        mid_times.append(mid_time)
        avg_complex_coherency_list.append(segment_avg_value)  # Append complex value

    if not mid_times:
        return np.array([], dtype="datetime64[ns]"), np.array(
            [], dtype=np.complex128
        )  # Return complex array

    return np.array(mid_times), np.array(
        avg_complex_coherency_list, dtype=np.complex128
    )  # Ensure output is complex
