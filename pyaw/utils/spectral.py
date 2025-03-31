from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pywt
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import welch

from pyaw.utils import other
import warnings # To warn about truncation


class FFT:
    """
    a class to get fft of a signal of sampling frequency fs
    """

    def __init__(self, array: NDArray, fs: float):
        self.array = array
        self.fs = fs

    def get_fft(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Only return the data of positive frequencies.
        (note: because the returned frequencies are determined by the length of the signal and the sampling rate,
        the previous frequencies returned by the high sampling rate signal are exactly the same as the frequencies of
        the low sampling rate signal.)
        Returns:

        """
        n = len(self.array)
        fft_values = np.fft.fft(self.array)  # fft
        magnitudes = np.abs(fft_values)  # magnitude of fft
        phases = np.angle(fft_values)  # phase of fft
        frequencies = np.fft.fftfreq(n, d=1 / self.fs)  # frequencies
        # only positive frequencies
        return frequencies[: n // 2], magnitudes[: n // 2], phases[: n // 2]

    def plot_fft(
        self, figsize=(10, 6), title="fft"
    ):  # todo: 绘图调用修改后，删除这个方法
        frequencies, amps, _ = self.get_fft()
        fig = plt.figure(figsize=figsize)
        plt.plot(frequencies, amps, color="red")
        plt.xscale("linear")
        plt.yscale("log")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude Spectra")
        plt.grid(which="both", linestyle="--", linewidth=0.5)
        plt.title(f"{title}: (fs={self.fs})")
        plt.show()
        return fig


class PSD:
    """
    a class to get psd of a signal of sampling frequency fs using welch method.
    default 'density'
    (note: 返回的频率由更多参数决定)
    """

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
            frequencies: Array of sample frequencies.
            Pxx: Power spectral density or power spectrum of x.

        """
        frequencies, Pxx = welch(
            self.array,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            window=self.window,
            scaling=self.scaling,
            nfft=self.nfft
        )
        return frequencies, Pxx


class CWT:
    """
    连续小波变换
    """

    def __init__(
        self,
        arr1: NDArray,
        arr2: NDArray,
        scales: NDArray = np.arange(1, 128),
        wavelet: str = "cmor1.5-1.0",
        fs: float = 16.0,
    ) -> None:
        """
        signal1 and signal2 are aligned in time.

        Args:
            arr1:
            arr2:
            scales:
            wavelet:
            fs:
        """
        self.arr1 = arr1
        self.arr2 = arr2
        self.scales = scales
        self.wavelet = wavelet
        self.sampling_period = 1 / fs

    def get_cross_spectral(self):
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
    Zxx1: NDArray, Zxx2: NDArray, cpsd_12: NDArray, step: int = 11
) -> NDArray:
    """
    The times and frequencies corresponding to Zxx1 and Zxx2 should be the same.
    Zxx1 can be Zxx_e or Zxx_b, the order doesn't affect the result.

    Args:
        Zxx1: Spectrogram or stft of x (refer to scipy doc).
        Zxx2: ~
        cpsd_12: the cross power spectral density of signal1 (array1) and signal2 (array2).
        step: Zxx1, Zxx2, cpsd_12 的拆分间隔
    """
    split_array = other.split_array
    cpsd12_split = split_array(cpsd_12, step=step)  # ls
    denominator1ls = split_array(np.abs(Zxx1**2), step=step)
    denominator2ls = split_array(np.abs(Zxx2**2), step=step)

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


def get_coherence_optimized(
    Zxx1: NDArray,
    Zxx2: NDArray,
    cpsd_12: NDArray,
    step: int = 11
) -> NDArray:
    """
    Calculates the average coherence magnitude over frequency for time segments.

    Assumes Zxx1, Zxx2, and cpsd_12 have the same shape (n_freqs, n_times)
    and correspond to the same frequencies and time points.

    The calculation averages the complex coherence over time windows of size 'step'
    for each frequency, takes the magnitude of this average complex coherence,
    and then averages these magnitudes across all frequencies for each time window.

    Args:
        Zxx1: STFT result for signal 1 (n_freqs, n_times).
        Zxx2: STFT result for signal 2 (n_freqs, n_times).
        cpsd_12: Cross power spectral density between signal 1 and 2 (n_freqs, n_times).
        step: The size of the non-overlapping time windows (number of time points)
              over which to average.

    Returns:
        NDArray[np.float64]: An array containing the average coherence magnitude
                             for each time segment (shape: n_segments,).

    Raises:
        ValueError: If input shapes are inconsistent or step size is invalid.
    """
    if not (Zxx1.shape == Zxx2.shape == cpsd_12.shape):
        raise ValueError("Input arrays Zxx1, Zxx2, and cpsd_12 must have the same shape.")
    if Zxx1.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional (n_freqs, n_times).")
    if step <= 0:
        raise ValueError("step must be a positive integer.")

    n_freqs, n_times = Zxx1.shape

    # Determine the number of full segments and the truncated time length
    n_segments = n_times // step
    if n_segments == 0:
        raise ValueError(f"step size ({step}) is larger than the time dimension ({n_times}). No full segments.")

    n_times_trunc = n_segments * step

    if n_times_trunc != n_times:
        warnings.warn(
            f"Time dimension ({n_times}) not perfectly divisible by step ({step}). "
            f"Truncating to {n_times_trunc} time points ({n_segments} segments).",
            stacklevel=2
        )

    # Calculate Power Spectral Densities (PSD)
    # Use np.abs(... )**2 for potentially better numerical stability/readability
    pxx = np.abs(Zxx1[:, :n_times_trunc])**2
    pyy = np.abs(Zxx2[:, :n_times_trunc])**2
    cpsd_trunc = cpsd_12[:, :n_times_trunc] # Use truncated CPSD

    # Reshape arrays to separate segments: (n_freqs, n_segments, step)
    pxx_segmented = pxx.reshape(n_freqs, n_segments, step)
    pyy_segmented = pyy.reshape(n_freqs, n_segments, step)
    cpsd_segmented = cpsd_trunc.reshape(n_freqs, n_segments, step)

    # Average over the time window (axis=2) for each segment and frequency
    pxx_avg = pxx_segmented.mean(axis=2) # Shape: (n_freqs, n_segments)
    pyy_avg = pyy_segmented.mean(axis=2) # Shape: (n_freqs, n_segments)
    cpsd_avg = cpsd_segmented.mean(axis=2) # Shape: (n_freqs, n_segments)

    # Calculate the denominator: sqrt(mean(Pxx) * mean(Pyy))
    # Use np.maximum to avoid sqrt of potentially tiny negative numbers due to precision
    denominator_squared = np.maximum(0, pxx_avg * pyy_avg)
    denominator = np.sqrt(denominator_squared)

    # Calculate complex coherence per frequency per segment
    # |Coherence|^2 = |mean(CPSD)|^2 / (mean(Pxx) * mean(Pyy))
    # Complex Coherence = mean(CPSD) / sqrt(mean(Pxx) * mean(Pyy))
    complex_coherence_f = np.divide(
        cpsd_avg,
        denominator,
        out=np.zeros_like(cpsd_avg, dtype=complex), # Output array for result
        where=denominator != 0  # Condition for division
    )

    # Calculate magnitude of complex coherence (this is coherence, 0 to 1)
    coherence_magnitude_f = np.abs(complex_coherence_f) # Shape: (n_freqs, n_segments)

    # Average coherence magnitude across frequencies (axis=0) for each segment
    avg_coherence_per_segment = coherence_magnitude_f.mean(axis=0) # Shape: (n_segments,)

    return avg_coherence_per_segment


def get_coherence_optimized_no_trunc(
    Zxx1: NDArray[np.complex_],
    Zxx2: NDArray[np.complex_],
    cpsd_12: NDArray[np.complex_],
    step: int = 11
) -> NDArray[np.float64]:
    """
    Calculates the average coherence magnitude over frequency for time segments,
    including a final segment potentially shorter than 'step'.

    Assumes Zxx1, Zxx2, and cpsd_12 have the same shape (n_freqs, n_times)
    and correspond to the same frequencies and time points.

    The function averages the complex coherence over time windows. Windows 0 to N-1
    have size 'step'. The final window N has size n_times % step (if non-zero).
    It takes the magnitude of the average complex coherence for each window at each
    frequency, and then averages these magnitudes across frequencies for each window.

    Args:
        Zxx1: STFT result for signal 1 (n_freqs, n_times).
        Zxx2: STFT result for signal 2 (n_freqs, n_times).
        cpsd_12: Cross power spectral density between signal 1 and 2 (n_freqs, n_times).
        step: The size of the time windows (number of time points) for averaging,
              except possibly the last one.

    Returns:
        NDArray[np.float64]: An array containing the average coherence magnitude
                             for each time segment (shape: n_segments + (1 if remainder else 0),).

    Raises:
        ValueError: If input shapes are inconsistent or step size is invalid.
    """
    if not (Zxx1.shape == Zxx2.shape == cpsd_12.shape):
        raise ValueError("Input arrays Zxx1, Zxx2, and cpsd_12 must have the same shape.")
    if Zxx1.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional (n_freqs, n_times).")
    if step <= 0:
        raise ValueError("step must be a positive integer.")

    n_freqs, n_times = Zxx1.shape
    if n_times == 0:
        return np.array([]) # Return empty if no time points

    # --- Calculation for full segments ---
    n_segments_full = n_times // step
    n_times_full = n_segments_full * step
    results_full = np.array([]) # Initialize empty array for results from full segments

    if n_segments_full > 0:
        # Extract data for full segments
        Zxx1_full = Zxx1[:, :n_times_full]
        Zxx2_full = Zxx2[:, :n_times_full]
        cpsd_12_full = cpsd_12[:, :n_times_full]

        # Calculate Power Spectral Densities (PSD) for full segments
        pxx_full = np.abs(Zxx1_full)**2
        pyy_full = np.abs(Zxx2_full)**2

        # Reshape arrays to separate segments: (n_freqs, n_segments, step)
        pxx_segmented = pxx_full.reshape(n_freqs, n_segments_full, step)
        pyy_segmented = pyy_full.reshape(n_freqs, n_segments_full, step)
        cpsd_segmented = cpsd_12_full.reshape(n_freqs, n_segments_full, step)

        # Average over the time window (axis=2) for each segment and frequency
        pxx_avg_full = pxx_segmented.mean(axis=2) # Shape: (n_freqs, n_segments_full)
        pyy_avg_full = pyy_segmented.mean(axis=2) # Shape: (n_freqs, n_segments_full)
        cpsd_avg_full = cpsd_segmented.mean(axis=2) # Shape: (n_freqs, n_segments_full)

        # Calculate complex coherence per frequency per full segment
        denominator_full_sq = np.maximum(0, pxx_avg_full * pyy_avg_full)
        denominator_full = np.sqrt(denominator_full_sq)
        complex_coherence_f_full = np.divide(
            cpsd_avg_full, denominator_full,
            out=np.zeros_like(cpsd_avg_full, dtype=complex), where=denominator_full != 0
        )

        # Average coherence magnitude across frequencies for each full segment
        coherence_magnitude_f_full = np.abs(complex_coherence_f_full)
        results_full = coherence_magnitude_f_full.mean(axis=0) # Shape: (n_segments_full,)

    # --- Calculation for the trailing segment (if any) ---
    results_trail = np.array([]) # Initialize empty array for trailing result

    if n_times_full < n_times:
        # Extract data for the trailing segment
        Zxx1_trail = Zxx1[:, n_times_full:]
        Zxx2_trail = Zxx2[:, n_times_full:]
        cpsd_12_trail = cpsd_12[:, n_times_full:]

        # Calculate PSDs for the trailing segment
        pxx_trail = np.abs(Zxx1_trail)**2
        pyy_trail = np.abs(Zxx2_trail)**2

        # Average over the remaining time points (axis=1)
        pxx_avg_trail = pxx_trail.mean(axis=1) # Shape: (n_freqs,)
        pyy_avg_trail = pyy_trail.mean(axis=1) # Shape: (n_freqs,)
        cpsd_avg_trail = cpsd_12_trail.mean(axis=1) # Shape: (n_freqs,)

        # Calculate complex coherence per frequency for the trailing segment
        denominator_trail_sq = np.maximum(0, pxx_avg_trail * pyy_avg_trail)
        denominator_trail = np.sqrt(denominator_trail_sq)
        complex_coherence_f_trail = np.divide(
            cpsd_avg_trail, denominator_trail,
            out=np.zeros_like(cpsd_avg_trail, dtype=complex), where=denominator_trail != 0
        )

        # Average coherence magnitude across frequencies for the trailing segment
        coherence_magnitude_f_trail = np.abs(complex_coherence_f_trail)
        # Result is a single scalar value, put into a 1-element array
        results_trail = np.array([coherence_magnitude_f_trail.mean(axis=0)])


    # --- Combine results ---
    # Concatenate the results from full segments and the trailing segment
    final_result = np.concatenate((results_full, results_trail))

    return final_result


def calculate_segmented_coherence(
    datetimes: NDArray[np.datetime64],
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    fs: float,
    segment_length_sec: float = None, # Define segment length in seconds
    nperseg: int = None,          # OR define segment length by samples
    nfft_coh: int = None,         # nfft for coherence calc within segment
    window_coh: str = 'hann',     # Window for coherence calc within segment
    noverlap_coh: int = None,     # Overlap for coherence calc within segment (usually None for Welch/CSD)
    freq_range: tuple[float, float] = None # Optional: (fmin, fmax) to average coherence over
) -> tuple[NDArray[np.datetime64], NDArray[np.float64]]:
    """
    Calculates average coherence over frequency for distinct time segments.

    Segments the input signals x and y based on time duration or number of samples.
    For each segment, computes the Magnitude Squared Coherence (MSC) across
    frequencies using Welch's method for PSD/CPSD estimation within the segment.
    Averages the MSC over a specified frequency range (or all frequencies) to
    get a single coherence value for that segment.

    Args:
        datetimes (NDArray): Datetime array corresponding to x and y.
        x (NDArray): First time series signal.
        y (NDArray): Second time series signal (same length as x and datetimes).
        fs (float): Sampling frequency of the time series.
        segment_length_sec (float, optional): Desired length of each analysis
            segment in seconds. One of segment_length_sec or nperseg must be provided.
        nperseg (int, optional): Desired length of each analysis segment in samples.
            One of segment_length_sec or nperseg must be provided.
        nfft_coh (int, optional): Length of FFT for Welch/CSD calculation *within*
            each segment. Defaults to nperseg if not given.
        window_coh (str, optional): Window function for Welch/CSD calculation.
            Defaults to 'hann'.
        noverlap_coh (int, optional): Overlap for Welch/CSD calculation.
            Defaults to nperseg // 2 if nperseg is specified for Welch/CSD.
            Set to 0 or None for no overlap if calculating coherence just based on the whole segment.
            **Note:** For Welch/CSD, `nperseg` used here refers to the internal FFT length,
            not necessarily the full segment length passed in `nperseg` argument of this function.
            Let's simplify and use `nfft_coh` as the primary control for FFT length within the segment.
        freq_range (tuple[float, float], optional): Tuple (fmin, fmax) specifying
            the frequency range over which to average the calculated coherence.
            If None, averages over all calculated frequencies. Defaults to None.


    Returns:
        tuple[NDArray[np.datetime64], NDArray[np.float64]]:
            - mid_times: Array of datetime objects representing the midpoint of each segment.
            - avg_coherence: Array of the average coherence values for each segment.

    Raises:
        ValueError: If inputs have inconsistent lengths, parameters are invalid,
                    or neither segment_length_sec nor nperseg is provided.
    """
    if not (len(datetimes) == len(x) == len(y)):
        raise ValueError("Input arrays datetimes, x, and y must have the same length.")
    if segment_length_sec is None and nperseg is None:
        raise ValueError("One of 'segment_length_sec' or 'nperseg' must be provided.")
    if segment_length_sec is not None and nperseg is not None:
        warnings.warn("Both 'segment_length_sec' and 'nperseg' provided. Using 'nperseg'.", stacklevel=2)
    if nperseg is None:
        nperseg = int(segment_length_sec * fs)
        if nperseg <= 0:
             raise ValueError("Resulting nperseg from segment_length_sec must be positive.")
    elif nperseg <= 0 :
         raise ValueError("nperseg must be positive.")

    n_times = len(datetimes)
    if n_times < nperseg:
        warnings.warn(f"Signal length ({n_times}) is shorter than segment length ({nperseg}). Calculating coherence over the entire signal.", stacklevel=2)
        nperseg = n_times # Use the whole signal as one segment

    # --- Parameters for Welch/CSD inside segments ---
    # Use nfft_coh if provided, otherwise default to the segment length nperseg
    # This nfft determines the frequency resolution *within* the segment average
    nfft_internal = nfft_coh if nfft_coh is not None else nperseg
    if noverlap_coh is None:
        noverlap_internal = nfft_internal // 2 # Default overlap for Welch/CSD
    else:
        noverlap_internal = noverlap_coh


    mid_times = []
    avg_coherence_list = []

    # Iterate through segments using the overall nperseg for segmentation
    for i_start in range(0, n_times, nperseg):
        i_end = min(i_start + nperseg, n_times)
        segment_len = i_end - i_start

        # Skip segments that are too short for the internal FFT calculation
        if segment_len < nfft_internal:
             warnings.warn(f"Skipping segment {i_start}-{i_end} (length {segment_len}) as it's shorter than internal nfft ({nfft_internal}).", stacklevel=2)
             continue
        if segment_len <= 0: # Should not happen if nperseg is positive, but safe check
             continue

        # Extract data for the current segment
        dt_segment = datetimes[i_start:i_end]
        x_segment = x[i_start:i_end]
        y_segment = y[i_start:i_end]

        # Calculate average PSDs and CPSD for this segment using Welch/CSD
        try:
            f_seg, Pxx_seg = signal.welch(
                x_segment, fs=fs, window=window_coh, nperseg=nfft_internal,
                noverlap=noverlap_internal, nfft=nfft_internal, scaling='density'
            )
            _, Pyy_seg = signal.welch(
                y_segment, fs=fs, window=window_coh, nperseg=nfft_internal,
                noverlap=noverlap_internal, nfft=nfft_internal, scaling='density'
            )
            _, Pxy_seg = signal.csd(
                x_segment, y_segment, fs=fs, window=window_coh, nperseg=nfft_internal,
                noverlap=noverlap_internal, nfft=nfft_internal, scaling='density'
            )
        except ValueError as e:
            # Handle cases where segment might be too short after windowing/detrending etc.
            warnings.warn(f"Welch/CSD failed for segment {i_start}-{i_end}: {e}. Skipping segment.", stacklevel=2)
            continue


        # Calculate Magnitude Squared Coherence (frequency-dependent) for this segment
        denominator = Pxx_seg * Pyy_seg
        Cxy_seg_f = np.divide(
            np.abs(Pxy_seg)**2,
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator > 1e-15 # Tolerance for denominator
        )
        Cxy_seg_f = np.clip(Cxy_seg_f, 0, 1) # Ensure range [0, 1]

        # Average coherence over the specified frequency range
        if freq_range:
            fmin, fmax = freq_range
            freq_mask = (f_seg >= fmin) & (f_seg <= fmax)
            if np.any(freq_mask):
                segment_avg_coherence = np.mean(Cxy_seg_f[freq_mask])
            else:
                warnings.warn(f"No frequencies found in range {freq_range} for segment {i_start}-{i_end}. Setting coherence to NaN.", stacklevel=2)
                segment_avg_coherence = np.nan # Or 0, depending on desired behavior
        else:
            # Average over all frequencies
            segment_avg_coherence = np.mean(Cxy_seg_f)

        # Calculate midpoint time for the segment
        # Midpoint calculation more robust for datetime64
        mid_time_ns = dt_segment[0].astype('datetime64[ns]').astype(np.int64) + \
                      (dt_segment[-1].astype('datetime64[ns]').astype(np.int64) - \
                       dt_segment[0].astype('datetime64[ns]').astype(np.int64)) // 2
        mid_time = mid_time_ns.astype('datetime64[ns]')


        mid_times.append(mid_time)
        avg_coherence_list.append(segment_avg_coherence)

    if not mid_times: # Handle case where no segments were processed
        return np.array([], dtype='datetime64[ns]'), np.array([], dtype=float)

    return np.array(mid_times), np.array(avg_coherence_list)


def calculate_segmented_complex_coherency( # Renamed function
    datetimes: NDArray[np.datetime64],
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    fs: float,
    segment_length_sec: float = None,
    nperseg: int = None,
    nfft_coh: int = None,
    window_coh: str = 'hann',
    noverlap_coh: int = None,
    freq_range: tuple[float, float] = None
) -> tuple[NDArray[np.datetime64], NDArray[np.complex128]]: # Changed return type
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
        warnings.warn("Both 'segment_length_sec' and 'nperseg' provided. Using 'nperseg'.", stacklevel=2)
    if nperseg is None:
        nperseg = int(segment_length_sec * fs)
        if nperseg <= 0:
             raise ValueError("Resulting nperseg from segment_length_sec must be positive.")
    elif nperseg <= 0 :
         raise ValueError("nperseg must be positive.")

    n_times = len(datetimes)
    if n_times < nperseg:
        warnings.warn(f"Signal length ({n_times}) is shorter than segment length ({nperseg}). Calculating over the entire signal.", stacklevel=2)
        nperseg = n_times

    # --- Parameters for Welch/CSD (same as before) ---
    nfft_internal = nfft_coh if nfft_coh is not None else nperseg
    if noverlap_coh is None:
        # If nfft_internal is used for Welch/CSD, overlap should be relative to it
        noverlap_internal = nfft_internal // 2
    else:
        noverlap_internal = noverlap_coh
    if noverlap_internal >= nfft_internal:
         warnings.warn(f"noverlap_coh ({noverlap_internal}) is greater or equal to internal nfft ({nfft_internal}). Setting overlap to nfft // 2.", stacklevel=2)
         noverlap_internal = nfft_internal // 2


    mid_times = []
    avg_complex_coherency_list = [] # Renamed list

    # Iterate through segments
    for i_start in range(0, n_times, nperseg):
        i_end = min(i_start + nperseg, n_times)
        segment_len = i_end - i_start

        if segment_len < nfft_internal:
             warnings.warn(f"Skipping segment {i_start}-{i_end} (length {segment_len}) as it's shorter than internal nfft ({nfft_internal}).", stacklevel=2)
             continue
        if segment_len <= 0:
             continue

        dt_segment = datetimes[i_start:i_end]
        x_segment = x[i_start:i_end]
        y_segment = y[i_start:i_end]

        try:
            # Calculate PSDs and CPSD (same as before)
            f_seg, Pxx_seg = signal.welch(
                x_segment, fs=fs, window=window_coh, nperseg=nfft_internal,
                noverlap=noverlap_internal, nfft=nfft_internal, scaling='density'
            )
            _, Pyy_seg = signal.welch(
                y_segment, fs=fs, window=window_coh, nperseg=nfft_internal,
                noverlap=noverlap_internal, nfft=nfft_internal, scaling='density'
            )
            _, Pxy_seg = signal.csd(
                x_segment, y_segment, fs=fs, window=window_coh, nperseg=nfft_internal,
                noverlap=noverlap_internal, nfft=nfft_internal, scaling='density'
            )
        except ValueError as e:
            warnings.warn(f"Welch/CSD failed for segment {i_start}-{i_end}: {e}. Skipping segment.", stacklevel=2)
            continue

        # --- Calculate Complex Coherency (frequency-dependent) ---
        # Denominator term: sqrt(Pxx * Pyy)
        # Use np.abs for safety inside sqrt, though PSDs should be non-negative
        denominator_sqrt = np.sqrt(np.abs(Pxx_seg * Pyy_seg))

        # Calculate complex coherency cxy(f) = Pxy / sqrt(Pxx * Pyy)
        complex_coherency_f = np.divide(
            Pxy_seg, # Numerator is complex CPSD
            denominator_sqrt,
            out=np.zeros_like(Pxy_seg, dtype=np.complex128), # Output is complex
            where=denominator_sqrt > 1e-15 # Tolerance for denominator
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
                warnings.warn(f"No frequencies found in range {freq_range} for segment {i_start}-{i_end}. Setting coherency to NaN.", stacklevel=2)
                segment_avg_value = np.nan + 0j # Complex NaN
        else:
            # Average the COMPLEX values over all frequencies
            segment_avg_value = np.mean(complex_coherency_f)

        # Calculate midpoint time (same as before)
        mid_time_ns = dt_segment[0].astype('datetime64[ns]').astype(np.int64) + \
                      (dt_segment[-1].astype('datetime64[ns]').astype(np.int64) - \
                       dt_segment[0].astype('datetime64[ns]').astype(np.int64)) // 2
        mid_time = mid_time_ns.astype('datetime64[ns]')

        mid_times.append(mid_time)
        avg_complex_coherency_list.append(segment_avg_value) # Append complex value

    if not mid_times:
        return np.array([], dtype='datetime64[ns]'), np.array([], dtype=np.complex128) # Return complex array

    return np.array(mid_times), np.array(avg_complex_coherency_list, dtype=np.complex128) # Ensure output is complex