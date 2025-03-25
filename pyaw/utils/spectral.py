from typing import Optional

import numpy as np
from nptyping import NDArray
import pywt
from matplotlib import pyplot as plt
from scipy.signal import welch

from pyaw.utils import other


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
    ):
        self.array = array
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window
        self.scaling = scaling

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
        cross_spectrum = coefficients_f * np.conj(coefficients_g)
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
