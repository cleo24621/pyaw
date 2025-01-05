# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/31
@DESCRIPTION: 
"""
from typing import Optional

import numpy as np
import pywt
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import welch, spectrogram, stft, correlate, coherence, csd

from pyaw import swarm, utils





class PSD:
    def __init__(self, array, fs, nperseg: Optional[int] = None, noverlap: Optional[int] = None, window='hann'):
        self.array = array
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window

    def get_psd(self):
        freqs, Pxx = welch(self.array, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap,
                           window=self.window)
        return freqs, Pxx


class Spectrogram:
    def __init__(self, signal, fs):
        self.signal = signal
        self.fs = fs

    def get_spectrogram(self,window='hann',window_length=4):
        nperseg = int(window_length * self.fs)  # 每个窗的采样点数
        noverlap = nperseg // 2  # 50%重叠
        frequencies, times, Sxx = spectrogram(self.signal.values, fs=self.fs,window=window, nperseg=nperseg, noverlap=noverlap,)
        return frequencies, times, Sxx

    def plot_spectrogram(self, figsize=(10, 4), shading='gouraud'):
        freqs, times, Sxx = self.get_spectrogram()
        plt.figure(figsize=figsize)
        plt.pcolormesh(times, freqs, 10 * np.log10(Sxx))
        plt.colorbar(label='csd module')
        plt.xlabel('UT Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Cross-Spectrogram module between Signal 1 and Signal 2')
        plt.show()
        return figure


class STFT:
    def __init__(self, signal, fs):
        self.signal = signal
        self.fs = fs

    def get_stft(self):
        frequencies, times, Sxx = stft(self.signal.values, fs=self.fs)
        return frequencies, times, Sxx

    def plot_stft(self, figsize=(10, 4), shading='gouraud'):
        freqs, times, Sxx = self.get_stft()
        Sxx = abs(Sxx)
        plt.figure(figsize=figsize)
        plt.pcolormesh(times, freqs, Sxx, shading=shading)
        plt.colorbar(label='csd module')
        plt.xlabel('UT Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Cross-Spectrogram module between Signal 1 and Signal 2')
        plt.show()
        return figure


class CrossCorrelation:
    def __init__(self, signal1, signal2, mode='full'):
        self.signal1 = signal1
        self.signal2 = signal2
        self.mode = mode

    def get_cross_correlation(self):
        cross_corr = correlate(self.signal1, self.signal2, mode='full')
        return cross_corr


class Coherence:
    def __init__(self, signal1, signal2, fs):
        self.signal1 = signal1
        self.signal2 = signal2
        self.fs = fs

    def get_coherence(self):
        frequencies, coherence_values = coherence(self.signal1, self.signal2, fs=self.fs)
        return frequencies, coherence_values

    def plot_coherence(self, figsize=(10, 6), title='coherence'):
        freqs, coherence_values = self.get_coherence()
        plt.figure(figsize=figsize)
        plt.plot(freqs, coherence_values, color='red')
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('coherence')
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.title(f'{title}: (fs={self.fs})')
        plt.show()
        return figure


class CSD:
    def __init__(self, signal1, signal2, fs):
        self.signal1 = signal1
        self.signal2 = signal2
        self.fs = fs

    def get_csd(self):
        freqs, csd_values = csd(self.signal1, self.signal2, fs=self.fs)
        return freqs, csd_values

    def plot_csd_module(self, figsize=(10, 4), title='csd module'):
        freqs, csd_values = self.get_csd()
        plt.figure(figsize=figsize)
        plt.plot(freqs, np.abs(csd_values))
        plt.title(f'{title}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('CSD module')
        plt.show()
        return figure

    def plot_csd_phase(self, figsize=(10, 4), title='csd phase'):
        freqs, csd_values = self.get_csd()
        plt.figure(figsize=figsize)
        plt.plot(freqs, np.abs(csd_values))
        plt.title(f'{title}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('CSD phase')
        plt.show()
        return figure


class TimeCSD:
    def __init__(self, signal1, signal2, fs):
        # todo: threshold
        self.signal1 = signal1
        self.signal2 = signal2
        self.fs = fs

    def get_time_csd(self, threshold: Optional[float] = None):
        stft1 = STFT(self.signal1, self.fs)
        stft2 = STFT(self.signal2, self.fs)
        frequencies1, times1, Sxx1 = stft1.get_stft()
        frequencies2, times2, Sxx2 = stft2.get_stft()
        assert all(frequencies1 == frequencies2), "frequencies1 must be equal to frequencies2"
        assert all(times1 == times2), "times1 must be equal to times2"
        # Compute the cross-spectrogram by taking the product of the two spectrograms
        cross_spectrogram = Sxx1 * np.conj(Sxx2)
        module = np.abs(cross_spectrogram)
        phase_difference = np.degrees(np.angle(cross_spectrogram))
        if threshold:
            phase_difference = np.where(module > threshold, phase_difference, np.nan)
        return module, phase_difference, times1, frequencies1

    def plot_module(self, figsize=(10, 4), shading='gouraud'):
        module, _, times, freqs = self.get_time_csd()
        plt.figure(figsize=figsize)
        plt.pcolormesh(times, freqs, module, shading=shading)
        plt.colorbar(label='csd module')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Cross-Spectrogram module between Signal 1 and Signal 2')
        plt.show()
        return figure

    def plot_phase(self, figsize=(10, 4), shading='gouraud'):
        _, phase_difference, times, freqs = self.get_time_csd()
        plt.figure(figsize=figsize)
        plt.pcolormesh(times, freqs, phase_difference, shading=shading)
        plt.colorbar(label='csd phase')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Cross-Spectrogram phase between Signal 1 and Signal 2')
        plt.show()
        return figure

    def get_phase_hist_counts(self, num_bins: int = 50):
        _, phase_difference, times, freqs = self.get_time_csd()
        hist_counts = np.zeros((len(freqs), num_bins - 1))  # num_bins-1 bins
        phase_bins = np.linspace(-180, 180, num_bins)
        # Loop over each frequency and calculate histogram for phases
        for i, freq in enumerate(freqs):
            hist_counts[i], _ = np.histogram(phase_difference[i], bins=phase_bins)
        # Normalize counts for better visualization
        hist_counts = hist_counts / np.max(hist_counts, axis=1, keepdims=True)
        return hist_counts, freqs

    def plot_phase_hist_counts(self, figsize=(10, 4)):
        hist_counts, freqs = self.get_phase_hist_counts()
        plt.figure(figsize=figsize)
        plt.imshow(hist_counts, extent=[-180, 180, freqs[-1], freqs[0]], aspect='auto', cmap='jet')
        plt.colorbar(label='Normalized Counts')
        plt.xlabel('Phase [degree]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Phase Histogram')
        plt.show()
        return figure


# def fft_(series_: pd.Series, fs: float, if_plot: bool = False, figsize=(10, 6), title='fft') -> pd.Series:
#     """
#     :param series_: [pd.series]. the type of index is pd.datetime.
#     :param fs: [float]. sampling rate.
#     :return: the index is frequency, the value is amplitude.
#     """
#     # make sure that the series doesn't have "nan" values
#     if series_.isna().any():
#         compare_before_after_interpolate(series_, method_='linear')
#         series_ = series_.interpolate(method='linear')
#     n = len(series_)
#     fft_values = np.fft.fft(series_.values)  # fft
#     amplitude_spectrum = np.abs(fft_values)  # magnitude of fft
#     f_values = np.fft.fftfreq(n, d=1 / fs)  # frequencies
#     # only positive frequencies
#     positive_frequencies = f_values[:n // 2]
#     positive_amplitude_spectrum = amplitude_spectrum[:n // 2]
#     if if_plot:
#         plt.figure(figsize=figsize)
#         plt.plot(positive_frequencies, positive_amplitude_spectrum, color='red')
#         plt.xscale('linear')
#         plt.yscale('log')
#         plt.xlabel('Frequency (Hz)')
#         plt.ylabel('Amplitude Spectra')
#         # plt.legend()
#         plt.grid(which='both', linestyle='--', linewidth=0.5)
#         plt.title(f'{title}: (fs={fs})')
#         plt.show()
#     return pd.Series(index=positive_frequencies, data=positive_amplitude_spectrum)

class CWT:
    def __init__(self, signal1, signal2, scales=np.arange(1, 128), wavelet='cmor1.5-1.0', sampling_period=1 / 16):
        """
        signal1 and signal2 are aligned in time.
        :param signal1:
        :param signal2:
        :param scales:
        :param wavelet:
        :param sampling_period:
        """
        self.signal1 = signal1
        self.signal2 = signal2
        self.scales = scales
        self.wavelet = wavelet
        self.sampling_period = sampling_period

    def get_cross_spectral(self):
        coefficients_f, freqs = pywt.cwt(self.signal1, self.scales, self.wavelet,
                                         sampling_period=self.sampling_period)  # CWT for signal1
        coefficients_g, _ = pywt.cwt(self.signal2, self.scales, self.wavelet,
                                     sampling_period=self.sampling_period)  # CWT for signal2
        cross_spectrum = coefficients_f * np.conj(coefficients_g)
        cross_spectrum_modulus = np.abs(cross_spectrum)
        cross_phase = np.degrees(np.angle(cross_spectrum))
        return cross_spectrum_modulus, cross_phase, freqs

    def plot_module(self, figsize=(10, 4)):
        cross_spectrum_modulus, _, freqs = self.get_cross_spectral()
        plt.figure(figsize=figsize)
        plt.imshow(cross_spectrum_modulus, extent=[self.signal1.index[0], self.signal1.index[-1], freqs[-1], freqs[0]],
                   aspect='auto', cmap='jet')
        plt.colorbar(label='Module')
        plt.xlabel('UT Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('One dimensional Continuous Wavelet Transform Modulus')
        plt.show()
        return figure

    def plot_phase(self, figsize=(10, 4)):
        _, cross_phase, freqs = self.get_cross_spectral()
        plt.figure(figsize=figsize)
        plt.imshow(cross_phase, extent=[self.signal1.index[0], self.signal1.index[-1], freqs[-1], freqs[0]],
                   aspect='auto', cmap='jet')
        plt.colorbar(label='Phase [degree]')
        plt.xlabel('UT Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Cross-Phase')
        plt.show()
        return figure

    def get_phase_hist_counts(self, num_bins=50):
        """
        :param num_bins: Number of bin edges for phase histogram
        :return:
        """
        cross_spectrum_modulus, cross_phase, freqs = self.get_cross_spectral()
        hist_counts = np.zeros((len(freqs), num_bins - 1))  # num_bins-1 bins
        phase_bins = np.linspace(-180, 180, num_bins)
        # Loop over each frequency and calculate histogram for phases
        for i, freq in enumerate(freqs):
            # # 仅考虑模值大于阈值的相位
            # valid_phases = cross_phase[i][cross_spectrum_modulus[i] > 100]
            # if len(valid_phases) > 0:
            #     hist_counts[i], _ = np.histogram(valid_phases, bins=phase_bins)
            hist_counts[i], _ = np.histogram(cross_phase[i], bins=phase_bins)
        # Normalize counts for better visualization
        hist_counts = hist_counts / np.max(hist_counts, axis=1, keepdims=True)
        return hist_counts, freqs

    def plot_phase_hist_counts(self, figsize=(10, 4)):
        hist_counts, freqs = self.get_phase_hist_counts()
        plt.figure(figsize=figsize)
        plt.imshow(hist_counts, extent=[-180, 180, freqs[-1], freqs[0]], aspect='auto', cmap='jet')
        plt.colorbar(label='Normalized Counts')
        plt.xlabel('Phase [degree]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Phase Histogram')
        plt.show()
        return figure


def main(fp1, fp2, start, end, compo='12'):
    pass
    # assert compo in ['12', '21']
    # df1 = swarm.pre_e(fp1, start, end, handle_outliers=True)
    # df2 = swarm.pre_b(fp2, start, end, handle_outliers=True)
    # df1['timestamp'] = df1.index.astype('int64')
    # df2['timestamp'] = df2.index.astype('int64')
    # if compo == '12':
    #     e = df1['eh1_enu2']
    #     b = df2['b1_enu1']
    # else:
    #     e = df1['eh1_enu1']
    #     b = df2['b1_enu2']
    # b = utils_preprocess.align_high2low(b, e)
    # cwt = CWT(e, b)
    # _ = cwt.plot_module()
    # _ = cwt.plot_phase()
    # _ = cwt.plot_phase_hist_counts()
    # return None


if __name__ == '__main__':
    fp1 = r"\\Diskstation1\file_three\aw\swarm\A\efi16\sw_efi16A_20160311T000000_20160311T235959_0.pkl"
    fp2 = r"\\Diskstation1\file_three\aw\swarm\A\vfm50\sw_vfm50A_20160311T060000_20160311T070000_0.pkl"
    start = '20160311T064705'
    end = '20160311T064725'
    compo = '21'
    # fp1 = input('Enter the file path of the first file: ')
    # fp2 = input('Enter the file path of the second file: ')
    # start = input('Enter the start time: ')
    # end = input('Enter the end time: ')
    # compo = input('Enter the component (12 or 21): ')
    main(fp1, fp2, start, end, compo)
