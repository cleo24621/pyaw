# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/20
@DESCRIPTION: 
"""

import numpy as np
import pandas as pd
import plotly.express as px
import pywt
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy.typing import ArrayLike
from scipy.signal import welch, spectrogram

from pyaw.paras import mu0, me, mp, e, kB


def get_va(B0: float | ndarray | pd.Series, np: float | ndarray | pd.Series) -> float | ndarray | pd.Series:
    """
    get local alfven velocity
    :param B0: (T SI)
    :param np: (m^{-3}) number density of proton
    :return: (m/s)
    """
    # todo:: may need modify because "n" quality problem.
    return B0 / np.sqrt(mu0 * (mp * np))


def get_ion_gyrofrequency(B: float | ndarray | pd.Series, mi: float, qi=e):
    """
    $\Omega_i$
    :param B: (T SI) measured magnetic field.
    :param qi: (C SI) in the study (in earth ionosphere and magnetosphere), all types of ions charge is e (elementary charge).
    :param mi: (kg SI) ion absolute mass. (consider different ions) may ues weight.
    :return: (rad/s SI) ion gyrofrequency
    """
    return (qi * B) / mi


def get_ion_gyroradius(Ti, mi, Omega_i):
    """
    $\rho_i$
    :param Ti: (K SI)
    :param mi: (kg SI) weight?
    :param Omega_i: (rad/s SI) ion gyrofrequency
    :return: (km SI)
    """
    return np.sqrt(Ti * mi) / Omega_i


def get_ion_acoustic_gyroradius(Te, mi, Omega_i):
    """
    $\rho_s$
    :param Te:
    :param mi:
    :param Omega_i:
    :return:
    """
    return np.sqrt(Te * mi) / Omega_i


def get_electron_inertial_length(ne, me=me, mu0=mu0, e=e):
    """
    $\lambda_e$
    :param me: (kg)
    :param mu0: (H/m)
    :param ne: (cm^{-3} SI)
    :param e: (C)
    :return: (km)
    """
    return np.sqrt(me / (mu0 * ne * e ** 2))


def get_beta(n: float, T: pd.Series | np.ndarray, B: pd.Series | np.ndarray):
    """
    $\beta = \frac{p}{p_mag} = \frac{nk_B T}{B^2 / 2\mu_0}$  refer to https://en.wikipedia.org/wiki/Plasma_beta
    (refer to "K. STASIEWICZ1, 1999, SMALL SCALE ALFVÉNIC STRUCTURE IN THE AURORA" [1])
    low-beta: $\beta < m_e / m_i$ [1]
    intermediate-beta: $m_e / m_i < \beta < 1$ [1]
    :param n: number density (suppose $n=n_i=n_e$)
    :param T: plasma temperature (suppose $T=(T_i + T_e) / 2$) [1]
    :param B: measured magnetic field
    :return:
    """
    return (2 * mu0 * kB * n * T) / (B ** 2)


def get_complex_impedance(mu0, va, Sigma_P, omega, z):
    """
    use "np.abs()" to get the magnitude of complex impedance
    :param mu0:
    :param va:
    :param Sigma_P:
    :param omega: angular frequency
    :param z: the distance from the reflection point (usually 100~200 km altitude)
    :return:
    """
    Gamma = (1 / Sigma_P - mu0 * va) / (1 / Sigma_P + mu0 * va)
    return mu0 * va * ((1 + Gamma * np.exp(-2j * omega * z / va)) / (1 - Gamma * np.exp(-2j * omega * z / va)))


def E_B_ratio_kaw(va, f, rho_i, rho_s, lambda_e, v_fit):
    """
    refer to: HULL A J, CHASTON C C, DAMIANO P A. Multipoint Cluster Observations of Kinetic Alfvén Waves, Electron Energization, and O + Ion Outflow Response in the Mid‐Altitude Cusp Associated With Solar Wind Pressure and/or IMF B  Z  Variations[J/OL]. Journal of Geophysical Research: Space Physics, 2023, 128(11): e2023JA031982. DOI:10.1029/2023JA031982.
    :param va:
    :param f: frequency
    :param rho_i:
    :param rho_s:
    :param lambda_e:
    :param v_fit: the velocity corresponding to the fitted curve
    :return:
    """
    k_transverse = (2 * np.pi * f) / v_fit
    return va * np.sqrt(
        (1 + (k_transverse ** 2) * (lambda_e ** 2)) / (1 + (k_transverse ** 2) * (rho_i ** 2 + rho_s ** 2))) * (
            1 + (k_transverse ** 2) * (rho_i ** 2))


# data process
def set_outliers_nan(series, threshold):
    # todo: may need improve
    series_copy = series.copy()
    diff = series_copy.diff()
    # 设置一个突变检测阈值
    spikes = diff[diff.abs() > threshold]
    print(spikes)
    series_copy.loc[diff.abs() > threshold] = np.nan
    # suppose a part of the data is [1,nan,100] and set the threshold 10, the outliers should be 100. But use series.diff(), it will return [nan,nan,nan], and none of them exceed the threshold, so the method cannot recognize the outlier 100.
    # to solve this problem, I design an addition process.
    series_copy[(np.abs(series_copy) / np.abs(series_copy).mean()) > 10] = np.nan
    return series_copy  # series_scores = (series - series.mean()) / series.std()  # # 设置阈值，通常 Z 分数大于 3 或小于 -3 的点可以认为是异常点，超过的设置为nan  # threshold = 2  # print(series[np.abs(series_scores) > threshold])  # series[np.abs(series_scores) > threshold] = np.nan  # return series


def move_average(series_: pd.Series, window: int, min_periods: int = 1, draw: bool = False,
                 savefig: bool = False) -> pd.Series:
    """
    :param series_:
    :param window:
    :param draw:
    :param savefig:
    :return:
    """
    series_mov_ave = series_.rolling(window=window, min_periods=min_periods).mean()
    # figure: before and after moving average comparison
    if draw:
        plt.figure()
        plt.plot(series_.index, series_)
        plt.plot(series_mov_ave.index, series_mov_ave)
        plt.xlabel('Time (UTC)')
        if savefig:
            plt.savefig(f'before and after moving average comparison')
    return series_mov_ave


# low, high, band pass filter
# butter()  # get butter object
# nyquist = 0.5 * fs
# low = lowcut / nyquist  # lowcut: low-cutoff frequency
# high = highcut / nyquist  # highcut: high-cutoff frequency
# b,a = butter(order,[low,high],btype)
# filtfilt(b,a,data)  # apply butter object to data (pd.Series)

# savgol_filter()

# wavelet
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


def fft_(series_: pd.Series, fs: float, if_plot: bool = False, figsize=(10, 6), title='fft') -> pd.Series:
    """
    :param series_: [pd.series]. the type of index is pd.datetime.
    :param fs: [float]. sampling rate.
    :return: the index is frequency, the value is amplitude.
    """
    # make sure that the series doesn't have "nan" values
    if series_.isna().any():
        compare_before_after_interpolate(series_, method_='linear')
        series_ = series_.interpolate(method='linear')
    n = len(series_)
    fft_values = np.fft.fft(series_.values)  # fft
    amplitude_spectrum = np.abs(fft_values)  # magnitude of fft
    f_values = np.fft.fftfreq(n, d=1 / fs)  # frequencies
    # only positive frequencies
    positive_frequencies = f_values[:n // 2]
    positive_amplitude_spectrum = amplitude_spectrum[:n // 2]
    if if_plot:
        plt.figure(figsize=figsize)
        plt.plot(positive_frequencies, positive_amplitude_spectrum, color='red')
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude Spectra')
        # plt.legend()
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.title(f'{title}: (fs={fs})')
        plt.show()
    return pd.Series(index=positive_frequencies, data=positive_amplitude_spectrum)


def psd_(series_, fs, nperseg, nperseg_denominator=2, window='hammind'):
    # todo:: may delete this function
    # todo:: understand psd
    return welch(series_.values, fs=fs, nperseg=nperseg, noverlap=nperseg / nperseg_denominator, window=window)


# plt
def plt_1f_2curve(x, y1, y2, title='', xlabel='', ylabel='', y1lable='', y2lable=''):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, label=y1lable)
    plt.plot(x, y2, label=y2lable)
    plt.legend()
    plt.show()


def compare_before_after_interpolate(series_: pd.Series, method_='linear', figsize=(10, 6)):
    """
    :param series_: the type of index is pd.datetime.
    :param method_:
    :param figsize:
    :return:
    """
    print(f'The number of NaN values: {series_.isna().sum()}')
    series_interpolate = series_.interpolate(method=method_)
    x = series_.index
    fig, axs = plt.subplots(3, figsize=figsize)
    axs[0].plot(x, series_, )
    axs[1].plot(x, series_interpolate, )
    axs[2].plot(x, series_, x)
    plt.show()


def plt_psd(frequencies, psd, method, figsize=(10, 6), xlim: tuple[float, float] = None):
    plt.figure(figsize=figsize)
    plt.plot(frequencies, psd)
    plt.xlim(xlim)
    plt.yscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"Power Spectral Density (PSD) using {method}")
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.show()


def plt_subplots(x: ArrayLike, y1: ArrayLike, y2: ArrayLike, y3: ArrayLike, y4: ArrayLike) -> None:
    """
    plot a figure including 5 sub figures. the 1st sub figure on a line.
    :param x:
    :param y1:
    :param y2:
    :param y3:
    :param y4:
    :return:
    """
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)  # 3 行 2 列的网格
    # 0
    ax1 = fig.add_subplot(gs[0, :])  # 第 0 行，跨越所有列
    ax1.plot(x, y1)
    ax1.plot(x, y2)
    ax1.plot(x, y3)
    ax1.plot(x, y4)
    ax1.grid(which='both', linestyle='--', linewidth=0.5)

    # 1,0
    ax10 = fig.add_subplot(gs[1, 0])  # 第 1 行，第 0 列
    ax10.plot(x, y1)
    ax10.set_title('10 title')
    # 1,1
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(x, y2)
    # 2,0
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.plot(x, y3)
    # 2,1
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.plot(x, y4)
    # show figure
    plt.tight_layout()
    plt.show()
    return None


class PlotSignal:
    def one_signal_ana_inc_spectrogram(self, signal: pd.Series, window):
        """

        :param signal: measurement signal
        :param window:
        :return:
        """
        signal_back = move_average(signal, window=window, draw=False)
        signal_perturbation = signal - signal_back
        d = fft_(signal_perturbation, fs=16, if_plot=False)
        signal_fft = pd.Series(data=d.values, index=d.index)
        f, pxx = welch(signal_perturbation.values, fs=16, nperseg=16 * 20, noverlap=16 * 10)
        signal_psd = pd.Series(data=pxx, index=f)
        frequencies, times, Sxx = spectrogram(signal_perturbation.values, fs=16, nperseg=2 ** 8)
        sig_spectrogram = {'frequencies': frequencies, 'times': times, 'Sxx': Sxx}
        self.one_signal_detailed(signal, signal_back, signal_perturbation, signal_fft, signal_psd, sig_spectrogram)

    def one_signal_simple(self, signal: pd.Series, fs: float) -> None:
        """
        plot some figure for potential insight.
        :param signal: usually a measurement signal
        :param fs: sample rate (Hz)
        :return:
        """
        fig = plt.figure()
        gs = fig.add_gridspec(3, 2)  # 3 行 2 列的网格
        # 0
        # original signal
        ax1 = fig.add_subplot(gs[0, :])  # 第 0 行，跨越所有列
        ax1.plot(signal.index, signal.values)
        ax1.grid(which='both', linestyle='--', linewidth=0.5)
        # 1,0
        # move average
        signal_mv = move_average(signal, window=20 * fs, draw=False)
        ax10 = fig.add_subplot(gs[1, 0])  # 第 1 行，第 0 列
        ax10.plot(signal.index, signal.values)
        ax10.plot(signal_mv.index, signal_mv.values)
        ax10.set_title('10 title')
        # 1,1
        ax11 = fig.add_subplot(gs[1, 1])
        ax11.plot(signal_mv.index, signal_mv.values)
        # 2,0
        # medfilt
        from scipy.signal import medfilt
        # Apply a median filter with a specified kernel size
        kernel_size = 20 * fs + 1  # Must be an odd number
        signal_medfilt = medfilt(signal, kernel_size=kernel_size)
        ax20 = fig.add_subplot(gs[2, 0])
        ax20.plot(signal.index, signal.values)
        ax20.plot(signal.index, signal_medfilt)
        # 2,1
        ax21 = fig.add_subplot(gs[2, 1])
        ax21.plot(signal.index, signal_medfilt)
        # show figure
        plt.tight_layout()
        plt.show()

    def one_signal_detailed(self, signal: pd.Series, signal_back: pd.Series, signal_perturbation: pd.Series,
                            signal_fft: pd.Series, sig_psd: pd.Series, sig_spectrogram: dict):
        """
        plot some figure for potential insight (compare to pre_plot, more information).
        :param signal: measurement signal
        :param signal_back: background signal of the measurement signal
        :param signal_perturbation: perturbation signal of the measurement signal
        :param signal_fft: usually fft of the perturbation signal. the index is frequencies (positive frequencies), and the values are amplitude corresponding to the frequencies.
        :param sig_psd: compare with 'signal_fft', use psd.
        :param sig_spectrogram: include time information (as the x-axis). its keys are 'times', 'frequencies', 'Sxx'.
        :return:
        """
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(4, 2)  # 3 行 2 列的网格
        # 00
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(signal.index, signal.values, color='r', label='signal')
        ax00.plot(signal_back.index, signal_back.values, color='b', label='signal_back')
        ax00.grid(which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.xticks(rotation=45)
        # 01
        ax01 = fig.add_subplot(gs[0, 1])
        ax01.plot(signal.index, signal.values, color='r', label='signal')
        ax01.plot(signal_perturbation.index, signal_perturbation.values, color='b', label='signal_perturbation')
        ax01.grid(which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.xticks(rotation=45)
        # 10
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.plot(signal_back.index, signal_back.values, color='r', label='signal_back')
        ax10.grid(which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.xticks(rotation=45)
        # 11
        ax11 = fig.add_subplot(gs[1, 1])
        ax11.plot(signal_perturbation.index, signal_perturbation.values, color='r', label='signal_perturbation')
        ax11.grid(which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.xticks(rotation=45)
        # 20
        ax20 = fig.add_subplot(gs[2, 0])
        ax20.plot(signal_fft.index, signal_fft.values, color='r', label='signal_perturbation_fft')
        plt.legend()
        plt.yscale('log')
        ax20.grid(which='both', linestyle='--', linewidth=0.5)
        # 21
        ax21 = fig.add_subplot(gs[2, 1])
        ax21.plot(sig_psd.index, sig_psd.values, color='r', label='signal_perturbation_psd')
        plt.legend()
        plt.yscale('log')
        ax21.grid(which='both', linestyle='--', linewidth=0.5)
        # 3
        ax3 = fig.add_subplot(gs[3, :])
        fig.colorbar(ax3.pcolormesh(sig_spectrogram['times'], sig_spectrogram['frequencies'],
                                    10 * np.log10(sig_spectrogram['Sxx']), shading='gouraud'), ax=ax3,
                     label='Power/Frequency (dB/Hz)')
        ax3.set_title("Spectrogram")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Frequency (Hz)")
        ax3.set_ylim(0, 16 / 2)
        # show figure
        plt.tight_layout()
        plt.show()

    def one_signal_stft(self, signal: pd.Series, sampling_rate):
        from scipy.signal import stft
        # Spectrogram for Signal
        frequencies, times, Sxx = stft(signal, fs=sampling_rate)
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times, frequencies, 10 * np.log10(np.abs(Sxx)), shading='gouraud')
        plt.colorbar(label="Power/Frequency (dB/Hz)")
        plt.title("Spectrogram of Signal 1")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.show()

    def double_signals_cross_correlation(self, signal1, signal2, sampling_rate=50):
        # 1. Cross-Correlation
        from scipy.signal import correlate
        cross_corr = correlate(signal1, signal2, mode='full')
        lags = np.arange(-len(signal1) + 1, len(signal2)) / sampling_rate
        plt.figure(figsize=(10, 4))
        plt.plot(lags, cross_corr)
        plt.title('Cross-Correlation between Signals')
        plt.xlabel('Lag [s]')
        plt.ylabel('Correlation')
        plt.show()

    def double_signals_coherence(self, signal1, signal2, sampling_rate=50):
        from scipy.signal import coherence
        # 2. Coherence
        frequencies, coherence_values = coherence(signal1, signal2, fs=sampling_rate)
        plt.figure(figsize=(10, 4))
        plt.semilogy(frequencies, coherence_values)
        plt.title('Coherence between Signals')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Coherence')
        plt.show()

    def double_signals_csd(self, signal1, signal2, sampling_rate=50):
        from scipy.signal import csd
        # 3. Cross-Spectral Density (CSD)
        freqs, csd_values = csd(signal1, signal2, fs=sampling_rate)
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, np.abs(csd_values))
        plt.title('Cross-Spectral Density (CSD) between Signals')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('CSD Magnitude')
        plt.show()

    def double_signals_hilbert_phase_diff(self, signal1, signal2):
        from scipy.signal import hilbert
        assert signal1.index.equals(signal2.index)
        time = signal1.index
        # Compute the analytic signal (for instantaneous phase)
        analytic_signal1 = hilbert(signal1)
        analytic_signal2 = hilbert(signal2)

        # Calculate the instantaneous phase
        phase1 = np.angle(analytic_signal1)
        phase2 = np.angle(analytic_signal2)

        # Compute the phase difference
        phase_difference = phase1 - phase2

        # Plot the phase difference over time
        plt.figure(figsize=(10, 4))
        plt.plot(time, phase_difference, label="Phase Difference (radians)")
        plt.xlabel("Time [s]")
        plt.ylabel("Phase Difference [radians]")
        plt.title("Instantaneous Phase Difference between Signals")
        plt.legend()
        plt.show()

    def double_signals_time_cspd(self, signal1, signal2, sampling_rate=50, magnitude_threshold=0.3):
        from scipy.signal import stft
        # Spectrogram for Signal 1
        frequencies1, times1, Sxx1 = stft(signal1, fs=sampling_rate)
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times1, frequencies1, 10 * np.log10(np.abs(Sxx1)), shading='gouraud')
        plt.colorbar(label="Power/Frequency (dB/Hz)")
        plt.title("Spectrogram of Signal 1")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.show()

        # Spectrogram for Signal 2
        frequencies2, times2, Sxx2 = stft(signal2, fs=sampling_rate)
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times2, frequencies2, 10 * np.log10(np.abs(Sxx2)), shading='gouraud')
        plt.colorbar(label="Power/Frequency (dB/Hz)")
        plt.title("Spectrogram of Signal 2")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.show()

        # Compute the cross-spectrogram by taking the product of the two spectrograms
        cross_spectrogram = Sxx1 * np.conj(Sxx2)
        # Plot the magnitude of the cross-spectrogram
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(times1, frequencies1, np.log10(np.abs(cross_spectrogram)), shading='gouraud')
        plt.colorbar(label='Cross-Power')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Cross-Spectrogram between Signal 1 and Signal 2')
        # plt.ylim(0, 100)  # Limiting frequency for clearer view (adjust as needed)
        plt.show()

        # Calculate phase difference
        phase_difference = np.angle(cross_spectrogram)
        # Set a threshold for the cross-spectrogram magnitude
        # magnitude_threshold = 0.3  # Adjust this threshold based on your signal characteristics
        magnitude = np.abs(cross_spectrogram)
        # Mask the phase difference where the magnitude is below the threshold
        phase_difference_masked = np.where(magnitude > magnitude_threshold, phase_difference, np.nan)

        # Plot the masked phase difference
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(times1, frequencies1, np.degrees(phase_difference_masked), shading='gouraud', cmap='twilight')
        plt.colorbar(label='Phase Difference (radians)')
        plt.ylim([0, 5])
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Phase Difference (only where Cross-Spectrogram > {magnitude_threshold})')
        plt.show()


# plotly
# interactive
def px_1f_2curve(x, y1, y2, title=''):
    # todo:: add code for real-time display of adjustment parameters
    df = pd.DataFrame({'x': x, 'y1': y1, 'y2': y2})
    fig = px.line(df, x='x', y=['y1', 'y2'], title=title)
    fig.show()


def px_beta(beta, me, mi):
    df = pd.DataFrame({'x': beta.index, 'y': beta.values})
    fig = px.line(df, x='x', y='y', title='Plasma beta')
    # add specific value horizontal line
    specific_v = me / mi
    fig.add_shape(type="line", x0=df['x'].min(), y0=specific_v, x1=df['x'].max(), y1=specific_v,
                  line=dict(color="Red", width=2, dash="dash"),  # type of line
                  )
    # add annotation
    fig.add_annotation(x=df['x'].max(), y=specific_v, text=f"$m_e/m_i$={specific_v}", showarrow=False, yshift=10)
    # title
    fig.update_layout(xaxis_title="Time (UT)", yaxis_title="$\\Beta$", )
    fig.show()

# cross product
# np.cross(v.values, B.values)
