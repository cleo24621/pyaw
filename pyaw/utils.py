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
from scipy.signal import welch

from paras import mu0, me, mp, e, kB


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
def baseline_correct(series_: pd.Series, baseline: pd.Series) -> pd.Series:
    """
    :param series_: be baseline corrected
    :param baseline: generally be the rolling average of the series_
    :return:
    """
    return series_ - baseline


def move_average(series_, window, draw=True, savefig=False):
    """
    :param series_:
    :param window:
    :param draw:
    :param savefig:
    :return:
    """
    series_mov_ave = series_.rolling(window=window).mean()
    # figure: before and after moving average comparison
    if draw:
        plt.figure()
        plt.plot(series_.index, series_, label=series_.name)
        plt.plot(series_mov_ave.index, series_mov_ave, label=series_mov_ave.name)
        plt.legend()
        plt.xlabel('Time (UTC)')
        plt.ylabel('B (nT)')
        plt.title('B-component before and after moving average comparison')
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


def fft_(series_: pd.Series, fs: float, if_plot: bool = False, figsize=(10, 6), title=''):
    """

    :param series_: [pd.series]. the type of index is pd.datetime.
    :param fs: [float]. sampling rate.
    :return: todo:: return type (i want pd.Series)???
    """
    # make sure that the series doesn't have "nan" values
    if series_.isna().any():
        compare_before_after_interpolate(series_, method_='linear')
        series_ = series_.interpolate(method='linear')
    n = len(series_)
    f_values = np.fft.fftfreq(n, d=1 / fs)  # frequencies
    fft_values = np.fft.fft(series_)  # fft
    amplitude_spectrum = np.abs(fft_values)  # magnitude of fft
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
        plt.title(title)
        plt.legend()
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.title(f'{title}: (fs={fs})')
        plt.show()
    return positive_frequencies, positive_amplitude_spectrum


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
