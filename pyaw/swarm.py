# -*- coding: utf-8 -*-
"""
@File        : swarm.py
@Author      : cleo
@Date        : 2024/9/23 9:55
@Project     : PyWave
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2024, cleo
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : cleo
@Last Modified Date: 2024/9/23 9:55
"""

import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt, welch
from viresclient import SwarmRequest

from pyaw import utils


def save_SW_EFIx_TCT16(start, end, satellite='A'):
    """
    can get 1d data
    :param start: '20240812T000000' format
    :param end: same as 'start'
    :param satellite: 'A', 'B' or 'C'
    :return:
    """
    # get data
    tct_vars = [  # Satellite velocity in NEC frame
        "VsatC", "VsatE", "VsatN",  # Geomagnetic field components derived from 1Hz product
        #  (in satellite-track coordinates)
        "Bx", "By", "Bz",  # Electric field components derived from -VxB with along-track ion drift
        #  (in satellite-track coordinates)
        # Eh: derived from horizontal sensor
        # Ev: derived from vertical sensor
        "Ehx", "Ehy", "Ehz", "Evx", "Evy", "Evz", # Ion drift corotation signal, removed from ion drift & electric field
        #  (in satellite-track coordinates)
        "Vicrx", "Vicry", "Vicrz",  # Ion drifts along-track from vertical (..v) and horizontal (..h) TII sensor
        "Vixv", "Vixh",  # Ion drifts cross-track (y from horizontal sensor, z from vertical sensor)
        #  (in satellite-track coordinates)
        "Viy", "Viz",  # Random error estimates for the above
        #  (Negative value indicates no estimate available)
        "Vixv_error", "Vixh_error", "Viy_error", "Viz_error",  # Quasi-dipole magnetic latitude and local time
        #  redundant with VirES auxiliaries, QDLat & MLT
        "Latitude_QD", "MLT_QD",  # Refer to release notes link above for details:
        "Calibration_flags", "Quality_flags", ]
    request = SwarmRequest()
    request.set_collection(f"SW_EXPT_EFI{satellite}_TCT16")
    request.set_products(measurements=tct_vars)
    data = request.get_between(start, end)
    df = data.as_dataframe()
    # save
    sdir = rf"V:\aw\swarm\{satellite}\efi16"
    sfn = f'sw_efi16{satellite}_{start}_{end}_0.pkl'
    os.makedirs(sdir, exist_ok=True)  # dir exit doesn't raise error
    sfp = os.path.join(sdir, sfn)
    if os.path.isfile(sfp):
        print(f"{sfn} already exists, skip save.")
        return None
    else:
        df.to_pickle(sfp)
        print(f"{sfn} saved")
        return None


def save_SW_MAGx_HR_1B(start, end, satellite='A'):
    """
    usually get 1h data, failed to get 1d data. So combined with `get_time_strs_forB()` to get 1d data.
    :param start: '20240812T000000' format
    :param end: same as 'start'
    :param satellite: 'A', 'B' or 'C'
    :return:
    """
    # get data
    request = SwarmRequest()
    request.set_collection(f"SW_OPER_MAG{satellite}_HR_1B")
    request.set_products(measurements=["B_NEC"], )
    data = request.get_between(start_time=start, end_time=end, asynchronous=False)
    df = data.as_dataframe()
    # save
    sdir = rf"V:\aw\swarm\{satellite}\vfm50"
    sfn = f'sw_vfm50{satellite}_{start}_{end}_0.pkl'
    os.makedirs(sdir, exist_ok=True)  # dir exit doesn't raise error
    sfp = os.path.join(sdir, sfn)
    if os.path.isfile(sfp):
        print(f"{sfn} already exists, skip save.")
        return None
    else:
        df.to_pickle(sfp)
        print(f"{sfn} saved")
        return None


def get_time_strs_forB(start: str, num_elements: int) -> list:
    """
    want to get 1d data of vfm50, because the data is too large, directly use "save_SW_MAGx_HR_1B()" to get 1d data will raise error, so use this to get the data for several hours.
    :param start: '20240812T000000' format
    :param num_elements: the whole time range (hour). for example, 24 means 24 hours.
    :return:
    """
    start = pd.to_datetime(start, format='%Y%m%dT%H%M%S')
    return [(start + timedelta(hours=i)).strftime("%Y%m%dT%H%M%S") for i in range(num_elements + 1)]


def pre_e(fp, start, end, handle_outliers=True, threshold=40):
    df = pd.read_pickle(fp)
    df = df.loc[start:end]
    theta = np.arccos(df['VsatN'] ** 2 / (np.abs(df['VsatN']) * np.sqrt(
        df['VsatN'] ** 2 + df['VsatE'] ** 2)))  # todo: all columns may include outliers?
    # without quality control to Ehx, Ehy; without outliers delete. (they will cancel each other?)
    df['eh_sc1'] = df['Ehx']
    df['eh_sc2'] = df['Ehy']
    if handle_outliers:
        df['eh_sc1'] = utils.set_outliers_nan(df['eh_sc1'], threshold=threshold)
        df['eh_sc2'] = utils.set_outliers_nan(df['eh_sc2'], threshold=threshold)
    df['Ehn'] = df['eh_sc1'] * np.cos(theta) - df['eh_sc2'] * np.sin(theta)
    df['Ehe'] = df['eh_sc1'] * np.sin(theta) + df['eh_sc2'] * np.cos(theta)
    df.rename(columns={'Ehn': 'eh_enu2', 'Ehe': 'eh_enu1'}, inplace=True)
    # df['Ehn_mv'] = utils.move_average(df['Ehn'], window=20 * configs.fs_efi16, draw=False)
    # df['Ehe_mv'] = utils.move_average(df['Ehe'], window=20 * configs.fs_efi16, draw=False)
    return df


def pre_b(fp, start, end, handle_outliers=True, threshold=1000):
    df = pd.read_pickle(fp)
    df = df.loc[start:end]
    b_enu1 = []
    b_enu2 = []
    b_enu3 = []
    for B_NEC in df['B_NEC']:
        b_enu1.append(B_NEC[1])
        b_enu2.append(B_NEC[0])
        b_enu3.append(-B_NEC[2])
    df['b_enu1'] = b_enu1
    df['b_enu2'] = b_enu2
    df['b_enu3'] = b_enu3
    if handle_outliers:
        df['b_enu1'] = utils.set_outliers_nan(df['b_enu1'], threshold=threshold)
        df['b_enu2'] = utils.set_outliers_nan(df['b_enu2'], threshold=threshold)
    return df


class Swarm:
    def __init__(self):
        self.fs_efi16 = 16
        self.fs_vfm50 = 50

    def get_BE(self, fp):
        # BE
        data_B = xr.open_dataset(fp)
        BE = data_B['B_NEC'][:, 1]
        BE = BE.to_dataframe()
        BE.drop(columns=['NEC'], inplace=True)
        BE.rename(columns={'B_NEC': 'E'}, inplace=True)
        BE = BE['E']
        BE = BE['2016-03-11 06:46:40':'2016-03-11 06:48:59']
        self.BE = BE

    def get_BN(self, fp):
        # BN
        data_B = xr.open_dataset(fp)
        BN = data_B['B_NEC'][:, 0]
        BN = BN.to_dataframe()
        BN.drop(columns=['NEC'], inplace=True)
        BN.rename(columns={'B_NEC': 'N'}, inplace=True)
        BN = BN['N']
        BN = BN['2016-03-11 06:46:40':'2016-03-11 06:48:59']
        self.BN = BN

    def get_E_NE(self, fp, if_draw=True):
        data_E = xr.open_dataset(fp)
        data_E = data_E.sel(Timestamp=slice('2016-03-11 06:46:40', '2016-03-11 06:48:59'))
        VsatN = data_E['VsatN']
        VsatE = data_E['VsatE']
        VsatC = data_E['VsatC']
        Ex = data_E['Ehx']
        Ey = data_E['Ehy']
        Ex = Ex.to_dataframe()
        Ey = Ey.to_dataframe()
        Ex = Ex['Ehx']
        Ey = Ey['Ehy']

        def set_outliers_nan(series):
            series_scores = (series - series.mean()) / series.std()
            # 设置阈值，通常 Z 分数大于 3 或小于 -3 的点可以认为是异常点，超过的设置为nan
            threshold = 2
            series[np.abs(series_scores) > threshold] = np.nan
            return series

        Ex = set_outliers_nan(Ex)
        Ey = set_outliers_nan(Ey)
        if if_draw:
            plt.figure(1)
            plt.plot(Ex.index, Ex, label='Ex')
            plt.plot(Ey.index, Ey, label='Ey')
            plt.legend()
            plt.xlabel('Time (UTC)')
            plt.ylabel('E (mV/m)')
        VsatN = VsatN.to_dataframe()
        VsatE = VsatE.to_dataframe()
        VsatC = VsatC.to_dataframe()
        VsatN = VsatN['VsatN']
        VsatE = VsatE['VsatE']
        VsatC = VsatC['VsatC']
        # sc to nec
        theta = np.arccos(VsatN ** 2 / (np.abs(VsatN) * np.sqrt(VsatN ** 2 + VsatE ** 2)))
        EN = Ex * np.cos(theta) - Ey * np.sin(theta)
        EE = Ey * np.cos(theta) + Ex * np.sin(theta)
        if if_draw:
            plt.figure(2)
            plt.plot(EN.index, EN, label='EN')
            plt.plot(EE.index, EE, label='EE')
            plt.legend()
            plt.xlabel('Time (UTC)')
            plt.ylabel('E (mV/m)')
            plt.title('delete outliers')
        return EN, EE

    def B_mov_ave(self, series, window, draw=True, savefig=False):
        series_mov_ave = series.rolling(window=window).mean()
        # figure: before and after moving average comparison
        if draw:
            plt.figure()
            plt.plot(series.index, series, label=series.name)
            plt.plot(series_mov_ave.index, series_mov_ave, label=series_mov_ave.name)
            plt.legend()
            plt.xlabel('Time (UTC)')
            plt.ylabel('B (nT)')
            plt.title('B-component before and after moving average comparison')
            if savefig:
                plt.savefig(f'before and after moving average comparison')
        return series_mov_ave

    def E_mov_ave(self, series, window, draw=True, savefig=False):
        series = series.interpolate(method='linear').bfill().ffill()
        series_mov_ave = series.rolling(window=window).mean()
        # figure: before and after moving average comparison
        if draw:
            plt.figure()
            plt.plot(series.index, series, label=series.name)
            plt.plot(series_mov_ave.index, series_mov_ave, label=series_mov_ave.name)
            plt.legend()
            plt.xlabel('Time (UTC)')
            plt.ylabel('E (mV/m)')
            plt.title('E-component before and after moving average comparison')
            if savefig:
                plt.savefig(f'before and after moving average comparison')

    def baselined_filter(self, B, B_mov_ave, E, E_mov_ave, if_draw=True):
        b = B - B_mov_ave
        b = b['2016-03-11 06:47:00':]
        if if_draw:
            plt.figure(1)
            plt.plot(b.index, b, label='before')
            plt.plot(b.index, b - b.mean(), label='after')
            # plt.legend()
            plt.xlabel('Time (UTC)')
            plt.ylabel('\\Delta B and \\Delta B after Baselined (nT)')
            plt.title(
                'B-component perturbation before and after baseline comparison')  # plt.savefig(f'b_{B_c_s} before and after baseline comparison')  # plt.close()
        e = E - E_mov_ave
        e = e['2016-03-11 06:47:00':]
        if if_draw:
            plt.figure(2)
            plt.plot(e.index, e, label='before')
            plt.plot(e.index, e - e.mean(), label='after')
            plt.xlabel('Time (UTC)')
            plt.ylabel('e (mV/m)')
            plt.title('E-component perturbation before and after baseline comparison')
        if if_draw:
            plt.figure(3)
            plt.plot(b.index, b - b.mean(), label='b')
            plt.plot(e.index, e - e.mean(), label='e')
            plt.xlabel('Time (UTC)')
            plt.ylabel('b and e')

        # filter
        # 1. 设计带通滤波器
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs  # 计算 Nyquist 频率
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype="band")
            return b, a

        # 2. 应用带通滤波器
        def bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = filtfilt(b, a, data)
            return y

        # 设置滤波参数
        lowcut = 0.2  # 带通滤波器下限频率
        highcut = 4.0  # 带通滤波器上限频率
        e_filter = bandpass_filter(e, lowcut, highcut, fs=self.fs_efi16, order=5)
        e_filter = pd.Series(e_filter, index=e.index)
        b_filter = bandpass_filter(b, lowcut, highcut, fs=self.fs_vfm50, order=5)
        b_filter = pd.Series(b_filter, index=b.index)
        if if_draw:
            plt.figure(4)
            plt.plot(e.index, e_filter, color='r', label=r'$E_{North}$ (mV/m)')
            plt.plot(b.index, b_filter, color='b', label=r'$B_{East}$ (nT)')
            plt.legend()
            plt.show()

    def fre_psd(ndarray, sample_rate, nperseg, window='hammind'):
        frequencies, psd = welch(ndarray, fs=sample_rate, nperseg=nperseg, noverlap=nperseg / 2, window=window)
        # 将 PSD 转换为振幅谱 (单边谱)
        # amp_spectrum = np.sqrt(2 * psd)
        # double
        # amp_spectrum = np.sqrt(len(ndarray) * psd)

        # # 绘制PSD和振幅谱的图像
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.semilogy(frequencies, psd)
        # plt.title("Power Spectral Density (PSD)")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("PSD (V^2/Hz)")
        # plt.grid()
        #
        # plt.subplot(2, 1, 2)
        # plt.semilogy(frequencies, amp_spectrum)
        # plt.title("Amplitude Spectrum (Amp)")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Amplitude (V)")
        # plt.grid()
        #
        # plt.tight_layout()
        # plt.show()

        # 绘制频谱
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, psd)
        plt.xlim((0.2, 8))
        plt.yscale('log')  # 幅值对数刻度
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (V^2/Hz)")
        plt.title("Power Spectral Density (PSD) using Welch's method")
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.show()

        # return
        # return frequencies, psd, amp_spectrum
        return frequencies, psd


def main():
    for i in get_time_strs_forB('20240812T000000', 24):
        print(i, type(i))


if __name__ == "__main__":
    main()
