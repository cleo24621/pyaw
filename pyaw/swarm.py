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
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from viresclient import SwarmRequest

from pyaw import utils_preprocess, configs, utils_spectral


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
        "Ehx", "Ehy", "Ehz", "Evx", "Evy", "Evz",
        # Ion drift corotation signal, removed from ion drift & electric field
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
    todo: add test
    want to get 1d data of vfm50, because the data is too large, directly use "save_SW_MAGx_HR_1B()" to get 1d data will raise error, so use this to get the data for several hours.
    :param start: '20240812T000000' format
    :param num_elements: the whole time range (hour). for example, 24 means 24 hours.
    :return:
    """
    start = pd.to_datetime(start, format='%Y%m%dT%H%M%S')
    return [(start + timedelta(hours=i)).strftime("%Y%m%dT%H%M%S") for i in range(num_elements + 1)]


class Swarm:
    def __init__(self, fp: str, payload: str, start: str = configs.swarm_start, end: str = configs.swarm_end,
                 handle_outliers: Optional[bool] = True, std_times: Optional[float] = None,
                 if_mv: Optional[bool] = None, window_sec: Optional[float] = None, threshold: Optional[float] = None):
        assert payload in ['efi16', 'vfm50'], "payload should be 'efi16' or 'vfm50'"
        self.fs_efi16 = 16
        self.fs_vfm50 = 50
        self.start = start
        self.end = end
        self.fp = fp
        self.df = pd.read_pickle(self.fp)
        self.df = self.df.loc[pd.to_datetime(self.start) - pd.Timedelta(20,'s'):pd.to_datetime(self.end) + pd.Timedelta(20,'s')]  # 解决滑动平均数据点个数小于窗口所需数据点个数的情况
        self.df = self.df.rename_axis('datetime')
        self.payload = payload
        self.handle_outliers = handle_outliers
        self.std_times = std_times
        self.if_mv = if_mv
        self.window_sec = window_sec
        self.threshold = threshold
        if self.payload == 'efi16':
            if self.std_times is None:  # todo: improve
                self.std_times = 1.0
            if self.if_mv is None:
                self.if_mv = True
            if self.window_sec is None:
                self.window_sec = 20.0
            self.df = self.preprocess_efi16(self.std_times, self.if_mv, self.window_sec)
        else:
            if self.threshold is None:
                self.threshold = 1000.0
            if self.if_mv is None:
                self.if_mv = True
            if self.window_sec is None:
                self.window_sec = 20.0
            self.df = self.preprocess_vfm50(self.threshold, self.if_mv, self.window_sec)

    def preprocess_efi16(self, std_times, if_mv, window_sec):
        """
        the electric field component of swarm should handle outliers (will do 1e1 effect to the ratio analysis).
        :param fp:
        :param start:
        :param end:
        :param handle_outliers: default use std method, not use diff method!
        :param threshold:
        :return:
        """
        # 坐标变换需要用到的角度
        theta = np.arccos(
            self.df['VsatN'] ** 2 / (np.abs(self.df['VsatN']) * np.sqrt(self.df['VsatN'] ** 2 + self.df['VsatE'] ** 2)))
        # todo: without quality control to Ehx, Ehy; without outliers delete. (they will cancel each other?)
        # todo: all columns may include outliers?
        self.df['eh_sc1'] = self.df['Ehx']
        self.df['eh_sc2'] = self.df['Ehy']
        if self.handle_outliers:
            print("self.df['eh_sc1'] set nan:")
            self.df['eh_sc1'] = utils_preprocess.set_bursts_nan_std(self.df['eh_sc1'], std_times=std_times, print_=True)
            print("\nself.df['eh_sc2'] set nan:")
            self.df['eh_sc2'] = utils_preprocess.set_bursts_nan_std(self.df['eh_sc2'], std_times=std_times, print_=True)
            print("\nfill nan value with linear, bfill, ffill in order")
            # 1
            # Identify NaN positions
            nan_positions = self.df['eh_sc1'].isna()
            # interpolate
            self.df['eh_sc1_interpolated'] = self.df['eh_sc1'].interpolate(method='linear').bfill().ffill()
            # Display detailed information about interpolated values
            interpolated_values = self.df.loc[nan_positions, ['eh_sc1', 'eh_sc1_interpolated']]
            print("Interpolated values at NaN positions:\n", interpolated_values)
            self.df['eh_sc1'] = self.df['eh_sc1_interpolated']
            # 2
            # Identify NaN positions
            nan_positions = self.df['eh_sc2'].isna()
            # interpolate
            self.df['eh_sc2_interpolated'] = self.df['eh_sc2'].interpolate(method='linear').bfill().ffill()
            # Display detailed information about interpolated values
            interpolated_values = self.df.loc[nan_positions, ['eh_sc2', 'eh_sc2_interpolated']]
            print("Interpolated values at NaN positions:\n", interpolated_values)
            self.df['eh_sc2'] = self.df['eh_sc2_interpolated']
            # drop
            self.df.drop(columns=['eh_sc1_interpolated', 'eh_sc2_interpolated'], inplace=True)
            print("complete fill nan")
        # 坐标变换
        self.df['eh_enu1'] = self.df['eh_sc1'] * np.sin(theta) + self.df['eh_sc2'] * np.cos(theta)
        self.df['eh_enu2'] = self.df['eh_sc1'] * np.cos(theta) - self.df['eh_sc2'] * np.sin(theta)
        if if_mv:
            print("\nget background electric field and disturb electric field using moving average:")
            self.df['eh0_enu1'] = utils_preprocess.move_average(self.df['eh_enu1'], window=int(16 * window_sec),
                                                                draw=False)
            self.df['eh0_enu2'] = utils_preprocess.move_average(self.df['eh_enu2'], window=int(16 * window_sec),
                                                                draw=False)
            self.df['eh1_enu1'] = self.df['eh_enu1'] - self.df['eh0_enu1']
            self.df['eh1_enu2'] = self.df['eh_enu2'] - self.df['eh0_enu2']
            print("complete get")
        return self.df  # todo: no return?

    def preprocess_vfm50(self, threshold: float, if_mv: bool, window_sec: float, if_igrf: bool = False):
        """
        without fill nan procedure
        :param fp:
        :param start:
        :param end:
        :param handle_outliers: default use diff method, not use std method!
        :param threshold:
        :return:
        """
        b_enu1 = []
        b_enu2 = []
        b_enu3 = []
        for B_NEC in self.df['B_NEC']:
            b_enu1.append(B_NEC[1])
            b_enu2.append(B_NEC[0])
            b_enu3.append(-B_NEC[2])
        self.df['b_enu1'] = b_enu1
        self.df['b_enu2'] = b_enu2
        self.df['b_enu3'] = b_enu3
        if self.handle_outliers:
            print("self.df['b_enu1'] set nan:")
            self.df['b_enu1'] = utils_preprocess.set_bursts_nan_diff(self.df['b_enu1'], threshold=threshold,
                                                                     print_=True)
            print("\nself.df['b_enu2'] set nan:")
            self.df['b_enu2'] = utils_preprocess.set_bursts_nan_diff(self.df['b_enu2'], threshold=threshold,
                                                                     print_=True)  # todo: add interpolate nan section
        if if_mv:
            print("\nget background magnetic field and disturb magnetic field using moving average:")
            self.df['b0_enu1'] = utils_preprocess.move_average(self.df['b_enu1'], window=int(50 * window_sec),
                                                               draw=False)
            self.df['b0_enu2'] = utils_preprocess.move_average(self.df['b_enu2'], window=int(50 * window_sec),
                                                               draw=False)
            self.df['b1_enu1'] = self.df['b_enu1'] - self.df['b0_enu1']
            self.df['b1_enu2'] = self.df['b_enu2'] - self.df['b0_enu2']
            print("complete get")
        if if_igrf:
            pass  # todo: get the igrf magnetic field of swarm orbit
        return self.df


# class SwarmSingleSignalPlot:
#     def __init__(self,fp_e:Optional[str]=None,fp_b:Optional[str]=None):
#         assert fp_e is not None or fp_b is not None, "fp_e and fp_b should not be None at the same time"
#         if fp_e is not None:
#             swarm_e = Swarm(fp_e, 'efi16')
#         if fp_b is not None:
#             swarm_b = Swarm(fp_b, 'vfm50')
#
#     def plot_e1_b1_baselined(self,figsize=(10,4)):
#         plt.figure(figsize=figsize)
#         x =
#         plt.plot()

def figure_baselined(signal1: pd.Series, signal2: pd.Series, label1='e (mV/m)', label2='b (nT)', figsize=(10, 4),
                     ylabel='e and b baseliend'):
    x1 = signal1.index
    x2 = signal2.index
    y1 = signal1.values
    y2 = signal2.values
    plt.figure(figsize=figsize)
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.legend()
    plt.xlabel('UT Time [s]')
    plt.ylabel(ylabel)
    plt.title(f"{label1} and {label2}")
    plt.show()
    return None


def figure_filter(signal1: pd.Series, signal2: pd.Series, lowcut=0.2, highcut=4.0, label1='e (mV/m)', label2='b (nT)',
                  figsize=(10, 4), ylabel='e and b filtered (0.2-4 Hz)'):
    band_filter_1 = utils_preprocess.LHBFilter(signal1, fs=16, lowcut=lowcut, highcut=highcut)
    band_filter_2 = utils_preprocess.LHBFilter(signal2, fs=50, lowcut=lowcut, highcut=highcut)
    signal1 = pd.Series(index=signal1.index, data=band_filter_1.apply_filter())
    signal2 = pd.Series(index=signal2.index, data=band_filter_2.apply_filter())
    x1 = signal1.index
    x2 = signal2.index
    y1 = signal1.values
    y2 = signal2.values
    plt.figure(figsize=figsize)
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.legend()
    plt.xlabel('UT Time [s]')
    plt.ylabel(ylabel)
    plt.title(f"{label1} and {label2}")
    plt.show()
    return None


def figure_spectrogram(signal, fs):
    spectrogram = utils_spectral.Spectrogram(signal, fs)
    spectrogram.plot_spectrogram()
    return None


def figure_csd_module(signal1, signal2, fs):
    timecsd = utils_spectral.TimeCSD(signal1, signal2, fs)
    timecsd.plot_module()
    return None


def figure_csd_phase(signal1, signal2, fs):
    timecsd = utils_spectral.TimeCSD(signal1, signal2, fs)
    timecsd.plot_phase()
    return None


def figure_csd_phase_hist_counts(signal1, signal2, fs):
    timecsd = utils_spectral.TimeCSD(signal1, signal2, fs)
    timecsd.plot_phase_hist_counts()
    return None


def figure_psd(signal1, signal2, fs, figsize=(5, 5)):
    assert all(signal1.index == signal2.index), "signal1 and signal2 must have the same index"
    psd1 = utils_spectral.PSD(signal1, fs)
    psd2 = utils_spectral.PSD(signal2, fs)
    freqs1, Pxx1 = psd1.get_psd()
    freqs2, Pxx2 = psd2.get_psd()
    assert all(freqs1 == freqs2), "freqs1 and freqs2 must be equal"
    x = freqs1
    y1 = np.sqrt(Pxx1)
    y2 = np.sqrt(Pxx2)
    plt.figure(figsize=figsize)
    plt.plot(x, y1, label='e')
    plt.plot(x, y2, label='b')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('PSD of e and b')
    plt.show()
    return None


def figure_ratio(signal1, signal2, fs, figsize=(5, 5)):
    psd1 = utils_spectral.PSD(signal1, fs)
    psd2 = utils_spectral.PSD(signal2, fs)
    freqs1, Pxx1 = psd1.get_psd()
    freqs2, Pxx2 = psd2.get_psd()
    assert all(freqs1 == freqs2), "freqs1 and freqs2 must be equal"
    x = freqs1
    y = np.sqrt(Pxx1 / Pxx2) * 1e6
    plt.figure(figsize=figsize)
    plt.plot(x, y, label='e/b')
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('e/b ratio')
    plt.show()
    return None


def figure_phase(signal1, signal2, figsize=(10, 5)):
    cwt = utils_spectral.CWT(signal1, signal2)
    cwt.plot_phase_hist_counts(figsize=figsize)
    return None


def main(compo='12'):
    assert compo in ['12', '21'], "compo should be '12' or '21'"
    fp_e = r"\\Diskstation1\file_three\aw\swarm\A\efi16\sw_efi16A_20160311T000000_20160311T235959_0.pkl"
    fp_b = r"\\Diskstation1\file_three\aw\swarm\A\vfm50\sw_vfm50A_20160311T060000_20160311T070000_0.pkl"
    swarm_e = Swarm(fp_e, 'efi16')
    swarm_b = Swarm(fp_b, 'vfm50')
    df_e = swarm_e.df
    df_b = swarm_b.df
    if compo == '21':
        e = df_e['eh1_enu2']
        b = df_b['b1_enu1']
    else:
        e = df_e['eh1_enu1']
        b = df_b['b1_enu2']
    # plot figure baselined
    signal1 = e - e.mean()
    signal2 = b - b.mean()
    figure_baselined(signal1, signal2)
    # plot figure filter, spectrogram
    signal1 = e
    signal2 = b
    figure_filter(signal1, signal2)
    spectrogram_e = utils_spectral.Spectrogram(signal1, 16)
    spectrogram_e.plot_spectrogram()
    spectrogram_b = utils_spectral.Spectrogram(signal2, 50)
    spectrogram_b.plot_spectrogram()
    # plot csd_module, phase
    signal1 = e
    signal2 = b
    signal2 = utils_preprocess.align_high2low(signal2, signal1)
    figure_csd_module(signal1, signal2, 16)
    figure_csd_phase(signal1, signal2, 16)
    # static
    # plot filter
    signal1 = e
    signal2 = b
    start = pd.to_datetime('20160311T064705')
    end = pd.to_datetime('20160311T064725')
    signal1 = signal1.loc[start:end]
    signal2 = signal2.loc[start:end]
    figure_filter(signal1, signal2)
    # plot psd, ratio, cwt_phase_hist_counts
    signal1 = e
    signal2 = b
    signal1 = signal1.loc[start:end]
    signal2 = signal2.loc[start:end]
    signal2 = utils_preprocess.align_high2low(signal2, signal1)
    figure_psd(signal1, signal2, 16)
    figure_ratio(signal1, signal2, 16)
    # plot
    figure_phase(signal1, signal2)
    # active
    # plot filter
    signal1 = e
    signal2 = b
    start = pd.to_datetime('20160311T064700')
    # start = pd.to_datetime('20160311T064735')
    end = pd.to_datetime('20160311T064900')
    # end = pd.to_datetime('20160311T064755')
    signal1 = signal1.loc[start:end]
    signal2 = signal2.loc[start:end]  # aligned
    figure_filter(signal1, signal2)
    # plot psd, ratio, cwt_phase_hist_counts
    signal1 = e
    signal2 = b
    signal1 = signal1.loc[start:end]
    signal2 = signal2.loc[start:end]
    signal2 = utils_preprocess.align_high2low(signal2, signal1)
    figure_psd(signal1, signal2, 16)
    figure_ratio(signal1, signal2, 16)
    # plot
    figure_phase(signal1, signal2)
    cwt = utils_spectral.CWT(signal1, signal2,sampling_period=1/16)
    cwt.plot_module()


if __name__ == "__main__":
    main()
