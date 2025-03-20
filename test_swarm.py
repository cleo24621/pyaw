# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/29
@DESCRIPTION: 
"""
import pandas as pd

from pyaw import utils_cal, configs, swarm

fp_e = r"\\Diskstation1\file_three\aw\swarm\A\efi16\sw_efi16A_20160311T000000_20160311T235959_0.pkl"
fp_b = r"\\Diskstation1\file_three\aw\swarm\A\vfm50\sw_vfm50A_20160311T060000_20160311T070000_0.pkl"
start = '20160311T064700'
end = '20160311T064900'

df_e = swarm.pre_e(fp_e,start,end,handle_outliers=True)
df_b = swarm.pre_b(fp_b,start,end,handle_outliers=True)

df_e['timestamp'] = df_e.index.astype('int64')
df_b['timestamp'] = df_b.index.astype('int64')

from scipy.interpolate import interp1d
interp_func1 = interp1d(df_b['timestamp'], df_b['b1_enu1'], kind='linear', fill_value="extrapolate")
interp_func2 = interp1d(df_b['timestamp'], df_b['b1_enu2'], kind='linear', fill_value="extrapolate")
df_e['b1_enu1_interp'] = interp_func1(df_e['timestamp'])
df_e['b1_enu2_interp'] = interp_func2(df_e['timestamp'])

start = pd.to_datetime('2016-03-11 06:47:35')
end = pd.to_datetime('2016-03-11 06:47:55')
df_e_clip = df_e.loc[start:end]
df_b_clip = df_b.loc[start:end]

plotsignal = utils.PlotSignal()
plotsignal.double_signals_time_cspd(df_e_clip['eh1_enu1'],df_e_clip['b1_enu2_interp'],sampling_rate=16)