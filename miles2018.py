#%%
#
# st = '20160311T064700'
# et = '20160311T064900'
# orbit_number = 12885
# st_clip = '20160311T064640'
# et_clip = '20160311T064920'
#
# request.set_collection('SW_OPER_MAGA_HR_1B')
# request.set_products(measurements=request.available_measurements(),
#                      models=['IGRF'],
#                      auxiliaries=request.available_auxiliaries())
# data = request.get_between

#%% file 电磁
import pandas as pd
fps = ['./SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl',
       './aux_SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl',
       './SW_EXPT_EFIA_TCT16_12885_20160311T061733_20160311T075106.pkl']
df_b = pd.read_pickle(fps[0])
df_b_aux = pd.read_pickle(fps[1])
df_e = pd.read_pickle(fps[2])

#%% 选取列，选取时间范围
st_clip = '20160311T064640'
et_clip = '20160311T064920'
df_b_clip = df_b[['B_NEC','Longitude','Latitude','Radius']]
df_b_clip = df_b_clip.loc[pd.Timestamp(st_clip):pd.Timestamp(et_clip)]
df_b_aux_clip = df_b_aux[['QDLat','QDLon','MLT']]
df_b_aux_clip = df_b_aux_clip.loc[pd.Timestamp(st_clip):pd.Timestamp(et_clip)]
df_e_clip = df_e[['Longitude','Latitude','Radius','VsatE','VsatN','Ehy','Ehx']]
df_e_clip = df_e_clip.loc[pd.Timestamp(st_clip):pd.Timestamp(et_clip)]


#%% 处理异常值
from pyaw import utils

ehx,ehy = df_e_clip['Ehx'].values,df_e_clip['Ehy'].values
ehx,ehy = utils.set_outliers_nan_std(ehx),utils.set_outliers_nan_std(ehy)
ehx,ehy = utils.get_array_interpolated(df_e_clip.index.values,ehx),utils.get_array_interpolated(df_e_clip.index.values,ehy)

#%% 坐标变换
rotation_matrix_2d_nec2sc, rotation_matrix_2d_sc2nec = utils.get_rotmat_nec2sc_sc2nec(df_e_clip['VsatN'].values, df_e_clip['VsatE'].values)
ehn, ehe = utils.do_rotation(-ehx, -ehy,rotation_matrix_2d_sc2nec)

#%% 滑动平均
import numpy as np
fs_e = 16
fs_b = 50

mv_window_seconds = 20  # int
window_e = fs_e * mv_window_seconds
ehn_mv = utils.move_average(ehn, window_e)
ehn1 = ehn - ehn_mv

window_b = fs_b * mv_window_seconds
_,be,_ = utils.get_3arrs(df_b_clip['B_NEC'])
be_mv = utils.move_average(be, window_b)
be1 = be - be_mv

datetimes_e = df_e_clip.index.values
datetimes_b = df_b_clip.index.values
st = '20160311T064700'
et = '20160311T064900'
mask_e = np.where((datetimes_e >= utils.convert_tstr2dt64(st)) & (datetimes_e <= utils.convert_tstr2dt64(et)))
mask_b = np.where((datetimes_b >= utils.convert_tstr2dt64(st)) & (datetimes_b <= utils.convert_tstr2dt64(et)))
datetimes_e = datetimes_e[mask_e]
datetimes_b = datetimes_b[mask_b]
ehn1 = ehn1[mask_e]
be1 = be1[mask_b]

#%% 谱分析

#%% 交叉谱
#%% 比值
#%% histgram2d

