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
# fps = ['./SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl',
#        './aux_SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl',
#        './SW_EXPT_EFIA_TCT16_12885_20160311T061733_20160311T075106.pkl']
fps = ['./SW_OPER_MAGB_HR_1B_1349_20140219T020540_20140219T034014.pkl',
       './aux_SW_OPER_MAGB_HR_1B_1349_20140219T020540_20140219T034014.pkl',
       './SW_EXPT_EFIB_TCT16_1349_20140219T020540_20140219T034014.pkl']
df_b = pd.read_pickle(fps[0])
df_b_aux = pd.read_pickle(fps[1])
df_e = pd.read_pickle(fps[2])

#%% 选取列，选取时间范围
st = '20140219T023100'
et = '20140219T024100'
df_b_clip = df_b[['B_NEC','Longitude','Latitude','Radius']]
df_b_clip = df_b_clip.loc[pd.Timestamp(st):pd.Timestamp(et)]
df_b_aux_clip = df_b_aux[['QDLat','QDLon','MLT']]
df_b_aux_clip = df_b_aux_clip.loc[pd.Timestamp(st):pd.Timestamp(et)]
df_e_clip = df_e[['Longitude','Latitude','Radius','VsatE','VsatN','Ehy','Ehx']]
df_e_clip = df_e_clip.loc[pd.Timestamp(st):pd.Timestamp(et)]


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
fs_b = 50
mv_window_seconds = 20  # int

window_b = fs_b * mv_window_seconds
_,be,_ = utils.get_3arrs(df_b_clip['B_NEC'])
be_mv = utils.move_average(be, window_b)
be1 = be - be_mv

datetimes_e = df_e_clip.index.values
datetimes_b = df_b_clip.index.values

be1 = utils.align_high2low(be1,datetimes_b,datetimes_e)

#%% 谱分析
from scipy.signal import stft

fs = 16
window = 'hann'
stft_window_seconds = 4  # second
nperseg = int(stft_window_seconds * fs)  # 每个窗的采样点数
noverlap = nperseg // 2  # 50%重叠

# get stft
freqs, ts, Zxx_e = stft(ehn, fs, window=window, nperseg=nperseg, noverlap=noverlap,
                        scaling='psd')
_, _, Zxx_b = stft(be1, fs, window=window, nperseg=nperseg, noverlap=noverlap,
                       scaling='psd')

ts_dt64 = datetimes_e[0] + [np.timedelta64(int(_), 's') for _ in ts]

#%% 绘制谱 e
import matplotlib.pyplot as plt

Zxx_e_m = np.abs(Zxx_e)
plt.pcolormesh(ts_dt64, freqs, np.log10(Zxx_e_m), shading='gouraud')  # 可以用初始值也可以用log10
plt.colorbar()
plt.xticks(rotation=45)
plt.show()

#%% 绘制谱 b
Zxx_b_m = np.abs(Zxx_b)
plt.pcolormesh(ts_dt64, freqs, np.log10(Zxx_b_m), shading='gouraud')
plt.colorbar()
plt.ylim([0, 8])
plt.xticks(rotation=45)
plt.show()

#%% 交叉谱
cross_e_b_spectral_density = Zxx_e * np.conj(Zxx_b)

#%% plot
cross_e_b_spectral_density_module = np.abs(cross_e_b_spectral_density)
plt.pcolormesh(ts_dt64, freqs, np.log10(cross_e_b_spectral_density_module), shading='gouraud')
plt.xticks(rotation=45)
plt.colorbar()
plt.show()

#%% modify and plot
cross_e_b_spectral_density_module_modified = cross_e_b_spectral_density_module.copy()
modify_per99 = np.percentile(cross_e_b_spectral_density_module_modified,99)
cross_e_b_spectral_density_module_modified[cross_e_b_spectral_density_module_modified>modify_per99] = modify_per99

plt.pcolormesh(ts_dt64, freqs, cross_e_b_spectral_density_module_modified, shading='gouraud')
plt.xticks(rotation=45)
plt.colorbar()
plt.show()

#%% cross mean
plt.figure()
plt.plot(ts_dt64, cross_e_b_spectral_density_module.mean(axis=0))
plt.xticks(rotation=45)
plt.show()

#%% coherence
def split_array(data,step=11):
    # Split the array
    result = [data[:, i:i + step] for i in range(0, data.shape[1] - step, step)]
    # Add the remaining columns to the last segment
    remainder = data[:, step * len(result):]
    if remainder.size > 0:
        if len(result) > 0:
            # Append remaining columns to the last split
            result[-1] = np.hstack((result[-1], remainder))
        else:
            # If there's no initial split, the remainder is the only result
            result.append(remainder)
    # # Convert result to a NumPy array (optional)
    # print(f"Number of resulting arrays: {len(result)}")
    # for idx, arr in enumerate(result):
    #     print(f"Shape of array {idx}: {arr.shape}")
    return result

#%% compute
cross_e_b_spectral_density_split = split_array(cross_e_b_spectral_density)  # ls
denominator1ls = split_array(np.abs(Zxx_e ** 2))
denominator2ls = split_array(np.abs(Zxx_b ** 2))

coherences_f = []
for i in range(len(cross_e_b_spectral_density_split)):
    nominator = cross_e_b_spectral_density_split[i].mean(axis=1)
    denominator = np.sqrt(denominator1ls[i].mean(axis=1)) * np.sqrt(denominator2ls[i].mean(axis=1))
    coherences_f.append(nominator / denominator)
    # denominator = sum(cross_e_b_spectral_density_module_split[i]) / len(cross_e_b_spectral_density_module_split[i])
    # nominator = np.sqrt(sum(nominator1ls[i]) / len(nominator1ls[i])) * np.sqrt(sum(nominator2ls[i]) / len(nominator2ls[i]))
    # coherences_spec.append(sum(np.abs(denominator / nominator)) / len(np.abs(denominator / nominator)))


coherences = []
for c_f in coherences_f:
    coherences.append(np.abs(c_f).mean())

plt.figure()
plt.plot(coherences)
plt.axhline(0.5)
plt.show()


#%%
S_By_Ex_split_spec = split_array(cross_e_b_spectral_density)
nominator1ls = split_array(np.abs(Zxx_e ** 2))
nominator2ls = split_array(np.abs(Zxx_b ** 2))
coherences_spec = []
for i in range(len(S_By_Ex_split_spec)):
    denominator = sum(S_By_Ex_split_spec[i]) / len(S_By_Ex_split_spec[i])
    nominator = np.sqrt(sum(nominator1ls[i]) / len(nominator1ls[i])) * np.sqrt(sum(nominator2ls[i]) / len(nominator2ls[i]))
    print("denominator:",denominator)
    print("nominator:",nominator)
    print("denominator/nominator",denominator/nominator)
    print("coherency:",sum(np.abs(denominator / nominator)) / len(np.abs(denominator / nominator)))
    print("----------------------------------------")
    coherences_spec.append(sum(np.abs(denominator / nominator)) / len(np.abs(denominator / nominator)))

plt.figure()
plt.plot(coherences_spec)
plt.axhline(0.5)
plt.show()

#%% 比值
#%% histgram2d

