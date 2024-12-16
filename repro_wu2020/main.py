# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/12/11
@DESCRIPTION: 
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import stft coherence

from pyaw import swarm, utils_spectral,utils_preprocess
from pyaw.swarm import Swarm


fp_e = "sw_efi16B_20140219T0231_20140219T0241_0.pkl"
fp_b = "sw_vfm50B_20140219T0231_20140219T0241_0.pkl"

swarm_e = Swarm(fp_e, 'efi16','20140219T0231','20140219T0241')
swarm_b = Swarm(fp_b, 'vfm50','20140219T0231','20140219T0241')
df_e = swarm_e.df
df_b = swarm_b.df


# IGRF B
IGRF_B = pd.read_pickle(r'pyIGRF/wu2020_IGRF_B_20140219T0231_20140219T0241.pkl')

def get_quaternions_NEC2VFM():
    # get quaternions
    B_VFM = df_b['B_VFM']
    B_NEC = df_b['B_NEC']
    # Preallocate
    n_iterations = len(B_VFM)
    array_shape = (4,)  # Shape of each array
    quaternions = np.empty((n_iterations, *array_shape))
    for i,vector_a,vector_b in zip(range(n_iterations),B_VFM,B_NEC):
        # Compute the rotation aligning vector_b to vector_a
        rotation, _ = R.align_vectors([vector_a], [vector_b])

        # Extract the quaternion (in scalar-last format by default)
        quaternion = rotation.as_quat()
        quaternions[i] = quaternion
    return quaternions

def get_residual_By():
    igrf_b_vfm_ls = []
    for igrf_b_nec, quaternion in zip(IGRF_B[['IGRF_B_N', 'IGRF_B_E', 'IGRF_B_C']].apply(lambda row: row.values, axis=1), get_quaternions_NEC2VFM()):
        rotation = R.from_quat(quaternion)
        igrf_b_vfm_ls.append(rotation.apply(igrf_b_nec))
    df_b['IGRF_B_VFM'] = igrf_b_vfm_ls
    By = df_b['B_VFM'].apply(lambda row: row[1])
    model_By = df_b['IGRF_B_VFM'].apply(lambda row: row[1])
    residual_By = By - model_By
    return residual_By

redisual_By = get_residual_By()
resisual_By_align = utils_preprocess.align_high2low(redisual_By,df_e['Ehx'])

from scipy.signal import spectrogram, coherence

window='hann'
window_length=4
fs = 16
nperseg = int(window_length * fs)  # 每个窗的采样点数
noverlap = nperseg // 2  # 50%重叠


f, t_stft, S_By = spectrogram(-resisual_By_align, fs=fs,window=window, nperseg=nperseg, noverlap=noverlap,mode='complex')
_, _, S_Ex = spectrogram(df_e['Ehx'], fs=fs,window=window, nperseg=nperseg, noverlap=noverlap,mode='complex')