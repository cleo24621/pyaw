# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/22
@DESCRIPTION: 
"""

import h5py
import numpy as np
import pandas as pd
from pymap3d import ecef
from scipy import interpolate

import utils.data
from core.read_data import get_dfs
from pyaw import configs, utils, utils_spectral
from utils.file_process import print_hdf5_variable_name


# class ZH1:
#     """
#     处理 hpm载荷的fgm和cdsm仪器的类，草稿。
#     """
#     def __init__(self):
#         self.fillvalue_fgm = (-999999, -9999)
#
#     def r_hpm_cdsm(self,fp):
#         # Open the HDF5 file in read mode
#         with h5py.File(fp, 'r') as hdf:
#             # Create an empty dictionary to hold dataset names and data
#             data_dict = {}
#
#             # Recursively iterate through groups and datasets
#             def recursively_extract_data(group, data_dict):
#                 for key in group.keys():
#                     item = group[key]
#                     if isinstance(item, h5py.Dataset):
#                         data = item[:]
#                         # Flatten the (n,1) dataset to a single column
#                         data_dict[key] = data.flatten()
#                     elif isinstance(item, h5py.Group):
#                         # If the item is a group, recursively extract its datasets
#                         recursively_extract_data(item, data_dict)
#
#             recursively_extract_data(hdf, data_dict)
#
#             # Convert the dictionary to a pandas DataFrame
#             df = pd.DataFrame(data_dict)
#             df['DATETIME'] = pd.to_datetime(df['UTC_TIME'].astype(str), format='%Y%m%d%H%M%S%f')
#         return df




def main():
    fp_scmulf = r"D:\cleo\master\pyaw\data\zh1\scm\CSES_01_SCM_1_L02_A2_178261_20210420_000623_20210420_004156_000.h5"
    scmulf = SCMULF(fp_scmulf)

    # fp_efd = r"\\Diskstation1\file_three\aw\zh1\efd\ulf\201911\CSES_01_EFD_1_L02_A1_096790_20191031_233350_20191101_000824_000.h5"
    # fp_scm = r"\\Diskstation1\file_three\aw\zh1\scm\ulf\201911\CSES_01_SCM_1_L02_A2_096790_20191031_233256_20191101_000821_000.h5"
    # start = pd.Timestamp('2019-10-31 23:34:08.0')
    # end = pd.Timestamp('2019-10-31 23:34:28.0')  # according to ratio fields 'df1c_efd' and 'df1c_scm'
    # efdscmclip = EFDSCMClip(start, end, fp_efd, fp_scm)
    # e_psd = utils_spectral.PSD(efdscmclip.data_preprocessed['e1_enu1'], efdscmclip.target_fs)
    # freqs_e, Pxx_e = e_psd.get_psd()
    # b_psd = utils_spectral.PSD(efdscmclip.data_preprocessed['b1_enu2'], efdscmclip.target_fs)
    # freqs_b, Pxx_b = b_psd.get_psd()
    # plt.figure(1)
    # plt.plot(freqs_e, np.sqrt(Pxx_e), color='r', label='e1_enu1')
    # plt.plot(freqs_b, np.sqrt(Pxx_b), color='b', label='b1_enu2')
    # plt.xlim([1, 16])
    # plt.legend()
    # plt.yscale('log')
    # plt.show()
    # plt.figure(2)
    # assert all(freqs_e == freqs_b), "freqs of e and b should be equal"
    # plt.plot(freqs_e, np.sqrt(Pxx_e / Pxx_b) * 1e6)
    # plt.xlim([1, 16])
    # plt.yscale('log')
    # plt.show()
    # cwt_ = utils_spectral.CWT(efdscmclip.data_preprocessed['e1_enu1'], efdscmclip.data_preprocessed['b1_enu2'],
    #                           sampling_period=1 / 32)  # todo:: use new CWT class in new utils.py file
    # cwt_.plot_module()  # cwt_.plot_phase()  # cwt_.plot_phase_hist_counts()
    # cwt_.plot_phase_hist_counts()


if __name__ == '__main__':
    main()

# # read all data of 1 d
# # according to time, slice to 24 hour
# dirp = r"D:\cleo\PycharmProjects\auroramaps\data\ZH-1\HPM\1111"
# fns = os.listdir(dirp)
# fps = [os.path.join(dirp, fn) for fn in fns]
# dfs = [ZH1.r_hpm_fgm(fp) for fp in fps]
# df_20191111 = pd.concat(dfs,ignore_index=True)
# df_20191111['HOUR'] = df_20191111['DATETIME'].dt.hour
# grouped = df_20191111.groupby('HOUR')
# # Step 3: Split into separate DataFrames by hour
# dfs_by_hour = {hour: group for hour, group in grouped}
# for hour, df in dfs_by_hour.items():
#     # Customize file name with hour and additional information
#     file_name = f'fgm_20191111_{hour}.pkl'
#
#     # Save the DataFrame as a .pkl file
#     df.to_pickle(file_name)
#
#     # You can print or log the saved file name for confirmation
#     print(f"Saved: {file_name}")
