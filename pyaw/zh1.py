# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/22
@DESCRIPTION: 
"""

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pymap3d import ecef
from scipy import interpolate

from pyaw import configs, utils, utils_spectral


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


def print_name(fp):
    """
    打印变量名
    :param fp: 文件路径
    :return:
    """
    with h5py.File(fp, 'r') as h5file:
        def print_name_(name):
            print(name)

        h5file.visit(print_name_)
    return None


def get_dfs(fp):
    """
    单个变量数据由DataFrame格式存储，所有变量数据存储在字典中，键对应变量名，值对应相应的DataFrame
    :param fp: 文件路径
    :return:
    """
    dataframes = {}
    with h5py.File(fp, 'r') as h5file:
        for name, dataset in h5file.items():
            data = dataset[()]
            if data.ndim == 1:
                # 1D array
                df = pd.DataFrame(data, columns=[name])
            elif data.ndim == 2:
                # 2D array
                df = pd.DataFrame(data)
            else:
                # For higher-dimensional data, additional processing may be required
                raise ValueError(f"Dataset {name} has {data.ndim} dimensions, which is not supported.")
            dataframes[name] = df
    return dataframes


class FGM:
    def __init__(self, fp):
        self.fp = fp

    def recursively_extract_data(self, group, data_dict):
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data = item[:]
                # Check the shape of the dataset
                if data.shape[1] == 1:
                    # Flatten the (n,1) dataset to a single column
                    data_dict[key] = data.flatten()
                elif data.shape[1] == 3:
                    # Split the (n,3) dataset into 3 separate columns
                    data_dict[f'{key}1'] = data[:, 0]
                    data_dict[f'{key}2'] = data[:, 1]
                    data_dict[f'{key}3'] = data[:, 2]
            elif isinstance(item, h5py.Group):
                # If the item is a group, recursively extract its datasets
                self.recursively_extract_data(item, data_dict)

    def get_df(self):
        # Open the HDF5 file in read mode
        with h5py.File(self.fp, 'r') as hdf:
            # Create an empty dictionary to hold dataset names and data
            data_dict = {}
            # Recursively iterate through groups and datasets
            self.recursively_extract_data(hdf, data_dict)
            return pd.DataFrame(data_dict)


class SCMULF:
    def __init__(self, fp):
        self.fs = 1024
        self.row_len = 4096  # DataFrame行长度
        # self.target_fs = 16
        self.fp = fp
        self.dfs = get_dfs(self.fp)
        verse_time = self.dfs['VERSE_TIME'].squeeze()
        self.datetime = pd.to_datetime(verse_time, origin='2009-01-01', unit='ms')
        self.start_time = self.datetime.iloc[0]
        self.df1c_scm = self.concat_data()

    def concat_data(self):
        """
        将所有变量数据对应的字典转换成DataFrame
        :return:
        """
        dict1c = {}
        for key in configs.scm_ulf_1c_vars:
            dict1c[key] = self.dfs[key].squeeze().values
        df1c = pd.DataFrame(index=self.datetime.values, data=dict1c)  # todo: index name
        return df1c

    def print_name(self):
        print_name(self.fp)
        return None

    # def _get_resample_signal(self, vn):  #     df_vn = self.dfs[vn]  #     return get_resample_signal(df_vn,self.start_time,self.original_fs,self.target_fs)


class EFDULF(SCMULF):
    def __init__(self, fp):
        super().__init__(fp)
        self.fs = 125
        self.row_len = 256
        # self.target_fs = 25
        self.df1c_efd = self.concat_data()

    def concat_data(self):
        dict1c = {}
        for key in configs.efd_ulf_1c_vars:
            dict1c[key] = self.dfs[key].squeeze().values
        df1c = pd.DataFrame(index=self.datetime.values, data=dict1c)
        return df1c


class EFDSCMClip:
    def __init__(self, st, et, fp_efd, fp_scm, threshold: float = 100):
        """
        :param st: the chosen start time
        :param et: the chosen end time
        :param fp_efd: the file path of efd
        :param fp_scm: the file path of scm
        :param threshold: the threshold of datetime difference between practical interval and the theory interval
        """
        self.efd = EFDULF(fp_efd)  # efd instance
        self.scm = SCMULF(fp_scm)  # scm instance
        self.lowcut = 16.0  # frequency domain maximum
        self.target_fs = 32  # target fs
        assert self.scm.fs % self.target_fs == 0, "Original frequency must be divisible by target frequency"
        # datetime clip
        self.scm_datetime_clip = self.scm.datetime.values[
            (self.scm.datetime.values >= st) & (self.scm.datetime.values <= et)]
        self.efd_datetime_clip = self.efd.datetime.values[
            (self.efd.datetime.values >= st) & (self.efd.datetime.values <= et)]
        # clipped datetime verification (whether the time interval meets the expectations)
        b_theory_interval = pd.Timedelta((self.scm.row_len + 1) / self.scm.fs, unit='s')
        e_theory_interval = pd.Timedelta((self.efd.row_len + 1) / self.efd.fs, unit='s')
        diff_threshold = pd.Timedelta(threshold, unit='ms')
        # todo: mode group
        # todo: if not satisfy the assert, fill with nan?
        assert all(np.diff(
            self.scm_datetime_clip) - b_theory_interval < 10*pd.Timedelta(1/self.scm.fs, unit='s')), "the datetime duration of clipped b minus the theory datetime duration should be less than the set threshold."
        assert all(np.diff(
            self.efd_datetime_clip) - e_theory_interval < pd.Timedelta(1/self.efd.fs, unit='s')), "the datetime duration of clipped e minus the theory datetime duration should be less than the set threshold."
        self.efd_geo = pd.DataFrame(index=self.efd.datetime.values,
                                    data={'lat': self.efd.dfs['GEO_LAT'].squeeze().values,
                                          'lon': self.efd.dfs['GEO_LON'].squeeze().values,
                                          'alt': self.efd.dfs['ALTITUDE'].squeeze().values})
        self.A111_W = pd.DataFrame(index=self.efd.datetime.values, data=self.efd.dfs['A111_W'].values)
        self.A112_W = pd.DataFrame(index=self.efd.datetime.values, data=self.efd.dfs['A112_W'].values)
        self.A113_W = pd.DataFrame(index=self.efd.datetime.values, data=self.efd.dfs['A113_W'].values)
        self.scm_geo = pd.DataFrame(index=self.scm.datetime.values,
                                    data={'lat': self.scm.dfs['GEO_LAT'].squeeze().values,
                                          'lon': self.scm.dfs['GEO_LON'].squeeze().values,
                                          'alt': self.scm.dfs['ALTITUDE'].squeeze().values})
        self.A231_W = pd.DataFrame(index=self.scm.datetime.values, data=self.scm.dfs['A231_W'].values)
        self.A232_W = pd.DataFrame(index=self.scm.datetime.values, data=self.scm.dfs['A232_W'].values)
        self.A233_W = pd.DataFrame(index=self.scm.datetime.values, data=self.scm.dfs['A233_W'].values)
        (self.df1c_efd_clip, self.df1c_scm_clip, self.efd_geo_clip, self.A111_W_clip, self.A112_W_clip,
         self.A113_W_clip, self.scm_geo_clip, self.A231_W_clip, self.A232_W_clip, self.A233_W_clip) = (i.loc[st:et] for
                                                                                                       i in (
                                                                                                           self.efd.df1c_efd,
                                                                                                           self.scm.df1c_scm,
                                                                                                           self.efd_geo,
                                                                                                           self.A111_W,
                                                                                                           self.A112_W,
                                                                                                           self.A113_W,
                                                                                                           self.scm_geo,
                                                                                                           self.A231_W,
                                                                                                           self.A232_W,
                                                                                                           self.A233_W))
        # from now, I use the clipped data for analysis.
        self.e_enu1, self.e_enu2, self.e_enu3 = self.efd_geo2enu()  # time clipped data
        self.b_enu1, self.b_enu2, self.b_enu3 = self.scm_geo2enu()  # same time clipped data
        # get the datetime corresponding to the target fs
        self.resample_factor = int(self.scm.fs / self.target_fs)
        interval = 1 / self.target_fs
        self.datetime = pd.date_range(start=self.scm_datetime_clip[0],
                                      periods=int(self.b_enu1.shape[0] * self.b_enu1.shape[1] / self.resample_factor),
                                      freq=f'{interval}s')  # the common time grid
        self.b_resampled_ls = self.b_resampled()
        self.e_datetime = pd.date_range(start=self.efd_datetime_clip[0],
                                        periods=int(self.e_enu1.shape[0] * self.e_enu1.shape[1]),
                                        freq=f'{1 / self.efd.fs}s')  # b_datetime is similar, but I don't need, I just need the resampled b_datetime that is 'self.datetime'
        self.e_resampled_ls = self.e_resampled()
        self.data = self.get_data()
        self.data_preprocessed = self.preprocess_data()

    def mode_choose(self):
        pass

    def efd_geo2enu(self):
        e_enu1_ls = []
        e_enu2_ls = []
        e_enu3_ls = []
        if len(self.A111_W_clip) == len(self.A112_W_clip) == len(self.A113_W_clip):  # have the same number of rows
            for (index1, row1), (index2, row2), (index3, row3) in zip(self.A111_W_clip.iterrows(),
                                                                      self.A112_W_clip.iterrows(),
                                                                      self.A113_W_clip.iterrows()):
                assert index1 == index2 == index3, "index not equal"
                lat = self.efd_geo_clip['lat'][index1]
                lon = self.efd_geo_clip['lon'][index1]
                alt = self.efd_geo_clip['alt'][index1] * 1e3
                e_enu1, e_enu2, e_enu3 = ecef.ecef2enuv(row1, row2, row3, lat, lon, alt)
                e_enu1_ls.append(e_enu1)
                e_enu2_ls.append(e_enu2)
                e_enu3_ls.append(e_enu3)
        return pd.concat(e_enu1_ls, axis=1).T, pd.concat(e_enu2_ls, axis=1).T, pd.concat(e_enu3_ls, axis=1).T

    def scm_geo2enu(self):
        b_enu1_ls = []
        b_enu2_ls = []
        b_enu3_ls = []
        if len(self.A231_W_clip) == len(self.A232_W_clip) == len(self.A233_W_clip):  # have the same number of rows
            for (index1, row1), (index2, row2), (index3, row3) in zip(self.A231_W_clip.iterrows(),
                                                                      self.A232_W_clip.iterrows(),
                                                                      self.A233_W_clip.iterrows()):
                assert index1 == index2 == index3, "index not equal"
                lat = self.scm_geo_clip['lat'][index1]
                lon = self.scm_geo_clip['lon'][index1]
                alt = self.scm_geo_clip['alt'][index1] * 1e3
                b_enu1, b_enu2, b_enu3 = ecef.ecef2enuv(row1, row2, row3, lat, lon, alt)
                b_enu1_ls.append(b_enu1)
                b_enu2_ls.append(b_enu2)
                b_enu3_ls.append(b_enu3)
        return pd.concat(b_enu1_ls, axis=1).T, pd.concat(b_enu2_ls, axis=1).T, pd.concat(b_enu3_ls, axis=1).T

    def b_resampled(self):
        """

        :return: 'b' of zh1 after filter and resample in order.
        """
        # low-pass filter: 16hz
        b_filters = [utils_preprocess.LHBFilter(i.values.flatten(), fs=self.scm.fs, lowcut=self.lowcut) for i in
                     [self.b_enu1, self.b_enu2, self.b_enu3]]
        b_filtered_ls = [i.apply_filter() for i in b_filters]
        b_resample_ls = []
        for b in b_filtered_ls:
            b_resample_ls.append(b[::self.resample_factor])
        return b_resample_ls

    def e_resampled(self):
        # scipy
        old_timestamps = [pd.Timestamp(i).timestamp() for i in self.e_datetime]
        new_timestamps = [pd.Timestamp(i).timestamp() for i in self.datetime]
        f_s = [interpolate.interp1d(old_timestamps, i.values.flatten(), fill_value='extrapolate') for i in
               [self.e_enu1, self.e_enu2, self.e_enu3]]
        e_resampled_ls = [f(new_timestamps) for f in f_s]
        return e_resampled_ls  # todo: add index of b and e equal test

    def get_data(self):
        data = pd.DataFrame(index=self.datetime)
        e_columns = ['e_enu1', 'e_enu2', 'e_enu3']
        b_columns = ['b_enu1', 'b_enu2', 'b_enu3']
        for column, e in zip(e_columns, self.e_resampled_ls):
            data[column] = e
        for column, b in zip(b_columns, self.b_resampled_ls):
            data[column] = b
        return data

    def preprocess_data(self):
        e0_columns = ['e0_enu1', 'e0_enu2', 'e0_enu3']
        e1_columns = ['e1_enu1', 'e1_enu2', 'e1_enu3']
        data_dict = {}
        for (column_name, column_data), (e0_col, e1_col) in zip(self.data[['e_enu1', 'e_enu2', 'e_enu3']].items(),
                                                                zip(e0_columns, e1_columns)):
            _ = utils_preprocess.move_average(column_data, self.target_fs * 20)
            data_dict[e0_col] = _
            data_dict[e1_col] = column_data - _
        _1 = pd.DataFrame(data=data_dict)
        b0_columns = ['b0_enu1', 'b0_enu2', 'b0_enu3']
        b1_columns = ['b1_enu1', 'b1_enu2', 'b1_enu3']
        data_dict = {}
        for (column_name, column_data), (b0_col, b1_col) in zip(self.data[['b_enu1', 'b_enu2', 'b_enu3']].items(),
                                                                zip(b0_columns, b1_columns)):
            _ = utils_preprocess.move_average(column_data, self.target_fs * 20)
            data_dict[b0_col] = _
            data_dict[b1_col] = column_data - _
        _2 = pd.DataFrame(data=data_dict)
        return pd.concat([self.data, _1, _2], axis=1)

    def get_enu_e_b_compo(self):
        pass

    def get_ratio(self, e, b):
        psd_e = utils_spectral.PSD(e, fs=125, nperseg=1000)
        freqs_e, Pxx_e = psd_e.get_psd()
        psd_b = utils_spectral.PSD(b, fs=1024, nperseg=4096 * 2)
        freqs_b, Pxx_b = psd_b.get_psd()
        e_end_idx = np.where(freqs_e == 10.0)[0][0]
        b_end_idx = np.where(freqs_b == 10.0)[0][0]
        assert e_end_idx == b_end_idx, "freqs before 10hz not equal"
        freqs_e_clip = freqs_e[:e_end_idx]
        Pxx_e_clip = Pxx_e[:e_end_idx]
        freqs_b_clip = freqs_b[:b_end_idx]
        Pxx_b_clip = Pxx_b[:b_end_idx]
        return freqs_e_clip, np.sqrt(Pxx_e_clip), np.sqrt(Pxx_b_clip)

    def plot_ratio(self):
        pass


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
