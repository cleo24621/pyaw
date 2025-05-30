import math

import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pymap3d import ecef
from scipy.interpolate import interpolate
from scipy.signal import filtfilt, butter, buttord

import pyaw.utils
from utils import customize_butter

# Todo: optimize this file.

FGM_VARS = (
    "A221",
    "A222",
    "A223",
    "ALTITUDE",
    "B_FGM1",
    "B_FGM2",
    "B_FGM3",
    "FLAG_MT",
    "FLAG_SHW",
    "FLAG_TBB",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "UTC_TIME",
    "VERSE_TIME",
)
SCM_ULF_VARS = (
    "A231_P",
    "A231_W",
    "A232_P",
    "A232_W",
    "A233_P",
    "A233_W",
    "ALTITUDE",
    "FLAG",
    "FREQ",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "PhaseX",
    "PhaseY",
    "PhaseZ",
    "UTC_TIME",
    "VERSE_TIME",
    "WORKMODE",
)
SCM_ULF_1C_VARS = (
    "ALTITUDE",
    "FLAG",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "UTC_TIME",
    "VERSE_TIME",
    "WORKMODE",
)
SCM_ULF_RESAMPLE_VARS = ["A231_W", "A232_W", "A233_W"]
EFD_ULF_VARS = (
    "A111_P",
    "A111_W",
    "A112_P",
    "A112_W",
    "A113_P",
    "A113_W",
    "ALTITUDE",
    "FREQ",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "UTC_TIME",
    "VERSE_TIME",
    "WORKMODE",
)
EFD_ULF_1C_VARS = (
    "ALTITUDE",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "UTC_TIME",
    "VERSE_TIME",
    "WORKMODE",
)
EFD_ULF_RESAMPLE_VARS = ["A111_W", "A112_W", "A113_W"]
SCMULF_FS = 1024

EFDULF_FS = 125

# Ascending (south to north); descending (north to south).
INDICATOR_DESCEND_ASCEND = {
    "0": "descending",
    "1": "ascending",
}


class FGM:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._get_df()

    def _recursively_extract_data(self, group, data_dict):
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
                    data_dict[f"{key}1"] = data[:, 0]
                    data_dict[f"{key}2"] = data[:, 1]
                    data_dict[f"{key}3"] = data[:, 2]
            elif isinstance(item, h5py.Group):
                # If the item is a group, recursively extract its datasets
                self._recursively_extract_data(item, data_dict)

    def _get_df(self) -> DataFrame:
        """
        Returns:
            单维变量数据为1列，多维变量数据拆分成多列
        """
        # Open the HDF5 file in read mode
        with h5py.File(self.file_path, "r") as hdf:
            # Create an empty dictionary to hold dataset names and data
            data_dict = {}
            # Recursively iterate through groups and datasets
            self._recursively_extract_data(hdf, data_dict)
        return pd.DataFrame(data_dict)


class SCM:
    dfs: dict
    datetimes: Series
    df1c: DataFrame

    fs = SCMULF_FS
    row_len = 4096  # DataFrame行长度 (fixed)
    new_datetimes_dfs_names = [
        "WORKMODE",
        "A231_W",
        "A232_W",
        "A233_W",
        "A231_P",
        "A232_P",
        "A233_P",
        "PhaseX",
        "PhaseY",
        "PhaseZ",
        "ALTITUDE",
        "MAG_LAT",
        "MAG_LON",
        "GEO_LAT",
        "GEO_LON",
    ]  # variable that will use new datetimes as index (remove variables that have different length of rows and variables that's not needed)
    # target_fs = 32
    # lowcut = target_fs / 2
    f_t = 1
    f_z = 200

    def __init__(
        self,
        file_path,
        set_time_delta=pd.Timedelta("4 seconds"),
        tolerance=pd.Timedelta("0.01 seconds"),
    ):
        self.file_path = file_path
        self.set_time_delta = set_time_delta
        self.tolerance = tolerance
        self.dfs = self._get_dfs()  # 所有变量对应的DataFrame组成的字典

        # self.datetimes_dfs = {
        #     k: self.dfs[k] for k in self.new_datetimes_dfs_names
        # }  # note that the dfs index is not datetime type

        verse_time = self.dfs["VERSE_TIME"].squeeze()
        self.datetimes = pd.to_datetime(
            verse_time, origin="2009-01-01", unit="ms"
        )  # element is Timestamp type
        self.start_time = self.datetimes.iloc[0]

        # test: check
        # self._some_checks()
        self._check_split()  # default the min length of slice is greater than 5

        self.df1c = self._concat_data()

        # ---

        # get split data
        self.df1c_split_list = self._get_split_data(self.df1c)
        self.datetimes_split_list = self._get_split_data(self.datetimes)
        # check length and index
        assert len(self.df1c_split_list) == len(self.datetimes_split_list)
        for i in range(len(self.df1c_split_list)):
            assert np.array_equal(
                self.df1c_split_list[i].index.values,
                self.datetimes_split_list[i].values,
            )

        # # wave data, i.e., magnetic field
        # self.A231_W_df = pd.DataFrame(index=self.datetimes.values, data=self.dfs['A231_W'].values)
        # self.A232_W_df = pd.DataFrame(index=self.datetimes.values, data=self.dfs['A232_W'].values)
        # self.A233_W_df = pd.DataFrame(index=self.datetimes.values, data=self.dfs['A233_W'].values)
        # # get split dfs
        # self.A231_W_df_split_list = self._get_split_data(self.A231_W_df)
        # self.A232_W_df_split_list = self._get_split_data(self.A232_W_df)
        # self.A233_W_df_split_list = self._get_split_data(self.A233_W_df)

        # todo: after explore
        # self.new_datetimes = self.fill_gaps(target=target, tolerance=tolerance)
        # self.new_datetimes_dfs = (
        #     self._get_new_datetimes_dfs()
        # )  # the dfs index is datetime type and is new datetimes.
        #

    def get_wave_data_split_list(self):
        # wave data, i.e., magnetic field
        A231_W_df = pd.DataFrame(
            index=self.datetimes.values, data=self.dfs["A231_W"].values
        )
        A232_W_df = pd.DataFrame(
            index=self.datetimes.values, data=self.dfs["A232_W"].values
        )
        A233_W_df = pd.DataFrame(
            index=self.datetimes.values, data=self.dfs["A233_W"].values
        )
        # get split dfs
        self.A231_W_df_split_list = self._get_split_data(A231_W_df)
        self.A232_W_df_split_list = self._get_split_data(A232_W_df)
        self.A233_W_df_split_list = self._get_split_data(A233_W_df)
        return (
            self.A231_W_df_split_list,
            self.A232_W_df_split_list,
            self.A233_W_df_split_list,
        )

    def _get_dfs(self):
        """
        单个变量数据由DataFrame格式存储，所有变量数据存储在字典中，键对应变量名，值对应相应的DataFrame
        """
        dataframes = {}
        with h5py.File(self.file_path, "r") as h5file:
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
                    raise ValueError(
                        f"Dataset {name} has {data.ndim} dimensions, which is not supported."
                    )
                dataframes[name] = df
        return dataframes

    def _concat_data(self) -> DataFrame:
        """
        获取拼接而成的DataFrame
        Returns:
            SCMUlf产品中形状为(n,1)的变量数据。DataFrame的索引为(DatetimeIndex,dtype=pd.Timestamp)，每一列的列名对应变量名，每一列的数据对应变量数据
        """
        dict1c = {}  # key: variable name; value: variable value
        for key in SCM_ULF_1C_VARS:
            dict1c[key] = self.dfs[key].squeeze().values
        df1c = pd.DataFrame(
            index=self.datetimes.values, data=dict1c
        )  # DataFrame. index is DatetimeIndex; data is 'dict1c'
        return df1c

    def _get_split_slices(self, min_length=5):
        dts_diff = self.datetimes.diff()
        dts_diff_diff = abs(dts_diff - self.set_time_delta)
        indices = dts_diff_diff[dts_diff_diff > self.tolerance].index
        indices = sorted(list(set(indices)))  # 去重并排序

        # 生成分割点（包含起始0和结束位置）
        split_points = [0] + indices + [len(self.datetimes)]

        slices = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            if (end - start) >= min_length:  # 切片长度需≥min_length
                slices.append(slice(start, end))

        return slices

    def _get_split_data(self, data: pd.Series | pd.DataFrame, min_length=5):
        slices = self._get_split_slices(min_length=min_length)
        split_data_list = []
        for slice_ in slices:
            split_data_list.append(data.iloc[slice_])
        return split_data_list

    def _check_split(self, min_length=5):
        """
        确保数据分割后其对应的时间索引约等于实际差值同根据起始时间、采样率计算的理论差值。
        Args:
            min_length:

        Returns:

        """
        dts = self.datetimes
        split_dts = self._get_split_data(data=dts, min_length=min_length)
        for split_dt in split_dts:
            split_dt_diff = split_dt.diff()
            split_dt_diff_diff = abs(split_dt_diff - self.set_time_delta)
            assert all(
                split_dt_diff_diff[1:] < self.tolerance
            )  # because use 'diff()' will result in the first element be NaT. So remove the first element.

    def ecef2enuv(self, lats, lons, alts, Axx1_W_flat, Axx2_W_flat, Axx3_W_flat):
        """

        Args:
            Axx1_W_flat: 波形数据，经过split之后，再进行自主选择的切片之后，将其从左至右、从上至下展平。例如A231_W经过前述操作返回的结果。
            Axx2_W_flat:
            Axx3_W_flat:
            lats: 和A231_W_flat等相对应

        Returns:

        """
        vector_east_list = []
        vector_north_list = []
        vector_up_list = []
        for lat, lon, alt, A231_W, A232_W, A233_W in zip(
            lats, lons, alts, Axx1_W_flat, Axx2_W_flat, Axx3_W_flat
        ):
            vector_east, vector_north, vector_up = ecef.ecef2enuv(
                A231_W, A232_W, A233_W, lat, lon, alt
            )
            vector_east_list.append(vector_east)
            vector_north_list.append(vector_north)
            vector_up_list.append(vector_up)
        return (
            np.array(vector_east_list),
            np.array(vector_north_list),
            np.array(vector_up_list),
        )

    def generate_datetimes(self, start_time: pd.Timestamp, periods, interval):
        """

        Args:
            start_time:
            periods: 多少个周期（数据点）
            interval: 周期时间 (s)

        Returns:

        """
        return pd.date_range(start=start_time, periods=periods, freq=f"{interval}s")

    def get_downsample_scm_data(self, data: np.ndarray):
        step = self.fs // self.target_fs
        assert len(data) % step == 0  # 确保后续的时间索引的正确处理
        # firstly, low-pass filter
        b, a = customize_butter(self.fs, f_t=self.f_t, f_z=self.f_z)
        data_filter = filtfilt(b, a, data)
        return data_filter[::step]

    # not use because some places time delta don't satisfy 4s or 2.04s (time delta of 1 row)
    def fill_gaps(
        self,
        target: pd.Timedelta,
        tolerance: pd.Timedelta,
        max_total_insertions: int = 1000,
    ) -> pd.Series:
        """Fill gaps in time series while limiting total inserted points.

        Args:
            target: Target time interval (e.g. pd.Timedelta('2s'))
            tolerance: Allowed deviation from target (e.g. pd.Timedelta('0.5s'))
            max_total_insertions: Maximum allowed inserted points for entire series

        Returns:
            pd.DatetimeIndex: New time series with controlled gap filling

        Raises:
            ValueError: If total required insertions exceed max_total_insertions
        """
        target_seconds = target.total_seconds()
        tolerance_seconds = tolerance.total_seconds()
        max_allowed = target_seconds + tolerance_seconds

        new_dates = [self.datetimes[0]]  # 初始化新时间列表
        total_insertions = 0

        for i in range(len(self.datetimes) - 1):
            current = self.datetimes[i]
            next_ = self.datetimes[i + 1]
            dt = next_ - current
            dt_seconds = dt.total_seconds()
            # 检查是否需要分割间隔
            if abs(dt_seconds - target_seconds) > tolerance_seconds:
                required_intervals = math.ceil(dt_seconds / max_allowed)
                insertions_needed = required_intervals - 1

                # Check global insertion limit
                if total_insertions + insertions_needed > max_total_insertions:
                    raise ValueError(
                        f"Insertions limit reached: {max_total_insertions}. "
                        f"Required {total_insertions + insertions_needed} total."
                    )

                interval_length = dt / required_intervals  # 每个间隔的长度
                assert (
                    interval_length - target
                ) < tolerance  # make sure the new generated datetimes don't have gaps.

                # Perform insertions
                for k in range(1, required_intervals):
                    new_time = current + k * interval_length
                    new_dates.append(new_time)
                    total_insertions += 1
                new_dates.append(next_)  # 添加原始的下一个时间点
            else:
                new_dates.append(next_)  # 直接添加，无需分割

        self.new_datetimes = pd.Series(new_dates)

        return pd.Series(new_dates)

    # not use because fill_gaps() don't use
    def _get_new_datetimes_dfs(self):
        """

        Returns:

        """

        new_dfs = []
        for df in self.datetimes_dfs.values():
            assert len(df) == len(self.datetimes)
            df.index = self.datetimes
            new_dfs.append(df.reindex(self.new_datetimes))
        return new_dfs


class EFD(SCM):
    fs = EFDULF_FS
    row_len = 256
    new_datetimes_dfs_names = [
        "WORKMODE",
        "A111_W",
        "A112_W",
        "A113_W",
        "A111_P",
        "A112_P",
        "A113_P",
        "ALTITUDE",
        "MAG_LAT",
        "MAG_LON",
        "GEO_LAT",
        "GEO_LON",
    ]  # variable that will use new datetimes as index (remove variables that have different length of rows and variables that's not needed)
    f_t = 0
    f_z = 16

    def __init__(
        self,
        file_path,
        set_time_delta=pd.Timedelta("2.048 seconds"),
        tolerance=pd.Timedelta("0.5 seconds"),
    ):
        super().__init__(file_path, set_time_delta=set_time_delta, tolerance=tolerance)

    def _concat_data(self):
        """
        refer to SCMUlf.concat_data()
        Returns:

        """
        dict1c = {}
        for key in EFD_ULF_1C_VARS:
            dict1c[key] = self.dfs[key].squeeze().values
        df1c = pd.DataFrame(index=self.datetimes.values, data=dict1c)
        return df1c

    def get_downsample_efd_data(self, data, dts, new_dts):
        interp1d_funcs = [interpolate.interp1d(dts, data, fill_value="extrapolate")]
        return np.array([interp1d_func(new_dts) for interp1d_func in interp1d_funcs])

    def _some_checks(self):
        pass

    def get_wave_data_split_list(self):
        # wave data, i.e., magnetic field
        A111_W_df = pd.DataFrame(
            index=self.datetimes.values, data=self.dfs["A111_W"].values
        )
        A112_W_df = pd.DataFrame(
            index=self.datetimes.values, data=self.dfs["A112_W"].values
        )
        A113_W_df = pd.DataFrame(
            index=self.datetimes.values, data=self.dfs["A113_W"].values
        )
        # get split dfs
        self.A111_W_df_split_list = self._get_split_data(A111_W_df)
        self.A112_W_df_split_list = self._get_split_data(A112_W_df)
        self.A113_W_df_split_list = self._get_split_data(A113_W_df)
        return (
            self.A111_W_df_split_list,
            self.A112_W_df_split_list,
            self.A113_W_df_split_list,
        )


class SCMEFDUlf:
    """
    根据起止时间将efd和scm产品做切片并组合
    """

    def __init__(self, st, et, fp_scm, fp_efd, threshold: float = 100):
        """
        :param st: the chosen start time
        :param et: the chosen end time
        :param fp_efd: the file path of efd
        :param fp_scm: the file path of scm
        :param threshold: the threshold of datetime difference between practical interval and the theory interval
        """
        self.scm = SCM(fp_scm)  # scm instance
        self.efd = EFD(fp_efd)  # efd instance
        self.lowcut = 16.0  # frequency domain maximum
        self.target_fs = 32  # target fs
        assert (
            self.scm.fs % self.target_fs == 0
        ), "Original frequency must be divisible by target frequency"
        # datetime clip
        self.scm_datetimes_clip = self.scm.datetimes.values[
            (self.scm.datetimes.values >= st) & (self.scm.datetimes.values <= et)
        ]
        self.efd_datetimes_clip = self.efd.datetimes.values[
            (self.efd.datetimes.values >= st) & (self.efd.datetimes.values <= et)
        ]
        self._check_datetimes_clip()

        # initialize other dataframes
        self.efd_geo = pd.DataFrame(
            index=self.efd.datetimes.values,
            data={
                "lat": self.efd.dfs["GEO_LAT"].squeeze().values,
                "lon": self.efd.dfs["GEO_LON"].squeeze().values,
                "alt": self.efd.dfs["ALTITUDE"].squeeze().values,
            },
        )
        self.A111_W = pd.DataFrame(
            index=self.efd.datetimes.values, data=self.efd.dfs["A111_W"].values
        )
        self.A112_W = pd.DataFrame(
            index=self.efd.datetimes.values, data=self.efd.dfs["A112_W"].values
        )
        self.A113_W = pd.DataFrame(
            index=self.efd.datetimes.values, data=self.efd.dfs["A113_W"].values
        )
        self.scm_geo = pd.DataFrame(
            index=self.scm.datetimes.values,
            data={
                "lat": self.scm.dfs["GEO_LAT"].squeeze().values,
                "lon": self.scm.dfs["GEO_LON"].squeeze().values,
                "alt": self.scm.dfs["ALTITUDE"].squeeze().values,
            },
        )
        self.A231_W = pd.DataFrame(
            index=self.scm.datetimes.values, data=self.scm.dfs["A231_W"].values
        )
        self.A232_W = pd.DataFrame(
            index=self.scm.datetimes.values, data=self.scm.dfs["A232_W"].values
        )
        self.A233_W = pd.DataFrame(
            index=self.scm.datetimes.values, data=self.scm.dfs["A233_W"].values
        )

        # clip
        (
            self.df1c_efd_clip,
            self.df1c_scm_clip,
            self.efd_geo_clip,
            self.A111_W_clip,
            self.A112_W_clip,
            self.A113_W_clip,
            self.scm_geo_clip,
            self.A231_W_clip,
            self.A232_W_clip,
            self.A233_W_clip,
        ) = (
            i.loc[st:et]
            for i in (
                self.efd.df1c,
                self.scm.df1c,
                self.efd_geo,
                self.A111_W,
                self.A112_W,
                self.A113_W,
                self.scm_geo,
                self.A231_W,
                self.A232_W,
                self.A233_W,
            )
        )

        # from now, I use the clipped data for analysis.
        self.e_enu1, self.e_enu2, self.e_enu3 = self.efd_geo2enu()  # time clipped data
        self.b_enu1, self.b_enu2, self.b_enu3 = (
            self.scm_geo2enu()
        )  # same time clipped data

        # get the datetime corresponding to the target fs
        self.resample_factor = int(self.scm.fs / self.target_fs)
        interval = 1 / self.target_fs
        self.datetime = pd.date_range(
            start=self.scm_datetimes_clip[0],
            periods=int(
                self.b_enu1.shape[0] * self.b_enu1.shape[1] / self.resample_factor
            ),
            freq=f"{interval}s",
        )  # the common time grid
        # resample b
        self.b_resampled_ls = self.b_resampled()
        # resampe e
        self.e_datetime = pd.date_range(
            start=self.efd_datetimes_clip[0],
            periods=int(self.e_enu1.shape[0] * self.e_enu1.shape[1]),
            freq=f"{1 / self.efd.fs}s",
        )  # b_datetime is similar, but I don't need, I just need the resampled b_datetime that is 'self.datetime'
        self.e_resampled_ls = self.e_resampled()
        self.data = self.get_data()  # the needed data (no process with nan...)
        self.data_concat_e0_e1_enu = (
            self.preprocess_data()
        )  # get df concated back and disturb info

    def _check_datetimes_clip(self):
        b_theory_interval = pd.Timedelta((self.scm.row_len + 1) / self.scm.fs, unit="s")
        e_theory_interval = pd.Timedelta((self.efd.row_len + 1) / self.efd.fs, unit="s")
        assert all(
            np.diff(self.scm_datetimes_clip) - b_theory_interval
            < 10 * pd.Timedelta(1 / self.scm.fs, unit="s")
        ), "the datetime duration of clipped b minus the theory datetime duration should be less than the set threshold."
        assert all(
            np.diff(self.efd_datetimes_clip) - e_theory_interval
            < pd.Timedelta(1 / self.efd.fs, unit="s")
        ), "the datetime duration of clipped e minus the theory datetime duration should be less than the set threshold."

    def efd_geo2enu(self) -> tuple[DataFrame]:  # todo
        e_enu1_ls = []
        e_enu2_ls = []
        e_enu3_ls = []
        if (
            len(self.A111_W_clip) == len(self.A112_W_clip) == len(self.A113_W_clip)
        ):  # have the same number of rows
            for (index1, row1), (index2, row2), (index3, row3) in zip(
                self.A111_W_clip.iterrows(),
                self.A112_W_clip.iterrows(),
                self.A113_W_clip.iterrows(),
            ):
                assert index1 == index2 == index3, "index not equal"
                lat = self.efd_geo_clip["lat"][index1]
                lon = self.efd_geo_clip["lon"][index1]
                alt = self.efd_geo_clip["alt"][index1] * 1e3
                e_enu1, e_enu2, e_enu3 = ecef.ecef2enuv(row1, row2, row3, lat, lon, alt)
                e_enu1_ls.append(e_enu1)
                e_enu2_ls.append(e_enu2)
                e_enu3_ls.append(e_enu3)
        return (
            pd.concat(e_enu1_ls, axis=1).T,
            pd.concat(e_enu2_ls, axis=1).T,
            pd.concat(e_enu3_ls, axis=1).T,
        )

    def scm_geo2enu(self) -> tuple[DataFrame]:
        b_enu1_ls = []
        b_enu2_ls = []
        b_enu3_ls = []
        if (
            len(self.A231_W_clip) == len(self.A232_W_clip) == len(self.A233_W_clip)
        ):  # have the same number of rows
            for (index1, row1), (index2, row2), (index3, row3) in zip(
                self.A231_W_clip.iterrows(),
                self.A232_W_clip.iterrows(),
                self.A233_W_clip.iterrows(),
            ):
                assert index1 == index2 == index3, "index not equal"
                lat = self.scm_geo_clip["lat"][index1]
                lon = self.scm_geo_clip["lon"][index1]
                alt = self.scm_geo_clip["alt"][index1] * 1e3
                b_enu1, b_enu2, b_enu3 = ecef.ecef2enuv(row1, row2, row3, lat, lon, alt)
                b_enu1_ls.append(b_enu1)
                b_enu2_ls.append(b_enu2)
                b_enu3_ls.append(b_enu3)
        return (
            pd.concat(b_enu1_ls, axis=1).T,
            pd.concat(b_enu2_ls, axis=1).T,
            pd.concat(b_enu3_ls, axis=1).T,
        )

    def b_resampled(self):
        """

        :return: 'b' of zh1 after filter and resample in order.
        """
        # low-pass filter: 16hz
        b, a = customize_butter(
            fs=self.scm.fs, f_t=self.scm.f_t, f_z=self.scm.f_z, type="lowpass"
        )
        b_filtered_ls = [
            filtfilt(b, a, i.values.flatten())
            for i in [self.b_enu1, self.b_enu2, self.b_enu3]
        ]
        b_resample_ls = []
        for b in b_filtered_ls:
            b_resample_ls.append(b[:: self.resample_factor])
        return b_resample_ls

    def e_resampled(self):
        # scipy
        old_timestamps = [pd.Timestamp(i).timestamp() for i in self.e_datetime]
        new_timestamps = [pd.Timestamp(i).timestamp() for i in self.datetime]
        f_s = [
            interpolate.interp1d(
                old_timestamps, i.values.flatten(), fill_value="extrapolate"
            )
            for i in [self.e_enu1, self.e_enu2, self.e_enu3]
        ]
        e_resampled_ls = [f(new_timestamps) for f in f_s]
        return e_resampled_ls  # todo: add index of b and e equal test

    def get_data(self):
        """
        concat b and e with base datetimes
        Returns:

        """
        data = pd.DataFrame(index=self.datetime)
        e_columns = ["e_enu1", "e_enu2", "e_enu3"]
        b_columns = ["b_enu1", "b_enu2", "b_enu3"]
        for column, e in zip(e_columns, self.e_resampled_ls):
            data[column] = e
        for column, b in zip(b_columns, self.b_resampled_ls):
            data[column] = b
        return data

    def preprocess_data(self, window_seconds=20, center=True, min_periods_seconds=20):
        # e
        window = self.target_fs * window_seconds
        min_periods = self.target_fs * min_periods_seconds
        e0_columns = ["e0_enu1", "e0_enu2", "e0_enu3"]  # from mov ave
        e1_columns = ["e1_enu1", "e1_enu2", "e1_enu3"]
        data_dict = {}
        for (column_name, column_data), (e0_col, e1_col) in zip(
            self.data[["e_enu1", "e_enu2", "e_enu3"]].items(),
            zip(e0_columns, e1_columns),
        ):
            arr_mov_ave = pyaw.utils.move_average(
                column_data.values,
                window=window,
                center=center,
                min_periods=min_periods,
            )
            data_dict[e0_col] = arr_mov_ave
            data_dict[e1_col] = column_data - arr_mov_ave
        _1 = pd.DataFrame(data=data_dict)
        # b
        # use mov ave, because igfr calculate need time and lat,lon,alt data,but after A111_W type flatten, we don't have the corresponding geo infos.
        b0_columns = ["b0_enu1", "b0_enu2", "b0_enu3"]
        b1_columns = ["b1_enu1", "b1_enu2", "b1_enu3"]
        data_dict = {}
        for (column_name, column_data), (b0_col, b1_col) in zip(
            self.data[["b_enu1", "b_enu2", "b_enu3"]].items(),
            zip(b0_columns, b1_columns),
        ):
            _ = pyaw.utils.move_average(
                column_data.values,
                window=window,
                center=center,
                min_periods=min_periods,
            )
            data_dict[b0_col] = _
            data_dict[b1_col] = column_data - _
        _2 = pd.DataFrame(data=data_dict)
        return pd.concat([self.data, _1, _2], axis=1)


def customize_butter(fs, f_t, f_z, type="lowpass"):
    """

    Args:
        fs: 采样率 (Hz)
        f_t: 通带截止频率 (Hz)。低，例如100
        f_z: 阻带截止频率 (Hz)。高，例如200
        type: ‘lowpass’, ‘highpass’, ‘bandpass’

    Returns:

    """
    # 归一化频率
    wp = f_t / (fs / 2)
    ws = f_z / (fs / 2)

    # 计算阶数（默认 gpass=3dB, gstop=40dB）
    order, wn = buttord(wp, ws, 3, 40)
    b, a = butter(order, wn, type)
    return b, a


def test_split():
    import os

    data_dir_path = r"G:\master\pyaw\data"
    # file_name = "CSES_01_SCM_1_L02_A2_096790_20191031_233256_20191101_000821_000.h5"
    file_name = (
        "../../data/CSES_01_SCM_1_L02_A2_175381_20210401_012104_20210401_015640_000.h5"
    )
    file_path = os.path.join(data_dir_path, file_name)
    scm = SCM(file_path)
    # dts = scm.datetimes
    # split_dts = scm.get_split_data(dts)
    scm._check_split()
    print("---")
