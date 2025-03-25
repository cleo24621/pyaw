import math

import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pymap3d import ecef
from scipy.interpolate import interpolate
from scipy.signal import filtfilt

from pyaw.configs import Zh1Configs
from utils.filter import customize_butter


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

    fs = Zh1Configs.scmulf_fs
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
    target_fs = 32
    lowcut = target_fs / 2
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

        self.datetimes_dfs = {
            k: self.dfs[k] for k in self.new_datetimes_dfs_names
        }  # note that the dfs index is not datetime type

        verse_time = self.dfs["VERSE_TIME"].squeeze()
        self.datetimes = pd.to_datetime(
            verse_time, origin="2009-01-01", unit="ms"
        )  # element is Timestamp type
        self.start_time = self.datetimes.iloc[0]

        # test: check
        self._some_checks()
        self._check_split()  # default the min length of slice is greater than 5

        # todo: after explore
        # self.new_datetimes = self.fill_gaps(target=target, tolerance=tolerance)
        # self.new_datetimes_dfs = (
        #     self._get_new_datetimes_dfs()
        # )  # the dfs index is datetime type and is new datetimes.
        #
        # self.df1c = self._concat_data()

    def _some_checks(self):
        assert self.fs % self.target_fs == 0

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
        for key in Zh1Configs.scm_ulf_1c_vars:
            dict1c[key] = self.dfs[key].squeeze().values
        df1c = pd.DataFrame(
            index=self.datetimes.values, data=dict1c
        )  # DataFrame. index is DatetimeIndex; data is 'dict1c'
        return df1c

    def get_split_slices(self, min_length=5):
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

    def get_split_data(self, data: pd.Series | pd.DataFrame, min_length=5):
        slices = self.get_split_slices(min_length=min_length)
        split_data_list = []
        for slice_ in slices:
            split_data_list.append(data.iloc[slice_])
        return split_data_list

    def _check_split(self, min_length=5):
        dts = self.datetimes
        split_dts = self.get_split_data(data=dts, min_length=min_length)
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
    fs = Zh1Configs.efdulf_fs
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
        for key in Zh1Configs.efd_ulf_1c_vars:
            dict1c[key] = self.dfs[key].squeeze().values
        df1c = pd.DataFrame(index=self.datetimes.values, data=dict1c)
        return df1c

    def get_downsample_efd_data(self, data, dts, new_dts):
        interp1d_funcs = [interpolate.interp1d(dts, data, fill_value="extrapolate")]
        return np.array([interp1d_func(new_dts) for interp1d_func in interp1d_funcs])


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


def test_split():
    import os
    from core import zh1

    data_dir_path = "G:\master\pyaw\data"
    # file_name = "CSES_01_SCM_1_L02_A2_096790_20191031_233256_20191101_000821_000.h5"
    file_name = "CSES_01_SCM_1_L02_A2_175381_20210401_012104_20210401_015640_000.h5"
    file_path = os.path.join(data_dir_path, file_name)
    scm = zh1.SCM(file_path)
    # dts = scm.datetimes
    # split_dts = scm.get_split_data(dts)
    scm._check_split()
    print("---")
