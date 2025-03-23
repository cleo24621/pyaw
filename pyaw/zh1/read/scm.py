"""
About ulf
"""

import h5py
import pandas as pd
from pandas import Series, DataFrame

from pyaw.zh1 import configs


def get_dfs(fp):
    """
    单个变量数据由DataFrame格式存储，所有变量数据存储在字典中，键对应变量名，值对应相应的DataFrame
    :param fp: 文件路径
    :return:
    """
    dataframes = {}
    with h5py.File(fp, "r") as h5file:
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


class SCMUlf:
    dfs: dict
    datetime: Series
    df1c_scm: DataFrame

    def __init__(self, file_path):
        self.fs = configs.scmulf_fs
        self.row_len = 4096  # DataFrame行长度 (fixed)
        # self.target_fs = 16
        self.fp = file_path
        self.dfs = get_dfs(file_path)  # 所有变量对应的DataFrame组成的字典
        verse_time = self.dfs["VERSE_TIME"].squeeze()
        self.datetime = pd.to_datetime(
            verse_time, origin="2009-01-01", unit="ms"
        )  # element is Timestamp type
        self.start_time = self.datetime.iloc[0]
        self.df1c_scm = self._concat_data()

    def _concat_data(self) -> DataFrame:
        """
        获取拼接而成的DataFrame
        Returns:
            SCMUlf产品中形状为(n,1)的变量数据。DataFrame的索引为(DatetimeIndex,dtype=pd.Timestamp)，每一列的列名对应变量名，每一列的数据对应变量数据
        """
        dict1c = {}  # key: variable name; value: variable value
        for key in configs.scm_ulf_1c_vars:
            dict1c[key] = self.dfs[key].squeeze().values
        df1c = pd.DataFrame(
            index=self.datetime.values, data=dict1c
        )  # DataFrame. index is DatetimeIndex; data is 'dict1c'
        return df1c
