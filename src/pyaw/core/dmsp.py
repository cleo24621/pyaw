import os.path
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from spacepy.pycdf import CDF

from src.pyaw import DMSPConfigs
import xarray as xr


class SPDF:
    class SSIES3:
        def __init__(self, file_path):
            self.file_path = file_path
            self.original_df = self._read()
            self.df = self._other_process()  # DataFrame after quality process and rename, reindex ...

        def _read(self, variables=DMSPConfigs.SPDF.ssies3_vars) -> DataFrame:
            """
            return original data
            Args:
                variables: needed ssies3 variables

            Returns:
                original data of selected variables
            """
            cdf = CDF(self.file_path)
            # read data
            df = pd.DataFrame()
            for var_name in cdf.keys():
                if var_name not in variables:
                    continue
                df[var_name] = cdf[var_name][...]
            return df

        def _quality_process(self) -> pd.DataFrame:
            """
            Use quality flag and valid value range to do quality process.
            Rename and Reindex the time column for the unified types.
            Rename and drop some columns.
            Returns:
                preprocessed data
            """
            df = self.original_df.copy()
            # velocity
            # pay attention: f17星的 vqual 的坏对应的只有 4
            df.loc[
                (df["vxqual"] == DMSPConfigs.SPDF.vx_qual_filter)
                | (df["vx"].abs() > DMSPConfigs.SPDF.vx_valid_value),
                "vx",
            ] = np.nan
            df.loc[
                (df["vyqual"] == DMSPConfigs.SPDF.vy_qual_filter)
                | (df["vy"].abs() > DMSPConfigs.SPDF.vy_valid_value),
                "vy",
            ] = np.nan
            df.loc[
                (df["vzqual"] == DMSPConfigs.SPDF.vz_qual_filter)
                | (df["vz"].abs() > DMSPConfigs.SPDF.vz_valid_value),
                "vz",
            ] = np.nan
            # ductdens (number density of ion)
            df.loc[
                (df["ductdens"] > DMSPConfigs.SPDF.ductdens_valid_value_max)
                | (df["ductdens"] < DMSPConfigs.SPDF.ductdens_valid_value_min),
                "ductdens",
            ] = np.nan
            # fraco, frache, frach
            # 因为变量说明中明确说明应该舍弃的异常值，所以没有使用[validmin, validmax]
            df.loc[
                (df["fraco"] > DMSPConfigs.SPDF.frac_valid_value_max)
                | (df["fraco"] < DMSPConfigs.SPDF.frac_valid_value_min)
                | (df["fracoqual"] == DMSPConfigs.SPDF.frac_qual_filter),
                "fraco",
            ] = np.nan
            df.loc[
                (df["frach"] > DMSPConfigs.SPDF.frac_valid_value_max)
                | (df["frach"] < DMSPConfigs.SPDF.frac_valid_value_min)
                | (df["frachqual"] == DMSPConfigs.SPDF.frac_qual_filter),
                "frach",
            ] = np.nan
            df.loc[
                (df["frache"] > DMSPConfigs.SPDF.frac_valid_value_max)
                | (df["frache"] < DMSPConfigs.SPDF.frac_valid_value_min)
                | (df["frachequal"] == DMSPConfigs.SPDF.frac_qual_filter),
                "frache",
            ] = np.nan
            # temp (ion temperature)
            df.loc[
                (df["temp"] > DMSPConfigs.SPDF.temp_valid_value_max)
                | (df["temp"] < DMSPConfigs.SPDF.temp_valid_value_min)
                | (df["tempqual"] == DMSPConfigs.SPDF.temp_qual_filter),
                "temp",
            ] = np.nan
            # te (electron temperature)
            df.loc[
                (df["te"] > DMSPConfigs.SPDF.te_valid_value_max)
                | (df["te"] < DMSPConfigs.SPDF.te_valid_value_min),
                "te",
            ] = np.nan
            return df

        def _other_process(self):
            df = (
                self._quality_process()
            )  # （没有使用.copy()）质量控制后返回的DataFrame的视图
            # rename
            df.rename(columns={"Epoch": "datetime"}, inplace=True)
            # reindex
            df.set_index("datetime", inplace=True)
            # interpolate Nan
            df = df.resample(
                "1s"
            ).mean()  # ssies3 data may have some time that there is not data.
            # 确保没有重复的时间
            assert not df.index.duplicated().any(), "duplicated datetime"
            # drop
            df.drop(
                [
                    "vxqual",
                    "vyqual",
                    "vzqual",
                    "tempqual",
                    "frachqual",
                    "frachequal",
                    "fracoqual",
                ],
                axis=1,
                inplace=True,
            )
            # rename again
            df.rename(
                columns={"vx": "v_s3_sc1", "vy": "v_s3_sc2", "vz": "v_s3_sc3"},
                inplace=True,
            )
            if all([e in df.columns for e in ["bx", "by", "bz"]]):
                df["bz"] = -df["bz"]  # change down to up
                df.rename(
                    columns={"bx": "bm_enu2", "by": "bm_enu1", "bz": "bm_enu3"},
                    inplace=True,
                )  # 'm' means IGRF model
            return df

    class SSM:
        def __init__(self, file_path):
            self.file_path = file_path
            self.original_df = self._read()
            self.df = self._process()

        def _read(self, variables=DMSPConfigs.SPDF.ssm_vars) -> pd.DataFrame:
            """
            将spdf托管的ssm载荷数据以pd.DataFrame的格式返回
            :param variables: read variables that you need
            :return: original data
            """
            # CDF instance
            cdf = CDF(self.file_path)

            # read data
            # delete data like [x,y,z] (label data)
            var_shape = {}
            for var in cdf:
                if len(cdf[var].shape) == 1 and cdf[var].shape[0] == 3:
                    continue
                var_shape[var] = cdf[var].shape
            # load the data into a df
            df = pd.DataFrame()
            for var, shape in var_shape.items():
                if var not in variables:
                    continue
                # if the data is 3-dimensional, decompose it into 3 columns, with each column
                # corresponding to a dimension of the original data.
                if (len(shape) == 2) and (shape[1] == 3):
                    df[f"{var}1"] = cdf[var][...][:, 0]
                    df[f"{var}2"] = cdf[var][...][:, 1]
                    df[f"{var}3"] = cdf[var][...][:, 2]
                else:
                    df[var] = cdf[var][...]
            return df

        def _process(self):
            """
            Don't have the quality process.
            Solve the "ns" seconds problem.
            Rename and reindex the time column.
            Resample with 1s.
            Rename some columns.
            :return: preprocessed data
            """
            df = (
                self.original_df.copy()
            )  # because `data_.columns = data_.columns.str.lower()` will change the input data's columns names, for not confused, use `copy`.
            # change "Epoch" from "ns" into "s" (use "proximity principle")
            df["Epoch"] = df["Epoch"].dt.round("s")
            # rename and reindex the time column
            df.rename(columns={"Epoch": "datetime"}, inplace=True)
            # lower case all the columns' names
            df.columns = df.columns.str.lower()
            df.set_index("datetime", inplace=True)
            # 2 same, get average. 1,3 without 2, set 2 np.nan.
            df = df.resample("1s").mean()
            df.rename(
                columns={
                    "delta_b_sc1": "b1_ssm_sc1",
                    "delta_b_sc2": "b1_ssm_sc2",
                    "delta_b_sc3": "b1_ssm_sc3",
                    "delta_b_geo1": "b1_enu1",
                    "delta_b_geo2": "b1_enu2",
                    "delta_b_geo3": "b1_enu3",
                    "b_sc_obs_orig1": "b_ssm_sc_orig1",
                    "b_sc_obs_orig2": "b_ssm_sc_orig2",
                    "b_sc_obs_orig3": "b_ssm_sc_orig3",
                },
                inplace=True,
            )
            return df

    class SSIES3CoupleSSM:
        def __init__(self, file_path_ssies3, file_path_ssm):
            self.file_path_ssies3 = file_path_ssies3
            self.file_path_ssm = file_path_ssm

            self.file_name_ssies3 = os.path.basename(file_path_ssies3)
            self.file_name_ssm = os.path.basename(file_path_ssm)

            assert self._check_same_day_condition(), "ssies3,ssm应为同一天"
            assert self._check_ssies3_orbit_file_not_cross_day(), "ssies3文件不跨天"

            ssies3 = SPDF.SSIES3(file_path_ssies3)
            ssm = SPDF.SSM(file_path_ssm)
            self.ssies3_df = ssies3.df
            self.ssm_df = ssm.df

            self.ssm_df_clip = self._clip_ssm_by_ssies3()

            self.ssies3_ssm_df = self._get_s3_ssm()
            # self.ssies3_ssm_df = self._get_b_enu_columns()

            # self.v_enu = self._s3_sc2enu(self.ssies3_ssm_df['v_s3_sc1'],self.ssies3_ssm_df['v_s3_sc2'],self.ssies3_ssm_df['v_s3_sc3'])
            # self.b_enu = self._s3_sc2enu(self.ssies3_ssm_df['v_s3_sc1'],self.ssies3_ssm_df['v_s3_sc2'],self.ssies3_ssm_df['v_s3_sc3'])

            # self.E_df = self._get_E(self.ssies3_ssm_df[['v_enu1', 'v_enu2', 'v_enu3']], self.ssies3_ssm_df[['b_enu1', 'b_enu2', 'b_enu3']])


        def _check_same_day_condition(self):
            # ssies3
            # 提取日期时间字符串部分
            parts = self.file_name_ssies3.split("_")
            dt_str = parts[3]

            # 解析为datetime对象
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M")

            # 获取日期和时间字符串
            date_str_ssies3 = dt.strftime("%Y%m%d")

            # ssm
            parts = self.file_name_ssm.split("_")
            dt_str = parts[3]
            dt = datetime.strptime(dt_str, "%Y%m%d")
            date_str_ssm = dt.strftime("%Y%m%d")

            is_same_day = date_str_ssies3 == date_str_ssm

            return is_same_day

        def _check_ssies3_orbit_file_not_cross_day(self) -> bool:
            """
            检验ssies3文件是否不跨天（类SSIES3CoupleSSM只能处理不跨天的SSIES文件）
            Returns:

            """
            # 提取日期时间字符串部分
            parts = self.file_name_ssies3.split("_")
            dt_str = parts[3]

            # 解析为datetime对象
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M")

            # 计算加上1轨的时间
            new_dt = dt + DMSPConfigs.SPDF.ssies3_orbit_time

            # 判断是否超过24小时（即是否跨天）
            is_over_24h = new_dt.date() > dt.date()

            return not is_over_24h

        def _clip_ssm_by_ssies3(
            self
        ) -> pd.DataFrame:
            """
            the ssm and the ssies3 data should in the same day.
            :param s3_df: preprocessed ssies3 data (without time preprocess, can not clip ssm well (the time of the clipped ssm data is different from the time of the ssies3 data))
            :param ssm_df: preprocessed ssm data (same)
            :return: clipped ssm data (1d) by ssies3 (1 orbit) time
            """
            s3_df = self.ssies3_df
            ssm_df = self.ssm_df
            s3_datetimes = s3_df.index.values
            ssm_datetimes = ssm_df.index.values
            st_idx = np.where(ssm_datetimes == s3_datetimes[0])[0][0]
            et_idx = np.where(ssm_datetimes == s3_datetimes[-1])[0][0]
            result_df = ssm_df.iloc[st_idx : et_idx + 1]
            result_datetiems = result_df.index.values
            assert np.array_equal(s3_datetimes, result_datetiems)
            return result_df


        def _get_s3_ssm(self) -> pd.DataFrame:
            """
            the last dataframe you use in the experiment.
            :param s3_df: the preprocessed ssies3 data
            :param ssm_df: the clipped ssm data
            :return:
            """
            df = pd.concat([self.ssies3_df,self.ssm_df_clip],axis=1)
            # df = self.ssm_df_clip.copy()  # concat
            # get b1_s3_sc
            b1_s3_sc1, b1_s3_sc2, b1_s3_sc3 = self.ssm_sc2s3_sc(df['b1_ssm_sc1'].values, df['b1_ssm_sc2'].values,
                                                                df['b1_ssm_sc3'].values)
            # df.drop(['b1_ssm_sc1', 'b1_ssm_sc2', 'b1_ssm_sc3'], axis=1, inplace=True)
            df['b1_s3_sc1'] = b1_s3_sc1
            df['b1_s3_sc2'] = b1_s3_sc2
            df['b1_s3_sc3'] = b1_s3_sc3
            # # drop no needed variables
            # df.drop(
            #     ['sc_along_geo1', 'sc_along_geo2', 'sc_along_geo3', 'sc_across_geo1', 'sc_across_geo2',
            #      'sc_across_geo3'],
            #     axis=1, inplace=True)
            # get b_s3_sc_orig
            b_s3_sc_orig1, b_s3_sc_orig2, b_s3_sc_orig3 = self.ssm_sc2s3_sc(df['b_ssm_sc_orig1'].values,
                                                                            df['b_ssm_sc_orig2'].values,
                                                                            df['b_ssm_sc_orig3'].values)
            # df.drop(['b_ssm_sc_orig1', 'b_ssm_sc_orig2', 'b_ssm_sc_orig3'], axis=1, inplace=True)
            df['b_s3_sc_orig1'] = b_s3_sc_orig1
            df['b_s3_sc_orig2'] = b_s3_sc_orig2
            df['b_s3_sc_orig3'] = b_s3_sc_orig3

            # get v_enu columns
            ## qua
            sc_along_geo1 = df['sc_along_geo1'].values
            sc_along_geo2 = df['sc_along_geo2'].values
            sc_across_geo1 = df['sc_across_geo1'].values
            sc_across_geo2 = df['sc_across_geo2'].values
            v_enu1,v_enu2 = self._s3_sc2enu(df['v_s3_sc1'].values,df['v_s3_sc2'].values,sc_along_geo1,sc_along_geo2,sc_across_geo1,sc_across_geo2)
            v_enu3 = df['v_s3_sc3'].values
            df['v_enu1'] = v_enu1
            df['v_enu2'] = v_enu2
            df['v_enu3'] = v_enu3
            # get b_enu columns
            df['b_enu1'] = df['bm_enu1'].values + df['b1_enu1'].values
            df['b_enu2'] = df['bm_enu2'].values + df['b1_enu2'].values
            df['b_enu3'] = df['bm_enu3'].values + df['b1_enu3'].values

            # get E
            E_df_enu = self._get_E(df[['v_enu1', 'v_enu2', 'v_enu3']], df[['b_enu1', 'b_enu2', 'b_enu3']],column_names=['E_enu1','E_enu2','E_enu3'])
            E_df_sc = self._get_E(df[['v_s3_sc1', 'v_s3_sc2', 'v_s3_sc3']], df[['b_s3_sc_orig1','b_s3_sc_orig2','b_s3_sc_orig3']],column_names=['E_s3_sc1','E_s3_sc2','E_s3_sc3'])
            df = pd.concat([df, E_df_enu,E_df_sc], axis=1)

            # return final concat df
            return df

        def ssm_sc2s3_sc(self, com1: np.ndarray, com2: np.ndarray, com3: np.ndarray) -> tuple[
            NDArray,NDArray,NDArray]:
            """
            B is verified
            :param com1: a component of a variable in ssm sc coordinate system
            :param com2: ~
            :param com3: ~
            :return: 3 components of the variable in ssies3 sc coordinate system
            """
            # return pd.DataFrame({'s3_sc1': com2, 's3_sc2': -com3, 's3_sc3': -com1})
            return com2, -com3, -com1

        # def _get_b_enu_columns(self):
        #     self.ssies3_ssm_df['b_enu1'] = self.ssies3_ssm_df['bm_enu1'] + self.ssies3_ssm_df['b1_enu1']
        #     self.ssies3_ssm_df['b_enu2'] = self.ssies3_ssm_df['bm_enu2'] + self.ssies3_ssm_df['b1_enu2']
        #     self.ssies3_ssm_df['b_enu3'] = self.ssies3_ssm_df['bm_enu3'] + self.ssies3_ssm_df['b1_enu3']
        #     return None



        # def _get_s3_sc2enu_change_df(self) -> pd.DataFrame:
        #     """
        #     :param ssm_pre_df: the ssm df should already be preprocessed
        #     :return:
        #     """
        #     return self.ssm_df_clip[
        #         ['sc_along_geo1', 'sc_along_geo2', 'sc_along_geo3', 'sc_across_geo1', 'sc_across_geo2',
        #          'sc_across_geo3']]

        def _s3_sc2enu(self, s3_sc_com1: NDArray, s3_sc_com2: NDArray,sc_along_geo1, sc_along_geo2, sc_across_geo1, sc_across_geo2):
            """
            :param s3_sc_com1: a component of a variable in ssies3 sc coordinate system
            :param s3_sc_com2: ~
            :param com3: ~
            :param along_across_df: the needed transformation DataFrame
            :return: 3 components of the variable in ENU coordinate system
            """
            # along_across_df = self._get_s3_sc2enu_change_df()
            e = (sc_along_geo1 * s3_sc_com1) + (sc_across_geo1 * s3_sc_com2)
            n = (sc_along_geo2 * s3_sc_com1) + (sc_across_geo2 * s3_sc_com2)
            return e,n

        def _get_E(self, v: pd.DataFrame, B: pd.DataFrame,column_names) -> DataFrame:
            """
            pay attention the v and B are in the same coordinate system, and the coordinate system should be relatively stationary with respect to the satellite. For example, the ssies3 coordinate system, the enu coordinate system.
            :param v: velocity of ion
            :param B: measurement magnetic field
            :return:
            """
            assert v.index.dtype == 'datetime64[ns]'
            assert np.all(np.equal(v.index.values, B.index.values))
            return pd.DataFrame(np.cross(v.values, B.values) * 1e-6 * -1, columns=column_names,
                                index=v.index.values)  # todo:: -v x b? 'np.cross()'?
        #
        # def compare_b1(self):
        #     pass  # todo:: compare b1 from different method


def r_madrigal_1s(fp):
    dataset = xr.open_dataset(fp)
    print(dataset)
    return dataset
