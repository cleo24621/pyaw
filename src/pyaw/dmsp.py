import os.path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from pandas import DataFrame
from spacepy.pycdf import CDF

# Select variables in SSIES3 and SSM of SPDF.
SSIES3_VARS = [
    "Epoch",
    "glat",
    "glon",
    "alt",
    "vx",
    "vxqual",
    "vy",
    "vyqual",
    "vz",
    "vzqual",
    "temp",
    "tempqual",
    "frach",
    "frachqual",
    "frache",
    "frachequal",
    "fraco",
    "fracoqual",
    "bx",
    "by",
    "bz",
    "ductdens",
    "te",
]
SSM_VARS = [
    "Epoch",
    "SC_GEOCENTRIC_LAT",
    "SC_GEOCENTRIC_LON",
    "SC_GEOCENTRIC_R",
    "SC_AACGM_LAT",
    "SC_AACGM_LON",
    "SC_AACGM_LTIME",
    "B_SC_OBS_ORIG",
    "DELTA_B_GEO",
    "DELTA_B_SC",
    "SC_ALONG_GEO",
    "AURORAL_REGION",
    "ORBIT_INDEX",
    "AURORAL_BOUNDARY_FOM",
    "SC_ACROSS_GEO",
]

# f17星的 v_qual 的坏对应的只有 4?
QUALITY_INDICES = {
    "vx_qual_filter": 4,
    "vy_qual_filter": 4,
    "vz_qual_filter": 4,
    "frac_qual_filter": 4,
    "temp_qual_filter": 4,
}

# valid min and max
VALID_VALUES = {
    "vx_valid_value": 2000,
    "vy_valid_value": 2000,
    "vz_valid_value": 2000,
    # number density of ion
    "ductdens_valid_value_min": 0,
    "ductdens_valid_value_max": 1e8,
    "frac_valid_value_min": 0,
    "frac_valid_value_max": 1.05,
    "temp_valid_value_min": 500,
    "temp_valid_value_max": 2e4,
    "te_valid_value_min": 500,
    "te_valid_value_max": 1e4,
}

# Slightly larger than the real orbit time.
SSIES3_ORBIT_TIME = timedelta(hours=1, minutes=45)


class SPDF:
    class SSIES3:
        def __init__(self, file_path):
            self.file_path = file_path
            self.original_df = self._read()

            # DataFrame after quality process and rename, reindex ...
            self.df = self._other_process()

        def _read(self) -> DataFrame:
            """
            Read selected original data from SSIES3 CDF file.
            """
            cdf = CDF(self.file_path)

            # read data
            df = pd.DataFrame()
            for var_name in cdf.keys():
                if var_name not in SSIES3_VARS:
                    continue
                df[var_name] = cdf[var_name][...]

            return df

        def _quality_process(self) -> pd.DataFrame:
            """
            Use quality flag and valid value range to do quality process.
            Rename and Reindex the time column for the unified types.
            Rename and drop some columns.
            Returns:
                Preprocessed data.
            """
            df = self.original_df.copy()

            # velocity
            df.loc[
                (df["vxqual"] == QUALITY_INDICES["vx_qual_filter"])
                | (df["vx"].abs() > VALID_VALUES["vx_valid_value"]),
                "vx",
            ] = np.nan
            df.loc[
                (df["vyqual"] == QUALITY_INDICES["vy_qual_filter"])
                | (df["vy"].abs() > VALID_VALUES["vy_valid_value"]),
                "vy",
            ] = np.nan
            df.loc[
                (df["vzqual"] == QUALITY_INDICES["vz_qual_filter"])
                | (df["vz"].abs() > VALID_VALUES["vz_valid_value"]),
                "vz",
            ] = np.nan

            # ductdens (number density of ion)
            df.loc[
                (df["ductdens"] > VALID_VALUES["ductdens_valid_value_max"])
                | (df["ductdens"] < VALID_VALUES["ductdens_valid_value_min"]),
                "ductdens",
            ] = np.nan

            # fraco, frache, frach
            # 因为变量说明中明确说明应该舍弃的异常值，所以没有使用[valid_value_min, valid_value_max]?
            df.loc[
                (df["fraco"] > VALID_VALUES["frac_valid_value_max"])
                | (df["fraco"] < VALID_VALUES["frac_valid_value_min"])
                | (df["fracoqual"] == QUALITY_INDICES["frac_qual_filter"]),
                "fraco",
            ] = np.nan
            df.loc[
                (df["frach"] > VALID_VALUES["frac_valid_value_max"])
                | (df["frach"] < VALID_VALUES["frac_valid_value_min"])
                | (df["frachqual"] == QUALITY_INDICES["frac_qual_filter"]),
                "frach",
            ] = np.nan
            df.loc[
                (df["frache"] > VALID_VALUES["frac_valid_value_max"])
                | (df["frache"] < VALID_VALUES["frac_valid_value_min"])
                | (df["frachequal"] == QUALITY_INDICES["frac_qual_filter"]),
                "frache",
            ] = np.nan

            # temp (ion temperature)
            df.loc[
                (df["temp"] > VALID_VALUES["temp_valid_value_max"])
                | (df["temp"] < VALID_VALUES["temp_valid_value_min"])
                | (df["tempqual"] == QUALITY_INDICES["temp_qual_filter"]),
                "temp",
            ] = np.nan

            # te (electron temperature)
            df.loc[
                (df["te"] > VALID_VALUES["te_valid_value_max"])
                | (df["te"] < VALID_VALUES["te_valid_value_min"]),
                "te",
            ] = np.nan
            return df

        def _other_process(self):
            # （没有使用.copy()）质量控制后返回的DataFrame的视图
            df = self._quality_process()

            # rename
            df.rename(columns={"Epoch": "datetime"}, inplace=True)

            # reindex
            df.set_index("datetime", inplace=True)

            # interpolate Nan
            # Note that SSIES3 data may have no data at some time points.
            df = df.resample("1s").mean()

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

        def _read(self) -> pd.DataFrame:
            """
            Read selected original data from SSM CDF file.
            """
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
                if var not in SSM_VARS:
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
            No quality process.
            Solve the "ns" seconds problem.
            Rename and reindex the time column.
            Resample with 1s.
            Rename some columns.
            Returns:
                Preprocessed data.
            """
            # Because `data_.columns = data_.columns.str.lower()` will change the input data's columns names.
            # For not confused, use `copy`.
            df = self.original_df.copy()

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

        def _check_same_day_condition(self):
            """
            Check if the SSIES3 and SSM files are from the same day.
            Returns:

            """
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
            检验ssies3文件是否不跨天（类SSIES3CoupleSSM只能处理不跨天的SSIES文件）.
            Returns:

            """
            # 提取日期时间字符串部分
            parts = self.file_name_ssies3.split("_")
            dt_str = parts[3]

            # 解析为datetime对象
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M")

            # 计算加上1轨的时间
            new_dt = dt + SSIES3_ORBIT_TIME

            # 判断是否超过24小时（即是否跨天）
            is_over_24h = new_dt.date() > dt.date()

            return not is_over_24h

        def _clip_ssm_by_ssies3(self) -> pd.DataFrame:
            """Clip the SSM data by the SSIES3 time range.

            Returns:
                Clipped SSM data.
            """
            s3_df = self.ssies3_df
            ssm_df = self.ssm_df
            s3_datetimes = s3_df.index.values
            ssm_datetimes = ssm_df.index.values
            st_idx = np.where(ssm_datetimes == s3_datetimes[0])[0][0]
            et_idx = np.where(ssm_datetimes == s3_datetimes[-1])[0][0]
            result_df = ssm_df.iloc[st_idx : et_idx + 1]
            result_datetimes = result_df.index.values

            assert np.array_equal(s3_datetimes, result_datetimes)

            return result_df

        def _get_s3_ssm(self) -> pd.DataFrame:
            """The final preprocessed dataframe of SSIES3 and SSM data."""
            df = pd.concat([self.ssies3_df, self.ssm_df_clip], axis=1)
            # df = self.ssm_df_clip.copy()  # concat

            # get b1_s3_sc
            b1_s3_sc1, b1_s3_sc2, b1_s3_sc3 = self.ssm_sc2s3_sc(
                df["b1_ssm_sc1"].values,
                df["b1_ssm_sc2"].values,
                df["b1_ssm_sc3"].values,
            )

            # df.drop(['b1_ssm_sc1', 'b1_ssm_sc2', 'b1_ssm_sc3'], axis=1, inplace=True)
            df["b1_s3_sc1"] = b1_s3_sc1
            df["b1_s3_sc2"] = b1_s3_sc2
            df["b1_s3_sc3"] = b1_s3_sc3
            # # drop no needed variables
            # df.drop(
            #     ['sc_along_geo1', 'sc_along_geo2', 'sc_along_geo3', 'sc_across_geo1', 'sc_across_geo2',
            #      'sc_across_geo3'],
            #     axis=1, inplace=True)

            # get b_s3_sc_orig
            b_s3_sc_orig1, b_s3_sc_orig2, b_s3_sc_orig3 = self.ssm_sc2s3_sc(
                df["b_ssm_sc_orig1"].values,
                df["b_ssm_sc_orig2"].values,
                df["b_ssm_sc_orig3"].values,
            )
            # df.drop(['b_ssm_sc_orig1', 'b_ssm_sc_orig2', 'b_ssm_sc_orig3'], axis=1, inplace=True)
            df["b_s3_sc_orig1"] = b_s3_sc_orig1
            df["b_s3_sc_orig2"] = b_s3_sc_orig2
            df["b_s3_sc_orig3"] = b_s3_sc_orig3

            # get v_enu columns
            ## qua
            sc_along_geo1 = df["sc_along_geo1"].values
            sc_along_geo2 = df["sc_along_geo2"].values
            sc_across_geo1 = df["sc_across_geo1"].values
            sc_across_geo2 = df["sc_across_geo2"].values
            v_enu1, v_enu2 = self._s3_sc2enu(
                df["v_s3_sc1"].values,
                df["v_s3_sc2"].values,
                sc_along_geo1,
                sc_along_geo2,
                sc_across_geo1,
                sc_across_geo2,
            )
            v_enu3 = df["v_s3_sc3"].values
            df["v_enu1"] = v_enu1
            df["v_enu2"] = v_enu2
            df["v_enu3"] = v_enu3

            # get b_enu columns
            df["b_enu1"] = df["bm_enu1"].values + df["b1_enu1"].values
            df["b_enu2"] = df["bm_enu2"].values + df["b1_enu2"].values
            df["b_enu3"] = df["bm_enu3"].values + df["b1_enu3"].values

            # get E
            e_df_enu = self._calculate_electric_field(
                df[["v_enu1", "v_enu2", "v_enu3"]],
                df[["b_enu1", "b_enu2", "b_enu3"]],
                column_names=["E_enu1", "E_enu2", "E_enu3"],
            )
            e_df_sc = self._calculate_electric_field(
                df[["v_s3_sc1", "v_s3_sc2", "v_s3_sc3"]],
                df[["b_s3_sc_orig1", "b_s3_sc_orig2", "b_s3_sc_orig3"]],
                column_names=["E_s3_sc1", "E_s3_sc2", "E_s3_sc3"],
            )
            df = pd.concat([df, e_df_enu, e_df_sc], axis=1)

            return df

        @staticmethod
        def ssm_sc2s3_sc(
            com1: NDArray, com2: NDArray, com3: NDArray
        ) -> tuple[NDArray, NDArray, NDArray]:
            """Transform SSM s/c coordinate system to SSIES3 s/c coordinate system. Magnetic field is verified.

            Args:
                com1: a component of a variable in ssm sc coordinate system.
                com2: ~
                com3: ~

            Returns:
                Three components of the variable in SSIES3 s/c coordinate system.
            """
            # return pd.DataFrame({'s3_sc1': com2, 's3_sc2': -com3, 's3_sc3': -com1})
            return com2, -com3, -com1

        @staticmethod
        def _s3_sc2enu(
            s3_sc_com1: NDArray,
            s3_sc_com2: NDArray,
            sc_along_geo1,
            sc_along_geo2,
            sc_across_geo1,
            sc_across_geo2,
        ):
            """Transform SSIES3 variable data from s/c coordinate system to enu coordinate system.

            Args:
                s3_sc_com1: a component of a variable in ssies3 sc coordinate system.
                s3_sc_com2: ~

            Returns:
                Three components of the variable data in enu coordinate system.
            """
            # along_across_df = self._get_s3_sc2enu_change_df()
            e = (sc_along_geo1 * s3_sc_com1) + (sc_across_geo1 * s3_sc_com2)
            n = (sc_along_geo2 * s3_sc_com1) + (sc_across_geo2 * s3_sc_com2)
            return e, n

        @staticmethod
        def _calculate_electric_field(
            v: pd.DataFrame, magnetic_field: pd.DataFrame, column_names
        ) -> DataFrame:
            """
            The velocity of ion and the magnetic field are in the same coordinate system,
            such as the ssies3 s/c coordinate system and the enu coordinate system.

            Args:
                v: velocity of ion.
                magnetic_field: measurement magnetic field.

            Returns:
                Electric field in the same coordinate system as the input velocity and magnetic field.
            """
            assert v.index.dtype == "datetime64[ns]"
            assert np.all(np.equal(v.index.values, magnetic_field.index.values))

            return pd.DataFrame(
                np.cross(v.values, magnetic_field.values) * 1e-6 * -1,
                columns=column_names,
                index=v.index.values,
            )


def r_madrigal_1s(fp):
    """Read a Madrigal 1s file and return a xarray Dataset."""
    dataset = xr.open_dataset(fp)
    print(dataset)
    return dataset
