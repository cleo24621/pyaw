# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/22
@DESCRIPTION: 
"""
import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from spacepy.pycdf import CDF

from pyaw.configs import s3_vars, ssm_vars


# class
# field
# fill value
# method
# read
# preprocess
class SPDF:
    """
    baseline corrected data of ssm data is not parallel translation. can use plot to see the effect.
    todo:: unit conversion
    """

    def __init__(self):
        self._vx_qual_filter = 4  # quality filter of v (velocity of ion?)  todo
        self._vy_qual_filter = 4
        self._vz_qual_filter = 4
        self._vx_valid_value = 2000  # valid min and max of v
        self._vy_valid_value = 2000
        self._vz_valid_value = 2000
        self._ductdens_valid_value_min = 0  # number density of ion
        self._ductdens_valid_value_max = 1e8
        self._frac_qual_filter = 4  # fraction of different ions
        self._frac_valid_value_min = 0
        self._frac_valid_value_max = 1.05
        self._temp_qual_filter = 4  # ion temperature
        self._temp_valid_value_min = 500
        self._temp_valid_value_max = 2e4
        self._te_valid_value_min = 500  # electron temperature
        self._te_valid_value_max = 1e4

    def r_s3(self, fp: str, vars_=s3_vars) -> pd.DataFrame:
        """
        将spdf托管的ssies3载荷数据以pd.DataFrame的格式返回
        :param vars_: read variables that you need
        :param fp: path of ssies3 file
        :return: original data with specified variables
        """
        cdf = CDF(fp)
        # read data
        df = pd.DataFrame()
        for var_name in cdf.keys():
            if var_name not in vars_:
                continue
            df[var_name] = cdf[var_name][...]
        return df

    def s3_pre(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        quality filter and valid value filter.
        rename and reindex the time column.
        rename and drop some columns.
        :param df: original data
        :return: preprocessed data
        """
        df_ = df.copy()
        # velocity
        # pay attention: f17星的 vqual 的坏对应的只有 4
        df_.loc[(df_['vxqual'] == self._vx_qual_filter) | (df_['vx'].abs() > self._vx_valid_value), 'vx'] = np.nan
        df_.loc[(df_['vyqual'] == self._vy_qual_filter) | (df_['vy'].abs() > self._vy_valid_value), 'vy'] = np.nan
        df_.loc[(df_['vzqual'] == self._vz_qual_filter) | (df_['vz'].abs() > self._vz_valid_value), 'vz'] = np.nan
        # ductdens (number density of ion)
        df_.loc[(df_['ductdens'] > self._ductdens_valid_value_max) | (
                df_['ductdens'] < self._ductdens_valid_value_min), 'ductdens'] = np.nan
        # fraco, frache, frach
        # 因为变量说明中明确说明应该舍弃的异常值，所以没有使用[validmin, validmax]
        df_.loc[(df_['fraco'] > self._frac_valid_value_max) | (df_['fraco'] < self._frac_valid_value_min) | (
                df_['fracoqual'] == self._frac_qual_filter), 'fraco'] = np.nan
        df_.loc[(df_['frach'] > self._frac_valid_value_max) | (df_['frach'] < self._frac_valid_value_min) | (
                df_['frachqual'] == self._frac_qual_filter), 'frach'] = np.nan
        df_.loc[(df_['frache'] > self._frac_valid_value_max) | (df_['frache'] < self._frac_valid_value_min) | (
                df_['frachequal'] == self._frac_qual_filter), 'frache'] = np.nan
        # temp (ion temperature)
        df_.loc[(df_['temp'] > self._temp_valid_value_max) | (df_['temp'] < self._temp_valid_value_min) | (
                df_['tempqual'] == self._temp_qual_filter), 'temp'] = np.nan
        # te (electron temperature)
        df_.loc[(df_['te'] > self._te_valid_value_max) | (df_['te'] < self._te_valid_value_min), 'te'] = np.nan
        df_.rename(columns={'Epoch': 'datetime'}, inplace=True)
        df_.set_index('datetime', inplace=True)
        df_ = df_.resample('1s').mean()  # ssies3 data may have some time that there is not data.
        assert not df_.index.duplicated().any(), "duplicated datetime"
        df_.drop(['vxqual', 'vyqual', 'vzqual', 'tempqual', 'frachqual', 'frachequal', 'fracoqual'], axis=1,
                 inplace=True)
        df_.rename(columns={'vx': 'v_s3_sc1', 'vy': 'v_s3_sc2', 'vz': 'v_s3_sc3'}, inplace=True)
        if all([e in df_.columns for e in ['bx', 'by', 'bz']]):
            df_['bz'] = -df_['bz']  # change down to up
            df_.rename(columns={'bx': 'bm_enu2', 'by': 'bm_enu1', 'bz': 'bm_enu3'}, inplace=True)  # 'm' means IGRF model
        return df_

    def r_ssm(self, fp: str, vars_=ssm_vars) -> pd.DataFrame:
        """
        将spdf托管的ssm载荷数据以pd.DataFrame的格式返回
        :param vars_: read variables that you need
        :param fp: path of ssm file
        :return: original data
        """
        # CDF instance
        cdf = CDF(fp)
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
            if var not in vars_:
                continue
            # if the data is 3-dimensional, decompose it into 3 columns, with each column
            # corresponding to a dimension of the original data.
            if (len(shape) == 2) and (shape[1] == 3):
                df[f'{var}1'] = cdf[var][...][:, 0]
                df[f'{var}2'] = cdf[var][...][:, 1]
                df[f'{var}3'] = cdf[var][...][:, 2]
            else:
                df[var] = cdf[var][...]
        return df

    def ssm_pre(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        solve the "ns" seconds problem.
        rename and reindex the time column.
        resample with 1s.
        rename some columns.
        :param df: original data
        :return: preprocessed data
        """
        df_ = df.copy()  # because `data_.columns = data_.columns.str.lower()` will change the input data's columns names, for not confused, use `copy`.
        # change "Epoch" from "ns" into "s" (use "proximity principle")
        df_['Epoch'] = df_['Epoch'].dt.round('s')
        # rename and reindex the time column
        df_.rename(columns={'Epoch': 'datetime'}, inplace=True)
        # lower case all the columns' names
        df_.columns = df_.columns.str.lower()
        df_.set_index('datetime', inplace=True)
        # 2 same, get average. 1,3 without 2, set 2 np.nan.
        df_ = df_.resample('1s').mean()
        df_.rename(columns={'delta_b_sc1': 'b1_ssm_sc1', 'delta_b_sc2': 'b1_ssm_sc2', 'delta_b_sc3': 'b1_ssm_sc3',
                            'delta_b_geo1': 'b1_enu1', 'delta_b_geo2': 'b1_enu2', 'delta_b_geo3': 'b1_enu3',
                            'b_sc_obs_orig1': 'b_ssm_sc_orig1', 'b_sc_obs_orig2': 'b_ssm_sc_orig2',
                            'b_sc_obs_orig3': 'b_ssm_sc_orig3'}, inplace=True)
        return df_

    def clip_ssm_by_ssies3(self, s3_df: pd.DataFrame, ssm_df: pd.DataFrame) -> pd.DataFrame:
        """
        the ssm and the ssies3 data should in the same day.
        :param s3_df: preprocessed ssies3 data (without time preprocess, can not clip ssm well (the time of the clipped ssm data is different from the time of the ssies3 data))
        :param ssm_df: preprocessed ssm data (same)
        :return: clipped ssm data (1d) by ssies3 (1 orbit) time
        """
        st_idx = np.where(ssm_df.index == s3_df.index[0])[0][0]
        et_idx = np.where(ssm_df.index == s3_df.index[-1])[0][0]
        result_df = ssm_df.iloc[st_idx:et_idx + 1]
        assert s3_df.index.equals(result_df.index)
        return result_df

    def get_s3_ssm(self, s3_df: pd.DataFrame, ssm_df: pd.DataFrame) -> pd.DataFrame:
        """
        the last dataframe you use in the experiment.
        :param s3_df: the preprocessed ssies3 data
        :param ssm_df: the clipped ssm data
        :return:
        """
        df = ssm_df.copy()
        # get b1_s3_sc
        b1_s3_sc1, b1_s3_sc2, b1_s3_sc3 = self.ssm_sc2s3_sc(df['b1_ssm_sc1'], df['b1_ssm_sc2'],
                                                            df['b1_ssm_sc3'])
        df.drop(['b1_ssm_sc1', 'b1_ssm_sc2', 'b1_ssm_sc3'], axis=1, inplace=True)
        df['b1_s3_sc1'] = b1_s3_sc1
        df['b1_s3_sc2'] = b1_s3_sc2
        df['b1_s3_sc3'] = b1_s3_sc3
        # drop no needed variables
        df.drop(
            ['sc_along_geo1', 'sc_along_geo2', 'sc_along_geo3', 'sc_across_geo1', 'sc_across_geo2', 'sc_across_geo3'],
            axis=1, inplace=True)
        # get b_s3_sc_orig
        b_s3_sc_orig1, b_s3_sc_orig2, b_s3_sc_orig3 = self.ssm_sc2s3_sc(df['b_ssm_sc_orig1'],
                                                                        df['b_ssm_sc_orig2'],
                                                                        df['b_ssm_sc_orig3'])
        df.drop(['b_ssm_sc_orig1', 'b_ssm_sc_orig2', 'b_ssm_sc_orig3'], axis=1, inplace=True)
        df['b_s3_sc_orig1'] = b_s3_sc_orig1
        df['b_s3_sc_orig2'] = b_s3_sc_orig2
        df['b_s3_sc_orig3'] = b_s3_sc_orig3
        # return final concat df
        return pd.concat([s3_df, df], axis=1)

    def ssm_sc2s3_sc(self, com1: pd.Series, com2: pd.Series, com3: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        B is verified
        :param com1: a component of a variable in ssm sc coordinate system
        :param com2: ~
        :param com3: ~
        :return: 3 components of the variable in ssies3 sc coordinate system
        """
        # return pd.DataFrame({'s3_sc1': com2, 's3_sc2': -com3, 's3_sc3': -com1})
        return com2, -com3, -com1

    def get_s3_sc2enu_change_df(self, ssm_pre_df: pd.DataFrame) -> pd.DataFrame:
        """
        :param ssm_pre_df: the ssm df should already be preprocessed
        :return:
        """
        return ssm_pre_df[
            ['sc_along_geo1', 'sc_along_geo2', 'sc_along_geo3', 'sc_across_geo1', 'sc_across_geo2', 'sc_across_geo3']]

    def s3_sc2enu(self, com1: pd.Series, com2: pd.Series, com3: pd.Series, along_across: pd.DataFrame) -> tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        :param com1: a component of a variable in ssies3 sc coordinate system
        :param com2: ~
        :param com3: ~
        :param along_across: the needed transformation DataFrame
        :return: 3 components of the variable in ENU coordinate system
        """
        e = (along_across['sc_along_geo1'] * com1) + (along_across['sc_across_geo1'] * com2)
        n = (along_across['sc_along_geo2'] * com1) + (along_across['sc_across_geo2'] * com2)
        u = com3
        return e, n, u

    def get_E(self, v: pd.DataFrame, B: pd.DataFrame) -> DataFrame:
        """
        pay attention the v and B are in the same coordinate system, and the coordinate system should be relatively stationary with respect to the satellite. For example, the ssies3 coordinate system, the enu coordinate system.
        :param v: velocity of ion
        :param B: measurement magnetic field
        :return:
        """
        assert v.index.dtype == 'datetime64[ns]'
        assert np.all(np.equal(v.index.values,B.index.values))
        return pd.DataFrame(np.cross(v.values, B.values) * 1e-6 * -1, columns=['1', '2', '3'],index=v.index.values)  # todo:: -v x b? 'np.cross()'?

    def compare_b1(self):
        pass  # todo:: compare b1 from different method


def r_madrigal_1s(fp):
    dataset = xr.open_dataset(fp)
    print(dataset)
    return dataset


# fp_s3 = r"D:\cleo\master\pyaw\data\dmsp-f18_ssies-3_thermal-plasma_201401010124_v01.cdf"
# fp_ssm = r"D:\cleo\master\pyaw\data\dmsp-f18_ssm_magnetometer_20140101_v1.0.4.cdf"
# spdf = SPDF()
# s3_df = spdf.r_s3(fp_s3)
# s3_df_pre = spdf.s3_pre(s3_df)
# ssm_df = spdf.r_ssm(fp_ssm)
# ssm_df_pre = spdf.ssm_pre(ssm_df)
# clipped_ssm_df = spdf.clip_ssm_by_ssies3(s3_df_pre, ssm_df_pre)
# s3_ssm_df = spdf.get_s3_ssm(s3_df_pre, clipped_ssm_df)
# spdf.get_E(s3_ssm_df[['v_s3_sc1', 'v_s3_sc2', 'v_s3_sc3']],
#            s3_ssm_df[['b_s3_sc_orig1', 'b_s3_sc_orig2', 'b_s3_sc_orig3']])

# plt.figure()
# assert clipped_ssm_d.index.equals(s3_d_pre.index)
# time = clipped_ssm_d.index
# plt.plot(time,s3_d_pre['glat'],time,clipped_ssm_d['sc_geocentric_lat'])
# plt.show()

# plt.figure()
# time = s3_ssm_df.index
# plt.plot(time, s3_ssm_df['bx'])
# plt.show()
