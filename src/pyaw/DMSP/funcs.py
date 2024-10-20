# -*- coding: utf-8 -*-
"""
@File        : funcs.py
@Author      : 13927
@Date        : 2024/9/14 18:15
@Project     : pyWave
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2024, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
@Last Modified Date: 2024/9/14 18:15
"""

import os

import numpy as np
import pandas as pd
import spacepy.pycdf

import plotly.express as px
import matplotlib.pyplot as plt

# basic parameter

mu0 = 4 * np.pi * 1e-7  # 真空磁导率 （维基）
epsilon0 = 8.854187817e-12  # 真空介电常数 （维基）
kB = 1.38e-23  # 玻尔兹曼常数 （维基）
me = 9.1093837139e-31  # kg
m_atom = 1.66054e-27  # kg
mo = m_atom * 16
mh = m_atom
mhe = m_atom * 4
me_mo_ratio = me / mo

# basic data process

# ssies3

def ssies3(fp):
    """
    return the data of ssies3.
    :return:
    """
    # path
    # CDF example
    cdf = spacepy.pycdf.CDF(fp)
    # read data
    data = pd.DataFrame()
    var_names = cdf.keys()
    for var_name in var_names:
        var = cdf[var_name][...]
        data[var_name] = var
    data.set_index('Epoch', inplace=True)
    return data


def process_ssies3_qual_ValidValue(data):
    # velocity
    # f17星的vqual的坏对应的只有4
    data.loc[(data['vxqual'] == 4) | (data['vx'].abs() > 2000), 'vx'] = np.nan
    data.loc[(data['vyqual'] == 4) | (data['vy'].abs() > 2000), 'vy'] = np.nan
    data.loc[(data['vzqual'] == 4) | (data['vz'].abs() > 2000), 'vz'] = np.nan
    # ductdens (离子数密度)
    data.loc[(data['ductdens'] > 1e8) | (
        data['ductdens'] < 0), 'ductdens'] = np.nan
    # fraco, frache, frach
    # 因为变量说明中明确说明应该舍弃的异常值，所以没有使用[validmin, validmax]
    data.loc[(data['fraco'] > 1.05) | (data['fraco'] < 0)
             | (data['fracoqual'] == 4), 'fraco'] = np.nan
    data.loc[(data['frach'] > 1.05) | (data['frach'] < 0)
             | (data['frachqual'] == 4), 'frach'] = np.nan
    data.loc[(data['frache'] > 1.05) | (data['frache'] < 0)
             | (data['frachequal'] == 4), 'frache'] = np.nan
    # temp (离子温度)
    data.loc[(data['temp'] > 2e4) | (data['temp'] < 500)
             | (data['tempqual'] == 4), 'temp'] = np.nan
    # te (电子温度)
    data.loc[(data['te'] > 1e4) | (data['te'] < 500), 'te'] = np.nan

    return data


def process_ssies3_time(data):
    """
    填充缺失时刻对应的数据为nan
    :param data:
    :return:
    """
    # 如果data的epoch索引列有重复的时间，则报错
    assert not data.index.duplicated().any()
    return data.resample('1s').mean()


def process_ssies3_data(fp):
    return process_ssies3_time(process_ssies3_qual_ValidValue(ssies3(fp)))


# ssm

def ssm(fp):
    """
    返回SSM的原始数据（不含labels变量）（格式为pd.DataFrame）
    """
    # CDF example
    cdf = spacepy.pycdf.CDF(fp)
    # read data
    # delete data like [x,y,z]
    var_shape = {}
    for var in cdf:
        if len(cdf[var].shape) == 1 and cdf[var].shape[0] == 3:
            continue
        var_shape[var] = cdf[var].shape
    # load the data into a df
    data = pd.DataFrame()
    for var, shape in var_shape.items():
        # if the data is 3-dimensional, decompose it into 3 columns, with each column
        # corresponding to a dimension of the original data.
        if (len(shape) == 2) and (shape[1] == 3):
            data[f'{var}_x'] = cdf[var][...][:, 0]
            data[f'{var}_y'] = cdf[var][...][:, 1]
            data[f'{var}_z'] = cdf[var][...][:, 2]
        else:
            data[var] = cdf[var][...]
    return data


def process_ssm_data_reduced(data):
    """
    将dataframe根据B_SC_OBS_ORIG_x列开始数据不为nan以及结尾数据不为nan缩减
    :param data: 原始ssm数据
    :return: pd.DataFrame
    """
    # 找到B_SC_OBS_ORIG_x列中第一个和最后一个非NaN的索引
    start_idx = data['B_SC_OBS_ORIG_x'].first_valid_index()
    end_idx = data['B_SC_OBS_ORIG_x'].last_valid_index()
    # 缩减 DataFrame，只保留B_SC_OBS_ORIG_x列中间的非NaN部分
    return data.loc[start_idx:end_idx].reset_index(drop=True)


def process_ssm_data_time1(data):
    """
    epoch 从 ms -> s 的相关处理
    :param data:
    :return:
    """
    data['Epoch'] = pd.to_datetime(
        data['Epoch']).astype(
        np.int64) / 1e9  # 将 ns 转换为 s
    # 对时间戳进行四舍五入到秒级
    data['Epoch'] = data['Epoch'].round(0)
    # 获取除了 'Epoch' 之外的其他列
    columns_to_aggregate = data.columns.difference(['Epoch'])
    # 按照四舍五入后的时间点进行分组，并对所有其他列取均值
    return data.groupby('Epoch')[columns_to_aggregate].mean().reset_index()


def process_ssm_data_time2(data):
    """
    填充缺失时刻对应的数据为nan
    :param data:
    :return:
    """
    # 将 Epoch 列转换为 pandas 的 Datetime 格式
    data['Epoch'] = pd.to_datetime(data['Epoch'], unit='s')
    # 将 Epoch 列设置为索引
    data.set_index('Epoch', inplace=True)

    # 使用 resample 以 1 秒为间隔生成完整时间序列，并进行插值
    ssm_data_resampled = data.resample('1s').mean()  # 生成 1 秒间隔的时间序列
    return ssm_data_resampled.interpolate(
        method='linear')  # 插值方法，可以选择 'linear' 或其他


def process_ssm_data(fp):
    """
    按照我的编写逻辑调用处理函数，得到处理后的ssm数据
    :param data:
    :return:
    """
    data = process_ssm_data_time2(
        process_ssm_data_time1(
            process_ssm_data_reduced(ssm(fp))))
    data.reset_index()
    # data.set_index('Epoch', inplace=True)
    return data


# advanced data process

def clip_ssm_by_ssies3(ssies3_data, ssm_data):

    st_idx = np.where(ssm_data.index == ssies3_data.index[0])[0][0]
    et_idx = np.where(ssm_data.index == ssies3_data.index[-1])[0][0]
    result = ssm_data.iloc[st_idx:et_idx + 1]
    assert ssies3_data.index.equals(result.index)
    return result



# calculate

def calculate_beta(n,T,B):
    return (2 * mu0 * kB * n * 1e6 * T) / ((B * 1e-9) ** 2)


def calculate_vA(B0,n,fraco,frach,frache):
    """
    Many NAN values are introduced when frach and frache are considered. So vA will have many NaN.
    :param B0: [DataFrame]
    :param n: [Series]
    :param fraco: [Series]
    :param frach: [Series]
    :param frache: [Series]
    :param mo: [float]
    :param mh: [float]
    :param mhe: [float]
    :return: [Series]
    """
    _ = {}
    denominator = np.sqrt(mu0 * (n * 1e6) * (fraco * mo + frach * mh + frache * mhe))
    for c in ['1','2','3']:
        _[c] = (B0[c] * 1e-9) / denominator
    return pd.DataFrame(_)



# coordinate

def ssm_sc_to_ssies_sc(para_x,para_y,para_z):
    """

    :param para_x: [Series] ssm parameter x component in ssm sc coordinate (B is verified, others not)
    :param para_y: ~
    :param para_z: ~
    :return: DataFrame with columns = ['x','y','z']
    """
    _ = {'1':para_y,'2':-para_z,'3':-para_x}
    return pd.DataFrame(_)


def ssies_sc_to_ENU(para_x,para_y,para_z,ssm_data_clip):
    """

    :param para_x: [Series] ssm parameter x component in ssies sc coordinate (B is verified, others not)
    :param para_y: ~
    :param para_z: ~
    :param along_GEO_x: ssm  'SC_ALONG_GEO_x'
    :param along_GEO_y: ~
    :param across_GEO_x: ~
    :param across_GEO_y: ~
    :return: DataFrame with columns = ['E','N','U']
    """
    (along_GEO_x,along_GEO_y,across_GEO_x,across_GEO_y) = (ssm_data_clip['SC_ALONG_GEO_x'],ssm_data_clip['SC_ALONG_GEO_y'],
    ssm_data_clip['SC_ACROSS_GEO_x'], ssm_data_clip['SC_ACROSS_GEO_y'])
    E = (along_GEO_x * para_x) + (across_GEO_x * para_y)
    N = (along_GEO_y * para_x) + (across_GEO_y * para_y)
    U = para_z
    _ = {'E':E,'N':N,'U':U}
    return pd.DataFrame(_)




# draw


def draw_beta(dates, values):
    """
    :param dates: DatetimeIndex
    :param values: Series
    :return:
    """
    # 创建 DataFrame
    df = pd.DataFrame({'Date': dates, 'Values': values})

    # 指定的横线值
    specified_value = me_mo_ratio

    # 使用 plotly 绘制交互式时间序列图
    fig = px.line(df, x='Date', y='Values', title='Plasma beta')

    # 添加横线 (y=指定值)
    fig.add_shape(
        type="line",
        x0=df['Date'].min(),  # 起始 x 位置
        y0=specified_value,  # 起始 y 位置
        x1=df['Date'].max(),  # 结束 x 位置
        y1=specified_value,  # 结束 y 位置
        line=dict(color="Red", width=2, dash="dash"),  # 可以设置线条样式
    )

    # 添加注释，标注横线的值
    fig.add_annotation(
        x=df['Date'].max(),  # 标注位置的 x 坐标
        y=specified_value,  # 标注位置的 y 坐标
        text=f"m_e/m_o",  # 显示文本
        showarrow=False,
        yshift=10  # 调整标注的垂直偏移
    )

    # 自定义 x 轴和 y 轴标题
    fig.update_layout(
        xaxis_title="Time (UT)",  # 自定义 x 轴标题
        yaxis_title="Beta",  # 自定义 y 轴标题
    )

    # 显示图表
    fig.show()


def draw_frac_ions(time, fraco, frach, frache, vlines):
    """

    :param time:
    :param fraco:
    :param frach:
    :param frache:
    :param vlines:  like "pd.to_datetime(['2014-01-01 08:45:00', '2014-01-01 09:15:00', '2014-01-01 09:45:00'])"
    :return:
    """
    y1 = fraco
    y2 = frach
    y3 = frache

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制三条线
    plt.plot(time, y1, label='fraco', color='blue')
    plt.plot(time, y2, label='frach', color='orange')
    plt.plot(time, y3, label='frache', color='green')

    # 添加图例
    plt.legend()

    # 添加竖线
    for vline in vlines:
        plt.axvline(x=vline, color='red', linestyle='--', label=f'Vertical line at {vline}')

    # 设置X轴和Y轴标签
    plt.xlabel('Time (UT)')
    plt.ylabel('fraction of ion')
    plt.title("fraction of O+, H+ and He+")

    # 自动调整X轴日期格式
    plt.xticks(rotation=45)

    # 显示图形
    plt.tight_layout()
    plt.show()


def plot_multiple_physical_quantities(x,b,v,vA,n):
    """

    :param x: [pd.datetime] ssies3_data.index
    :param b: [Series]
    :param v: [Series]
    :param vA: [Series]
    :param n: [Series]
    :return:
    """
    fig, axes = plt.subplots(4,3,sharex=True,figsize=(20,16))
    # 1
    for i in range(3):
        # 1
        axes[0,i].plot(x,b.iloc[:,i])
        # 2
        axes[1, i].plot(x, v.iloc[:, i])
        # 3
        axes[2, i].plot(x, vA.iloc[:, i])
        # 4
        axes[3, i].plot(x, n)
        axes[3, i].set_xlabel('UT time [s]')
    axes[0,0].set_ylabel('b [nT]')
    axes[1, 0].set_ylabel('v [m/s]')
    axes[2, 0].set_ylabel(r'$v_A [m/s]$')
    axes[3, 0].set_ylabel('n [/cc]')

    plt.show()


# 测试
# # 检查 process_ssm_data_time1之后得到的Epoch 列中是否有重复的值
# duplicate_epochs = ssm_data_grouped['Epoch'].duplicated()
#
# # 输出重复的时间戳行
# if duplicate_epochs.any():
#     print("存在重复的 Epoch 值:")
#     print(ssm_data_grouped[duplicate_epochs])
# else:
#     print("没有重复的 Epoch 值。")





def main():
    pass

if __name__ == "__main__":
    main()
