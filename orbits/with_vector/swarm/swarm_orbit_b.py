import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import os
import matplotlib.path as mpath  # 添加此行！

import utils.data
from pyaw import utils

def process_data(df_b,df_b_igrf):
    # 提取数据
    lats = df_b['Latitude'].values
    lons = df_b['Longitude'].values

    Bn, Be, _ = utils.data.get_3arrs(df_b['B_NEC'].values)
    bn_igrf, be_igrf, _ = utils.data.get_3arrs(df_b_igrf['B_NEC_IGRF'].values)
    bn = Bn - bn_igrf
    be = Be - be_igrf

    # 北半球筛选（使用布尔索引）
    mask_northern = lats >= 0
    orbit_lats = lats[mask_northern]
    orbit_lons = lons[mask_northern]
    be_nor = be[mask_northern]
    bn_nor = bn[mask_northern]

    # 正则化矢量
    magnitudes = np.sqrt(be_nor**2 + bn_nor**2)
    min_mag = 1e-8
    max_mag = np.max(np.clip(magnitudes, a_min=min_mag, a_max=None))
    scale_factor = 1.8
    be_nor_norm = (be_nor / max_mag) * scale_factor
    bn_nor_norm = (bn_nor / max_mag) * scale_factor

    return orbit_lons, orbit_lats, be_nor_norm, bn_nor_norm

def plot_map(orbit_lons, orbit_lats, E_east, E_north):
    # 创建投影
    proj = ccrs.NorthPolarStereo(central_longitude=0)
    geodetic = ccrs.PlateCarree()

    # 创建画布
    with plt.ioff():
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([-180, 180, 0, 90], crs=geodetic)

        # 添加圆形边界
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = mpath.Path(
            np.vstack([np.sin(theta), np.cos(theta)]).T * 0.45 + 0.5
        )
        ax.set_boundary(circle, transform=ax.transAxes)

        # 绘制轨道和矢量（减少点数）
        step = 50
        ax.plot(
            orbit_lons[::step], orbit_lats[::step],
            color='navy', linewidth=1.5, transform=geodetic, zorder=2
        )
        # # 绘制散点（可选,'s'控制散点大小）
        # ax.scatter(
        #     orbit_lons[::step], orbit_lats[::step],
        #     color='navy', s=15, zorder=3, transform=geodetic
        # )

        # 转换矢量并绘制
        u_proj, v_proj = proj.transform_vectors(
            geodetic,
            orbit_lons[::step], orbit_lats[::step],
            E_east[::step], E_north[::step]
        )
        q = ax.quiver(
            orbit_lons[::step], orbit_lats[::step],
            u_proj, v_proj,
            scale=35,  # 增大scale值以适应更大范围
            width=0.0025,  # 更细的箭头杆
            headwidth=0,
            headlength=0,
            headaxislength=0,  # 箭头轴长度（可选）(default=4.5，设置不同值的影响查看示例)（此绘制最好设置为0）
            color='crimson',
            transform=geodetic,
            zorder=4
        )

        # 添加网格线
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            xlocs=np.arange(-180, 181, 45),
            ylocs=np.arange(0, 91, 15),
            xpadding=15,
            ypadding=15
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.rotate_labels = False

        ax.quiverkey(
            q, X=0.82, Y=0.12, U=0.3,
            label=f'Normalized Measured Minus Model Magnetic Field (nT)', labelpos='E',  # 标签在箭头右侧
            coordinates='axes'
        )

        # 标题
        ax.set_title(
            'North Hemisphere Map with Satellite Orbit and Measured Minus Model Magnetic Field (nT)\n'
            '(NorthPolarStereo Projection)',
            fontsize=12, pad=18
        )

        plt.show()

if __name__ == "__main__":
    # 读取数据（增加异常处理）
    try:
        # modify
        file_path = r"V:\aw\swarm\vires\SW_OPER_MAGA_HR_1B\SW_OPER_MAGA_HR_1B_12019_20160114T234831_20160115T012206.pkl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df_b = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit()
    try:
        # modify (same orbit number)
        file_path = r"V:\aw\swarm\vires\IGRF\SW_OPER_MAGA_HR_1B\IGRF_SW_OPER_MAGA_HR_1B_12019_20160114T234831_20160115T012206.pkl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df_b_igrf = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit()

    # 处理数据并绘图
    orbit_lons, orbit_lats, E_east, E_north = process_data(df_b,df_b_igrf)
    plot_map(orbit_lons, orbit_lats, E_east, E_north)