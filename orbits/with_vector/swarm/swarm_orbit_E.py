import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import os
import matplotlib.path as mpath  # 添加此行！

import utils.coordinate
from pyaw import utils

def process_data(df_e):
    # 提取数据
    lats = df_e['Latitude'].values
    lons = df_e['Longitude'].values
    Ehx = df_e['Ehx'].values
    Ehy = df_e['Ehy'].values
    VsatN = df_e['VsatN'].values
    VsatE = df_e['VsatE'].values

    # 计算旋转矩阵
    rotmat_nec2sc, rotmat_sc2nec = utils.coordinate.get_rotmat_nec2sc_sc2nec(VsatN, VsatE)
    e_north, e_east = utils.coordinate.do_rotation(-Ehx, -Ehy, rotmat_sc2nec)

    # 北半球筛选（使用布尔索引）
    mask_northern = lats >= 0
    orbit_lats = lats[mask_northern]
    orbit_lons = lons[mask_northern]
    E_east = e_east[mask_northern]
    E_north = e_north[mask_northern]

    # 正则化矢量
    magnitudes = np.sqrt(E_east**2 + E_north**2)
    min_mag = 1e-8
    max_mag = np.max(np.clip(magnitudes, a_min=min_mag, a_max=None))
    scale_factor = 1.8
    E_east_norm = (E_east / max_mag) * scale_factor
    E_north_norm = (E_north / max_mag) * scale_factor

    return orbit_lons, orbit_lats, E_east_norm, E_north_norm

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
        step = 16
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
            label=f'Normalized Electric Vector', labelpos='E',  # 标签在箭头右侧
            coordinates='axes'
        )

        # 标题
        ax.set_title(
            'North Hemisphere Map with Satellite Orbit and Electric Fields\n'
            '(NorthPolarStereo Projection)',
            fontsize=12, pad=18
        )

        plt.show()

if __name__ == "__main__":
    # 读取数据（增加异常处理）
    try:
        # modify
        file_path = r"V:\aw\swarm\vires\SW_EXPT_EFIA_TCT16\SW_EXPT_EFIA_TCT16_12019_20160114T234831_20160115T012206.pkl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df_e = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit()

    # 处理数据并绘图
    orbit_lons, orbit_lats, E_east, E_north = process_data(df_e)
    plot_map(orbit_lons, orbit_lats, E_east, E_north)