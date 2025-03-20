# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/19/2025 20:37
@Project     : pyaw
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

from pyaw.dmsp import SPDF
from utils import get_split_indices


def plot_orbits(file_paths, proj_method="NorthPolarStereo", central_longitude=0,
               figsize=(12, 12), lon_lat_ext=None, title=None, if_cfeature=False, ax=None):
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    # 设置默认的extent和标题
    if proj_method == "NorthPolarStereo":
        default_ext = (-180, 180, 0, 90)
        default_title = 'North Hemisphere Map using NorthPolarStereo Projection'
    else:
        default_ext = (-180, 180, -90, 0)
        default_title = 'South Hemisphere Map using SouthPolarStereo Projection'

    current_ext = lon_lat_ext if lon_lat_ext is not None else default_ext
    current_title = title if title is not None else default_title

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': proj})
    else:
        fig = ax.figure

    ax.set_extent(list(current_ext), crs=geodetic)

    # 创建圆形边界
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.45 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)

    if if_cfeature:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND.with_scale('50m'), alpha=0.6)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.4)
        ax.coastlines(resolution='50m', linewidth=0.5)

    # 设置ylocs
    if proj_method == "NorthPolarStereo":
        ylocs = np.arange(0, 91, 15)
    else:
        ylocs = np.arange(-90, 0, 15)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      xlocs=np.arange(-180, 181, 45), ylocs=ylocs,
                      xpadding=15, ypadding=15)
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False

    for file_path in file_paths:
        spdf = SPDF()
        df = spdf.r_s3(str(file_path))
        lats = df['glat'].values
        indices = get_split_indices(lats)
        if proj_method == "NorthPolarStereo":
            northern_slice = slice(*indices[0])
            orbit_lons_north = df['glon'].values[northern_slice]
            orbit_lats_north = lats[northern_slice]
            # 绘制轨道
            ax.plot(*proj.transform_points(geodetic, orbit_lons_north, orbit_lats_north)[:, :2].T,
                    'b-', lw=1.5)
        else:
            southern_slice = slice(*indices[1])
            orbit_lons_south = df['glon'].values[southern_slice]
            orbit_lats_south = lats[southern_slice]
            # 绘制轨道
            ax.plot(*proj.transform_points(geodetic, orbit_lons_south, orbit_lats_south)[:, :2].T,
                    'b-', lw=1.5)
    ax.set_title(current_title, fontsize=12, pad=18)

    return ax



def main():
    pass


if __name__ == "__main__":
    dir = r"V:\aw\DMSP\spdf\f16\ssies3\2014"
    day = '20140101'
    fns = [f for f in os.listdir(dir) if day in f and f.endswith('.cdf')]
    file_paths = [Path(dir) / fn for fn in fns]

    # 创建图形和子图
    fig = plt.figure(figsize=(24, 12))

    # 北半球子图
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
    plot_orbits(file_paths, proj_method="NorthPolarStereo", ax=ax1)

    # 南半球子图
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=0))
    plot_orbits(file_paths, proj_method="SouthPolarStereo", ax=ax2)

    plt.suptitle(f'DMSP F16, {day}')
    # plt.tight_layout()  # 调整布局 (可能会使图中某些部分消失)
    plt.show()