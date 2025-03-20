import os.path
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np

from utils import get_orbit_num_indicator_st_et, get_split_indices
from zh1 import zh1


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
    # 初始化一个文本对象用于显示轨道信息
    orbit_text = ax.text(0.98, 0.02, '', transform=ax.transAxes,
                         ha='right', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        orbit_number, indicator, st, et = get_orbit_num_indicator_st_et(file_name)
        efd = zh1.EFDULF(file_path)
        dfs = efd.dfs
        lats = dfs['GEO_LAT'].squeeze().values
        indices = get_split_indices(lats, indicator)
        lons = dfs['GEO_LON'].squeeze().values
        if proj_method == "NorthPolarStereo":
            northern_slice = slice(*indices[0])
            orbit_lats_north = lats[northern_slice]
            orbit_lons_north = lons[northern_slice]
            # 绘制轨道
            ax.plot(*proj.transform_points(geodetic, orbit_lons_north, orbit_lats_north)[:, :2].T,
                    'b-', lw=1.5)
        else:
            southern_slice = slice(*indices[1])
            orbit_lats_south = lats[southern_slice]
            orbit_lons_south = lons[southern_slice]
            # 绘制轨道
            ax.plot(*proj.transform_points(geodetic, orbit_lons_south, orbit_lats_south)[:, :2].T,
                    'b-', lw=1.5)
        # 更新并显示当前轨道信息
        orbit_info = f"orbit number: {orbit_number}\ndescend_ascend: {indicator} (1 ascend, 0 descend)\nstart time: {st}\nend time: {et}"
        orbit_text.set_text(orbit_info)
        plt.draw()
        plt.pause(1)  # 暂停1秒观察
    # 清除轨道信息文本
    orbit_text.remove()
    ax.set_title(current_title, fontsize=12, pad=18)

    return ax


file_dir = Path(r'V:\aw\zh1\efd\ulf\2a\20210401_20210630')


def get_1d_file_names(file_dir, condition="20210401"):
    # 轨道号和升降轨号一样的文件可能不唯一，例如‘175381’，此时选择较大的那个文件，因为包含更多的信息
    # 获取所有符合条件的文件（20210401）
    files = [f for f in os.listdir(file_dir) if condition in f and f.endswith(".h5")]

    # 创建字典存储每个轨道号的最大文件
    orbit_files = {}

    for filename in files:
        # 提取轨道号（第七个下划线分隔字段）
        parts = filename.split('_')
        if len(parts) >= 7:
            orbit = parts[6]
            file_path = os.path.join(file_dir, filename)
            size = os.path.getsize(file_path)

            # 更新字典中该轨道号的最大文件
            if orbit not in orbit_files or size > orbit_files[orbit]["size"]:
                orbit_files[orbit] = {
                    "name": filename,
                    "size": size
                }
    return [i['name'] for i in orbit_files.values()]


day = "20210401"
file_names = get_1d_file_names(file_dir, condition=day)
file_paths = [file_dir / file_name for file_name in file_names]

# 创建图形和子图
fig = plt.figure(figsize=(24, 12))

# 北半球子图
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
plot_orbits(file_paths, proj_method="NorthPolarStereo", ax=ax1)

# 南半球子图
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=0))
plot_orbits(file_paths, proj_method="SouthPolarStereo", ax=ax2)

plt.suptitle(f'ZH-1, {day}')
# plt.tight_layout()  # 调整布局 (可能会使图中某些部分消失)
plt.show()

if __name__ == '__main__':
    pass
