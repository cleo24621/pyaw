import os.path
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio  # 确保使用 v2 版本
import subprocess  # 用于 ffmpeg 方案

from utils import get_orbit_num_indicator_st_et
from utils import get_nor_sou_split_indices_zh1
from zh1 import zh1

def plot_orbits(file_paths, proj_method="NorthPolarStereo", central_longitude=0,
                figsize=(12, 12), lon_lat_ext=None, title=None, if_cfeature=False, ax=None, save_gif_path=None):
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

    # 设置标题（提前设置，确保所有帧都有标题）
    ax.set_title(current_title, fontsize=12, pad=18)

    # 初始化文本对象
    orbit_text = ax.text(0.98, 0.02, '', transform=ax.transAxes,
                         ha='right', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    frame_count = 0
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        orbit_number, indicator, st, et = get_orbit_num_indicator_st_et(file_name)
        efd = zh1.EFDULF(file_path)
        dfs = efd.dfs
        lats = dfs['GEO_LAT'].squeeze().values
        indices = get_nor_sou_split_indices_zh1(lats, indicator)
        lons = dfs['GEO_LON'].squeeze().values

        if proj_method == "NorthPolarStereo":
            northern_slice = slice(*indices[0])
            orbit_lats_north = lats[northern_slice]
            orbit_lons_north = lons[northern_slice]
            ax.plot(*proj.transform_points(geodetic, orbit_lons_north, orbit_lats_north)[:, :2].T,
                    'b-', lw=1.5)
        else:
            southern_slice = slice(*indices[1])
            orbit_lats_south = lats[southern_slice]
            orbit_lons_south = lons[southern_slice]
            ax.plot(*proj.transform_points(geodetic, orbit_lons_south, orbit_lats_south)[:, :2].T,
                    'b-', lw=1.5)

        # 更新轨道信息
        orbit_info = f"orbit number: {orbit_number}\ndescend_ascend: {indicator} (1 ascend, 0 descend)\nstart time: {st}\nend time: {et}"
        orbit_text.set_text(orbit_info)

        # 保存当前图像为帧（包含标题和当前轨道信息）
        if save_gif_path:
            plt.savefig(f"{save_gif_path}_{frame_count:03d}.png", bbox_inches='tight')
            frame_count += 1

    # 清除文本对象
    orbit_text.remove()

    return ax


def create_gif(frame_prefix, output_filename, duration_per_frame=0.5):
    # 方案1：使用 imageio
    images = []
    i = 0
    while True:
        filename = f"{frame_prefix}_{i:03d}.png"
        if not os.path.exists(filename):
            break
        images.append(imageio.imread(filename))
        i += 1
    if images:
        # 确保仅设置 duration，不使用 fps
        imageio.mimsave(output_filename, images, duration=duration_per_frame)


def create_gif_with_ffmpeg(frame_prefix, output_filename, duration_per_frame):
    # 需要安装 ffmpeg
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-framerate', str(1/duration_per_frame),
        '-i', f'{frame_prefix}_%03d.png',
        '-vf', f'fps={1/duration_per_frame}',
        '-loop', '0',
        output_filename
    ]
    subprocess.run(cmd, check=True)


def get_1d_file_names(file_dir, condition="20210401"):
    files = [f for f in os.listdir(file_dir) if condition in f and f.endswith(".h5")]
    orbit_files = {}
    for filename in files:
        parts = filename.split('_')
        if len(parts) >= 7:
            orbit = parts[6]
            file_path = os.path.join(file_dir, filename)
            size = os.path.getsize(file_path)
            if orbit not in orbit_files or size > orbit_files[orbit]["size"]:
                orbit_files[orbit] = {"name": filename, "size": size}
    return [i['name'] for i in orbit_files.values()]


if __name__ == '__main__':
    file_dir = Path(r'V:\aw\zh1\efd\ulf\2a\20210401_20210630')
    day = "20210401"
    file_names = get_1d_file_names(file_dir, condition=day)
    file_paths = [file_dir / file_name for file_name in file_names]

    # 生成北半球GIF
    fig_north = plt.figure(figsize=(12, 12))
    ax_north = fig_north.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
    plot_orbits(file_paths, proj_method="NorthPolarStereo", ax=ax_north, save_gif_path='north_frame')
    plt.close(fig_north)

    # 生成南半球GIF
    fig_south = plt.figure(figsize=(12, 12))
    ax_south = fig_south.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    plot_orbits(file_paths, proj_method="SouthPolarStereo", ax=ax_south, save_gif_path='south_frame')
    plt.close(fig_south)

    # # 生成 GIF（使用 imageio）
    # create_gif('north_frame', 'north_globe.gif', duration_per_frame=0.25)  # 每帧 0.25 秒
    # create_gif('south_frame', 'south_globe.gif', duration_per_frame=0.25)

    # 或者使用 ffmpeg（需要安装）
    create_gif_with_ffmpeg('north_frame', 'north_globe.gif', 0.25)

    # 清理临时文件
    for filename in os.listdir('../../../orbits/zh1'):
        if filename.startswith(('north_frame', 'south_frame')):
            os.remove(filename)