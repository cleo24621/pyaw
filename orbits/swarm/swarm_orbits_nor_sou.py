from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import numpy as np
import pandas as pd


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
        df = pd.read_pickle(file_path)
        lats = df['Latitude'].values
        indices = get_split_indices(lats)
        if proj_method == "NorthPolarStereo":
            northern_slice = slice(*indices[0])
            orbit_lons_north = df['Longitude'].values[northern_slice]
            orbit_lats_north = lats[northern_slice]
            # 绘制轨道
            ax.plot(*proj.transform_points(geodetic, orbit_lons_north, orbit_lats_north)[:, :2].T,
                    'b-', lw=1.5)
        else:
            southern_slice = slice(*indices[1])
            orbit_lons_south = df['Longitude'].values[southern_slice]
            orbit_lats_south = lats[southern_slice]
            # 绘制轨道
            ax.plot(*proj.transform_points(geodetic, orbit_lons_south, orbit_lats_south)[:, :2].T,
                    'b-', lw=1.5)
    ax.set_title(current_title, fontsize=12, pad=18)

    return ax


# 主程序部分
fns = [
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12729_20160301T030258_20160301T043631.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12730_20160301T043631_20160301T061005.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12731_20160301T061005_20160301T074338.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12732_20160301T074338_20160301T091712.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12733_20160301T091712_20160301T105045.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12734_20160301T105045_20160301T122419.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12735_20160301T122419_20160301T135752.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12736_20160301T135752_20160301T153125.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12737_20160301T153125_20160301T170459.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12738_20160301T170459_20160301T183832.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12739_20160301T183832_20160301T201206.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12740_20160301T201206_20160301T214539.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12741_20160301T214539_20160301T231913.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12742_20160301T231913_20160302T005246.pkl",
]
file_paths = [Path(r"V:\aw\swarm\vires\gdcoors\SW_OPER_MAGA_LR_1B") / Path(i) for i in fns]

# 创建图形和子图
fig = plt.figure(figsize=(24, 12))

# 北半球子图
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
plot_orbits(file_paths, proj_method="NorthPolarStereo", ax=ax1)

# 南半球子图
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=0))
plot_orbits(file_paths, proj_method="SouthPolarStereo", ax=ax2)

plt.suptitle('SwarmA, 20160301')
# plt.tight_layout()  # 调整布局 (可能会使图中某些部分消失)
plt.show()