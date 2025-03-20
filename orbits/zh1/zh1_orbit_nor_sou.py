from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import numpy as np

from zh1 import zh1
from utils import get_orbit_num_indicator_st_et,get_split_indices


def plot_orbit(lons, lats, proj_method="NorthPolarStereo", central_longitude=0,
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

    ax.plot(*proj.transform_points(geodetic, lons, lats)[:, :2].T,
            'b-', lw=1.5, label=f'Satellite Orbit')
    ax.legend(loc='upper right')
    ax.set_title(current_title, fontsize=12, pad=18)

    return ax


### modify
## ---
# 17538, 1 ascending
# file_name = 'CSES_01_EFD_1_L2A_A1_175381_20210401_012158_20210401_015642_000.h5'
# file_path = r"V:\aw\zh1\efd\ulf\2a\20210401_20210630\CSES_01_EFD_1_L2A_A1_175381_20210401_012158_20210401_015642_000.h5"
## ---
# 17538, 0 descending
file_name = 'CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5'
file_path = r"V:\aw\zh1\efd\ulf\2a\20210401_20210630\CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5"
## ---
# file with abnormal longitudes
# 在经度0处有突变（合理的突变处应为经度正负180处）
# file_dir = Path(r"V:\aw\zh1\efd\ulf\2a\abnormal_lons")
# file_name = 'CSES_01_EFD_1_L2A_A1_175391_20210401_025642_20210401_033127_000.h5'
# file_path = file_dir / Path(file_name)
### modify

orbit_number,indicator,st,et = get_orbit_num_indicator_st_et(file_name)
efd = zh1.EFDULF(file_path)
dfs = efd.dfs
lats = dfs['GEO_LAT'].squeeze().values
indices = get_split_indices(lats, indicator)
northern_slice = slice(*indices[0])
southern_slice = slice(*indices[1])
orbit_lats_north = lats[northern_slice]
orbit_lats_south = lats[southern_slice]
lons = dfs['GEO_LON'].squeeze().values
orbit_lons_north = lons[northern_slice]
orbit_lons_south = lons[southern_slice]
# 创建图形和子图
fig = plt.figure(figsize=(24, 12))
# 北半球子图
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
plot_orbit(orbit_lons_north, orbit_lats_north, proj_method="NorthPolarStereo", ax=ax1)
# 南半球子图
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=0))
plot_orbit(orbit_lons_south, orbit_lats_south, proj_method="SouthPolarStereo", ax=ax2)
if indicator == "1":
    plt.suptitle(f'ZH-1, orbit{orbit_number} ascending, from {st} to {et}')
else:
    plt.suptitle(f'ZH-1, orbit{orbit_number} descending, from {st} to {et}')
# plt.tight_layout()  # 调整布局
plt.show()




if __name__ == '__main__':
    pass