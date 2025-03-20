from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import numpy as np
import pandas as pd


from utils import get_split_indices


def plot_orbit(lons, lats, proj_method="NorthPolarStereo", central_longitude=0,
               figsize=(12, 12), lon_lat_ext=(-180, 180, 0, 90),
               title='North Hemisphere Map using NorthPolarStereo Projection',
               if_cfeature=False):
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': proj})
    ax.set_extent(list(lon_lat_ext), crs=geodetic)

    # Create circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.45 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)

    if if_cfeature:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND.with_scale('50m'), alpha=0.6)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.4)
        ax.coastlines(resolution='50m', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      xlocs=np.arange(-180, 181, 45), ylocs=np.arange(0, 91, 15),
                      xpadding=15, ypadding=15)
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False

    ax.plot(*proj.transform_points(geodetic, lons, lats)[:, :2].T,
            'b-', lw=1.5, label='Satellite Orbit')
    ax.legend(loc='upper right')
    ax.set_title(title, fontsize=12, pad=18)
    plt.show()


# 主程序部分
df_path = Path(
    r"V:\aw\swarm\vires\AHY9U3~9\SW_OPER_MAGA_LR_1B") / "aux_only_gdcoors_SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl"
df = pd.read_pickle(df_path)

lats = df['Latitude'].values
indices = get_split_indices(lats)
northern_slice = slice(*indices[0])
orbit_lats = lats[northern_slice]
orbit_lons = df['Longitude'].values[northern_slice]

plot_orbit(orbit_lons, orbit_lats)