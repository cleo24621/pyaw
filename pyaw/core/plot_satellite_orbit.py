# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/23/2025 1:28
@Project     : pyaw
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import numpy as np
from nptyping import NDArray
import pandas as pd

def plot_northern_hemisphere_satellite_orbit(
        lons: NDArray,
        lats: NDArray,
        proj_method: str = "NorthPolarStereo",
        central_longitude: float = 0,
        figsize: tuple[float] = (12, 12),
        lon_lat_ext: tuple[float] = (-180, 180, 0, 90),
        title: str = "North Hemisphere Map using NorthPolarStereo Projection",
        if_cfeature: bool = False,
) -> None:
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    ax.set_extent(list(lon_lat_ext), crs=geodetic)

    # Create circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(
        np.column_stack([np.sin(theta), np.cos(theta)]) * 0.45 + 0.5
    )
    ax.set_boundary(circle, transform=ax.transAxes)

    if if_cfeature:
        import cartopy.feature as cfeature

        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4)
        ax.coastlines(resolution="50m", linewidth=0.5)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        xlocs=np.arange(-180, 181, 45),
        ylocs=np.arange(0, 91, 15),
        xpadding=15,
        ypadding=15,
    )
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False

    ax.plot(
        *proj.transform_points(geodetic, lons, lats)[:, :2].T,
        "b-",
        lw=1.5,
        label="Satellite Orbit"
    )
    ax.legend(loc="upper right")
    ax.set_title(title, fontsize=12, pad=18)
    plt.show()




def main():
    pass


if __name__ == "__main__":
    main()
