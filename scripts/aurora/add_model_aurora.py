import os.path
import pickle
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.feature.nightshade import Nightshade
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# data config
ORBIT_NUM = 12864  # modify
AURORA_FN = "12864_aurora_data_snapshot_20160309T221900.pkl"  # modify
AURORA_DIR = r"G:\master\pyaw\scripts\results\aw_cases\archive\aurora_data"
SATELLITE_ROOT_DIR = (
    rf"G:\master\pyaw\scripts\results\aw_cases\archive\orbits\{ORBIT_NUM}"
)
assert str(ORBIT_NUM) in AURORA_FN

# other config
SAVE = False
save_dir = Path(r"G:\master\pyaw\scripts\results\aw_cases\archive\aurora_fig")
plt.style.use("seaborn-v0_8-paper")
SATELLITE_TRACK_WIDTH = 5.0
SATELLITE_TRACK_ALPHA = 1.0
SATELLITE_TRACK_COLOR = "red"

# aurora data
aurora_path = os.path.join(AURORA_DIR, AURORA_FN)
with open(aurora_path, "rb") as f:
    aurora_data = pickle.load(f)
ts = aurora_data["ts"]
eb = aurora_data["eb"]
ovation_img = aurora_data["ovation_img"]
boundary_lon = eb["long"]
assert eb["smooth"].shape[0] == 1
boundary_lat_smooth = eb["smooth"][0, :]

# satellite data
list_of_tracks = []
pkl_paths = list(Path(SATELLITE_ROOT_DIR).glob("*.pkl"))
for pkl_path in pkl_paths:
    df = pd.read_pickle(pkl_path)
    df_track = df[["Latitude", "Longitude"]].copy()
    list_of_tracks.append(df_track.values)
print(f"Number of tracks to plot: {len(list_of_tracks)}")
if list_of_tracks:
    print(f"Shape of first track: {list_of_tracks[0].shape}")
# plot
# config
pole = "N"  # 'N' for North Pole, 'S' for South Pole
lat_limit_display = 55  # Geographic latitude boundary
polar_cap_latitude = (
    66.5  # Define the polar cap boundary latitude (e.g., Arctic Circle)
)
fig = plt.figure(figsize=(7, 7))
# --- Choose Projection ---
if pole == "N":
    proj = ccrs.NorthPolarStereo(central_longitude=0)
    map_extent = [-180, 180, lat_limit_display, 90]
    parallels = np.arange(60.0, 91.0, 10.0)
    meridians = np.arange(-180.0, 181.0, 30.0)
    title = "Satellite Tracks - North Pole"
else:  # South Pole
    proj = ccrs.SouthPolarStereo(central_longitude=0)
    map_extent = [-180, 180, -90, -lat_limit_display]
    parallels = np.arange(-80.0, -59.0, 10.0)  # Adjust South Pole parallels if needed
    meridians = np.arange(-180.0, 181.0, 30.0)
    polar_cap_latitude = -66.5  # Antarctic Circle for South Pole view
    title = "Satellite Tracks - South Pole"
# basemap, feature, gridline
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color="black")
ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.5)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    color="gray",
    alpha=0.7,
    linestyle="--",
)
gl.xlocator = mticker.FixedLocator(meridians)
gl.ylocator = mticker.FixedLocator(parallels)
gl.xlabel_style = {"size": 10, "color": "gray"}
gl.ylabel_style = {"size": 10, "color": "gray"}
gl.top_labels = False
gl.right_labels = False
ax.set_title(title)
# night border
try:
    border1 = ax.add_feature(Nightshade(ts[0]), alpha=0.5, zorder=3)
except Exception as e_nightshade:
    print(f"Could not add Nightshade: {e_nightshade}")
# aurora
min_level = 0.1  # aurora level
max_level = 5
global_mapextent = (-180, 180, -90, 90)

# --- Create Custom Colormap with Transparent Low End ---

# 1. Choose a base colormap
base_cmap_name = "viridis"
# base_cmap_name = 'inferno' # Inferno/Magma often work well for intensity plots
# base_cmap_name = 'gist_heat'

base_cmap = plt.cm.get_cmap(base_cmap_name)

# 2. Get the colors (e.g., 256 samples)
num_colors = 256
colors = base_cmap(np.linspace(0, 1, num_colors))  # Shape (256, 4) RGBA

# 3. Modify the Alpha channel for the low end
# Option A: Make the very first color (lowest value) fully transparent
# colors[0, 3] = 0.0 # Set alpha of the first color entry to 0

# Option B: Create a linear fade-in for the first few percent of colors
fade_percentage = 30  # Let the bottom 15% of the colormap fade in
num_fade = int(num_colors * fade_percentage / 100)
fade_alphas = np.linspace(0.0, 1.0, num_fade)  # Linear alpha from 0 to 1
colors[:num_fade, 3] = fade_alphas  # Apply these alphas to the first num_fade colors

# Option C: Quadratic fade-in (starts transparent longer)
# fade_percentage = 20
# num_fade = int(num_colors * fade_percentage / 100)
# fade_alphas = np.linspace(0.0, 1.0, num_fade)**2 # Quadratic alpha from 0 to 1
# colors[:num_fade, 3] = fade_alphas

# 4. Create the new colormap object
transparent_cmap = ListedColormap(colors)
try:
    img1 = ax.imshow(
        (
            ovation_img[:, :, 0] if ovation_img.ndim == 3 else ovation_img
        ),  # Adjust indexing if needed
        vmin=min_level,
        vmax=max_level,
        transform=ccrs.PlateCarree(),  # Assuming image is on PlateCarree grid
        extent=global_mapextent,  # Geographic bounds of the image
        origin="lower",
        zorder=4,  # Above nightshade
        alpha=0.5,  # Apply overall alpha if desired
        cmap=transparent_cmap,
        # cmap='inferno'
    )  # <-- USE THE NEW CUSTOM CMAP

    # Add Colorbar (it will reflect the new colormap)
    cbar = plt.colorbar(
        img1, ax=ax, orientation="horizontal", fraction=0.03, pad=0.06, shrink=0.8
    )
    cbar.set_label("Aurora Intensity / Flux")  # Adjust label
    cbar.ax.tick_params(labelsize=8)

except Exception as e_imshow:
    print(f"Could not plot aurora image with imshow: {e_imshow}")
# boundary
bordercolor_e1 = "orange"
bordercolor_v1 = "lightgreen"
border_alpha = 1.0
bound_e1 = ax.plot(
    boundary_lon,
    boundary_lat_smooth,
    transform=ccrs.Geodetic(),  # Data is geographic
    color=bordercolor_e1,
    alpha=border_alpha,
    linewidth=1.2,  # Example linewidth
    zorder=5,  # Above aurora image
    label="Equatorial Boundary",
)  # Label for legend

# Plot the offset viewing line (-8 degrees from the main boundary)
bound_v1 = ax.plot(
    boundary_lon,
    boundary_lat_smooth - 8,
    transform=ccrs.Geodetic(),
    color=bordercolor_v1,
    linestyle="--",
    alpha=border_alpha,
    linewidth=1.0,
    zorder=5,
    label="Viewing Line after Case et al. 2016",
)
# satellite
# Loop through each track in the list
for i, track_array in enumerate(list_of_tracks):
    if (
        track_array is None
        or track_array.ndim != 2
        or track_array.shape[1] != 2
        or track_array.shape[0] < 2
    ):
        print(f"Skipping track {i} due to invalid data format or insufficient points.")
        continue
    track_latitudes = track_array[:, 0]
    track_longitudes = track_array[:, 1]
    ax.plot(
        track_longitudes,
        track_latitudes,
        color=SATELLITE_TRACK_COLOR,
        linewidth=SATELLITE_TRACK_WIDTH,
        linestyle="-",
        transform=ccrs.Geodetic(),
        alpha=SATELLITE_TRACK_ALPHA,
        zorder=6,
    )

# --- Legend ---
legend_handles = []
legend_labels = []
if "bound_e1" in locals():  # Check if boundary plot exists
    legend_handles.append(
        Line2D([0], [0], color=bordercolor_e1, lw=1.2, alpha=border_alpha)
    )
    legend_labels.append("Equatorial Boundary")
if "bound_v1" in locals():  # Check if offset plot exists
    legend_handles.append(
        Line2D([0], [0], color=bordercolor_v1, lw=1.0, ls="--", alpha=border_alpha)
    )
    legend_labels.append("Viewing Line after Case et al. 2016")
# Add satellite track legend entry if tracks were plotted
legend_handles.append(
    Line2D([0], [0], color=SATELLITE_TRACK_COLOR, lw=SATELLITE_TRACK_WIDTH)
)
legend_labels.append("Satellite Tracks")

if legend_handles:  # Only show legend if there's something to label
    ax.legend(
        legend_handles, legend_labels, fontsize=8, loc="lower left", framealpha=0.8
    )

# --- Final Touches ---
title = f"OVATION Aurora ({title})\n{ts[0].strftime('%Y-%m-%d %H:%M:%S')} UTC"
ax.set_title(title)

# --- Saving ---
if SAVE:
    # Construct filename based on data mode and plot mode
    save_n = f"{ORBIT_NUM}.png"
    save_path = save_dir / save_n  # Use Path object for joining
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Use bbox_inches='tight'
    print(f"Plot saved to: {save_path}")

plt.show()

# Optional: Save the figure
# output_directory = "./output_plots" # Define your output directory
# os.makedirs(output_directory, exist_ok=True)
# output_filename = os.path.join(output_directory, f"aurora_map_{ts[0].strftime('%Y%m%d_%H%M%S')}.png")
# try:
#     fig.savefig(output_filename, dpi=150, bbox_inches='tight')
#     print(f"Figure saved to {output_filename}")
# except Exception as e_save:
#     print(f"Error saving figure: {e_save}")
# plt.close(fig) # Close figure after saving if running multiple times
