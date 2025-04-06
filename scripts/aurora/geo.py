import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from matplotlib.lines import Line2D  # Needed for custom legend entry

# --- Configuration ---
pole = "S"  # 'N' for North Pole, 'S' for South Pole
lat_limit_display = 55  # Geographic latitude boundary
polar_cap_latitude = (
    66.5  # Define the polar cap boundary latitude (e.g., Arctic Circle)
)

# --- Plotting ---
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

# --- Draw Polar Cap Boundary Circle ---
# Generate points along the constant latitude circle
polar_cap_lons = np.linspace(-180, 180, 150)  # Longitudes from -180 to 180
polar_cap_lats = np.full_like(polar_cap_lons, polar_cap_latitude)  # Constant latitude

ax.plot(
    polar_cap_lons,
    polar_cap_lats,
    color="red",  # Distinct color (e.g., red)
    linewidth=1.5,  # Slightly thicker?
    linestyle="-",  # Solid line
    transform=ccrs.Geodetic(),  # Input coords are Geographic
    label=f"{abs(polar_cap_latitude):.1f}° Latitude Boundary",  # Label for legend
)

# --- Load or Define Your LIST of Satellite Track Data ---
# Each element of the list should be a NumPy array of shape (n_i, 2)
# where column 0 is latitude, column 1 is longitude.

list_of_tracks = []


# get all fps
def get_pkl_files(root_dir):
    pkl_files = []

    # 遍历根目录下的所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pkl"):
                # 获取完整的文件路径
                full_path = os.path.join(dirpath, filename)
                pkl_files.append(full_path)

    return pkl_files


# 指定要搜索的目录
target_dir = r"G:\master\pyaw\scripts\results\aw_cases\archive\orbits"

# 获取所有pkl文件路径
pkl_paths = get_pkl_files(target_dir)

for pkl_path in pkl_paths:
    df = pd.read_pickle(pkl_path)
    df_track = df[["Latitude", "Longitude"]].copy()
    list_of_tracks.append(df_track.values)

print(f"Number of tracks to plot: {len(list_of_tracks)}")
if list_of_tracks:
    print(f"Shape of first track: {list_of_tracks[0].shape}")

# --- Plot Multiple Satellite Tracks ---
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
        color="darkorange",
        linewidth=1.0,
        linestyle="-",
        transform=ccrs.Geodetic(),
    )
    # Optional: Start/End markers
    ax.plot(
        track_longitudes[0],
        track_latitudes[0],
        marker="o",
        color="green",
        markersize=4,
        linestyle="",
        transform=ccrs.Geodetic(),
    )
    ax.plot(
        track_longitudes[-1],
        track_latitudes[-1],
        marker="x",
        color="red",
        markersize=5,
        linestyle="",
        transform=ccrs.Geodetic(),
    )


# --- Add Legend ---
# Create legend entries manually to combine similar items
custom_lines = [
    Line2D([0], [0], color="darkorange", lw=1),  # For satellite tracks
    Line2D([0], [0], color="red", lw=1.5),
]  # For polar cap boundary
ax.legend(
    custom_lines,
    ["Satellite Tracks", f"{abs(polar_cap_latitude):.1f}° Lat Boundary"],
    fontsize=9,
    loc="lower left",
)


# --- Final Touches ---
plt.show()
