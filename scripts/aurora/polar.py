# """
# 地理坐标系，卫星轨迹（模式：所有卫星轨迹；单个轨道的几个卫星轨迹），一般情况下的极光带圈（纬度66.5）
# 新模式：并排绘制北极和南极，或仅绘制其中一个。
# """
import os
import sys
from pathlib import Path
import warnings # Import warnings module

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from matplotlib.lines import Line2D  # Needed for custom legend entry
from matplotlib import MatplotlibDeprecationWarning
# Suppress specific Cartopy warning about geometry simplification
warnings.filterwarnings("ignore", message="Approximating coordinate system")
# Suppress specific Matplotlib warning about deprecated style
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning, message="The seaborn styles")


plt.style.use("seaborn-v0_8-paper")

# --- Configuration ---
SAVE = False
# 'both': Plot North and South side-by-side
# 'north': Plot only North Pole
# 'south': Plot only South Pole
PLOT_POLES = 'south'  # Choose 'both', 'north', or 'south'

# Data loading mode ('single' or 'all')
DATA_MODE = 'all'

# Geographic latitude boundary for display (absolute value)
lat_limit_display = 55
# Define the polar cap boundary latitude (absolute value, e.g., 66.5 for Auroral Oval approximation)
polar_cap_latitude_abs = 66.5

satellite_lw = 2

# --- Data Loading ---

def get_pkl_files_pathlib(root_dir):
    """Finds all .pkl files recursively in a directory."""
    root_path = Path(root_dir)
    pkl_files = list(root_path.rglob("*.pkl"))
    return pkl_files

list_of_tracks = []

# --- Define Paths (MODIFY THESE PATHS AS NEEDED) ---
# Make sure the drive letter and paths are correct for your system
base_data_dir = Path(r"G:\master\pyaw\scripts\results\aw_cases\archive\orbits")
single_orbit_dir = base_data_dir / "12728" # Example single orbit directory
all_orbits_dir = base_data_dir
save_dir = Path(r"G:\master\pyaw\scripts\results\aw_cases\archive\polar")
save_dir.mkdir(parents=True, exist_ok=True) # Ensure save directory exists

# --- Select Data Source Based on DATA_MODE ---
if DATA_MODE == 'single':
    if not single_orbit_dir.is_dir():
         raise FileNotFoundError(f"Single orbit directory not found: {single_orbit_dir}")
    target_dir = single_orbit_dir
    pkl_paths = list(target_dir.glob("*.pkl"))
    save_prefix = f"single_{single_orbit_dir.name}" # Use directory name for prefix
elif DATA_MODE == 'all':
    if not all_orbits_dir.is_dir():
        raise FileNotFoundError(f"All orbits directory not found: {all_orbits_dir}")
    target_dir = all_orbits_dir
    pkl_paths = get_pkl_files_pathlib(target_dir)
    save_prefix = "all_awcases"
else:
    raise ValueError(f"Invalid DATA_MODE: {DATA_MODE}. Choose 'single' or 'all'.")


# --- Load Track Data ---
print(f"Loading data from: {target_dir}")
for pkl_path in pkl_paths:
    try:
        df = pd.read_pickle(pkl_path)
        # Ensure columns exist and are numeric before proceeding
        if "Latitude" in df.columns and "Longitude" in df.columns:
             if pd.api.types.is_numeric_dtype(df["Latitude"]) and pd.api.types.is_numeric_dtype(df["Longitude"]):
                 # Select and copy, dropping NaN values just in case
                 df_track = df[["Latitude", "Longitude"]].dropna().copy()
                 if not df_track.empty:
                    list_of_tracks.append(df_track.values)
             else:
                 print(f"Warning: Non-numeric Lat/Lon data in {pkl_path}. Skipping.")
        else:
             print(f"Warning: Missing 'Latitude' or 'Longitude' column in {pkl_path}. Skipping.")
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}. Skipping.")


print(f"Number of valid tracks loaded: {len(list_of_tracks)}")
if not list_of_tracks:
    print("No valid track data loaded. Exiting.")
    sys.exit() # Exit if no data to plot

if list_of_tracks:
    print(f"Shape of first track: {list_of_tracks[0].shape}")


# --- Plotting Function ---
def plot_polar_view(ax, pole_type, lat_limit, polar_cap_lat, tracks_data):
    """
    Plots satellite tracks and features on a polar stereographic map.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        pole_type (str): 'N' for North Pole, 'S' for South Pole.
        lat_limit (float): The latitude limit for the map extent (absolute value).
        polar_cap_lat (float): The latitude for the polar cap boundary circle (signed).
        tracks_data (list): A list of NumPy arrays, each containing [lat, lon] points.
    """
    if pole_type == "N":
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        map_extent = [-180, 180, lat_limit, 90]
        parallels = np.arange(60.0, 91.0, 10.0)
        meridians = np.arange(-180.0, 181.0, 30.0)
        title = "Satellite Tracks - North Pole"
    elif pole_type == "S":
        proj = ccrs.SouthPolarStereo(central_longitude=0)
        map_extent = [-180, 180, -90, -lat_limit]
        parallels = np.arange(-80.0, -lat_limit + 1, 10.0) # Adjust parallels based on lat_limit
        meridians = np.arange(-180.0, 181.0, 30.0)
        title = "Satellite Tracks - South Pole"
    else:
        raise ValueError("pole_type must be 'N' or 'S'")

    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, color="black", zorder=3)
    ax.add_feature(cfeature.LAND.with_scale('50m'), color="lightgray", alpha=0.5, zorder=1)
    # ax.add_feature(cfeature.OCEAN.with_scale('50m'), color='aliceblue', alpha=0.5, zorder=1) # Optional: Add ocean color

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.8, # Thinner gridlines
        color="gray",
        alpha=0.6,
        linestyle=":", # Dotted lines
        zorder=2 # Ensure gridlines are below coastlines but above land/ocean
    )
    gl.xlocator = mticker.FixedLocator(meridians)
    gl.ylocator = mticker.FixedLocator(parallels)
    gl.xlabel_style = {"size": 8, "color": "dimgray"} # Smaller labels
    gl.ylabel_style = {"size": 8, "color": "dimgray"}
    gl.top_labels = False
    gl.right_labels = False
    # Customize label positions for polar plots (can be tricky)
    # This attempts to place longitude labels nicely around the circle
    gl.xformatter = mticker.FuncFormatter(lambda v, pos: f"{int(v)}°")
    gl.yformatter = mticker.FuncFormatter(lambda v, pos: f"{int(v)}°")


    ax.set_title(title, fontsize=12)

    # --- Draw Polar Cap Boundary Circle ---
    polar_cap_lons = np.linspace(-180, 180, 150)
    polar_cap_lats = np.full_like(polar_cap_lons, polar_cap_lat)

    ax.plot(
        polar_cap_lons,
        polar_cap_lats,
        color="red",
        linewidth=1.2, # Slightly thinner
        linestyle="--", # Dashed line
        transform=ccrs.Geodetic(),
        label=f"{abs(polar_cap_lat):.1f}° Latitude Boundary",
        zorder=4 # Ensure boundary is on top
    )

    # --- Plot Multiple Satellite Tracks ---
    for i, track_array in enumerate(tracks_data):
        if (
            track_array is None
            or track_array.ndim != 2
            or track_array.shape[1] != 2
            or track_array.shape[0] < 2
        ):
            # print(f"Skipping track {i} due to invalid data format or insufficient points.") # Less verbose
            continue
        track_latitudes = track_array[:, 0]
        track_longitudes = track_array[:, 1]

        # Filter tracks to be within the plot's latitude bounds *before* plotting
        # This helps prevent Cartopy issues with lines wrapping excessively
        if pole_type == 'N':
            valid_indices = track_latitudes >= lat_limit
        else: # pole_type == 'S'
            valid_indices = track_latitudes <= -lat_limit

        if not np.any(valid_indices):
             continue # Skip if no part of the track is within the view

        # Plot segments (helps with tracks crossing the pole or map edge)
        # Find where the track goes out of bounds and break the line
        valid_lat = track_latitudes[valid_indices]
        valid_lon = track_longitudes[valid_indices]

        # A simple approach is to just plot the valid points. More complex segmenting
        # could be done, but this is often sufficient for visualization.
        ax.plot(
            valid_lon,
            valid_lat,
            color="darkorange",
            linewidth=satellite_lw, # Thinner lines for many tracks
            linestyle="-",
            transform=ccrs.Geodetic(),
            zorder=3 # Ensure tracks are above land/ocean but below boundary/coastlines
        )

    # --- Add Legend ---
    custom_lines = [
        Line2D([0], [0], color="darkorange", lw=satellite_lw),
        Line2D([0], [0], color="red", lw=1.2, linestyle="--"),
    ]
    ax.legend(
        custom_lines,
        ["Satellite Tracks", f"{abs(polar_cap_lat):.1f}° Lat Boundary"],
        fontsize=8, # Smaller legend
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01) # Fine-tune position slightly away from corner
    )


# --- Main Plotting Logic ---

poles_to_plot = []
if PLOT_POLES == 'both':
    poles_to_plot = ['N', 'S']
    fig_size = (14, 7) # Wider figure for two plots
    ncols = 2
    nrows = 1
elif PLOT_POLES == 'north':
    poles_to_plot = ['N']
    fig_size = (7, 7.5) # Slightly taller single plot
    ncols = 1
    nrows = 1
elif PLOT_POLES == 'south':
    poles_to_plot = ['S']
    fig_size = (7, 7.5) # Slightly taller single plot
    ncols = 1
    nrows = 1
else:
    raise ValueError("Invalid PLOT_POLES value. Choose 'both', 'north', or 'south'.")

# Create figure, but don't specify projection globally here
fig = plt.figure(figsize=fig_size)

# Keep track of created axes if needed later (though maybe not necessary here)
axes_list = []

# Plot each requested pole
for i, pole in enumerate(poles_to_plot):
    # --- Determine the correct projection for this subplot ---
    if pole == "N":
        proj = ccrs.NorthPolarStereo(central_longitude=0)
    else: # 'S'
        proj = ccrs.SouthPolarStereo(central_longitude=0)

    # --- Add subplot WITH the correct projection ---
    ax = fig.add_subplot(nrows, ncols, i + 1, projection=proj)
    axes_list.append(ax) # Optional: store the axis

    # Calculate the signed polar cap latitude for this pole
    current_polar_cap_lat = polar_cap_latitude_abs if pole == 'N' else -polar_cap_latitude_abs

    # --- Call the plotting function with the newly created GeoAxes ---
    # The plot_polar_view function now receives a proper GeoAxes object
    plot_polar_view(ax, pole, lat_limit_display, current_polar_cap_lat, list_of_tracks)

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0) # Add some padding

# --- Saving ---
if SAVE:
    # Construct filename based on data mode and plot mode
    save_n = f"{save_prefix}_polar_{PLOT_POLES}.png"
    save_path = save_dir / save_n # Use Path object for joining
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # Use bbox_inches='tight'
    print(f"Plot saved to: {save_path}")

# --- Display Plot ---
plt.show()