import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path # Using pathlib now

# --- Configuration ---
pole = 'N' # <--- Set to 'S' for South Pole test

# --- !!! Conditional Settings based on Pole !!! ---
if pole == 'N':
    mlat_grid_circles = np.arange(60, 81, 10)  # [60, 70, 80] for North
    plot_mlat_outer_boundary = 58             # Outer boundary MLat for North
    mlat_label_mlt = 3.5                      # MLT position for North Pole MLat labels (upper right)
else: # South Pole settings
    mlat_grid_circles = np.arange(-80, -59, 10) # [-80, -70, -60] for South
    plot_mlat_outer_boundary = -58            # Outer boundary MLat for South (negative)
    mlat_label_mlt = 3.5                      # MLT position for South Pole MLat labels (can adjust if needed, e.g., lower right: 20.5)
# -------------------------------------------------

mlt_grid_lines = np.arange(0, 24, 3) # MLT lines to draw (every 3 hours)
mlt_labels_list = [0, 6, 12, 18] # MLT labels to show
target_dir = r"G:\master\pyaw\scripts\results\aw_cases\archive\orbis_geomag" # Data directory
track_color = 'blue' # Color for the tracks

# --- Control Option ---
enable_hover_info = False
# --------------------

if enable_hover_info:
    try:
        import mplcursors
    except ImportError:
        print("Warning: mplcursors library not found. Hover info disabled.")
        print("Install it using: pip install mplcursors")
        enable_hover_info = False

# --- Helper Functions (Unchanged - logic handles pole='S') ---
def mlat_to_radius(mlat, pole='N'):
    if pole == 'N': return 90.0 - np.asarray(mlat)
    else: return 90.0 + np.asarray(mlat) # Correct for South Pole
def mlt_to_theta(mlt):
    return (np.asarray(mlt) / 24.0 - 0.25) * 2 * np.pi
def radius_to_mlat(r, pole='N'):
    if pole == 'N': return 90.0 - r
    else: return r - 90.0 # Correct for South Pole
def theta_to_mlt(theta):
    theta_normalized = theta % (2 * np.pi)
    mlt = (theta_normalized / (2 * np.pi) + 0.25) * 24.0
    return mlt % 24.0

# --- Data Loading (Using pathlib - Unchanged) ---
list_of_tracks = []
def get_pkl_files_pathlib(root_dir):
    root_path = Path(root_dir)
    pkl_files = list(root_path.rglob("*.pkl"))
    return pkl_files

pkl_paths = get_pkl_files_pathlib(target_dir)
print(f"Found {len(pkl_paths)} .pkl files.")

loaded_track_count = 0
for pkl_path in pkl_paths:
    try:
        df = pd.read_pickle(pkl_path)
        if "QDLat" in df.columns and "MLT" in df.columns:
            # Filter based on pole BEFORE processing - Assumes data contains both N/S
            if pole == 'N':
                df_pole_filtered = df[df["QDLat"] >= 0]
            else: # pole == 'S'
                df_pole_filtered = df[df["QDLat"] <= 0]

            if not df_pole_filtered.empty:
                 df_track_geomag = df_pole_filtered[["QDLat", "MLT"]].dropna().astype(float)
                 if not df_track_geomag.empty:
                     list_of_tracks.append(df_track_geomag.values)
                     loaded_track_count += 1
                 # else: print(f"Warning: Track from {pkl_path.name} ({pole}) empty after dropna.")
            # else: print(f"Warning: No {pole}-pole data found in {pkl_path.name}.")
        # else: print(f"Warning: Columns 'QDLat' or 'MLT' not found in {pkl_path.name}. Skipping.")
    except Exception as e:
        print(f"Error loading or processing {pkl_path}: {e}")

print(f"Successfully loaded {loaded_track_count} tracks for {pole}-pole.")
if list_of_tracks:
    print(f"Shape of first loaded track: {list_of_tracks[0].shape}")

# --- Plotting ---
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='polar')

# --- Set Plot Limits and Appearance (Uses conditional boundary) ---
max_radius = mlat_to_radius(plot_mlat_outer_boundary, pole)
print(f"Pole: {pole}, Outer MLat Boundary: {plot_mlat_outer_boundary}, Max Radius: {max_radius}")
ax.set_ylim(0, max_radius)
ax.set_facecolor('white')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['polar'].set_visible(False)

# --- Draw MLat Circles (Uses conditional mlat_grid_circles) ---
thetas_for_circle = np.linspace(0, 2 * np.pi, 150)
mlat_label_positions = {}
# mlat_label_mlt is now set conditionally above

for mlat_val in mlat_grid_circles: # Now iterates through correct negative values for 'S' pole
    r = mlat_to_radius(mlat_val, pole)
    # Check if radius is within plot limits before drawing
    if r <= max_radius:
        ax.plot(thetas_for_circle, np.full_like(thetas_for_circle, r), 'k--', linewidth=1)
        # Store label position using the correct radius
        label_theta = mlt_to_theta(mlat_label_mlt)
        mlat_label_positions[mlat_val] = (label_theta, r)
    else:
         print(f"Skipping MLat circle {mlat_val} (radius {r} > max_radius {max_radius})")


# --- Draw MLT Meridians (Unchanged logic) ---
mlt_label_positions = {}
r_line_min = 0
r_line_max = max_radius
for mlt_val in mlt_grid_lines:
    theta = mlt_to_theta(mlt_val)
    ax.plot([theta, theta], [r_line_min, r_line_max], 'k--', linewidth=1)
    if mlt_val in mlt_labels_list:
         mlt_label_positions[mlt_val] = (theta, max_radius * 1.05)

# --- Add Labels (Uses correct mlat_val for 'S' pole) ---
# MLat Labels
for mlat_val, (theta, r) in mlat_label_positions.items():
     # Add a small radial offset outward for clarity
     # Offset calculation remains the same, uses the correct 'r'
     ax.text(theta, r + 0.03 * max_radius, f'{int(mlat_val)}$^\circ$', color='black', # Use relative offset
             ha='center', va='center', fontsize=10)

# MLT Labels (Unchanged logic)
for mlt_val, (theta, r) in mlt_label_positions.items():
    label_text = f'{int(mlt_val)}'
    if mlt_val == 0: label_text = f'{int(mlt_val)} MLT'
    if mlt_val == 12: ha, va = 'center', 'bottom'
    elif mlt_val == 6: ha, va = 'left', 'center'
    elif mlt_val == 0: ha, va = 'center', 'top'
    elif mlt_val == 18: ha, va = 'right', 'center'
    else: ha, va = 'center', 'center'
    ax.text(theta, r, label_text, color='black', ha=ha, va=va, fontsize=11, weight='medium')

# --- Plot Loaded Tracks (Added check for radius) ---
plotted_lines = []

if not list_of_tracks:
    print("No tracks loaded to plot.")
else:
    for track_data in list_of_tracks:
        mlats = track_data[:, 0]
        mlts = track_data[:, 1]
        thetas = mlt_to_theta(mlts)
        radii = mlat_to_radius(mlats, pole)

        # --- Filter points outside the plot boundary ---
        valid_indices = np.where(radii <= max_radius)[0]
        if len(valid_indices) < 2 : continue # Need at least 2 points to draw a line segment

        # Plot segments only where consecutive points are valid
        # Find breaks in consecutive valid indices
        breaks = np.where(np.diff(valid_indices) != 1)[0] + 1
        segment_indices = np.split(valid_indices, breaks)

        for segment in segment_indices:
            if len(segment) >= 2: # Only plot segments with 2 or more points
                 line, = ax.plot(thetas[segment], radii[segment], color=track_color, linewidth=1.0)
                 plotted_lines.append(line) # Add segment line for potential hover
            # else: print(f"Skipping segment with < 2 points: {segment}")


# --- Conditionally Add Hover Capability (Uses correct pole in callbacks) ---
if enable_hover_info:
    if plotted_lines:
        print("Hover info enabled.")

        def on_add(sel):
            theta_hover, radius_hover = sel.target
            # Pass the current 'pole' setting to the conversion functions
            mlat_hover = radius_to_mlat(radius_hover, pole)
            mlt_hover = theta_to_mlt(theta_hover)
            annotation_text = f"MLT: {mlt_hover:.2f}\nMLat: {mlat_hover:.2f}"
            sel.annotation.set(text=annotation_text)
            sel.annotation.get_bbox_patch().set(facecolor='white', alpha=0.7)
            sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)

        cursor = mplcursors.cursor(plotted_lines, hover=True)
        cursor.connect("add", on_add)

    else:
        print("Hover info was enabled, but no valid track lines were plotted.")
else:
    print("Hover info disabled.")

# --- Final Touches ---
plt.show()