import os
from pathlib import Path  # Using pathlib now

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Configuration ---
# Set pole to 'N', 'S', or 'Both'
pole_mode = (
    "S"  # <--- Set to 'S' for South Pole test, 'N' for North, 'Both' for side-by-side
)

# style

plt.style.use("seaborn-v0_8-paper")

# --- NEW: Auroral Oval Configuration ---
PLOT_AURORAL_OVAL = True  # Set to True to draw an approximate auroral oval band
# --- Approximate Oval Parameters (MLat degrees) ---
# These are simplified representations and don't account for activity levels
OVAL_EQUATORWARD_MLAT_AVG_N = 62.5  # Average equatorward boundary (North)
OVAL_POLEWARD_MLAT_AVG_N = 72.5  # Average poleward boundary (North)
OVAL_DAY_NIGHT_MLAT_DIFF = 2.5  # How much more equatorward at night vs day
# OVAL_DAY_NIGHT_MLAT_DIFF = 0
OVAL_COLOR = "lightgreen"
OVAL_ALPHA = 0.3
# --------------------------------------

SAVE = False
save_dir = r"G:\master\pyaw\scripts\results\aw_cases"

target_dir = (
    r"G:\master\pyaw\scripts\results\aw_cases\archive\orbis_geomag"  # Data directory
)
SATELLITE_TRACK_COLOR = "red"  # Color for the tracks
SATELLITE_TRACK_WIDTH = 1.0

# --- Control Option ---
enable_hover_info = False
# --------------------

# --- Constants (Defaults for North Pole) ---
DEFAULT_MLAT_GRID_CIRCLES = np.arange(60, 81, 10)  # [60, 70, 80]
DEFAULT_PLOT_MLAT_OUTER_BOUNDARY = 58
DEFAULT_MLAT_LABEL_MLT = 3.5  # Upper right

MLT_GRID_LINES = np.arange(0, 24, 3)  # MLT lines to draw (every 3 hours)
MLT_LABELS_LIST = [0, 6, 12, 18]  # MLT labels to show

# --- mplcursors Check ---
if enable_hover_info:
    try:
        import mplcursors
    except ImportError:
        print("Warning: mplcursors library not found. Hover info disabled.")
        print("Install it using: pip install mplcursors")
        enable_hover_info = False


# --- Helper Functions (Unchanged - logic handles pole='S') ---
def mlat_to_radius(mlat, pole_type="N"):
    """Converts Magnetic Latitude (MLat) to polar radius."""
    if pole_type == "N":
        return 90.0 - np.asarray(mlat)
    else:
        return 90.0 + np.asarray(mlat)  # Correct for South Pole


def mlt_to_theta(mlt):
    """Converts Magnetic Local Time (MLT) to polar angle (theta) in radians."""
    return (np.asarray(mlt) / 24.0 - 0.25) * 2 * np.pi


def radius_to_mlat(r, pole_type="N"):
    """Converts polar radius back to Magnetic Latitude (MLat)."""
    if pole_type == "N":
        return 90.0 - r
    else:
        return r - 90.0  # Correct for South Pole


def theta_to_mlt(theta):
    """Converts polar angle (theta) back to Magnetic Local Time (MLT)."""
    theta_normalized = theta % (2 * np.pi)
    mlt = (theta_normalized / (2 * np.pi) + 0.25) * 24.0
    return mlt % 24.0


# --- Data Loading (Loads all potential data first) ---
all_loaded_tracks = []  # Store tuples of (dataframe, original_filename)


def get_pkl_files_pathlib(root_dir):
    root_path = Path(root_dir)
    pkl_files = list(root_path.rglob("*.pkl"))
    return pkl_files


pkl_paths = get_pkl_files_pathlib(target_dir)
print(f"Found {len(pkl_paths)} potential .pkl files.")

loaded_file_count = 0
for pkl_path in pkl_paths:
    try:
        df = pd.read_pickle(pkl_path)
        # Check essential columns exist BEFORE filtering by pole
        if "QDLat" in df.columns and "MLT" in df.columns:
            df_geomag = df[["QDLat", "MLT"]].dropna().astype(float)
            if not df_geomag.empty:
                all_loaded_tracks.append(
                    (df_geomag, pkl_path.name)
                )  # Store DF and filename
                loaded_file_count += 1
            # else: print(f"Info: Data from {pkl_path.name} empty after dropna/astype.") # Optional info
        # else: print(f"Warning: Columns 'QDLat' or 'MLT' not found in {pkl_path.name}. Skipping.") # Optional warning
    except Exception as e:
        print(f"Error loading or processing {pkl_path}: {e}")

print(f"Successfully loaded data from {loaded_file_count} files.")
if all_loaded_tracks:
    print(f"Shape of first loaded track data: {all_loaded_tracks[0][0].shape}")
else:
    print("Warning: No valid track data loaded. Cannot plot.")
    exit()  # Exit if no data is loaded


# --- Core Plotting Function ---
def plot_polar_view(ax, pole_type, track_data_list):
    """
    Plots satellite tracks and optionally an auroral oval representation
    on a given polar axes object for a specific pole.
    # ... (rest of docstring) ...
    """
    print(f"\n--- Plotting for {pole_type}-Pole ---")

    # --- Determine Pole-Specific Settings ---
    if pole_type == "N":
        mlat_grid_circles = DEFAULT_MLAT_GRID_CIRCLES
        plot_mlat_outer_boundary = DEFAULT_PLOT_MLAT_OUTER_BOUNDARY
        mlat_label_mlt = DEFAULT_MLAT_LABEL_MLT
        pole_title = "North Pole"
        # Use North Pole oval parameters directly
        oval_eq_avg = OVAL_EQUATORWARD_MLAT_AVG_N
        oval_pol_avg = OVAL_POLEWARD_MLAT_AVG_N
        # Day-night difference makes oval extend further equatorward (lower MLat) at night
        oval_eq_sign_factor = -1
        oval_pol_sign_factor = -1
    else:  # South Pole settings
        mlat_grid_circles = np.arange(-80, -59, 10)
        plot_mlat_outer_boundary = -58
        mlat_label_mlt = 3.5
        pole_title = "South Pole"
        # Use negative of North Pole average MLats for South
        oval_eq_avg = -OVAL_EQUATORWARD_MLAT_AVG_N
        oval_pol_avg = -OVAL_POLEWARD_MLAT_AVG_N
        # Day-night difference makes oval extend further equatorward (more negative MLat) at night
        oval_eq_sign_factor = (
            +1
        )  # Add difference to move towards equator (less negative)
        oval_pol_sign_factor = +1

    # --- Set Plot Limits and Appearance ---
    max_radius = mlat_to_radius(plot_mlat_outer_boundary, pole_type)
    print(
        f"  Outer MLat Boundary: {plot_mlat_outer_boundary}, Max Radius: {max_radius}"
    )
    ax.set_ylim(0, max_radius)
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # --- Set Subplot Title ---
    ax.set_title(pole_title, weight="bold", fontsize=12, pad=25)  # Increased pad

    # --- Draw MLat Circles and MLT Meridians ---
    # ... (Grid line drawing code remains the same) ...
    thetas_for_circle = np.linspace(0, 2 * np.pi, 150)
    mlat_label_positions = {}
    for mlat_val in mlat_grid_circles:
        r = mlat_to_radius(mlat_val, pole_type)
        if r <= max_radius:
            ax.plot(
                thetas_for_circle,
                np.full_like(thetas_for_circle, r),
                "k--",
                linewidth=0.8,
                alpha=0.7,
            )
            label_theta = mlt_to_theta(mlat_label_mlt)
            mlat_label_positions[mlat_val] = (label_theta, r)

    mlt_label_positions = {}
    r_line_min = 0
    r_line_max = max_radius
    for mlt_val in MLT_GRID_LINES:
        theta = mlt_to_theta(mlt_val)
        ax.plot(
            [theta, theta], [r_line_min, r_line_max], "k--", linewidth=0.8, alpha=0.7
        )
        if mlt_val in MLT_LABELS_LIST:
            mlt_label_positions[mlt_val] = (theta, max_radius * 1.06)

    # --- Add MLat and MLT Labels ---
    # ... (Label drawing code remains the same) ...
    for mlat_val, (theta, r) in mlat_label_positions.items():
        ax.text(
            theta,
            r + 0.02 * max_radius,
            f"{int(abs(mlat_val))}$^\circ$",
            color="dimgray",
            ha="center",
            va="center",
            fontsize=9,
        )
    for mlt_val, (theta, r) in mlt_label_positions.items():
        label_text = f"{int(mlt_val)}"
        if mlt_val == 12:
            ha, va = "center", "bottom"
        elif mlt_val == 6:
            ha, va = "left", "center"
        elif mlt_val == 0:
            ha, va = "center", "top"
            label_text = f"{int(mlt_val)} MLT"
        elif mlt_val == 18:
            ha, va = "right", "center"
        else:
            ha, va = "center", "center"
        ax.text(
            theta,
            r,
            label_text,
            color="black",
            ha=ha,
            va=va,
            fontsize=10,
            weight="medium",
        )

    # --- *** NEW: Plot Auroral Oval (if enabled) *** ---
    oval_plotted = False  # Flag to track if the oval was actually plotted
    if PLOT_AURORAL_OVAL:
        print(f"  Plotting approximate auroral oval...")
        n_points_oval = 200
        thetas_oval = np.linspace(0, 2 * np.pi, n_points_oval)
        mlts_oval = theta_to_mlt(
            thetas_oval
        )  # For potential MLT-dependent model (using cosine)

        # Calculate MLat boundaries with day-night asymmetry
        # cos(theta - pi/2) is max (+1) at midnight (theta=3pi/2), min (-1) at noon (theta=pi/2)
        cos_term = np.cos(thetas_oval - np.pi / 2.0)

        # Equatorward boundary MLat (lower abs(MLat))
        mlat_eq = (
            oval_eq_avg
            + oval_eq_sign_factor * OVAL_DAY_NIGHT_MLAT_DIFF * (1 - cos_term) / 2.0
        )
        # Poleward boundary MLat (higher abs(MLat))
        mlat_pol = (
            oval_pol_avg
            + oval_pol_sign_factor * OVAL_DAY_NIGHT_MLAT_DIFF * (1 - cos_term) / 2.0
        )

        # Convert MLat boundaries to radii
        radius_eq = mlat_to_radius(mlat_eq, pole_type)
        radius_pol = mlat_to_radius(mlat_pol, pole_type)

        # Ensure radii are within plot limits (using max_radius calculated earlier)
        # We plot between the poleward radius (smaller) and equatorward radius (larger)
        # Clip radii to be within the plot boundaries (0 to max_radius)
        radius_pol_clipped = np.clip(radius_pol, 0, max_radius)
        radius_eq_clipped = np.clip(radius_eq, 0, max_radius)

        # Plot the filled region
        ax.fill_between(
            thetas_oval,
            radius_pol_clipped,
            radius_eq_clipped,
            color=OVAL_COLOR,
            alpha=OVAL_ALPHA,
            label="Auroral Oval (Approx.)",
        )
        oval_plotted = True  # Mark that we attempted to plot it (and added the label)
    # --- *** End of Auroral Oval Plotting *** ---

    # --- Plot Tracks for the specific pole ---
    plotted_lines = []
    tracks_plotted_count = 0
    legend_label_track_added = False  # Reset track label flag for each subplot

    for df_track_geomag, filename in track_data_list:
        # --- Define df_pole_filtered first based on pole_type ---
        if pole_type == "N":
            df_pole_filtered = df_track_geomag[df_track_geomag["QDLat"] >= 0]
        else:  # pole_type == 'S'
            df_pole_filtered = df_track_geomag[df_track_geomag["QDLat"] <= 0]
        # ----------------------------------------------------------

        # --- Now process the correctly filtered dataframe ---
        if not df_pole_filtered.empty:
            track_data = df_pole_filtered[
                ["QDLat", "MLT"]
            ].values  # Convert to numpy here
            if track_data.shape[0] >= 2:  # Need at least 2 points
                # Convert coordinates
                mlats = track_data[:, 0]
                mlts = track_data[:, 1]
                thetas = mlt_to_theta(mlts)
                radii = mlat_to_radius(mlats, pole_type)

                # Filter points outside the plot boundary
                valid_indices = np.where(radii <= max_radius)[0]

                if len(valid_indices) >= 2:
                    # Find breaks to plot segments
                    breaks = np.where(np.diff(valid_indices) != 1)[0] + 1
                    segment_indices = np.split(valid_indices, breaks)
                    segment_plotted_for_this_file = False

                    for segment in segment_indices:
                        if len(segment) >= 2:
                            label_to_add = None
                            # Add satellite track label only once *per subplot*
                            if not legend_label_track_added:
                                label_to_add = "Satellite Track"
                                legend_label_track_added = True

                            (line,) = ax.plot(
                                thetas[segment],
                                radii[segment],
                                color=SATELLITE_TRACK_COLOR,
                                linewidth=SATELLITE_TRACK_WIDTH,
                                alpha=0.8,
                                label=label_to_add,  # Pass label
                                zorder=5,
                            )  # Ensure tracks are drawn on top of oval
                            plotted_lines.append(line)
                            segment_plotted_for_this_file = True

                    if segment_plotted_for_this_file:
                        tracks_plotted_count += 1
            # else: print(f"  Skipping track from {filename} ({pole_type}): Less than 2 data points after filtering.") # Optional info
        # else: print(f"  Skipping track from {filename} ({pole_type}): No data for this pole.") # Optional info

    print(f"  Plotted {tracks_plotted_count} track segments for {pole_type}-pole.")
    # --- Update text condition check ---
    # Check if ONLY the oval was plotted (no tracks) OR nothing was plotted at all
    if not legend_label_track_added and not oval_plotted:  # Nothing plotted
        ax.text(
            0.5,
            0.5,
            "No data\nin range",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            transform=ax.transAxes,
        )
    elif (
        not legend_label_track_added and oval_plotted
    ):  # Oval plotted, but no tracks in range
        ax.text(
            0.5,
            0.5,
            "No track data\nin range",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            transform=ax.transAxes,
            zorder=6,
        )  # Ensure text is on top

    # --- Add Legend (if anything was labeled) ---
    # Call legend() if either the oval or the track label was added
    if oval_plotted or legend_label_track_added:
        ax.legend(loc="lower right", fontsize=9)

    return plotted_lines


# --- Figure Creation and Plotting Execution ---


# Setup hover callback function (used by both single and dual plots)
def create_hover_callback(pole_type_for_callback):
    """Creates a callback function for mplcursors specific to a pole."""

    def on_add(sel):
        theta_hover, radius_hover = sel.target
        # Use the specific pole_type captured by the closure
        mlat_hover = radius_to_mlat(radius_hover, pole_type_for_callback)
        mlt_hover = theta_to_mlt(theta_hover)
        annotation_text = f"MLT: {mlt_hover:.2f}\nMLat: {mlat_hover:.2f}"
        sel.annotation.set(text=annotation_text, fontsize=9)  # Smaller font
        # Use default annotation appearance
        sel.annotation.get_bbox_patch().set(facecolor="yellow", alpha=0.7)
        # sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7) # Optional arrow styling

    return on_add


# --- Main Plotting Logic ---
if pole_mode == "N" or pole_mode == "S":
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    plotted_lines = plot_polar_view(ax, pole_mode, all_loaded_tracks)

    # Setup hover info if enabled and lines were plotted
    if enable_hover_info and plotted_lines:
        print("Hover info enabled for single plot.")
        cursor = mplcursors.cursor(plotted_lines, hover=True)
        cursor.connect("add", create_hover_callback(pole_mode))  # Pass the correct pole
    elif enable_hover_info:
        print("Hover info enabled, but no valid track lines were plotted.")

elif pole_mode == "Both":
    # Create figure with two subplots side-by-side
    fig, axs = plt.subplots(
        1, 2, figsize=(14, 7.5), subplot_kw={"projection": "polar"}
    )  # Adjusted figsize

    # Plot North Pole on the left (axs[0])
    plotted_lines_n = plot_polar_view(axs[0], "N", all_loaded_tracks)

    # Plot South Pole on the right (axs[1])
    plotted_lines_s = plot_polar_view(axs[1], "S", all_loaded_tracks)

    # Add overall figure title
    fig.suptitle(
        "Satellite Tracks (North and South Pole Views)", fontsize=16, y=0.99
    )  # Adjust y position

    # Setup hover info separately for each subplot if enabled
    if enable_hover_info:
        if plotted_lines_n:
            print("Hover info enabled for North Pole plot.")
            cursor_n = mplcursors.cursor(plotted_lines_n, hover=True)
            cursor_n.connect(
                "add", create_hover_callback("N")
            )  # Create callback specific to 'N'
        else:
            print(
                "Hover info enabled, but no valid North Pole track lines were plotted."
            )

        if plotted_lines_s:
            print("Hover info enabled for South Pole plot.")
            cursor_s = mplcursors.cursor(plotted_lines_s, hover=True)
            cursor_s.connect(
                "add", create_hover_callback("S")
            )  # Create callback specific to 'S'
        else:
            print(
                "Hover info enabled, but no valid South Pole track lines were plotted."
            )

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Add rect to make space for suptitle

else:
    print(f"Error: Invalid pole_mode '{pole_mode}'. Choose 'N', 'S', or 'Both'.")
    exit()

# --- Final Touches ---
if not enable_hover_info:
    print("Hover info disabled.")

# --- Saving ---
if SAVE:
    if not PLOT_AURORAL_OVAL:
        save_n = f"mlt_{pole_mode}_all_cases.png"
    else:
        save_n = f"mlt_{pole_mode}_add_approx_aurora_oval_all_cases.png"
    save_path = os.path.join(save_dir, save_n)  # Use Path object for joining
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Use bbox_inches='tight'
    print(f"Plot saved to: {save_path}")

plt.show()
