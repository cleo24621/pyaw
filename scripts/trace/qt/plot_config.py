# plot_config.py
import os
import json
import re
import sys
import numpy as np
import pandas as pd
import pyvista as pv

# --- Qt Imports ---
try:
    from qtpy import QtCore, QtWidgets
    import pyvistaqt as pvqt

    QT_AVAILABLE = True
    print("Config: Qt and pyvistaqt imported successfully.")
except ImportError as e:
    print(
        f"Config Warning: Failed to import Qt modules ({e}). BackgroundPlotter might not work reliably."
    )
    QT_AVAILABLE = False
# -----------------
from spacepy import coordinates as coord
from spacepy.time import Ticktock
import glob

print("--- Initializing plot_config ---")

# %% --- Default Workflow Control ---
# Individual scripts can override this based on their needs or arguments
INTERACTIVE_MODE_DEFAULT = True  # Default behavior if script doesn't set it

# %% --- Constants, Paths, Styles --- (Same as before)
R_E = 1.0
Re_km = 6371.2
SWARM_RADIUS = 6816838.5 * 1e-3
SWARM_ALTITUDE = SWARM_RADIUS - Re_km
EARTH_TEXTURE_PATH = "eo_base_2020_clean_3600x1800.png"
TRACE_FILE_DIR = (
    r"G:\master\pyaw\scripts\results\aw_cases\archive\trace_points\pkl\12728"
)
SATELLITE_DIR = r"G:\master\pyaw\scripts\results\aw_cases\archive\orbits\12728"
# CAMERA_POS_FILE = "camera_positions.json" # REMOVED - Handled by each script

FIG_BG_COLOR = "#282c34"
TEXT_COLOR = "#abb2bf"
AXIS_LINE_COLOR = "#4b5263"
EARTH_COLOR = "#678094"
EARTH_SPECULAR = 0.2
EARTH_SPECULAR_POWER = 10
EARTH_SMOOTH_SHADING = True
FIELD_LINE_COLOR = "#ff0000"
FIELD_LINE_OPACITY = 0.65
FIELD_LINE_WIDTH = 1.5
POINT_COLOR = "#ffffff"
POINT_SIZE = 5
POINT_ALPHA = 0.9
SAT_TRACK_COLOR = "#56b6c2"
SAT_TRACK_WIDTH = 20
SAT_TRACK_OPACITY = 0.9
SAT_POS_COLOR = "#e06c75"
SAT_POS_SIZE = 10
POLE_MARKER_COLOR = "#ffffff"
POLE_MARKER_RADIUS = 0.03
AURORA_COLOR = "#98c379"
AURORA_OPACITY = 0.4  # Keep if add_aurora is defined
MAX_PLOT_RADIUS = 10.0

# %% --- Screenshot Resolution ---
BASE_WINDOW_WIDTH = 1500
BASE_WINDOW_HEIGHT = 800
IMAGE_SCALE = 4

# %% --- Initialize QApplication --- (Keep as before)
app = None
if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance()
    if app is None:
        print("Config: Creating new QApplication.")
        qt_args = [arg for arg in sys.argv if not arg.lower().endswith(".py")]
        if not qt_args:
            qt_args = [sys.executable]
        try:
            app = QtWidgets.QApplication(qt_args)
        except Exception as e:
            print(f"Config ERROR: Failed creating QApplication: {e}")
            QT_AVAILABLE = False
    else:
        print("Config: Using existing QApplication.")


# %% --- Data Loading Helpers & Data --- (Keep definitions and loading logic here)
# ... geosph2geocar, find_contiguous_segments, extract_timestamps ...
# ... Texture, Trace, Satellite data loading into module variables ...
# (Same as previous plot_config.py)
def geosph2geocar(times, alts, lats, lons):  # ... (full implementation)
    arr_n_3 = np.column_stack((alts, lats, lons))
    c = coord.Coords(arr_n_3, "GEO", "sph", ticks=Ticktock(times, "ISO"))
    return c.convert("GEO", "car")


def find_contiguous_segments(mask):  # ... (full implementation)
    segments = []
    mask = np.asarray(mask, dtype=bool)
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for start, end in zip(starts, ends):
        start = max(0, start)
        end = min(len(mask), end)
    if start < end:
        segments.append((start, end))
        return segments


# %% get time for
def extract_timestamps(filename):
    """提取文件名中的时间信息（自动区分start/end）"""
    match = re.search(r"_case_(\d{8}T\d{6})_(\d{8}T\d{6})\.pkl$", filename)
    if not match:
        raise ValueError(f"文件名格式不符: {filename}")

    if filename.startswith("start_"):
        return match.group(1)  # start文件返回第一个时间
    elif filename.startswith("end_"):
        return match.group(2)  # end文件返回第二个时间
    else:
        raise ValueError("文件名必须以 'start_' 或 'end_' 开头")


texture = None  # ... (texture loading block) ...
if os.path.exists(EARTH_TEXTURE_PATH):  # ... (texture loading logic) ...
    try:
        texture = pv.read_texture(EARTH_TEXTURE_PATH)
        print(f"Config: Loaded texture: {EARTH_TEXTURE_PATH}")
    except Exception as e:
        print(f"Config Error: loading texture: {e}")
else:
    print(f"Config Warning: Texture not found: {EARTH_TEXTURE_PATH}")
magnetic_field_lines_data = []  # ... (trace loading block) ...
try:
    TRACE_FILE_PATHS = glob.glob(os.path.join(TRACE_FILE_DIR, "*.pkl"))
    TRACE_FILE_NAMES = [
        os.path.basename(path) for path in TRACE_FILE_PATHS
    ]
except Exception as e:
    print(f"Config Error finding trace files: {e}")
print(f"Loading {len(TRACE_FILE_PATHS)} trace magnetic field lines...")
for (i, trace_path), trace_fn in zip(enumerate(TRACE_FILE_PATHS), TRACE_FILE_NAMES):
    print(f"Processing trace magnetic field line {i+1}: {trace_path}")
    try:
        df_trace = pd.read_pickle(trace_path)
        df_trace_np = df_trace.values
        df_trace_np_alt_re = df_trace_np.copy()
        df_trace_np_alt_re[:, 0] = (df_trace_np_alt_re[:, 0] + Re_km) / Re_km
        time_trace = extract_timestamps(trace_fn)
        times_trace = np.array([str(time_trace)] * len(df_trace_np_alt_re[:, 0]))
        geo_car_trace = geosph2geocar(
            times_trace,
            df_trace_np_alt_re[:, 0],
            df_trace_np_alt_re[:, 1],
            df_trace_np_alt_re[:, 2],
        )
        magnetic_field_lines_data.append(geo_car_trace.data)
        print(
            f"trace magnetic field line {i+1} loaded successfully ({geo_car_trace.data.shape[0]} points)."
        )
    except Exception as e:
        print(f"Error processing trace points file {trace_path}: {e}. Skipping.")
if not magnetic_field_lines_data:
    print("Warning: No trace points file loaded successfully.")

satellite_tracks_data = []  # ... (satellite loading block) ...
satellite_markers_data = [[]]
try:
    SATELLITE_PATHS = glob.glob(
        os.path.join(SATELLITE_DIR, "*.pkl")
    )  # ... (rest of satellite loading logic) ...
except Exception as e:
    print(f"Config Error finding satellite files: {e}")
print(f"Loading {len(SATELLITE_PATHS)} satellite track(s)...")
if len(SATELLITE_PATHS) != len(satellite_markers_data):
    print("Warning: Mismatch between satellite paths and marker lists!")

for i, sat_path in enumerate(SATELLITE_PATHS):
    print(f"Processing track {i+1}: {sat_path}")
    try:
        df_sa = pd.read_pickle(sat_path)
        lats_sa = df_sa["Latitude"].values
        lons_sa = df_sa["Longitude"].values
        alts_sa = (np.full(len(lats_sa), SWARM_ALTITUDE) + Re_km) / Re_km
        times_sa = df_sa.index.values
        times_sa_str = np.array([str(t) for t in times_sa])
        geo_car_sa = geosph2geocar(times_sa_str, alts_sa, lats_sa, lons_sa)
        satellite_tracks_data.append(geo_car_sa.data)
        print(f"Track {i+1} loaded successfully ({geo_car_sa.data.shape[0]} points).")
    except Exception as e:
        print(f"Error processing satellite file {sat_path}: {e}. Skipping.")
if not satellite_tracks_data:
    print("Warning: No satellite tracks loaded successfully.")


# %% --- Function to Load ONE Camera Position ---
def load_camera_position(filename):
    """Loads a single camera position from a JSON file."""
    position = None
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                # Expecting the file to contain just the list for one position
                pos_list = json.load(f)
                if isinstance(pos_list, list) and len(pos_list) == 3:
                    position = [
                        tuple(pos_list[0]),
                        tuple(pos_list[1]),
                        tuple(pos_list[2]),
                    ]
                else:
                    print(f"Config Warning: Invalid format in {filename}")
            print(f"Config: Loaded camera position from {filename}")
        except Exception as e:
            print(f"Config Warning: Load position failed from {filename}: {e}")
    else:
        print(f"Config Warning: Camera file {filename} not found.")
    return position


# %% --- Function to Save ONE Camera Position ---
def save_camera_position(filename, position_list):
    """Saves a single camera position (list of lists) to a JSON file."""
    if not (isinstance(position_list, list) and len(position_list) == 3):
        print(f"Config Error: Invalid position format for saving to {filename}")
        return
    try:
        with open(filename, "w") as f:
            json.dump(position_list, f, indent=4)
        print(f"Config: Saved camera position to {filename}")
    except Exception as e:
        print(f"Config Warning: Could not save position to {filename}: {e}")


# %% --- Signal Emitter Class (needed for button interaction) ---
if QT_AVAILABLE:

    class SignalEmitter(QtCore.QObject):
        proceed_signal = QtCore.Signal()


# %% --- Plotting Functions (Defined here) ---
# create_aurora_texture, add_aurora, add_earth_features
# add_fieldlines_and_points, add_satellite_track, apply_final_settings
# (Definitions remain the same)
# ... (Copy the 6 function definitions here) ...
def create_aurora_texture(
    color_hex, opacity, height=64
):  # ... (full implementation) ...
    try:
        color_obj = pv.Color(color_hex, default_opacity=opacity)
        rgb_tuple = color_obj.int_rgb
    except Exception as e:
        print(f"Error creating aurora texture (color): {e}")
        return None
    if rgb_tuple is None:
        return None
        color_r, color_g, color_b = rgb_tuple
        max_alpha = int(opacity * 255)
    gradient = np.concatenate(
        [
            np.linspace(0, max_alpha, height // 2),
            np.linspace(max_alpha, 0, height - height // 2),
        ]
    ).astype(np.uint8)
    tex_arr = np.zeros((height, 1, 4), dtype=np.uint8)
    tex_arr[:, 0, :3] = [color_r, color_g, color_b]
    tex_arr[:, 0, 3] = gradient
    return pv.Texture(tex_arr)


def add_aurora(
    plotter_instance,
    lat=70,
    thickness=0.15,
    color=AURORA_COLOR,
    opacity=AURORA_OPACITY,
    texture_height=128,
):  # ... (full implementation) ...
    try:
        radius = R_E * np.cos(np.radians(lat))
        z_offset = R_E * np.sin(np.radians(lat))
        aurora_tex = create_aurora_texture(color, opacity, height=texture_height)
        if not aurora_tex:
            return False
        for pole, direction_z, name in [("North", 1, "north"), ("South", -1, "south")]:
            center = (0, 0, direction_z * z_offset)
            aurora_cyl = pv.Cylinder(
                center=center,
                direction=(0, 0, direction_z),
                radius=radius,
                height=thickness,
                capping=False,
                resolution=100,
            )
            aurora_cyl.texture_map_to_plane(inplace=True, use_bounds=True)
            plotter_instance.add_mesh(
                aurora_cyl,
                texture=aurora_tex,
                opacity=1.0,
                show_edges=False,
                smooth_shading=True,
                rgb=False,
                name=f"{name}_aurora",
                use_transparency=True,
            )
        return True
    except Exception as e:
        print(f"ERROR adding aurora: {e}")
        return False


def add_earth_features(plotter_instance):  # ... (full implementation) ...
    earth = pv.Sphere(radius=R_E, theta_resolution=120, phi_resolution=120)
    try:
        earth.texture_map_to_sphere(inplace=True)
    except Exception as e:
        print(f"Warning: Earth TCoord failed: {e}")
    try:
        if texture:
            plotter_instance.add_mesh(
                earth,
                texture=texture,
                smooth_shading=EARTH_SMOOTH_SHADING,
                specular=EARTH_SPECULAR,
                specular_power=EARTH_SPECULAR_POWER,
                show_edges=False,
                rgb=False,
                name="earth",
            )
        else:
            plotter_instance.add_mesh(
                earth,
                color=EARTH_COLOR,
                smooth_shading=EARTH_SMOOTH_SHADING,
                specular=EARTH_SPECULAR,
                specular_power=EARTH_SPECULAR_POWER,
                show_edges=False,
                name="earth",
            )
    except Exception as e:
        print(f"Warning: Earth mesh failed: {e}")
        plotter_instance.add_mesh(earth, color=EARTH_COLOR, name="earth_fallback")
    try:
        north_pole = pv.Sphere(radius=POLE_MARKER_RADIUS, center=(0, 0, R_E))
        south_pole = pv.Sphere(radius=POLE_MARKER_RADIUS, center=(0, 0, -R_E))
        plotter_instance.add_mesh(
            north_pole, color=POLE_MARKER_COLOR, name="north_pole_marker"
        )
        plotter_instance.add_mesh(
            south_pole, color=POLE_MARKER_COLOR, name="south_pole_marker"
        )
        plotter_instance.add_point_labels(
            [(0, 0, R_E * 1.1)],
            ["N"],
            point_size=10,
            font_size=12,
            text_color=TEXT_COLOR,
            shape=None,
            name="north_label",
        )
        plotter_instance.add_point_labels(
            [(0, 0, -R_E * 1.1)],
            ["S"],
            point_size=10,
            font_size=12,
            text_color=TEXT_COLOR,
            shape=None,
            name="south_label",
        )
    except Exception as e:
        print(f"Warning: Failed poles/labels: {e}")
    return plotter_instance


def add_fieldlines_and_points(
    plotter_instance, lines_data, max_radius
):  # ... (full implementation) ...
    all_endpoints = []
    line_count = 0
    for line_data in lines_data:
        if line_data is None or line_data.shape[0] < 2:
            continue
        r = np.linalg.norm(line_data, axis=1)
        mask = r <= max_radius
        for start, end in find_contiguous_segments(np.asarray(mask)):
            segment = line_data[start:end]
            if segment.shape[0] >= 2:
                line_count += 1
                lines = np.c_[
                    np.full(len(segment) - 1, 2),
                    np.arange(len(segment) - 1),
                    np.arange(1, len(segment)),
                ]
                line_poly = pv.PolyData(segment, lines=lines)
                plotter_instance.add_mesh(
                    line_poly,
                    color=FIELD_LINE_COLOR,
                    opacity=FIELD_LINE_OPACITY,
                    line_width=FIELD_LINE_WIDTH,
                )
                all_endpoints.extend([segment[0], segment[-1]])
            elif segment.shape[0] == 1:
                all_endpoints.append(segment[0])
    if all_endpoints:
        coords = np.array(all_endpoints)
        plotter_instance.add_points(
            coords,
            color=POINT_COLOR,
            point_size=POINT_SIZE,
            render_points_as_spheres=True,
            opacity=POINT_ALPHA,
            label=f"Field Endpoints ({line_count})",
        )
    return plotter_instance


def add_satellite_track(
    plotter_instance, tracks_list, marker_indices_list, max_radius
):  # ... (full implementation) ...
    safe_markers = marker_indices_list + [[]] * (
        len(tracks_list) - len(marker_indices_list)
    )
    legend_added = False
    for i, (track_data, markers) in enumerate(zip(tracks_list, safe_markers)):
        if track_data is None or track_data.shape[0] < 2:
            continue
        r_sat = np.linalg.norm(track_data, axis=1)
        mask_sat = r_sat <= max_radius
        for start, end in find_contiguous_segments(np.asarray(mask_sat)):
            segment = track_data[start:end]
            if segment.shape[0] >= 2:
                lines = np.c_[
                    np.full(len(segment) - 1, 2),
                    np.arange(len(segment) - 1),
                    np.arange(1, len(segment)),
                ]
                track_poly = pv.PolyData(segment, lines=lines)
                plotter_instance.add_mesh(
                    track_poly,
                    color=SAT_TRACK_COLOR,
                    line_width=SAT_TRACK_WIDTH,
                    opacity=SAT_TRACK_OPACITY,
                    label="Satellite Track" if not legend_added else None,
                )
                legend_added = True
        valid_pts = [
            track_data[idx]
            for idx in markers
            if 0 <= idx < len(track_data)
            and np.linalg.norm(track_data[idx]) <= max_radius
        ]
        if valid_pts:
            plotter_instance.add_points(
                np.array(valid_pts),
                color=SAT_POS_COLOR,
                point_size=SAT_POS_SIZE,
                render_points_as_spheres=True,
                opacity=POINT_ALPHA,
                label=f"Sat Markers (T{i+1})" if i == 0 else None,
            )
    return plotter_instance


def apply_final_settings(plotter_instance, max_radius):  # ... (full implementation) ...
    print("Config: Applying final settings...")
    plotter_instance.add_axes(interactive=True, line_width=2, color=TEXT_COLOR)
    title = f"Alfven Wave Magnetic Conjugacy (View range: {max_radius} Rₑ)"
    plotter_instance.add_text(
        title, position="upper_edge", color=TEXT_COLOR, font_size=12, name="final_title"
    )
    return plotter_instance


print("--- plot_config initialized ---")
