# %%
import os
import json
import re

import numpy as np
import pandas as pd
import pyvista as pv
from spacepy import coordinates as coord
from spacepy.time import Ticktock
import glob

# %% --- Backend Setup --- (remains the same)
try:
    pv.set_jupyter_backend("")
    pv.global_theme.trame.server_proxy_enabled = False
    print("Attempting to use default desktop (VTK/Qt) backend.")
except Exception as e:
    print(f"Could not configure backend, using PyVista default: {e}")

# %% --- Workflow and Control Flags ---
INTERACTIVE_MODE = True
PLOT_AURORA = False
CAMERA_POS_FILE = "camera_positions.json"  # pass to screenshot
# ---------------------------------

# %% --- Constants and Paths ---
R_E = 1.0
Re_km = 6371.2
SWARM_RADIUS = 6816838.5 * 1e-3  # km
SWARM_ALTITUDE = SWARM_RADIUS - Re_km  # km
EARTH_TEXTURE_PATH = "eo_base_2020_clean_3600x1800.png"
TRACE_FILE_DIR = (
    r"G:\master\pyaw\scripts\results\aw_cases\archive\trace_points\pkl\12728"
)
TRACE_FILE_PATHS = glob.glob(os.path.join(TRACE_FILE_DIR, "*.pkl"))
TRACE_FILE_NAMES = [os.path.basename(path) for path in TRACE_FILE_PATHS]
SATELLITE_DIR = r"G:\master\pyaw\scripts\results\aw_cases\archive\orbits\12728"
SATELLITE_PATHS = glob.glob(os.path.join(SATELLITE_DIR, "*.pkl"))
# SATELLITE_MARKER_INDICES = [[10000, 25000, 40000]]
SATELLITE_MARKER_INDICES = [[]]

# --- Style Settings ---
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
AURORA_OPACITY = 0.4
MAX_PLOT_RADIUS = 10.0
# WINDOW_WIDTH = 1500
# WINDOW_HEIGHT = 800
WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160
# WINDOW_WIDTH = 7680
# WINDOW_HEIGHT = 4320
IMAGE_SCALE = 4

# ---------------------

# %% --- Earth Texture Loading ---
texture = None
if not os.path.exists(EARTH_TEXTURE_PATH):
    print(f"Warning: Earth texture file not found at '{EARTH_TEXTURE_PATH}'.")
else:
    try:
        texture = pv.read_texture(EARTH_TEXTURE_PATH)
        print(f"Successfully loaded Earth texture from '{EARTH_TEXTURE_PATH}'.")
    except Exception as e:
        print(f"Error loading texture '{EARTH_TEXTURE_PATH}': {e}.")


# %% --- Data Loading Functions ---
def geosph2geocar(times, alts, lats, lons):
    arr_n_3 = np.column_stack((alts, lats, lons))
    c = coord.Coords(arr_n_3, "GEO", "sph", ticks=Ticktock(times, "ISO"))
    return c.convert("GEO", "car")


def find_contiguous_segments(mask):
    segments = []
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for start, end in zip(starts, ends):
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


# %% --- Load Trace Data ---
magnetic_field_lines_data = []
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


# %% --- Load Satellite Data ---
satellite_tracks_data = []
satellite_markers_data = SATELLITE_MARKER_INDICES  # Assign to the variable used later
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


# %% --- Plotting Helper Functions ---


def show_or_screenshot_step(plotter, step_description, camera_pos_dict, step_key):
    """Handles showing plot or taking screenshot, manages camera positions."""
    plotter.add_text(
        f"Step: {step_description}",
        position="lower_edge",
        font_size=10,
        color=TEXT_COLOR,
        name="step_text",
    )
    if INTERACTIVE_MODE:
        print(f"\n--- Showing Interactive Step: {step_description} ---")
        print(">>> Adjust view. Close window to continue AND SAVE VIEW. <<<")
        plotter.show()
        try:
            raw_camera_pos = plotter.camera_position
            camera_pos_list = [
                list(raw_camera_pos[0]),
                list(raw_camera_pos[1]),
                list(raw_camera_pos[2]),
            ]
            camera_pos_dict[step_key] = camera_pos_list
            print(f"Camera position for '{step_key}' saved.")
        except Exception as e:
            print(f"Warning: Could not get/save camera position for '{step_key}': {e}")
    else:  # Batch mode
        filename = f"step_{step_key}_screenshot.png"
        print(f"\n--- Generating Screenshot: {step_description} ({filename}) ---")
        if step_key in camera_pos_dict:
            try:
                plotter.camera_position = camera_pos_dict[step_key]
                print(f"Applied saved camera position for '{step_key}'.")
            except Exception as e:
                print(f"Warning: Could not apply camera position for '{step_key}': {e}")
        else:
            print(f"Warning: No saved camera position for '{step_key}'. Using default.")
        try:
            plotter.render()
            plotter.screenshot(filename, transparent_background=False)
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"ERROR taking screenshot for '{step_key}': {e}")


def create_aurora_texture(color_hex, opacity, height=64):
    """Creates the aurora gradient texture."""
    try:
        color_obj = pv.Color(color_hex, default_opacity=opacity)
        rgb_tuple = color_obj.int_rgb
        if rgb_tuple is None:
            raise ValueError("Invalid color")
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
        # print(f"Created aurora texture: ({color_r}, {color_g}, {color_b})") # Less verbose
        return pv.Texture(tex_arr)
    except Exception as e:
        print(f"Error creating aurora texture: {e}")
        return None


def add_aurora(
    plotter_instance,
    lat=70,
    thickness=0.15,
    color=AURORA_COLOR,
    opacity=AURORA_OPACITY,
    texture_height=128,
):
    """Adds aurora rings to the plotter."""
    # print("Attempting to add aurora...") # Less verbose
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
            # print(f"Added {pole} aurora mesh.") # Less verbose
        return True
    except Exception as e:
        print(f"ERROR adding aurora: {e}")
        return False


def add_earth_features(plotter_instance):
    """Adds Earth sphere, poles, and labels."""
    # print("--- Running add_earth_features ---") # Less verbose
    earth = pv.Sphere(radius=R_E, theta_resolution=120, phi_resolution=120)
    try:
        earth.texture_map_to_sphere(inplace=True)
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
        print(f"Warning: Earth processing failed: {e}")
        if not plotter_instance.renderer.actors.get("earth"):
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
        print(f"Warning: Failed adding poles/labels: {e}")
    # print("--- Exiting add_earth_features ---") # Less verbose
    return plotter_instance


def add_fieldlines_and_points(plotter_instance, lines_data, max_radius):
    """Adds magnetic field lines and their endpoints."""
    all_endpoints = []
    # print(f"Processing {len(lines_data)} field line(s)...") # Less verbose
    line_count = 0
    for line_data in lines_data:
        if line_data is None or line_data.shape[0] < 2 or line_data.shape[1] != 3:
            continue
        r = np.linalg.norm(line_data, axis=1)
        mask = r <= max_radius
        segments = find_contiguous_segments(np.asarray(mask))
        for start_idx, end_idx in segments:
            segment_data = line_data[start_idx:end_idx]
            if segment_data.shape[0] >= 2:
                line_count += 1
                points = segment_data
                n_points = len(points)
                lines_array = np.hstack(
                    [[2, k, k + 1] for k in range(n_points - 1)]
                ).astype(int)
                line_polydata = pv.PolyData(points, lines=lines_array)
                plotter_instance.add_mesh(
                    line_polydata,
                    color=FIELD_LINE_COLOR,
                    opacity=FIELD_LINE_OPACITY,
                    line_width=FIELD_LINE_WIDTH,
                )
                all_endpoints.extend([segment_data[0, :], segment_data[-1, :]])
            elif segment_data.shape[0] == 1:
                all_endpoints.append(segment_data[0, :])
    if all_endpoints:
        endpoints_coords = np.array(all_endpoints)
        plotter_instance.add_points(
            endpoints_coords,
            color=POINT_COLOR,
            point_size=POINT_SIZE,
            render_points_as_spheres=True,
            opacity=POINT_ALPHA,
            label=f"Field Line Endpoints ({line_count})",
        )
    return plotter_instance


def add_satellite_track(plotter_instance, tracks_list, marker_indices_list, max_radius):
    """Adds satellite tracks and markers."""
    # print(f"Processing {len(tracks_list)} satellite track(s)...") # Less verbose
    safe_markers = marker_indices_list + [[]] * (
        len(tracks_list) - len(marker_indices_list)
    )
    legend_added = False
    for i, (track_data, markers) in enumerate(zip(tracks_list, safe_markers)):
        # print(f" Track {i+1}: {track_data.shape[0]} points, {len(markers)} markers.") # Less verbose
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
        # Plot markers
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
            # print(f" Track {i+1}: Plotted {len(valid_pts)} markers.") # Less verbose
    return plotter_instance


def apply_final_settings(plotter_instance, max_radius):
    """Applies final camera, axes, and title."""
    print("Applying final settings...")
    try:
        if INTERACTIVE_MODE or "final" not in camera_positions:
            # Simplified default view adjustment
            plotter_instance.camera.zoom(
                0.7
            )  # Zoom out a bit from default instead of complex calc
            print("Set default zoom.")
    except Exception as e:
        print(f"Warning: Could not set default camera zoom - {e}.")
    plotter_instance.add_axes(interactive=True, line_width=2, color=TEXT_COLOR)
    title = f"Alfven Wave Magnetic Conjugacy (View range: {max_radius} Re)"
    plotter_instance.add_text(
        title, position="upper_edge", color=TEXT_COLOR, font_size=12, name="final_title"
    )
    return plotter_instance


# %% --- Refactored Plotting Function ---


def plot_step(
    step_key,
    step_description,
    camera_positions,
    # Pass necessary data for this step
    plot_fieldlines=None,
    plot_satellite=None,
    is_final=None,
):
    """Handles setup, plotting, and showing/screenshotting for a single step."""

    print(f"\n===== Processing Step: {step_description} =====")
    plotter = pv.Plotter(
        window_size=[WINDOW_WIDTH, WINDOW_HEIGHT], off_screen=not INTERACTIVE_MODE,image_scale=IMAGE_SCALE
    )
    plotter.enable_depth_peeling()

    plotter.set_background(FIG_BG_COLOR)
    add_earth_features(plotter)
    if PLOT_AURORA:
        add_aurora(plotter)

    # Add step-specific elements based on flags
    if plot_fieldlines:
        add_fieldlines_and_points(plotter, magnetic_field_lines_data, MAX_PLOT_RADIUS)
    if plot_satellite:
        add_satellite_track(
            plotter, satellite_tracks_data, satellite_markers_data, MAX_PLOT_RADIUS
        )

    # Apply final settings ONLY on the final step
    if is_final:
        apply_final_settings(plotter, MAX_PLOT_RADIUS)
        plotter.remove_actor(
            "step_text", render=False
        )  # Remove step text for final view

    # Show interactively or take screenshot
    show_or_screenshot_step(plotter, step_description, camera_positions, step_key)

    plotter.close()
    print(f"===== Completed Step: {step_description} =====")


# %% --- Main Execution ---

print(
    f"Starting PyVista plotting in {'INTERACTIVE' if INTERACTIVE_MODE else 'BATCH'} mode..."
)
pv.set_plot_theme("paraview")

# Load camera positions if file exists or if in batch mode
camera_positions = {}
if not INTERACTIVE_MODE or os.path.exists(CAMERA_POS_FILE):
    if os.path.exists(CAMERA_POS_FILE):
        try:
            with open(CAMERA_POS_FILE, "r") as f:
                loaded_pos = json.load(f)
                # Convert lists back to tuples
                for key, pos_list in loaded_pos.items():
                    if isinstance(pos_list, list) and len(pos_list) == 3:
                        camera_positions[key] = [
                            tuple(pos_list[0]),
                            tuple(pos_list[1]),
                            tuple(pos_list[2]),
                        ]
                    else:
                        camera_positions[key] = pos_list
                print(
                    f"Loaded {len(camera_positions)} camera positions from {CAMERA_POS_FILE}"
                )
        except Exception as e:
            print(
                f"Warning: Could not load camera positions from {CAMERA_POS_FILE}: {e}"
            )
            if not INTERACTIVE_MODE:
                print("Proceeding with default views.")
    elif not INTERACTIVE_MODE:
        print(
            f"Warning: Camera position file '{CAMERA_POS_FILE}' not found. Using default views."
        )

# --- Define and Run Steps ---
steps_definition = [
    {
        "key": "step1",
        "desc": "1_Add_Earth_Features" + ("_Aurora" if PLOT_AURORA else ""),
        "plot_fieldlines": False,
        "plot_satellite": False,
        "is_final": False,
    },
    {
        "key": "step2",
        "desc": "2_Add_Trace_Magnetic_Field_Lines",
        "plot_fieldlines": True,
        "plot_satellite": False,
        "is_final": False,
    },
    {
        "key": "step3",
        "desc": "3_Add_Satellite_Tracks",
        "plot_fieldlines": True,
        "plot_satellite": True,
        "is_final": False,
    },
    {
        "key": "final",
        "desc": "4_Final_Plot",
        "plot_fieldlines": True,
        "plot_satellite": True,
        "is_final": True,
    },  # Mark as final
]

for step_info in steps_definition:
    plot_step(
        step_key=step_info["key"],
        step_description=step_info["desc"],
        camera_positions=camera_positions,
        plot_fieldlines=step_info["plot_fieldlines"],
        plot_satellite=step_info["plot_satellite"],
        is_final=step_info["is_final"],
    )

# --- Save Camera Positions ---
if INTERACTIVE_MODE and camera_positions:
    try:
        # camera_positions already contains list-of-lists from the helper func
        with open(CAMERA_POS_FILE, "w") as f:
            json.dump(camera_positions, f, indent=4)
        print(f"\nSaved {len(camera_positions)} camera positions to {CAMERA_POS_FILE}")
    except Exception as e:
        print(f"Warning: Could not save camera positions: {e}")

print("\n绘图完成.")
