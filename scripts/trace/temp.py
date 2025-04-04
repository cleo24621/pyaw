# %%
import os
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import pyvista as pv
# --- Qt Imports ---
from qtpy import QtCore, QtWidgets # Need both
import pyvistaqt as pvqt
# -----------------
from spacepy import coordinates as coord
from spacepy.time import Ticktock
import glob

# %% --- Backend Setup ---
print("Using pyvistaqt with BackgroundPlotter.")

# %% --- Workflow and Control Flags ---
INTERACTIVE_MODE = True
PLOT_AURORA = False
CAMERA_POS_FILE = "camera_positions.json"
# ---------------------------------

# %% --- Constants, Paths, Styles --- (Keep as before)
# ... (R_E, Re_km, paths, colors, window sizes, etc.) ...
R_E = 1.0
Re_km = 6371.2
SWARM_RADIUS = 6816838.5 * 1e-3
SWARM_ALTITUDE = SWARM_RADIUS - Re_km
EARTH_TEXTURE_PATH = "eo_base_2020_clean_3600x1800.png"
TRACE_FILE_DIR = r"G:\master\pyaw\scripts\results\aw_cases\archive\trace_points\pkl\12728"
TRACE_FILE_PATHS = glob.glob(os.path.join(TRACE_FILE_DIR, "*.pkl"))
TRACE_FILE_NAMES = [os.path.basename(path) for path in TRACE_FILE_PATHS]
SATELLITE_DIR = r"G:\master\pyaw\scripts\results\aw_cases\archive\orbits\12728"
SATELLITE_PATHS = glob.glob(os.path.join(SATELLITE_DIR, "*.pkl"))
SATELLITE_MARKER_INDICES = [[]]
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
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 800
IMAGE_SCALE = 4

# %% --- Earth Texture Loading --- (Keep as before)
# ... texture loading ...
texture = None
if not os.path.exists(EARTH_TEXTURE_PATH): print(f"Warning: Texture not found: {EARTH_TEXTURE_PATH}")
else:
    try:
        texture = pv.read_texture(EARTH_TEXTURE_PATH)
        print(f"Loaded texture: {EARTH_TEXTURE_PATH}")
    except Exception as e: print(f"Error loading texture: {e}")

# %% --- Data Loading and Helper Functions --- (Keep as before)
# geosph2geocar, find_contiguous_segments, extract_timestamps
# %% --- Load Trace and Satellite Data --- (Keep as before)
# ... (Loading loops for trace and satellite data) ...
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

def extract_timestamps(filename):
    match = re.search(r"_case_(\d{8}T\d{6})_(\d{8}T\d{6})\.pkl$", filename)
    if not match: raise ValueError(f"Filename format incorrect: {filename}")
    return match.group(2) if filename.startswith("end_") else match.group(1)

magnetic_field_lines_data = []
print(f"Loading {len(TRACE_FILE_PATHS)} trace lines...")
for (i, trace_path), trace_fn in zip(enumerate(TRACE_FILE_PATHS), TRACE_FILE_NAMES):
    try:
        df_trace = pd.read_pickle(trace_path)
        df_trace_np = df_trace.values
        df_trace_np_alt_re = df_trace_np.copy()
        df_trace_np_alt_re[:, 0] = (df_trace_np_alt_re[:, 0] + Re_km) / Re_km
        time_trace = extract_timestamps(trace_fn)
        times_trace = np.array([str(time_trace)] * len(df_trace_np_alt_re))
        geo_car_trace = geosph2geocar(times_trace, df_trace_np_alt_re[:, 0], df_trace_np_alt_re[:, 1], df_trace_np_alt_re[:, 2])
        magnetic_field_lines_data.append(geo_car_trace.data)
        print(f" Trace line {i+1} loaded ({geo_car_trace.data.shape[0]} pts).")
    except Exception as e: print(f"Error loading trace {trace_path}: {e}")
if not magnetic_field_lines_data: print("Warning: No trace data loaded.")

satellite_tracks_data = []
satellite_markers_data = SATELLITE_MARKER_INDICES
print(f"Loading {len(SATELLITE_PATHS)} satellite track(s)...")
if len(SATELLITE_PATHS) != len(satellite_markers_data): print("Warning: Sat path/marker mismatch!")
for i, sat_path in enumerate(SATELLITE_PATHS):
    try:
        df_sa = pd.read_pickle(sat_path)
        lats_sa = df_sa["Latitude"].values
        lons_sa = df_sa["Longitude"].values
        alts_sa = (np.full(len(lats_sa), SWARM_ALTITUDE) + Re_km) / Re_km
        times_sa = df_sa.index.values
        times_sa_str = np.array([str(t) for t in times_sa])
        geo_car_sa = geosph2geocar(times_sa_str, alts_sa, lats_sa, lons_sa)
        satellite_tracks_data.append(geo_car_sa.data)
        print(f" Track {i+1} loaded ({geo_car_sa.data.shape[0]} pts).")
    except Exception as e: print(f"Error loading satellite {sat_path}: {e}")
if not satellite_tracks_data: print("Warning: No satellite data loaded.")


# %% --- Plotting Helper Functions ---
# Keep create_aurora_texture, add_aurora, add_earth_features
# Keep add_fieldlines_and_points, add_satellite_track
# Keep apply_final_settings
# --- (Definitions of these 6 functions are required here, unchanged) ---
def create_aurora_texture(color_hex, opacity, height=64):
    try:
        color_obj = pv.Color(color_hex, default_opacity=opacity)
        rgb_tuple = color_obj.int_rgb
        if rgb_tuple is None: raise ValueError("Invalid color")
        color_r, color_g, color_b = rgb_tuple
        max_alpha = int(opacity * 255)
        gradient = np.concatenate([np.linspace(0, max_alpha, height // 2), np.linspace(max_alpha, 0, height - height // 2)]).astype(np.uint8)
        tex_arr = np.zeros((height, 1, 4), dtype=np.uint8)
        tex_arr[:, 0, :3] = [color_r, color_g, color_b]
        tex_arr[:, 0, 3] = gradient
        return pv.Texture(tex_arr)
    except Exception as e: print(f"Error creating aurora texture: {e}"); return None

def add_aurora(plotter_instance, lat=70, thickness=0.15, color=AURORA_COLOR, opacity=AURORA_OPACITY, texture_height=128):
    try:
        radius = R_E * np.cos(np.radians(lat))
        z_offset = R_E * np.sin(np.radians(lat))
        aurora_tex = create_aurora_texture(color, opacity, height=texture_height)
        if not aurora_tex: return False
        for pole, direction_z, name in [("North", 1, "north"), ("South", -1, "south")]:
            center = (0, 0, direction_z * z_offset)
            aurora_cyl = pv.Cylinder(center=center, direction=(0, 0, direction_z), radius=radius, height=thickness, capping=False, resolution=100)
            aurora_cyl.texture_map_to_plane(inplace=True, use_bounds=True)
            plotter_instance.add_mesh(aurora_cyl, texture=aurora_tex, opacity=1.0, show_edges=False, smooth_shading=True, rgb=False, name=f"{name}_aurora", use_transparency=True)
        return True
    except Exception as e: print(f"ERROR adding aurora: {e}"); return False

def add_earth_features(plotter_instance):
    earth = pv.Sphere(radius=R_E, theta_resolution=120, phi_resolution=120)
    try:
        earth.texture_map_to_sphere(inplace=True)
        if texture: plotter_instance.add_mesh(earth, texture=texture, smooth_shading=EARTH_SMOOTH_SHADING, specular=EARTH_SPECULAR, specular_power=EARTH_SPECULAR_POWER, show_edges=False, rgb=False, name="earth")
        else: plotter_instance.add_mesh(earth, color=EARTH_COLOR, smooth_shading=EARTH_SMOOTH_SHADING, specular=EARTH_SPECULAR, specular_power=EARTH_SPECULAR_POWER, show_edges=False, name="earth")
    except Exception as e:
        print(f"Warning: Earth processing failed: {e}")
        if not plotter_instance.renderer.actors.get("earth"): plotter_instance.add_mesh(earth, color=EARTH_COLOR, name="earth_fallback")
    try:
        north_pole = pv.Sphere(radius=POLE_MARKER_RADIUS, center=(0, 0, R_E))
        south_pole = pv.Sphere(radius=POLE_MARKER_RADIUS, center=(0, 0, -R_E))
        plotter_instance.add_mesh(north_pole, color=POLE_MARKER_COLOR, name="north_pole_marker")
        plotter_instance.add_mesh(south_pole, color=POLE_MARKER_COLOR, name="south_pole_marker")
        plotter_instance.add_point_labels([(0, 0, R_E * 1.1)], ["N"], point_size=10, font_size=12, text_color=TEXT_COLOR, shape=None, name="north_label")
        plotter_instance.add_point_labels([(0, 0, -R_E * 1.1)], ["S"], point_size=10, font_size=12, text_color=TEXT_COLOR, shape=None, name="south_label")
    except Exception as e: print(f"Warning: Failed adding poles/labels: {e}")
    return plotter_instance

def add_fieldlines_and_points(plotter_instance, lines_data, max_radius):
    all_endpoints = []
    line_count = 0
    for line_data in lines_data:
        if line_data is None or line_data.shape[0] < 2: continue
        r = np.linalg.norm(line_data, axis=1)
        mask = r <= max_radius
        for start, end in find_contiguous_segments(np.asarray(mask)):
            segment = line_data[start:end]
            if segment.shape[0] >= 2:
                line_count += 1
                lines = np.c_[np.full(len(segment)-1, 2), np.arange(len(segment)-1), np.arange(1, len(segment))]
                line_poly = pv.PolyData(segment, lines=lines)
                plotter_instance.add_mesh(line_poly, color=FIELD_LINE_COLOR, opacity=FIELD_LINE_OPACITY, line_width=FIELD_LINE_WIDTH)
                all_endpoints.extend([segment[0], segment[-1]])
            elif segment.shape[0] == 1: all_endpoints.append(segment[0])
    if all_endpoints:
        coords = np.array(all_endpoints)
        plotter_instance.add_points(coords, color=POINT_COLOR, point_size=POINT_SIZE, render_points_as_spheres=True, opacity=POINT_ALPHA, label=f"Field Endpoints ({line_count})")
    return plotter_instance

def add_satellite_track(plotter_instance, tracks_list, marker_indices_list, max_radius):
    safe_markers = marker_indices_list + [[]] * (len(tracks_list) - len(marker_indices_list))
    legend_added = False
    for i, (track_data, markers) in enumerate(zip(tracks_list, safe_markers)):
        if track_data is None or track_data.shape[0] < 2: continue
        r_sat = np.linalg.norm(track_data, axis=1)
        mask_sat = r_sat <= max_radius
        for start, end in find_contiguous_segments(np.asarray(mask_sat)):
            segment = track_data[start:end]
            if segment.shape[0] >= 2:
                lines = np.c_[np.full(len(segment)-1, 2), np.arange(len(segment)-1), np.arange(1, len(segment))]
                track_poly = pv.PolyData(segment, lines=lines)
                plotter_instance.add_mesh(track_poly, color=SAT_TRACK_COLOR, line_width=SAT_TRACK_WIDTH, opacity=SAT_TRACK_OPACITY, label="Satellite Track" if not legend_added else None)
                legend_added = True
        valid_pts = [track_data[idx] for idx in markers if 0 <= idx < len(track_data) and np.linalg.norm(track_data[idx]) <= max_radius]
        if valid_pts:
            plotter_instance.add_points(np.array(valid_pts), color=SAT_POS_COLOR, point_size=SAT_POS_SIZE, render_points_as_spheres=True, opacity=POINT_ALPHA, label=f"Sat Markers (T{i+1})" if i==0 else None)
    return plotter_instance

def apply_final_settings(plotter_instance, max_radius):
    print("Applying final settings...")
    try:
        if INTERACTIVE_MODE or 'final' not in camera_positions:
             plotter_instance.camera.zoom(0.7)
             print("Set default zoom.")
    except Exception as e: print(f"Warning: Could not set default camera zoom - {e}.")
    plotter_instance.add_axes(interactive=True, line_width=2, color=TEXT_COLOR)
    title = f"Alfven Wave Magnetic Conjugacy (View range: {max_radius} Rₑ)" # Use Unicode
    plotter_instance.add_text(title, position="upper_edge", color=TEXT_COLOR, font_size=12, name="final_title")
    return plotter_instance


# %% --- Define a simple class to emit signals ---
# This is needed because the callback function needs to emit a signal
class SignalEmitter(QtCore.QObject):
    proceed_signal = QtCore.Signal()

# %% --- Step Interaction / Screenshot Handler with Button and Signal ---

def handle_step_interaction_or_screenshot(plotter, step_description, camera_pos_dict, step_key, is_final=False):
    """
    Handles pausing/saving view in interactive mode (using a button + signal),
    or setting view/screenshotting in batch mode.
    """
    step_text_actor = None
    if not is_final:
        step_text_actor = plotter.add_text(
            f"Step: {step_description}", position="lower_edge", font_size=10, color=TEXT_COLOR, name="step_text",
        )

    plotter.render()

    if INTERACTIVE_MODE:
        prompt = "\n--- Paused at Step: {} ---".format(step_description)
        if is_final:
            prompt += "\n>>> This is the FINAL view. Adjust as needed. Close window to EXIT script. <<<"
            print(prompt)
            # Save camera before final app.exec_
            try:
                raw_camera_pos = plotter.camera_position
                camera_pos_list = [list(raw_camera_pos[0]), list(raw_camera_pos[1]), list(raw_camera_pos[2])]
                camera_pos_dict[step_key] = camera_pos_list
                print(f"Camera position for FINAL step '{step_key}' saved.")
            except Exception as e: print(f"Warning: Could not get/save FINAL camera position for '{step_key}': {e}")

        else: # Intermediate step with button
            prompt += "\n>>> INTERACTIVE view. Adjust view. Click the 'Next Step' button in the plot window to continue AND SAVE VIEW. <<<"
            print(prompt)

            local_app = QtWidgets.QApplication.instance()
            if not local_app:
                print("ERROR: QApplication not found! Cannot use button pause. Falling back to input().")
                input("Press Enter in console to continue (window might be unresponsive)...")
            else:
                event_loop = QtCore.QEventLoop()
                button_actor = None
                emitter = SignalEmitter() # Create signal emitter instance

                # --- Callback emits signal ---
                def proceed_callback():
                    print("'Next Step' button clicked.")
                    # Remove button and text before emitting signal
                    if button_actor:
                         try: plotter.remove_actor(button_actor, render=False)
                         except: print("Warning: could not remove button actor.")
                    if step_text_actor:
                         try: plotter.remove_actor(step_text_actor, render=False)
                         except: print("Warning: could not remove step text actor.")
                    plotter.render() # Update display
                    emitter.proceed_signal.emit() # Emit the signal

                # --- Connect signal to event loop quit ---
                emitter.proceed_signal.connect(event_loop.quit)
                # -----------------------------------------

                # --- Add the button widget ---
                try:
                    print("Adding 'Next Step' button...")
                    button_actor = plotter.add_button_widget(proceed_callback, value=False, pass_widget=False, title="Next Step >") # pass_widget=False might be needed
                    print("Button added.")
                    plotter.render()
                except Exception as e:
                     print(f"ERROR adding button widget: {e}. Falling back to input().")
                     emitter.proceed_signal.disconnect(event_loop.quit) # Disconnect if fallback
                     input("Press Enter in console to continue...")
                     if step_text_actor: plotter.remove_actor(step_text_actor, render=True)

                else: # Only run event loop if button was added
                    print("Starting event loop, waiting for button click...")
                    event_loop.exec_() # Start loop - waits for proceed_signal to trigger quit()
                    print("Event loop finished.")

            # --- Save camera position AFTER event loop finishes ---
            try:
                raw_camera_pos = plotter.camera_position
                camera_pos_list = [list(raw_camera_pos[0]), list(raw_camera_pos[1]), list(raw_camera_pos[2])]
                camera_pos_dict[step_key] = camera_pos_list
                print(f"Camera position for '{step_key}' saved.")
            except Exception as e:
                print(f"Warning: Could not get/save camera position for '{step_key}': {e}")

    else: # Batch screenshot mode
        # ... (batch mode logic remains the same) ...
        filename = f"step_{step_key}_screenshot.png"
        print(f"\n--- Generating Screenshot: {step_description} ({filename}) ---")
        if step_key in camera_pos_dict:
            try: plotter.camera_position = camera_pos_dict[step_key]; print(f"Applied camera position.")
            except Exception as e: print(f"Warning: Could not apply camera position: {e}")
        else: print(f"Warning: No saved camera position.")
        try:
            if not is_final and step_text_actor: plotter.render() # Render if text was added
            plotter.screenshot(filename)
            print(f"Screenshot saved: {filename}")
            if not is_final and step_text_actor: plotter.remove_actor(step_text_actor, render=False)
        except Exception as e: print(f"ERROR taking screenshot: {e}")


# %% --- Main Execution ---

print(f"Starting PyVista plotting in {'INTERACTIVE' if INTERACTIVE_MODE else 'BATCH'} mode...")
pv.set_plot_theme("paraview")

# --- Initialize Qt Application ---
print("Initializing Qt Application...")
app = QtWidgets.QApplication.instance()
if app is None:
    print("Creating new QApplication.")
    app = QtWidgets.QApplication(sys.argv)
else:
    print("Using existing QApplication.")
# ---------------------------------

# Load camera positions
camera_positions = {}
if not INTERACTIVE_MODE or os.path.exists(CAMERA_POS_FILE):
    # ... (loading logic) ...
    if os.path.exists(CAMERA_POS_FILE):
        try:
            with open(CAMERA_POS_FILE, 'r') as f:
                loaded_pos = json.load(f)
                for key, pos_list in loaded_pos.items():
                     if isinstance(pos_list, list) and len(pos_list) == 3: camera_positions[key] = [tuple(pos_list[0]), tuple(pos_list[1]), tuple(pos_list[2])]
                     else: camera_positions[key] = pos_list
                print(f"Loaded {len(camera_positions)} positions from {CAMERA_POS_FILE}")
        except Exception as e: print(f"Warning: Load positions failed: {e}")
    elif not INTERACTIVE_MODE: print(f"Warning: {CAMERA_POS_FILE} not found. Using defaults.")


# --- Create the SINGLE BackgroundPlotter ---
effective_image_scale = IMAGE_SCALE if not INTERACTIVE_MODE else 1
plotter = pvqt.BackgroundPlotter(
    window_size=(WINDOW_WIDTH, WINDOW_HEIGHT),
    image_scale=effective_image_scale,
    show=INTERACTIVE_MODE,
    title="Magnetosphere Visualization"
)
print(f"Created BackgroundPlotter (image_scale={effective_image_scale}, show={INTERACTIVE_MODE})")
plotter.enable_depth_peeling()
print("Depth peeling enabled.")
plotter.set_background(FIG_BG_COLOR)

# --- Add Base Features (Once) ---
print("\nAdding base features...")
add_earth_features(plotter)
if PLOT_AURORA: add_aurora(plotter)
print("Base features added.")
# --------------------------------

# --- Define and Process Steps Cumulatively ---
steps_definition = [
    {'key': 'step1', 'desc': "1_Base_Features" + ("_Aurora" if PLOT_AURORA else ""), 'add_fieldlines': False, 'add_satellite': False, 'is_final': False},
    {'key': 'step2', 'desc': '2_Add_Field_Lines', 'add_fieldlines': True, 'add_satellite': False, 'is_final': False},
    {'key': 'step3', 'desc': '3_Add_Satellite', 'add_fieldlines': False, 'add_satellite': True, 'is_final': False},
    {'key': 'final', 'desc': '4_Final_Plot', 'add_fieldlines': False, 'add_satellite': False, 'is_final': True}
]

fieldlines_added = False
satellite_added = False

for step_info in steps_definition:
    print(f"\n===== Processing Step: {step_info['desc']} =====")

    # Add elements specific to this step IF NOT ALREADY ADDED
    if step_info['add_fieldlines'] and not fieldlines_added:
        print("Adding field lines...")
        add_fieldlines_and_points(plotter, magnetic_field_lines_data, MAX_PLOT_RADIUS)
        fieldlines_added = True
    if step_info['add_satellite'] and not satellite_added:
        print("Adding satellite track...")
        add_satellite_track(plotter, satellite_tracks_data, satellite_markers_data, MAX_PLOT_RADIUS)
        satellite_added = True

    # Apply final settings if this is the last step
    if step_info['is_final']:
        apply_final_settings(plotter, MAX_PLOT_RADIUS)
        plotter.remove_actor("step_text", render=False) # Final removal before showing

    # Handle interaction or screenshot for the current cumulative view
    handle_step_interaction_or_screenshot(
        plotter,
        step_description=step_info['desc'],
        camera_pos_dict=camera_positions,
        step_key=step_info['key'],
        is_final=step_info['is_final']
    )

# --- Finalization ---
# Save Camera Positions if interactive mode was run
if INTERACTIVE_MODE and camera_positions:
    try:
        with open(CAMERA_POS_FILE, 'w') as f:
            json.dump(camera_positions, f, indent=4)
        print(f"\nSaved {len(camera_positions)} positions to {CAMERA_POS_FILE}")
    except Exception as e: print(f"Warning: Could not save positions: {e}")

# --- Keep window open in interactive mode (only needed if the loop didn't handle the final step pause) ---
if INTERACTIVE_MODE:
    # Check if the plotter window might still exist (e.g., if final step saving happened before app.exec_)
    window_potentially_open = False
    try:
        if plotter and plotter.iren and plotter.iren.interactor and plotter.iren.interactor.GetRenderWindow().IsDrawable():
            window_potentially_open = True
    except:
        pass # Ignore errors if plotter is already gone

    if window_potentially_open:
        print("\nMain script finished. Starting final Qt event loop.")
        print("Close the PyVista window to exit.")
        app.exec_() # Start the Qt event loop for the final persistent window
    else:
        print("\nMain script finished (window likely closed during final step).")


print("\n绘图完成.")