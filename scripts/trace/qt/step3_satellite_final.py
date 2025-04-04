# step3_fieldlines.py
import sys
import pyvista as pv
import plot_config as cfg # Import shared config/functions/data
import json # Need json for saving directly here

# --- Script Specific Settings ---
STEP_KEY = 'step3'
CAMERA_FILE_THIS_STEP = f"camera_pos_{STEP_KEY}.json"
OUTPUT_FILENAME = f"step_{STEP_KEY}_field_lines.png"
PLOT_AURORA_THIS_STEP = False
# --- CHOOSE MODE FOR THIS SCRIPT RUN ---
# Set this specifically for how you want to run THIS script right now
INTERACTIVE_MODE = True # Set True to adjust view, False to generate screenshot
# ------------------------------------

# --- Check Qt ---
if not cfg.QT_AVAILABLE: sys.exit("ERROR: Qt / pyvistaqt not available.")
if cfg.app is None and INTERACTIVE_MODE: sys.exit("ERROR: Qt App not initialized.")


print(f"\n--- Running Step 2: Field Lines ({'INTERACTIVE' if INTERACTIVE_MODE else 'BATCH'}) ---")

# --- Load Specific Camera Position (for BATCH mode) ---
camera_position = None
if not INTERACTIVE_MODE: # Only load if in batch mode
    camera_position = cfg.load_camera_position(CAMERA_FILE_THIS_STEP)

# --- Create BackgroundPlotter ---
# NOTE: For interactive mode, image_scale=1. For batch, use cfg.IMAGE_SCALE.
plotter_scale = 1 if INTERACTIVE_MODE else cfg.IMAGE_SCALE
plotter = cfg.pvqt.BackgroundPlotter(
    window_size=(cfg.BASE_WINDOW_WIDTH, cfg.BASE_WINDOW_HEIGHT),
    image_scale=plotter_scale,
    show=INTERACTIVE_MODE, # Show window only if interactive
    title="Step 2: Field Lines (Press 'C' to Save View)" # Updated title
)
plotter.enable_depth_peeling(); plotter.set_background(cfg.FIG_BG_COLOR)

# --- Add Elements ---
print("Adding Earth features...")
cfg.add_earth_features(plotter)
# if PLOT_AURORA_THIS_STEP: cfg.add_aurora(plotter)
print("Adding field lines...")
cfg.add_fieldlines_and_points(plotter, cfg.magnetic_field_lines_data, cfg.MAX_PLOT_RADIUS)
plotter.render()
print("Adding Satellite Tracks")
cfg.add_satellite_track(plotter, cfg.satellite_tracks_data,cfg.satellite_markers_data, cfg.MAX_PLOT_RADIUS)
cfg.apply_final_settings(plotter,cfg.MAX_PLOT_RADIUS)

# --- Define Camera Saving Callback ---
# This function will be triggered by a key press in interactive mode
def save_current_view():
    try:
        raw_cam_pos = plotter.camera_position
        pos_list = [list(raw_cam_pos[0]), list(raw_cam_pos[1]), list(raw_cam_pos[2])]
        cfg.save_camera_position(CAMERA_FILE_THIS_STEP, pos_list) # Use config save function
        print(f"\n*** Camera position saved to {CAMERA_FILE_THIS_STEP} ***")
    except Exception as e:
        print(f"\n*** ERROR saving camera position: {e} ***")

# --- Handle Interaction or Screenshot ---
if INTERACTIVE_MODE:
    # Add key press event 'c' to trigger saving the camera view
    plotter.add_key_event('c', save_current_view)
    print("\n>>> INTERACTIVE MODE: Adjust view in the PyVista window.")
    print(">>> Press 'C' key (ensure window has focus) to SAVE the current view.")
    print(">>> Close the window MANUALLY when finished.")
    # NO MORE LOCAL EVENT LOOP OR BUTTONS HERE
    # The script will continue to the end, but the window stays open via cfg.app

else: # BATCH MODE
    print(f"\n--- Generating Screenshot: {OUTPUT_FILENAME} ---")
    if camera_position:
        try: plotter.camera_position = camera_position; print(f"Applied camera.")
        except Exception as e: print(f"Warning: Could not apply camera: {e}")
    else: print(f"Warning: No camera position loaded.")
    try:
        plotter.render(); plotter.screenshot(OUTPUT_FILENAME)
        print(f"Screenshot saved: {OUTPUT_FILENAME}")
    except Exception as e: print(f"ERROR taking screenshot: {e}")
    # Close plotter immediately in batch mode as we are done
    print("Closing plotter (batch mode).")
    plotter.close()

# --- Script End ---
print(f"--- Finished script execution for Step 2 ---")

# --- Keep Qt app alive ONLY in interactive mode ---
# This allows the window to stay open after the script finishes plotting.
if INTERACTIVE_MODE and cfg.app:
    print("\nStarting Qt event loop (interactive mode). Close window to exit.")
    cfg.app.exec_()
    print("Qt event loop finished.")
elif not INTERACTIVE_MODE:
     # Optional: process events briefly for cleanup in batch? Usually not needed.
     # if cfg.app: cfg.app.processEvents()
     pass