# generate_final.py
import pyvista as pv
import plot_config as cfg # Import the configuration module

# --- Script Settings ---
STEP_KEY = 'final'
OUTPUT_FILENAME = f'step_{STEP_KEY}_final_plot.png'
PLOT_AURORA_THIS_STEP = False # Control aurora specific to this script if needed

# --- Check if Qt is available ---
if not cfg.QT_AVAILABLE:
    print("ERROR: Qt / pyvistaqt not available. Cannot use BackgroundPlotter.")
    exit()

print(f"\n--- Generating Plot: {OUTPUT_FILENAME} (using BackgroundPlotter) ---")

# --- Create BackgroundPlotter ---
plotter = cfg.pvqt.BackgroundPlotter(
    window_size=(cfg.BASE_WINDOW_WIDTH, cfg.BASE_WINDOW_HEIGHT),
    image_scale=cfg.IMAGE_SCALE,
    show=False,
    title=f"Final Plot" # Set initial title, will be overwritten by apply_final_settings
)
print(f"Created BackgroundPlotter (image_scale={cfg.IMAGE_SCALE})")

plotter.enable_depth_peeling()
plotter.set_background(cfg.FIG_BG_COLOR)

# --- Add Elements for this step ---
print("Adding Earth features...")
cfg.add_earth_features(plotter)
# if PLOT_AURORA_THIS_STEP: cfg.add_aurora(plotter)

print("Adding field lines...")
cfg.add_fieldlines_and_points(plotter, cfg.magnetic_field_lines_data, cfg.MAX_PLOT_RADIUS)

print("Adding satellite track...")
cfg.add_satellite_track(plotter, cfg.satellite_tracks_data, cfg.satellite_markers_data, cfg.MAX_PLOT_RADIUS)

# --- Apply Final Settings FIRST ---
cfg.apply_final_settings(plotter, cfg.MAX_PLOT_RADIUS)

# --- Apply Camera Position AFTER final settings ---
if STEP_KEY in cfg.camera_positions:
    try:
        plotter.camera_position = cfg.camera_positions[STEP_KEY]
        print(f"Applied final camera position for '{STEP_KEY}'.")
    except Exception as e:
        print(f"Warning: Could not apply final camera position '{STEP_KEY}': {e}")
else:
    print(f"Warning: No saved position for '{STEP_KEY}', using default view set by apply_final_settings.")

# --- Take Screenshot ---
try:
    plotter.render() # Ensure rendering is complete after camera set
    plotter.screenshot(OUTPUT_FILENAME)
    print(f"Screenshot saved: {OUTPUT_FILENAME}")
except Exception as e:
    print(f"ERROR taking screenshot: {e}")

# --- Clean up ---
plotter.close()
print(f"--- Finished Plot: {OUTPUT_FILENAME} ---")