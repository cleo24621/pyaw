import plot_config as cfg
from pyvistaqt import BackgroundPlotter


print("\n--- Initializing Final Plot ---")

# Determine the main title
plot_title = f"Alfven Wave Magnetic Conjugate (View Range: {cfg.MAX_PLOT_RADIUS} Re)"

# --- CHANGE HERE: Use BackgroundPlotter directly ---
plotter = BackgroundPlotter(
    window_size=(cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT),
    title=plot_title, # Set title on creation
    auto_update=True
)
# --------------------------------------------------

# Set background color and enable depth peeling
plotter.set_background(cfg.FIG_BG_COLOR)
plotter.enable_depth_peeling()

# Add Earth and optional Aurora
print("Adding Earth features...")
cfg.add_earth_features(plotter)
if cfg.PLOT_AURORA:
    print("Adding Aurora...")
    cfg.add_aurora(plotter)

# Add all field line segments and their endpoint markers
print(f"Adding {len(cfg.magnetic_field_lines_segments)} field line segments...")
for idx, data in enumerate(cfg.magnetic_field_lines_segments):
    cfg.add_single_fieldline_segment(plotter, data, idx)

# Add all full satellite tracks
print(f"Adding {len(cfg.all_satellite_tracks_data)} satellite tracks...")
for idx, track_data in enumerate(cfg.all_satellite_tracks_data):
    cfg.add_full_satellite_track(plotter, track_data, idx)

# Add all satellite markers
print(f"Adding {len(cfg.all_satellite_markers_data)} satellite markers...")
if cfg.all_satellite_markers_data:
    cfg.add_satellite_markers(plotter, cfg.all_satellite_markers_data)

# Add final axes
print("Adding axes...")
plotter.add_axes(interactive=True, line_width=2, color=cfg.TEXT_COLOR)

# --- REMOVED Camera Loading Block ---
print("Using default camera view.")

print("\n--- Plotting Complete ---")
print("The plot window should now be open and interactive.")
print("Close the plot window manually when finished.")

# --- ADD THIS LINE AT THE VERY END ---
# Start the Qt event loop to keep the window open and interactive
# This call is blocking and will only return when the window is closed.
if hasattr(plotter, 'app') and plotter.app is not None:
    print("Starting Qt event loop to keep window open...")
    plotter.app.exec_()
else:
    # Fallback if app isn't directly accessible or using a different setup
    # This keeps the main thread alive but requires Ctrl+C to exit script
    print("Qt app not found directly on plotter. Using input() to keep script alive.")
    print("Press Enter in the console to exit the script after closing the window.")
    input("...") # Or use a time.sleep loop: while True: time.sleep(1)

print("Plot window closed. Exiting script.")
# --- END OF ADDITION ---

# --- END OF CORRECTED FILE ---