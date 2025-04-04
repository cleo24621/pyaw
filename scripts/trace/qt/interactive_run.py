# interactive_run.py
import sys
import json
import pyvista as pv
import plot_config as cfg # Import shared config/functions/data
from qtpy import QtCore, QtWidgets # Needed for button interaction

print("--- Running INTERACTIVE View Setter ---")

# Ensure Qt App exists (created in plot_config)
if not cfg.QT_AVAILABLE or cfg.app is None:
    print("ERROR: Qt Application not initialized in config. Exiting.")
    sys.exit(1)

# --- Define Signal Emitter ---
class SignalEmitter(QtCore.QObject):
    proceed_signal = QtCore.Signal()

# --- Define Step Handler with Button ---
def handle_interactive_step(plotter, step_description, camera_pos_dict, step_key, is_final=False):
    step_text_actor = None
    if not is_final:
        step_text_actor = plotter.add_text(
            f"Step: {step_description}", position="lower_edge", font_size=10, color=cfg.TEXT_COLOR, name="step_text",
        )
    plotter.render()

    prompt = f"\n--- Paused at Step: {step_description} ---"
    if is_final:
        prompt += "\n>>> FINAL view. Adjust as needed. Close window to EXIT & SAVE FINAL VIEW. <<<"
        print(prompt)
        # Final step relies on main app.exec_(), save happens before that
    else:
        prompt += "\n>>> Adjust view. Click 'Next Step >' button to continue AND SAVE VIEW. <<<"
        print(prompt)

        event_loop = QtCore.QEventLoop()
        button_actor = None
        emitter = SignalEmitter()

        def proceed_callback():
            print("'Next Step' button clicked.")
            if button_actor:
                 try: plotter.remove_actor(button_actor, render=False)
                 except: pass # Ignore error if already removed
            if step_text_actor:
                 try: plotter.remove_actor(step_text_actor, render=False)
                 except: pass
            plotter.render()
            emitter.proceed_signal.emit()

        emitter.proceed_signal.connect(event_loop.quit)

        try:
            button_actor = plotter.add_button_widget(proceed_callback, value=False, pass_widget=False, title="Next Step >")
            plotter.render() # Show button
            print("Starting event loop, waiting for button click...")
            event_loop.exec_() # PAUSE here until button clicked
            print("Event loop finished.")
        except Exception as e:
             print(f"ERROR adding button or running loop: {e}. Script might hang.")
             # No easy fallback here without potentially breaking things
             return # Exit function if button fails critically

    # --- Save camera position AFTER pause/button click (or before final exit) ---
    try:
        raw_camera_pos = plotter.camera_position
        camera_pos_list = [list(raw_camera_pos[0]), list(raw_camera_pos[1]), list(raw_camera_pos[2])]
        camera_pos_dict[step_key] = camera_pos_list
        print(f"Camera position for '{step_key}' saved.")
    except Exception as e:
        print(f"Warning: Could not get/save camera position for '{step_key}': {e}")

# --- Main Interactive Logic ---
camera_positions = {} # Start with empty positions

# Create the single BackgroundPlotter for the interactive session
print("Creating persistent BackgroundPlotter for interactive session...")
plotter = cfg.pvqt.BackgroundPlotter(
    window_size=(cfg.BASE_WINDOW_WIDTH, cfg.BASE_WINDOW_HEIGHT),
    image_scale=1, # No scaling needed for interactive view
    show=True,     # Show the window immediately
    title="Interactive View Setup"
)
plotter.enable_depth_peeling()
plotter.set_background(cfg.FIG_BG_COLOR)

# --- Step 1: Base ---
print("\n===== Setting up Step 1: Base Features =====")
cfg.add_earth_features(plotter)
# if cfg.PLOT_AURORA: cfg.add_aurora(plotter) # Add if needed
handle_interactive_step(plotter, "1_Base_Features", camera_positions, 'step1')

# --- Step 2: Add Field Lines ---
print("\n===== Setting up Step 2: Field Lines =====")
cfg.add_fieldlines_and_points(plotter, cfg.magnetic_field_lines_data, cfg.MAX_PLOT_RADIUS)
handle_interactive_step(plotter, "2_Add_Field_Lines", camera_positions, 'step2')

# --- Step 3: Add Satellite ---
print("\n===== Setting up Step 3: Satellite =====")
cfg.add_satellite_track(plotter, cfg.satellite_tracks_data, cfg.satellite_markers_data, cfg.MAX_PLOT_RADIUS)
handle_interactive_step(plotter, "3_Add_Satellite", camera_positions, 'step3')

# --- Step 4: Final Setup (Apply Settings) ---
print("\n===== Setting up Step 4: Final Plot =====")
cfg.apply_final_settings(plotter, cfg.MAX_PLOT_RADIUS)
plotter.remove_actor("step_text", render=False) # Remove any previous step text
# Handle the final pause differently - save position now, then start main loop
handle_interactive_step(plotter, "4_Final_Plot", camera_positions, 'final', is_final=True)

# --- Save the collected camera positions ---
if camera_positions:
    try:
        with open(cfg.CAMERA_POS_FILE, 'w') as f:
            json.dump(camera_positions, f, indent=4)
        print(f"\nSaved {len(camera_positions)} camera positions to {cfg.CAMERA_POS_FILE}")
    except Exception as e:
        print(f"Warning: Could not save camera positions: {e}")

# --- Start main Qt loop to keep final window open ---
print("\nSetup complete. Starting final Qt event loop.")
print("Close the PyVista window to exit.")
cfg.app.exec_()

print("\nInteractive run finished.")