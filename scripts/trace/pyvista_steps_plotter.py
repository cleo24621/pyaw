import pathlib

import plot_config as cfg

import os
import json

import pyvista as pv


#  --- Workflow and Control Flags ---
INTERACTIVE_MODE = True  # 设置为 True 进行交互式查看，False 进行截图
CAMERA_POS_FILE = f"camera_positions/{cfg.ORBIT_NUM}/camera_positions_incremental.json"  # 相机位置文件
SCREENSHOT_DIR = f"screenshot/{cfg.ORBIT_NUM}"


def show_or_screenshot_step(
    plotter, step_description, camera_pos_dict, step_key, is_final_view=False
):
    """
    Handles showing plot or taking screenshot, manages camera positions.
    Sets a specific final title if is_final_view is True.
    """
    title_text = (
        f"Alfven Wave Magnetic Conjugate (View Range: {cfg.MAX_PLOT_RADIUS} Re)"
    )
    if is_final_view:
        plotter.add_text(
            title_text,
            position="upper_edge",
            color=cfg.TEXT_COLOR,
            font_size=12,
            name="final_title",
        )
    else:
        # Add step description text (always at the bottom)
        plotter.add_text(
            f"Step {step_description}",
            position="lower_edge",
            font_size=10,
            color=cfg.TEXT_COLOR,
            name="step_text",
        )
        plotter.add_text(
            title_text,
            position="upper_edge",
            color=cfg.TEXT_COLOR,
            font_size=12,
            name="main_title",
        )
    # --------------------------------------------------------

    # --- Core show/screenshot logic (remains mostly the same) ---
    if INTERACTIVE_MODE:
        print(f"\n--- Showing Interactive Step: {step_description} ---")
        print(">>> Adjust view. Close window to continue AND SAVE VIEW. <<<")
        plotter.add_axes(interactive=True, line_width=2, color=cfg.TEXT_COLOR)

        # Apply saved camera position if available for this step_key
        if step_key in camera_pos_dict:
            try:
                plotter.camera_position = camera_pos_dict[step_key]
                print(f"Applied saved camera position for '{step_key}'.")
            except Exception as e:
                print(f"Warning: Could not apply camera position for '{step_key}': {e}")

        plotter.show(title=step_description)  # Display window

        # Save camera position using the step_key
        try:
            raw_camera_pos = plotter.camera_position
            camera_pos_list = [list(pos) for pos in raw_camera_pos]
            camera_pos_dict[step_key] = camera_pos_list
            print(f"Camera position for '{step_key}' saved.")
        except Exception as e:
            print(f"Warning: Could not get/save camera position for '{step_key}': {e}")

    else:  # Batch mode
        if not os.path.exists(SCREENSHOT_DIR):
            os.makedirs(SCREENSHOT_DIR)  # 自动创建不存在的父目录
            print(f"目录已创建: {SCREENSHOT_DIR}")
        filename = f"{step_key}_screenshot.png".replace(" ", "_").replace(":", "_")
        print(f"\n--- Generating Screenshot: {step_description} ({filename}) ---")
        plotter.add_axes(interactive=False, line_width=2, color=cfg.TEXT_COLOR)

        # Apply camera position
        if step_key in camera_pos_dict:
            try:
                plotter.camera_position = camera_pos_dict[step_key]
                print(f"Applied saved camera position for '{step_key}'.")
            except Exception as e:
                print(f"Warning: Could not apply camera position for '{step_key}': {e}")
        else:
            print(
                f"Warning: No saved camera position for '{step_key}'. Using default view."
            )

        # Screenshot
        try:
            plotter.screenshot(
                os.path.join(SCREENSHOT_DIR,filename), scale=cfg.IMAGE_SCALE, transparent_background=False
            )
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"ERROR taking screenshot for '{step_key}': {e}")


#  --- Main Execution (Incremental Plotting) ---

print(
    f"Starting PyVista plotting in {'INTERACTIVE' if INTERACTIVE_MODE else 'BATCH'} mode (Incremental)..."
)
pv.set_plot_theme("paraview")  # 或者 "document"

# 加载相机位置
camera_positions = {}
if not INTERACTIVE_MODE or os.path.exists(CAMERA_POS_FILE):
    if os.path.exists(CAMERA_POS_FILE):
        try:
            with open(CAMERA_POS_FILE, "r") as f:
                loaded_pos = json.load(f)
                # JSON 加载的是 list of lists, PyVista 需要 list of tuples
                for key, pos_list in loaded_pos.items():
                    if (
                        isinstance(pos_list, list)
                        and len(pos_list) == 3
                        and all(isinstance(sublist, list) for sublist in pos_list)
                    ):
                        camera_positions[key] = [tuple(p) for p in pos_list]
                    else:
                        # 如果格式不符，尝试跳过或记录警告
                        print(
                            f"Warning: Invalid camera position format for key '{key}' in {CAMERA_POS_FILE}. Skipping."
                        )
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


# --- Step 0: Plot Earth ---
print("\n===== Step 0: Plotting Earth =====")
step_key_earth = "step_0_Earth"
plotter = pv.Plotter(
    window_size=[cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT],
    off_screen=not INTERACTIVE_MODE,
    image_scale=cfg.IMAGE_SCALE,
)
plotter.enable_depth_peeling()  # 启用以处理透明度
plotter.set_background(cfg.FIG_BG_COLOR)
cfg.add_earth_features(plotter)
if cfg.PLOT_AURORA:
    cfg.add_aurora(plotter)
show_or_screenshot_step(
    plotter,
    "0: Earth and Aurora" if cfg.PLOT_AURORA else "0: Earth",
    camera_positions,
    step_key_earth,
)
plotter.close()
print("===== Completed Step 0 =====")


# --- Step 1: Plot Field Line Segments Incrementally ---
print(
    f"\n===== Step 1: Plotting {len(cfg.magnetic_field_lines_segments)} Field Line Segments ====="
)
plotted_fieldline_segment_data = []  # 用于累积绘制的数据
for i, segment_data in enumerate(cfg.magnetic_field_lines_segments):
    step_key = f"step_1_fieldline_{i+1}"
    step_desc = (
        f"1: Added Field Line Segment {i+1}/{len(cfg.magnetic_field_lines_segments)}"
    )
    print(f"--- Processing: {step_desc} ---")

    plotted_fieldline_segment_data.append(segment_data)  # 添加当前段到累积列表

    plotter = pv.Plotter(
        window_size=[cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT],
        off_screen=not INTERACTIVE_MODE,
        image_scale=cfg.IMAGE_SCALE,
    )
    plotter.enable_depth_peeling()
    plotter.set_background(cfg.FIG_BG_COLOR)

    # 绘制基础地球和极光
    cfg.add_earth_features(plotter)
    if cfg.PLOT_AURORA:
        cfg.add_aurora(plotter)

    # 重新绘制所有已添加的磁力线段
    for idx, data in enumerate(plotted_fieldline_segment_data):
        cfg.add_single_fieldline_segment(plotter, data, idx)

    # 显示或截图
    show_or_screenshot_step(plotter, step_desc, camera_positions, step_key)
    plotter.close()

print("===== Completed Step 1 =====")


# --- Step 2: Plot Full Satellite Tracks Incrementally (REVISED Logic for Final Step) ---
print(
    f"\n===== Step 2: Plotting {len(cfg.all_satellite_tracks_data)} Full Satellite Tracks ====="
)
plotted_satellite_tracks = []
plotted_markers = []
num_tracks = len(cfg.all_satellite_tracks_data)

for i, current_track_data in enumerate(cfg.all_satellite_tracks_data):
    track_index = i
    is_last_track = i == num_tracks - 1  # Check if this is the last iteration

    # --- Determine Step Key, Description, and Final View Flag ---
    if is_last_track:
        step_key = "step_2_final_view"  # Use a distinct key for the final satellite step camera view
        step_desc = f"Final Plot: Added Track {track_index+1}/{num_tracks}"
        is_final_view_flag = True
    else:
        step_key = f"step_2_full_track_{track_index+1}"  # Incremental key
        step_desc = f"2: Added Full Satellite Track {track_index+1}/{num_tracks}"
        is_final_view_flag = False
    # ----------------------------------------------------------

    print(f"--- Processing: {step_desc} ---")

    # Add data for plotting in this step
    plotted_satellite_tracks.append(current_track_data)
    new_markers_for_this_track = [
        m for m in cfg.all_satellite_markers_data if m["track_index"] == track_index
    ]
    plotted_markers.extend(new_markers_for_this_track)

    # --- Create Plotter for this step ---
    plotter = pv.Plotter(
        window_size=[cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT],
        off_screen=not INTERACTIVE_MODE,
        image_scale=cfg.IMAGE_SCALE,
    )
    plotter.enable_depth_peeling()
    plotter.set_background(cfg.FIG_BG_COLOR)

    # 1. Plot Base Features
    cfg.add_earth_features(plotter)
    if cfg.PLOT_AURORA:
        cfg.add_aurora(plotter)

    # 2. Plot ALL Field Line Segments
    # print(f"   Adding {len(magnetic_field_lines_segments)} field line segments...") # Less verbose now
    for idx, data in enumerate(cfg.magnetic_field_lines_segments):
        cfg.add_single_fieldline_segment(plotter, data, idx)

    # 3. Plot ALL satellite tracks added UP TO this point
    # print(f"   Adding {len(plotted_satellite_tracks)} satellite tracks...")
    for idx, track_data in enumerate(plotted_satellite_tracks):
        cfg.add_full_satellite_track(plotter, track_data, idx)

    # 4. Plot ALL markers associated with the tracks added so far
    # print(f"   Adding {len(plotted_markers)} satellite markers...")
    if plotted_markers:
        cfg.add_satellite_markers(plotter, plotted_markers)

    # 5. Show or Screenshot - Pass the determined key, description, and final view flag
    show_or_screenshot_step(
        plotter, step_desc, camera_positions, step_key, is_final_view=is_final_view_flag
    )
    plotter.close()

print("===== Completed Step 2 (Final view is the last track step) =====")


# --- 保存相机位置 ---
if INTERACTIVE_MODE and camera_positions:
    try:
        # 将字符串路径转换为 Path 对象
        file_path = pathlib.Path(CAMERA_POS_FILE)

        # 获取父目录的 Path 对象
        parent_dir = file_path.parent

        # 创建父目录，包括所有必要的中间目录
        # parents=True: 创建所有父目录
        # exist_ok=True: 如果目录已存在，不引发错误
        parent_dir.mkdir(parents=True, exist_ok=True)

        # camera_positions 字典的值已经是 list of lists
        # 使用 Path 对象直接打开文件
        with open(file_path, "w") as f:
            json.dump(camera_positions, f, indent=4)

        print(f"\nSaved {len(camera_positions)} camera positions to {file_path}")  # 可以直接打印 Path 对象

    # 添加基本的错误处理
    except OSError as e:
        print(f"Error: Could not create directory or write file '{CAMERA_POS_FILE}'. Reason: {e}")
    except TypeError as e:
        print(f"Error: Data in 'camera_positions' is not JSON serializable. Reason: {e}")
    except Exception as e:  # 捕获其他意外错误
        print(f"An unexpected error occurred: {e}")

print("\n增量绘图完成 (最终图像为最后一次卫星轨迹添加步骤).")
