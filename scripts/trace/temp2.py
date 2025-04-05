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
    # 尝试使用 'trame' 或 'ipygany' 以获得更好的Jupyter体验
    # 如果在非Jupyter环境（如脚本直接运行），可能需要注释掉或选择 'None'
    pv.set_jupyter_backend("trame") # 或者 'ipygany', 'panel', None
    pv.global_theme.trame.server_proxy_enabled = False # 根据环境调整
    print("Attempting to use 'trame' backend.")
except Exception as e:
    try:
        pv.set_jupyter_backend(None) # 退回到默认桌面 (VTK/Qt)
        print(f"Could not configure jupyter backend, using default desktop (VTK/Qt) backend: {e}")
    except Exception as e_vtk:
         print(f"Could not configure any backend, using PyVista default: {e_vtk}")


# %% --- Workflow and Control Flags ---
INTERACTIVE_MODE = True # 设置为 True 进行交互式查看，False 进行截图
PLOT_AURORA = False
CAMERA_POS_FILE = "camera_positions_incremental.json"  # 新的相机位置文件
# ---------------------------------

# %% --- Constants and Paths --- (保持不变)
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
# SATELLITE_MARKER_INDICES = [[10000, 25000, 40000]] # 示例
SATELLITE_MARKER_INDICES = [[]] * len(SATELLITE_PATHS) # 确保与路径数量一致

# --- Style Settings --- (保持不变)
FIG_BG_COLOR = "#282c34"
TEXT_COLOR = "#abb2bf"
AXIS_LINE_COLOR = "#4b5263"
EARTH_COLOR = "#678094"
EARTH_SPECULAR = 0.2
EARTH_SPECULAR_POWER = 10
EARTH_SMOOTH_SHADING = True
FIELD_LINE_COLOR = "#ff0000"
FIELD_LINE_OPACITY = 0.65
FIELD_LINE_WIDTH = 0.1
POINT_COLOR = "#ffffff"
POINT_SIZE = 5
POINT_ALPHA = 0.9
SAT_TRACK_COLOR = "#56b6c2"
SAT_TRACK_WIDTH = 5 # 减小一点宽度以便看清增量
SAT_TRACK_OPACITY = 0.9
SAT_POS_COLOR = "#e06c75"
SAT_POS_SIZE = 10
POLE_MARKER_COLOR = "#ffffff"
POLE_MARKER_RADIUS = 0.03
AURORA_COLOR = "#98c379"
AURORA_OPACITY = 0.4
MAX_PLOT_RADIUS = 10.0 # 保持一致
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 800
IMAGE_SCALE = 2 # 截图时可以适当调小以加快速度

# ---------------------

# %% --- Earth Texture Loading --- (保持不变)
texture = None
if not os.path.exists(EARTH_TEXTURE_PATH):
    print(f"Warning: Earth texture file not found at '{EARTH_TEXTURE_PATH}'.")
else:
    try:
        texture = pv.read_texture(EARTH_TEXTURE_PATH)
        print(f"Successfully loaded Earth texture from '{EARTH_TEXTURE_PATH}'.")
    except Exception as e:
        print(f"Error loading texture '{EARTH_TEXTURE_PATH}': {e}.")


# %% --- Data Loading Functions --- (保持不变)
def geosph2geocar(times, alts, lats, lons):
    arr_n_3 = np.column_stack((alts, lats, lons))
    c = coord.Coords(arr_n_3, "GEO", "sph", ticks=Ticktock(times, "ISO"))
    return c.convert("GEO", "car")

def find_contiguous_segments_revised(mask):
    """
    查找布尔掩码中连续 True 块的开始和结束索引。

    Args:
        mask (np.ndarray or list): 一维布尔数组或列表。

    Returns:
        list: 一个元组列表，每个元组是 (start_index, end_index)。
              end_index 是排他的（不包含在段内），适用于切片 mask[start:end]。
              如果找不到 True 块或掩码为空，则返回空列表。
    """
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)

    if mask.ndim != 1:
        raise ValueError("Mask must be 1-dimensional.")
    if len(mask) == 0:
        return [] # 处理空掩码

    # 确保是布尔类型，以便正确转换为 int (0 或 1)
    mask_int = mask.astype(bool).astype(int)

    # 使用 np.diff 并前后填充 0 来查找变化点
    # prepend=0 处理掩码以 True 开始的情况
    # append=0 处理掩码以 True 结束的情况
    diff = np.diff(mask_int, prepend=0, append=0)

    # 查找 True 块开始的位置 (0 -> 1 的转变)
    starts = np.where(diff == 1)[0]

    # 查找 True 块结束的位置 (1 -> 0 的转变)
    # 'ends' 中的索引是最后一个 True 值 *之后* 的索引。
    ends = np.where(diff == -1)[0]

    # 由于我们用 0 填充了，starts 的长度应该总是等于 ends 的长度
    if len(starts) != len(ends):
        # 理论上这不应该发生，但加个检查以防万一
        print(f"Warning: Mismatch in starts ({len(starts)}) and ends ({len(ends)}). This is unexpected.")
        # 尝试安全地 zip
        min_len = min(len(starts), len(ends))
        starts = starts[:min_len]
        ends = ends[:min_len]

    # 将开始和结束索引配对。'end' 索引对于切片是排他的。
    segments = list(zip(starts, ends))

    # --- 不再需要复杂的边界处理或过滤 ---
    # 如果需要过滤掉长度小于2（即只有一个点）的段：
    # segments_min_len_2 = [(s, e) for s, e in segments if e > s + 1]
    # return segments_min_len_2
    # 这里我们返回所有找到的段，包括可能只有一个点的段（如果需要的话）

    return segments


def extract_timestamps(filename):
    match = re.search(r"_case_(\d{8}T\d{6})_(\d{8}T\d{6})\.pkl$", filename)
    if not match:
        # 尝试匹配不带 case_ 的格式
        match_alt = re.search(r"(\d{8}T\d{6})_(\d{8}T\d{6})\.pkl$", filename)
        if not match_alt:
            raise ValueError(f"文件名格式不符: {filename}")
        match = match_alt # 使用备用匹配

    # 假设 start_ 或 end_ 总是存在
    if filename.startswith("start_"):
        return match.group(1)
    elif filename.startswith("end_"):
        return match.group(2)
    else:
        # 如果没有 start_/end_ 前缀，我们可以尝试根据时间猜测，但这不可靠
        # 或者返回两个时间让调用者决定
        # 这里我们强制要求前缀
        raise ValueError("文件名必须以 'start_' 或 'end_' 开头")

# %% --- Load Trace Data --- (处理逻辑稍作调整，预处理数据)
magnetic_field_lines_segments = [] # 存储所有有效的、在半径内的磁力线段
print(f"Loading and processing {len(TRACE_FILE_PATHS)} trace magnetic field lines...")
for (i, trace_path), trace_fn in zip(enumerate(TRACE_FILE_PATHS), TRACE_FILE_NAMES):
    print(f"Processing trace magnetic field line file {i+1}: {trace_path}")
    try:
        df_trace = pd.read_pickle(trace_path)
        if df_trace.empty:
            print(f"  Skipping empty file: {trace_path}")
            continue
        df_trace_np = df_trace.values
        df_trace_np_alt_re = df_trace_np.copy()
        # 确保列数正确
        if df_trace_np_alt_re.shape[1] < 3:
             print(f"  Skipping file with insufficient columns: {trace_path}")
             continue

        df_trace_np_alt_re[:, 0] = (df_trace_np_alt_re[:, 0] + Re_km) / Re_km
        time_trace = extract_timestamps(trace_fn)
        times_trace = np.array([str(time_trace)] * len(df_trace_np_alt_re[:, 0]))
        geo_car_trace = geosph2geocar(
            times_trace,
            df_trace_np_alt_re[:, 0],
            df_trace_np_alt_re[:, 1],
            df_trace_np_alt_re[:, 2],
        )
        line_data = geo_car_trace.data

        # --- 在加载时就进行半径裁剪和分段 ---
        if line_data is not None and line_data.shape[0] >= 2 and line_data.shape[1] == 3:
            r = np.linalg.norm(line_data, axis=1)
            mask = r <= MAX_PLOT_RADIUS
            segments_indices = find_contiguous_segments_revised(mask) # 使用修正后的函数
            print(f"  Found {len(segments_indices)} segments within radius for file {i+1}.")
            for start_idx, end_idx in segments_indices:
                segment_data = line_data[start_idx:end_idx]
                if segment_data.shape[0] >= 2: # 确保段至少有两个点
                    magnetic_field_lines_segments.append(segment_data)
                # else:
                    # print(f"  Segment from {start_idx} to {end_idx} too short ({segment_data.shape[0]} points), skipping.")

        print(f"  File {i+1} processed. Total segments collected so far: {len(magnetic_field_lines_segments)}")

    except Exception as e:
        print(f"Error processing trace points file {trace_path}: {e}. Skipping.")

if not magnetic_field_lines_segments:
    print("Warning: No valid magnetic field line segments loaded or processed.")
else:
    print(f"Successfully processed {len(magnetic_field_lines_segments)} magnetic field line segments.")


# %% --- Load Satellite Data --- (处理逻辑稍作调整，预处理数据)
satellite_tracks_segments = [] # 存储所有有效的、在半径内的卫星轨迹段
satellite_markers_processed = [] # 存储处理过的、在半径内的标记点及其对应的轨迹索引

print(f"Loading and processing {len(SATELLITE_PATHS)} satellite track(s)...")
if len(SATELLITE_PATHS) != len(SATELLITE_MARKER_INDICES):
    print(f"Warning: Mismatch between satellite paths ({len(SATELLITE_PATHS)}) and marker lists ({len(SATELLITE_MARKER_INDICES)})!")
    # 尝试修正，使用空列表填充或截断
    SATELLITE_MARKER_INDICES = (SATELLITE_MARKER_INDICES + [[]] * len(SATELLITE_PATHS))[:len(SATELLITE_PATHS)]

for i, sat_path in enumerate(SATELLITE_PATHS):
    print(f"Processing satellite track file {i+1}: {sat_path}")
    track_segments_for_this_file = []
    try:
        df_sa = pd.read_pickle(sat_path)
        if df_sa.empty:
             print(f"  Skipping empty file: {sat_path}")
             continue

        lats_sa = df_sa["Latitude"].values
        lons_sa = df_sa["Longitude"].values
        alts_sa = (np.full(len(lats_sa), SWARM_ALTITUDE) + Re_km) / Re_km
        times_sa = df_sa.index.to_numpy() # 使用 .to_numpy() 获取 ndarray
        times_sa_str = np.array([str(pd.Timestamp(t)) for t in times_sa]) # 确保时间格式正确
        geo_car_sa = geosph2geocar(times_sa_str, alts_sa, lats_sa, lons_sa)
        track_data = geo_car_sa.data

        current_markers = SATELLITE_MARKER_INDICES[i]
        markers_in_radius = []

        # --- 在加载时就进行半径裁剪和分段 ---
        if track_data is not None and track_data.shape[0] >= 2 and track_data.shape[1] == 3:
            r_sat = np.linalg.norm(track_data, axis=1)
            mask_sat = r_sat <= MAX_PLOT_RADIUS
            segments_indices = find_contiguous_segments_revised(mask_sat) # 使用修正后的函数
            print(f"  Found {len(segments_indices)} track segments within radius for file {i+1}.")
            for start, end in segments_indices:
                segment = track_data[start:end]
                if segment.shape[0] >= 2: # 确保段至少有两个点
                    # 存储属于这个轨迹文件的段，并标记其来源文件索引 i
                    satellite_tracks_segments.append({"data": segment, "track_index": i})
                    track_segments_for_this_file.append(segment)
                # else:
                    # print(f"  Track segment from {start} to {end} too short ({segment.shape[0]} points), skipping.")

            # 处理标记点：检查原始索引对应的点是否在半径内
            for idx in current_markers:
                 if 0 <= idx < len(track_data) and mask_sat[idx]: # 检查索引有效且点在半径内
                     markers_in_radius.append({"point": track_data[idx], "track_index": i})

            satellite_markers_processed.extend(markers_in_radius)
            print(f"  File {i+1} processed. Total track segments: {len(satellite_tracks_segments)}. Total markers in radius: {len(satellite_markers_processed)}")


    except Exception as e:
        print(f"Error processing satellite file {sat_path}: {e}. Skipping.")

if not satellite_tracks_segments:
    print("Warning: No valid satellite track segments loaded or processed.")
else:
    print(f"Successfully processed {len(satellite_tracks_segments)} satellite track segments.")
    print(f"Successfully processed {len(satellite_markers_processed)} satellite markers within radius.")


# %% --- Plotting Helper Functions --- (部分保持不变, 部分修改/新增)

def show_or_screenshot_step(plotter, step_description, camera_pos_dict, step_key):
    """Handles showing plot or taking screenshot, manages camera positions."""
    plotter.add_text(
        f"Step: {step_description}",
        position="lower_edge",
        font_size=10,
        color=TEXT_COLOR,
        name="step_text", # 确保可以移除
    )
    if INTERACTIVE_MODE:
        print(f"\n--- Showing Interactive Step: {step_description} ---")
        print(">>> Adjust view. Close window to continue AND SAVE VIEW. <<<")
        # 添加轴和标题在显示之前
        plotter.add_axes(interactive=True, line_width=2, color=TEXT_COLOR)
        plotter.add_text(
             f"Alfven Wave Trace (View: {MAX_PLOT_RADIUS} Re)",
             position="upper_edge", color=TEXT_COLOR, font_size=12, name="main_title"
        )
        # 尝试应用已保存的相机位置（如果存在）
        if step_key in camera_pos_dict:
             try:
                 plotter.camera_position = camera_pos_dict[step_key]
                 print(f"Applied saved camera position for '{step_key}'.")
             except Exception as e:
                 print(f"Warning: Could not apply camera position for '{step_key}': {e}")
        else:
             # 如果没有保存的位置，设置一个默认的（可选）
             # plotter.camera.zoom(0.8)
             pass

        plotter.show(title=step_description) # 显示窗口

        # 保存相机位置
        try:
            # camera_position 返回的是一个包含3个元组的列表，每个元组是(x, y, z)
            raw_camera_pos = plotter.camera_position
            # 转换为 list of lists 以便 JSON 序列化
            camera_pos_list = [list(pos) for pos in raw_camera_pos]
            camera_pos_dict[step_key] = camera_pos_list
            print(f"Camera position for '{step_key}' saved.")
        except Exception as e:
            print(f"Warning: Could not get/save camera position for '{step_key}': {e}")
    else:  # Batch mode
        filename = f"{step_key}_screenshot.png".replace(" ", "_").replace(":", "_") # 清理文件名
        print(f"\n--- Generating Screenshot: {step_description} ({filename}) ---")
        # 添加轴和标题
        plotter.add_axes(interactive=False, line_width=2, color=TEXT_COLOR) # 非交互式轴
        plotter.add_text(
             f"Alfven Wave Trace (View: {MAX_PLOT_RADIUS} Re)",
             position="upper_edge", color=TEXT_COLOR, font_size=12, name="main_title"
        )

        if step_key in camera_pos_dict:
            try:
                plotter.camera_position = camera_pos_dict[step_key]
                print(f"Applied saved camera position for '{step_key}'.")
            except Exception as e:
                print(f"Warning: Could not apply camera position for '{step_key}': {e}")
        else:
            print(f"Warning: No saved camera position for '{step_key}'. Using default view.")
            # plotter.camera.zoom(0.8) # 可以为截图设置默认缩放

        try:
            plotter.screenshot(filename, scale=IMAGE_SCALE, transparent_background=False)
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"ERROR taking screenshot for '{step_key}': {e}")

# create_aurora_texture and add_aurora 保持不变 (如果 PLOT_AURORA=True 会用到)
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
                opacity=1.0, # 纹理自带透明度
                show_edges=False,
                smooth_shading=True,
                rgb=False,
                name=f"{name}_aurora",
                use_transparency=True, # 启用透明度渲染
            )
        return True
    except Exception as e:
        print(f"ERROR adding aurora: {e}")
        return False


# add_earth_features 保持不变
def add_earth_features(plotter_instance):
    """Adds Earth sphere, poles, and labels."""
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
        # Corrected check:
        if 'earth' not in plotter_instance.actors:
            # Fallback: Add the earth mesh if it wasn't added successfully before
            print("Attempting to add fallback Earth mesh.")  # 添加日志以便调试
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
            [(0, 0, R_E * 1.1)], ["N"], point_size=10, font_size=12,
            text_color=TEXT_COLOR, shape=None, name="north_label", always_visible=True
        )
        plotter_instance.add_point_labels(
            [(0, 0, -R_E * 1.1)], ["S"], point_size=10, font_size=12,
            text_color=TEXT_COLOR, shape=None, name="south_label", always_visible=True
        )
    except Exception as e:
        print(f"Warning: Failed adding poles/labels: {e}")
    return plotter_instance

# 新增：绘制单条磁力线段
def add_single_fieldline_segment(plotter_instance, segment_data, index):
    """Adds a single magnetic field line segment to the plotter."""
    if segment_data is None or segment_data.shape[0] < 2:
        return
    points = segment_data
    n_points = len(points)
    # 创建 PolyData 线条: [2, p0_idx, p1_idx, 2, p1_idx, p2_idx, ...]
    lines_array = np.hstack([[2, k, k + 1] for k in range(n_points - 1)]).astype(int)
    line_polydata = pv.PolyData(points, lines=lines_array)
    plotter_instance.add_mesh(
        line_polydata,
        color=FIELD_LINE_COLOR,
        opacity=FIELD_LINE_OPACITY,
        line_width=FIELD_LINE_WIDTH,
        name=f"fieldline_segment_{index}" # 为每个段指定唯一名称
    )
    # 添加端点（可选，如果需要的话）
    # plotter_instance.add_points(
    #     [segment_data[0, :], segment_data[-1, :]],
    #     color=POINT_COLOR, point_size=POINT_SIZE,
    #     render_points_as_spheres=True, opacity=POINT_ALPHA,
    #     name=f"fieldline_endpoints_{index}"
    # )

# 新增：绘制单条卫星轨迹段
def add_single_track_segment(plotter_instance, segment_data, index):
    """Adds a single satellite track segment to the plotter."""
    if segment_data is None or segment_data.shape[0] < 2:
        return
    points = segment_data
    n_points = len(points)
    lines_array = np.hstack([[2, k, k + 1] for k in range(n_points - 1)]).astype(int)
    track_poly = pv.PolyData(points, lines=lines_array)
    plotter_instance.add_mesh(
        track_poly,
        color=SAT_TRACK_COLOR,
        line_width=SAT_TRACK_WIDTH,
        opacity=SAT_TRACK_OPACITY,
        name=f"track_segment_{index}" # 为每个段指定唯一名称
    )

# 新增：绘制卫星标记点
def add_satellite_markers(plotter_instance, marker_list):
    """Adds satellite markers to the plotter."""
    if not marker_list:
        return
    marker_points = np.array([m["point"] for m in marker_list])
    if marker_points.size > 0:
        plotter_instance.add_points(
            marker_points,
            color=SAT_POS_COLOR,
            point_size=SAT_POS_SIZE,
            render_points_as_spheres=True,
            opacity=POINT_ALPHA,
            name="satellite_markers" # 可以使用一个统一的名称
        )


# %% --- Main Execution (Incremental Plotting) ---

print(
    f"Starting PyVista plotting in {'INTERACTIVE' if INTERACTIVE_MODE else 'BATCH'} mode (Incremental)..."
)
pv.set_plot_theme("paraview") # 或者 "document"

# 加载相机位置
camera_positions = {}
if not INTERACTIVE_MODE or os.path.exists(CAMERA_POS_FILE):
    if os.path.exists(CAMERA_POS_FILE):
        try:
            with open(CAMERA_POS_FILE, "r") as f:
                loaded_pos = json.load(f)
                 # JSON 加载的是 list of lists, PyVista 需要 list of tuples
                for key, pos_list in loaded_pos.items():
                     if isinstance(pos_list, list) and len(pos_list) == 3 and all(isinstance(sublist, list) for sublist in pos_list):
                         camera_positions[key] = [tuple(p) for p in pos_list]
                     else:
                         # 如果格式不符，尝试跳过或记录警告
                         print(f"Warning: Invalid camera position format for key '{key}' in {CAMERA_POS_FILE}. Skipping.")
                print(f"Loaded {len(camera_positions)} camera positions from {CAMERA_POS_FILE}")
        except Exception as e:
            print(f"Warning: Could not load camera positions from {CAMERA_POS_FILE}: {e}")
            if not INTERACTIVE_MODE:
                print("Proceeding with default views.")
    elif not INTERACTIVE_MODE:
        print(f"Warning: Camera position file '{CAMERA_POS_FILE}' not found. Using default views.")


# --- Step 0: Plot Earth ---
print("\n===== Step 0: Plotting Earth =====")
step_key_earth = "step_0_Earth"
plotter = pv.Plotter(
    window_size=[WINDOW_WIDTH, WINDOW_HEIGHT], off_screen=not INTERACTIVE_MODE, image_scale=IMAGE_SCALE
)
plotter.enable_depth_peeling() # 启用以处理透明度
plotter.set_background(FIG_BG_COLOR)
add_earth_features(plotter)
if PLOT_AURORA:
    add_aurora(plotter)
show_or_screenshot_step(plotter, "0: Earth and Aurora" if PLOT_AURORA else "0: Earth", camera_positions, step_key_earth)
plotter.close()
print("===== Completed Step 0 =====")


# --- Step 1: Plot Field Line Segments Incrementally ---
print(f"\n===== Step 1: Plotting {len(magnetic_field_lines_segments)} Field Line Segments =====")
plotted_fieldline_segment_data = [] # 用于累积绘制的数据
for i, segment_data in enumerate(magnetic_field_lines_segments):
    step_key = f"step_1_fieldline_{i+1}"
    step_desc = f"1: Added Field Line Segment {i+1}/{len(magnetic_field_lines_segments)}"
    print(f"--- Processing: {step_desc} ---")

    plotted_fieldline_segment_data.append(segment_data) # 添加当前段到累积列表

    plotter = pv.Plotter(
        window_size=[WINDOW_WIDTH, WINDOW_HEIGHT], off_screen=not INTERACTIVE_MODE, image_scale=IMAGE_SCALE
    )
    plotter.enable_depth_peeling()
    plotter.set_background(FIG_BG_COLOR)

    # 绘制基础地球和极光
    add_earth_features(plotter)
    if PLOT_AURORA:
        add_aurora(plotter)

    # 重新绘制所有已添加的磁力线段
    for idx, data in enumerate(plotted_fieldline_segment_data):
        add_single_fieldline_segment(plotter, data, idx)

    # 显示或截图
    show_or_screenshot_step(plotter, step_desc, camera_positions, step_key)
    plotter.close()

print("===== Completed Step 1 =====")


# --- Step 2: Plot Satellite Track Segments Incrementally ---
print(f"\n===== Step 2: Plotting {len(satellite_tracks_segments)} Satellite Track Segments =====")
plotted_track_segment_data = [] # 用于累积绘制的轨迹段数据
plotted_markers = [] # 用于累积绘制的标记点

# 先对轨迹段排序，确保来自同一个卫星的段连续处理（如果需要按卫星显示）
# 如果不需要按卫星分组显示，可以去掉排序
# satellite_tracks_segments.sort(key=lambda x: x['track_index'])

# Track index of the last plotted segment to manage marker plotting logic
last_plotted_track_index = -1
markers_to_plot_this_step = []

for i, segment_info in enumerate(satellite_tracks_segments):
    segment_data = segment_info["data"]
    current_track_index = segment_info["track_index"]
    step_key = f"step_2_track_{current_track_index}_segment_{i+1}" # 使用更详细的 key
    step_desc = f"2: Added Track Segment {i+1}/{len(satellite_tracks_segments)} (from Track {current_track_index+1})"
    print(f"--- Processing: {step_desc} ---")

    plotted_track_segment_data.append(segment_data) # 添加当前段

    # 决定在这一步绘制哪些标记点
    # 简单策略：当绘制属于某卫星的任何段时，绘制该卫星 *所有* 在半径内的标记点
    # （这样标记点会随着第一个相关段出现，并一直存在）
    markers_to_plot_this_step = [m for m in satellite_markers_processed if m['track_index'] <= current_track_index]
    # 或者只绘制当前 track_index 的 markers:
    # markers_to_plot_this_step = [m for m in satellite_markers_processed if m['track_index'] == current_track_index]
    # 或者累积所有见过的 markers:
    if current_track_index != last_plotted_track_index:
         new_markers = [m for m in satellite_markers_processed if m['track_index'] == current_track_index]
         plotted_markers.extend(new_markers) # Add markers when the first segment of a new track appears
         last_plotted_track_index = current_track_index

    plotter = pv.Plotter(
        window_size=[WINDOW_WIDTH, WINDOW_HEIGHT], off_screen=not INTERACTIVE_MODE, image_scale=IMAGE_SCALE
    )
    plotter.enable_depth_peeling()
    plotter.set_background(FIG_BG_COLOR)

    # 绘制基础地球和极光
    add_earth_features(plotter)
    if PLOT_AURORA:
        add_aurora(plotter)

    # 绘制所有已添加的磁力线段
    for idx, data in enumerate(plotted_fieldline_segment_data):
        add_single_fieldline_segment(plotter, data, idx)

    # 绘制所有已添加的卫星轨迹段
    for idx, data in enumerate(plotted_track_segment_data):
        add_single_track_segment(plotter, data, idx)

    # 绘制累积的标记点
    if plotted_markers:
        add_satellite_markers(plotter, plotted_markers)

    # 显示或截图
    show_or_screenshot_step(plotter, step_desc, camera_positions, step_key)
    plotter.close()

print("===== Completed Step 2 =====")


# --- Step 3: Final Plot (Optional, shows everything at once) ---
print("\n===== Step 3: Final Combined Plot =====")
step_key_final = "step_3_Final"
step_desc_final = "3: Final Combined Plot"

plotter = pv.Plotter(
    window_size=[WINDOW_WIDTH, WINDOW_HEIGHT], off_screen=not INTERACTIVE_MODE, image_scale=IMAGE_SCALE
)
plotter.enable_depth_peeling()
plotter.set_background(FIG_BG_COLOR)

# 绘制基础地球和极光
add_earth_features(plotter)
if PLOT_AURORA:
    add_aurora(plotter)

# 绘制所有磁力线段
for idx, data in enumerate(plotted_fieldline_segment_data):
    add_single_fieldline_segment(plotter, data, idx)

# 绘制所有卫星轨迹段
for idx, data in enumerate(plotted_track_segment_data):
    add_single_track_segment(plotter, data, idx)

# 绘制所有标记点
if plotted_markers:
    add_satellite_markers(plotter, plotted_markers)

# 应用最终设置（移除步骤文本，添加最终标题等）
# apply_final_settings(plotter, MAX_PLOT_RADIUS) # 如果有这个函数
# 这里手动添加最终标题和轴
plotter.add_axes(interactive=INTERACTIVE_MODE, line_width=2, color=TEXT_COLOR)
title = f"Final Plot: Alfven Wave Trace (View: {MAX_PLOT_RADIUS} Re)"
plotter.add_text(
    title, position="upper_edge", color=TEXT_COLOR, font_size=12, name="final_title"
)


show_or_screenshot_step(plotter, step_desc_final, camera_positions, step_key_final)
plotter.close()
print("===== Completed Step 3 =====")


# --- 保存相机位置 ---
if INTERACTIVE_MODE and camera_positions:
    try:
        # camera_positions 字典的值已经是 list of lists
        with open(CAMERA_POS_FILE, "w") as f:
            json.dump(camera_positions, f, indent=4)
        print(f"\nSaved {len(camera_positions)} camera positions to {CAMERA_POS_FILE}")
    except Exception as e:
        print(f"Warning: Could not save camera positions: {e}")

print("\n增量绘图完成.")