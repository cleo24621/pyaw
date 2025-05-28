import os
import json
import re

import numpy as np
import pandas as pd
import pyvista as pv
from spacepy import coordinates as coord
from spacepy.time import Ticktock
import glob
from scipy.spatial.distance import cdist

#  --- Backend Setup ---
try:
    # 尝试使用 'trame' 或 'ipygany' 以获得更好的Jupyter体验
    # 如果在非Jupyter环境（如脚本直接运行），可能需要注释掉或选择 ''
    pv.set_jupyter_backend("trame")  # 或者 'ipygany', 'panel', None
    pv.global_theme.trame.server_proxy_enabled = False  # 根据环境调整
    print("Attempting to use 'trame' backend.")
except Exception as e:
    try:
        pv.set_jupyter_backend("")  # 退回到默认桌面 (VTK/Qt)
        print(
            f"Could not configure jupyter backend, using default desktop (VTK/Qt) backend: {e}"
        )
    except Exception as e_vtk:
        print(f"Could not configure any backend, using PyVista default: {e_vtk}")


#  --- Constants and Paths ---
R_E = 1.0
Re_km = 6371.2
SWARM_RADIUS = 6816838.5 * 1e-3  # km
SWARM_ALTITUDE = SWARM_RADIUS - Re_km  # km
EARTH_TEXTURE_PATH = "eo_base_2020_clean_3600x1800.png"
ORBIT_NUM = 12738  # modify
TRACE_FILE_DIR = (
    f"scripts/results/aw_cases/archive/trace_points/pkl/{ORBIT_NUM}"
)
TRACE_FILE_PATHS = glob.glob(os.path.join(TRACE_FILE_DIR, "*.pkl"))
TRACE_FILE_NAMES = [os.path.basename(path) for path in TRACE_FILE_PATHS]
SATELLITE_DIR = rf"scripts\results\aw_cases\archive\orbits\{ORBIT_NUM}"
SATELLITE_PATHS = glob.glob(os.path.join(SATELLITE_DIR, "*.pkl"))
# SATELLITE_MARKER_INDICES = [[0, -1]] * len(SATELLITE_PATHS)  # 确保与路径数量一致
SATELLITE_MARKER_INDICES = [[]] * len(SATELLITE_PATHS)  # 确保与路径数量一致

# --- Style Settings ---
FIG_BG_COLOR = "#282c34"
TEXT_COLOR = "#abb2bf"
AXIS_LINE_COLOR = "#4b5263"
EARTH_COLOR = "#678094"
EARTH_SPECULAR = 0.2
EARTH_SPECULAR_POWER = 10
EARTH_SMOOTH_SHADING = True
FIELD_LINE_COLOR = "#ff0000"
FIELD_LINE_OPACITY = 1
FIELD_LINE_WIDTH = 5
# Field Line Points
POINT_COLOR = "#ffffff"  # 颜色：磁力线端点的颜色
POINT_SIZE = 15  # 大小：磁力线端点的尺寸 (像素或点单位，取决于后端)
POINT_ALPHA = 0.9  # 透明度：磁力线端点的不透明度 (0=透明, 1=不透明)
#
SAT_TRACK_COLOR = "#56b6c2"
SAT_TRACK_WIDTH = 50  # 减小一点宽度以便看清增量
SAT_TRACK_OPACITY = 1
SAT_POS_COLOR = "#e06c75"
SAT_POS_SIZE = 8
POLE_MARKER_COLOR = "#ffffff"
POLE_MARKER_RADIUS = 0.03
PLOT_AURORA = False
AURORA_COLOR = "#98c379"
AURORA_OPACITY = 0.4
MAX_PLOT_RADIUS = 10.0  # 保持一致
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 800
# WINDOW_WIDTH = 2000
# WINDOW_HEIGHT = 1000
IMAGE_SCALE = 10  # 截图时可以适当调小以加快速度

DISTANCE_THRESHOLD = 0.01  # (in Re) Adjust this threshold as needed!
REMOVE_OVERLAPPING = False  # Set to True to remove duplicates, False to just warn
# ---------------------

#  --- Earth Texture Loading ---
texture = None
if not os.path.exists(EARTH_TEXTURE_PATH):
    print(f"Warning: Earth texture file not found at '{EARTH_TEXTURE_PATH}'.")
else:
    try:
        texture = pv.read_texture(EARTH_TEXTURE_PATH)
        print(f"Successfully loaded Earth texture from '{EARTH_TEXTURE_PATH}'.")
    except Exception as e:
        print(f"Error loading texture '{EARTH_TEXTURE_PATH}': {e}.")


#  --- Data Loading Functions ---
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
        return []  # 处理空掩码

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
        print(
            f"Warning: Mismatch in starts ({len(starts)}) and ends ({len(ends)}). This is unexpected."
        )
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
        match = match_alt  # 使用备用匹配

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

#  --- Load Trace Data ---
magnetic_field_lines_segments = []  # 存储所有有效的、在半径内的磁力线段
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
        if (
            line_data is not None
            and line_data.shape[0] >= 2
            and line_data.shape[1] == 3
        ):
            r = np.linalg.norm(line_data, axis=1)
            mask = r <= MAX_PLOT_RADIUS
            segments_indices = find_contiguous_segments_revised(
                mask
            )  # 使用修正后的函数
            print(
                f"  Found {len(segments_indices)} segments within radius for file {i+1}."
            )
            for start_idx, end_idx in segments_indices:
                segment_data = line_data[start_idx:end_idx]
                if segment_data.shape[0] >= 2:  # 确保段至少有两个点
                    magnetic_field_lines_segments.append(segment_data)
                # else:
                # print(f"  Segment from {start_idx} to {end_idx} too short ({segment_data.shape[0]} points), skipping.")

        print(
            f"  File {i+1} processed. Total segments collected so far: {len(magnetic_field_lines_segments)}"
        )

    except Exception as e:
        print(f"Error processing trace points file {trace_path}: {e}. Skipping.")

if not magnetic_field_lines_segments:
    print("Warning: No valid magnetic field line segments loaded or processed.")
else:
    print(
        f"Successfully processed {len(magnetic_field_lines_segments)} magnetic field line segments."
    )

print(
    f"\n--- Checking for potentially overlapping field line segments (Threshold: {DISTANCE_THRESHOLD} Re) ---"
)

original_segment_count = len(magnetic_field_lines_segments)
segments_to_keep_indices = set(
    range(original_segment_count)
)  # Start assuming we keep all
potentially_overlapping_pairs = []

# Compare each segment with every other segment that comes after it
for i in range(original_segment_count):
    # If segment 'i' was already marked for removal by a previous comparison, skip it
    if i not in segments_to_keep_indices:
        continue

    segment_a = magnetic_field_lines_segments[i]

    for j in range(i + 1, original_segment_count):
        # If segment 'j' was already marked for removal, skip it
        if j not in segments_to_keep_indices:
            continue

        segment_b = magnetic_field_lines_segments[j]

        # Ensure segments have points before calculating distance
        if segment_a.shape[0] == 0 or segment_b.shape[0] == 0:
            continue

        try:
            # Calculate all pairwise distances between points of segment_a and segment_b
            dist_matrix = cdist(segment_a, segment_b)

            # Find the minimum distance between any point in A and any point in B
            min_dist = np.min(dist_matrix)

            if min_dist < DISTANCE_THRESHOLD:
                potentially_overlapping_pairs.append((i, j, min_dist))
                print(
                    f"  -> Potential overlap detected: Segment {i} and Segment {j} (Min Dist: {min_dist:.4f} Re)"
                )

                # Decide how to handle overlap (e.g., remove the second one)
                if REMOVE_OVERLAPPING:
                    # Mark segment 'j' for removal
                    if j in segments_to_keep_indices:
                        segments_to_keep_indices.remove(j)
                        # print(f"     Marked segment {j} for removal.")

        except Exception as e:
            print(f"  Error calculating distance between segment {i} and {j}: {e}")


#  --- Load Satellite Data ---
all_satellite_tracks_data = []  # List to store full track data arrays
all_satellite_markers_data = (
    []
)  # List to store markers {point: ndarray, track_index: int}

print(f"Loading {len(SATELLITE_PATHS)} satellite track(s) (full tracks)...")
if len(SATELLITE_PATHS) != len(SATELLITE_MARKER_INDICES):
    print(
        f"Warning: Mismatch between satellite paths ({len(SATELLITE_PATHS)}) and marker lists ({len(SATELLITE_MARKER_INDICES)})!"
    )
    SATELLITE_MARKER_INDICES = (SATELLITE_MARKER_INDICES + [[]] * len(SATELLITE_PATHS))[
        : len(SATELLITE_PATHS)
    ]

for i, sat_path in enumerate(SATELLITE_PATHS):
    print(f"Processing satellite track file {i+1}: {sat_path}")
    try:
        df_sa = pd.read_pickle(sat_path)
        if df_sa.empty:
            print(f"  Skipping empty file: {sat_path}")
            continue

        lats_sa = df_sa["Latitude"].values
        lons_sa = df_sa["Longitude"].values
        alts_sa = (np.full(len(lats_sa), SWARM_ALTITUDE) + Re_km) / Re_km
        times_sa = df_sa.index.to_numpy()
        times_sa_str = np.array(
            [str(pd.Timestamp(t)) for t in times_sa]
        )  # Ensure correct timestamp format
        geo_car_sa = geosph2geocar(times_sa_str, alts_sa, lats_sa, lons_sa)
        track_data = geo_car_sa.data  # This is the full track in GEO Cartesian (Re)

        if (
            track_data is not None
            and track_data.shape[0] >= 2
            and track_data.shape[1] == 3
        ):
            # Store the full track data
            all_satellite_tracks_data.append(track_data)

            # Process and store markers associated with this track index
            current_markers_indices = SATELLITE_MARKER_INDICES[i]
            num_points = len(track_data)  # Get length AFTER loading track_data

            print(
                f"  Track {i + 1} loaded ({num_points} points). Requested marker indices: {current_markers_indices}"
            )

            processed_markers_count = 0
            actual_indices_to_mark = (
                set()
            )  # Use a set to avoid duplicates if 0 and -1 resolve to the same for very short tracks

            for requested_idx in current_markers_indices:
                actual_idx = -999  # Default invalid index
                if requested_idx == 0:
                    actual_idx = 0
                elif requested_idx == -1:
                    # Translate -1 to the actual last index
                    actual_idx = num_points - 1
                elif requested_idx > 0:
                    # Keep other positive indices as they are
                    actual_idx = requested_idx
                else:
                    # Handle other negative indices or invalid values if necessary
                    print(
                        f"  Warning: Unsupported marker index {requested_idx} encountered for track {i + 1}. Skipping."
                    )
                    continue  # Skip to next requested index

                # Check if the *actual* index is valid for this track
                if 0 <= actual_idx < num_points:
                    actual_indices_to_mark.add(actual_idx)
                else:
                    # This could happen if track is very short (e.g., 1 point, 0 and -1 are same) or user gave out-of-bounds positive index
                    print(
                        f"  Warning: Resolved marker index {actual_idx} (from {requested_idx}) is out of bounds for track {i + 1} (length {num_points}). Skipping."
                    )

            print(
                f"  Actual indices to be marked: {sorted(list(actual_indices_to_mark))}"
            )

            # Now iterate through the valid, resolved indices
            for idx in actual_indices_to_mark:
                # Store marker point and its track index
                all_satellite_markers_data.append(
                    {"point": track_data[idx], "track_index": i}
                )
                processed_markers_count += 1
            # --- MODIFICATION END ---

            print(
                f"  Successfully processed {processed_markers_count} markers for this track."
            )
            for idx in current_markers_indices:
                if 0 <= idx < len(track_data):  # Check index validity
                    # Store marker point and its track index
                    all_satellite_markers_data.append(
                        {"point": track_data[idx], "track_index": i}
                    )
                else:
                    print(
                        f"  Warning: Marker index {idx} out of bounds for track {i+1} (length {len(track_data)}). Skipping marker."
                    )
            print(
                f"  Track {i+1} loaded successfully ({track_data.shape[0]} points). Associated markers processed: {len([m for m in all_satellite_markers_data if m['track_index'] == i])}"
            )
        else:
            print(
                f"  Skipping track {i+1} due to insufficient data points after conversion."
            )

    except Exception as e:
        print(f"Error processing satellite file {sat_path}: {e}. Skipping.")

if not all_satellite_tracks_data:
    print("Warning: No satellite tracks loaded successfully.")
else:
    print(
        f"Successfully loaded {len(all_satellite_tracks_data)} full satellite tracks."
    )
    print(f"Total processed markers: {len(all_satellite_markers_data)}")


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
                opacity=1.0,  # 纹理自带透明度
                show_edges=False,
                smooth_shading=True,
                rgb=False,
                name=f"{name}_aurora",
                use_transparency=True,  # 启用透明度渲染
            )
        return True
    except Exception as e:
        print(f"ERROR adding aurora: {e}")
        return False


# add_earth_features
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
        if "earth" not in plotter_instance.actors:
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
            [(0, 0, R_E * 1.0)],
            ["N"],
            point_size=0.1,
            font_size=20,
            text_color=TEXT_COLOR,
            shape=None,
            name="north_label",
            always_visible=True,
        )
        plotter_instance.add_point_labels(
            [(0, 0, -R_E * 1.0)],
            ["S"],
            point_size=0.1,
            font_size=20,
            text_color=TEXT_COLOR,
            shape=None,
            name="south_label",
            always_visible=True,
        )
    except Exception as e:
        print(f"Warning: Failed adding poles/labels: {e}")
    return plotter_instance


def add_single_fieldline_segment(plotter_instance, segment_data, index):
    """
    Adds a single magnetic field line segment AND markers at its endpoints
    to the plotter.
    """
    if segment_data is None or segment_data.shape[0] < 2:
        # Segment is too short to have a start and end, or is invalid
        return

    # 1. Add the line segment itself (as before)
    points = segment_data
    n_points = len(points)
    lines_array = np.hstack([[2, k, k + 1] for k in range(n_points - 1)]).astype(int)
    line_polydata = pv.PolyData(points, lines=lines_array)
    plotter_instance.add_mesh(
        line_polydata,
        color=FIELD_LINE_COLOR,
        opacity=FIELD_LINE_OPACITY,
        line_width=FIELD_LINE_WIDTH,
        name=f"fieldline_segment_{index}",  # Unique name for the line
    )

    # --- NEW: Add markers for the start and end points ---
    start_point = segment_data[0, :]  # First point of the segment
    end_point = segment_data[-1, :]  # Last point of the segment
    endpoints_coords = np.array([start_point, end_point])

    # Use the globally defined POINT styles for consistency, or define new ones
    plotter_instance.add_points(
        endpoints_coords,
        color=POINT_COLOR,  # e.g., '#ffffff'
        point_size=POINT_SIZE,  # e.g., 5
        render_points_as_spheres=True,  # Makes points more visible
        opacity=POINT_ALPHA,  # e.g., 0.9
        name=f"fieldline_endpoints_{index}",  # Unique name for the points actor
    )


def add_full_satellite_track(plotter_instance, track_data, track_index):
    """Adds a complete satellite track line to the plotter."""
    if track_data is None or track_data.shape[0] < 2:
        print(
            f"Warning: Skipping plotting track {track_index} due to insufficient points ({track_data.shape[0] if track_data is not None else 'None'})."
        )
        return
    points = track_data
    n_points = len(points)
    # Create lines array for PolyData: [2, p0, p1, 2, p1, p2, ...]
    lines_array = np.hstack([[2, k, k + 1] for k in range(n_points - 1)]).astype(int)
    track_poly = pv.PolyData(points, lines=lines_array)
    plotter_instance.add_mesh(
        track_poly,
        color=SAT_TRACK_COLOR,
        line_width=SAT_TRACK_WIDTH,
        opacity=SAT_TRACK_OPACITY,
        name=f"satellite_track_{track_index}",  # Unique name for the full track
    )


def add_satellite_markers(plotter_instance, marker_list):
    """Adds satellite markers to the plotter."""
    if not marker_list:
        return
    # Extract points from the list of dictionaries
    marker_points = np.array([m["point"] for m in marker_list])
    if marker_points.size > 0:
        plotter_instance.add_points(
            marker_points,
            color=SAT_POS_COLOR,
            point_size=SAT_POS_SIZE,
            render_points_as_spheres=True,
            opacity=POINT_ALPHA,
            name="satellite_markers",  # Single actor for all markers is fine
        )