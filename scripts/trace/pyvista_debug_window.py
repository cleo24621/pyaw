import numpy as np
import pandas as pd
import pyvista as pv
from spacepy import coordinates as coord
from spacepy.time import Ticktock

# Explicitly set the backend
# Option A: Try the default VTK backend (often needs PyQt5)
try:
    # Setting QT_API environment variable might help sometimes
    # os.environ['QT_API'] = 'pyqt5' # Or 'pyside2' if you have that
    pv.set_jupyter_backend(None)  # Ensure it doesn't use a jupyter backend
    pv.global_theme.trame.server_proxy_enabled = False  # Disable trame just in case
    # Attempt to set a suitable backend explicitly or let PyVista choose default desktop
    # Forcing 'vtk' is not a direct API call, it's usually the default desktop backend
    # pv.rcParams['backend'] = 'vtk' # Setting rcParams might work in some versions
    print("Attempting to use default desktop (VTK/Qt) backend.")
except Exception as e:
    print(f"Could not configure backend, using PyVista default: {e}")

# Option B: If default/vtk fails, try 'panel' if installed
# try:
#    pv.set_jupyter_backend(None)
#    pv.set_plot_theme('panel') # Panel backend might have different behavior
#    pv.global_theme.trame.server_proxy_enabled = False
#    print("Attempting to use 'panel' backend.")
# except Exception as e:
#    print(f"Could not set 'panel' backend: {e}")


# --- 1. 加载或准备你的数据 ---
def geosph2geocar(
    times: np.ndarray, alts: np.ndarray, lats: np.ndarray, lons: np.ndarray
):
    """
    for points
    Args:
        times: element is np.datetime64
        alts: unit is Re
        lats: degree
        lons: ~

    Returns:
        'geo,car' Coords instance
    """
    arr_n_3 = np.column_stack((alts, lats, lons))
    c = coord.Coords(arr_n_3, "GEO", "sph", ticks=Ticktock(times, "ISO"))
    return c.convert("GEO", "car")


Re_km = 6371.2

# trace
file_path = r"G:\master\pyaw\scripts\results\aw_cases\archive\trace_points\end_trace_points_SwarmA_12728_case_20160301T014840_20160301T014900.pkl"
df_trace = pd.read_pickle(file_path)
df_trace_np = df_trace.values
df_trace_np_alt_re = df_trace_np.copy()
df_trace_np_alt_re[:, 0] = df_trace_np_alt_re[:, 0] + Re_km
df_trace_np_alt_re[:, 0] = df_trace_np_alt_re[:, 0] / Re_km
time_trace = np.datetime64("2016-03-01T01:49:00")  # the corresponding trace time
times_trace = np.array([str(time_trace) for _ in range(len(df_trace_np_alt_re[:, 0]))])
geo_car_trace = geosph2geocar(
    times_trace,
    df_trace_np_alt_re[:, 0],
    df_trace_np_alt_re[:, 1],
    df_trace_np_alt_re[:, 2],
)

# satellite
satellite_path = r"V:\aw\swarm\vires\measurements\SW_EXPT_EFIA_TCT16\SW_EXPT_EFIA_TCT16_12728_20160301T012924_20160301T030258.pkl"
df_sa = pd.read_pickle(satellite_path)
swarm_radius = 6816838.5 * 1e-3  # km
swarm_altitude = swarm_radius - Re_km  # km（初始高度）
lats_sa = df_sa["Latitude"].values
lons_sa = df_sa["Longitude"].values
# km convert to Re
alts_sa = np.full(len(lats_sa), swarm_altitude)
alts_sa = alts_sa + Re_km
alts_sa = alts_sa / Re_km
times_sa = df_sa.index.values
times_sa_str = np.array([str(t) for t in times_sa])
geo_car_sa = geosph2geocar(times_sa_str, alts_sa, lats_sa, lons_sa)

magnetic_field_lines = [geo_car_trace.data]  # units: Re,Re,Re
satellite_track = geo_car_sa.data
satellite_current_pos_index = 40000


# --- 辅助函数：寻找连续段落 (保持不变) ---
def find_contiguous_segments(mask):
    """根据布尔 mask 找到 True 的连续段落的起始和结束索引"""
    segments = []
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for start, end in zip(starts, ends):
        segments.append((start, end))
    return segments


# --- 颜色和样式设定 ---
R_E = 1.0
FIG_BG_COLOR = "#282c34"
TEXT_COLOR = "#abb2bf"
AXIS_LINE_COLOR = "#4b5263"

EARTH_COLOR = "#678094"  # 灰蓝色
EARTH_SPECULAR = 0.2
EARTH_SPECULAR_POWER = 10
EARTH_SMOOTH_SHADING = True

FIELD_LINE_COLOR = "#ff0000"  # 红色
FIELD_LINE_OPACITY = 0.65
FIELD_LINE_WIDTH = 1.5

POINT_COLOR = "#ffffff"  # 白色端点
POINT_SIZE = 6
POINT_ALPHA = 0.9

SAT_TRACK_COLOR = "#56b6c2"  # 青蓝色
SAT_TRACK_WIDTH = 2.5
SAT_TRACK_OPACITY = 0.9

SAT_POS_COLOR = "#ffffff"
SAT_POS_SIZE = 10
SAT_POS_EDGECOLOR = SAT_TRACK_COLOR  # 使用轨道颜色作为边缘

# 新增：地球特征颜色
POLE_MARKER_COLOR = "#ffffff"
POLE_MARKER_RADIUS = 0.03  # R_E units
AURORA_COLOR = "#98c379"  # 淡绿色
AURORA_OPACITY = 0.4

MAX_PLOT_RADIUS = 10.0
# ------------------------


# --- Helper function for showing steps ---
def show_current_plot(plotter, step_description):
    plotter.add_text(
        f"Current Step: {step_description}",
        position="lower_edge",
        font_size=10,
        color=TEXT_COLOR,
        name="step_text",
    )
    print(f"\n--- Showing Step: {step_description} ---")
    print(
        ">>> Adjust view if needed. Press 's' to save screenshot. Close window to continue. <<<"
    )  # 添加截图提示
    plotter.show()


# --- 更新绘图函数以包含新元素 ---
def add_earth_features(plotter_instance):
    # 地球本体
    earth = pv.Sphere(radius=R_E, theta_resolution=120, phi_resolution=120)
    plotter_instance.add_mesh(
        earth,
        color=EARTH_COLOR,
        smooth_shading=EARTH_SMOOTH_SHADING,
        specular=EARTH_SPECULAR,
        specular_power=EARTH_SPECULAR_POWER,
        show_edges=False,
    )
    # 南北极标记 (小球体)
    north_pole = pv.Sphere(radius=POLE_MARKER_RADIUS, center=(0, 0, R_E))
    south_pole = pv.Sphere(radius=POLE_MARKER_RADIUS, center=(0, 0, -R_E))
    plotter_instance.add_mesh(north_pole, color=POLE_MARKER_COLOR)
    plotter_instance.add_mesh(south_pole, color=POLE_MARKER_COLOR)
    # (可选) 添加文字标签 N/S
    plotter_instance.add_point_labels(
        [(0, 0, R_E * 1.1)],
        ["N"],
        point_size=10,
        font_size=12,
        text_color=TEXT_COLOR,
        shape=None,
    )
    plotter_instance.add_point_labels(
        [(0, 0, -R_E * 1.1)],
        ["S"],
        point_size=10,
        font_size=12,
        text_color=TEXT_COLOR,
        shape=None,
    )

    # 极光带示意环 (简化版 - 围绕地理极)
    # 参数需要调整以获得合适外观
    aurora_lat = 70  # 大致纬度 (度)
    aurora_radius_on_surface = R_E * np.cos(np.radians(aurora_lat))
    aurora_z = R_E * np.sin(np.radians(aurora_lat))
    aurora_thickness = 0.05  # 环的厚度 (R_E)

    # 北极光环
    north_aurora = pv.Cylinder(
        center=(0, 0, aurora_z),
        direction=(0, 0, 1),
        radius=aurora_radius_on_surface,
        height=aurora_thickness,
        capping=False,
        resolution=100,
    )
    # 南极光环
    south_aurora = pv.Cylinder(
        center=(0, 0, -aurora_z),
        direction=(0, 0, -1),
        radius=aurora_radius_on_surface,
        height=aurora_thickness,
        capping=False,
        resolution=100,
    )

    plotter_instance.add_mesh(
        north_aurora, color=AURORA_COLOR, opacity=AURORA_OPACITY, show_edges=False
    )
    plotter_instance.add_mesh(
        south_aurora, color=AURORA_COLOR, opacity=AURORA_OPACITY, show_edges=False
    )

    return plotter_instance


# --- 其他绘图函数 (add_fieldlines_and_points, add_satellite_track, apply_final_settings) ---
# 保持不变，或者根据新的颜色变量更新内部颜色设置（如果之前硬编码了的话）
# 例如，在 add_fieldlines_and_points 中确保使用新的 POINT_COLOR
# 在 add_satellite_track 中确保使用新的 SAT_POS_COLOR 和 SAT_POS_EDGECOLOR (add_points 不直接支持 edge color, 可忽略或用小球体代替)


def add_fieldlines_and_points(plotter_instance, lines_data, max_radius):
    all_endpoints = []
    print(f"Processing {len(lines_data)} magnetic field line(s)...")
    for i, line_data in enumerate(lines_data):
        if line_data is None or line_data.shape[0] < 2 or line_data.shape[1] != 3:
            continue
        r = np.linalg.norm(line_data, axis=1)
        mask = r <= max_radius
        segments = find_contiguous_segments(mask)
        for start_idx, end_idx in segments:
            segment_data = line_data[start_idx:end_idx]
            if segment_data.shape[0] >= 2:
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
                )  # Use new FIELD_LINE_... vars
                all_endpoints.append(segment_data[0, :])
                all_endpoints.append(segment_data[-1, :])
            elif segment_data.shape[0] == 1:
                all_endpoints.append(segment_data[0, :])
    if all_endpoints:
        endpoints_coords = np.array(all_endpoints)
        plotter_instance.add_points(
            endpoints_coords,
            color=POINT_COLOR,
            point_size=POINT_SIZE,  # Use new POINT_... vars
            render_points_as_spheres=True,
            opacity=POINT_ALPHA,
            label=f"磁力线端点 ({len(lines_data)}条)",
        )
    return plotter_instance


def add_satellite_track(plotter_instance, track_data, pos_index, max_radius):
    print("Processing satellite track...")
    plotted_sat_track = False
    if track_data is not None and track_data.shape[0] > 1 and track_data.shape[1] == 3:
        r_sat = np.linalg.norm(track_data, axis=1)
        mask_sat = r_sat <= max_radius
        sat_segments = find_contiguous_segments(mask_sat)
        for start_idx, end_idx in sat_segments:
            segment_data = track_data[start_idx:end_idx]
            if segment_data.shape[0] >= 2:
                points = segment_data
                n_points = len(points)
                lines_array = np.hstack(
                    [[2, k, k + 1] for k in range(n_points - 1)]
                ).astype(int)
                track_polydata = pv.PolyData(points, lines=lines_array)
                plotter_instance.add_mesh(
                    track_polydata,
                    color=SAT_TRACK_COLOR,
                    line_width=SAT_TRACK_WIDTH,  # Use new SAT_... vars
                    opacity=SAT_TRACK_OPACITY,
                    label="卫星轨道" if not plotted_sat_track else None,
                )
                plotted_sat_track = True
        if 0 <= pos_index < track_data.shape[0]:
            sat_pos = track_data[pos_index, :]
            if np.linalg.norm(sat_pos) <= max_radius:
                # Using add_points doesn't easily support edge color.
                # If edge color is crucial, replace with add_mesh(pv.Sphere(...))
                plotter_instance.add_points(
                    sat_pos,
                    color=SAT_POS_COLOR,
                    point_size=SAT_POS_SIZE,  # Use new SAT_POS_... vars
                    render_points_as_spheres=True,
                    label="卫星位置示意",
                )
        else:
            print("Satellite position index invalid or position out of range.")
    else:
        print("Satellite track data invalid or insufficient.")
    return plotter_instance


def apply_final_settings(plotter_instance, max_radius):
    print("Applying final settings (Camera, Axes, Title)...")
    try:
        default_cam_settings = plotter_instance.get_default_cam_pos()
        default_position = default_cam_settings[0]
        default_focal_point = default_cam_settings[1]
        default_viewup = default_cam_settings[2]
        dist_factor = 2.0
        new_position = tuple(coord * dist_factor for coord in default_position)
        plotter_instance.camera_position = [
            new_position,
            default_focal_point,
            default_viewup,
        ]
    except Exception as e:
        print(f"Warning: Could not set camera position - {e}.")
    # Ensure TEXT_COLOR is used for axes
    plotter_instance.add_axes(interactive=True, line_width=2, color=TEXT_COLOR)
    title = f"地球磁场线与卫星轨道 (视图范围: {max_radius} $R_E$)"
    # Ensure TEXT_COLOR is used for title
    plotter_instance.add_text(
        title, position="upper_edge", color=TEXT_COLOR, font_size=12, name="final_title"
    )
    return plotter_instance


# --- 2. 主绘图程序 (使用新的绘图函数和步骤) ---
print("开始使用 PyVista 绘图 (Step-by-Step with Enhancements)...")
pv.set_plot_theme("paraview")

# 增大窗口尺寸以获得更高分辨率的 's' 键截图
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200

# === STEP 1: Draw Earth with Features ===
plotter1 = pv.Plotter(window_size=[WINDOW_WIDTH, WINDOW_HEIGHT], off_screen=False)

# === 添加调试信息 ===
try:
    # 尝试打印交互渲染器的类型，这通常能反映后端
    print(f"Plotter Interactive Renderer Type: {type(plotter1.iren)}")
    # 也可以检查全局主题设置的后端（但这不总是最终使用的那个）
    print(f"PyVista Global Backend Theme: {pv.global_theme.backend}")
except Exception as e:
    print(f"Could not get detailed backend info: {e}")
# =====================

plotter1.set_background(FIG_BG_COLOR)
add_earth_features(plotter1)  # 使用新函数
show_current_plot(plotter1, "1. Earth with Features")
plotter1.close()
# ======================================

# === STEP 2: Earth + Field Lines & Endpoints ===
plotter2 = pv.Plotter(window_size=[WINDOW_WIDTH, WINDOW_HEIGHT], off_screen=False)
plotter2.set_background(FIG_BG_COLOR)
add_earth_features(plotter2)  # 重新绘制地球和特征
add_fieldlines_and_points(plotter2, magnetic_field_lines, MAX_PLOT_RADIUS)
show_current_plot(plotter2, "2. Field Lines & Endpoints")
plotter2.close()
# ====================================================

# === STEP 3: Earth + Lines/Points + Satellite ===
plotter3 = pv.Plotter(window_size=[WINDOW_WIDTH, WINDOW_HEIGHT], off_screen=False)
plotter3.set_background(FIG_BG_COLOR)
add_earth_features(plotter3)
add_fieldlines_and_points(plotter3, magnetic_field_lines, MAX_PLOT_RADIUS)
add_satellite_track(
    plotter3, satellite_track, satellite_current_pos_index, MAX_PLOT_RADIUS
)
show_current_plot(plotter3, "3. Satellite Track & Position")
plotter3.close()
# ====================================================

# === STEP 4: Final Plot with All Settings ===
plotter_final = pv.Plotter(window_size=[WINDOW_WIDTH, WINDOW_HEIGHT], off_screen=False)
plotter_final.set_background(FIG_BG_COLOR)
add_earth_features(plotter_final)
add_fieldlines_and_points(plotter_final, magnetic_field_lines, MAX_PLOT_RADIUS)
add_satellite_track(
    plotter_final, satellite_track, satellite_current_pos_index, MAX_PLOT_RADIUS
)
apply_final_settings(plotter_final, MAX_PLOT_RADIUS)

print("\n--- Showing Final Plot ---")
print(">>> Interact with the final plot. Adjust view as needed. <<<")
print(
    ">>> Press 's' key IN THE WINDOW to save a screenshot (check script directory). <<<"
)
print(">>> Close the window to exit the script. <<<")
plotter_final.remove_actor("step_text", render=False)  # 移除步骤文字
plotter_final.show()
# No automatic screenshot here, user saves manually via 's' key
plotter_final.close()
# =======================================

print("绘图完成.")
