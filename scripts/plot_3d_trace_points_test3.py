import numpy as np
import matplotlib.pyplot as plt
import matplotlib  # 导入 matplotlib 库本身
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
# from PIL import Image # Optional

# --- 添加字体配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 常量与样式设定 (再次改进) ---
R_E = 1.0
FIG_BG_COLOR = '#282c34'  # 更深的蓝灰色背景 (类似 IDE 暗色主题)
AXES_BG_COLOR = '#282c34' # 坐标系背景与图形背景一致
TEXT_COLOR = '#abb2bf'   # 稍亮的灰色文本 (类似 IDE 文本色)
AXIS_LINE_COLOR = '#4b5263' # 坐标轴线颜色

EARTH_COLOR = '#569cd6'   # 柔和的蓝色 (类似 VS Code 蓝)
EARTH_RESOLUTION_U = 180
EARTH_RESOLUTION_V = 90

# FIELD_LINE_COLOR = '#c678dd' # 柔和的紫色
FIELD_LINE_COLOR = '#e5c07b' # 柔和的金色/黄色
FIELD_LINE_ALPHA = 0.7
FIELD_LINE_WIDTH = 1.0    # 可以细一点

POINT_COLOR = FIELD_LINE_COLOR # 端点颜色与线条一致
POINT_SIZE = 8            # 更小的端点标记
POINT_MARKER = '.'         # 使用小点标记
POINT_ALPHA = 0.8

SAT_TRACK_COLOR = '#e06c75' # 柔和的红色/珊瑚色
SAT_TRACK_WIDTH = 1.8
SAT_TRACK_ALPHA = 0.9

SAT_POS_COLOR = '#ffffff'   # 白色标记
SAT_POS_SIZE = 30         # 更小的星号
SAT_POS_MARKER = '*'
SAT_POS_EDGECOLOR = SAT_TRACK_COLOR # 边缘用轨道颜色

MAX_PLOT_RADIUS = 10.0

# --- 绘图函数 (plot_earth_shaded 确保 alpha=1.0) ---
def plot_earth_shaded(ax, radius=R_E, color=EARTH_COLOR, edge_color=None, texture_path=None, light_source_angles=(315, 45), res_u=EARTH_RESOLUTION_U, res_v=EARTH_RESOLUTION_V):
    u = np.linspace(0, 2 * np.pi, res_u)
    v = np.linspace(0, np.pi, res_v)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ls = LightSource(azdeg=light_source_angles[0], altdeg=light_source_angles[1])
    from matplotlib.colors import ListedColormap, to_rgba
    try:
        base_cmap = matplotlib.colormaps['Blues_r']
    except AttributeError:
        base_cmap = plt.cm.get_cmap('Blues_r')

    base_color_rgba = list(to_rgba(color))
    base_color_rgba[3] = 1.0
    shaded_cmap = ListedColormap(base_cmap(np.linspace(0.3, 1.0, 256)) * np.array(base_color_rgba) )
    rgb = ls.shade(z, cmap=shaded_cmap, vert_exag=0.1, blend_mode='soft')

    # !!! 确保 alpha=1.0 完全不透明 !!!
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           facecolors=rgb,
                           linewidth=0, antialiased=True, alpha=1.0, zorder=1) # zorder=1 在最底层

# --- 辅助函数：寻找连续段落 ---
def find_contiguous_segments(mask):
    """根据布尔 mask 找到 True 的连续段落的起始和结束索引"""
    segments = []
    # 使用差分找到边界，+1 表示从 False 变为 True (段落开始)，-1 表示从 True 变为 False (段落结束)
    diff = np.diff(mask.astype(int), prepend=0, append=0) # 前后补0处理边界情况
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for start, end in zip(starts, ends):
        # end 索引是变为 False 的那个点，所以段落实际包含到 end-1
        segments.append((start, end)) # 返回 [start, end) 区间
    return segments

# --- 1. 加载或准备你的数据 (保持不变) ---
# ... (你的数据加载代码) ...
# 示例占位数据 (你需要替换！)
num_lines = 50
magnetic_field_lines = []
for i in range(num_lines):
    if i % 5 == 0:
        theta = np.linspace(np.pi / 6, 5 * np.pi / 6, 150)
        r_val = 1.5 + 15 * np.sin(theta)**2 # 模拟一些延伸较远的线
    else:
        theta = np.linspace(np.pi / 4, 3 * np.pi / 4, 100)
        r_val = 1.5 + 3 * np.sin(theta)**2 # 最高约 4.5 RE
    phi = (i / num_lines) * 2 * np.pi
    x = r_val * np.sin(theta) * np.cos(phi)
    y = r_val * np.sin(theta) * np.sin(phi)
    z = r_val * np.cos(theta)
    line_data = np.vstack((x, y, z)).T
    magnetic_field_lines.append(line_data)

theta_sat = np.linspace(0, 2 * np.pi, 200)
a_sat = 4 * R_E
e_sat = 0.1
x_sat = a_sat * (np.cos(theta_sat) - e_sat)
y_sat = a_sat * np.sqrt(1 - e_sat**2) * np.sin(theta_sat)
z_sat = 0.5 * R_E * np.sin(theta_sat*2)
satellite_track = np.vstack((x_sat, y_sat, z_sat)).T
satellite_current_pos_index = 50

# --- 3. 主绘图程序 ---
print("开始绘图 (再次改进)...")
fig = plt.figure(figsize=(12, 10)) # 可以尝试调整 figsize 比例，例如 (11, 10)
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor(FIG_BG_COLOR)
ax.set_facecolor(AXES_BG_COLOR)

# 绘制地球
plot_earth_shaded(ax, radius=R_E, color=EARTH_COLOR)
print("地球绘制完成.")

# 绘制磁力线 (!!! 实现分段绘制 !!!)
print(f"绘制 {len(magnetic_field_lines)} 条磁力线 (截断在 {MAX_PLOT_RADIUS} RE, 分段绘制)...")
all_segment_endpoints = [] # 记录所有绘制线段的端点
num_plotted_segments = 0
for i, line_data in enumerate(magnetic_field_lines):
    if line_data.shape[0] == 3 and line_data.shape[1] > 3: line_data = line_data.T
    elif line_data.shape[1] != 3: continue
    if line_data.shape[0] < 2: continue

    # 计算半径并生成 mask
    r = np.sqrt(np.sum(line_data**2, axis=1))
    mask = r <= MAX_PLOT_RADIUS

    # 找到所有在范围内的连续段落
    segments = find_contiguous_segments(mask)

    # 对每个段落分别绘制
    for start_idx, end_idx in segments:
        segment_data = line_data[start_idx:end_idx] # 提取段落数据
        if segment_data.shape[0] >= 2: # 至少需要两个点才能画线
            ax.plot(segment_data[:, 0], segment_data[:, 1], segment_data[:, 2],
                    color=FIELD_LINE_COLOR, alpha=FIELD_LINE_ALPHA, linewidth=FIELD_LINE_WIDTH,
                    zorder=2) # zorder=2 在地球之上
            # 记录该段的起点和终点
            all_segment_endpoints.append(segment_data[0, :])
            all_segment_endpoints.append(segment_data[-1, :])
            num_plotted_segments += 1
        elif segment_data.shape[0] == 1: # 如果段落只有一个点，也记录下来
             all_segment_endpoints.append(segment_data[0, :])
             num_plotted_segments += 1 # 算作一个点

# 统一绘制所有线段的端点
if all_segment_endpoints:
    endpoints = np.array(all_segment_endpoints)
    ax.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2],
               color=POINT_COLOR, marker=POINT_MARKER, s=POINT_SIZE, alpha=POINT_ALPHA,
               label=f'磁力线端点 ({len(magnetic_field_lines)}条线)', zorder=4) # 标签显示原始线条数
print("磁力线绘制完成.")

# 绘制卫星轨道 (同样可以应用分段逻辑，虽然轨道一般是连续的)
print("绘制卫星轨道...")
if satellite_track.shape[0] > 1:
    if satellite_track.shape[0] == 3 and satellite_track.shape[1] > 3: satellite_track = satellite_track.T
    elif satellite_track.shape[1] != 3: print(f"警告: 卫星轨道数据形状错误")
    else:
        r_sat = np.sqrt(np.sum(satellite_track**2, axis=1))
        mask_sat = r_sat <= MAX_PLOT_RADIUS
        sat_segments = find_contiguous_segments(mask_sat)
        plotted_sat_track = False
        for start_idx, end_idx in sat_segments:
             segment_data = satellite_track[start_idx:end_idx]
             if segment_data.shape[0] >= 2:
                 # 只给第一个绘制的轨道段添加标签
                 current_label = '卫星轨道' if not plotted_sat_track else None
                 ax.plot(segment_data[:, 0], segment_data[:, 1], segment_data[:, 2],
                         color=SAT_TRACK_COLOR, linewidth=SAT_TRACK_WIDTH, alpha=SAT_TRACK_ALPHA,
                         label=current_label, zorder=3)
                 plotted_sat_track = True

        # 标记卫星当前位置
        if 0 <= satellite_current_pos_index < satellite_track.shape[0]:
            sat_pos = satellite_track[satellite_current_pos_index, :]
            r_sat_pos = np.sqrt(np.sum(sat_pos**2))
            if r_sat_pos <= MAX_PLOT_RADIUS:
                ax.scatter(sat_pos[0], sat_pos[1], sat_pos[2],
                           color=SAT_POS_COLOR, marker=SAT_POS_MARKER, s=SAT_POS_SIZE,
                           label='卫星位置示意', zorder=5, edgecolors=SAT_POS_EDGECOLOR, linewidths=0.5)
else:
    print("警告: 卫星轨道数据点不足")
print("卫星轨道绘制完成.")


# --- 4. 精调外观与设置 ---

# 设置坐标轴标签
ax.set_xlabel('X [$R_E$]', color=TEXT_COLOR, fontsize=12)
ax.set_ylabel('Y [$R_E$]', color=TEXT_COLOR, fontsize=12)
ax.set_zlabel('Z [$R_E$]', color=TEXT_COLOR, fontsize=12)

# 设置标题
ax.set_title(f'地球磁场线与卫星轨道 (视图范围: {MAX_PLOT_RADIUS} $R_E$)', color=TEXT_COLOR, fontsize=14, weight='bold')

# 设置坐标轴范围
plot_limit = MAX_PLOT_RADIUS * 1.05
ax.set_xlim([-plot_limit, plot_limit])
ax.set_ylim([-plot_limit, plot_limit])
ax.set_zlim([-plot_limit, plot_limit])

# !!! 强制设置立方体比例，解决地球变形问题 !!!
ax.set_box_aspect([1,1,1]) # 关键：设置 XYZ 轴视觉比例为 1:1:1

# 设置刻度颜色
ax.tick_params(axis='x', colors=TEXT_COLOR)
ax.tick_params(axis='y', colors=TEXT_COLOR)
ax.tick_params(axis='z', colors=TEXT_COLOR)

# 设置坐标轴面板和边缘透明
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')

# 设置坐标轴线颜色
ax.xaxis.line.set_color(AXIS_LINE_COLOR)
ax.yaxis.line.set_color(AXIS_LINE_COLOR)
ax.zaxis.line.set_color(AXIS_LINE_COLOR)

# --- 移除网格线 ---
# ax.grid(True, color=GRID_COLOR, linestyle=GRID_LINESTYLE, linewidth=0.5, alpha=GRID_ALPHA)
ax.grid(False) # 明确禁用网格

# 添加图例 (使用新的颜色和样式)
legend = ax.legend(facecolor='#3c4049', edgecolor='#5c6370', framealpha=0.85, loc='upper left', bbox_to_anchor=(0.02, 0.98))
plt.setp(legend.get_texts(), color=TEXT_COLOR)

# 调整视角
ax.view_init(elev=25., azim=-110)

# plt.tight_layout() # tight_layout 有时会与 set_box_aspect 冲突，可以先注释掉观察效果

# --- 5. 保存图像 ---
output_filename_png = f'magnetosphere_plot_final_{MAX_PLOT_RADIUS}RE.png'
plt.savefig(output_filename_png, dpi=300, facecolor=FIG_BG_COLOR, bbox_inches='tight')
print(f"图形已保存为 {output_filename_png}")
# plt.show()