import numpy as np
import matplotlib.pyplot as plt
import matplotlib  # 导入 matplotlib 库本身
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
# from PIL import Image # Optional

# --- 添加字体配置 ---
# 解决中文显示问题 (选择一个你系统上有的字体)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Source Han Sans CN'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- 常量与样式设定 (改进版) ---
R_E = 1.0
FIG_BG_COLOR = '#1c1c1c' # 更柔和的深灰色背景
AXES_BG_COLOR = '#1c1c1c' # 坐标系背景与图形背景一致
TEXT_COLOR = '#d0d0d0'   # 浅灰色文本，比纯白柔和
GRID_COLOR = '#404040'    # 深灰色网格线
GRID_ALPHA = 0.5          # 网格线透明度
GRID_LINESTYLE = ':'      # 网格线样式改为点线

EARTH_COLOR = '#678BB4'   # 柔和一点的蓝色
# EARTH_EDGE_COLOR = '#90b0d0' # 可选：地球边缘光晕色
EARTH_RESOLUTION_U = 180 # 提高地球表面分辨率
EARTH_RESOLUTION_V = 90

FIELD_LINE_COLOR = 'palegreen' # 使用柔和的淡绿色
# FIELD_LINE_COLOR = 'lightsteelblue' # 或者试试钢蓝色系
FIELD_LINE_ALPHA = 0.65   # 可以稍微增加一点不透明度
FIELD_LINE_WIDTH = 1.2    # 适当加粗磁力线

# 使用同一种颜色标记磁力线端点，简化图例
POINT_COLOR = 'skyblue'    # 使用天蓝色标记端点
POINT_SIZE = 15           # 减小标记尺寸
POINT_MARKER = 'o'        # 可以继续用圆点，或试试 '.'
POINT_ALPHA = 0.8         # 标记透明度

SAT_TRACK_COLOR = 'orange' # 使用橙色作为卫星轨道颜色
SAT_TRACK_WIDTH = 2.0     # 卫星轨道线宽
SAT_TRACK_ALPHA = 0.9     # 卫星轨道透明度

SAT_POS_COLOR = 'white'   # 使用白色标记卫星当前位置
SAT_POS_SIZE = 40         # 显著减小卫星标记尺寸
SAT_POS_MARKER = '*'      # 星号标记
SAT_POS_EDGECOLOR = 'red' # 可以给标记加个红色边缘，使其更突出但不刺眼

MAX_PLOT_RADIUS = 10.0 # 保持或调整视图范围

# --- 绘图函数 (plot_earth_shaded 稍作调整) ---
def plot_earth_shaded(ax, radius=R_E, color=EARTH_COLOR, edge_color=None, texture_path=None, light_source_angles=(315, 45), res_u=EARTH_RESOLUTION_U, res_v=EARTH_RESOLUTION_V):
    u = np.linspace(0, 2 * np.pi, res_u) # 使用更高分辨率
    v = np.linspace(0, np.pi, res_v)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ls = LightSource(azdeg=light_source_angles[0], altdeg=light_source_angles[1])
    from matplotlib.colors import ListedColormap, to_rgba

    # 使用 matplotlib.colormaps 获取 colormap (推荐)
    try:
        base_cmap = matplotlib.colormaps['Blues_r'] # 尝试新 API
    except AttributeError: # 兼容旧版本 matplotlib
        base_cmap = plt.cm.get_cmap('Blues_r')

    base_color_rgba = list(to_rgba(color))
    base_color_rgba[3] = 1.0
    # 调整颜色映射的亮度范围，可以使阴影更柔和
    shaded_cmap = ListedColormap(base_cmap(np.linspace(0.3, 1.0, 256)) * np.array(base_color_rgba) )

    rgb = ls.shade(z, cmap=shaded_cmap, vert_exag=0.1, blend_mode='soft')

    # 绘制表面，使用 antialiased=True 使边缘平滑
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, # r/cstride=1 匹配高分辨率
                           facecolors=rgb,
                           linewidth=0, antialiased=True, alpha=0.95, zorder=1)
    # if edge_color: # 边缘线有时会破坏平滑感，在高分辨率下可省略
    #      surf.set_edgecolor(edge_color)
    #      surf.set_linewidth(0.5)

# --- 1. 加载或准备你的数据 (保持不变) ---
# ... (你的数据加载代码) ...
# 示例占位数据 (你需要替换！)
num_lines = 50
magnetic_field_lines = []
for i in range(num_lines):
    if i % 5 == 0:
        theta = np.linspace(np.pi / 6, 5 * np.pi / 6, 150)
        r = 1.5 + 15 * np.sin(theta)**2
    else:
        theta = np.linspace(np.pi / 4, 3 * np.pi / 4, 100)
        r = 1.5 + 3 * np.sin(theta)**2
    phi = (i / num_lines) * 2 * np.pi
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
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
print("开始绘图 (改进版)...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor(FIG_BG_COLOR)
ax.set_facecolor(AXES_BG_COLOR)

# 绘制地球 (使用新设置)
plot_earth_shaded(ax, radius=R_E, color=EARTH_COLOR)
print("地球绘制完成.")

# 绘制磁力线 (数据截断逻辑不变)
print(f"绘制 {len(magnetic_field_lines)} 条磁力线 (截断在 {MAX_PLOT_RADIUS} RE)...")
all_endpoints = [] # 合并起点和终点
num_plotted_lines = 0
for i, line_data in enumerate(magnetic_field_lines):
    # ... (数据格式检查和转置不变) ...
    if line_data.shape[0] == 3 and line_data.shape[1] > 3: line_data = line_data.T
    elif line_data.shape[1] != 3: continue
    if line_data.shape[0] < 2: continue

    r = np.sqrt(np.sum(line_data**2, axis=1))
    mask = r <= MAX_PLOT_RADIUS
    plot_data = line_data[mask]

    if plot_data.shape[0] >= 2:
        ax.plot(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2],
                color=FIELD_LINE_COLOR, alpha=FIELD_LINE_ALPHA, linewidth=FIELD_LINE_WIDTH,
                zorder=2)
        # 记录端点
        all_endpoints.append(plot_data[0, :])
        all_endpoints.append(plot_data[-1, :])
        num_plotted_lines += 1
    elif plot_data.shape[0] == 1:
        all_endpoints.append(plot_data[0, :]) # 记录单个点
        num_plotted_lines += 1

# 统一绘制端点 (使用同一种颜色)
if all_endpoints:
    endpoints = np.array(all_endpoints)
    ax.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2],
               color=POINT_COLOR, marker=POINT_MARKER, s=POINT_SIZE, alpha=POINT_ALPHA,
               label=f'磁力线端点 ({num_plotted_lines}条)', zorder=4) # 修改图例标签
print("磁力线绘制完成.")

# 绘制卫星轨道 (数据截断逻辑不变)
print("绘制卫星轨道...")
if satellite_track.shape[0] > 1:
    # ... (数据格式检查和转置不变) ...
    if satellite_track.shape[0] == 3 and satellite_track.shape[1] > 3: satellite_track = satellite_track.T
    elif satellite_track.shape[1] != 3: print(f"警告: 卫星轨道数据形状错误")
    else:
        r_sat = np.sqrt(np.sum(satellite_track**2, axis=1))
        mask_sat = r_sat <= MAX_PLOT_RADIUS
        plot_sat_track = satellite_track[mask_sat]

        if plot_sat_track.shape[0] >= 2:
            ax.plot(plot_sat_track[:, 0], plot_sat_track[:, 1], plot_sat_track[:, 2],
                    color=SAT_TRACK_COLOR, linewidth=SAT_TRACK_WIDTH, alpha=SAT_TRACK_ALPHA,
                    label='卫星轨道', zorder=3)

        # 标记卫星当前位置 (使用新样式)
        if 0 <= satellite_current_pos_index < satellite_track.shape[0]:
            sat_pos = satellite_track[satellite_current_pos_index, :]
            r_sat_pos = np.sqrt(np.sum(sat_pos**2))
            if r_sat_pos <= MAX_PLOT_RADIUS:
                ax.scatter(sat_pos[0], sat_pos[1], sat_pos[2],
                           color=SAT_POS_COLOR, marker=SAT_POS_MARKER, s=SAT_POS_SIZE,
                           label='卫星位置示意', zorder=5, edgecolors=SAT_POS_EDGECOLOR, linewidths=0.5) # 添加边缘色
            # else: print(f"信息: 卫星位置超出范围")
        # else: print("警告: 卫星位置索引无效")
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

# 设置等比例坐标轴
limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
ax.set_box_aspect(np.ptp(limits, axis=1))

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

# 设置坐标轴线颜色 (可选，如果想要更柔和可以设为深灰)
# ax.xaxis.line.set_color(GRID_COLOR)
# ax.yaxis.line.set_color(GRID_COLOR)
# ax.zaxis.line.set_color(GRID_COLOR)
ax.xaxis.line.set_color(TEXT_COLOR) # 保持与刻度一致
ax.yaxis.line.set_color(TEXT_COLOR)
ax.zaxis.line.set_color(TEXT_COLOR)


# 设置网格线 (使用新样式)
ax.grid(True, color=GRID_COLOR, linestyle=GRID_LINESTYLE, linewidth=0.5, alpha=GRID_ALPHA)

# 添加图例 (修改标签，调整位置和样式)
legend = ax.legend(facecolor='#333333', edgecolor='#cccccc', framealpha=0.85, loc='upper left', bbox_to_anchor=(0.02, 0.98))
plt.setp(legend.get_texts(), color=TEXT_COLOR)

# 调整视角
ax.view_init(elev=25., azim=-110)

plt.tight_layout()

# --- 5. 保存图像 ---
output_filename_png = f'magnetosphere_plot_improved_{MAX_PLOT_RADIUS}RE.png'
plt.savefig(output_filename_png, dpi=300, facecolor=FIG_BG_COLOR, bbox_inches='tight')
print(f"图形已保存为 {output_filename_png}")
# plt.show()