import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly import express as px
import matplotlib.dates as mdates
import string  # To get lowercase letters
from numpy.typing import NDArray  # Keep this for NumPy arrays

# Import standard types from the 'typing' module
from typing import List, Dict, Any, Optional, Tuple, Union


def plot_xticks_of_times_lats_qdlats_mlts(
    datetimes: np.ndarray,
    values: np.ndarray,
    latitudes: np.ndarray,
    qdlats: np.ndarray,
    mlts: np.ndarray,
    step: int = 20000,
) -> tuple[plt.Figure, plt.Axes]:
    """
    e.g., plot_with_x_dt_lat_qdlat_mlt(df.index.values,bn_disturb,df['Latitude'].values,df_aux['QDLat'].values,df_aux['MLT'].values)
    :param datetimes:
    :param values:
    :param latitudes:
    :param qdlats: 地磁纬度
    :param mlts: 磁地方时
    :param step:
    :return:
    """
    # 创建图像
    fig, ax = plt.subplots(figsize=(18, 8))
    # 绘制数据
    ax.plot(datetimes, values, label="disturb magnetic field")
    datetime_ls = [
        np.datetime64("2015-12-31T23:06:00"),
        np.datetime64("2015-12-31T23:08:30"),
        np.datetime64("2015-12-31T23:20:00"),
        np.datetime64("2015-12-31T23:27:00"),
        np.datetime64("2015-12-31T23:53:00"),
        np.datetime64("2015-12-31T23:57:00"),
        np.datetime64("2016-01-01T00:04:00"),
        np.datetime64("2016-01-01T00:13:00"),
    ]
    for datetime_ in datetime_ls:
        plt.axvline(datetime_, color="r", linestyle="--")
    # plt.text(np.datetime64('2015-12-31T23:06'), max(values) * 0.9, f"{np.datetime64('2015-12-31T23:06')}", rotation=90, color='r', ha='right', va='top')
    # ax.set_ylabel('Value', color='b')
    # ax.tick_params(axis='y', labelcolor='b')

    # 设置时间轴标签
    datetime_ticks = datetimes[::step]
    latitude_ticks = latitudes[::step]
    qdlat_ticks = qdlats[::step]
    mlt_ticks = mlts[::step]
    ax.set_xticks(datetime_ticks)
    datetime_ticks_formatted = [
        t[11:19] for t in np.datetime_as_string(datetime_ticks, unit="s")
    ]
    ax.set_xticklabels(
        [
            (
                f"time: {t}\nlat: {lat:.2f}°\nqdlat: {qdlat:.2f}\nmlt: {mlt:.2f}"
                if i == 0
                else f"{t}\n{lat:.2f}\n{qdlat:.2f}\n{mlt:.2f}°"
            )
            for i, t, lat, qdlat, mlt in zip(
                range(len(datetime_ticks_formatted)),
                datetime_ticks_formatted,
                latitude_ticks,
                qdlat_ticks,
                mlt_ticks,
            )
        ]
    )

    return fig, ax


def plot_where_is(series):
    """
    :param series: pd.Series
    :return: None
    """
    plt.plot(series.isna(), marker=".", linestyle="None", color="red")
    plt.title("NaN Positions in the Series")
    plt.show()
    return None


def compare_before_after_interpolate(
    series_: pd.Series, method_="linear", figsize=(10, 6)
):
    """
    :param series_: the type of index is pd.datetime.
    :param method_:
    :param figsize:
    :return:
    """
    print(f"The number of NaN values: {series_.isna().sum()}")
    series_interpolate = series_.interpolate(method=method_)
    x = series_.index
    fig, axs = plt.subplots(3, figsize=figsize)
    axs[0].plot(
        x,
        series_,
    )
    axs[1].plot(
        x,
        series_interpolate,
    )
    axs[2].plot(x, series_, x)
    plt.show()


def plt_1f_2curve(x, y1, y2, title="", xlabel="", ylabel="", y1lable="", y2lable=""):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, label=y1lable)
    plt.plot(x, y2, label=y2lable)
    plt.legend()
    plt.show()


def plt_subplots(x, y1, y2, y3, y4) -> None:
    """
    plot a figure including 5 sub figures. the 1st sub figure on a line.
    :param x:
    :param y1:
    :param y2:
    :param y3:
    :param y4:
    :return:
    """
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)  # 3 行 2 列的网格
    # 0
    ax1 = fig.add_subplot(gs[0, :])  # 第 0 行，跨越所有列
    ax1.plot(x, y1)
    ax1.plot(x, y2)
    ax1.plot(x, y3)
    ax1.plot(x, y4)
    ax1.grid(which="both", linestyle="--", linewidth=0.5)

    # 1,0
    ax10 = fig.add_subplot(gs[1, 0])  # 第 1 行，第 0 列
    ax10.plot(x, y1)
    ax10.set_title("10 title")
    # 1,1
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(x, y2)
    # 2,0
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.plot(x, y3)
    # 2,1
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.plot(x, y4)
    # show figure
    plt.tight_layout()
    plt.show()
    return None


def plot_2signals_baselined(signal1: pd.Series, signal2: pd.Series, figsize=(10, 4)):
    assert all(
        signal1.index == signal2.index
    ), "signal1 and signal2 must have the same index"
    pass


def plt_mark_nan(series):
    plt.plot(series.isna(), marker=".", linestyle="None", color="red")
    plt.title("NaN Positions in b_baselined")
    plt.show()


def px_1f_2curve(x, y1, y2, title=""):
    # todo:: add code for real-time display of adjustment parameters
    df = pd.DataFrame({"x": x, "y1": y1, "y2": y2})
    fig = px.line(df, x="x", y=["y1", "y2"], title=title)
    fig.show()


def px_beta(beta, me, mi):
    df = pd.DataFrame({"x": beta.index, "y": beta.values})
    fig = px.line(df, x="x", y="y", title="Plasma beta")
    # add specific value horizontal line
    specific_v = me / mi
    fig.add_shape(
        type="line",
        x0=df["x"].min(),
        y0=specific_v,
        x1=df["x"].max(),
        y1=specific_v,
        line=dict(color="Red", width=2, dash="dash"),  # type of line
    )
    # add annotation
    fig.add_annotation(
        x=df["x"].max(),
        y=specific_v,
        text=f"$m_e/m_i$={specific_v}",
        showarrow=False,
        yshift=10,
    )
    # title
    fig.update_layout(
        xaxis_title="Time (UT)",
        yaxis_title="$\\Beta$",
    )
    fig.show()


# def plot_multi_panel_aligned(
#     subplot_definitions: list[dict],
#     figsize: tuple = None,
#     width_ratios: list = [15, 1],
#     global_cmap: str = 'turbo',
#     use_shared_clims: bool = True,
#     clim_percentiles: tuple = (1.0, 99.0),
#     global_vmin: float = None,
#     global_vmax: float = None,
#     log_scale_z: bool = True,
#     log_epsilon: float = 1e-15,
#     use_grid: bool = True,
#     figure_title: str = None,
#     xlabel: str = "Time (UTC)",
#     date_format_str: str = None,
#     rotate_xticklabels: float = 0
# ):
#     """
#     Creates a multi-panel plot with vertically stacked subplots, sharing a common
#     time axis. Supports line plots and pcolormesh plots with aligned colorbars.
#
#     Args:
#         subplot_definitions (list[dict]): A list where each dictionary defines one
#             subplot from top to bottom. Required keys depend on 'plot_type':
#             - Common keys: 'plot_type' ('line' or 'pcolormesh'), 'title' (str), 'ylabel' (str).
#             - For 'line':
#                 'x_data' (np.ndarray), 'y_data' (np.ndarray),
#                 'label' (str, optional), 'color' (str, optional),
#                 'linewidth' (float, optional).
#             - For 'pcolormesh':
#                 'x_data' (np.ndarray), 'y_data' (np.ndarray), 'z_data' (np.ndarray),
#                 'clabel' (str, optional), 'cmap' (str, optional),
#                 'vmin' (float, optional), 'vmax' (float, optional),
#                 'shading' (str, optional): Shading mode ('auto', 'flat', 'gouraud'). Defaults to 'auto'.
#
#         figsize (tuple, optional): Overall figure size (width, height). Defaults to calculated.
#         width_ratios (list, optional): Ratio [main_plot, cbar]. Defaults to [15, 1].
#         global_cmap (str, optional): Default cmap for pcolormesh. Defaults to 'turbo'.
#         use_shared_clims (bool, optional): Calc shared vmin/vmax. Defaults to True.
#         clim_percentiles (tuple, optional): Percentiles for shared limits. Defaults to (1.0, 99.0).
#         global_vmin (float, optional): Force global min color limit. Defaults to None.
#         global_vmax (float, optional): Force global max color limit. Defaults to None.
#         log_scale_z (bool, optional): Apply 10*log10 to pcolormesh z_data. Defaults to True.
#         log_epsilon (float, optional): Epsilon for log10. Defaults to 1e-15.
#         use_grid (bool, optional): Display grid lines. Defaults to True.
#         figure_title (str, optional): Overall figure title. Defaults to None.
#         xlabel (str, optional): Shared x-axis label. Defaults to "Time (UTC)".
#         date_format_str (str, optional): Specific x-axis date format string. Defaults to None.
#         rotate_xticklabels (float, optional): Rotation angle for x-tick labels. Defaults to 0.
#
#     Returns:
#         tuple[plt.Figure, np.ndarray[plt.Axes]]: Figure and axes array.
#
#     Raises:
#         ValueError: If subplot_definitions is invalid.
#     """
#     num_plots = len(subplot_definitions)
#     if num_plots == 0:
#         raise ValueError("subplot_definitions list cannot be empty.")
#
#     # --- Calculate Figure Size if not provided ---
#     if figsize is None:
#         fig_height_per_plot = 2.2 # Base height per plot
#         # Increase base height slightly if titles/labels might be long
#         fig_height = fig_height_per_plot * num_plots + 1.0 # Add extra inch for spacing
#         fig_width = 13
#         figsize = (fig_width, fig_height)
#         print(f"Figure size not specified, using calculated figsize={figsize}")
#
#     # --- Create Figure and Axes Grid ---
#     fig, axes = plt.subplots(
#         nrows=num_plots, ncols=2, sharex='col', figsize=figsize,
#         layout='constrained', gridspec_kw={'width_ratios': width_ratios}
#     )
#     if num_plots == 1: axes = axes.reshape(1, 2)
#
#     # --- Determine Color Limits (if needed - same logic as before) ---
#     calc_vmin, calc_vmax = None, None
#     if global_vmin is None and global_vmax is None and use_shared_clims:
#         # ...(calculation logic remains the same)...
#         all_z_data_for_clim = []
#         for i, plot_info in enumerate(subplot_definitions):
#              if plot_info.get('plot_type') == 'pcolormesh':
#                 z_data = plot_info.get('z_data')
#                 if z_data is not None:
#                     if log_scale_z: data_to_calc = 10 * np.log10(np.maximum(np.abs(z_data), log_epsilon))
#                     else: data_to_calc = z_data
#                     if np.any(np.isfinite(data_to_calc)): all_z_data_for_clim.append(data_to_calc[np.isfinite(data_to_calc)].flatten())
#         if all_z_data_for_clim:
#             concatenated_data = np.concatenate(all_z_data_for_clim)
#             if concatenated_data.size > 0:
#                 try:
#                     calc_vmin = np.percentile(concatenated_data, clim_percentiles[0]); calc_vmax = np.percentile(concatenated_data, clim_percentiles[1])
#                     print(f" Calculated shared vmin={calc_vmin:.2f}, vmax={calc_vmax:.2f}")
#                 except IndexError: calc_vmin, calc_vmax = None, None; print("Warning: Could not calculate percentiles.")
#             else: calc_vmin, calc_vmax = None, None; print("Warning: No valid data for shared limits.")
#         else: calc_vmin, calc_vmax = None, None; print("Warning: No pcolormesh data for shared limits.")
#
#
#     # --- Store all x-data to determine overall limits later ---
#     all_x_data = []
#
#     # --- Plotting Loop ---
#     for i, plot_info in enumerate(subplot_definitions):
#         ax_main = axes[i, 0]
#         ax_cbar = axes[i, 1]
#         plot_type = plot_info.get('plot_type')
#
#         if use_grid: ax_main.grid(True, linestyle='--', alpha=0.6)
#
#         if plot_type == 'line':
#             # ...(line plot logic remains the same)...
#             x_data = plot_info.get('x_data'); y_data = plot_info.get('y_data')
#             if x_data is None or y_data is None: raise ValueError(f"Missing data for line subplot {i}")
#             all_x_data.append(x_data)
#             ax_main.plot(x_data, y_data, label=plot_info.get('label'), color=plot_info.get('color'), linewidth=plot_info.get('linewidth', 1.5))
#             if plot_info.get('label'): ax_main.legend(loc='best', fontsize='small')
#             ax_cbar.axis('off')
#
#         elif plot_type == 'pcolormesh':
#             x_data = plot_info.get('x_data'); y_data = plot_info.get('y_data'); z_data = plot_info.get('z_data')
#             if x_data is None or y_data is None or z_data is None: raise ValueError(f"Missing data for pcolormesh subplot {i}")
#             # ...(shape warning remains the same)...
#             if z_data.shape[1] != len(x_data) or z_data.shape[0] != len(y_data): print(f"Warning: z_data shape {z_data.shape} vs x({len(x_data)})/y({len(y_data)}) for subplot {i}.")
#
#             all_x_data.append(x_data)
#
#             if log_scale_z: plot_z_data = 10 * np.log10(np.maximum(np.abs(z_data), log_epsilon)); effective_clabel = plot_info.get('clabel', "Value (dB)")
#             else: plot_z_data = z_data; effective_clabel = plot_info.get('clabel', "Value")
#
#             vmin_plot = global_vmin if global_vmin is not None else plot_info.get('vmin', calc_vmin)
#             vmax_plot = global_vmax if global_vmax is not None else plot_info.get('vmax', calc_vmax)
#
#             # *** MODIFICATION: Get shading mode ***
#             shading_mode = plot_info.get('shading', 'auto') # Get shading or default
#
#             im = ax_main.pcolormesh(
#                 x_data, y_data, plot_z_data,
#                 cmap=plot_info.get('cmap', global_cmap),
#                 shading=shading_mode, # *** USE SHADING MODE HERE ***
#                 vmin=vmin_plot,
#                 vmax=vmax_plot
#             )
#             cbar = fig.colorbar(im, cax=ax_cbar)
#             cbar.set_label(effective_clabel)
#
#         else:
#             raise ValueError(f"Unknown 'plot_type': {plot_type} at index {i}. Use 'line' or 'pcolormesh'.")
#
#         # --- Set Common Labels/Titles (remains the same) ---
#         ax_main.set_title(plot_info.get('title', f'Subplot {i+1}'))
#         ax_main.set_ylabel(plot_info.get('ylabel', ''))
#         if i < num_plots - 1: ax_main.tick_params(labelbottom=False)
#
#     # --- Configure Final X-axis (remains the same) ---
#     if all_x_data:
#         ax_last_main = axes[num_plots - 1, 0]; ax_last_main.set_xlabel(xlabel)
#         try:
#             min_time = min(np.min(arr) for arr in all_x_data if arr.size > 0); max_time = max(np.max(arr) for arr in all_x_data if arr.size > 0)
#             ax_last_main.set_xlim(min_time, max_time)
#         except (ValueError, TypeError) as e: print(f"Warning: Could not set shared x-limits: {e}")
#         if date_format_str: formatter = mdates.DateFormatter(date_format_str)
#         else: locator = mdates.AutoDateLocator(minticks=6, maxticks=10); formatter = mdates.ConciseDateFormatter(locator); ax_last_main.xaxis.set_major_locator(locator)
#         ax_last_main.xaxis.set_major_formatter(formatter)
#         if rotate_xticklabels != 0: plt.setp(ax_last_main.get_xticklabels(), rotation=rotate_xticklabels, ha='right')
#
#     # --- Add Figure Title (remains the same) ---
#     if figure_title: fig.suptitle(figure_title, fontsize=16)
#
#     return fig, axes


# def plot_multi_panel_aligned(
#     subplot_definitions: list[dict],
#     # --- X-axis Multi-label Data (Optional) ---
#     x_datetime_ref: np.ndarray = None,
#     x_aux_data: dict[str, np.ndarray] = None,
#     x_label_step: int = None,
#     # --- General Plotting Options ---
#     figsize: tuple = None,
#     width_ratios: list = [15, 1],
#     global_cmap: str = 'viridis',
#     use_shared_clims: bool = True,
#     clim_percentiles: tuple = (1.0, 99.0),
#     global_vmin: float = None,
#     global_vmax: float = None,
#     use_grid: bool = True,
#     figure_title: str = None,
#     xlabel: str = None,
#     date_format_str: str = None,
#     rotate_xticklabels: float = 0,
#     # --- Font Size Control ---
#     base_fontsize: float = 10,
#     title_fontsize: float = 12,
#     label_fontsize: float = 10,
#     tick_label_fontsize: float = 9,
#     legend_fontsize: float = 9,
#     annotation_fontsize: float = 8,
#     panel_label_fontsize: float = 12, # Fontsize for (a), (b)...
#     # --- Annotation Style Defaults (Optional) ---
#     vline_color: str = 'red',
#     vline_linestyle: str = '--',
#     vline_label_color: str = 'red',
#     block_label_color: str = None
# ):
#     """
#     Creates a multi-panel plot with aligned axes, subplot labels (a, b, ...),
#     multi-line x-axis labels, subplot-specific annotations, and font size control.
#
#     Args:
#         subplot_definitions (list[dict]): Defines subplots.
#         x_datetime_ref (np.ndarray, optional): Ref times for multi-labels.
#         x_aux_data (dict[str, np.ndarray], optional): Aux data for multi-labels.
#         x_label_step (int, optional): Thinning step for multi-labels.
#         figsize (tuple, optional): Figure size.
#         width_ratios (list, optional): Ratio [main_plot, cbar]. Defaults [15, 1].
#         global_cmap (str, optional): Default cmap. Defaults to 'viridis'.
#         use_shared_clims (bool, optional): Calc shared vmin/vmax.
#         clim_percentiles (tuple, optional): Percentiles for shared clim.
#         global_vmin (float, optional): Force global min color limit.
#         global_vmax (float, optional): Force global max color limit.
#         use_grid (bool, optional): Display grid lines.
#         figure_title (str, optional): Overall figure title.
#         xlabel (str, optional): Shared x-axis label.
#         date_format_str (str, optional): x-axis date format string.
#         rotate_xticklabels (float, optional): Rotation for x-tick labels.
#         base_fontsize (float, optional): Base font size.
#         title_fontsize (float, optional): Font size for subplot titles.
#         label_fontsize (float, optional): Font size for axis/cbar labels.
#         tick_label_fontsize (float, optional): Font size for tick labels.
#         legend_fontsize (float, optional): Font size for legends.
#         annotation_fontsize (float, optional): Font size for block/vline text labels.
#         panel_label_fontsize (float, optional): Font size for (a), (b)... labels.
#         vline_color, vline_linestyle, vline_label_color: Vline style defaults.
#         block_label_color: Block label color default.
#
#     Returns:
#         tuple[plt.Figure, np.ndarray[plt.Axes]]: Figure and axes array.
#     """
#     num_plots = len(subplot_definitions)
#     if num_plots == 0: raise ValueError("subplot_definitions list empty.")
#
#     if figsize is None:
#         fig_height = (2.4 * num_plots) + 1.5; fig_width = 13
#         figsize = (fig_width, fig_height)
#
#     fig, axes = plt.subplots(
#         nrows=num_plots, ncols=2, sharex='col', figsize=figsize,
#         layout='constrained', gridspec_kw={'width_ratios': width_ratios}
#     )
#     if num_plots == 1: axes = axes.reshape(1, 2)
#
#     # --- Determine Color Limits (if needed - raw z_data) ---
#     calc_vmin, calc_vmax = None, None
#     if global_vmin is None and global_vmax is None and use_shared_clims:
#         # ...(calculation logic remains the same)...
#         all_z_data=[];
#         for info in subplot_definitions:
#              if info.get('plot_type') == 'pcolormesh' and info.get('z_data') is not None:
#                 z=info['z_data'];
#                 if np.any(np.isfinite(z)): all_z_data.append(z[np.isfinite(z)].flatten());
#         if all_z_data:
#             cat=np.concatenate(all_z_data);
#             if cat.size > 0:
#                 try: calc_vmin, calc_vmax = np.percentile(cat, clim_percentiles); print(f" Calc shared vmin={calc_vmin:.2f}, vmax={calc_vmax:.2f} (raw data)")
#                 except IndexError: print("Warn: Bad clim percentiles."); calc_vmin, calc_vmax = None, None
#             else: print("Warn: No valid data for shared clim."); calc_vmin, calc_vmax = None, None
#         else: print("Warn: No pcolormesh for shared clim."); calc_vmin, calc_vmax = None, None
#
#     all_x_data = []
#
#     # --- Plotting Loop ---
#     panel_labels = string.ascii_lowercase # Get 'a', 'b', 'c', ...
#     for i, plot_info in enumerate(subplot_definitions):
#         ax_main = axes[i, 0]; ax_cbar = axes[i, 1]
#         plot_type = plot_info.get('plot_type')
#         if use_grid: ax_main.grid(True, linestyle='--', alpha=0.6)
#
#         x_data = plot_info.get('x_data');
#         if x_data is None: raise ValueError(f"Missing 'x_data' for subplot {i}")
#         all_x_data.append(x_data)
#
#         # Plot main content (Line or Pcolormesh - same logic)
#         if plot_type == 'line':
#             y_data=plot_info.get('y_data'); ax_cbar.axis('off');
#             if y_data is None: raise ValueError(f"Missing 'y_data' for line {i}")
#             lw = plot_info.get('linewidth', 1.5)
#             ax_main.plot(x_data, y_data, label=plot_info.get('label'), color=plot_info.get('color'), linewidth=lw);
#             if plot_info.get('label'): ax_main.legend(loc='best', fontsize=legend_fontsize)
#         elif plot_type == 'pcolormesh':
#             y_data=plot_info.get('y_data'); z_data=plot_info.get('z_data');
#             if y_data is None or z_data is None: raise ValueError(f"Missing data for pcolormesh {i}")
#             plot_z_data = z_data
#             clabel=plot_info.get('clabel', 'Value'); vmin_plot=global_vmin if global_vmin is not None else plot_info.get('vmin', calc_vmin); vmax_plot=global_vmax if global_vmax is not None else plot_info.get('vmax', calc_vmax); shading=plot_info.get('shading', 'auto'); cmap=plot_info.get('cmap', global_cmap);
#             im=ax_main.pcolormesh(x_data, y_data, plot_z_data, cmap=cmap, shading=shading, vmin=vmin_plot, vmax=vmax_plot);
#             cbar=fig.colorbar(im, cax=ax_cbar); cbar.set_label(clabel, size=label_fontsize); cbar.ax.tick_params(labelsize=tick_label_fontsize)
#         else: raise ValueError(f"Unknown plot_type: {plot_type}")
#
#         # --- Add Annotations: Blocks or Vlines ---
#         subplot_blocks = plot_info.get('blocks', [])
#         subplot_vlines = plot_info.get('vlines')
#         if subplot_blocks: # Use truthiness of list
#             for block in subplot_blocks:
#                 start=block.get('start'); end=block.get('end'); color=block.get('color')
#                 if start is not None and end is not None and color is not None:
#                     alpha=block.get('alpha', 0.2); ax_main.axvspan(start, end, color=color, alpha=alpha, zorder=-10)
#                     label=block.get('label');
#                     if label:
#                         start_num=mdates.date2num(start); end_num=mdates.date2num(end); mid_dt=mdates.num2date(start_num + (end_num - start_num) / 2); y_txt=ax_main.get_ylim()[0] + (ax_main.get_ylim()[1] - ax_main.get_ylim()[0]) * 0.95; lbl_color = block_label_color if block_label_color is not None else color; ax_main.text(mid_dt, y_txt, label, ha='center', va='top', fontsize=annotation_fontsize, color=lbl_color, clip_on=True, fontweight='bold')
#         elif isinstance(subplot_vlines, dict):
#             for label, dt_value in subplot_vlines.items():
#                  if dt_value is not None:
#                     ax_main.axvline(dt_value, color=vline_color, linestyle=vline_linestyle, alpha=0.7, lw=1)
#                     y_txt = ax_main.get_ylim()[0] + (ax_main.get_ylim()[1] - ax_main.get_ylim()[0]) * 0.95
#                     ax_main.text(dt_value, y_txt, f" {label}", rotation=90, color=vline_label_color, ha='left', va='top', fontsize=annotation_fontsize, clip_on=True)
#
#         # --- Add Panel Label (a), (b), ... ---
#         panel_label = f"({panel_labels[i]})"
#         # Position in top-left corner using axes coordinates (0,0 is bottom-left, 1,1 is top-right)
#         ax_main.text(0.01, 0.98, panel_label, transform=ax_main.transAxes,
#                      fontsize=panel_label_fontsize, fontweight='bold', va='top', ha='left')
#
#         # Set Title/YLabel/Ticks using specified font sizes
#         ax_main.set_title(plot_info.get('title', f'Subplot {i+1}'), fontsize=title_fontsize)
#         ax_main.set_ylabel(plot_info.get('ylabel', ''), fontsize=label_fontsize)
#         ax_main.tick_params(axis='y', labelsize=tick_label_fontsize)
#         if i < num_plots - 1: ax_main.tick_params(labelbottom=False)
#         else: ax_main.tick_params(axis='x', labelsize=tick_label_fontsize)
#
#
#     # --- Configure Final X-axis (same logic) ---
#     ax_last_main = axes[num_plots - 1, 0]; final_xlabel = xlabel
#     use_multi_labels = (x_datetime_ref is not None and x_aux_data is not None and x_label_step is not None)
#     if use_multi_labels:
#         if final_xlabel is None: final_xlabel = f"Time (UTC) / {' / '.join(x_aux_data.keys())}"
#         dt_ticks = x_datetime_ref[::x_label_step]; aux_ticks = {key: np.asarray(arr)[::x_label_step] for key, arr in x_aux_data.items()}; time_labels = [np.datetime_as_string(t, unit='s')[11:19] for t in dt_ticks]; xticklabels = ["\n".join([time_labels[i]] + [f"{aux_ticks[key][i]:.1f}" for key in x_aux_data.keys()]) for i in range(len(dt_ticks))]; ax_last_main.set_xticks(dt_ticks); ax_last_main.set_xticklabels(xticklabels, rotation=rotate_xticklabels, ha='right' if rotate_xticklabels!=0 else 'center', fontsize=tick_label_fontsize) # Use tick_label_fontsize
#     else: # Standard formatting
#         if final_xlabel is None: final_xlabel = "Time (UTC)";
#         if date_format_str: formatter = mdates.DateFormatter(date_format_str)
#         else: locator = mdates.AutoDateLocator(minticks=6, maxticks=10); formatter = mdates.ConciseDateFormatter(locator); ax_last_main.xaxis.set_major_locator(locator)
#         ax_last_main.xaxis.set_major_formatter(formatter)
#         if rotate_xticklabels != 0: plt.setp(ax_last_main.get_xticklabels(), rotation=rotate_xticklabels, ha='right')
#     ax_last_main.set_xlabel(final_xlabel, fontsize=label_fontsize)
#     if all_x_data:
#         try: min_t=min(np.min(arr) for arr in all_x_data if arr.size > 0); max_t=max(np.max(arr) for arr in all_x_data if arr.size > 0); ax_last_main.set_xlim(min_t, max_t)
#         except Exception as e: print(f"Warn: Could not set x-limits: {e}")
#
#     if figure_title: fig.suptitle(figure_title, fontsize=title_fontsize + 2)
#
#     return fig, axes


def plot_multi_panel(
    subplot_definitions: list[dict],
    # --- X-axis Multi-label Data (Optional) ---
    x_datetime_ref: np.ndarray = None,
    x_aux_data: dict[str, np.ndarray] = None,
    x_label_step: int = None,
    # --- General Plotting Options ---
    figsize: tuple = None,
    width_ratios: list = [15, 1],
    global_cmap: str = "viridis",
    use_shared_clims: bool = True,
    clim_percentiles: tuple = (1.0, 99.0),
    global_vmin: float = None,
    global_vmax: float = None,
    use_grid: bool = True,
    figure_title: str = None,
    xlabel: str = None,
    date_format_str: str = None,
    rotate_xticklabels: float = 0,
    # --- Font Size Control ---
    base_fontsize: float = 10,
    title_fontsize: float = 12,
    label_fontsize: float = 10,
    tick_label_fontsize: float = 9,
    legend_fontsize: float = 9,
    annotation_fontsize: float = 8,
    panel_label_fontsize: float = 12,
    # --- Annotation Style Defaults ---
    vline_color: str = "red",
    vline_linestyle: str = "--",
    vline_label_color: str = "red",
    block_label_color: str = None,  # None uses block color
    hline_color: str = "blue",  # Default Hline color
    hline_linestyle: str = ":",  # Default Hline style
    hline_label_color: str = "blue",  # Default Hline label color
):
    """
    Creates a multi-panel plot with aligned axes, subplot labels (a, b, ...),
    multi-line x-axis labels, subplot-specific annotations (blocks/vlines/hlines),
    and font size control.

    Args:
        subplot_definitions (list[dict]): Defines subplots. Keys include:
            'plot_type', 'title', 'ylabel', 'x_data', 'y_data',
            ['z_data', 'clabel', 'cmap', 'vmin', 'vmax', 'shading'],
            ['label', 'color', 'linewidth'],
            ['blocks': list[dict] {'start', 'end', 'color', ['label', 'alpha']}],
            ['vlines': dict {'Label': datetime64}],
            ['hlines': list[dict] {'y', 'color', 'linestyle', ['label']}]. # NEW
        x_datetime_ref (np.ndarray, optional): Ref times for multi-labels.
        x_aux_data (dict[str, np.ndarray], optional): Aux data for multi-labels.
        x_label_step (int, optional): Thinning step for multi-labels.
        figsize (tuple, optional): Figure size.
        width_ratios (list, optional): Ratio [main_plot, cbar]. Defaults [15, 1].
        global_cmap (str, optional): Default cmap. Defaults to 'viridis'.
        use_shared_clims (bool, optional): Calc shared vmin/vmax.
        clim_percentiles (tuple, optional): Percentiles for shared clim.
        global_vmin (float, optional): Force global min color limit.
        global_vmax (float, optional): Force global max color limit.
        use_grid (bool, optional): Display grid lines.
        figure_title (str, optional): Overall figure title.
        xlabel (str, optional): Shared x-axis label.
        date_format_str (str, optional): x-axis date format string.
        rotate_xticklabels (float, optional): Rotation for x-tick labels.
        base_fontsize ... panel_label_fontsize: Font size controls.
        vline_..., hline_..., block_...: Annotation style defaults.

    Returns:
        tuple[plt.Figure, np.ndarray[plt.Axes]]: Figure and axes array.
    """
    num_plots = len(subplot_definitions)
    if num_plots == 0:
        raise ValueError("subplot_definitions list empty.")

    if figsize is None:
        fig_height = (2.4 * num_plots) + 1.5
        fig_width = 13
        figsize = (fig_width, fig_height)

    fig, axes = plt.subplots(
        nrows=num_plots,
        ncols=2,
        sharex="col",
        figsize=figsize,
        layout="constrained",
        gridspec_kw={"width_ratios": width_ratios},
    )
    if num_plots == 1:
        axes = axes.reshape(1, 2)

    # --- Determine Color Limits (if needed - raw z_data - same logic) ---
    calc_vmin, calc_vmax = None, None
    if global_vmin is None and global_vmax is None and use_shared_clims:
        all_z_data = []
        for info in subplot_definitions:
            if info.get("plot_type") == "pcolormesh" and info.get("z_data") is not None:
                z = info["z_data"]
                if np.any(np.isfinite(z)):
                    all_z_data.append(z[np.isfinite(z)].flatten())
        if all_z_data:
            cat = np.concatenate(all_z_data)
            if cat.size > 0:
                try:
                    calc_vmin, calc_vmax = np.percentile(cat, clim_percentiles)
                    print(
                        f" Calc shared vmin={calc_vmin:.2f}, vmax={calc_vmax:.2f} (raw data)"
                    )
                except IndexError:
                    print("Warn: Bad clim percentiles.")
                    calc_vmin, calc_vmax = None, None
            else:
                print("Warn: No valid data for shared clim.")
                calc_vmin, calc_vmax = None, None
        else:
            print("Warn: No pcolormesh for shared clim.")
            calc_vmin, calc_vmax = None, None

    all_x_data = []

    # --- Plotting Loop ---
    panel_labels = string.ascii_lowercase
    for i, plot_info in enumerate(subplot_definitions):
        ax_main = axes[i, 0]
        ax_cbar = axes[i, 1]
        plot_type = plot_info.get("plot_type")
        if use_grid:
            ax_main.grid(True, linestyle="--", alpha=0.6)

        x_data = plot_info.get("x_data")
        if x_data is None:
            raise ValueError(f"Missing 'x_data' for subplot {i}")
        all_x_data.append(x_data)

        # Plot main content (Line or Pcolormesh - same logic)
        if plot_type == "line":
            y_data = plot_info.get("y_data")
            ax_cbar.axis("off")
            if y_data is None:
                raise ValueError(f"Missing 'y_data' for line {i}")
            lw = plot_info.get("linewidth", 1.5)
            ax_main.plot(
                x_data,
                y_data,
                label=plot_info.get("label"),
                color=plot_info.get("color"),
                linewidth=lw,
            )
            if plot_info.get("label"):
                ax_main.legend(loc="best", fontsize=legend_fontsize)
        elif plot_type == "pcolormesh":
            y_data = plot_info.get("y_data")
            z_data = plot_info.get("z_data")
            if y_data is None or z_data is None:
                raise ValueError(f"Missing data for pcolormesh {i}")
            plot_z_data = z_data
            clabel = plot_info.get("clabel", "Value")
            vmin_plot = (
                global_vmin
                if global_vmin is not None
                else plot_info.get("vmin", calc_vmin)
            )
            vmax_plot = (
                global_vmax
                if global_vmax is not None
                else plot_info.get("vmax", calc_vmax)
            )
            shading = plot_info.get("shading", "auto")
            cmap = plot_info.get("cmap", global_cmap)
            im = ax_main.pcolormesh(
                x_data,
                y_data,
                plot_z_data,
                cmap=cmap,
                shading=shading,
                vmin=vmin_plot,
                vmax=vmax_plot,
            )
            cbar = fig.colorbar(im, cax=ax_cbar)
            cbar.set_label(clabel, size=label_fontsize)
            cbar.ax.tick_params(labelsize=tick_label_fontsize)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        # --- Add Annotations: Blocks, Vlines, Hlines ---
        subplot_blocks = plot_info.get("blocks", [])
        subplot_vlines = plot_info.get("vlines")
        subplot_hlines = plot_info.get("hlines", [])  # NEW: Get hlines list

        # --- Draw Blocks ---
        if subplot_blocks:
            for block in subplot_blocks:
                start = block.get("start")
                end = block.get("end")
                color = block.get("color")
                if start is not None and end is not None and color is not None:
                    alpha = block.get("alpha", 0.2)
                    ax_main.axvspan(start, end, color=color, alpha=alpha, zorder=-10)
                    label = block.get("label")
                    if label:
                        start_num = mdates.date2num(start)
                        end_num = mdates.date2num(end)
                        mid_dt = mdates.num2date(start_num + (end_num - start_num) / 2)
                        y_txt = (
                            ax_main.get_ylim()[0]
                            + (ax_main.get_ylim()[1] - ax_main.get_ylim()[0]) * 0.95
                        )
                        lbl_color = (
                            block_label_color
                            if block_label_color is not None
                            else color
                        )
                        ax_main.text(
                            mid_dt,
                            y_txt,
                            label,
                            ha="center",
                            va="top",
                            fontsize=annotation_fontsize,
                            color=lbl_color,
                            clip_on=True,
                            fontweight="bold",
                        )

        # --- Draw Vertical Lines ---
        elif isinstance(subplot_vlines, dict):
            for label, dt_value in subplot_vlines.items():
                if dt_value is not None:
                    ax_main.axvline(
                        dt_value,
                        color=vline_color,
                        linestyle=vline_linestyle,
                        alpha=0.7,
                        lw=1,
                    )
                    y_txt = (
                        ax_main.get_ylim()[0]
                        + (ax_main.get_ylim()[1] - ax_main.get_ylim()[0]) * 0.95
                    )
                    ax_main.text(
                        dt_value,
                        y_txt,
                        f" {label}",
                        rotation=90,
                        color=vline_label_color,
                        ha="left",
                        va="top",
                        fontsize=annotation_fontsize,
                        clip_on=True,
                    )

        # --- Draw Horizontal Lines --- NEW ---
        if subplot_hlines:
            for hline_def in subplot_hlines:
                y_val = hline_def.get("y")
                if y_val is not None:
                    color = hline_def.get(
                        "color", hline_color
                    )  # Use specific or default
                    ls = hline_def.get("linestyle", hline_linestyle)
                    ax_main.axhline(y_val, color=color, linestyle=ls, alpha=0.8, lw=1)
                    label = hline_def.get("label")
                    if label:
                        # Position text near right edge, slightly above the line
                        # Use blended transform: x in axes coords, y in data coords
                        # Adjust x position (0.98) and vertical offset (e.g., +0.01*diff) as needed
                        y_low, y_high = ax_main.get_ylim()
                        y_offset = (
                            y_high - y_low
                        ) * 0.01  # Small offset above the line
                        lbl_color = hline_def.get("label_color", hline_label_color)
                        ax_main.text(
                            0.98,
                            y_val + y_offset,
                            f"{label} ",  # Add space for padding
                            transform=ax_main.get_yaxis_transform(),  # Key for positioning
                            ha="right",
                            va="bottom",
                            fontsize=annotation_fontsize,
                            color=lbl_color,
                            clip_on=True,
                        )

        # --- Add Panel Label (a), (b), ... ---
        panel_label = f"({panel_labels[i]})"
        ax_main.text(
            0.01,
            0.98,
            panel_label,
            transform=ax_main.transAxes,
            fontsize=panel_label_fontsize,
            fontweight="bold",
            va="top",
            ha="left",
        )

        # Set Title/YLabel/Ticks using specified font sizes
        ax_main.set_title(
            plot_info.get("title", f"Subplot {i+1}"), fontsize=title_fontsize
        )
        ax_main.set_ylabel(plot_info.get("ylabel", ""), fontsize=label_fontsize)
        ax_main.tick_params(axis="y", labelsize=tick_label_fontsize)
        if i < num_plots - 1:
            ax_main.tick_params(labelbottom=False)
        else:
            ax_main.tick_params(axis="x", labelsize=tick_label_fontsize)

    # --- Configure Final X-axis (same logic) ---
    ax_last_main = axes[num_plots - 1, 0]
    final_xlabel = xlabel
    use_multi_labels = (
        x_datetime_ref is not None
        and x_aux_data is not None
        and x_label_step is not None
    )
    if use_multi_labels:
        if final_xlabel is None:
            final_xlabel = f"Time (UTC) / {' / '.join(x_aux_data.keys())}"
        dt_ticks = x_datetime_ref[::x_label_step]
        aux_ticks = {
            key: np.asarray(arr)[::x_label_step] for key, arr in x_aux_data.items()
        }
        time_labels = [np.datetime_as_string(t, unit="s")[11:19] for t in dt_ticks]
        xticklabels = [
            "\n".join(
                [time_labels[i]]
                + [f"{aux_ticks[key][i]:.1f}" for key in x_aux_data.keys()]
            )
            for i in range(len(dt_ticks))
        ]
        ax_last_main.set_xticks(dt_ticks)
        ax_last_main.set_xticklabels(
            xticklabels,
            rotation=rotate_xticklabels,
            ha="right" if rotate_xticklabels != 0 else "center",
            fontsize=tick_label_fontsize,
        )
    else:  # Standard formatting
        if final_xlabel is None:
            final_xlabel = "Time (UTC)"
        if date_format_str:
            formatter = mdates.DateFormatter(date_format_str)
        else:
            locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            ax_last_main.xaxis.set_major_locator(locator)
        ax_last_main.xaxis.set_major_formatter(formatter)
        if rotate_xticklabels != 0:
            plt.setp(
                ax_last_main.get_xticklabels(), rotation=rotate_xticklabels, ha="right"
            )
    ax_last_main.set_xlabel(final_xlabel, fontsize=label_fontsize)
    if all_x_data:
        try:
            min_t = min(np.min(arr) for arr in all_x_data if arr.size > 0)
            max_t = max(np.max(arr) for arr in all_x_data if arr.size > 0)
            ax_last_main.set_xlim(min_t, max_t)
        except Exception as e:
            print(f"Warn: Could not set x-limits: {e}")

    if figure_title:
        fig.suptitle(figure_title, fontsize=title_fontsize + 2)

    return fig, axes


# def plot_gridded_panels(
#     plot_definitions: List[List[Optional[Dict[str, Any]]]],
#     nrows: int = 4,
#     ncols_main: int = 2, # Number of main plotting columns (e.g., 2, potentially 3)
#     add_shared_cbar: bool = True, # Add a shared colorbar in the last column?
#     # --- Layout ---
#     figsize: Optional[Tuple[float, float]] = None,
#     main_col_ratio: float = 10, # Relative width of main columns
#     cbar_col_ratio: float = 0.5, # Relative width of the colorbar column
#     layout_engine: str = 'constrained', # 'constrained' or 'tight'
#     # --- Shared Colorbar ---
#     shared_cbar_label: str = 'Value',
#     global_cmap: str = 'viridis',
#     use_shared_clims: bool = True,
#     clim_percentiles: Tuple[float, float] = (1.0, 99.0),
#     global_vmin: Optional[float] = None,
#     global_vmax: Optional[float] = None,
#     # --- General Plotting ---
#     use_grid: bool = True,
#     figure_title: Optional[str] = None,
#     # --- Font Size Control ---
#     title_fontsize: float = 11,
#     label_fontsize: float = 10,
#     tick_label_fontsize: float = 9,
#     legend_fontsize: float = 8,
#     panel_label_fontsize: float = 11,
# ):
#     """
#     Creates a grid of plots (e.g., 4x3) with potentially different plot types per cell.
#     Axes are NOT shared. A single shared colorbar can be added in the last column
#     for pcolormesh plots in the last row. Panel labels (a, b, ...) are added.
#
#     Args:
#         plot_definitions (List[List[Optional[Dict]]]): A 2D list (list of rows)
#             where each element plot_definitions[row][col] is a dictionary defining
#             the plot for that cell in the main columns (up to ncols_main), or None
#             to leave the cell blank. Dictionary keys:
#             - 'plot_type' (str): 'line2y', 'logline2y', 'pcolormesh'.
#             - 'title' (str, optional).
#             - 'xlabel' (str, optional).
#             - 'ylabel' (str, optional).
#             - For 'line2y', 'logline2y':
#                 'x_data' (NDArray), 'y_data1' (NDArray), 'y_data2' (NDArray),
#                 'label1' (str, optional), 'label2' (str, optional),
#                 'color1' (str, optional), 'color2' (str, optional),
#                 'linewidth' (float, optional).
#             - For 'pcolormesh':
#                 'x_data' (NDArray), 'y_data' (NDArray), 'z_data' (NDArray),
#                 'cmap' (str, optional), 'vmin' (float, optional),
#                 'vmax' (float, optional), 'shading' (str, optional, default 'auto').
#         nrows (int): Number of rows in the grid.
#         ncols_main (int): Number of columns containing actual plots.
#         add_shared_cbar (bool): If True, adds a narrow column for a shared colorbar
#                                 associated with pcolormesh plots in the last row.
#         figsize (tuple, optional): Figure size (width, height).
#         main_col_ratio (float): Relative width for each main plot column.
#         cbar_col_ratio (float): Relative width for the colorbar column.
#         layout_engine (str): Layout engine ('constrained' or 'tight').
#         shared_cbar_label (str): Label for the shared colorbar.
#         global_cmap (str): Default colormap for pcolormesh.
#         use_shared_clims (bool): Calculate shared vmin/vmax for ALL pcolormesh plots.
#         clim_percentiles (tuple): Percentiles for shared clim calculation.
#         global_vmin (float, optional): Force global min color limit.
#         global_vmax (float, optional): Force global max color limit.
#         use_grid (bool): Display grid lines on main plots.
#         figure_title (str, optional): Overall figure title.
#         title_fontsize ... panel_label_fontsize: Font size controls.
#
#     Returns:
#         tuple[plt.Figure, np.ndarray[plt.Axes]]: Figure and the full 2D array of axes.
#     """
#     if not plot_definitions or not isinstance(plot_definitions[0], list):
#          raise ValueError("plot_definitions must be a list of lists.")
#     if len(plot_definitions) != nrows:
#          warnings.warn(f"Number of rows in plot_definitions ({len(plot_definitions)}) does not match nrows ({nrows}). Plotting may be incomplete.", stacklevel=2)
#     # Ensure inner lists have at least ncols_main elements (can be None)
#     for r, row_def in enumerate(plot_definitions):
#          if len(row_def) < ncols_main:
#               raise ValueError(f"Row {r} in plot_definitions has only {len(row_def)} columns defined, needs at least {ncols_main}.")
#
#     ncols_total = ncols_main + 1 if add_shared_cbar else ncols_main
#
#     # Define width ratios for GridSpec
#     width_ratios = [main_col_ratio] * ncols_main
#     if add_shared_cbar:
#         width_ratios.append(cbar_col_ratio)
#
#     if figsize is None:
#         fig_height_per_plot = 2.5 # Adjust as needed
#         fig_height = fig_height_per_plot * nrows + 1.0 # Add margin
#         # Width depends on ratios
#         total_ratio = sum(width_ratios)
#         base_width_unit = 5 # Adjust basic width unit
#         fig_width = (total_ratio / main_col_ratio) * base_width_unit + 1.0
#         figsize = (fig_width, fig_height)
#         print(f"Figure size not specified, using calculated figsize={figsize}")
#
#
#     fig, axes = plt.subplots(
#         nrows=nrows,
#         ncols=ncols_total,
#         figsize=figsize,
#         layout=layout_engine,
#         gridspec_kw={'width_ratios': width_ratios}
#     )
#     # Ensure axes is always 2D
#     if nrows == 1 and ncols_total == 1: axes = np.array([[axes]])
#     elif nrows == 1: axes = axes.reshape(1, ncols_total)
#     elif ncols_total == 1: axes = axes.reshape(nrows, 1)
#
#     # --- Determine Color Limits (if needed, based on ALL pcolormesh plots) ---
#     calc_vmin, calc_vmax = None, None
#     mappable_for_cbar = None # Store one mappable from the last row for the cbar
#     if add_shared_cbar and (global_vmin is None and global_vmax is None and use_shared_clims):
#         all_z_data = []
#         print("Calculating shared color limits for all pcolormesh plots...")
#         for r in range(nrows):
#             for c in range(ncols_main):
#                 if r < len(plot_definitions) and c < len(plot_definitions[r]):
#                     info = plot_definitions[r][c]
#                     if info and info.get('plot_type') == 'pcolormesh':
#                         z = info.get('z_data')
#                         if z is not None and np.any(np.isfinite(z)):
#                             all_z_data.append(z[np.isfinite(z)].flatten())
#         if all_z_data:
#             cat = np.concatenate(all_z_data)
#             if cat.size > 0:
#                 try: calc_vmin, calc_vmax = np.percentile(cat, clim_percentiles); print(f" Calculated shared vmin={calc_vmin:.2f}, vmax={calc_vmax:.2f} (raw data)")
#                 except IndexError: print("Warn: Bad clim percentiles."); calc_vmin, calc_vmax = None, None
#             else: print("Warn: No valid data for shared clim."); calc_vmin, calc_vmax = None, None
#         else: print("Warn: No pcolormesh found for shared clim."); calc_vmin, calc_vmax = None, None
#
#
#     # --- Plotting Loop ---
#     panel_labels = string.ascii_lowercase
#     panel_idx = 0
#     for r in range(nrows):
#         for c in range(ncols_main):
#             ax = axes[r, c]
#             plot_info = None
#             # Get plot definition if available
#             if r < len(plot_definitions) and c < len(plot_definitions[r]):
#                 plot_info = plot_definitions[r][c]
#
#             if plot_info is None:
#                 ax.axis('off') # Turn off unused axes
#                 continue # Skip to next cell
#
#             plot_type = plot_info.get('plot_type')
#             if use_grid: ax.grid(True, linestyle='--', alpha=0.6)
#
#             x_data = plot_info.get('x_data')
#             if x_data is None: raise ValueError(f"Missing 'x_data' for subplot [{r},{c}]")
#
#             im = None # To store pcolormesh result
#
#             # --- Plot based on type ---
#             if plot_type in ['line2y', 'logline2y']:
#                 y1 = plot_info.get('y_data1'); y2 = plot_info.get('y_data2')
#                 if y1 is None or y2 is None: raise ValueError(f"Missing 'y_data1' or 'y_data2' for subplot [{r},{c}]")
#                 lbl1 = plot_info.get('label1'); lbl2 = plot_info.get('label2')
#                 c1 = plot_info.get('color1'); c2 = plot_info.get('color2')
#                 lw = plot_info.get('linewidth', 1.5)
#                 ax.plot(x_data, y1, label=lbl1, color=c1, linewidth=lw)
#                 ax.plot(x_data, y2, label=lbl2, color=c2, linewidth=lw)
#                 if lbl1 or lbl2: ax.legend(loc='best', fontsize=legend_fontsize)
#                 if plot_type == 'logline2y':
#                     ax.set_yscale('log')
#
#             elif plot_type == 'pcolormesh':
#                 y_data = plot_info.get('y_data'); z_data = plot_info.get('z_data')
#                 if y_data is None or z_data is None: raise ValueError(f"Missing data for pcolormesh [{r},{c}]")
#                 plot_z_data = z_data # Plot raw data
#                 vmin_plot = global_vmin if global_vmin is not None else plot_info.get('vmin', calc_vmin)
#                 vmax_plot = global_vmax if global_vmax is not None else plot_info.get('vmax', calc_vmax)
#                 shading = plot_info.get('shading', 'auto')
#                 cmap = plot_info.get('cmap', global_cmap)
#                 im = ax.pcolormesh(x_data, y_data, plot_z_data, cmap=cmap, shading=shading, vmin=vmin_plot, vmax=vmax_plot)
#                 # Store the mappable if it's in the last row (for shared colorbar)
#                 if add_shared_cbar and r == nrows - 1:
#                      mappable_for_cbar = im # Use the last one encountered in the row
#
#             else:
#                 warnings.warn(f"Unknown plot_type '{plot_type}' for subplot [{r},{c}]. Leaving blank.", stacklevel=2)
#                 ax.axis('off')
#                 continue
#
#             # --- Add Panel Label ---
#             panel_label = f"({panel_labels[panel_idx]})"
#             ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=panel_label_fontsize, fontweight='bold', va='top', ha='left', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none')) # Add white background
#             panel_idx += 1
#
#             # --- Set Titles and Labels ---
#             ax.set_title(plot_info.get('title', ''), fontsize=title_fontsize)
#             ax.set_ylabel(plot_info.get('ylabel', ''), fontsize=label_fontsize)
#             ax.set_xlabel(plot_info.get('xlabel', ''), fontsize=label_fontsize) # Set xlabel per plot now
#             ax.tick_params(axis='both', labelsize=tick_label_fontsize)
#
#     # --- Configure Colorbar Column ---
#     if add_shared_cbar:
#         for r in range(nrows):
#             ax_cbar_col = axes[r, ncols_main] # Axes in the last column
#             if r == nrows - 1 and mappable_for_cbar is not None:
#                 # Add the shared colorbar to the last row's dedicated axes
#                 cbar = fig.colorbar(mappable_for_cbar, cax=ax_cbar_col)
#                 cbar.set_label(shared_cbar_label, size=label_fontsize)
#                 cbar.ax.tick_params(labelsize=tick_label_fontsize)
#             else:
#                 # Turn off all other axes in the colorbar column
#                 ax_cbar_col.axis('off')
#
#     # --- Add Figure Title ---
#     if figure_title:
#         fig.suptitle(figure_title, fontsize=title_fontsize + 2)
#
#     return fig, axes


# Define default colors to cycle through if not provided
DEFAULT_LINE_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_gridded_panels(
        plot_definitions: List[List[Optional[Dict[str, Any]]]],
        nrows: int = 4,
        ncols_main: int = 2,
        add_shared_cbar: bool = True,
        # --- Layout ---
        figsize: Optional[Tuple[float, float]] = None,
        main_col_ratio: float = 10,
        cbar_col_ratio: float = 0.5,
        layout_engine: str = "constrained",
        # --- Shared Colorbar ---
        shared_cbar_label: str = "Value",
        global_cmap: str = "viridis",
        use_shared_clims: bool = True,
        clim_percentiles: Tuple[float, float] = (1.0, 99.0),
        global_vmin: Optional[float] = None,
        global_vmax: Optional[float] = None,
        # --- General Plotting ---
        use_grid: bool = True,
        figure_title: Optional[str] = None,
        # --- Font Size Control ---
        base_fontsize: float = 10,
        title_fontsize: float = 12,
        label_fontsize: float = 10,
        tick_label_fontsize: float = 9,
        legend_fontsize: float = 9,
        annotation_fontsize: float = 8,
        panel_label_fontsize: float = 12,
        # --- Annotation Style Defaults ---
        vline_color: str = "red",
        vline_linestyle: str = "--",
        vline_label_color: str = "red",
        hline_color: str = "red",
        hline_linestyle: str = ":",
        hline_label_color: str = "red",
        block_label_color: str = None,  # None uses block color
        # --- Axis Label Control ---
        rotate_xticklabels: bool = False  # Add control for x-axis label rotation
):
    """
    Creates a grid of plots supporting various types, annotations per subplot,
    log scale grids, and a shared colorbar.

    Args:
        plot_definitions (List[List[Optional[Dict]]]): Defines plots per cell. Dict keys:
            - Common: 'plot_type' ('line' or 'pcolormesh'), 'title', 'xlabel', 'ylabel'.
            - For 'line':
                'x_data' (NDArray), 'y_data_list' (List[NDArray]), 
                'yscale' (str, optional): 'linear' (default) or 'log'.
                'labels', 'colors', 'linewidths' (optional lists or single value).
            - For 'pcolormesh':
                'x_data', 'y_data', 'z_data' (NDArrays),
                'cmap', 'vmin', 'vmax', 'shading' (optional).
            - Optional Annotation keys (for ANY plot_type):
                'blocks': list[dict] {'start', 'end', 'color', ['label', 'alpha']}
                'vlines': dict {'Label': x_value (datetime or float)}
                'hlines': list[dict] {'y', 'color', 'linestyle', ['label', 'label_color']}
        nrows, ncols_main: Grid dimensions for main plots.
        add_shared_cbar: Add shared colorbar column.
        figsize, main_col_ratio, cbar_col_ratio, layout_engine: Layout controls.
        shared_cbar_label, global_cmap, use_shared_clims, ... global_vmax: Colorbar/clim controls.
        use_grid: Display grid lines.
        figure_title: Overall figure title.
        title_fontsize ... panel_label_fontsize: Font size controls.
        vline_..., hline_..., block_...: Annotation style defaults.
        rotate_xticklabels: Whether to rotate x-axis tick labels 45 degrees.

    Returns:
        tuple[plt.Figure, np.ndarray[plt.Axes]]: Figure and the full 2D array of axes.
    """
    if not plot_definitions or not isinstance(plot_definitions[0], list):
        raise ValueError("plot_definitions must be list of lists.")
    if len(plot_definitions) != nrows:
        warnings.warn(f"Num rows mismatch", stacklevel=2)
    for r, row_def in enumerate(plot_definitions):
        if len(row_def) < ncols_main:
            raise ValueError(f"Row {r} needs {ncols_main} cols.")

    ncols_total = ncols_main + 1 if add_shared_cbar else ncols_main
    width_ratios_list = [main_col_ratio] * ncols_main
    if add_shared_cbar:
        width_ratios_list.append(cbar_col_ratio)

    if figsize is None:
        fig_height = (2.5 * nrows) + 1.0
        total_ratio = sum(width_ratios_list)
        base_width_unit = 5
        fig_width = (total_ratio / main_col_ratio) * base_width_unit + 1.0
        figsize = (fig_width, fig_height)
        print(f"Using calculated figsize={figsize}")

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols_total,
        figsize=figsize,
        layout=layout_engine,
        gridspec_kw={"width_ratios": width_ratios_list},
    )
    if nrows == 1 and ncols_total == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, ncols_total)
    elif ncols_total == 1:
        axes = axes.reshape(nrows, 1)

    # --- Determine Color Limits (if needed - raw z_data - same logic) ---
    calc_vmin, calc_vmax = None, None
    mappable_for_cbar = None
    if add_shared_cbar and (
            global_vmin is None and global_vmax is None and use_shared_clims
    ):
        all_z_data = []
        print("Calculating shared color limits...")
        for r in range(nrows):
            for c in range(ncols_main):
                if r < len(plot_definitions) and c < len(plot_definitions[r]):
                    info = plot_definitions[r][c]
                    if info and info.get("plot_type") == "pcolormesh":
                        z = info.get("z_data")
                        if z is not None and np.any(np.isfinite(z)):
                            all_z_data.append(z[np.isfinite(z)].flatten())
        if all_z_data:
            cat = np.concatenate(all_z_data)
            if cat.size > 0:
                try:
                    calc_vmin, calc_vmax = np.percentile(cat, clim_percentiles)
                    print(
                        f" Calculated shared vmin={calc_vmin:.2f}, vmax={calc_vmax:.2f} (raw data)"
                    )
                except IndexError:
                    print("Warn: Bad clim percentiles.")
                    calc_vmin, calc_vmax = None, None
            else:
                print("Warn: No valid data for shared clim.")
                calc_vmin, calc_vmax = None, None
        else:
            print("Warn: No pcolormesh found for shared clim.")
            calc_vmin, calc_vmax = None, None

    panel_labels = string.ascii_lowercase
    panel_idx = 0
    for r in range(nrows):
        for c in range(ncols_main):
            ax = axes[r, c]
            plot_info = None
            if r < len(plot_definitions) and c < len(plot_definitions[r]):
                plot_info = plot_definitions[r][c]
            if plot_info is None:
                ax.axis("off")
                continue

            plot_type = plot_info.get("plot_type")
            x_data = plot_info.get("x_data")
            if x_data is None:
                raise ValueError(f"Missing 'x_data' for subplot [{r},{c}]")
            im = None
            yscale = "linear"  # Default yscale

            # --- Plot main content ---
            if plot_type == "line":
                y_data_list = plot_info.get("y_data_list")
                if not isinstance(y_data_list, list) or not y_data_list:
                    raise ValueError(f"Invalid 'y_data_list' for subplot [{r},{c}]")
                labels = plot_info.get("labels", [None] * len(y_data_list))
                colors = plot_info.get("colors", [None] * len(y_data_list))
                linewidths = plot_info.get("linewidths", 1.5)
                yscale = plot_info.get("yscale", "linear")
                if not isinstance(labels, list) or len(labels) != len(y_data_list):
                    labels = [None] * len(y_data_list)
                if not isinstance(colors, list) or len(colors) != len(y_data_list):
                    colors = [None] * len(y_data_list)
                if isinstance(linewidths, (int, float)):
                    linewidths = [linewidths] * len(y_data_list)
                elif not isinstance(linewidths, list) or len(linewidths) != len(
                        y_data_list
                ):
                    linewidths = [1.5] * len(y_data_list)
                any_label_present = False
                for idx, y_data in enumerate(y_data_list):
                    if len(x_data) != len(y_data):
                        warnings.warn(
                            f"Length mismatch x/y[{idx}] subplot [{r},{c}].",
                            stacklevel=2,
                        )
                        continue
                    color = (
                        colors[idx]
                        if colors[idx] is not None
                        else DEFAULT_LINE_COLORS[idx % len(DEFAULT_LINE_COLORS)]
                    )
                    label = labels[idx]
                    lw = linewidths[idx]
                    ax.plot(x_data, y_data, label=label, color=color, linewidth=lw)
                    if label is not None:
                        any_label_present = True
                if any_label_present:
                    ax.legend(loc="best", fontsize=legend_fontsize)
                ax.set_yscale(yscale)
                if add_shared_cbar and r != nrows - 1:
                    axes[r, ncols_main].axis("off")  # Turn off non-last cbar axes

            elif plot_type == "pcolormesh":
                y_data = plot_info.get("y_data")
                z_data = plot_info.get("z_data")
                if y_data is None or z_data is None:
                    raise ValueError(f"Missing data for pcolormesh [{r},{c}]")
                if z_data.shape[0] != len(y_data) or z_data.shape[1] != len(x_data):
                    warnings.warn(
                        f"z_data shape mismatch subplot [{r},{c}]", stacklevel=2
                    )
                plot_z_data = z_data
                vmin_plot = (
                    global_vmin
                    if global_vmin is not None
                    else plot_info.get("vmin", calc_vmin)
                )
                vmax_plot = (
                    global_vmax
                    if global_vmax is not None
                    else plot_info.get("vmax", calc_vmax)
                )
                shading = plot_info.get("shading", "auto")
                cmap = plot_info.get("cmap", global_cmap)
                im = ax.pcolormesh(
                    x_data,
                    y_data,
                    plot_z_data,
                    cmap=cmap,
                    shading=shading,
                    vmin=vmin_plot,
                    vmax=vmax_plot,
                )
                if add_shared_cbar and r == nrows - 1:
                    mappable_for_cbar = im
                # Turn off cbar axis for pcolormesh if not last row 
                if add_shared_cbar and r != nrows - 1:
                    axes[r, ncols_main].axis("off")

            else:
                warnings.warn(
                    f"Unknown plot_type '{plot_type}' subplot [{r},{c}].", stacklevel=2
                )
                ax.axis("off")
                continue

            # --- Configure Grid based on Y Scale ---
            if use_grid:
                # Use 'both' for major/minor on log, 'major' on linear
                which_grid = "both" if yscale == "log" else "major"
                grid_linestyle = ":" if yscale == "log" else "--"  # Finer grid for log
                ax.grid(True, which=which_grid, linestyle=grid_linestyle, alpha=0.6)

            # --- Add Annotations (Blocks, Vlines, Hlines) ---
            # Note: This logic works for both 'line' and 'pcolormesh' axes
            subplot_blocks = plot_info.get("blocks", [])
            subplot_vlines = plot_info.get("vlines")
            subplot_hlines = plot_info.get("hlines", [])

            if subplot_blocks:
                for block in subplot_blocks:
                    start = block.get("start")
                    end = block.get("end")
                    color = block.get("color")
                    if start is not None and end is not None and color is not None:
                        alpha = block.get("alpha", 0.2)
                        ax.axvspan(start, end, color=color, alpha=alpha, zorder=-10)
                        label = block.get("label")
                        if label:
                            try:
                                start_num = mdates.date2num(start)
                                end_num = mdates.date2num(end)
                                mid_dt = mdates.num2date(
                                    start_num + (end_num - start_num) / 2
                                )  # Assumes datetime
                            except TypeError:
                                mid_dt = (
                                        start + (end - start) / 2
                                )  # Assume numeric otherwise
                            y_txt = (
                                    ax.get_ylim()[0]
                                    + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95
                            )
                            lbl_color = (
                                block_label_color
                                if block_label_color is not None
                                else color
                            )
                            ax.text(
                                mid_dt,
                                y_txt,
                                label,
                                ha="center",
                                va="top",
                                fontsize=annotation_fontsize,
                                color=lbl_color,
                                clip_on=True,
                                fontweight="bold",
                            )

            if isinstance(subplot_vlines, dict):
                for label, x_val in subplot_vlines.items():
                    if x_val is not None:
                        ax.axvline(
                            x_val,
                            color=vline_color,
                            linestyle=vline_linestyle,
                            alpha=0.7,
                            lw=1,
                        )
                        y_txt = (
                                ax.get_ylim()[0]
                                + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95
                        )
                        ax.text(
                            x_val,
                            y_txt,
                            f" {label}",
                            rotation=90,
                            color=vline_label_color,
                            ha="left",
                            va="top",
                            fontsize=annotation_fontsize,
                            clip_on=True,
                        )

            if subplot_hlines:
                for hline_def in subplot_hlines:
                    y_val = hline_def.get("y")
                    if y_val is not None:
                        color = hline_def.get("color", hline_color)
                        ls = hline_def.get("linestyle", hline_linestyle)
                        ax.axhline(y_val, color=color, linestyle=ls, alpha=0.8, lw=1)
                        label = hline_def.get("label")
                        if label:
                            y_low, y_high = ax.get_ylim()
                            y_offset = (
                                (y_high - y_low) * 0.01
                                if yscale == "linear"
                                else y_val * 0.1
                            )  # Different offset logic for log? Careful
                            lbl_color = hline_def.get("label_color", hline_label_color)
                            # Use blended transform for robust positioning
                            ax.text(
                                0.98,
                                y_val,
                                f"{label} ",
                                transform=ax.get_yaxis_transform(which="grid"),
                                ha="right",
                                va="bottom",
                                fontsize=annotation_fontsize,
                                color=lbl_color,
                                clip_on=True,
                                bbox=dict(
                                    boxstyle="round,pad=0.1",
                                    fc="white",
                                    alpha=0.5,
                                    ec="none",
                                ),
                            )  # Added background box

            # --- Add Panel Label ---
            panel_label = f"({panel_labels[panel_idx]})"
            ax.text(
                0.02,
                0.98,
                panel_label,
                transform=ax.transAxes,
                fontsize=panel_label_fontsize,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
            )
            panel_idx += 1

            # --- Set Titles and Labels ---
            ax.set_title(plot_info.get("title", ""), fontsize=title_fontsize)
            ax.set_ylabel(plot_info.get("ylabel", ""), fontsize=label_fontsize)
            ax.set_xlabel(plot_info.get("xlabel", ""), fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)

            # Rotate x-axis labels if requested
            if rotate_xticklabels:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # --- Configure Colorbar Column (same logic) ---
    if add_shared_cbar:
        for r in range(nrows):
            ax_cbar_col = axes[r, ncols_main]
            if r == nrows - 1 and mappable_for_cbar is not None:
                cbar = fig.colorbar(mappable_for_cbar, cax=ax_cbar_col)
                cbar.set_label(shared_cbar_label, size=label_fontsize)
                cbar.ax.tick_params(labelsize=tick_label_fontsize)
            else:
                ax_cbar_col.axis("off")

    # --- Add Figure Title (remains the same) ---
    if figure_title:
        fig.suptitle(figure_title, fontsize=title_fontsize + 2)

    return fig, axes
