import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from configs import ProjectConfigs
from utils import get_3arrs


def plot_multi_disturb_field(
    datetimes: np.ndarray,
    fields_list: list[np.ndarray],
    latitudes: np.ndarray,
    qdlats: np.ndarray,
    mlts: np.ndarray,
    figure_title: str,
    subplot_titles_list: list[str],
    ylabel_list: list[str],
    line_labels_list: list[str],
    vertical_lines_dict: dict[str, np.datetime64] = None, # e.g., {"Max Lat": dt_max_lat, "Event 1": dt_event1}
    step: int = 200, # Adjusted default step for potentially denser labels
    figsize: tuple[float, float] = (18, 10), # Adjusted default height
):
    """
    Plots multiple time series datasets against a shared datetime axis in separate subplots,
    with shared latitude, QD latitude, and MLT information on the bottom x-axis labels,
    and customizable vertical lines.

    Args:
        datetimes (np.ndarray): Array of datetime64 objects for the shared x-axis.
        fields_list (list[np.ndarray]): List of 1D numpy arrays, each representing a dataset for a subplot.
        latitudes (np.ndarray): Array of geographic latitudes corresponding to datetimes.
        qdlats (np.ndarray): Array of quasi-dipole latitudes corresponding to datetimes.
        mlts (np.ndarray): Array of magnetic local times corresponding to datetimes.
        figure_title (str): The main title for the entire figure.
        subplot_titles_list (list[str]): List of titles for each subplot. Must match length of fields_list.
        ylabel_list (list[str]): List of y-axis labels for each subplot. Must match length of fields_list.
        line_labels_list (list[str]): List of labels for each plotted line (used in legend). Must match length of fields_list.
        vertical_lines_dict (dict[str, np.datetime64], optional): Dictionary where keys are labels (str)
            and values are datetime64 objects indicating where to draw vertical lines. Defaults to None.
        step (int, optional): Step size for thinning the x-axis labels. Defaults to 200.
        figsize (tuple[float, float], optional): Figure size. Defaults to (18, 10).

    Returns:
        tuple[plt.Figure, np.ndarray[plt.Axes]]: The figure and array of axes objects.

    Raises:
        ValueError: If the lengths of fields_list, subplot_titles_list, ylabel_list,
                    or line_labels_list do not match.
    """
    num_subplots = len(fields_list)
    if not (num_subplots == len(subplot_titles_list) == len(ylabel_list) == len(line_labels_list)):
        raise ValueError("Input lists (fields_list, subplot_titles_list, ylabel_list, line_labels_list) must have the same length.")

    if vertical_lines_dict is None:
        vertical_lines_dict = {} # Ensure it's an empty dict if None

    # Create figure and subplots - sharex=True is key!
    fig, axes = plt.subplots(num_subplots, 1, figsize=figsize, sharex=True)

    # Ensure axes is always iterable, even if num_subplots is 1
    axes = np.atleast_1d(axes)

    # --- Plotting Loop ---
    for i, ax in enumerate(axes):
        field = fields_list[i]
        subplot_title = subplot_titles_list[i]
        ylabel = ylabel_list[i]
        line_label = line_labels_list[i]

        # Plot the main data for this subplot
        ax.plot(datetimes, field, label=line_label)
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_title(subplot_title)
        ax.grid(True, linestyle='--', alpha=0.6) # Add subtle grid

        # Plot vertical lines on this subplot
        for label, dt_value in vertical_lines_dict.items():
            if dt_value is not None: # Check if datetime value exists
                try:
                    # Find y-value at or near the vertical line time to position text
                    # This is approximate, finds closest index
                    time_diff = np.abs(datetimes - dt_value)
                    closest_idx = np.argmin(time_diff)
                    y_pos_text = field[closest_idx] # Use y-value near the line
                    # Alternative: Place text relative to ylim
                    # y_pos_text = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95

                    ax.axvline(dt_value, color='r', linestyle='--', alpha=0.8)
                    # Position text slightly offset from the line, adjust as needed
                    # Using ha='center' and placing slightly above max might work better
                    ax.text(dt_value, y_pos_text, f" {label}", rotation=90, color='b',
                            ha='left', va='center', fontsize=8) # Adjust fontsize, positioning
                except Exception as e:
                    print(f"Warning: Could not plot vertical line or text for '{label}' at {dt_value}: {e}")


    # --- Configure the Shared X-axis (on the last subplot) ---
    last_ax = axes[-1]

    # Thin the data for ticks
    datetime_ticks = datetimes[::step]
    latitude_ticks = latitudes[::step]
    qdlat_ticks = qdlats[::step]
    mlt_ticks = mlts[::step]

    # Set tick positions
    last_ax.set_xticks(datetime_ticks)

    # Format tick labels - More robust time formatting
    datetime_ticks_formatted = [np.datetime_as_string(t, unit='s')[11:19] for t in datetime_ticks] # HH:MM:SS

    # Create labels - consider rotating for readability
    # Option 1: Multi-line labels (can overlap if step is small)
    xticklabels = [
        f"{t}\n{lat:.1f}°\n{qdlat:.1f}°\n{mlt:.1f}h"
        for t, lat, qdlat, mlt in zip(datetime_ticks_formatted, latitude_ticks, qdlat_ticks, mlt_ticks)
    ]
    last_ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=8) # Rotate labels

    # Option 2: Add secondary axes (more complex, but cleaner separation) - Example commented out
    # Requires more careful handling of scales and linking
    # secax = last_ax.secondary_xaxis('top') # Or another location
    # secax.set_xticks(datetime_ticks)
    # secax.set_xticklabels([f"{lat:.1f}°/{qdlat:.1f}°/{mlt:.1f}h" for lat, qdlat, mlt in zip(latitude_ticks, qdlat_ticks, mlt_ticks)], rotation=45, ha='left', fontsize=8)
    # last_ax.set_xticklabels(datetime_ticks_formatted, rotation=45, ha='right', fontsize=8) # Only time on bottom

    last_ax.set_xlabel("UTC / Lat / QDLat / MLT", fontsize=10) # General label for the combined info

    # --- Final Figure Touches ---
    if figure_title:
        fig.suptitle(figure_title, fontsize=16)

    # Adjust layout to prevent title overlap and give space for rotated labels
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # rect=[left, bottom, right, top]

    plt.show()
    return fig, axes

# --- Helper function to find datetime extremes (Example) ---
def find_datetime_extremes(datetimes, values):
    """Finds datetimes corresponding to min and max of values."""
    if len(datetimes) != len(values):
        raise ValueError("datetimes and values must have the same length.")
    if len(values) == 0:
        return None, None
    min_idx = np.nanargmin(values) # Use nanargmin to handle potential NaNs
    max_idx = np.nanargmax(values)
    return datetimes[min_idx], datetimes[max_idx]

# --- Example Usage ---
if __name__ == '__main__':

    file_name = "SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl"
    file_name_igrf = "IGRF_SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl"
    file_name_aux = "aux_SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl"
    data_dir_path = ProjectConfigs.data_dir_path
    file_path = os.path.join(data_dir_path,file_name)
    file_path_igrf = os.path.join(data_dir_path,file_name_igrf)
    file_path_aux = os.path.join(data_dir_path,file_name_aux)

    df = pd.read_pickle(file_path)
    df_igrf = pd.read_pickle(file_path_igrf)
    df_aux = pd.read_pickle(file_path_aux)

    datetimes = df.index.values

    B_N, B_E, B_C = get_3arrs(df["B_NEC"].values)
    B_N_IGRF, B_E_IGRF, B_C_IGRF = get_3arrs(df_igrf["B_NEC_IGRF"].values)
    delta_B_E = B_E - B_E_IGRF
    delta_B_N = B_N - B_N_IGRF
    delta_B_C = B_C - B_C_IGRF

    latitudes = df['Latitude'].values
    qdlats = df_aux['QDLat']
    mlts = df_aux['MLT']


    fields_list = [delta_B_N, delta_B_E, delta_B_C]
    subplot_titles = ["North Component of Disturb Magnetic Field", "East Component of Disturb Magnetic Field", "Centric Component of Disturb Magnetic Field"]
    ylabel_list = ["Magnetic Flux Density (nT)", "Magnetic Flux Density (nT)", "Magnetic Flux Density (nT)"]
    line_labels = ["b_North", "b_East", "b_Centric"]

    # Find extremes for vertical lines (example)
    min_lat_dt, max_lat_dt = find_datetime_extremes(datetimes, latitudes)
    min_qdlat_dt, max_qdlat_dt = find_datetime_extremes(datetimes, qdlats)

    # Create the dictionary for vertical lines
    vertical_lines = {
        "Min Lat": min_lat_dt,
        "Max Lat": max_lat_dt,
        "Min QDLat": min_qdlat_dt,
        "Max QDLat": max_qdlat_dt,
        # "Event Mid": datetimes[num_points // 2] # Add another arbitrary event
    }


    # Call the plotting function
    fig, axes = plot_multi_disturb_field(
        datetimes=datetimes,
        fields_list=fields_list,
        latitudes=latitudes,
        qdlats=qdlats,
        mlts=mlts,
        figure_title="Three Component of Disturb Magnetic Field in NEC Frame",
        subplot_titles_list=subplot_titles,
        ylabel_list=ylabel_list,
        line_labels_list=line_labels,
        vertical_lines_dict=vertical_lines,
        step=500 # Adjust step for label density in example
    )
    # todo: 标签问题、标注问题