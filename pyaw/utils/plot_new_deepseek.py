"""Visualization utilities for time series and scientific data analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.gridspec import GridSpec
from nptyping import NDArray, Datetime64, Float64
from typing import Optional, Tuple


def plot_multi_parameter_xticks(
        timestamps: NDArray[Datetime64],
        values: NDArray[Float64],
        latitudes: NDArray[Float64],
        qdlats: NDArray[Float64],
        mlts: NDArray[Float64],
        step: int = 20000,
        highlight_times: Optional[list[Datetime64]] = None,
        highlight_color: str = "r",
        figsize: Tuple[int, int] = (18, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots time series with composite x-axis labels showing multiple parameters.

    Args:
        timestamps: Array of datetime values
        values: Primary data series to plot
        latitudes: Geographic latitudes corresponding to timestamps
        qdlats: Quasi-Dipole latitudes
        mlts: Magnetic Local Times
        step: Interval for x-axis label sampling
        highlight_times: List of timestamps to highlight with vertical lines
        highlight_color: Color for vertical highlights
        figsize: Figure dimensions

    Returns:
        Tuple containing matplotlib Figure and Axes objects

    Example:
        >>> plot_multi_parameter_xticks(times, data, lats, qdlats, mlts, step=1000)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(timestamps, values, label="Magnetic Disturbance", lw=1.2)

    # Add event markers
    if highlight_times:
        for event_time in highlight_times:
            ax.axvline(event_time, color=highlight_color, ls="--", alpha=0.7)

    # Configure x-axis labels
    sample_indices = slice(None, None, step)
    label_times = timestamps[sample_indices]
    label_lats = latitudes[sample_indices]
    label_qdlats = qdlats[sample_indices]
    label_mlts = mlts[sample_indices]

    ax.set_xticks(label_times)
    ax.set_xticklabels([
        f"{t:%H:%M:%S}\nLat: {lat:.1f}°\nQDLat: {qlat:.1f}\nMLT: {mlt:.1f}"
        for t, lat, qlat, mlt in zip(
            pd.to_datetime(label_times),
            label_lats,
            label_qdlats,
            label_mlts
        )
    ])

    ax.set_ylabel("Magnetic Field (nT)")
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    return fig, ax


def visualize_nan_positions(
        series: pd.Series,
        figsize: Tuple[int, int] = (10, 4),
        marker_size: int = 10
) -> plt.Figure:
    """Visualizes NaN positions in a pandas Series.

    Args:
        series: Input data series with potential NaN values
        figsize: Figure dimensions
        marker_size: Size of NaN indicator markers

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.isna(), 'r.', ms=marker_size, alpha=0.6)
    ax.set_title("NaN Distribution in Time Series")
    ax.set_xlabel("Time Index")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Valid", "NaN"])
    return fig


def compare_interpolation(
        original: pd.Series,
        method: str = "linear",
        figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Compares original and interpolated time series.

    Args:
        original: Series with missing values
        method: Interpolation method (see pandas.Series.interpolate)
        figsize: Figure dimensions

    Returns:
        matplotlib Figure object
    """
    interpolated = original.interpolate(method=method)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    original.plot(ax=ax1, title="Original Data", marker='o', ms=3)
    interpolated.plot(ax=ax2, title=f"{method.title()} Interpolation", marker='o', ms=3)

    fig.tight_layout()
    return fig


def plot_dual_signals(
        x: pd.Index,
        y1: pd.Series,
        y2: pd.Series,
        title: str = "Signal Comparison",
        ylabel: str = "Amplitude"
) -> plt.Figure:
    """Plots two aligned time series with shared x-axis.

    Args:
        x: Shared x-axis values
        y1: First signal series
        y2: Second signal series
        title: Figure title
        ylabel: Y-axis label

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y1, label=y1.name or "Signal 1")
    ax.plot(x, y2, label=y2.name or "Signal 2")

    ax.set(title=title, xlabel="Time", ylabel=ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def create_dashboard(
        signals: dict[str, pd.Series],
        figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """Creates a multi-panel dashboard of time series plots.

    Args:
        signals: Dictionary of {panel_title: time_series}
        figsize: Overall figure dimensions

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    n_plots = len(signals)
    gs = GridSpec(n_plots, 1)

    axes = []
    for idx, (title, data) in enumerate(signals.items()):
        ax = fig.add_subplot(gs[idx, 0])
        data.plot(ax=ax, title=title, lw=1)
        axes.append(ax)

    fig.tight_layout()
    return fig


def interactive_plot(
        df: pd.DataFrame,
        x_col: str,
        y_cols: list[str],
        title: str = "Interactive Plot"
) -> px.line:
    """Creates an interactive Plotly visualization.

    Args:
        df: Input DataFrame
        x_col: X-axis column name
        y_cols: List of y-axis columns to plot
        title: Figure title

    Returns:
        Plotly Express line figure
    """
    fig = px.line(df, x=x_col, y=y_cols, title=title)
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title="Value",
        hovermode="x unified"
    )
    return fig


def plot_plasma_ratio(
        beta: pd.Series,
        mass_ratio: float,
        title: str = "Plasma Beta Ratio"
) -> px.line:
    """Generates annotated plasma beta plot with reference line.

    Args:
        beta: Plasma beta values
        mass_ratio: Electron-to-ion mass ratio (me/mi)
        title: Figure title

    Returns:
        Plotly Express line figure
    """
    fig = px.line(beta, title=title, labels={"value": "Beta", "index": "Time"})

    fig.add_hline(
        y=mass_ratio,
        line_dash="dot",
        annotation_text=f"m_e/m_i = {mass_ratio:.2e}",
        annotation_position="bottom right"
    )

    fig.update_layout(
        xaxis_title="Time (UT)",
        yaxis_title="Beta",
        showlegend=False
    )
    return fig