#!/usr/bin/env python3
"""
coastal_evolution_animation/generate_animation.py

Generates an animated GIF visualizing coastal evolution:
- Left Column: Map showing satellite basemap (Esri World Imagery at 70% transparency),
  profiles (with ID_Perfil 18844 and 18756 highlighted in red dashed line and labeled with their ID),
  and shoreline evolution colored with the 'viridis' colormap by year.
  An inset zoom box in the bottom-left corner displays the same layers zoomed to ROI_2.
  Bottom horizontal colorbar with yearly ticks, and large white text in the top-left
  indicating the current year.
- Right Column: Two stacked horizontally elongated scatter plots showing time series of shoreline positions ("Distancia")
  over time, colored consistently with the map by year using 'viridis':
  * Top: Profile ID 18844
  * Bottom: Profile ID 18756
  * Both share the exact same vertical scale for direct comparison.
  * Vertical axis title: "Avance de la línea de costa respecto la primera fecha (m)"
  * Scatter points have no black outline, and plot spines are clean and frameless.
  * Optional vertical lines for specific event dates (e.g., beach nourishment) with
    a label "Avance de ≈ X metros".
- Frequency: 2 frames per year (accumulating lines every 6 months).
- Pause: The animation pauses on the last frame for 2.0 seconds.
"""

import os
import sys
import io
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as cx
from PIL import Image
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow

# Hardcoded default event dates and advance values for specific profiles:
# Format: {ID_Perfil: [fecha, valor_X_avance]}
DEFAULT_EVENT_DATES: Dict[int, List] = {
    18844: ["2024-07-28 13:00:17", 80],
    18756: ["2024-08-12 13:00:19", 25],
}


def load_datasets(
    shorelines_path: str,
    profiles_path: str,
    rois_path: str,
    seatool_path: str,
    quadrant: int = 61,
    target_profile_ids: List[int] = None,
):
    """
    Load and filter all necessary spatial layers and time series data.
    """
    if target_profile_ids is None:
        target_profile_ids = [18844, 18756]

    print("--> Loading ROIs...")
    roi1 = gpd.read_file(rois_path, layer="ROI_1")
    roi2 = gpd.read_file(rois_path, layer="ROI_2")

    print(f"--> Loading shorelines from {shorelines_path}...")
    # Read shorelines filtering by Cuadrante == quadrant and is_filtered != 0
    gdf_shorelines = gpd.read_file(shorelines_path, where=f"Cuadrante = {quadrant}")
    gdf_shorelines = gdf_shorelines[gdf_shorelines["is_filtered"] != 0].copy()

    # Create year and datetime columns
    date_col = "datetime" if "datetime" in gdf_shorelines.columns else "Fecha"
    gdf_shorelines["datetime"] = pd.to_datetime(gdf_shorelines[date_col].astype(str))
    gdf_shorelines["year"] = gdf_shorelines["datetime"].dt.year
    gdf_shorelines = gdf_shorelines.sort_values("datetime").reset_index(drop=True)
    print(f"    Loaded {len(gdf_shorelines)} filtered shorelines ({gdf_shorelines['year'].min()}-{gdf_shorelines['year'].max()}).")

    print(f"--> Loading profiles from {profiles_path}...")
    gdf_profiles = gpd.read_file(profiles_path)
    
    # Filter profiles intersecting ROI_1 for fast drawing
    roi1_geom = roi1.geometry.iloc[0]
    profiles_in_roi1 = gdf_profiles[gdf_profiles.intersects(roi1_geom)].copy()
    
    target_profiles_gdf = gdf_profiles[gdf_profiles["ID_Perfil"].isin(target_profile_ids)].copy()
    print(f"    Loaded {len(profiles_in_roi1)} profiles inside ROI_1. Target profiles found: {len(target_profiles_gdf)}.")

    print(f"--> Loading time series from {seatool_path}...")
    df_sea = pd.read_csv(seatool_path)
    
    # Determine distance column
    if "Distancia" in df_sea.columns:
        dist_col = "Distancia"
    elif "Distancia_Suavizada" in df_sea.columns:
        dist_col = "Distancia_Suavizada"
    else:
        raise KeyError("Neither 'Distancia' nor 'Distancia_Suavizada' found in S2_seatool.csv")

    df_sea["Distancia"] = df_sea[dist_col]
    df_sea["datetime"] = pd.to_datetime(df_sea["Fecha"].astype(str))
    df_sea["year"] = df_sea["datetime"].dt.year

    dict_target_df = {}
    for pid in target_profile_ids:
        sub_df = df_sea[df_sea["ID_Perfil"] == pid].sort_values("datetime").reset_index(drop=True).copy()
        dict_target_df[pid] = sub_df
        print(f"    Loaded {len(sub_df)} points for profile {pid}.")

    return roi1, roi2, gdf_shorelines, profiles_in_roi1, target_profiles_gdf, dict_target_df


def prefetch_basemaps(roi1: gpd.GeoDataFrame, roi2: gpd.GeoDataFrame, alpha: float):
    """
    Pre-fetch and cache Esri World Imagery basemaps for ROI_1 and ROI_2 once
    so that individual frames render instantaneously without re-requesting tiles.
    """
    print("--> Pre-fetching Esri World Imagery basemap for ROI_1 and ROI_2...")
    r1_bounds = roi1.total_bounds
    r2_bounds = roi2.total_bounds

    # Main map basemap
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_xlim(r1_bounds[0], r1_bounds[2])
    ax1.set_ylim(r1_bounds[1], r1_bounds[3])
    cx.add_basemap(ax1, crs=roi1.crs.to_string(), source=cx.providers.Esri.WorldImagery, alpha=alpha)
    img_roi1 = ax1.images[0].get_array()
    ext_roi1 = ax1.images[0].get_extent()
    plt.close(fig1)

    # Inset zoom basemap
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_xlim(r2_bounds[0], r2_bounds[2])
    ax2.set_ylim(r2_bounds[1], r2_bounds[3])
    cx.add_basemap(ax2, crs=roi2.crs.to_string(), source=cx.providers.Esri.WorldImagery, alpha=alpha)
    img_roi2 = ax2.images[0].get_array()
    ext_roi2 = ax2.images[0].get_extent()
    plt.close(fig2)

    print("    Basemap tiles successfully cached in memory.")
    return (img_roi1, ext_roi1), (img_roi2, ext_roi2)


def generate_animation(
    shorelines_path: str = "data/shorelines/LineasCosta_with_filter_flags.gpkg",
    profiles_path: str = "data/profiles/perfiles_recortados.gpkg",
    rois_path: str = "coastal_evolution_animation/ROIs.gpkg",
    seatool_path: str = "S2_seatool.csv",
    output_gif_path: str = "coastal_evolution_animation/coastal_evolution.gif",
    quadrant: int = 61,
    target_profile_ids: List[int] = None,
    event_dates: Optional[Dict[int, str]] = None,
    event_label_template: str = "Avance de ≈ X metros",
    event_only_when_reached: bool = True,
    pause_last_frame_seconds: float = 3.0,
    basemap_transparency: float = 0.70,  # 70% transparency -> alpha = 0.30
    fps: float = 2.0,
    dpi: int = 120,
):
    """
    Builds the animation and saves it as a GIF.
    """
    if target_profile_ids is None:
        target_profile_ids = [18844, 18756]

    # Use hardcoded default event dates if not specified
    if event_dates is None:
        event_dates = DEFAULT_EVENT_DATES.copy()

    # Pre-parse event dates and advance values
    parsed_event_dates = {}
    for pid, info in event_dates.items():
        if isinstance(info, (list, tuple)) and len(info) >= 2:
            d_val, adv_val = info[0], info[1]
        elif isinstance(info, (list, tuple)) and len(info) == 1:
            d_val, adv_val = info[0], "X"
        else:
            d_val, adv_val = info, "X"
        parsed_event_dates[int(pid)] = (pd.to_datetime(d_val), adv_val)
    print(f"--> Event dates configured: {parsed_event_dates}")

    basemap_alpha = 1.0 - basemap_transparency

    # 1. Load data
    roi1, roi2, gdf_s, profiles_roi1, target_profs, dict_target_df = load_datasets(
        shorelines_path=shorelines_path,
        profiles_path=profiles_path,
        rois_path=rois_path,
        seatool_path=seatool_path,
        quadrant=quadrant,
        target_profile_ids=target_profile_ids,
    )

    r1_bounds = roi1.total_bounds
    r2_bounds = roi2.total_bounds

    # 2. Pre-fetch basemaps
    (img1, ext1), (img2, ext2) = prefetch_basemaps(roi1, roi2, alpha=basemap_alpha)

    # 3. Setup time intervals: 2 frames per year (every 6 months)
    min_year = int(gdf_s["year"].min())
    max_year = int(gdf_s["year"].max())

    periods = []
    for y in range(min_year, max_year + 1):
        cutoff_s1 = pd.Timestamp(f"{y}-06-30 23:59:59")
        cutoff_s2 = pd.Timestamp(f"{y}-12-31 23:59:59")
        
        if (gdf_s["datetime"] <= cutoff_s1).any():
            periods.append((y, 1, cutoff_s1, f"{y} (Sem. 1)"))
        if (gdf_s["datetime"] <= cutoff_s2).any():
            periods.append((y, 2, cutoff_s2, f"{y} (Sem. 2)"))

    print(f"--> Total animation frames to generate: {len(periods)}")

    # 4. Color normalization (viridis by year)
    norm = Normalize(vmin=min_year, vmax=max_year)
    cmap = mpl.colormaps["viridis"]

    # Calculate overall scatter plot date range
    all_dates = pd.concat([df["datetime"] for df in dict_target_df.values()])
    scatter_min_date = all_dates.min() - pd.Timedelta(days=90)
    scatter_max_date = all_dates.max() + pd.Timedelta(days=90)

    # Calculate SHARED Y-axis limits across all target profiles for uniform scale
    all_distances = pd.concat([df["Distancia"] for df in dict_target_df.values()])
    shared_y_min, shared_y_max = all_distances.min(), all_distances.max()
    y_margin_bottom = (shared_y_max - shared_y_min) * 0.08
    y_margin_top = (shared_y_max - shared_y_min) * 0.14  # Extra headroom for event label
    shared_scatter_ylim = (shared_y_min - y_margin_bottom, shared_y_max + y_margin_top)
    print(f"--> Shared vertical scale: [{shared_scatter_ylim[0]:.1f}, {shared_scatter_ylim[1]:.1f}] m")

    # Y position for the event label near the top of the plot
    event_label_y = shared_scatter_ylim[1] - 0.07 * (shared_scatter_ylim[1] - shared_scatter_ylim[0])

    # Pre-calculate label coordinates for each target profile
    profile_label_positions = {}
    for _, row in target_profs.iterrows():
        pid = row["ID_Perfil"]
        coords = list(row.geometry.coords)
        x0, y0 = coords[0]
        profile_label_positions[pid] = (x0 - 50, y0 + 35)

    frames = []

    print("--> Generating frames...")
    for idx, (current_year, semester, cutoff, label_period) in enumerate(periods):
        current_lines = gdf_s[gdf_s["datetime"] <= cutoff]

        # Figure setup: Wider scatter column (width_ratios=[1.0, 1.6]) so plots are longer horizontally
        fig = plt.figure(figsize=(20, 8.5), dpi=dpi)
        gs = gridspec.GridSpec(
            2, 2, width_ratios=[1.0, 1.6], height_ratios=[1.0, 1.0], figure=fig, hspace=0.32, wspace=0.18
        )

        ax_map = fig.add_subplot(gs[:, 0])
        ax_scat_top = fig.add_subplot(gs[0, 1])
        ax_scat_bottom = fig.add_subplot(gs[1, 1])

        # ---------------------------------------------------------------------
        # LEFT COLUMN: MAP
        # ---------------------------------------------------------------------
        ax_map.set_xlim(r1_bounds[0], r1_bounds[2])
        ax_map.set_ylim(r1_bounds[1], r1_bounds[3])
        ax_map.set_aspect("equal", adjustable="box")

        # Esri Basemap
        ax_map.imshow(img1, extent=ext1, zorder=1)

        # Profiles
        profiles_roi1.plot(ax=ax_map, color="gray", linewidth=1.5, zorder=2)
        
        # Highlighted profiles (red, dashed, linewidth 3)
        if len(target_profs) > 0:
            target_profs.plot(
                ax=ax_map, color="red", linewidth=3.0, linestyle="--", zorder=4
            )
            # Add text labels next to each profile
            for pid, (lx, ly) in profile_label_positions.items():
                ax_map.text(
                    lx,
                    ly,
                    f"ID: {pid}",
                    color="red",
                    fontsize=10.5,
                    fontweight="bold",
                    va="bottom",
                    ha="right",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    zorder=6,
                )

        # Accumulated shorelines colored by year with viridis
        if len(current_lines) > 0:
            for yr, group in current_lines.groupby("year"):
                group.plot(
                    ax=ax_map,
                    color=cmap(norm(yr)),
                    linewidth=1.2,
                    zorder=3,
                )

        # Inset zoom box in BOTTOM-LEFT corner (based on ROI_2)
        ax_inset = ax_map.inset_axes([0.03, 0.04, 0.44, 0.44])
        ax_inset.set_xlim(r2_bounds[0], r2_bounds[2])
        ax_inset.set_ylim(r2_bounds[1], r2_bounds[3])
        ax_inset.set_aspect("equal", adjustable="box")

        # Inset basemap and layers
        ax_inset.imshow(img2, extent=ext2, zorder=1)
        profiles_roi1.plot(ax=ax_inset, color="gray", linewidth=1.5, zorder=2)
        if len(target_profs) > 0:
            target_profs.plot(
                ax=ax_inset, color="red", linewidth=3.0, linestyle="--", zorder=4
            )
            if 18844 in profile_label_positions:
                ax_inset.text(
                    0.5,
                    0.3,
                    "ID: 18844",
                    transform=ax_inset.transAxes,
                    color="red",
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    ha="center",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    zorder=6,
                )
        if len(current_lines) > 0:
            for yr, group in current_lines.groupby("year"):
                group.plot(
                    ax=ax_inset,
                    color=cmap(norm(yr)),
                    linewidth=1.6,
                    zorder=3,
                )

        # Inset styling
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        for spine in ax_inset.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(1.8)

        # Draw ROI_2 boundary box on the main map
        roi2.boundary.plot(
            ax=ax_map, color="white", linewidth=1.5, linestyle=":", zorder=5
        )

        # Map axes labels and styling
        ax_map.set_xlabel("UTM X (m)", fontsize=11)
        ax_map.set_ylabel("UTM Y (m)", fontsize=11)
        ax_map.tick_params(labelsize=9)

        # Top-left current year text: white, large, with subtle outline for readability
        ax_map.text(
            0.04,
            0.95,
            f"{current_year}",
            transform=ax_map.transAxes,
            color="white",
            fontsize=32,
            fontweight="bold",
            va="top",
            ha="left",
            path_effects=[pe.withStroke(linewidth=4, foreground="black")],
            zorder=10,
        )

        # North arrow in top-right corner (matplotlib-map-utils)
        north_arrow(ax_map, location="upper right", scale=0.5, zorder=20)

        # Scale bar in meters in top-right corner (matplotlib-scalebar)
        scalebar = ScaleBar(
            dx=1.0,
            units="m",
            fixed_value=500,
            fixed_units="m",
            location="upper right",
            bbox_to_anchor=(0.98, 0.84),
            bbox_transform=ax_map.transAxes,
            box_alpha=0.65,
            box_color="white",
            color="black",
            font_properties={"weight": "bold", "size": 9.5},
            pad=0.25,
            border_pad=0.35,
            scale_loc="bottom",
        )
        ax_map.add_artist(scalebar)

        # Colorbar below map with ticks for each year
        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes("bottom", size="4%", pad=0.45)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="horizontal",
            ticks=range(min_year, max_year + 1),
        )
        cbar.set_ticklabels([str(y) for y in range(min_year, max_year + 1)])
        cbar.set_label("Año", fontsize=11, fontweight="bold")
        cbar.ax.tick_params(labelsize=9)

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: SCATTER PLOTS (TOP: 18844, BOTTOM: 18756)
        # ---------------------------------------------------------------------
        scat_axes = [
            (ax_scat_top, target_profile_ids[0]),
            (ax_scat_bottom, target_profile_ids[1]),
        ]

        y_axis_label = "Avance de la línea de costa\nrespecto la primera fecha (m)"

        for ax_s, pid in scat_axes:
            df_p = dict_target_df[pid]
            current_pts = df_p[df_p["datetime"] <= cutoff]

            # Background reference points (faint gray) - no black outline
            ax_s.scatter(
                df_p["datetime"],
                df_p["Distancia"],
                color="#d8d8d8",
                s=30,
                alpha=0.55,
                edgecolors="none",
                zorder=2,
            )

            # Accumulated points colored by year with viridis - NO black outline (edgecolors='none')
            if len(current_pts) > 0:
                ax_s.scatter(
                    current_pts["datetime"],
                    current_pts["Distancia"],
                    c=current_pts["year"],
                    cmap=cmap,
                    norm=norm,
                    edgecolors="none",
                    s=55,
                    zorder=4,
                )

            # Vertical cutoff line (current animation progress)
            ax_s.axvline(
                cutoff,
                color="#d9534f",
                linestyle="--",
                linewidth=1.4,
                alpha=0.7,
                zorder=3,
            )

            # Optional/Hardcoded vertical event line for this profile
            if pid in parsed_event_dates:
                event_dt, adv_val = parsed_event_dates[pid]
                show_event = (not event_only_when_reached) or (cutoff >= event_dt)
                if show_event:
                    # Solid vertical event line
                    ax_s.axvline(
                        event_dt,
                        color="#1f2d3d",
                        linestyle="-",
                        linewidth=1.6,
                        alpha=0.9,
                        zorder=5,
                    )
                    # Text label aligned to the right (placed to the left of the vertical line)
                    label_text = f"Avance de ≈ {adv_val} metros"
                    ax_s.text(
                        event_dt - pd.Timedelta(days=15),
                        event_label_y,
                        label_text,
                        color="#1f2d3d",
                        fontsize=9.5,
                        fontweight="bold",
                        ha="right",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="#ffffff",
                            edgecolor="#1f2d3d",
                            linewidth=0.8,
                            alpha=0.92,
                        ),
                        zorder=6,
                    )

            # Synchronized X limits and SHARED Y scale
            ax_s.set_xlim(scatter_min_date, scatter_max_date)
            ax_s.set_ylim(shared_scatter_ylim)
            ax_s.set_ylabel(y_axis_label, fontsize=10.5, fontweight="bold")
            ax_s.set_title(
                f"Serie temporal - Perfil {pid}",
                fontsize=13,
                fontweight="bold",
                pad=8,
            )
            ax_s.grid(True, linestyle="--", alpha=0.45, zorder=1)
            ax_s.tick_params(labelsize=9.5)

            # Remove black plot contour box (spines) for a clean modern look
            for spine in ax_s.spines.values():
                spine.set_visible(False)

        # X-label only on bottom scatter plot
        ax_scat_bottom.set_xlabel("Tiempo", fontsize=11, fontweight="bold")

        # Render frame to PIL Image (without plt.tight_layout to avoid UserWarnings)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)

        print(f"    [{idx + 1}/{len(periods)}] Frame {label_period} completed.")

    # 5. Save GIF with pause in the last frame
    print(f"--> Saving animation to {output_gif_path}...")
    output_path = Path(output_gif_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_duration_ms = int(1000.0 / fps)
    durations = [frame_duration_ms] * len(frames)
    if pause_last_frame_seconds > 0 and len(durations) > 0:
        durations[-1] = int(pause_last_frame_seconds * 1000.0)

    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )

    total_anim_duration_s = sum(durations) / 1000.0
    print(f"--> Animation successfully generated and saved at: {output_gif_path}")
    print(f"    Total frames: {len(frames)}, Duration: {total_anim_duration_s:.1f} s (including {pause_last_frame_seconds:.1f}s final pause)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate coastal evolution GIF animation."
    )
    parser.add_argument(
        "--shorelines",
        default="data/shorelines/LineasCosta_with_filter_flags.gpkg",
        help="Path to shorelines GPKG",
    )
    parser.add_argument(
        "--profiles",
        default="data/profiles/perfiles_recortados.gpkg",
        help="Path to profiles GPKG",
    )
    parser.add_argument(
        "--rois",
        default="coastal_evolution_animation/ROIs.gpkg",
        help="Path to ROIs GPKG",
    )
    parser.add_argument(
        "--seatool",
        default="S2_seatool.csv",
        help="Path to S2_seatool.csv",
    )
    parser.add_argument(
        "--output",
        default="coastal_evolution_animation/coastal_evolution.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--quadrant",
        type=int,
        default=61,
        help="Quadrant filter for shorelines (default: 61)",
    )
    parser.add_argument(
        "--target-profiles",
        nargs="+",
        type=int,
        default=[18844, 18756],
        help="Target profile IDs (default: 18844 18756)",
    )
    parser.add_argument(
        "--event-dates",
        type=str,
        default=None,
        help="JSON string with custom event dates dict, e.g. '{\"18844\": [\"2024-07-28 13:00:17\", 25], \"18756\": [\"2024-08-12 13:00:19\", 80]}'",
    )
    parser.add_argument(
        "--event-label",
        type=str,
        default="Avance de ≈ X metros",
        help="Label text above event line (default: 'Avance de ≈ X metros')",
    )
    parser.add_argument(
        "--show-events-always",
        dest="event_only_when_reached",
        action="store_false",
        default=True,
        help="Show event vertical lines from the beginning rather than only once the date is reached",
    )
    parser.add_argument(
        "--pause-last-frame",
        type=float,
        default=3.0,
        help="Pause duration on the last frame in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--transparency",
        type=float,
        default=0.70,
        help="Basemap transparency fraction (default: 0.70 = 70%% transparency)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second in GIF (default: 2.0)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure DPI (default: 120)",
    )

    args = parser.parse_args()

    parsed_events = None
    if args.event_dates:
        try:
            parsed_events = json.loads(args.event_dates)
            parsed_events = {int(k): v for k, v in parsed_events.items()}
        except Exception as e:
            print(f"Warning: could not parse --event-dates as JSON ({e}), using default dates.")
            parsed_events = None

    generate_animation(
        shorelines_path=args.shorelines,
        profiles_path=args.profiles,
        rois_path=args.rois,
        seatool_path=args.seatool,
        output_gif_path=args.output,
        quadrant=args.quadrant,
        target_profile_ids=args.target_profiles,
        event_dates=parsed_events,
        event_label_template=args.event_label,
        event_only_when_reached=args.event_only_when_reached,
        pause_last_frame_seconds=args.pause_last_frame,
        basemap_transparency=args.transparency,
        fps=args.fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
