"""Interactive shoreline evolution visualization for Jupyter notebooks."""

def create_shoreline_dashboard(
    profiles,
    shorelines,
    profiles_shoreline_intersections,
):
    """Create and display the shoreline evolution widget dashboard.

    Parameters
    ----------
    profiles : geopandas.GeoDataFrame
        Coastal profile geometries. Must contain ``profile_id``.
    shorelines : geopandas.GeoDataFrame
        Shoreline geometries and attributes, including ``Municipio``,
        ``Playa`` and ``NivelTotal``.
    profiles_shoreline_intersections : geopandas.GeoDataFrame
        Point intersections with ``profile_id``, ``Municipio``, ``Playa``,
        ``Fecha``, ``NivelTotal`` and ``shoreline_position_m`` columns.

    Returns
    -------
    ipywidgets.Widget
        The complete dashboard container.
    """
    import folium
    import ipywidgets as widgets
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from IPython.display import clear_output, display

    intersections = profiles_shoreline_intersections.copy()
    intersections["clean_date"] = pd.to_datetime(
        intersections["Fecha"].astype(str).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )

    municipalities = sorted(intersections["Municipio"].dropna().unique())
    dropdown_municipality = widgets.Dropdown(
        options=municipalities,
        description="Municipality:",
    )
    dropdown_beach = widgets.Dropdown(description="Beach:")
    select_profiles = widgets.SelectMultiple(
        description="Profiles:",
        tooltip="Use Ctrl/Cmd to select multiple",
    )

    min_level = float(intersections["NivelTotal"].min())
    max_level = float(intersections["NivelTotal"].max())
    slider_sea_level = widgets.FloatRangeSlider(
        value=[min_level, max_level],
        min=min_level,
        max=max_level,
        step=0.1,
        description="Sea Level (m):",
        layout=widgets.Layout(width="50%"),
    )

    min_date = intersections["clean_date"].min().date()
    max_date = intersections["clean_date"].max().date()
    date_start = widgets.DatePicker(description="Start Date:", value=min_date)
    date_end = widgets.DatePicker(description="End Date:", value=max_date)
    button_update = widgets.Button(
        description="Render Dashboard",
        button_style="primary",
    )
    output_plot = widgets.Output()
    output_map = widgets.Output()

    def update_beach_options(*args):
        selected_municipality = dropdown_municipality.value
        beaches = sorted(
            intersections[
                intersections["Municipio"] == selected_municipality
            ]["Playa"].dropna().unique()
        )
        dropdown_beach.options = beaches
        if beaches:
            dropdown_beach.value = beaches[0]

    def update_profile_options(*args):
        valid_profiles = sorted(
            intersections[
                (intersections["Municipio"] == dropdown_municipality.value)
                & (intersections["Playa"] == dropdown_beach.value)
            ]["profile_id"].dropna().unique()
        )
        select_profiles.options = valid_profiles
        select_profiles.value = tuple()

    def render_dashboard(_button):
        municipality = dropdown_municipality.value
        beach = dropdown_beach.value
        level_min, level_max = slider_sea_level.value
        selected_profiles = select_profiles.value
        start_date = pd.to_datetime(date_start.value)
        end_date = pd.to_datetime(date_end.value)

        valid_profile_ids = intersections[
            (intersections["Municipio"] == municipality)
            & (intersections["Playa"] == beach)
        ]["profile_id"].unique()
        active_profiles = selected_profiles or valid_profile_ids

        df_beach = intersections[
            (intersections["Municipio"] == municipality)
            & (intersections["Playa"] == beach)
            & intersections["clean_date"].notna()
            & (intersections["clean_date"] >= start_date)
            & (intersections["clean_date"] <= end_date)
            & intersections["profile_id"].isin(active_profiles)
        ].sort_values(by="clean_date")

        profiles_filt = profiles[profiles["profile_id"].isin(valid_profile_ids)]
        shorelines_filt = shorelines[
            (shorelines["Municipio"] == municipality)
            & (shorelines["Playa"] == beach)
            & (shorelines["NivelTotal"] >= level_min)
            & (shorelines["NivelTotal"] <= level_max)
        ]
        profiles_wgs84 = profiles_filt.to_crs(epsg=4326)
        shorelines_wgs84 = shorelines_filt.to_crs(epsg=4326)

        with output_plot:
            clear_output(wait=True)
            if df_beach.empty:
                print("No data available for the selected parameters.")
            else:
                figure = go.Figure()
                colors = px.colors.qualitative.Plotly
                for index, profile_id in enumerate(active_profiles):
                    profile_data = df_beach[
                        df_beach["profile_id"] == profile_id
                    ]
                    in_range = profile_data[
                        (profile_data["NivelTotal"] >= level_min)
                        & (profile_data["NivelTotal"] <= level_max)
                    ]
                    out_range = profile_data[
                        (profile_data["NivelTotal"] < level_min)
                        | (profile_data["NivelTotal"] > level_max)
                    ]
                    marker_color = colors[index % len(colors)]

                    if not in_range.empty:
                        figure.add_trace(
                            go.Scatter(
                                x=in_range["clean_date"],
                                y=in_range["shoreline_position_m"],
                                mode="markers",
                                marker=dict(
                                    color=marker_color,
                                    size=7,
                                    line=dict(width=0.5, color="white"),
                                ),
                                name=f"Profile {profile_id}",
                            )
                        )
                    if not out_range.empty:
                        figure.add_trace(
                            go.Scatter(
                                x=out_range["clean_date"],
                                y=out_range["shoreline_position_m"],
                                mode="markers",
                                marker=dict(color="grey", size=5, opacity=0.25),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

                figure.update_layout(
                    title=f"Shoreline Evolution - {beach}",
                    xaxis_title="Date",
                    yaxis_title="Cross-shore Position (m)",
                    hovermode="closest",
                    template="plotly_white",
                )
                figure.show()

        with output_map:
            clear_output(wait=True)
            if profiles_wgs84.empty:
                print("No spatial data available to render the map.")
                return

            bounds = profiles_wgs84.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            map_view = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
            )
            folium.TileLayer(
                tiles=(
                    "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ),
                attr="Esri",
                name="Esri Satellite",
                overlay=False,
                control=True,
            ).add_to(map_view)

            for _, row in profiles_wgs84.iterrows():
                is_selected = (
                    row["profile_id"] in selected_profiles
                    if selected_profiles
                    else False
                )
                line_color = "red" if is_selected else "white"
                line_weight = 4 if is_selected else 2
                line_opacity = 1.0 if is_selected else 0.7
                profile_layer = folium.GeoJson(
                    row.geometry,
                    style_function=lambda _feature, color=line_color, weight=line_weight, opacity=line_opacity, selected=is_selected: {
                        "color": color,
                        "weight": weight,
                        "opacity": opacity,
                        "dashArray": None if selected else "5, 5",
                    },
                )
                folium.Tooltip(
                    f"Profile ID: {row['profile_id']}"
                ).add_to(profile_layer)
                profile_layer.add_to(map_view)

            for _, row in shorelines_wgs84.iterrows():
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda _feature: {
                        "color": "cyan",
                        "weight": 1.5,
                        "opacity": 0.6,
                    },
                ).add_to(map_view)
            display(map_view)

    dropdown_municipality.observe(update_beach_options, "value")
    dropdown_beach.observe(update_profile_options, "value")
    button_update.on_click(render_dashboard)

    update_beach_options()
    update_profile_options()
    dashboard = widgets.VBox(
        [
            widgets.HBox(
                [dropdown_municipality, dropdown_beach, select_profiles]
            ),
            widgets.HBox([date_start, date_end]),
            slider_sea_level,
            button_update,
            widgets.VBox([output_plot, output_map]),
        ]
    )
    display(dashboard)
    return dashboard


__all__ = ["create_shoreline_dashboard"]