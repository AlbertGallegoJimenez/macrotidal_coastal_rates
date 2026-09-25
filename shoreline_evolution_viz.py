"""Interactive shoreline evolution visualization for Jupyter notebooks."""

def create_shoreline_dashboard(
    profiles,
    shorelines,
    profiles_shoreline_intersections,
    display_widget: bool = False,
):
    """Create the shoreline evolution widget dashboard.

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
    display_widget : bool, default False
        Whether to display the dashboard immediately after creation.

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
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde, linregress
    from IPython.display import clear_output, display

    intersections = profiles_shoreline_intersections.copy()
    intersections["clean_date"] = pd.to_datetime(
        intersections["Fecha"].astype(str).str[:8],
        format="%Y%m%d",
        errors="coerce",
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
    output_histogram = widgets.Output()
    output_map = widgets.Output()

    def update_beach_options(*args):
        beaches = sorted(intersections["Playa"].dropna().unique())
        dropdown_beach.options = beaches
        if beaches:
            dropdown_beach.value = beaches[0]

    def update_profile_options(*args):
        valid_profiles = sorted(
            intersections[
                intersections["Playa"] == dropdown_beach.value
            ]["profile_id"].dropna().unique()
        )
        select_profiles.options = valid_profiles
        select_profiles.value = tuple()

    def render_dashboard(_button):
        beach = dropdown_beach.value
        level_min, level_max = slider_sea_level.value
        selected_profiles = select_profiles.value
        start_date = pd.to_datetime(date_start.value)
        end_date = pd.to_datetime(date_end.value)

        valid_profile_ids = intersections[
            intersections["Playa"] == beach
        ]["profile_id"].unique()
        active_profiles = selected_profiles or valid_profile_ids

        df_beach = intersections[
            (intersections["Playa"] == beach)
            & intersections["clean_date"].notna()
            & (intersections["clean_date"] >= start_date)
            & (intersections["clean_date"] <= end_date)
            & intersections["profile_id"].isin(active_profiles)
        ].sort_values(by="clean_date")

        all_levels = intersections[
            intersections["Playa"] == beach
        ]["NivelTotal"].dropna()
        filtered_levels = df_beach[
            (df_beach["NivelTotal"] >= level_min)
            & (df_beach["NivelTotal"] <= level_max)
        ]["NivelTotal"].dropna()

        profiles_filt = profiles[profiles["profile_id"].isin(valid_profile_ids)]
        shorelines_filt = shorelines[
            (shorelines["Playa"] == beach)
            & (shorelines["NivelTotal"] >= level_min)
            & (shorelines["NivelTotal"] <= level_max)
        ]
        profiles_wgs84 = profiles_filt.to_crs(epsg=4326)
        shorelines_wgs84 = shorelines_filt.to_crs(epsg=4326)

        with output_histogram:
            clear_output(wait=True)
            if all_levels.empty:
                print("No sea-level data available for the selected beach.")
            else:
                x_min = float(all_levels.min())
                x_max = float(all_levels.max())
                bin_size = (x_max - x_min) / 30 if x_min < x_max else 1.0
                shared_bins = dict(
                    start=x_min,
                    end=x_max + bin_size,
                    size=bin_size,
                )
                histogram = go.Figure()
                histogram.add_trace(
                    go.Histogram(
                        x=all_levels,
                        histnorm="probability density",
                        xbins=shared_bins,
                        marker_color="lightgray",
                        marker_line_color="white",
                        marker_line_width=0.5,
                        name="All data",
                        opacity=0.75,
                    )
                )
                histogram.add_trace(
                    go.Histogram(
                        x=filtered_levels,
                        histnorm="probability density",
                        xbins=shared_bins,
                        marker_color="#1976D2",
                        marker_line_color="white",
                        marker_line_width=0.5,
                        name="Filtered data",
                        opacity=0.65,
                    )
                )

                kde_data = [(all_levels, "All data KDE", "#616161")]
                if not filtered_levels.empty:
                    kde_data.append(
                        (filtered_levels, "Filtered data KDE", "#1976D2")
                    )

                if x_min < x_max:
                    x_values = np.linspace(x_min, x_max, 200)
                    for values, name, color in kde_data:
                        if len(values) > 1 and values.nunique() > 1:
                            density = gaussian_kde(values)(x_values)
                            histogram.add_trace(
                                go.Scatter(
                                    x=x_values,
                                    y=density,
                                    mode="lines",
                                    line=dict(color=color, width=2),
                                    name=name,
                                )
                            )

                histogram.update_layout(
                    title=f"Sea Level Distribution - {beach}",
                    xaxis_title="Sea Level (m)",
                    yaxis_title="Density",
                    barmode="overlay",
                    template="plotly_white",
                    hovermode="x unified",
                )
                histogram.show()

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
                        # 1. Dibujar los puntos filtrados
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
                        
                        # 2. Ajuste lineal y estadísticos si solo hay 1 perfil seleccionado
                        if len(active_profiles) == 1 and len(in_range) > 2:
                            # Convertir fechas a años decimales
                            dias_desde_origen = (in_range["clean_date"] - pd.Timestamp("1970-01-01")).dt.days
                            x_years = dias_desde_origen / 365.2425
                            y_pos = in_range["shoreline_position_m"]

                            # Get the number of observations
                            n_obs = len(in_range)
                            
                            # Regresión lineal con SciPy
                            res = linregress(x_years, y_pos)
                            y_fit = res.slope * x_years + res.intercept
                            
                            # Cálculos de bondad de ajuste
                            r_squared = res.rvalue**2
                            p_value = res.pvalue
                            ci_95 = 1.96 * res.stderr  # Intervalo de confianza al 95%
                            
                            # Evaluar la fiabilidad estadística
                            is_significant = p_value < 0.05
                            sig_color = "green" if is_significant else "red"
                            sig_text = "Significativo" if is_significant else "No significativo"
                            
                            # Trazar la línea de ajuste
                            figure.add_trace(
                                go.Scatter(
                                    x=in_range["clean_date"],
                                    y=y_fit,
                                    mode="lines",
                                    line=dict(color=sig_color, width=2, dash="dash"),
                                    name=f"Trend OLS",
                                )
                            )
                            
                            # Preparar y añadir el recuadro con las métricas
                            annotation_text = (
                                f"<b>Observaciones:</b> {n_obs}<br>"
                                f"<b>Tasa:</b> {res.slope:.2f} ± {ci_95:.2f} m/año<br>"
                                f"<b>R²:</b> {r_squared:.2f}<br>"
                                f"<b>p-valor:</b> {p_value:.3f} (<span style='color:{sig_color}'><b>{sig_text}</b></span>)"
                            )
                            
                            figure.add_annotation(
                                text=annotation_text,
                                xref="paper", yref="paper",
                                x=0.02, y=0.95,
                                showarrow=False,
                                bgcolor="rgba(255, 255, 255, 0.9)",
                                bordercolor="black",
                                borderwidth=1,
                                font=dict(size=13, color="black"),
                                align="left"
                            )

                    if not out_range.empty:
                        # 3. Dibujar los puntos fuera de rango (descartados por marea)
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

    dropdown_beach.observe(update_profile_options, "value")
    button_update.on_click(render_dashboard)

    update_beach_options()
    update_profile_options()
    dashboard = widgets.VBox(
        [
            widgets.HBox([dropdown_beach, select_profiles]),
            widgets.HBox([date_start, date_end]),
            slider_sea_level,
            button_update,
            widgets.VBox([output_histogram, output_plot, output_map]),
        ]
    )
    if display_widget:
        display(dashboard)
    return dashboard


__all__ = ["create_shoreline_dashboard"]