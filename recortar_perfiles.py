"""
recortar_perfiles.py
--------------------
Recorta cada perfil costero del archivo 'perfiles_limpios.gpkg' tomando como
punto central el valor del campo 'dist_lref' (distancia en metros desde el
inicio del perfil) y extendiéndose EXTENSION_M metros en cada dirección a lo
largo de la orientación original del perfil.

El resultado se exporta a 'perfiles_recortados.gpkg' (misma carpeta que el
archivo de entrada).

Uso:
    python recortar_perfiles.py

Parámetro configurable:
    EXTENSION_M  – metros a extender en cada dirección desde dist_lref (defecto: 100).
"""

import os
import geopandas as gpd
from shapely.geometry import LineString

# ---------------------------------------------------------------------------
# PARÁMETRO CONFIGURABLE
# ---------------------------------------------------------------------------
EXTENSION_M = 250          # metros a extender en cada dirección desde dist_lref

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
INPUT_GPKG  = os.path.join("data", "profiles", "perfiles_limpios.gpkg")
INPUT_LAYER = "perfiles_limpios"
OUTPUT_GPKG  = os.path.join("data", "profiles", "perfiles_recortados.gpkg")
OUTPUT_LAYER = "perfiles_recortados"


def interpolate_point_along_line(line: LineString, distance: float):
    """
    Devuelve el punto situado a 'distance' metros desde el inicio de 'line'.
    Shapely garantiza que project/interpolate respetan el orden de los vértices
    (inicio → fin), por lo que el sentido del perfil se conserva.
    """
    # Clamp a los extremos para evitar problemas numéricos
    distance = max(0.0, min(distance, line.length))
    return line.interpolate(distance)


def clip_line_between_distances(line: LineString,
                                dist_start: float,
                                dist_end: float) -> LineString:
    """
    Extrae la subsección de 'line' comprendida entre dist_start y dist_end
    (ambas medidas en metros desde el inicio del perfil).

    El algoritmo recorre los vértices del LineString original y reconstruye
    únicamente los que caen dentro del intervalo, interpolando los extremos
    exactos para no desplazar el resultado.
    """
    total = line.length
    # Clamp al rango real de la línea
    d0 = max(0.0, dist_start)
    d1 = min(total, dist_end)

    if d0 >= d1:
        # Intervalo degenerado: devolvemos un segmento mínimo en el punto central
        pt = line.interpolate(max(d0, 0.0))
        return LineString([pt, pt])

    # Punto de inicio y fin exactos (interpolados sobre la línea)
    pt_start = line.interpolate(d0)
    pt_end   = line.interpolate(d1)

    # Acumular distancias de cada vértice original
    coords = list(line.coords)
    accumulated = 0.0
    result_coords = [pt_start]

    for i in range(len(coords) - 1):
        segment_len = (
            LineString([coords[i], coords[i + 1]]).length
        )
        d_vertex_next = accumulated + segment_len

        # El vértice coords[i+1] está dentro del intervalo
        if d0 < d_vertex_next <= d1:
            result_coords.append(coords[i + 1])

        accumulated = d_vertex_next

    result_coords.append(pt_end)

    # Eliminar duplicados consecutivos (puede ocurrir si pt_start o pt_end
    # coinciden exactamente con un vértice original)
    cleaned = [result_coords[0]]
    for pt in result_coords[1:]:
        if pt != cleaned[-1]:
            cleaned.append(pt)

    if len(cleaned) < 2:
        cleaned = [pt_start, pt_end]

    return LineString(cleaned)


def recortar_perfiles(input_gpkg: str,
                      input_layer: str,
                      output_gpkg: str,
                      output_layer: str,
                      extension_m: float = EXTENSION_M) -> None:
    """
    Lee los perfiles, los recorta y guarda el resultado.

    Parameters
    ----------
    input_gpkg  : ruta al GeoPackage de entrada.
    input_layer : nombre de la capa dentro del GeoPackage de entrada.
    output_gpkg : ruta al GeoPackage de salida.
    output_layer: nombre de la capa dentro del GeoPackage de salida.
    extension_m : metros a extender en cada dirección desde dist_lref.
    """
    print(f"Leyendo '{input_layer}' desde '{input_gpkg}'…")
    gdf = gpd.read_file(input_gpkg, layer=input_layer)
    print(f"  {len(gdf)} perfiles leídos. CRS: {gdf.crs}")

    if "dist_lref" not in gdf.columns:
        raise ValueError(
            "El campo 'dist_lref' no existe en la capa. "
            f"Columnas disponibles: {gdf.columns.tolist()}"
        )

    n_skipped = 0
    new_geoms = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        dist_lref = row["dist_lref"]

        # Comprobaciones básicas
        if geom is None or geom.is_empty:
            print(f"  [AVISO] Perfil idx={idx} tiene geometría nula/vacía. Se conserva sin cambios.")
            new_geoms.append(geom)
            n_skipped += 1
            continue

        if geom.geom_type != "LineString":
            print(
                f"  [AVISO] Perfil idx={idx} tiene tipo '{geom.geom_type}' "
                "(se esperaba LineString). Se conserva sin cambios."
            )
            new_geoms.append(geom)
            n_skipped += 1
            continue

        if dist_lref is None or (dist_lref != dist_lref):  # NaN check
            print(f"  [AVISO] Perfil idx={idx} tiene 'dist_lref' nulo. Se conserva sin cambios.")
            new_geoms.append(geom)
            n_skipped += 1
            continue

        total_length = geom.length
        d_centro = float(dist_lref)
        d_inicio = d_centro - extension_m
        d_fin    = d_centro + extension_m

        # Advertir si el recorte queda fuera del perfil original
        if d_inicio < 0 or d_fin > total_length:
            print(
                f"  [AVISO] Perfil idx={idx} (longitud={total_length:.1f} m, "
                f"dist_lref={d_centro:.1f} m): el recorte [{d_inicio:.1f}, {d_fin:.1f}] "
                "supera los límites del perfil. Se recortará hasta el extremo disponible."
            )

        new_geom = clip_line_between_distances(geom, d_inicio, d_fin)
        new_geoms.append(new_geom)

    gdf = gdf.copy()
    gdf["geometry"] = new_geoms

    print(f"\nExportando resultado a '{output_gpkg}' (capa '{output_layer}')…")
    gdf.to_file(output_gpkg, layer=output_layer, driver="GPKG")
    print(f"  Hecho. {len(gdf) - n_skipped} perfiles recortados, {n_skipped} sin cambios.")
    print(f"  Extensión aplicada: ±{extension_m} m desde dist_lref.")


if __name__ == "__main__":
    recortar_perfiles(
        input_gpkg=INPUT_GPKG,
        input_layer=INPUT_LAYER,
        output_gpkg=OUTPUT_GPKG,
        output_layer=OUTPUT_LAYER,
        extension_m=EXTENSION_M,
    )
