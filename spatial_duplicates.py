"""Spatial duplicate detection and removal for line GeoDataFrames."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

_LINE_TYPES = {"LineString", "MultiLineString"}


def drop_spatial_duplicates(
    gdf: gpd.GeoDataFrame,
    tolerance: float = 0.0,
    keep: str = "first",
    drop_empty: bool = False,
    return_duplicates: bool = False,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Remove duplicated line geometries using a purely spatial criterion.

    Attribute values are ignored. Two features are duplicates when their
    geometries are the same after:
      * normalizing vertex order / direction (A->B equals B->A),
      * normalizing the order of parts in MultiLineStrings,
      * (optionally) snapping coordinates to a grid of size `tolerance`.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data with LineString / MultiLineString geometries.
    tolerance : float, default 0.0
        Grid size, in CRS units, used to snap coordinates before comparing.
        0 means exact coordinate comparison. Use e.g. 0.01 to treat lines
        that differ by less than ~1 cm (in a metric CRS) as duplicates.
    keep : {"first", "last"}, default "first"
        Which occurrence to keep within each group of duplicates.
    drop_empty : bool, default False
        If True, null/empty geometries are also removed.
        If False, they are kept untouched and never flagged as duplicates.
    return_duplicates : bool, default False
        If True, also return a GeoDataFrame with the removed features.

    Returns
    -------
    GeoDataFrame, or (GeoDataFrame, GeoDataFrame) if `return_duplicates`.
    The original index is preserved.
    """
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'.")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0.")

    geom_types = set(gdf.geom_type.dropna().unique())
    if not geom_types <= _LINE_TYPES:
        raise TypeError(
            f"Only LineString/MultiLineString are supported, got: {sorted(geom_types)}"
        )

    geoms = np.asarray(gdf.geometry.values, dtype=object)
    missing = shapely.is_missing(geoms) | shapely.is_empty(geoms)
    valid = ~missing

    work = geoms[valid]
    if tolerance > 0:
        # "pointwise" only rounds coordinates: it never drops or rebuilds
        # geometries, so collapsed lines still get a comparable key.
        work = shapely.set_precision(work, grid_size=tolerance, mode="pointwise")

    # normalize() gives a canonical form (direction, ring/part order), so
    # equal geometries produce identical WKB bytes.
    work = shapely.normalize(work)
    keys = shapely.to_wkb(work, output_dimension=2)  # ignore Z on purpose

    is_dup = np.zeros(len(gdf), dtype=bool)
    is_dup[valid] = pd.Series(keys).duplicated(keep=keep).to_numpy()
    if drop_empty:
        is_dup |= missing

    cleaned = gdf.loc[~is_dup].copy()
    if return_duplicates:
        return cleaned, gdf.loc[is_dup].copy()
    return cleaned


if __name__ == "__main__":
    import os
    from shapely.geometry import LineString

    profiles = gpd.read_file(os.path.join("data", "profiles", "perfiles_playas_limpios",  "all_profiles_to_clean.shp"))
    # Sort profiles by 'ID_perfil' to ensure consistent order for duplicate removal
    profiles = profiles.sort_values(by="ID_Perfil").reset_index(drop=True)
    profiles['ID_Perfil'] = profiles['ID_Perfil'].astype(int)  # Ensure 'ID_Perfil' is of integer type

    # Remove duplicates with a tolerance of 0.01 (1 cm in metric CRS)
    cleaned, duplicates = drop_spatial_duplicates(
        profiles, tolerance=0.01, keep="first", drop_empty=True, return_duplicates=True
    )

    print(f"Removed {len(duplicates)} duplicates.")
    print(f"Kept {len(cleaned)} unique profiles.")

    # Check the province (or provinces) of the duplicates
    if not duplicates.empty:
        provinces = duplicates["provincia"].unique()
        print(f"Duplicates found in provinces: {', '.join(map(str, provinces))}")

    # Drop automatic attributes generated after running Merge geoprocess in QGIS
    cleaned = cleaned.drop(columns=['OBJECTID_1', 'layer', 'path'])
    
    # Save the cleaned data to a new shapefile
    cleaned.to_file(os.path.join("data", "profiles", "cleaned_profiles_no_duplicates.shp"))

