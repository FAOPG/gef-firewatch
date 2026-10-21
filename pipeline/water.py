"""Mapped rivers and lakes (HydroSHEDS) for finding the nearest water to each fire alert.

inputs/water.json is made once from HydroRIVERS v1.0 (Australasia) and HydroLAKES v1.0, clipped to
the four project provinces and 12 km around every site:

    python pipeline/water.py HydroRIVERS_v10_au.shp HydroLAKES_polys_v10.shp

HydroRIVERS holds rivers with an upstream catchment of at least 10 km²; HydroLAKES holds lakes of
at least 10 ha. Small creeks, springs, tanks and village water points are not mapped.
Source: Lehner & Grill (2013), HydroSHEDS, www.hydrosheds.org (CC BY 4.0).

Compact format: {"rivers": [[order, flow_m3s, upland_km2, x0, y0, dx1, dy1, ...]], "lakes": [[name, area_km2, x0, y0, dx1, ...]]}
with coordinates in 1e-4 degrees, delta-encoded.
"""
import json, os, sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(ROOT, "inputs")
SRC = os.path.join(INP, "water.json")


def enc(coords):
    a = np.round(np.asarray(coords)[:, :2] * 1e4).astype(int)
    d = np.vstack([a[:1], np.diff(a, axis=0)])
    d = d[np.any(d != 0, axis=1) | (np.arange(len(d)) == 0)]
    return d.ravel().tolist()


def dec(v):
    a = np.cumsum(np.asarray(v, dtype=float).reshape(-1, 2), axis=0) / 1e4
    return a


def prepare(rivers_shp, lakes_shp):
    prov = gpd.read_file(os.path.join(INP, "provinces.geojson"))
    sites = gpd.read_file(os.path.join(INP, "sites.geojson"))
    aoi = pd.concat([prov.geometry, sites.to_crs(3857).buffer(12000).to_crs(4326)]).union_all()
    b = aoi.bounds
    R = gpd.read_file(rivers_shp, bbox=b)
    R = R[R.intersects(aoi)]
    L = gpd.read_file(lakes_shp, bbox=b)
    L = L[L.intersects(aoi)]
    rv = []
    for r in R.itertuples():
        g = r.geometry.simplify(0.0003)
        lines = g.geoms if g.geom_type == "MultiLineString" else [g]
        for ln in lines:
            rv.append([int(r.ORD_STRA), round(float(r.DIS_AV_CMS), 2), round(float(r.UPLAND_SKM))] + enc(ln.coords))
    lk = []
    for r in L.itertuples():
        g = r.geometry.simplify(0.0003)
        polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
        for p in polys:
            lk.append([r.Lake_name if isinstance(r.Lake_name, str) else "", round(float(r.Lake_area), 2)] + enc(p.exterior.coords))
    json.dump({"source": "HydroRIVERS v1.0 and HydroLAKES v1.0 (HydroSHEDS, CC BY 4.0)", "rivers": rv, "lakes": lk},
              open(SRC, "w"), separators=(",", ":"))
    print(len(rv), "rivers,", len(lk), "lakes ->", SRC, os.path.getsize(SRC) // 1024, "KB")


def load():
    """GeoDataFrame of water features, index = position in rivers then lakes (as published)."""
    j = json.load(open(SRC))
    rows = []
    for r in j["rivers"]:
        rows.append({"kind": "river", "order": r[0], "flow": r[1], "upland": r[2], "name": "", "geometry": LineString(dec(r[3:]))})
    for l in j["lakes"]:
        xy = dec(l[2:])
        rows.append({"kind": "lake", "order": None, "flow": None, "upland": None, "name": l[0], "area": l[1],
                     "geometry": Polygon(xy) if len(xy) >= 3 else LineString(xy)})
    return gpd.GeoDataFrame(rows, crs=4326)


def nearest(lon, lat, W, max_km):
    """Nearest mapped water for each point: (index into W or -1, distance in m)."""
    if not len(lon):
        return np.array([], int), np.array([], float)
    P = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.asarray(lon, float), np.asarray(lat, float)), crs=4326).to_crs(3857)
    W3 = W[["geometry"]].to_crs(3857)
    k = 1 / np.cos(np.radians(-6.5))  # Web Mercator scale near PNG
    j = gpd.sjoin_nearest(P, W3, how="left", max_distance=max_km * 1000 * k, distance_col="dist")
    j = j[~j.index.duplicated()]
    idx = j["index_right"].fillna(-1).astype(int).values
    # true distance: Mercator distance times cos(latitude)
    dist = (j["dist"].values * np.cos(np.radians(np.asarray(lat, dtype=float))))
    dist = np.where(idx >= 0, dist, np.nan)
    return idx, dist


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2])
