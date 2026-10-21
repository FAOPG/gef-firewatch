"""Site buffers and boundary sections for firebreak planning.

Each site boundary is cut into sections of about one kilometre (longer for very large sites).
Fire alerts in the edge zone (from `fb_outer_km` outside the boundary to `fb_inner_km` inside it)
are given to the nearest section. Sections where fire comes back year after year are candidates
for firebreaks or early burning, to be checked on the ground.

    python pipeline/firebreak.py      # (re)builds inputs/firebreak_history.json

update.py calls build_history() itself when the file is missing, out of date, or a boundary changed.
"""
import datetime as dt, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree, force_2d
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring

sys.path.insert(0, os.path.dirname(__file__))
from gfw import query  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(ROOT, "inputs")
CFG = json.load(open(os.path.join(ROOT, "config.json")))
HIST = os.path.join(INP, "firebreak_history.json")
PV = "nasa_viirs_fire_alerts"


def utm_of(geom):
    return gpd.GeoSeries([geom], crs=4326).estimate_utm_crs()


def to_utm(geom, crs):
    return gpd.GeoSeries([geom], crs=4326).to_crs(crs).iloc[0]


def geom_hash(geom):
    return hashlib.md5(geom.wkb).hexdigest()[:12]


def buffer_geom(geom, km):
    """Site plus a buffer of km, measured in the site's UTM zone, returned in WGS84."""
    crs = utm_of(geom)
    g = to_utm(geom, crs).buffer(km * 1000).simplify(100)
    return gpd.GeoSeries([g], crs=crs).to_crs(4326).iloc[0]


def sections(geom):
    """Boundary sections in UTM: (crs, [LineString], section length in m)."""
    crs = utm_of(geom)
    g = force_2d(to_utm(geom, crs)).simplify(30)
    b = g.boundary
    lines = list(b.geoms) if isinstance(b, MultiLineString) else [b]
    per = sum(l.length for l in lines)
    L = max(CFG.get("fb_section_km", 1) * 1000, per / CFG.get("fb_max_sections", 300))
    out = []
    for l in lines:
        n = max(1, int(round(l.length / L)))
        step = l.length / n
        out += [substring(l, i * step, (i + 1) * step) for i in range(n)]
    return crs, g, out, L


def assign(geom, lon, lat):
    """For points (arrays), return section index (-1 outside the edge zone), zone code and signed
    distance to the boundary in m (negative inside).
    zone: 0 inside the site, 1 within 0–ring_km outside, 2 within ring_km–map_ring_km, 3 farther."""
    crs, g, segs, _ = sections(geom)
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(crs)
    inside = pts.within(g).values
    d = pts.distance(g.boundary).values
    outer, inner = CFG.get("fb_outer_km", 2) * 1000, CFG.get("fb_inner_km", 1) * 1000
    in_edge = np.where(inside, d <= inner, d <= outer)
    seg = np.full(len(pts), -1)
    if in_edge.any():
        tree = STRtree(segs)
        seg[in_edge] = tree.query_nearest(list(pts[in_edge]), all_matches=False)[1]
    r1, r2 = CFG["ring_km"] * 1000, CFG.get("map_ring_km", 10) * 1000
    zone = np.where(inside, 0, np.where(d <= r1, 1, np.where(d <= r2, 2, 3)))
    return seg, zone, np.where(inside, -d, d)


def year_points(gs, y, through):
    end = min(dt.date(y, 12, 31), through)
    rows = query(PV, f"SELECT latitude, longitude, alert__date FROM results WHERE alert__date >= '{y}-01-01' AND alert__date <= '{end}'", gs)
    return pd.DataFrame(rows)


def build_history(sites, last_year, log=print):
    """Alerts per boundary section and calendar year, baseline_first_year..last_year."""
    first = CFG["baseline_first_year"]
    old = json.load(open(HIST)) if os.path.exists(HIST) else {"sites": {}}
    out = {"first_year": first, "last_year": last_year, "outer_km": CFG.get("fb_outer_km", 2), "inner_km": CFG.get("fb_inner_km", 1), "sites": {}}
    todo = []
    for s in sites.itertuples():
        h = geom_hash(s.geometry)
        o = old["sites"].get(s.id)
        if o and o.get("hash") == h and old.get("first_year") == first and old.get("outer_km") == out["outer_km"] and old.get("inner_km") == out["inner_km"]:
            out["sites"][s.id] = o
            years = [y for y in range(first, last_year + 1) if str(y) not in o["years_done"]]
        else:
            out["sites"][s.id] = {"hash": h, "counts": {}, "years_done": []}
            years = list(range(first, last_year + 1))
        todo += [(s.id, s.geometry, s.geostore10, y) for y in years]
    log(f"firebreak history: {len(todo)} site-years to query")

    def job(t):
        sid, geom, gs, y = t
        df = year_points(gs, y, dt.date(last_year, 12, 31))
        if not len(df):
            return sid, y, {}
        seg, _, _ = assign(geom, df.longitude.values, df.latitude.values)
        u, n = np.unique(seg[seg >= 0], return_counts=True)
        return sid, y, {int(a): int(b) for a, b in zip(u, n)}

    with ThreadPoolExecutor(5) as ex:
        for sid, y, cnt in ex.map(job, todo):
            o = out["sites"][sid]
            for k, v in cnt.items():
                o["counts"].setdefault(str(k), {})[str(y)] = v
            o["years_done"] = sorted(set(o["years_done"]) | {str(y)})
    json.dump(out, open(HIST, "w"), separators=(",", ":"))
    log("firebreak history written")
    return out


if __name__ == "__main__":
    s = gpd.read_file(os.path.join(INP, "sites.geojson"))
    build_history(s, dt.date.today().year - 1)
