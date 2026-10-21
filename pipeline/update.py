"""Daily data update for GEF Fire Watch PNG.

Queries the GFW Data API for the four project provinces and the 40 project sites, then writes
the website's data files into the build folder (default _site/). Maps and PDF reports are made
afterwards by outputs.py.

Every count is made twice: with all alerts, and without low-confidence alerts (the "nl" fields),
so the dashboard can switch between them.

    python pipeline/update.py --out _site [--force]

If the live site (config "site_url" or env SITE_URL) already shows the latest alert date, the
script writes "changed=false" to $GITHUB_OUTPUT and stops, unless --force is given.
"""
import argparse, datetime as dt, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import mapping

sys.path.insert(0, os.path.dirname(__file__))
from gfw import query, create_geostore  # noqa: E402
import firebreak as FB  # noqa: E402
import water as WT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(ROOT, "inputs")
CFG = json.load(open(os.path.join(ROOT, "config.json")))
PV = "nasa_viirs_fire_alerts"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gh_output(k, v):
    p = os.environ.get("GITHUB_OUTPUT")
    if p:
        with open(p, "a") as f:
            f.write(f"{k}={v}\n")


def shift_year(d, years):
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year + years, day=28)


def season_bounds(latest):
    mm, dd = map(int, CFG["season_start"].split("-"))
    start = dt.date(latest.year, mm, dd)
    if latest < start:
        start = dt.date(latest.year - 1, mm, dd)
    return start, latest


def window_counts(daily, start, end, first_year, k=0):
    """Alerts in the same calendar window (start..end) for every year since first_year.
    daily: {date: [all, not_low]}; k picks the component."""
    s = pd.Series({d: v[k] for d, v in daily.items()}, dtype=float)
    s.index = pd.to_datetime(s.index)
    out = {}
    span_years = end.year - start.year
    for y in range(first_year, start.year + 1):
        k = y - start.year
        a, b = shift_year(start, k), shift_year(end, k)
        out[str(y + span_years) if span_years else str(y)] = int(s[(s.index >= str(a)) & (s.index <= str(b))].sum())
    return out


def daily_counts(geostore, since):
    """{date: [all alerts, alerts that are not low confidence]}"""
    rows = query(PV, f"SELECT alert__date, confidence__cat, COUNT(*) AS n FROM results WHERE alert__date >= '{since}' "
                 "GROUP BY alert__date, confidence__cat", geostore)
    out = {}
    for r in rows:
        n = int(r.get("n", r.get("count", 0)))
        v = out.setdefault(r["alert__date"], [0, 0])
        v[0] += n
        if r.get("confidence__cat") != "l":
            v[1] += n
    return out


def compare(yrs):
    """Current season against earlier years."""
    cur_key = max(yrs)
    hist = [v for k, v in yrs.items() if k != cur_key]
    mean = float(np.mean(hist)) if hist else 0.0
    cur = yrs[cur_key]
    return {"years": yrs, "fire_now": cur, "fire_mean": round(mean, 1), "fire_2015": yrs.get("2015", 0), "fire_max": max(hist) if hist else 0,
            "ratio": round(cur / mean, 1) if mean > 0 else None, "fire_rank": 1 + sum(h > cur for h in hist)}


def month_keys(start, end):
    keys, d = [], dt.date(start.year, start.month, 1)
    while d <= end:
        keys.append(d.strftime("%Y-%m"))
        d = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "_site"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = a.out
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    os.makedirs(os.path.join(out, "downloads"), exist_ok=True)

    prov = gpd.read_file(os.path.join(INP, "provinces.geojson"))
    dist = gpd.read_file(os.path.join(INP, "districts.geojson"))
    sites = gpd.read_file(os.path.join(INP, "sites.geojson"))

    # 1. latest data date
    latest = max(query(PV, "SELECT MAX(alert__date) AS d FROM results", g)[0].get("d") or "1900-01-01" for g in prov.geostore)
    latest = dt.date.fromisoformat(latest[:10])
    start, end = season_bounds(latest)
    log("latest alert date", latest, "season", start, "to", end)

    site_url = os.environ.get("SITE_URL") or CFG.get("site_url") or ""
    if site_url and not a.force:
        try:
            live = requests.get(site_url.rstrip("/") + "/data/stats.json", timeout=60).json()
            if live.get("season", {}).get("end") == str(end):
                log("live site already has data to", end, "- nothing to do")
                gh_output("changed", "false")
                return
        except Exception as e:
            log("could not read live site, building anyway:", e)
    gh_output("changed", "true")

    first = CFG["baseline_first_year"]
    since = f"{first}-01-01"
    months = month_keys(start, end)

    # 2. fire points for the season, by province and month
    cols = ("latitude, longitude, alert__date, alert__time_utc, confidence__cat, "
            "umd_tree_cover_density_2000__threshold, is__umd_regional_primary_forest_2001, is__ifl_intact_forest_landscapes_2016")
    colnames = [c.strip() for c in cols.split(",")]
    key = ["latitude", "longitude", "alert__date", "alert__time_utc"]
    jobs = []
    for p in prov.itertuples():
        for m in months:
            m0 = max(dt.date.fromisoformat(m + "-01"), start)
            nxt = dt.date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
            jobs.append((p.name, p.geostore, str(m0), str(min(nxt - dt.timedelta(days=1), end))))

    def get_pts(j):
        name, g, d0, d1 = j
        rows = query(PV, f"SELECT {cols} FROM results WHERE alert__date >= '{d0}' AND alert__date <= '{d1}'", g)
        return pd.DataFrame(rows, columns=colnames)

    with ThreadPoolExecutor(4) as ex:
        frames = list(ex.map(get_pts, jobs))
    pts = pd.concat(frames, ignore_index=True).drop_duplicates(subset=key)
    P = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.longitude, pts.latitude), crs=4326)
    P = gpd.sjoin(P, prov[["name", "geometry"]].rename(columns={"name": "province"}), predicate="within", how="inner").drop(columns="index_right")
    P = P.drop_duplicates(subset=key)
    log("fire points in the four provinces", len(P))

    # 3. sites: geostores for the boundary and for boundary + map_ring_km
    sites = sites.copy()
    if "geostore10" not in sites.columns:
        sites["geostore10"] = None
    changed_gs = False
    for i, s in sites.iterrows():
        if not s.get("geostore"):
            sites.at[i, "geostore"] = create_geostore(mapping(s.geometry.simplify(0.0005)))
            changed_gs = True
        if not s.get("geostore10"):
            sites.at[i, "geostore10"] = create_geostore(mapping(FB.buffer_geom(s.geometry, CFG.get("map_ring_km", 10))))
            changed_gs = True
    if changed_gs:
        sites.to_file(os.path.join(INP, "sites.geojson"), driver="GeoJSON")

    # 4. season points within map_ring_km of every site (they can fall outside the four provinces)
    def site_pts(s):
        rows = query(PV, f"SELECT {cols} FROM results WHERE alert__date >= '{start}' AND alert__date <= '{end}'", s.geostore10)
        return s.id, pd.DataFrame(rows, columns=colnames)
    with ThreadPoolExecutor(5) as ex:
        SP = dict(ex.map(site_pts, list(sites.itertuples())))
    extra = pd.concat(list(SP.values()), ignore_index=True).drop_duplicates(subset=key)
    extra = extra.merge(P[key], on=key, how="left", indicator=True)
    extra = extra[extra._merge == "left_only"].drop(columns="_merge")
    if len(extra):
        allp = gpd.read_file(os.path.join(INP, "png_provinces_all.geojson"))[["NAME_1", "geometry"]].rename(columns={"NAME_1": "province"})
        E = gpd.GeoDataFrame(extra, geometry=gpd.points_from_xy(extra.longitude, extra.latitude), crs=4326)
        E = gpd.sjoin(E, allp, predicate="within", how="left").drop(columns="index_right").drop_duplicates(subset=key)
        E["province"] = E.province.fillna("Outside PNG").str.replace(r" Province$", "", regex=True)
        P = pd.concat([P, E], ignore_index=True)
    P = gpd.GeoDataFrame(P, geometry="geometry", crs=4326).reset_index(drop=True)
    P["latitude"] = P.latitude.astype(float)
    P["longitude"] = P.longitude.astype(float)
    P = gpd.sjoin(P, dist[["name", "geometry"]].rename(columns={"name": "district"}), predicate="within", how="left").drop(columns="index_right")
    P = P[~P.index.duplicated()].copy()
    P["month"] = P.alert__date.str[:7]
    P["nl"] = P.confidence__cat != "l"
    P["in4"] = P.province.isin(list(prov.name))
    log("fire points incl. site surroundings", len(P), "of which outside the four provinces", int((~P.in4).sum()))

    # nearest mapped water for every point
    W = WT.load()
    widx, wdist = WT.nearest(P.longitude.values, P.latitude.values, W, CFG.get("water_max_km", 30))
    P["water_idx"] = widx
    P["water_km"] = np.round(wdist / 1000, 2)
    P["water_kind"] = [W.kind.iat[i] if i >= 0 else "" for i in widx]
    P["river_order"] = [W.order.iat[i] if i >= 0 and W.kind.iat[i] == "river" else None for i in widx]
    log("nearest water done")

    # 5. provinces: daily counts since baseline
    def prov_job(p):
        return p.name, daily_counts(p.geostore, since)
    with ThreadPoolExecutor(4) as ex:
        pdaily = dict(ex.map(prov_job, list(prov.itertuples())))
    ptcl = json.load(open(os.path.join(INP, "province_tree_cover_loss.json")))["provinces"]
    P4 = P[P.in4]
    provinces = {}
    for p in prov.itertuples():
        pp = P4[P4.province == p.name]
        def part(k, sub):
            return {"years": window_counts(pdaily[p.name], start, end, first, k),
                    "monthly": {m: int((sub.month == m).sum()) for m in months}, "season_points": int(len(sub))}
        provinces[p.name] = {"project": p.project, "slug": p.slug, **part(0, pp), "nl": part(1, pp[pp.nl]), "tcl": ptcl.get(p.name, {})}

    # 6. districts
    dist = dist.copy()
    dist["area_km2"] = (dist.to_crs(6933).area / 1e6).round(1)
    drows = []
    for d in dist.itertuples():
        dp = P4[P4.district == d.name]
        def part(sub):
            r = {"monthly": {m: int((sub.month == m).sum()) for m in months}, "fires": int(len(sub)),
                 "prim": int(sub.is__umd_regional_primary_forest_2001.fillna(False).astype(bool).sum()),
                 "ifl": int(sub.is__ifl_intact_forest_landscapes_2016.fillna(False).astype(bool).sum())}
            r["density"] = round(100 * r["fires"] / d.area_km2, 2) if d.area_km2 else 0
            return r
        drows.append({"name": d.name, "province": d.province, "project": d.project, "area_km2": d.area_km2, **part(dp), "nl": part(dp[dp.nl])})

    # 7. sites
    ia_since = f"{start.year - 1}-{CFG['season_start']}"

    def site_job(s):
        gs = s.geostore
        daily = daily_counts(gs, since)
        tcl = query("umd_tree_cover_loss", "SELECT umd_tree_cover_loss__year, SUM(area__ha) AS ha FROM results "
                    "WHERE umd_tree_cover_density_2000__threshold >= 30 GROUP BY umd_tree_cover_loss__year", gs)
        tc = query("umd_tree_cover_density_2000", "SELECT SUM(area__ha) AS ha FROM results WHERE umd_tree_cover_density_2000__threshold >= 30", gs)
        ia = query("gfw_integrated_alerts", "SELECT gfw_integrated_alerts__date, gfw_integrated_alerts__confidence, COUNT(*) AS n FROM results "
                   f"WHERE gfw_integrated_alerts__date >= '{ia_since}' GROUP BY gfw_integrated_alerts__date, gfw_integrated_alerts__confidence", gs)
        return s.id, daily, tcl, tc, ia

    with ThreadPoolExecutor(5) as ex:
        res = {r[0]: r[1:] for r in ex.map(site_job, list(sites.itertuples()))}

    # firebreak history (alerts per boundary section and year); rebuilt when missing or out of date
    last_hist = start.year - 1
    H = json.load(open(FB.HIST)) if os.path.exists(FB.HIST) else {}
    if (H.get("last_year") != last_hist or set(H.get("sites", {})) != set(sites.id)
            or any(H["sites"][s.id].get("hash") != FB.geom_hash(s.geometry) for s in sites.itertuples())):
        H = FB.build_history(sites, last_hist, log)
        gh_output("history_changed", "true")
    hy = [str(y) for y in range(H["first_year"], H["last_year"] + 1)]

    P["rowid"] = np.arange(len(P))
    srows, site_fires, fb_feats, fb_rows, sp_rows = [], {}, [], [], []
    for s in sites.itertuples():
        daily, tcl, tc, ia = res[s.id]
        A = compare(window_counts(daily, start, end, first, 0))
        N = compare(window_counts(daily, start, end, first, 1))
        dd = pd.Series({d: v[0] for d, v in daily.items()}, dtype=float)
        dn = pd.Series({d: v[1] for d, v in daily.items()}, dtype=float)
        mon = lambda ser: {m: int(ser[[k[:7] == m for k in ser.index]].sum()) if len(ser) else 0 for m in months}
        tcl_d = {str(int(r["umd_tree_cover_loss__year"])): round(float(r.get("ha") or 0), 2) for r in tcl}
        tc2000 = float((tc[0].get("ha") if tc else 0) or 0)
        tot = sum(tcl_d.values())
        iad = pd.DataFrame(ia)

        def iac(a, b):
            if iad.empty:
                return 0
            m = (iad.gfw_integrated_alerts__date >= str(a)) & (iad.gfw_integrated_alerts__date <= str(b))
            return int(iad[m][[c for c in iad.columns if c in ("n", "count")][0]].sum())

        # points around the site: zone, boundary section and distance, measured in the site's UTM zone
        sp = SP[s.id].drop_duplicates(subset=key)
        if len(sp):
            seg, zone, dkm = FB.assign(s.geometry, sp.longitude.values.astype(float), sp.latitude.values.astype(float))
        else:
            seg = zone = dkm = np.array([], int)
        sp = sp.assign(section=seg, zone=zone, boundary_km=np.round(np.asarray(dkm, float) / 1000, 2))
        sp = sp[sp.zone <= 2]
        sp = sp.merge(P[key + ["rowid"]], on=key, how="left")
        site_fires[s.id] = [int(v) for pair in zip(sp.rowid.fillna(-1).astype(int), sp.zone) for v in pair]
        snl = sp.confidence__cat != "l"
        ring = lambda sub, zs: int(sub.zone.isin(zs).sum())

        # boundary sections
        crs, g, segs, seglen = FB.sections(s.geometry)
        hist = H["sites"][s.id]["counts"]
        now_cnt = sp[sp.section >= 0].section.value_counts().to_dict()
        segs4326 = gpd.GeoSeries(segs, crs=crs).to_crs(4326)
        secs = []
        for k, ln in enumerate(segs4326):
            hc = hist.get(str(k), {})
            yrs_fire = sum(1 for y in hy if hc.get(y, 0) > 0)
            tot_h = sum(hc.values())
            now_k = int(now_cnt.get(k, 0))
            cls = 0 if yrs_fire == 0 else 1 if yrs_fire <= 2 else 2 if yrs_fire <= 5 else 3 if yrs_fire <= 9 else 4
            mid = ln.interpolate(0.5, normalized=True)
            secs.append((k, yrs_fire, tot_h, now_k, cls, mid))
            fb_feats.append({"type": "Feature", "properties": {"s": s.id, "k": k, "y": yrs_fire, "t": tot_h, "n": now_k, "c": cls},
                             "geometry": {"type": "LineString", "coordinates": [[round(x, 5), round(y, 5)] for x, y in ln.coords]}})
            c0, c1 = ln.coords[0], ln.coords[-1]
            fb_rows.append({"site_id": s.id, "site": s.site, "section": k + 1, "length_m": round(segs[k].length),
                            "years_with_fire": yrs_fire, "of_years": len(hy), f"alerts_{hy[0]}_{hy[-1]}": tot_h, "alerts_this_season": now_k,
                            "recurrence": ["none", "low (1-2 yrs)", "moderate (3-5 yrs)", "high (6-9 yrs)", "very high (10+ yrs)"][cls],
                            "mid_lat": round(mid.y, 5), "mid_lon": round(mid.x, 5), "start_lat": round(c0[1], 5), "start_lon": round(c0[0], 5),
                            "end_lat": round(c1[1], 5), "end_lon": round(c1[0], 5)})
        top = sorted(secs, key=lambda t: (-t[1], -t[3], -t[2]))[:5]
        fb = {"sections": len(secs), "section_km": round(seglen / 1000, 2), "years": [hy[0], hy[-1]],
              "by_class": [sum(1 for t in secs if t[4] == c) for c in range(5)],
              "recurring": sum(1 for t in secs if t[1] >= 6), "edge_now": int(sum(now_cnt.values())),
              "top": [[t[0] + 1, t[1], t[2], t[3], round(t[5].y, 5), round(t[5].x, 5)] for t in top if t[1] > 0 or t[3] > 0]}

        for r in sp.itertuples():
            sp_rows.append({"site_id": s.id, "site": s.site, "latitude": r.latitude, "longitude": r.longitude, "alert__date": r.alert__date,
                            "alert__time_utc": r.alert__time_utc, "confidence__cat": r.confidence__cat,
                            "zone": ["inside site", f"0-{CFG['ring_km']} km outside", f"{CFG['ring_km']}-{CFG.get('map_ring_km', 10)} km outside"][r.zone],
                            "km_from_boundary": r.boundary_km, "boundary_section": int(r.section) + 1 if r.section >= 0 else None})

        peak = max(tcl_d, key=tcl_d.get) if tcl_d else None
        area = float(gpd.GeoSeries([s.geometry], crs=4326).to_crs(6933).area.iloc[0] / 1e4)
        srows.append({
            "id": s.id, "site": s.site, "project": s.project, "province": s.province, "type": s.type, "status": s.status,
            "source": s.source, "geostore": s.geostore, "area_ha": round(area, 1), "tc2000_ha": round(tc2000, 1),
            **A, "monthly": mon(dd), "ring_now": ring(sp, [1]), "ring10_now": ring(sp, [1, 2]), "inside_pts": ring(sp, [0]),
            "nl": {**N, "monthly": mon(dn), "ring_now": ring(sp[snl], [1]), "ring10_now": ring(sp[snl], [1, 2]), "inside_pts": ring(sp[snl], [0])},
            "ia_now": iac(start, end), "ia_prev": iac(shift_year(start, -1), shift_year(end, -1)),
            "tcl": tcl_d, "tcl_total": round(tot, 1), "tcl_pct": round(100 * tot / tc2000, 2) if tc2000 else None,
            "tcl_recent5": round(sum(v for k, v in tcl_d.items() if int(k) >= (max(map(int, tcl_d)) - 4 if tcl_d else 0)), 1),
            "tcl_peak_year": peak, "tcl_peak": tcl_d.get(peak, 0) if peak else 0, "firebreak": fb,
        })
    log("sites done", len(srows))

    # 8. write data
    D = os.path.join(out, "data")
    season = {"start": str(start), "end": str(end), "months": months,
              "label": f"{start.day} {start:%b}–{end.day} {end:%b %Y}" if start.year == end.year else f"{start.day} {start:%b %Y}–{end.day} {end:%b %Y}",
              "first_year": first, "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    conf = {"title": CFG["title"], "credit": CFG["credit"], "ring_km": CFG["ring_km"], "map_ring_km": CFG.get("map_ring_km", 10),
            "fb_outer_km": CFG.get("fb_outer_km", 2), "fb_inner_km": CFG.get("fb_inner_km", 1), "water_max_km": CFG.get("water_max_km", 30)}
    json.dump({"season": season, "config": conf, "provinces": provinces, "districts": drows, "sites": srows},
              open(os.path.join(D, "stats.json"), "w"), separators=(",", ":"))
    d0 = pd.Timestamp(start)
    pv = list(prov.name) + sorted(set(P.province) - set(prov.name))
    pproj = {p.name: p.project for p in prov.itertuples()}
    extra_proj = {}  # provinces outside the four take the projects of the sites they surround
    for s in sites.itertuples():
        ids = [i for i in site_fires[s.id][0::2] if i >= 0]
        for pn in set(P.province.iloc[ids]) - set(prov.name):
            extra_proj.setdefault(pn, set()).update(x.strip() for x in s.project.split("&"))
    pproj.update({k: " & ".join(sorted(v)) for k, v in extra_proj.items()})
    P["d"] = (pd.to_datetime(P.alert__date) - d0).dt.days
    LOW = 8
    flags = ((pd.to_numeric(P.umd_tree_cover_density_2000__threshold, errors="coerce").fillna(0) >= 30).astype(int)
             + 2 * P.is__umd_regional_primary_forest_2001.fillna(False).astype(bool).astype(int)
             + 4 * P.is__ifl_intact_forest_landscapes_2016.fillna(False).astype(bool).astype(int)
             + LOW * (~P.nl).astype(int))
    rows = [[round(x, 4), round(y, 4), int(d), pv.index(p), int(f), int(w)] for x, y, d, p, f, w in zip(P.longitude, P.latitude, P.d, P.province, flags, P.water_idx)]
    json.dump({"start": str(start), "provinces": pv, "projects": [pproj.get(p, "") for p in pv], "low_bit": LOW,
               "fields": ["lon", "lat", "day", "prov", "flags", "water"], "rows": rows},
              open(os.path.join(D, "fires.json"), "w"), separators=(",", ":"))
    json.dump(site_fires, open(os.path.join(D, "site_fires.json"), "w"), separators=(",", ":"))
    json.dump({"type": "FeatureCollection", "features": fb_feats}, open(os.path.join(D, "firebreak.geojson"), "w"), separators=(",", ":"))
    with open(WT.SRC) as a_, open(os.path.join(D, "water.json"), "w") as b_:
        b_.write(a_.read())

    def save_geo(g, name, tol, nd=4):
        g = g.copy()
        g["geometry"] = g.geometry.simplify(tol, preserve_topology=True)
        gj = json.loads(g.to_json(drop_id=True))
        def r(c):
            return [round(c[0], nd), round(c[1], nd)] if isinstance(c[0], (int, float)) else [r(x) for x in c]
        for f in gj["features"]:
            f["geometry"]["coordinates"] = r(f["geometry"]["coordinates"])
        json.dump(gj, open(os.path.join(D, name), "w"), separators=(",", ":"))
    save_geo(prov[["name", "slug", "project", "geometry"]], "provinces.geojson", 0.003)
    save_geo(dist[["name", "province", "project", "geometry"]], "districts.geojson", 0.003)
    save_geo(sites[["id", "site", "geometry"]], "sites.geojson", 0.0005, 5)
    rings = [{"id": s.id, "km": km, "geometry": FB.buffer_geom(s.geometry, km).boundary}
             for s in sites.itertuples() for km in (CFG["ring_km"], CFG.get("map_ring_km", 10))]
    save_geo(gpd.GeoDataFrame(rings, crs=4326), "site_rings.geojson", 0.0008)
    # optional project overlays: published only if the files are in inputs/ and config allows it
    if CFG.get("publish_overlays", False):
        for f in ("logging_existing", "logging_proposed", "mining_el"):
            src = os.path.join(INP, f + ".geojson")
            if os.path.exists(src):
                with open(src) as a_, open(os.path.join(D, f + ".geojson"), "w") as b_:
                    b_.write(a_.read())

    # downloads (tables)
    W_ = os.path.join(out, "downloads")
    flat = lambda r: {**{k: v for k, v in r.items() if k not in ("years", "monthly", "tcl", "nl", "firebreak")},
                      **{k + "_excl_low": v for k, v in r["nl"].items() if k not in ("years", "monthly")},
                      "firebreak_sections": r["firebreak"]["sections"], "sections_fire_6plus_years": r["firebreak"]["recurring"]}
    sdf = pd.DataFrame([flat(r) for r in srows])
    sdf.to_csv(os.path.join(W_, "site_summary.csv"), index=False)
    pd.DataFrame({r["site"]: r["years"] for r in srows}).T.to_csv(os.path.join(W_, "site_fire_alerts_by_year.csv"), index_label="site")
    pd.DataFrame({r["site"]: r["nl"]["years"] for r in srows}).T.to_csv(os.path.join(W_, "site_fire_alerts_by_year_excl_low.csv"), index_label="site")
    pd.DataFrame({r["site"]: r["monthly"] for r in srows}).T.to_csv(os.path.join(W_, "site_fire_alerts_by_month.csv"), index_label="site")
    pd.DataFrame({r["site"]: r["tcl"] for r in srows}).T.sort_index(axis=1).to_csv(os.path.join(W_, "site_tree_cover_loss_ha.csv"), index_label="site")
    pd.DataFrame([{**{k: r[k] for k in ("name", "province", "project", "area_km2", "fires", "density", "prim", "ifl")}, "fires_excl_low": r["nl"]["fires"], **r["monthly"]} for r in drows]).to_csv(os.path.join(W_, "district_fire_alerts.csv"), index=False)
    pd.DataFrame({n: v["years"] for n, v in provinces.items()}).T.to_csv(os.path.join(W_, "province_fire_alerts_by_year.csv"), index_label="province")
    pd.DataFrame({n: v["nl"]["years"] for n, v in provinces.items()}).T.to_csv(os.path.join(W_, "province_fire_alerts_by_year_excl_low.csv"), index_label="province")
    P.drop(columns=["geometry", "d", "rowid", "nl", "in4", "water_idx"]).rename(columns={"water_km": "nearest_water_km", "water_kind": "nearest_water_type"}) \
        .to_csv(os.path.join(W_, "viirs_fire_alerts_season.csv"), index=False)
    spd = pd.DataFrame(sp_rows)
    if len(spd):
        wi = spd.merge(P[key + ["water_km", "water_kind", "river_order"]], on=key, how="left")
        spd = wi.rename(columns={"water_km": "nearest_water_km", "water_kind": "nearest_water_type"})
    spd.to_csv(os.path.join(W_, "site_fire_alerts_10km.csv"), index=False)
    pd.DataFrame(fb_rows).to_csv(os.path.join(W_, "firebreak_sections.csv"), index=False)
    g = sites.drop(columns=["geostore10"]).merge(sdf.drop(columns=["site", "project", "province", "type", "status", "source", "geostore"]), on="id")
    g.to_file(os.path.join(W_, "gef_sites_fire.geojson"), driver="GeoJSON")
    log("data written to", out)


if __name__ == "__main__":
    main()
