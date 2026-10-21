"""Daily data update for GEF Fire Watch PNG.

Queries the GFW Data API for the four project provinces and the 40 project sites, then writes
the website's data files into the build folder (default _site/). Maps and PDF reports are made
afterwards by outputs.py.

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


def window_counts(daily, start, end, first_year):
    """Alerts in the same calendar window (start..end) for every year since first_year."""
    s = pd.Series(daily, dtype=float)
    s.index = pd.to_datetime(s.index)
    out = {}
    span_years = end.year - start.year
    for y in range(first_year, start.year + 1):
        k = y - start.year
        a, b = shift_year(start, k), shift_year(end, k)
        out[str(y + span_years) if span_years else str(y)] = int(s[(s.index >= str(a)) & (s.index <= str(b))].sum())
    return out


def daily_counts(geostore, since):
    rows = query(PV, f"SELECT alert__date, COUNT(*) AS n FROM results WHERE alert__date >= '{since}' GROUP BY alert__date", geostore)
    return {r["alert__date"]: int(r.get("n", r.get("count", 0))) for r in rows}


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
    jobs = []
    for p in prov.itertuples():
        for m in months:
            m0 = max(dt.date.fromisoformat(m + "-01"), start)
            nxt = dt.date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
            jobs.append((p.name, p.geostore, str(m0), str(min(nxt - dt.timedelta(days=1), end))))

    def get_pts(j):
        name, g, d0, d1 = j
        rows = query(PV, f"SELECT {cols} FROM results WHERE alert__date >= '{d0}' AND alert__date <= '{d1}'", g)
        return pd.DataFrame(rows)

    with ThreadPoolExecutor(4) as ex:
        frames = list(ex.map(get_pts, jobs))
    pts = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame(columns=cols.replace(" ", "").split(","))
    pts = pts.drop_duplicates(subset=["latitude", "longitude", "alert__date", "alert__time_utc"])
    P = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.longitude, pts.latitude), crs=4326)
    P = gpd.sjoin(P, prov[["name", "geometry"]].rename(columns={"name": "province"}), predicate="within", how="inner").drop(columns="index_right")
    P = gpd.sjoin(P, dist[["name", "geometry"]].rename(columns={"name": "district"}), predicate="within", how="left").drop(columns="index_right")
    P = P.drop_duplicates(subset=["latitude", "longitude", "alert__date", "alert__time_utc"])
    P["month"] = P.alert__date.str[:7]
    log("fire points", len(P))

    # 3. provinces: daily counts since baseline
    def prov_job(p):
        d = daily_counts(p.geostore, since)
        return p.name, d
    with ThreadPoolExecutor(4) as ex:
        pdaily = dict(ex.map(prov_job, list(prov.itertuples())))
    ptcl = json.load(open(os.path.join(INP, "province_tree_cover_loss.json")))["provinces"]
    provinces = {}
    for p in prov.itertuples():
        yrs = window_counts(pdaily[p.name], start, end, first)
        pp = P[P.province == p.name]
        provinces[p.name] = {"project": p.project, "slug": p.slug, "years": yrs,
                             "monthly": {m: int((pp.month == m).sum()) for m in months},
                             "season_points": int(len(pp)), "tcl": ptcl.get(p.name, {})}

    # 4. districts
    dist = dist.copy()
    dist["area_km2"] = (dist.to_crs(6933).area / 1e6).round(1)
    drows = []
    for d in dist.itertuples():
        dp = P[P.district == d.name]
        r = {"name": d.name, "province": d.province, "project": d.project, "area_km2": d.area_km2,
             "monthly": {m: int((dp.month == m).sum()) for m in months}, "fires": int(len(dp)),
             "prim": int(dp.is__umd_regional_primary_forest_2001.fillna(False).astype(bool).sum()),
             "ifl": int(dp.is__ifl_intact_forest_landscapes_2016.fillna(False).astype(bool).sum())}
        r["density"] = round(100 * r["fires"] / d.area_km2, 2) if d.area_km2 else 0
        drows.append(r)

    # 5. sites
    sites = sites.copy()
    changed_gs = False
    for i, s in sites.iterrows():
        if not s.get("geostore"):
            sites.at[i, "geostore"] = create_geostore(mapping(s.geometry.simplify(0.0005)))
            changed_gs = True
    if changed_gs:
        sites.to_file(os.path.join(INP, "sites.geojson"), driver="GeoJSON")
    S_m = sites.to_crs(32755) if True else None
    Pm = P.to_crs(32755)
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
    srows = []
    ring = CFG["ring_km"] * 1000
    for s, sm in zip(sites.itertuples(), S_m.itertuples()):
        daily, tcl, tc, ia = res[s.id]
        yrs = window_counts(daily, start, end, first)
        cur_key = max(yrs)
        hist = [v for k, v in yrs.items() if k != cur_key]
        mean = float(np.mean(hist)) if hist else 0.0
        cur = yrs[cur_key]
        dd = pd.Series(daily, dtype=float)
        dd.index = pd.to_datetime(dd.index) if len(dd) else dd.index
        monthly = {m: int(dd[dd.index.strftime("%Y-%m") == m].sum()) if len(dd) else 0 for m in months}
        tcl_d = {str(int(r["umd_tree_cover_loss__year"])): round(float(r.get("ha") or 0), 2) for r in tcl}
        tc2000 = float((tc[0].get("ha") if tc else 0) or 0)
        tot = sum(tcl_d.values())
        iad = pd.DataFrame(ia)
        def iac(a, b):
            if iad.empty:
                return 0
            m = (iad.gfw_integrated_alerts__date >= str(a)) & (iad.gfw_integrated_alerts__date <= str(b))
            return int(iad[m][[c for c in iad.columns if c in ("n", "count")][0]].sum())
        g = sm.geometry
        inner = Pm[Pm.within(g.buffer(ring))]
        ring_n = int((~inner.within(g)).sum())
        peak = max(tcl_d, key=tcl_d.get) if tcl_d else None
        area = float(gpd.GeoSeries([s.geometry], crs=4326).to_crs(6933).area.iloc[0] / 1e4)
        srows.append({
            "id": s.id, "site": s.site, "project": s.project, "province": s.province, "type": s.type, "status": s.status,
            "source": s.source, "geostore": s.geostore, "area_ha": round(area, 1), "tc2000_ha": round(tc2000, 1),
            "fire_now": cur, "fire_mean": round(mean, 1), "fire_2015": yrs.get("2015", 0), "fire_max": max(hist) if hist else 0,
            "ratio": round(cur / mean, 1) if mean > 0 else None, "fire_rank": 1 + sum(h > cur for h in hist),
            "ring_now": ring_n, "years": yrs, "monthly": monthly,
            "ia_now": iac(start, end), "ia_prev": iac(shift_year(start, -1), shift_year(end, -1)),
            "tcl": tcl_d, "tcl_total": round(tot, 1), "tcl_pct": round(100 * tot / tc2000, 2) if tc2000 else None,
            "tcl_recent5": round(sum(v for k, v in tcl_d.items() if int(k) >= (max(map(int, tcl_d)) - 4 if tcl_d else 0)), 1),
            "tcl_peak_year": peak, "tcl_peak": tcl_d.get(peak, 0) if peak else 0,
        })
    log("sites done", len(srows))

    # 6. write data
    D = os.path.join(out, "data")
    season = {"start": str(start), "end": str(end), "months": months,
              "label": f"{start.day} {start:%b}–{end.day} {end:%b %Y}" if start.year == end.year else f"{start.day} {start:%b %Y}–{end.day} {end:%b %Y}",
              "first_year": first, "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    json.dump({"season": season, "config": {"title": CFG["title"], "credit": CFG["credit"], "ring_km": CFG["ring_km"]},
               "provinces": provinces, "districts": drows, "sites": srows},
              open(os.path.join(D, "stats.json"), "w"), separators=(",", ":"))
    d0 = pd.Timestamp(start)
    pv = list(prov.name)
    P["d"] = (pd.to_datetime(P.alert__date) - d0).dt.days
    flags = ((pd.to_numeric(P.umd_tree_cover_density_2000__threshold, errors="coerce").fillna(0) >= 30).astype(int)
             + 2 * P.is__umd_regional_primary_forest_2001.fillna(False).astype(bool).astype(int)
             + 4 * P.is__ifl_intact_forest_landscapes_2016.fillna(False).astype(bool).astype(int))
    rows = [[round(x, 4), round(y, 4), int(d), pv.index(p), int(f)] for x, y, d, p, f in zip(P.longitude, P.latitude, P.d, P.province, flags)]
    json.dump({"start": str(start), "provinces": pv, "fields": ["lon", "lat", "day", "prov", "flags"], "rows": rows},
              open(os.path.join(D, "fires.json"), "w"), separators=(",", ":"))

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
    for f in ("logging_existing", "logging_proposed", "mining_el"):
        with open(os.path.join(INP, f + ".geojson")) as src, open(os.path.join(D, f + ".geojson"), "w") as dst:
            dst.write(src.read())

    # downloads (tables)
    W = os.path.join(out, "downloads")
    sdf = pd.DataFrame([{k: v for k, v in r.items() if k not in ("years", "monthly", "tcl")} for r in srows])
    sdf.to_csv(os.path.join(W, "site_summary.csv"), index=False)
    pd.DataFrame({r["site"]: r["years"] for r in srows}).T.to_csv(os.path.join(W, "site_fire_alerts_by_year.csv"), index_label="site")
    pd.DataFrame({r["site"]: r["monthly"] for r in srows}).T.to_csv(os.path.join(W, "site_fire_alerts_by_month.csv"), index_label="site")
    pd.DataFrame({r["site"]: r["tcl"] for r in srows}).T.sort_index(axis=1).to_csv(os.path.join(W, "site_tree_cover_loss_ha.csv"), index_label="site")
    pd.DataFrame([{**{k: r[k] for k in ("name", "province", "project", "area_km2", "fires", "density", "prim", "ifl")}, **r["monthly"]} for r in drows]).to_csv(os.path.join(W, "district_fire_alerts.csv"), index=False)
    pd.DataFrame({n: v["years"] for n, v in provinces.items()}).T.to_csv(os.path.join(W, "province_fire_alerts_by_year.csv"), index_label="province")
    P.drop(columns=["geometry", "d"]).to_csv(os.path.join(W, "viirs_fire_alerts_season.csv"), index=False)
    g = sites.merge(sdf.drop(columns=["site", "project", "province", "type", "status", "source", "geostore"]), on="id")
    g.to_file(os.path.join(W, "gef_sites_fire.geojson"), driver="GeoJSON")
    log("data written to", out)


if __name__ == "__main__":
    main()
