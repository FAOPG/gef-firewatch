"""Maps and PDF reports for GEF Fire Watch PNG, made from the data written by update.py.

    python pipeline/outputs.py --out _site
"""
import argparse, json, os, sys, textwrap, warnings
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(ROOT, "inputs")
GREY, ACC, INK = "#9a9a92", "#c0392b", "#1d2327"
MN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MNL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
ABBR = {"EHP": "Eastern Highlands", "WHP": "Western Highlands", "SHP": "Southern Highlands"}
OFF = {"Velotige CCA": (0.05, 0.2), "Mount Otto KBA": (0.45, 0.12), "Proposed Artrohialika Lulume Park": (-0.45, 0.12),
       "Natauka Zamu Nature Park": (0.55, 0.02), "Mt Gahavisuka Provincial Park": (0.6, -0.12), "Proposed Mt Dorohere CA": (-0.5, -0.1),
       "Proposed Yasina Nature Park": (0.45, -0.25), "Jimi (Ruti) Valley NP": (0.45, 0.08), "Sali River Jimi Valley KBA": (-0.25, 0.1),
       "Kuk Early Agricultural Site": (0.35, 0.0), "Tonda WMA": (1.1, -0.05), "Proposed Tonda Extension WMA": (-0.1, 0.55),
       "Tonda KBA": (1.0, 0.45), "Proposed Weriaver WMA": (-0.05, 1.0), "Proposed Suki-Aramba WMA": (0.9, 0.35),
       "Maza WMA": (0.6, -0.1), "Dabarambua KBA": (0.3, -0.15), "Proposed Lakekamu CA": (-0.3, 0.3)}
# Site outlines by category (same scheme as the web map): protected and conserved areas green, KBAs blue,
# proposed areas dashed. No red, so outlines are not confused with fire alerts.
PA_COL, KBA_COL = "#0f9d74", "#2f5fd0"          # on light maps
PA_IMG, KBA_IMG = "#2ee6a6", "#ffffff"          # on satellite imagery
SCOL = {"CCA": PA_COL, "Nature park": PA_COL, "Proposed CA": PA_COL, "Proposed park": PA_COL, "Proposed PA": PA_COL,
        "KBA": KBA_COL, "WMA": PA_COL, "Provincial park": PA_COL, "National park": PA_COL, "Sanctuary": PA_COL,
        "World Heritage site": PA_COL}
# Fire alerts by month: newest month bright yellow, earlier months fading through pink and violet to blue.
# No orange-red, so the map does not read as a "red zone".
RECENCY = ["#ffd400", "#ff5fb7", "#b14cff", "#5a6cf0", "#3a4bb0", "#2b3480", "#232a63", "#1d2250", "#181c42", "#141737", "#10132d", "#0d1025"]
# Firebreak planning: years with fire alerts near each boundary section
FB_COL = ["#b9c2be", "#fff3a0", "#ffc04d", "#f08a24", "#c2410c"]
FB_LAB = ["No alerts nearby", "Alerts in 1–2 years", "Alerts in 3–5 years", "Alerts in 6–9 years", "Alerts in 10+ years"]
WATER = "#4fc3f7"


def mlabel(m, long=False, season=None):
    y, mo = int(m[:4]), int(m[5:])
    lab = (MNL if long else MN)[mo - 1]
    if season and m == season["end"][:7] and int(season["end"][8:]) < 28:
        lab = f"1–{int(season['end'][8:])} {lab}"
    return lab


def month_colors(months):
    n = len(months)
    return {m: RECENCY[min(n - 1 - i, len(RECENCY) - 1)] for i, m in enumerate(months)}


def is_kba(t):
    return t == "KBA"


def is_proposed(r):
    return str(r.get("status", "")).startswith("Proposed") or str(r.get("type", "")).startswith("Proposed")


def fmt(v, d=0):
    return "–" if v is None else f"{v:,.{d}f}"


def ratio_txt(s):
    return ("new" if s["fire_now"] else "–") if s["ratio"] is None else f"{s['ratio']:.1f}×"


def bars(ax, keys, vals, hl, title, ylab, ref=None, reflab=None, lab=lambda k: k, valfmt=lambda v: f"{v:,.0f}"):
    x = np.arange(len(keys))
    ax.bar(x, vals, color=[ACC if h else GREY for h in hl], width=0.7)
    for i, (v, h) in enumerate(zip(vals, hl)):
        if h and v > 0:
            ax.text(i, v, valfmt(v), ha="center", va="bottom", fontsize=7.5, color=INK)
    if ref:
        ax.axhline(ref, color="#555", ls="--", lw=0.8)
        ax.text(-0.4, ref, f"{reflab} ", va="bottom", ha="left", fontsize=7, color="#555")
    step = max(1, len(keys) // 13)
    ax.set_xticks(x[::step], [lab(k) for k in keys][::step], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=7.5)


def header(fig, title, sub):
    fig.patches.append(plt.Rectangle((0, 0.93), 1, 0.07, transform=fig.transFigure, color="#1f3b2d", zorder=0))
    fig.text(0.05, 0.965, title, fontsize=16, color="white", fontweight="bold", va="center")
    fig.text(0.05, 0.94, sub, fontsize=8.5, color="#d8e6dc", va="center")


def kpis(fig, items, y=0.83):
    w = 0.9 / len(items)
    for i, (lab, val, note) in enumerate(items):
        x = 0.05 + i * w
        fig.text(x, y + 0.045, lab, fontsize=7, color="#555")
        fig.text(x, y + 0.012, val, fontsize=15, color=INK, fontweight="bold")
        fig.text(x, y - 0.008, note, fontsize=6.5, color="#666")


def table(ax, rows, cols, widths=None, fs=7.2):
    ax.axis("off")
    if not rows:
        return
    t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="left", colWidths=widths)
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    t.scale(1, 1.18)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#eef2ef")
            cell.set_text_props(fontweight="bold")


class Ctx:
    def __init__(self, out):
        self.out = out
        self.st = json.load(open(os.path.join(out, "data", "stats.json")))
        self.season = self.st["season"]
        self.months = self.season["months"]
        self.mcol = month_colors(self.months)
        self.credit = self.st["config"]["credit"]
        self.src = (f"Fire: NASA VIIRS via Global Forest Watch, to {self.season['end']}. Boundaries: CEPA/FAO GEF7 & GEF8 project sites; provinces and districts OCHA COD-AB / PNG NSO (CC BY-IGO). "
                    f"Generated automatically by {self.credit}, {self.season['updated']}.")
        pts = pd.read_csv(os.path.join(out, "downloads", "viirs_fire_alerts_season.csv"))
        self.P = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.longitude, pts.latitude), crs=4326)
        self.prov = gpd.read_file(os.path.join(INP, "provinces.geojson"))
        self.dist = gpd.read_file(os.path.join(INP, "districts.geojson"))
        self.allp = gpd.read_file(os.path.join(INP, "png_provinces_all.geojson"))
        self.sites = gpd.read_file(os.path.join(INP, "sites.geojson"))
        self.S = {s["site"]: s for s in self.st["sites"]}
        self.D = {d["name"]: d for d in self.st["districts"]}
        for d in ("maps/overview", "maps/sites", "reports/sites", "reports/provinces"):
            os.makedirs(os.path.join(out, d), exist_ok=True)


# ---------------- overview maps ----------------
def overview_all(c):
    fig, ax = plt.subplots(figsize=(14, 10))
    c.allp.plot(ax=ax, color="#f2f2ee", edgecolor="#bbbbbb", lw=0.5)
    c.prov.plot(ax=ax, color=["#e3ecf7" if p == "GEF7" else "#e6f0e1" for p in c.prov["project"]], edgecolor="none")
    P4 = c.P[c.P.province.isin(list(c.prov.name))]
    for m in c.months:
        s = P4[P4.month == m]
        ax.scatter(s.longitude, s.latitude, s=2.2, c=c.mcol[m], edgecolors="#333", linewidths=0.15, alpha=0.9, zorder=3)
    c.dist.boundary.plot(ax=ax, color="#777777", lw=0.4, zorder=4)
    c.prov.boundary.plot(ax=ax, color="#111111", lw=1.4, zorder=5)
    for r in c.prov.itertuples():
        pt = r.geometry.representative_point()
        n = c.st["provinces"][r.name]["season_points"]
        ax.text(pt.x, pt.y, f"{r.name}\n({r.project})\n{n:,} alerts", fontsize=10, fontweight="bold", ha="center", zorder=6,
                bbox=dict(fc="white", alpha=0.75, lw=0, pad=2))
    b = c.prov.total_bounds
    ax.set_xlim(b[0] - 0.3, b[2] + 0.3); ax.set_ylim(b[1] - 0.3, b[3] + 0.3); ax.set_aspect("equal"); ax.tick_params(labelsize=8)
    h = [Line2D([], [], marker="o", ls="", mfc=c.mcol[m], mec="#333", mew=0.4, ms=7, label=mlabel(m, True, c.season)) for m in c.months]
    h += [Patch(fc="#e3ecf7", label="GEF7 province"), Patch(fc="#e6f0e1", label="GEF8 province")]
    ax.legend(handles=h, title="VIIRS fire alerts", loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(f"GEF7 and GEF8 provinces: fire alerts, {c.season['label']}", loc="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, c.src, fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(c.out, "maps/overview/overview_all.png"), dpi=130); plt.close(fig)


def monthly_maps(c):
    for p in c.prov.itertuples():
        p1 = c.prov[c.prov.name == p.name]
        p2 = c.dist[c.dist.province == p.name].copy()
        p2["area"] = p2.name.map(lambda n: c.D[n]["area_km2"])
        dens = np.concatenate([[100 * c.D[n]["monthly"][m] / c.D[n]["area_km2"] for n in p2.name] for m in c.months])
        br = np.unique(np.round(np.quantile(dens[dens > 0], [0, .2, .4, .6, .8, 1]), 1)) if (dens > 0).any() else np.array([0, 1])
        br = np.concatenate([[0], br[1:]]) if len(br) > 1 else np.array([0, 1])
        cmap = ListedColormap(["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"][:len(br) - 1]); norm = BoundaryNorm(br, cmap.N)
        b = p1.total_bounds; asp = (b[2] - b[0]) / (b[3] - b[1])
        n = len(c.months); nc = 2 if n > 1 else 1; nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(13, min(9 * nr, 13 / nc / asp * nr * 1.02 + 1.6)), squeeze=False)
        ms = 0.8 if p.name == "Western" else 4
        for ax in axs.flat[n:]:
            ax.axis("off")
        for ax, m in zip(axs.flat, c.months):
            c.allp.plot(ax=ax, color="#f2f2ee", edgecolor="#cccccc", lw=0.4)
            p2.assign(dens=[100 * c.D[x]["monthly"][m] / c.D[x]["area_km2"] for x in p2.name]).plot(ax=ax, column="dens", cmap=cmap, norm=norm, edgecolor="#555", lw=0.6)
            s = c.P[(c.P.province == p.name) & (c.P.month == m)]
            ax.scatter(s.longitude, s.latitude, s=ms, c="#1d1d1d", lw=0, alpha=0.6, zorder=3)
            p1.boundary.plot(ax=ax, color="#111", lw=1.3, zorder=4)
            for r in p2.itertuples():
                pt = r.geometry.representative_point()
                ax.text(pt.x, pt.y, f"{r.name}\n{c.D[r.name]['monthly'][m]:,}", fontsize=7.5, ha="center", va="center", zorder=6,
                        bbox=dict(fc="white", alpha=0.7, lw=0, pad=1))
            px, py = (b[2] - b[0]) * 0.04, (b[3] - b[1]) * 0.04
            ax.set_xlim(b[0] - px, b[2] + px); ax.set_ylim(b[1] - py, b[3] + py); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{mlabel(m, True, c.season)} {m[:4]}: {len(s):,} alerts", loc="left", fontsize=12, fontweight="bold")
        hand = [Patch(fc=cmap(i), ec="#555", label=f"{br[i]:g}–{br[i + 1]:g}") for i in range(cmap.N)]
        hand += [Line2D([], [], marker="o", ls="", mfc="#1d1d1d", mec="none", ms=4, label="VIIRS fire alert")]
        fig.legend(handles=hand, title="Alerts per 100 km² (district)", loc="lower center", ncol=len(hand), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.018))
        fig.suptitle(f"{p.name} Province ({p.project}): fire alerts by district and month, {c.season['label']}", x=0.01, ha="left", fontsize=15, fontweight="bold")
        fig.text(0.01, 0.005, "Labels: district and alert count for the month. " + c.src, fontsize=7, color="#555")
        plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.085, wspace=0.03, hspace=0.1)
        fig.savefig(os.path.join(c.out, f"maps/overview/monthly_{p.slug}.png"), dpi=110); plt.close(fig)


def sites_maps(c):
    for proj, provs, ext, legloc in [("GEF7", ["Western Highlands", "Eastern Highlands"], (143.6, 146.4, -7.0, -5.1), "lower left"),
                                      ("GEF8", ["Western", "Gulf"], (140.8, 146.3, -9.6, -5.0), "upper right")]:
        fig, ax = plt.subplots(figsize=(13, 9.5))
        c.allp.plot(ax=ax, color="#f3f1ea", edgecolor="#999", lw=0.6)
        c.dist[c.dist.province.isin(provs)].boundary.plot(ax=ax, color="#c9c4b5", lw=0.4)
        c.prov[c.prov.name.isin(provs)].boundary.plot(ax=ax, color="#555", lw=1.2)
        p = c.P[c.P.province.isin(provs)]
        ax.scatter(p.longitude, p.latitude, s=1.2, c="#5a5a5a", alpha=0.45, lw=0, zorder=2)
        ss = c.sites[c.sites["project"].str.contains(proj)]
        for t, col in SCOL.items():
            sub = ss[ss["type"] == t]
            if len(sub):
                sub.plot(ax=ax, facecolor=col + "30", edgecolor=col, lw=1.3, zorder=3, linestyle="--" if "Proposed" in t else "-")
        for r in ss.itertuples():
            if proj == "GEF8" and r.site == "Crater Mountain WMA":
                pass
            pt = r.geometry.representative_point(); dx, dy = OFF.get(r.site, (0, 0))
            ax.annotate(f"{r.site}\n{c.S[r.site]['fire_now']:,} alerts", (pt.x, pt.y), xytext=(pt.x + dx, pt.y + dy), fontsize=6.5 if proj == "GEF7" else 7,
                        ha="center", va="center", zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                        arrowprops=dict(arrowstyle="-", color="#444", lw=0.6) if (dx or dy) else None)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        h = [Line2D([], [], marker="o", ls="", mfc="#5a5a5a", mec="none", ms=5, label=f"VIIRS fire alert, {c.season['label']}"),
             Line2D([], [], color=PA_COL, lw=2, label="Protected or conserved area (CCA, WMA, park)"),
             Line2D([], [], color=KBA_COL, lw=2, label="Key Biodiversity Area"), Line2D([], [], color=PA_COL, lw=2, ls="--", label="Proposed protected area"),
             Line2D([], [], color="#555", lw=1.2, label="Province"), Line2D([], [], color="#c9c4b5", lw=1, label="District")]
        ax.legend(handles=h, loc=legloc, fontsize=8.5, framealpha=0.9)
        ttl = "Western and Eastern Highlands" if proj == "GEF7" else "Western and Gulf provinces"
        ax.set_title(f"{proj} sites and fire alerts — {ttl}", loc="left", fontsize=14, fontweight="bold")
        fig.text(0.01, 0.01, c.src, fontsize=7, color="#555")
        fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(c.out, f"maps/overview/sites_{proj.lower()}.png"), dpi=130); plt.close(fig)


# ---------------- site maps and reports ----------------
def scalebar(ax):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); span = x1 - x0
    lat = np.degrees(2 * np.arctan(np.exp(((y0 + y1) / 2) / 6378137)) - np.pi / 2); k = 1 / np.cos(np.radians(lat))
    target = span / k / 5; mag = 10 ** np.floor(np.log10(target)); r = target / mag
    L = (1 if r < 2 else 2 if r < 5 else 5) * mag
    bx, by = x0 + span * 0.04, y0 + (y1 - y0) * 0.05
    ax.plot([bx, bx + L * k], [by, by], color="white", lw=5, solid_capstyle="butt", zorder=20)
    ax.plot([bx, bx + L * k], [by, by], color="black", lw=2, solid_capstyle="butt", zorder=21)
    ax.text(bx + L * k / 2, by + (y1 - y0) * 0.02, f"{L / 1000:g} km", ha="center", color="white", fontsize=8, fontweight="bold", zorder=21)
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.86), xycoords="axes fraction", ha="center", color="white", fontweight="bold",
                arrowprops=dict(facecolor="white", edgecolor="black", width=3, headwidth=9), zorder=21)


def load_water(c):
    """Rivers and lakes (HydroSHEDS) in Web Mercator, from the published compact file."""
    sys.path.insert(0, os.path.dirname(__file__))
    import water as WT
    W = WT.load()
    W["lw"] = W.order.fillna(3).map(lambda o: 0.35 + 0.28 * o)
    return W.to_crs(3857)


def frame(ax, geom, pad_m, aspect=1.25):
    minx, miny, maxx, maxy = geom.buffer(pad_m).bounds
    cxx, cyy = (minx + maxx) / 2, (miny + maxy) / 2; s = max(maxx - minx, (maxy - miny) * aspect) * 1.04
    ax.set_xlim(cxx - s / 2, cxx + s / 2); ax.set_ylim(cyy - s / 2 / aspect, cyy + s / 2 / aspect)


def basemap(ax, name):
    import contextily as cx
    try:
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, attribution=False, zoom="auto")
    except Exception as e:
        print("basemap failed", name, e)
        ax.set_facecolor("#dfe3dc")
    ax.text(0.995, 0.005, "Imagery © Esri, Maxar, Earthstar Geographics · Rivers: HydroSHEDS", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="white", bbox=dict(fc="black", alpha=0.45, lw=0, pad=1.5), zorder=25)


def draw_water(c, ax):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w = c.W.cx[x0:x1, y0:y1]
    if len(w):
        lk = w[w.kind == "lake"]
        rv = w[w.kind == "river"]
        if len(lk):
            lk.plot(ax=ax, facecolor=WATER + "80", edgecolor=WATER, lw=0.6, zorder=4)
        if len(rv):
            rv.plot(ax=ax, color=WATER, linewidth=rv.lw.values, alpha=0.9, zorder=4)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def site_outline(ax, geom, r, lw=2.2):
    col = KBA_IMG if is_kba(r["type"]) else PA_IMG
    gpd.GeoSeries([geom], crs=3857).boundary.plot(ax=ax, color="black", lw=lw + 1.6, alpha=0.55, zorder=6)
    gpd.GeoSeries([geom], crs=3857).boundary.plot(ax=ax, color=col, lw=lw, ls="--" if is_proposed(r) else "-", zorder=6)
    return col


def site_map(c, ax, r, G, mode="ring"):
    """mode: 'ring'   = alerts inside the site and up to map_ring_km outside, with both rings
             'inside' = alerts inside the site only
             'firebreak' = boundary sections coloured by years with fire nearby"""
    cfg = c.st["config"]; k = 1 / np.cos(np.radians(-6.5))
    geom = G[G["id"] == r["id"]].iloc[0].geometry
    sp = c.SP[c.SP.site_id == r["id"]]
    if mode == "inside":
        frame(ax, geom, 1500 * k); sp = sp[sp.zone_i == 0]
    elif mode == "firebreak":
        frame(ax, geom, (cfg["fb_outer_km"] + 1.5) * 1000 * k)
    else:
        frame(ax, geom, cfg["map_ring_km"] * 1000 * k * 1.02)
    basemap(ax, r["site"]); draw_water(c, ax)
    h = []
    if mode == "ring":
        for km, ls in ((cfg["ring_km"], "--"), (cfg["map_ring_km"], ":")):
            gpd.GeoSeries([geom.buffer(km * 1000 * k)], crs=3857).boundary.plot(ax=ax, color="#eeeeee", lw=1, ls=ls, zorder=5)
            h.append(Line2D([], [], color="#bbbbbb", lw=1, ls=ls, label=f"{km:g} km from boundary"))
    col = site_outline(ax, geom, r, 2.2 if mode != "firebreak" else 1)
    h.insert(0, Line2D([], [], color=col, lw=2, ls="--" if is_proposed(r) else "-", label=("Key Biodiversity Area" if is_kba(r["type"]) else "Site boundary") + (" (proposed)" if is_proposed(r) else "")))
    if mode == "firebreak":
        fb = c.FB[c.FB.s == r["id"]]
        for cl in range(5):
            sub = fb[fb.c == cl]
            if len(sub):
                sub.plot(ax=ax, color="black", lw=6.5 if cl else 3, alpha=0.6, zorder=7)
                sub.plot(ax=ax, color=FB_COL[cl], lw=4.5 if cl else 1.6, zorder=8)
            h.append(Line2D([], [], color=FB_COL[cl], lw=4, label=f"{FB_LAB[cl]}, {c.season['first_year']}–{r['firebreak']['years'][1]}"))
        top = r["firebreak"]["top"][:5]
        for t in top:
            seg = fb[fb.k == t[0] - 1]
            if len(seg):
                pt = seg.geometry.iloc[0].interpolate(0.5, normalized=True)
                ax.annotate(str(t[0]), (pt.x, pt.y), fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=12,
                            bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", lw=0.6))
        edge = sp[sp.boundary_section.notna()]
        if len(edge):
            ax.scatter(edge.geometry.x, edge.geometry.y, s=7, c="white", edgecolors="black", linewidths=0.3, zorder=9)
            h.append(Line2D([], [], marker="o", ls="", mfc="white", mec="k", mew=0.4, ms=4, label=f"Fire alert near boundary, {c.season['label']}"))
    else:
        n = len(sp)
        for m in c.months:
            sub = sp[sp.month == m]
            if len(sub):
                ax.scatter(sub.geometry.x, sub.geometry.y, s=(30 if mode == "inside" else 16) if n < 1500 else 7 if n < 6000 else 3, c=c.mcol[m], edgecolors="black", linewidths=0.35, zorder=9)
        h += [Line2D([], [], marker="o", ls="", mfc=c.mcol[m], mec="k", mew=0.4, ms=6, label=f"Fire alert, {mlabel(m, True, c.season)}") for m in c.months]
    h.append(Line2D([], [], color=WATER, lw=1.5, label="River or lake (HydroSHEDS, approx. ±500 m)"))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel(""); scalebar(ax)
    lg = ax.legend(handles=h, loc="upper left", fontsize=6.8, framealpha=0.88); lg.set_zorder(30)


def site_text(c, r):
    f, m = r["fire_now"], r["fire_mean"]; lab = c.season["label"]; out = []; cfg = c.st["config"]
    if f == 0:
        out.append(f"No VIIRS fire alerts fell inside {r['site']} in {lab}.")
    else:
        rel = ("for the first time in this period since records began in " + str(c.season["first_year"]) if m == 0 else
               f"{f / m:.1f} times its average of {m:,.1f} for the same period in earlier years" if f > m else f"compared with an average of {m:,.1f} in earlier years")
        out.append(f"{r['site']} had {f:,} VIIRS fire alerts in {lab}, {rel}.")
        if r["fire_rank"] == 1 and m > 0:
            out.append(f"This is the highest count for this period since {c.season['first_year']}.")
    out.append(f"Another {r['ring_now']:,} alerts fell within {cfg['ring_km']} km of the boundary, and {r['ring10_now']:,} within {cfg['map_ring_km']} km.")
    out.append(f"GFW integrated deforestation alerts in the same period: {r['ia_now']:,} (a year earlier: {r['ia_prev']:,}). These count 10–30 m alert pixels, not hectares.")
    if r["tcl"]:
        out.append(f"Tree cover loss since 2001 totals {r['tcl_total']:,.1f} ha ({r['tcl_pct'] or 0:.2f}% of 2000 tree cover), peaking in {r['tcl_peak_year']} at {r['tcl_peak']:,.1f} ha.")
    return " ".join(out)


def fb_text(c, r):
    fb = r["firebreak"]; cfg = c.st["config"]; y0, y1 = fb["years"]; n = int(y1) - int(y0) + 1
    t = (f"The boundary is split into {fb['sections']} sections of about {fb['section_km']:g} km. For each section the map counts in how many of the "
         f"{n} years {y0}–{y1} VIIRS alerts fell within {cfg['fb_outer_km']:g} km outside to {cfg['fb_inner_km']:g} km inside it. "
         f"{fb['recurring']} sections had fire in 6 or more years. This season {fb['edge_now']:,} alerts fell in that edge zone. ")
    t += ("Sections where fire returns most years are candidates for firebreaks, early-season burning or community fire patrols. "
          "Check each one on the ground with the community: slope, wind, vegetation, gardens, roads and rivers that already stop fire, and who will maintain it.")
    if is_proposed(r) and "circle" in r["status"]:
        t += " This site has no surveyed boundary yet (approximate circle), so the sections are indicative only."
    return t


def site_outputs(c, r, G):
    name, slug = r["site"], r["id"]; cfg = c.st["config"]
    titles = {"ring": f"{name}: fire alerts inside and within {cfg['map_ring_km']} km, {c.season['label']}",
              "inside": f"{name}: fire alerts inside the site, {c.season['label']}",
              "firebreak": f"{name}: where fire keeps reaching the boundary, {c.season['first_year']}–{r['firebreak']['years'][1]}"}
    for mode, suffix in (("ring", ""), ("inside", "_inside"), ("firebreak", "_firebreak")):
        fig, ax = plt.subplots(figsize=(11, 8.8))
        site_map(c, ax, r, G, mode)
        ax.set_title(titles[mode], loc="left", fontsize=13, fontweight="bold")
        fig.text(0.01, 0.01, textwrap.fill("Imagery: Esri, Maxar, Earthstar Geographics. Rivers and lakes: HydroSHEDS (CC BY 4.0). " + c.src, 200), fontsize=6.3, color="#555")
        fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(os.path.join(c.out, f"maps/sites/{slug}{suffix}.jpg"), dpi=120, pil_kwargs={"quality": 85}); plt.close(fig)
    yk = sorted(r["years"]); cur = yk[-1]
    with PdfPages(os.path.join(c.out, f"reports/sites/{slug}.pdf")) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        header(fig, name, f"{r['project']} · {r['province']} · {r['type']} · {r['status']} · {r['area_ha']:,.0f} ha")
        kpis(fig, [(f"Fire alerts\n{c.season['label']}", fmt(r["fire_now"]), f"avg {r['fire_mean']:,.1f} in earlier years"),
                   ("Versus average", ratio_txt(r), f"2015: {r['fire_2015']:,}"),
                   (f"Fire alerts\n0–{cfg['ring_km']} km outside", fmt(r["ring_now"]), f"0–{cfg['map_ring_km']} km: {r['ring10_now']:,}"),
                   ("Integrated alerts\nsame period", fmt(r["ia_now"]), f"year before: {r['ia_prev']:,}"),
                   ("Tree cover loss\nsince 2001", f"{r['tcl_total']:,.1f} ha", f"{r['tcl_pct'] or 0:.2f}% of 2000 cover")])
        ax = fig.add_axes([0.05, 0.27, 0.9, 0.52]); site_map(c, ax, r, G, "ring")
        ax.set_title(f"Fire alerts inside the site and within {cfg['map_ring_km']} km", loc="left", fontsize=9.5, fontweight="bold")
        fig.text(0.05, 0.245, "Summary", fontsize=10, fontweight="bold", color=INK)
        fig.text(0.05, 0.235, textwrap.fill(site_text(c, r), 118), fontsize=7.8, va="top", color=INK, linespacing=1.5)
        fig.text(0.05, 0.03, textwrap.fill("Boundary: " + r["source"] + ". Imagery: Esri, Maxar, Earthstar Geographics. Rivers: HydroSHEDS. " + c.src, 150), fontsize=6, color="#666")
        pdf.savefig(fig); plt.close(fig)
        # page 2: inside only
        fig = plt.figure(figsize=(8.27, 11.69)); header(fig, name, f"Fire alerts inside the site only · {c.season['label']}")
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7]); site_map(c, ax, r, G, "inside")
        fig.text(0.05, 0.17, textwrap.fill(f"{r['inside_pts']:,} VIIRS fire alerts fell inside the boundary in {c.season['label']}. Alerts outside the site are not shown. "
                                           "Coordinates of every alert are in site_fire_alerts_10km.csv on the dashboard's Downloads page.", 118), fontsize=7.8, va="top", color=INK, linespacing=1.5)
        fig.text(0.05, 0.03, textwrap.fill(c.src, 150), fontsize=6, color="#666")
        pdf.savefig(fig); plt.close(fig)
        # page 3: firebreak planning
        fig = plt.figure(figsize=(8.27, 11.69)); header(fig, name, "Firebreak planning: where fire keeps reaching the boundary")
        ax = fig.add_axes([0.05, 0.36, 0.9, 0.55]); site_map(c, ax, r, G, "firebreak")
        fig.text(0.05, 0.335, textwrap.fill(fb_text(c, r), 118), fontsize=7.6, va="top", color=INK, linespacing=1.45)
        top = r["firebreak"]["top"]
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.17]); ax.set_title("Boundary sections with the most years of fire (numbered on the map)", loc="left", fontsize=9, fontweight="bold")
        table(ax, [[str(t[0]), f"{t[1]} of {int(r['firebreak']['years'][1]) - int(r['firebreak']['years'][0]) + 1}", f"{t[2]:,}", f"{t[3]:,}", f"{t[4]:.5f}, {t[5]:.5f}"] for t in top],
              ["Section", "Years with fire", f"Alerts {r['firebreak']['years'][0]}–{r['firebreak']['years'][1]}", "Alerts this season", "Midpoint (lat, lon)"],
              [0.1, 0.18, 0.2, 0.18, 0.3], fs=7.4)
        if not top:
            ax.text(0, 0.5, "No fire alerts near the boundary since " + str(c.season["first_year"]) + ".", fontsize=8)
        fig.text(0.05, 0.025, textwrap.fill("All sections with coordinates: firebreak_sections.csv on the Downloads page. " + c.src, 150), fontsize=6, color="#666")
        pdf.savefig(fig); plt.close(fig)
        # page 4: trends
        fig = plt.figure(figsize=(8.27, 11.69)); header(fig, name, "Fire and forest change trends")
        ax = fig.add_axes([0.09, 0.64, 0.85, 0.23])
        bars(ax, yk, [r["years"][y] for y in yk], [y == cur for y in yk], f"Fire alerts, {c.season['label'][:-5]} of each year", "alerts",
             ref=r["fire_mean"] or None, reflab="earlier-years average", lab=lambda k: "'" + k[2:])
        mo = list(r["monthly"]); vals = [r["monthly"][k] for k in mo]; mx = max(vals) if vals else 0
        ax = fig.add_axes([0.09, 0.36, 0.85, 0.21])
        bars(ax, mo, vals, [v == mx and v > 0 for v in vals], "Fire alerts by month this season", "alerts", lab=lambda k: mlabel(k))
        ty = sorted(r["tcl"]); tv = [r["tcl"][k] for k in ty]; tm = max(tv) if tv else 0
        ax = fig.add_axes([0.09, 0.08, 0.85, 0.21])
        bars(ax, ty, tv, [v == tm and v > 0 for v in tv], "Tree cover loss (ha, canopy over 30%)", "ha", lab=lambda k: "'" + k[2:], valfmt=lambda v: f"{v:,.1f}")
        fig.text(0.05, 0.035, textwrap.fill(f"Method: NASA VIIRS 375 m fire alerts (all confidence levels) counted inside the boundary with the GFW Data API (geostore {r['geostore']}), compared with the same calendar period in every year since {c.season['first_year']}. Hansen/UMD tree cover loss.", 160), fontsize=6.3, color="#666")
        pdf.savefig(fig); plt.close(fig)


def province_report(c, n, d):
    slug = d["slug"]; yk = sorted(d["years"]); cur = yk[-1]
    avg = float(np.mean([d["years"][y] for y in yk[:-1]])) if len(yk) > 1 else 0.0; f = d["years"][cur]
    with PdfPages(os.path.join(c.out, f"reports/provinces/{slug}.pdf")) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69)); header(fig, f"{n} Province", f"{d['project']} · fire situation, {c.season['label']}")
        bm = max(d["monthly"], key=d["monthly"].get) if d["monthly"] else None
        kpis(fig, [(f"Fire alerts\n{c.season['label']}", f"{f:,}", f"avg {avg:,.0f} in earlier years"),
                   ("Versus average", f"{f / avg:.1f}×" if avg else "–", ""),
                   ("Versus 2015 El Niño", f"{100 * f / d['years']['2015']:.0f}%" if d["years"].get("2015") else "–", f"2015: {d['years'].get('2015', 0):,}"),
                   ("Busiest month", mlabel(bm, True) if bm else "–", f"{d['monthly'].get(bm, 0):,} alerts" if bm else "")])
        im = plt.imread(os.path.join(c.out, f"maps/overview/monthly_{slug}.png"))
        ax = fig.add_axes([0.03, 0.2, 0.94, 0.6]); ax.imshow(im); ax.axis("off")
        ds = sorted([x for x in c.st["districts"] if x["province"] == n], key=lambda x: -x["fires"])
        ax = fig.add_axes([0.05, 0.03, 0.9, 0.15])
        table(ax, [[x["name"]] + [f"{x['monthly'][m]:,}" for m in c.months] + [f"{x['fires']:,}", f"{x['density']:.1f}"] for x in ds],
              ["District"] + [mlabel(m) for m in c.months] + ["Total", "per 100 km²"])
        pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=(8.27, 11.69)); header(fig, f"{n} Province", "Trends and GEF sites")
        ax = fig.add_axes([0.1, 0.66, 0.84, 0.21])
        bars(ax, yk, [d["years"][y] for y in yk], [y == cur for y in yk], f"Fire alerts, {c.season['label'][:-5]} of each year", "alerts",
             ref=avg, reflab="earlier-years average", lab=lambda k: "'" + k[2:])
        tcl = d.get("tcl", {}); ty = sorted(tcl)
        if ty:
            ax = fig.add_axes([0.1, 0.39, 0.84, 0.2]); tv = [tcl[k]["loss"] for k in ty]; tm = max(tv)
            bars(ax, ty, tv, [v == tm for v in tv], "Tree cover loss (ha, canopy over 30%)", "ha", lab=lambda k: "'" + k[2:])
        inprov = lambda s: n in [ABBR.get(x.strip(), x.strip()) for x in s["province"].split("/")]
        ss = sorted([s for s in c.st["sites"] if inprov(s)], key=lambda s: -s["fire_now"])
        ax = fig.add_axes([0.05, 0.04, 0.9, 0.3]); ax.set_title("GEF sites in this province", loc="left", fontsize=9.5, fontweight="bold")
        table(ax, [[s["site"], s["type"], f"{s['fire_now']:,}", f"{s['fire_mean']:,.1f}", ratio_txt(s), f"{s['ring_now']:,}", f"{s['tcl_total']:,.1f}"] for s in ss[:22]],
              ["Site", "Type", "Fires now", "Avg", "vs avg", f"0–{c.st['config']['ring_km']} km", "Loss since 2001 (ha)"],
              [0.32, 0.14, 0.1, 0.08, 0.08, 0.1, 0.14], fs=6.8)
        note = "Western Highlands tree cover loss uses GFW's older, smaller province boundary. " if n == "Western Highlands" else ""
        fig.text(0.05, 0.015, textwrap.fill(note + c.src, 160), fontsize=6, color="#666")
        pdf.savefig(fig); plt.close(fig)


def summary_report(c):
    st = c.st; lab = c.season["label"]
    P = st["provinces"]; names = list(P)
    cur = max(P[names[0]]["years"])
    tot = sum(P[n]["years"][cur] for n in names); tot15 = sum(P[n]["years"].get("2015", 0) for n in names)
    sites = st["sites"]
    above = [s for s in sites if s["fire_now"] > s["fire_mean"] and s["fire_now"] > 0]
    rec = [s for s in sites if s["fire_rank"] == 1 and s["fire_now"] > 0]
    with PdfPages(os.path.join(c.out, "reports/fire_situation_summary.pdf")) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        header(fig, "Fire situation summary", f"GEF7 & GEF8 provinces, Papua New Guinea · {lab} · updated {c.season['updated']}")
        kpis(fig, [(f"Fire alerts, 4 provinces\n{lab}", f"{tot:,}", f"2015 El Niño: {tot15:,}"),
                   ("Sites above their average", f"{len(above)} of {len(sites)}", "same period, earlier years"),
                   ("Sites at a record", f"{len(rec)}", f"highest since {c.season['first_year']}")])
        lines = []
        for n in names:
            y = P[n]["years"]; avg = np.mean([y[k] for k in sorted(y)[:-1]])
            rel = f"{y[cur] / avg:.1f}× the average for this period" if avg else "no alerts in this period in earlier years"
            lines.append(f"{n} ({P[n]['project']}): {y[cur]:,} alerts, {rel}"
                         + (f" and {100 * y[cur] / y['2015']:.0f}% of 2015." if y.get("2015") else "."))
        top = sorted(sites, key=lambda s: -s["fire_now"])[:3]
        lines.append("Most alerts among project sites: " + ", ".join(f"{s['site']} ({s['fire_now']:,})" for s in top) + ".")
        topd = sorted(st["districts"], key=lambda d: -d["fires"])[:4]
        lines.append("Districts with the most alerts: " + ", ".join(f"{d['name']} ({d['fires']:,})" for d in topd) + ".")
        y0 = 0.76
        fig.text(0.05, y0 + 0.012, "Key points", fontsize=10, fontweight="bold", color=INK)
        for t in lines:
            w = textwrap.fill("•  " + t, 112, subsequent_indent="    ")
            fig.text(0.05, y0, w, fontsize=8, va="top", color=INK); y0 -= 0.017 * (w.count("\n") + 1) + 0.006
        im = plt.imread(os.path.join(c.out, "maps/overview/overview_all.png"))
        ax = fig.add_axes([0.03, 0.06, 0.94, y0 - 0.08]); ax.imshow(im); ax.axis("off")
        fig.text(0.05, 0.025, textwrap.fill(c.src, 150), fontsize=6, color="#666")
        pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=(8.27, 11.69)); header(fig, "Fire situation summary", "Provinces and districts")
        for i, n in enumerate(names):
            y = P[n]["years"]; yk = sorted(y); avg = np.mean([y[k] for k in yk[:-1]])
            ax = fig.add_axes([0.08 + 0.47 * (i % 2), 0.66 - 0.2 * (i // 2), 0.4, 0.14])
            bars(ax, yk, [y[k] for k in yk], [k == cur for k in yk], f"{n} ({P[n]['project']})", "alerts", ref=avg, reflab="avg", lab=lambda k: "'" + k[2:])
        ds = sorted(st["districts"], key=lambda d: -d["fires"])
        ax = fig.add_axes([0.05, 0.04, 0.9, 0.36]); ax.set_title(f"Districts, {lab}", loc="left", fontsize=9.5, fontweight="bold")
        table(ax, [[d["name"], d["province"]] + [f"{d['monthly'][m]:,}" for m in c.months] + [f"{d['fires']:,}", f"{d['density']:.1f}"] for d in ds],
              ["District", "Province"] + [mlabel(m) for m in c.months] + ["Total", "per 100 km²"], fs=6.8)
        pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=(8.27, 11.69)); header(fig, "Fire situation summary", "GEF7 and GEF8 sites")
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.86])
        rows = [[s["site"], s["project"], s["type"], f"{s['fire_now']:,}", f"{s['fire_mean']:,.1f}", ratio_txt(s), f"{s['ring_now']:,}"]
                for s in sorted(sites, key=lambda s: (-(s["ratio"] or (99 if s["fire_now"] else -1)), -s["fire_now"]))]
        table(ax, rows, ["Site", "Project", "Type", "Fires now", "Avg", "vs avg", f"0–{st['config']['ring_km']} km"], [0.36, 0.1, 0.14, 0.1, 0.08, 0.08, 0.1], fs=6.6)
        fig.text(0.05, 0.02, textwrap.fill("Sorted by fire alerts against the average for the same period in earlier years. 'new' = alerts where none were recorded before. " + c.src, 160), fontsize=6, color="#666")
        pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "_site")); ap.add_argument("--skip-sites", action="store_true")
    a = ap.parse_args(); c = Ctx(a.out)
    overview_all(c); monthly_maps(c); sites_maps(c); print("overview maps done", flush=True)
    for n, d in c.st["provinces"].items():
        province_report(c, n, d)
    summary_report(c); print("summary and province reports done", flush=True)
    if a.skip_sites:
        return
    G = c.sites.to_crs(3857)
    sp = pd.read_csv(os.path.join(c.out, "downloads", "site_fire_alerts_10km.csv"))
    sp["month"] = sp.alert__date.str[:7]
    sp["zone_i"] = sp.zone.map(lambda z: 0 if z == "inside site" else 1)
    c.SP = gpd.GeoDataFrame(sp, geometry=gpd.points_from_xy(sp.longitude, sp.latitude), crs=4326).to_crs(3857)
    c.FB = gpd.read_file(os.path.join(c.out, "data", "firebreak.geojson")).to_crs(3857)
    c.W = load_water(c)
    only = os.environ.get("ONLY_SITES", "").split(",") if os.environ.get("ONLY_SITES") else None
    for r in c.st["sites"]:
        if only and r["id"] not in only:
            continue
        site_outputs(c, r, G)
    print("site maps and reports done", flush=True)


if __name__ == "__main__":
    main()
