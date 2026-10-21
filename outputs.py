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
SCOL = {"CCA": "#1b7f3b", "Nature park": "#1b7f3b", "Proposed CA": "#7b3fa0", "Proposed park": "#7b3fa0", "Proposed PA": "#7b3fa0",
        "KBA": "#1f5fbf", "WMA": "#0e8c8c", "Provincial park": "#0e8c8c", "National park": "#0e8c8c", "Sanctuary": "#0e8c8c",
        "World Heritage site": "#b8860b"}


def mlabel(m, long=False, season=None):
    y, mo = int(m[:4]), int(m[5:])
    lab = (MNL if long else MN)[mo - 1]
    if season and m == season["end"][:7] and int(season["end"][8:]) < 28:
        lab = f"1–{int(season['end'][8:])} {lab}"
    return lab


def month_colors(months):
    cm = plt.get_cmap("YlOrRd")
    n = len(months)
    return {m: matplotlib.colors.to_hex(cm(0.25 + 0.7 * (i / max(1, n - 1)))) for i, m in enumerate(months)}


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
        self.src = (f"Fire: NASA VIIRS via Global Forest Watch, to {self.season['end']}. Boundaries: CEPA/FAO GEF7 & GEF8 project GIS, GADM 4.1. "
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
    for m in c.months:
        s = c.P[c.P.month == m]
        ax.scatter(s.longitude, s.latitude, s=1.2, c=c.mcol[m], lw=0, alpha=0.85, zorder=3)
    c.dist.boundary.plot(ax=ax, color="#777777", lw=0.4, zorder=4)
    c.prov.boundary.plot(ax=ax, color="#111111", lw=1.4, zorder=5)
    for r in c.prov.itertuples():
        pt = r.geometry.representative_point()
        n = c.st["provinces"][r.name]["season_points"]
        ax.text(pt.x, pt.y, f"{r.name}\n({r.project})\n{n:,} alerts", fontsize=10, fontweight="bold", ha="center", zorder=6,
                bbox=dict(fc="white", alpha=0.75, lw=0, pad=2))
    b = c.prov.total_bounds
    ax.set_xlim(b[0] - 0.3, b[2] + 0.3); ax.set_ylim(b[1] - 0.3, b[3] + 0.3); ax.set_aspect("equal"); ax.tick_params(labelsize=8)
    h = [Line2D([], [], marker="o", ls="", mfc=c.mcol[m], mec="none", ms=7, label=mlabel(m, True, c.season)) for m in c.months]
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
        cmap = ListedColormap(["#fff5eb", "#fdd0a2", "#fd8d3c", "#e6550d", "#a63603"][:len(br) - 1]); norm = BoundaryNorm(br, cmap.N)
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
            ax.scatter(s.longitude, s.latitude, s=ms, c="#3b0a0a", lw=0, alpha=0.6, zorder=3)
            p1.boundary.plot(ax=ax, color="#111", lw=1.3, zorder=4)
            for r in p2.itertuples():
                pt = r.geometry.representative_point()
                ax.text(pt.x, pt.y, f"{r.name}\n{c.D[r.name]['monthly'][m]:,}", fontsize=7.5, ha="center", va="center", zorder=6,
                        bbox=dict(fc="white", alpha=0.7, lw=0, pad=1))
            px, py = (b[2] - b[0]) * 0.04, (b[3] - b[1]) * 0.04
            ax.set_xlim(b[0] - px, b[2] + px); ax.set_ylim(b[1] - py, b[3] + py); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{mlabel(m, True, c.season)} {m[:4]}: {len(s):,} alerts", loc="left", fontsize=12, fontweight="bold")
        hand = [Patch(fc=cmap(i), ec="#555", label=f"{br[i]:g}–{br[i + 1]:g}") for i in range(cmap.N)]
        hand += [Line2D([], [], marker="o", ls="", mfc="#3b0a0a", mec="none", ms=4, label="VIIRS fire alert")]
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
        ax.scatter(p.longitude, p.latitude, s=1.2, c="#e0531f", alpha=0.45, lw=0, zorder=2)
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
        h = [Line2D([], [], marker="o", ls="", mfc="#e0531f", mec="none", ms=5, label=f"VIIRS fire alert, {c.season['label']}"),
             Line2D([], [], color="#1b7f3b", lw=2, label="CCA / nature park"), Line2D([], [], color="#0e8c8c", lw=2, label="WMA / park / sanctuary"),
             Line2D([], [], color="#1f5fbf", lw=2, label="Key Biodiversity Area"), Line2D([], [], color="#7b3fa0", lw=2, ls="--", label="Proposed protected area"),
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


def site_map(c, ax, name, G, P3):
    import contextily as cx
    geom = G[G["site"] == name].iloc[0].geometry
    ring = geom.buffer(c.st["config"]["ring_km"] * 1000 / np.cos(np.radians(-6)))
    minx, miny, maxx, maxy = ring.bounds
    cxx, cyy = (minx + maxx) / 2, (miny + maxy) / 2; s = max(maxx - minx, (maxy - miny) * 1.25) * 1.08
    ax.set_xlim(cxx - s / 2, cxx + s / 2); ax.set_ylim(cyy - s / 2 / 1.25, cyy + s / 2 / 1.25)
    try:
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, attribution=False, zoom="auto")
    except Exception as e:
        print("basemap failed", name, e)
        ax.set_facecolor("#dfe3dc")
    gpd.GeoSeries([ring], crs=3857).boundary.plot(ax=ax, color="white", lw=1, ls="--", zorder=5)
    gpd.GeoSeries([geom], crs=3857).boundary.plot(ax=ax, color="#00d5ff", lw=2, zorder=6)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    vis = P3[(P3.geometry.x > x0) & (P3.geometry.x < x1) & (P3.geometry.y > y0) & (P3.geometry.y < y1)]
    for m in c.months:
        sub = vis[vis.month == m]
        if len(sub):
            ax.scatter(sub.geometry.x, sub.geometry.y, s=14 if len(vis) < 3000 else 4, c=c.mcol[m], edgecolors="black", linewidths=0.3, zorder=7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel(""); scalebar(ax)
    h = [Line2D([], [], color="#00d5ff", lw=2, label="Site boundary"), Line2D([], [], color="white", lw=1, ls="--", label=f"{c.st['config']['ring_km']} km ring")]
    h += [Line2D([], [], marker="o", ls="", mfc=c.mcol[m], mec="k", mew=0.4, ms=6, label=f"Fire alert, {mlabel(m, True, c.season)}") for m in c.months]
    lg = ax.legend(handles=h, loc="upper left", fontsize=7, framealpha=0.85); lg.set_zorder(30)


def site_text(c, r):
    f, m = r["fire_now"], r["fire_mean"]; lab = c.season["label"]; out = []
    if f == 0:
        out.append(f"No VIIRS fire alerts fell inside {r['site']} in {lab}.")
    else:
        rel = ("for the first time in this period since records began in " + str(c.season["first_year"]) if m == 0 else
               f"{f / m:.1f} times its average of {m:,.1f} for the same period in earlier years" if f > m else f"compared with an average of {m:,.1f} in earlier years")
        out.append(f"{r['site']} had {f:,} VIIRS fire alerts in {lab}, {rel}.")
        if r["fire_rank"] == 1 and m > 0:
            out.append(f"This is the highest count for this period since {c.season['first_year']}.")
    out.append(f"Another {r['ring_now']:,} alerts fell within {c.st['config']['ring_km']} km of the boundary.")
    out.append(f"GFW integrated deforestation alerts in the same period: {r['ia_now']:,} (a year earlier: {r['ia_prev']:,}). These count 10–30 m alert pixels, not hectares.")
    if r["tcl"]:
        out.append(f"Tree cover loss since 2001 totals {r['tcl_total']:,.1f} ha ({r['tcl_pct'] or 0:.2f}% of 2000 tree cover), peaking in {r['tcl_peak_year']} at {r['tcl_peak']:,.1f} ha.")
    return " ".join(out)


def site_outputs(c, r, G, P3):
    name, slug = r["site"], r["id"]
    fig, ax = plt.subplots(figsize=(11, 8.8))
    site_map(c, ax, name, G, P3)
    ax.set_title(f"{name}: fire alerts, {c.season['label']}", loc="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Imagery: Esri, Maxar, Earthstar Geographics. " + c.src, fontsize=6.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(c.out, f"maps/sites/{slug}.jpg"), dpi=120, pil_kwargs={"quality": 85}); plt.close(fig)
    yk = sorted(r["years"]); cur = yk[-1]
    with PdfPages(os.path.join(c.out, f"reports/sites/{slug}.pdf")) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        header(fig, name, f"{r['project']} · {r['province']} · {r['type']} · {r['status']} · {r['area_ha']:,.0f} ha")
        kpis(fig, [(f"Fire alerts\n{c.season['label']}", fmt(r["fire_now"]), f"avg {r['fire_mean']:,.1f} in earlier years"),
                   ("Versus average", ratio_txt(r), f"2015: {r['fire_2015']:,}"),
                   (f"Fire alerts\n0–{c.st['config']['ring_km']} km outside", fmt(r["ring_now"]), "same period"),
                   ("Integrated alerts\nsame period", fmt(r["ia_now"]), f"year before: {r['ia_prev']:,}"),
                   ("Tree cover loss\nsince 2001", f"{r['tcl_total']:,.1f} ha", f"{r['tcl_pct'] or 0:.2f}% of 2000 cover")])
        ax = fig.add_axes([0.05, 0.27, 0.9, 0.52]); site_map(c, ax, name, G, P3)
        fig.text(0.05, 0.245, "Summary", fontsize=10, fontweight="bold", color=INK)
        fig.text(0.05, 0.235, textwrap.fill(site_text(c, r), 118), fontsize=7.8, va="top", color=INK, linespacing=1.5)
        fig.text(0.05, 0.03, textwrap.fill("Boundary: " + r["source"] + ". Imagery: Esri, Maxar, Earthstar Geographics. " + c.src, 150), fontsize=6, color="#666")
        pdf.savefig(fig); plt.close(fig)
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
        fig.text(0.05, 0.035, textwrap.fill(f"Method: NASA VIIRS 375 m fire alerts counted inside the boundary with the GFW Data API (geostore {r['geostore']}), compared with the same calendar period in every year since {c.season['first_year']}. Hansen/UMD tree cover loss.", 160), fontsize=6.3, color="#666")
        pdf.savefig(fig); plt.close(fig)


def province_report(c, n, d):
    slug = d["slug"]; yk = sorted(d["years"]); cur = yk[-1]
    avg = float(np.mean([d["years"][y] for y in yk[:-1]])); f = d["years"][cur]
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
            lines.append(f"{n} ({P[n]['project']}): {y[cur]:,} alerts, {y[cur] / avg:.1f}× the average for this period"
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
    P3 = c.P.to_crs(3857)
    for r in c.st["sites"]:
        site_outputs(c, r, G, P3)
    print("site maps and reports done", flush=True)


if __name__ == "__main__":
    main()
