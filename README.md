# GEF Fire Watch PNG

A live fire dashboard for the CEPA–FAO GEF7 project (Western and Eastern Highlands) and GEF8 project (Western and Gulf), Papua New Guinea.

It shows:

- **Map:** NASA VIIRS fire alerts for the current fire season, coloured by month (newest bright yellow), with GEF project sites, districts, rivers and lakes. Site outlines are coloured by type: green for protected and conserved areas, white (blue on the street map) for KBAs, dashed for proposed areas.
- **Low-confidence filter:** "Exclude low-confidence alerts" switches every number, chart and table. Both versions are computed in each run.
- **Site pages:** charts comparing this season with the same period in every year since 2012, alerts 0–5 km and 0–10 km outside, and a "Show alerts" switch for all alerts, inside the site only, or within 10 km.
- **Nearest water:** clicking a fire alert shows the nearest mapped river or lake (HydroSHEDS), its distance and direction, and draws a line to it.
- **Firebreak planning:** each site boundary is cut into ~1 km sections, coloured by how many years since 2012 fire alerts reached that section. The site page lists the sections that burn most often.
- **Downloads:** a summary report, province reports, four-page site reports (PDF), three printable maps per site (10 km, inside only, firebreak), and data (CSV and GeoJSON).

**It updates itself.** A GitHub Action runs twice a day. It checks Global Forest Watch for new fire alerts. When there are any, it rebuilds all numbers, maps and reports, then republishes the site. Nothing needs to be run by hand.

## Set up (once)

1. **Create the repository.**
   1. On github.com, create a new **public** repository, for example `gef-firewatch`.
   2. Upload everything in this folder, including the hidden `.github` folder, so that `index.html` sits at the top level.
      - With the GitHub website, drag the folder contents in. Then check that `.github/workflows/update.yml` is there, and create it with **Add file → Create new file** if it is missing.
      - With git:
        ```
        git init && git add . && git commit -m "GEF Fire Watch PNG"
        git branch -M main
        git remote add origin https://github.com/<you>/gef-firewatch.git
        git push -u origin main
        ```
2. **Turn on Pages.** Go to **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**, then save.
3. **Allow the workflow to run.** Go to **Settings → Actions → General → Workflow permissions** and select **Read and write permissions**. Save.
4. **Add a GFW API key (required for dependable runs)** so the daily queries go through the official Global Forest Watch Data API:
   1. Create a free account at globalforestwatch.org.
   2. Get an API key. See https://data-api.globalforestwatch.org/#tag/Authentication.
   3. Go to **Settings → Secrets and variables → Actions → New repository secret**, name it `GFW_API_KEY` and paste the key.
   - Without a key, the pipeline uses the Global Nature Watch proxy. It works, but it is not an official public service.
5. **(Recommended) Tell the workflow where the site lives** so it can skip runs when there is no new data. Under **Settings → Secrets and variables → Actions → Variables**, add `SITE_URL` with the value `https://<you>.github.io/gef-firewatch`.
6. **Run it the first time.** Go to **Actions → Update GEF Fire Watch → Run workflow**, tick **force**, and run it. A run takes about 10 minutes. The site then appears at `https://<you>.github.io/gef-firewatch/`.

After that it runs on its own at about 08:15 and 20:15 PNG time. A run also starts whenever you push a change to the page, inputs or pipeline.

## Changing things

| To change | Edit |
|---|---|
| Sites (add, remove, update a boundary) | `inputs/sites.geojson`. Each feature needs `id`, `site`, `project`, `province`, `type`, `status` and `source`. Leave `geostore` empty for a new or changed boundary, and the next run creates one. |
| Start of the fire season (default 1 July) | `season_start` in `config.json` |
| First comparison year (default 2012) | `baseline_first_year` in `config.json` |
| Rings around sites (default 5 km, and 10 km for printed maps) | `ring_km` and `map_ring_km` in `config.json` |
| Firebreak edge zone (default 2 km outside to 1 km inside) and section length (1 km, at most 300 per site) | `fb_outer_km`, `fb_inner_km`, `fb_section_km`, `fb_max_sections`. After changing any of these, delete `inputs/firebreak_history.json`; the next run rebuilds it (about 4 minutes). |
| How far to look for water (default 30 km) | `water_max_km` in `config.json` |
| Credit line on maps and reports | `credit` in `config.json` |
| Logging and mining layers (CEPA) | Not published by default. Once CEPA approves sharing, add `inputs/logging_existing.geojson`, `inputs/logging_proposed.geojson` and `inputs/mining_el.geojson`, and set `publish_overlays` to `true` in `config.json`. |
| Province tree cover loss (yearly) | `inputs/province_tree_cover_loss.json`. Update it when GFW releases the new year's loss data. |

## Run locally

```
pip install -r pipeline/requirements.txt
python pipeline/update.py --out _site --force
python pipeline/outputs.py --out _site
cp index.html _site/
cd _site && python -m http.server 8000
```

Then open http://localhost:8000.

## How it works

1. **`pipeline/update.py`** finds the latest VIIRS alert date. If the live site already has it, the script stops.
2. **Otherwise it queries the GFW Data API**:
   - the season's fire points for the four provinces;
   - the season's fire points within 10 km of every site (using a second geostore per site, `geostore10`);
   - daily alert counts since 2012, split by confidence, for each province and each site;
   - integrated deforestation alerts and tree cover loss for each site.

   It joins the points to districts, finds the nearest mapped water for each point (`pipeline/water.py`), assigns points near each boundary to a section (`pipeline/firebreak.py`), and writes `data/*.json` and the download tables.
   - **Firebreak baseline.** `inputs/firebreak_history.json` holds alerts per boundary section and calendar year since 2012. It is rebuilt automatically when it is missing, when a new season starts, or when a site boundary changes, and the workflow commits it.
3. **`pipeline/outputs.py`** draws:
   - the overview and monthly district maps;
   - three satellite maps per site (10 km, inside only, firebreak), using Esri imagery;
   - the PDF reports: one summary, 4 provinces and 40 sites.
4. **The workflow publishes `_site/`** to GitHub Pages. Generated files are not committed, so the repository stays small.
5. **If a run fails,** the workflow opens a GitHub issue labelled `update-failure` (or comments on the open one). Click **Watch → All activity** (or at least Issues) on the repository to get these by email. The site keeps showing the last good update.

## Before making it public

Some boundaries are proposed or community-sensitive, such as proposed CCAs that are not yet gazetted. Confirm with CEPA and the FAO GEF Coordinator which sites can be shown publicly. Remove any that should stay internal from `inputs/sites.geojson`.

## Data and credits

- Fire alerts: NASA VIIRS (375 m) via the Global Forest Watch Data API.
- Tree cover loss: Hansen/UMD.
- Integrated alerts: GFW (GLAD-L, GLAD-S2, RADD).
- Site, protected area, KBA, logging and mining layers: CEPA/FAO GEF7 and GEF8 project GIS.
- Province and district boundaries: OCHA COD-AB / PNG National Statistical Office, 2019 (CC BY-IGO).
- Rivers and lakes: HydroRIVERS and HydroLAKES v1.0, HydroSHEDS (Lehner & Grill 2013), CC BY 4.0. Stored in `inputs/water.json`, clipped to the project provinces and 12 km around each site; rebuild with `python pipeline/water.py <HydroRIVERS_v10_au.shp> <HydroLAKES_polys_v10.shp>`. HydroRIVERS maps streams with at least 10 km² of upstream catchment, and line positions are approximate (often within 500 m).
- Satellite imagery in the site maps and PDFs: Esri World Imagery (Esri, Maxar, Earthstar Geographics), credited on every map.
- VIIRS satellites: GFW's VIIRS series is Suomi NPP only (VNP14IMGT), so the record since 2012 comes from one sensor. All confidence levels are counted by default; the dashboard can exclude low-confidence alerts. For Western Province, dropping them lowers totals by 7–14% but leaves year-to-year comparisons almost unchanged. PDFs, printed maps and firebreak counts use all alerts.
- Basemaps: Esri World Imagery and OpenStreetMap.
