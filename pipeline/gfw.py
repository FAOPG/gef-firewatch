"""Global Forest Watch Data API helper.

With a GFW_API_KEY environment variable (free key from globalforestwatch.org) queries go to the
official Data API. Without one they go through the Global Nature Watch proxy, which needs no key
but is not a documented public service, so set a key for a dependable daily run.
"""
import json, os, time, requests

KEY = os.environ.get("GFW_API_KEY", "").strip()
OFFICIAL = "https://data-api.globalforestwatch.org/dataset/{ds}/latest/query/json"
PROXY = "https://globalnaturewatch.org/api/data/dataset/{ds}/latest/query/json/"
GEOSTORE = "https://production-api.globalforestwatch.org/v1/geostore"
S = requests.Session()


def query(ds, sql, geostore=None, tries=5):
    params = {"sql": sql}
    if geostore:
        params.update(geostore_id=geostore, geostore_origin="rw")
    url = (OFFICIAL if KEY else PROXY).format(ds=ds)
    headers = {"x-api-key": KEY} if KEY else {}
    last = None
    for i in range(tries):
        try:
            r = S.get(url, params=params, headers=headers, timeout=600)
            j = r.json()
            if "data" in j:
                return j["data"]
            last = str(j)[:300]
        except Exception as e:  # network or JSON error
            last = str(e)[:300]
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"GFW query failed for {ds}: {last}")


def create_geostore(geometry_geojson):
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": geometry_geojson}]}
    r = S.post(GEOSTORE, json={"geojson": fc}, timeout=300)
    r.raise_for_status()
    return r.json()["data"]["id"]
