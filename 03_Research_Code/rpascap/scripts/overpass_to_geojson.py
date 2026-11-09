"""Convert an Overpass API JSON response (buildings, 'out geom') into osm_buildings.geojson.
Used when OpenStreetMap buildings are downloaded manually in a browser instead of with osmnx.
Usage: python scripts/overpass_to_geojson.py <overpass.json> <processed_dir>/osm_buildings.geojson
Only outer rings are kept (the exposure code samples outer footprints only)."""
import json, sys


def ring(geom):
    pts = [[g["lon"], g["lat"]] for g in geom if g]
    if len(pts) >= 3 and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts if len(pts) >= 4 else None


def main(src, dst):
    d = json.load(open(src))
    feats = []
    for el in d.get("elements", []):
        tags = el.get("tags", {})
        if "building" not in tags:
            continue
        if el["type"] == "way" and "geometry" in el:
            r = ring(el["geometry"])
            if r:
                feats.append({"type": "Feature", "properties": {"osm_id": f"way/{el['id']}", "building": tags["building"]},
                              "geometry": {"type": "Polygon", "coordinates": [r]}})
        elif el["type"] == "relation":
            polys = [[r] for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m
                     for r in [ring(m["geometry"])] if r]
            if polys:
                feats.append({"type": "Feature", "properties": {"osm_id": f"relation/{el['id']}", "building": tags["building"]},
                              "geometry": {"type": "MultiPolygon", "coordinates": polys}})
    json.dump({"type": "FeatureCollection", "features": feats,
               "source": "OpenStreetMap contributors (ODbL), via Overpass API",
               "osm_base": d.get("osm3s", {}).get("timestamp_osm_base")}, open(dst, "w"))
    print(f"{len(feats)} building footprints -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
