"""End-to-end API test for vein branch maps and follow-vein line scans."""
import json
import tempfile

from folioleaf.web import create_app
from folioleaf.store import Store


def client():
    tmp = tempfile.mkdtemp(prefix="folioleaf-vein-")
    app = create_app(data_dir=tmp)
    app.testing = True
    return app.test_client(), tmp


def main():
    failures = []
    api, _ = client()

    seeded = api.post("/api/training", json={})
    assert seeded.status_code in (200, 201), seeded.data
    specimen = seeded.get_json()["items"][0]
    tile_id = specimen["id"]
    tile = api.get(f"/api/specimens/{tile_id}").get_json()
    photo_id = next(p["id"] for p in tile["photos"] if p["registration_state"] == "base")
    region_id = tile["regions"][0]["id"]

    # Mark anchors on the synthetic leaf's spine/ribs in normalized plane coords.
    # The generated leaf runs horizontally; pick points on visible dark veins.
    vein_payload = {
        "kind": "veinmap",
        "photo_id": photo_id,
        "region_id": region_id,
        "apply_neutral": True,
        "options": {
            "radius": 1.4, "sensitivity": 0.72, "join_px": 1,
            "component_minimum": 12, "minimum_cell": 24, "preferred_junction": 1,
            "midrib": [[0.50, 0.47], [0.60, 0.50]],
            "lateral_tips": [[0.36, 0.34], [0.66, 0.62]],
            "junctions": [[0.52, 0.485], [0.55, 0.515]],
        },
    }
    response = api.post(f"/api/specimens/{tile_id}/results", json=vein_payload)
    if response.status_code != 201:
        print("veinmap status:", response.status_code, response.data.decode()[:400])
        failures.append("veinmap request rejected")
        report(failures)
        return
    result = response.get_json()
    if result["kind"] != "veinmap":
        failures.append("wrong result kind")
    metrics = result["metrics"]
    if len(metrics["laterals"]) != 2:
        failures.append("expected two lateral records")
    for lateral in metrics["laterals"]:
        if lateral.get("trace") is None and lateral.get("stump") is None:
            failures.append(f"lateral {lateral['lateral']} kept no observed line")
    overlay = api.get(f"/api/results/{result['id']}/image/overlay")
    if overlay.status_code != 200 or overlay.mimetype != "image/png":
        failures.append("overlay image not served")

    # Validation: duplicate lateral tips must be rejected.
    bad = json.loads(json.dumps(vein_payload))
    bad["options"]["lateral_tips"][1] = bad["options"]["lateral_tips"][0]
    if api.post(f"/api/specimens/{tile_id}/results", json=bad).status_code != 422:
        failures.append("duplicate lateral tips were not rejected")

    # Follow-vein scan using the saved branch map.
    follow_payload = {
        "kind": "profile",
        "photo_id": photo_id,
        "region_id": region_id,
        "apply_neutral": True,
        "options": {
            "scan_mode": "follow_vein",
            "vein_result_id": result["id"],
            "lateral_number": 1,
            "start": [0.1, 0.1], "end": [0.9, 0.9],
            "samples": 90, "band_pixels": 3,
        },
    }
    scan_response = api.post(f"/api/specimens/{tile_id}/results", json=follow_payload)
    if scan_response.status_code != 201:
        print("follow scan status:", scan_response.status_code, scan_response.data.decode()[:400])
        failures.append("follow-vein scan rejected")
    else:
        scan = scan_response.get_json()["metrics"]
        if scan["mode"] != "follow_vein":
            failures.append("scan did not record follow_vein mode")
        if not any(row["state"] == "gap" for row in scan["rows"]) and scan["gap_station"] is not None:
            failures.append("gap stations not flagged")
        if scan["supported_stations"] < 3:
            failures.append("too few measured stations")

    # A follow-vein scan referencing a non-veinmap result must be rejected.
    straight = api.post(f"/api/specimens/{tile_id}/results", json={
        "kind": "profile", "photo_id": photo_id, "region_id": region_id,
        "apply_neutral": True,
        "options": {"scan_mode": "straight", "vein_result_id": None, "lateral_number": 1,
                    "start": [0.2, 0.4], "end": [0.8, 0.6], "samples": 60, "band_pixels": 5},
    })
    if straight.status_code != 201:
        failures.append(f"ordinary straight line scan broke: {straight.status_code}")
    else:
        straight_id = straight.get_json()["id"]
        wrong_ref = json.loads(json.dumps(follow_payload))
        wrong_ref["options"]["vein_result_id"] = straight_id
        if api.post(f"/api/specimens/{tile_id}/results", json=wrong_ref).status_code != 422:
            failures.append("follow scan accepted a non-veinmap reference")

    report(failures)


def report(failures):
    if failures:
        print("FAIL")
        for item in failures:
            print(" -", item)
        raise SystemExit(1)
    print("OK: API branch map, validation, follow-vein scan, straight scan backward compatibility")


if __name__ == "__main__":
    main()
