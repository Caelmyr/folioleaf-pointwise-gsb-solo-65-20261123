#!/usr/bin/env python3
"""End-to-end HTTP checks for fine-band row analysis and recheck."""
import sys
import tempfile

sys.path.insert(0, "/workspace")
for path in ("/opt/pointwise-env/site-packages", "/home/node/.local/lib/python3.11/site-packages"):
    if path not in sys.path:
        sys.path.append(path)

from folioleaf.demo import load_training
from folioleaf.store import Store
from folioleaf.web import create_app


def main():
    with tempfile.TemporaryDirectory() as directory:
        store = Store(directory)
        tiles, created = load_training(store)
        assert created
        app = create_app(directory)
        client = app.test_client()

        tile = client.get(f"/api/specimens/{tiles[0]['id']}").get_json()
        inspection = [r for r in tile["regions"] if r["purpose"] == "inspection"]
        assert len(inspection) >= 2
        base_photo = tile["plane"]["photo_id"]
        # Reuse the inspection regions as the row cells by creating extra region rows.
        region_ids = [r["id"] for r in inspection[:2]]
        # Need at least three cells; add a small region through the API.
        response = client.post(f"/api/specimens/{tile['id']}/regions", json={
            "version": tile["version"],
            "name": "Band cell extra",
            "purpose": "inspection",
            "vertices": [[0.30, 0.40], [0.44, 0.40], [0.44, 0.58], [0.30, 0.58]],
        })
        assert response.status_code == 201, response.get_json()
        updated_tile = response.get_json()
        new_region = next(r for r in updated_tile["regions"] if r["name"] == "Band cell extra")
        region_ids.append(new_region["id"])

        payload = {
            "photo_id": base_photo,
            "region_ids": region_ids,
            "reference_axis": 12,
            "tolerance": 28,
            "minimum_pitch": 5,
            "maximum_pitch": 30,
            "turn_limit": 38,
        }
        response = client.post(f"/api/specimens/{tile['id']}/band-rows", json=payload)
        assert response.status_code == 201, response.get_json()
        row = response.get_json()
        assert row["kind"] == "band_row"
        assert len(row["metrics"]["cells"]) == 3
        for cell in row["metrics"]["cells"]:
            used = [f["peak"] for f in cell["families"] if f["peak"] is not None]
            assert len(used) == len(set(used))
        assert row["metrics"]["conflict_policy"]
        assert row["pictures"]["overlay"] and row["pictures"]["source"]

        # Overlay and source images resolve.
        image = client.get(f"/api/band-rows/{row['id']}/image/overlay")
        assert image.status_code == 200 and image.data[:8] == b"\x89PNG\r\n\x1a\n"
        csv_response = client.get(f"/api/band-rows/{row['id']}/metrics.csv")
        assert csv_response.status_code == 200
        csv_text = csv_response.data.decode("utf-8-sig")
        assert "cell,region_name,family" in csv_text and csv_text.count("\r\n") >= 4
        listing = client.get(f"/api/specimens/{tile['id']}/band-rows")
        assert listing.status_code == 200 and len(listing.get_json()["items"]) == 1

        # Recheck: confirm the strongest peak of the middle cell into family B.
        middle = row["metrics"]["cells"][1]
        if not middle["peaks"]:
            raise AssertionError("middle cell produced no peaks to recheck")
        target_peak = middle["peaks"][0]["peak"]
        recheck = client.post(f"/api/band-rows/{row['id']}/recheck", json={
            "version": row["version"], "cell": 2, "family": "B", "peak": target_peak,
            "note": "on-image confirmation",
        })
        assert recheck.status_code == 201, recheck.get_json()
        revised = recheck.get_json()
        assert revised["recheck_of"] == row["id"]
        assert revised["metrics"]["reassignment"]["peak"] == target_peak
        assert revised["metrics"]["reassignment"]["family"] == "B"
        confirmed = next(f for f in revised["metrics"]["cells"][1]["families"] if f["family"] == "B")
        assert confirmed["status"] == "confirmed" and confirmed["peak"] == target_peak

        # The original reading still opens unchanged.
        original = client.get(f"/api/band-rows/{row['id']}").get_json()
        assert original["version"] == 1 and original["metrics"]["reassignment"] is None
        assert client.get(f"/api/specimens/{tile['id']}/band-rows").get_json()["items"][0]["id"] == revised["id"]

        # Stale version is rejected with a conflict.
        stale = client.post(f"/api/band-rows/{row['id']}/recheck", json={
            "version": 999, "cell": 1, "family": "A", "peak": 1,
        })
        assert stale.status_code == 409

        # A peak that does not exist in the chosen cell is rejected.
        bad_peak = client.post(f"/api/band-rows/{row['id']}/recheck", json={
            "version": row["version"], "cell": 1, "family": "A", "peak": 99,
        })
        assert bad_peak.status_code == 422

        # Validation: repeated regions rejected.
        bad = client.post(f"/api/specimens/{tile['id']}/band-rows", json={
            "photo_id": base_photo, "region_ids": [region_ids[0], region_ids[0], region_ids[1]],
        })
        assert bad.status_code == 422
        print("end-to-end band row HTTP checks passed")


if __name__ == "__main__":
    main()
