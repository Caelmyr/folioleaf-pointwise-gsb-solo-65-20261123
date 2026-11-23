"""Synthetic smoke test for vein branch maps and follow-vein scans.

Builds a registered plane with:
- a curved midrib broken by a horizontal mounting strip (excluded domain),
- two lateral veins joining candidate junction J1 above the strip,
- a press-strip edge line that also reaches candidate J2 (the route to reject),
- one very fine lateral that must still be retained.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

from folioleaf.geometry import SampleFrame
from folioleaf.vein import build_branch_map, follow_vein_scan


W = H = 320


def draw_polyline(draw, points, width, fill):
    draw.line(points, fill=fill, width=width, joint="curve")


def make_frame():
    bg = np.full((H, W, 3), 0.62, dtype=np.float32)
    image = Image.fromarray(np.rint(bg * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)

    # Curved midrib, split where the mounting strip will cover it.
    midrib_top = [(160, 12), (156, 60), (152, 100), (150, 141)]
    midrib_bottom = [(150, 177), (148, 220), (146, 260), (144, 308)]
    draw_polyline(draw, midrib_top, 3, (20, 24, 18))
    draw_polyline(draw, midrib_bottom, 3, (20, 24, 18))

    # J1 sits on the midrib just above the strip (strip covers rows 142-175).
    j1 = (150, 136)
    # Lateral A: solid vein from tip, physically joining the midrib at J1.
    lateral_a = [(58, 74), (96, 96), (126, 118), j1]
    draw_polyline(draw, lateral_a, 2, (26, 30, 22))

    # Lateral B: a deliberately very fine vein from tip, also joining J1.
    lateral_b = [(262, 82), (226, 98), (190, 116), (168, 128), j1]
    draw_polyline(draw, lateral_b, 1, (44, 48, 40))

    # Press-strip edge impression below the strip: a darker line from the
    # lower-right to J2 on the lower midrib — a real line, but the unwanted
    # alternative junction (it follows the strip edge rather than the vein).
    j2 = (150, 184)
    press_edge = [(258, 196), (224, 190), (192, 186), (170, 185), j2]
    draw_polyline(draw, press_edge, 2, (38, 34, 30))

    linear = np.asarray(image, dtype=np.float32) / 255.0
    valid = np.ones((H, W), dtype=bool)
    clipped = np.zeros((H, W), dtype=bool)

    # Exclude the mounting strip band itself: hidden veins leave real gaps.
    area = np.ones((H, W), dtype=bool)
    area[142:176, :] = False

    photo = {"id": "0" * 24, "name": "synthetic", "role": "diffuse"}
    frame = SampleFrame(linear, valid, photo, (1.0, 0, 0, 0, 1.0, 0, 0, 0),
                        scale=(0.1, 0.1), clipped=clipped)
    return frame, area


def main():
    frame, area = make_frame()
    options = {
        "radius": 1.4,
        "sensitivity": 0.72,
        "join_px": 1,
        "component_minimum": 8,
        "minimum_cell": 24,
        "preferred_junction": 0,
        "midrib": [[0.50, 0.05], [0.47, 0.40], [0.46, 0.72]],
        "lateral_tips": [[0.18, 0.23], [0.82, 0.26]],
        "junctions": [[0.469, 0.425], [0.469, 0.575]],
    }
    metrics, overlay, network = build_branch_map(frame, frame.linear, area, options)
    overlay.save("/tmp/vein_branch_overlay.png")

    failures = []

    midrib = metrics["midrib"]
    if not midrib["segments"]:
        failures.append("midrib retained no observed segment")
    if not midrib["occluded_gaps"]:
        failures.append("midrib did not report the mounting-strip gap")

    decisions = {item["lateral"]: item for item in metrics["junction_decisions"]}
    if decisions[1]["adopted_junction"] != 0 or decisions[2]["adopted_junction"] != 0:
        failures.append(f"expected both laterals to adopt J1, got {decisions}")

    for lateral in metrics["laterals"]:
        trace = lateral.get("trace") or lateral.get("stump")
        if trace is None or trace["pixel_count"] < 12:
            failures.append(f"lateral {lateral['lateral']} lost its observed trace")
        if lateral.get("open_gap"):
            failures.append(f"lateral {lateral['lateral']} unexpectedly open")

    # The fine lateral B must survive as its own trace to J1, not the press edge.
    fine = next(item for item in metrics["laterals"] if item["lateral"] == 2)
    fine_end = fine["trace"]["points"][-1]
    if abs(fine_end[1] - 0.43) > 0.04:
        failures.append(f"fine lateral ends at {fine_end}, expected near J1 (~0.43 v)")

    # Follow-vein scan along lateral B: curved reading, stations beyond a gap
    # would be empty; here the full trace is supported.
    scan = follow_vein_scan(frame, frame.linear, area, metrics, 2, 120, 3)
    measured = [row for row in scan["rows"] if row["state"] == "measured"]
    if len(measured) < 100:
        failures.append(f"follow scan only measured {len(measured)} stations")
    positions = np.array([row["relative_position"] for row in measured])
    straight = np.linalg.norm(positions[-1] - positions[0])
    walked = sum(
        np.linalg.norm(np.array(scan["rows"][i + 1]["relative_position"])
                       - np.array(scan["rows"][i]["relative_position"]))
        for i in range(len(measured) - 1)
    )
    if walked <= straight * 1.02:
        failures.append("follow scan did not follow the curved vein geometry")

    # Gap behavior: cut the domain midway along lateral B and confirm the scan
    # stops with empty stations instead of bridging.
    cut_area = area.copy()
    cut_area[100:130, :] = False
    gapped = follow_vein_scan(frame, frame.linear, cut_area, metrics, 2, 120, 3)
    gap_station = gapped["gap_station"]
    if gap_station is None or gap_station >= len(gapped["rows"]) - 2:
        failures.append("scan failed to stop at the introduced gap")
    if any(row["state"] == "measured" for row in gapped["rows"][gap_station + 1:]):
        failures.append("scan bridged across the gap instead of leaving stations empty")

    # Both-junctions-blocked case: cut lateral A's corridor into J1 and the
    # ridge corridor leading toward J2, so neither candidate is reachable;
    # the vein should keep its visible stump with an open gap rather than
    # inventing a connection.
    stump_area = area.copy()
    stump_area[128:142, 140:165] = False
    stump_area[118:132, 96:132] = False
    stump_metrics, _, _ = build_branch_map(frame, frame.linear, stump_area, options)
    a = next(item for item in stump_metrics["laterals"] if item["lateral"] == 1)
    if a.get("adopted_junction") is not None:
        failures.append("occluded lateral unexpectedly reached a junction")
    if not a.get("open_gap") or not a.get("stump"):
        failures.append("occluded lateral was not retained as an open-gap stump")

    if failures:
        print("FAIL")
        for item in failures:
            print(" -", item)
        sys.exit(1)
    print("OK: branch map, junction choice, fine-vein retention, gap stumps, follow scan")
    print("  midrib segments:", len(midrib["segments"]), "gaps:", len(midrib["occluded_gaps"]))
    print("  decisions:", decisions)
    print("  follow stations:", f"{len(measured)}/{scan['requested_stations']}",
          "gap_station:", gapped["gap_station"])


if __name__ == "__main__":
    main()
