#!/usr/bin/env python3
"""Standalone verification for the fine-band row engine.

Run with -I (isolated). The repo root and the interpreter site-packages are
added explicitly so no working-directory or third-party code is imported
implicitly. Synthetic texture images are generated in memory; no untrusted
files are read.
"""
import sys

sys.path.insert(0, "/workspace")
for path in ("/opt/pointwise-env/site-packages", "/home/node/.local/lib/python3.11/site-packages"):
    if path not in sys.path:
        sys.path.append(path)

import numpy as np

from folioleaf.bandrow import (
    HARMONIC_FREQUENCY_RATIO,
    anchor_families,
    evaluate_row,
    mark_harmonics,
    match_families,
    select_peaks,
)
from folioleaf.geometry import SampleFrame
from folioleaf.weave import axial_difference


def frame_with_image(image, scale=None):
    height, width = image.shape[:2]
    valid = np.ones((height, width), dtype=bool)
    return SampleFrame(image, valid, {"id": "synthetic", "name": "synthetic", "role": "diffuse",
                                      "working_digest": "x" * 64, "source_digest": "y" * 64,
                                      "original_size": [width, height], "width": width, "height": height,
                                      "working_color_space": "sRGB", "synthetic": True},
                       (), scale, np.zeros((height, width), dtype=bool))


def band_signal(width, height, angle_deg, pitch_px, amplitude, phase=0.0):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    theta = np.radians(angle_deg)
    # Bands perpendicular to the reported band direction.
    projection = xx * np.cos(theta) + yy * np.sin(theta)
    return amplitude * np.sin(2 * np.pi * projection / pitch_px + phase)


def make_cell_image(systems, size=(140, 140), noise=0.012, seed=0):
    rng = np.random.default_rng(seed)
    signal = np.zeros(size, dtype=np.float32)
    for angle, pitch, amplitude in systems:
        signal += band_signal(size[1], size[0], angle, pitch, amplitude)
    signal += rng.normal(0, noise, size).astype(np.float32)
    gray = np.clip(0.5 + signal, 0, 1).astype(np.float32)
    return np.repeat(gray[:, :, None], 3, axis=2)


def region(rid, index):
    left = 0.02 + index * 0.19
    return {"id": rid, "name": f"Cell {index + 1}", "purpose": "inspection",
            "vertices": [[left, 0.08], [min(left + 0.16, 0.96), 0.08],
                         [min(left + 0.16, 0.96), 0.92], [left, 0.92]]}


def tile_with(regions):
    return {"regions": regions, "id": "tile" * 4, "version": 1}


SETTINGS = {"reference_axis": 10.0, "tolerance": 24.0, "minimum_pitch": 5.0,
            "maximum_pitch": 32.0, "turn_limit": 38.0}


def test_harmonic_flagged_never_fundamental():
    image = make_cell_image([(20.0, 14.0, 0.10), (22.0, 7.0, 0.06)], seed=1)
    frame = frame_with_image(image)
    domain = np.ones(image.shape[:2], dtype=bool)
    from folioleaf.bandrow import cell_spectrum
    data = cell_spectrum(frame, image, domain, SETTINGS)
    peaks = select_peaks(data)
    fundamentals = [p for p in peaks if p["harmonic_of"] is None]
    harmonics = [p for p in peaks if p["harmonic_of"] is not None]
    assert fundamentals, "expected at least one fundamental"
    assert harmonics, "expected the double-frequency peak to be flagged as a harmonic"
    for harmonic in harmonics:
        parent = next(p for p in peaks if p["peak"] == harmonic["harmonic_of"])
        ratio = harmonic["pitch"] / parent["pitch"]
        assert HARMONIC_FREQUENCY_RATIO[0] <= ratio <= HARMONIC_FREQUENCY_RATIO[1]
    print("ok: harmonic peak flagged and excluded from fundamentals")


def test_peak_exclusivity():
    peaks = [
        {"peak": 1, "angle_degrees": 12.0, "pitch": 13.0, "peak_to_background": 20.0, "harmonic_of": None},
        {"peak": 2, "angle_degrees": 102.0, "pitch": 9.0, "peak_to_background": 18.0, "harmonic_of": None},
    ]
    rows, conflicts = match_families(peaks, {"A": 10.0, "B": 100.0}, 38.0)
    used = [row["peak"] for row in rows.values() if row["peak"] is not None]
    assert len(used) == len(set(used)), "a peak was assigned to two families"
    assert rows["A"]["peak"] == 1 and rows["B"]["peak"] == 2
    print("ok: distinct peaks assigned once each")


def test_blank_when_ambiguous():
    # Two fundamentals at nearly the same angular distance to family A: tie rule blanks A.
    peaks = [
        {"peak": 1, "angle_degrees": 20.0, "pitch": 12.0, "peak_to_background": 20.0, "harmonic_of": None},
        {"peak": 2, "angle_degrees": 23.0, "pitch": 11.0, "peak_to_background": 19.0, "harmonic_of": None},
        {"peak": 3, "angle_degrees": 110.0, "pitch": 8.0, "peak_to_background": 17.0, "harmonic_of": None},
    ]
    rows, conflicts = match_families(peaks, {"A": 21.5, "B": 110.0}, 38.0)
    assert rows["A"]["angle_degrees"] is None
    assert "competing_peaks" in rows["A"]["conflicts"]
    assert rows["B"]["peak"] == 3
    print("ok: ambiguous family left blank instead of guessed")


def test_sharp_turn_preserved():
    # A 55-degree jump exceeds the 38-degree step limit and must be flagged, not bridged.
    peaks = [
        {"peak": 1, "angle_degrees": 70.0, "pitch": 13.0, "peak_to_background": 22.0, "harmonic_of": None},
    ]
    rows, _ = match_families(peaks, {"A": 15.0}, 38.0)
    assert rows["A"]["status"] == "sharp_turn"
    assert rows["A"]["sharp_turn"] is True
    assert rows["A"]["angle_degrees"] == 70.0
    print("ok: genuine sharp turn kept and flagged")


def test_full_row_scenario():
    # Six cells: two clear anchors, approach, strong+half-pitch harmonic, separation, edge turn.
    scenarios = [
        [(10.0, 14.0, 0.10), (100.0, 9.0, 0.09)],
        [(12.0, 14.0, 0.10), (98.0, 9.0, 0.09)],
        [(22.0, 13.5, 0.09), (82.0, 9.0, 0.08)],
        [(26.0, 13.0, 0.11), (78.0, 9.2, 0.07), (27.0, 6.5, 0.05)],
        [(34.0, 13.0, 0.08), (70.0, 9.0, 0.07)],
        [(66.0, 12.5, 0.09), (52.0, 8.8, 0.06)],
    ]
    regions = [region(f"{i:024d}", i) for i in range(6)]
    tile = tile_with(regions)
    width = height = 520
    canvas = np.full((height, width, 3), 0.5, dtype=np.float32)
    frame = frame_with_image(canvas)
    # Evaluate by replacing the masked window per cell through monkeypatched cell builder.
    from folioleaf import bandrow

    original = bandrow.cell_spectrum
    images = [make_cell_image(systems, seed=10 + i) for i, systems in enumerate(scenarios)]

    def fake_cell_spectrum(frame_arg, linear_arg, domain, settings):
        index = getattr(fake_cell_spectrum, "index", 0)
        result = original(frame_with_image(images[index]), images[index],
                          np.ones(images[index].shape[:2], dtype=bool), settings)
        fake_cell_spectrum.index = index + 1
        return result

    bandrow.cell_spectrum = fake_cell_spectrum
    try:
        cells = evaluate_row(frame, canvas, tile, SETTINGS, regions)
    finally:
        bandrow.cell_spectrum = original

    assert len(cells) == 6
    # Anchors: both families present in cells 1 and 2.
    first = {row["family"]: row for row in cells[0]["families"]}
    assert first["A"]["status"] == "anchored" and first["B"]["status"] == "anchored"
    # No peak shared across families anywhere.
    for cell in cells:
        used = [row["peak"] for row in cell["families"] if row["peak"] is not None]
        assert len(used) == len(set(used))
    # Cell 4: the half-pitch peak is a harmonic, never the second family.
    fourth = cells[3]
    harmonic_peaks = [peak for peak in fourth["peaks"] if peak["harmonic_of"] is not None]
    assert harmonic_peaks, "the half-spacing secondary peak should be marked harmonic"
    family_peaks = {row["peak"] for row in fourth["families"] if row["peak"] is not None}
    assert not family_peaks.intersection(h["peak"] for h in harmonic_peaks)
    # Last cell: family A turns sharply (~34 -> 66) and is flagged.
    last = {row["family"]: row for row in cells[-1]["families"]}
    assert last["A"]["status"] == "sharp_turn"
    print("ok: full row tracks two families, isolates harmonics and preserves the edge turn")


def test_shared_peak_never_double_counted():
    # Family A can reach both peaks; family B can only reach peak 1. Peak 1 must
    # be assigned once, and B must not reuse it.
    peaks = [
        {"peak": 1, "angle_degrees": 40.0, "pitch": 12.0, "peak_to_background": 24.0, "harmonic_of": None},
        {"peak": 2, "angle_degrees": 20.0, "pitch": 13.0, "peak_to_background": 20.0, "harmonic_of": None},
    ]
    rows, conflicts = match_families(peaks, {"A": 25.0, "B": 55.0}, 38.0)
    codes = [item["code"] for item in conflicts]
    assert "shared_peak" in codes
    used = [row["peak"] for row in rows.values() if row["peak"] is not None]
    assert len(used) == len(set(used))
    assert 1 in used and 2 in used
    print("ok: shared peak assigned once; no family reuses another family's peak")


def test_anchor_and_blank_chain():
    # First cell has one family only; it must anchor A and leave B blank.
    peaks = [{"peak": 1, "angle_degrees": 12.0, "pitch": 14.0,
              "peak_to_background": 21.0, "harmonic_of": None}]
    state, rows = anchor_families(peaks, 10.0)
    assert rows["A"]["peak"] == 1
    assert rows["B"]["angle_degrees"] is None
    assert "anchor_missing" in rows["B"]["conflicts"]
    print("ok: missing anchor family stays blank")


if __name__ == "__main__":
    test_harmonic_flagged_never_fundamental()
    test_peak_exclusivity()
    test_shared_peak_never_double_counted()
    test_blank_when_ambiguous()
    test_sharp_turn_preserved()
    test_anchor_and_blank_chain()
    test_full_row_scenario()
    print("all band row engine checks passed")
