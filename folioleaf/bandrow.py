"""Row-of-cells fine band analysis for two tracked texture families.

A pressed-leaf close-up can carry two visible fine-band systems at once:
the abaxial venation of the blade and the gauze weave pressed over it.  A
single measurement window only reports two orthogonal search sectors, so a
row of slightly overlapping windows is used here.

The engine keeps the two families apart with three hard rules requested by
the inspection workflow:

* one spectral peak can be assigned to one family only (peak exclusivity);
* a harmonic (a near double-frequency, same-direction peak) stays flagged as
  a harmonic and is never counted as the other family's fundamental;
* when the evidence is ambiguous the missing family is left blank rather
  than filled in for the sake of continuity.

Genuine sharp turns near the blade edge are preserved: continuity only
guides allocation inside a bounded angular step, and a larger observed turn
is reported as a sharp turn instead of being smoothed away.
"""

import math

import numpy as np
from PIL import ImageDraw

from .art import base_rgba, font
from .defects import gaussian_supported
from .geometry import active_region, region_domain
from .model import InputError, collection, dictionary, enumeration, identity, integer, scalar, string
from .photos import luminance
from .weave import axial_difference, spectrum

FAMILY_COLORS = {"A": (103, 231, 200), "B": (248, 207, 108)}
FAMILIES = ("A", "B")
MAX_CELLS = 12
MIN_CELLS = 3
MIN_SUPPORTED_PIXELS = 400
MIN_SIDE = 32
MIN_SUPPORT_FRACTION = 0.65
MIN_TRUSTED_FRACTION = 0.55
PEAK_RATIO_FLOOR = 7.0
PEAK_ENERGY_FRACTION = 0.012
MAX_PEAKS_PER_CELL = 6
HARMONIC_FREQUENCY_RATIO = (1.75, 2.30)
HARMONIC_ANGLE_LIMIT = 12.0
TIE_ANGLE_DELTA = 4.0
DEFAULT_TURN_LIMIT = 38.0
MIN_TURN_LIMIT = 20.0
MAX_TURN_LIMIT = 70.0

CONFLICT_POLICY = [
    {
        "code": "shared_peak",
        "rule": "A single spectral peak is never counted for both families. When two families "
                "both reach the same peak, the better-continued family takes it and the other "
                "family is left blank with a shared_peak conflict.",
    },
    {
        "code": "harmonic_fundamental",
        "rule": "A near double-frequency peak on the same direction is labelled a harmonic of the "
                "lower-frequency fundamental. Harmonics are never promoted to the other family.",
    },
    {
        "code": "competing_peaks",
        "rule": "When two peaks of comparable strength both continue one family within the angular "
                "step and neither is clearly closer, the family is left blank for that cell rather "
                "than guessed.",
    },
    {
        "code": "sharp_turn",
        "rule": "A direction jump larger than the angular step limit is treated as a real turn. It "
                "is never bridged or smoothed; the family restarts from the local peak and the turn "
                "is flagged.",
    },
    {
        "code": "no_local_peak",
        "rule": "If no local peak clears the peak-to-background threshold, the family is left blank. "
                "A blank never invents a reading from neighbouring cells.",
    },
    {
        "code": "anchor_missing",
        "rule": "The first two cells anchor the two families from their strongest distinct "
                "fundamentals. A family that cannot be anchored there starts blank and is picked up "
                "only when a clear local fundamental appears later.",
    },
]


def transect_options(payload):
    data = dictionary(payload, "Band row analysis")
    region_ids = collection(data.get("region_ids"), "Band row cells", minimum=MIN_CELLS, maximum=MAX_CELLS)
    options = {
        "region_ids": [identity(item, "Sampling region") for item in region_ids],
        "photo_id": identity(data["photo_id"], "Photo ID") if data.get("photo_id") else None,
        "reference_axis": scalar(data.get("reference_axis", 0.0), "Approximate first band direction",
                                 lower=0.0, upper=179.0),
        "tolerance": scalar(data.get("tolerance", 24.0), "Search half-angle", lower=6.0, upper=35.0),
        "minimum_pitch": scalar(data.get("minimum_pitch", 5.0), "Minimum spacing", lower=3.0, upper=25.0),
        "maximum_pitch": scalar(data.get("maximum_pitch", 32.0), "Maximum spacing", lower=8.0, upper=52.0),
        "turn_limit": scalar(data.get("turn_limit", DEFAULT_TURN_LIMIT), "Angular step limit",
                             lower=MIN_TURN_LIMIT, upper=MAX_TURN_LIMIT),
    }
    if options["minimum_pitch"] >= options["maximum_pitch"]:
        raise InputError("Maximum pitch must exceed minimum pitch.")
    if len(set(options["region_ids"])) != len(options["region_ids"]):
        raise InputError("Each band row cell can be used once; remove repeated regions.")
    return options


def reassign_options(payload):
    data = dictionary(payload, "Band row recheck")
    return {
        "version": integer(data.get("version"), "Version", lower=1, upper=2_000_000_000),
        "cell": integer(data.get("cell"), "Cell number", lower=1, upper=MAX_CELLS),
        "family": enumeration(data.get("family"), FAMILIES, "Confirmed family"),
        "peak": integer(data.get("peak"), "Confirmed peak number", lower=1, upper=MAX_PEAKS_PER_CELL),
        "note": string(data.get("note", ""), "Recheck note", maximum=240, required=False),
    }


def cell_spectrum(frame, linear, domain, settings):
    visible = domain & frame.valid & ~frame.clipped
    rows, cols = np.nonzero(visible)
    if len(rows) < MIN_SUPPORTED_PIXELS:
        return None
    ys = slice(int(rows.min()), int(rows.max()) + 1)
    xs = slice(int(cols.min()), int(cols.max()) + 1)
    support = visible[ys, xs]
    height, width = support.shape
    if min(height, width) < MIN_SIDE or support.mean() < MIN_SUPPORT_FRACTION:
        return None
    scale = tuple(frame.scale or (1.0, 1.0))
    min_px, max_px = settings["minimum_pitch"], settings["maximum_pitch"]
    lower, upper = min_px * min(scale), max_px * max(scale)
    gray = luminance(linear[ys, xs])
    background, weight = gaussian_supported(gray, support, max(3.0, max_px / 2))
    trusted = support & (weight > 0.63)
    if trusted.mean() < MIN_TRUSTED_FRACTION:
        return None
    power, frequencies, angles = spectrum(gray - background, trusted, scale)
    return {
        "power": power,
        "frequencies": frequencies,
        "angles": angles,
        "period_range": (lower, upper),
        "trusted": trusted,
        "window": (ys, xs),
        "supported_pixels": int(np.count_nonzero(trusted)),
    }


def select_peaks(spectrum_data):
    power = spectrum_data["power"]
    frequencies = spectrum_data["frequencies"]
    angles = spectrum_data["angles"]
    lower, upper = spectrum_data["period_range"]
    in_range = (frequencies >= 1 / upper) & (frequencies <= 1 / lower)
    readings = power[in_range]
    if readings.size < 8:
        return []
    noise = float(np.median(readings))
    energy_floor = max(float(readings.sum()) * PEAK_ENERGY_FRACTION, 1e-9)
    threshold = max(noise * PEAK_RATIO_FLOOR, energy_floor)
    candidates = in_range & (power >= threshold)
    # Collapse adjacent FFT bins so one broad peak is one selectable peak.
    chosen = []
    while np.any(candidates):
        index = np.unravel_index(np.argmax(np.where(candidates, power, -np.inf)), power.shape)
        chosen.append(index)
        row, col = index
        candidates[max(0, row - 1):row + 2, max(0, col - 1):col + 2] = False
        if len(chosen) >= MAX_PEAKS_PER_CELL:
            break
    peaks = []
    for number, index in enumerate(chosen, start=1):
        ratio = float(power[index]) / max(noise, 1e-9)
        peaks.append({
            "peak": number,
            "index": index,
            "angle_degrees": round(float(angles[index]), 2),
            "pitch": round(1 / float(frequencies[index]), 4),
            "peak_to_background": round(ratio, 2),
            "harmonic_of": None,
        })
    mark_harmonics(peaks)
    # Peaks stay power-ranked; only fundamentals may anchor a family.
    return peaks


def mark_harmonics(peaks):
    for peak in peaks:
        if peak["harmonic_of"] is not None:
            continue
        for other in peaks:
            if other is peak or other["harmonic_of"] is not None:
                continue
            ratio = other["pitch"] and peak["pitch"] / other["pitch"]
            aligned = abs(axial_difference(peak["angle_degrees"], other["angle_degrees"])) <= HARMONIC_ANGLE_LIMIT
            if aligned and ratio and HARMONIC_FREQUENCY_RATIO[0] <= ratio <= HARMONIC_FREQUENCY_RATIO[1]:
                peak["harmonic_of"] = other["peak"]
                break


def is_fundamental(peak):
    return peak["harmonic_of"] is None


def blank_family(family, conflicts):
    return {
        "family": family,
        "angle_degrees": None,
        "pitch": None,
        "peak_to_background": None,
        "peak": None,
        "harmonic_of": None,
        "status": "blank",
        "conflicts": list(conflicts),
    }


def filled_family(family, peak, status, conflicts, *, sharp_turn=False):
    return {
        "family": family,
        "angle_degrees": peak["angle_degrees"],
        "pitch": peak["pitch"],
        "peak_to_background": peak["peak_to_background"],
        "peak": peak["peak"],
        "harmonic_of": peak["harmonic_of"],
        "status": status,
        "sharp_turn": sharp_turn,
        "conflicts": list(conflicts),
    }


def anchor_families(peaks, reference):
    """Anchor families A and B on the first two cells from distinct fundamentals."""
    fundamentals = [peak for peak in peaks if is_fundamental(peak)]
    state = {}
    rows = {}
    used = set()
    for family, direction in (("A", reference), ("B", (reference + 90) % 180)):
        available = [peak for peak in fundamentals if peak["peak"] not in used]
        if not available:
            rows[family] = blank_family(family, ["anchor_missing"])
            continue
        best = min(available, key=lambda peak: abs(axial_difference(peak["angle_degrees"], direction)))
        used.add(best["peak"])
        state[family] = best["angle_degrees"]
        rows[family] = filled_family(family, best, "anchored", [])
    return state, rows


def match_families(peaks, state, turn_limit):
    """Assign fundamentals to the tracked families without sharing a peak."""
    fundamentals = [peak for peak in peaks if is_fundamental(peak)]
    edges = []
    for family, direction in state.items():
        for peak in fundamentals:
            delta = abs(axial_difference(peak["angle_degrees"], direction))
            if delta <= turn_limit:
                edges.append((delta, family, peak["peak"]))
    edges.sort(key=lambda edge: (edge[0], edge[2]))

    assigned_peak = {}
    assigned_family = {}
    family_competition = {family: [] for family in state}
    for delta, family, peak_number in edges:
        family_competition[family].append((delta, peak_number))
        if peak_number in assigned_peak or family in assigned_family:
            continue
        assigned_peak[peak_number] = family
        assigned_family[family] = peak_number

    conflicts = []
    # Two comparably close peaks for one family is an ambiguous allocation.
    for family, candidates in family_competition.items():
        if len(candidates) >= 2 and (candidates[1][0] - candidates[0][0]) <= TIE_ANGLE_DELTA:
            conflicts.append({"code": "competing_peaks", "family": family,
                              "peaks": [number for _, number in candidates[:2]]})
            assigned_peak.pop(assigned_family.pop(family, None), None)

    peak_lookup = {peak["peak"]: peak for peak in peaks}
    # Precompute which peaks fall inside more than one family's angular window.
    overlaps = {}
    for peak in fundamentals:
        contenders = [family for family in state
                      if abs(axial_difference(peak["angle_degrees"], state[family])) <= turn_limit]
        if len(contenders) > 1:
            overlaps[peak["peak"]] = contenders
            conflicts.append({"code": "shared_peak", "peak": peak["peak"], "families": contenders})

    rows = {}
    claimed = set(assigned_family.values())
    for family in FAMILIES:
        if family not in state:
            rows[family] = blank_family(family, ["anchor_missing"])
            continue
        peak_number = assigned_family.get(family)
        if peak_number is None:
            local_conflicts = [item["code"] for item in conflicts if item.get("family") == family]
            lost_shared = [
                "shared_peak"
                for overlap_peak, contenders in overlaps.items()
                if family in contenders and overlap_peak in assigned_peak
            ]
            local_conflicts.extend(lost_shared)
            # A visible fundamental outside the angular step is a genuine sharp turn.
            turned = next((peak for peak in fundamentals
                           if peak["peak"] not in claimed
                           and abs(axial_difference(peak["angle_degrees"], state[family])) > turn_limit), None)
            if turned is not None:
                claimed.add(turned["peak"])
                rows[family] = filled_family(family, turned, "sharp_turn",
                                             ["sharp_turn", *local_conflicts], sharp_turn=True)
                state[family] = turned["angle_degrees"]
            else:
                if not local_conflicts:
                    local_conflicts = ["no_local_peak"]
                rows[family] = blank_family(family, local_conflicts)
            continue
        peak = peak_lookup[peak_number]
        delta = abs(axial_difference(peak["angle_degrees"], state[family]))
        sharp = delta > turn_limit * 0.8
        rows[family] = filled_family(family, peak, "tracked", [], sharp_turn=sharp)
        state[family] = peak["angle_degrees"]

    return rows, conflicts


def public_peaks(peaks):
    return [{key: peak[key] for key in ("peak", "angle_degrees", "pitch", "peak_to_background", "harmonic_of")}
            for peak in peaks]


def anchor_missing(peaks, state, rows, reference):
    """Anchor a family that was not resolved in the first cell."""
    used = {row["peak"] for row in rows.values() if row["peak"] is not None}
    for family, direction in (("A", reference), ("B", (reference + 90) % 180)):
        if family in state:
            continue
        available = [peak for peak in peaks if is_fundamental(peak) and peak["peak"] not in used]
        if not available:
            rows[family] = blank_family(family, ["anchor_missing"])
            continue
        best = min(available, key=lambda peak: abs(axial_difference(peak["angle_degrees"], direction)))
        used.add(best["peak"])
        state[family] = best["angle_degrees"]
        rows[family] = filled_family(family, best, "anchored", [])
    return state, rows


def evaluate_row(frame, linear, tile, settings, regions, *, forced=None):
    reference = settings["reference_axis"]
    state = {}
    cells = []
    for index, region in enumerate(regions, start=1):
        domain = region_domain(frame.extent, tile, region)
        spectrum_data = cell_spectrum(frame, linear, domain, settings)
        cell_conflicts = []
        if spectrum_data is None:
            peaks = []
            rows = {family: blank_family(family, ["no_local_peak"]) for family in FAMILIES}
            supported = 0
        else:
            supported = spectrum_data["supported_pixels"]
            peaks = select_peaks(spectrum_data)
            if forced is not None and index == forced["cell"]:
                rows, state = forced_allocation(peaks, state, forced)
            elif index == 1:
                state, rows = anchor_families(peaks, reference)
            elif index == 2 and len(state) < 2:
                rows, cell_conflicts = match_families(peaks, dict(state), settings["turn_limit"])
                state, rows = anchor_missing(peaks, state, rows, reference)
            else:
                rows, cell_conflicts = match_families(peaks, dict(state), settings["turn_limit"])
        cell = {
            "cell": index,
            "region_id": region["id"],
            "region_name": region["name"],
            "supported_pixels": supported,
            "peaks": public_peaks(peaks),
            "families": [rows[family] for family in FAMILIES],
            "conflicts": cell_conflicts,
        }
        cells.append(cell)
    return cells


def forced_allocation(peaks, state, forced):
    peak = next((item for item in peaks if item["peak"] == forced["peak"]), None)
    if peak is None:
        raise InputError("The confirmed peak is no longer visible in that cell. Recheck the current image.")
    rows = {}
    used = {forced["peak"]}
    rows[forced["family"]] = filled_family(forced["family"], peak, "confirmed", [])
    state[forced["family"]] = peak["angle_degrees"]
    other = "B" if forced["family"] == "A" else "A"
    remaining = [item for item in peaks if is_fundamental(item) and item["peak"] not in used]
    if other in state and remaining:
        best = min(remaining, key=lambda item: abs(axial_difference(item["angle_degrees"], state[other])))
        rows[other] = filled_family(other, best, "tracked", [])
        state[other] = best["angle_degrees"]
    elif other in state:
        rows[other] = blank_family(other, ["no_local_peak"])
    else:
        rows[other] = blank_family(other, ["anchor_missing"])
    return rows, state


def row_overlay(frame, regions, cells):
    image = base_rgba(frame)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    sx, sy = frame.scale or (1.0, 1.0)
    label_font = font(max(11, min(18, round(min(width, height) / 42))))
    for cell, region in zip(cells, regions):
        points = [(round(x * width), round(y * height)) for x, y in region["vertices"]]
        draw.line([*points, points[0]], fill=(236, 241, 232), width=2, joint="curve")
        center_x = round(sum(point[0] for point in points) / len(points))
        center_y = round(sum(point[1] for point in points) / len(points))
        span = min(width, height) * 0.055
        draw.text((center_x - 8, center_y - 22), str(cell["cell"]), fill=(236, 241, 232),
                  font=label_font, stroke_width=2, stroke_fill=(30, 38, 43))
        for row in cell["families"]:
            if row["angle_degrees"] is None:
                continue
            color = FAMILY_COLORS[row["family"]]
            theta = math.radians(row["angle_degrees"])
            delta = (math.cos(theta) * span / sx, math.sin(theta) * span / sy)
            first = (center_x - delta[0], center_y - delta[1])
            last = (center_x + delta[0], center_y + delta[1])
            draw.line((first, last), fill=(16, 31, 29), width=5)
            draw.line((first, last), fill=color, width=2)
    return image


def family_track(cells):
    tracks = []
    for family in FAMILIES:
        angles, pitches, blanks, turns = [], [], 0, 0
        for cell in cells:
            row = next(item for item in cell["families"] if item["family"] == family)
            angles.append(row["angle_degrees"])
            pitches.append(row["pitch"])
            if row["angle_degrees"] is None:
                blanks += 1
            if row.get("sharp_turn"):
                turns += 1
        tracks.append({"family": family, "angles": angles, "pitches": pitches,
                       "blank_cells": blanks, "sharp_turns": turns})
    return tracks


def conflict_summary(cells):
    codes = {}
    for cell in cells:
        entries = list(cell["conflicts"])
        entries.extend(code for row in cell["families"] for code in row["conflicts"])
        for conflict in entries:
            code = conflict["code"] if isinstance(conflict, dict) else conflict
            codes[code] = codes.get(code, 0) + 1
    return [{"code": code, "count": count} for code, count in sorted(codes.items())]


def resolve_regions(tile, region_ids):
    regions = []
    for region_id in region_ids:
        regions.append(active_region(tile, region_id))
    return regions


def build_row_metrics(frame, linear, tile, settings, regions, *, forced=None, revision=None):
    cells = evaluate_row(frame, linear, tile, settings, regions, forced=forced)
    unit = "mm" if frame.scale else "px"
    metrics = {
        "unit": unit,
        "reference_axis": settings["reference_axis"],
        "turn_limit": settings["turn_limit"],
        "pitch_range": [settings["minimum_pitch"], settings["maximum_pitch"]],
        "family_labels": {"A": "Family A (first anchored system)", "B": "Family B (second anchored system)"},
        "cells": cells,
        "tracks": family_track(cells),
        "conflict_summary": conflict_summary(cells),
        "conflict_policy": CONFLICT_POLICY,
        "reassignment": forced,
        "revision_note": "Re-evaluation after an on-image recheck; the previous reading remains stored unchanged."
        if revision else "Initial row reading. Blanks are unresolved families, not zero spacing.",
        "note": "Two fine-band families are tracked across the overlapping cells. Each peak belongs to one "
                "family, harmonics stay flagged, ambiguous families are left blank, and sharp turns are "
                "reported rather than smoothed out.",
    }
    image = row_overlay(frame, regions, cells)
    return metrics, image
