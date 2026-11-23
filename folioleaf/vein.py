"""Vein branch maps and follow-vein scans.

A branch map is built on top of the fine-line network centerlines.  The user
marks trusted anchor points (a midrib stretch, two lateral vein tips, two
candidate junctions near the mounting strip).  Paths between anchors are
traced only through pixels that are actually part of the observed network, so
a vein hidden under a mounting strip is reported as an open gap rather than
joined by a straight interpolation.  Of the two candidate junctions exactly
one is adopted; the rejected junction and the trace it would have needed are
kept in the record so the trade-off stays visible on the overlay.
"""

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import INK, SHADOW, base_rgba
from .color import linear_lab
from .craze import (
    contrast_suppress,
    gradient_field,
    retained_regions,
    threshold_centerline,
)
from .geometry import measure_line_samples
from .model import InputError, collection, numeric_pair
from .photos import luminance


# Distances in normalized units; the fork region is deliberately small.
ANCHOR_SNAP = 0.035
MAX_TRACE_PX = 9000
NEIGHBOR_STEPS = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)
MIDRIB_COLOR = (134, 220, 255)
LATERAL_COLORS = ((126, 231, 158), (196, 205, 116))
GAP_COLOR = (237, 122, 110)
REJECTED_COLOR = (232, 158, 96)
ANCHOR_COLOR = (255, 245, 225)


def point_list(value, label, *, minimum, maximum):
    raw = collection(value, label, minimum=minimum, maximum=maximum)
    return [numeric_pair(item, label, lower=0, upper=1) for item in raw]


def pixel_anchor(point, extent, walkable):
    """Nearest observed line pixel to a normalized anchor."""
    width, height = extent
    target = np.array([point[0] * width, point[1] * height], dtype=np.float64)
    rows, cols = np.nonzero(walkable)
    if not len(rows):
        raise InputError("The fine-line network contains no traceable lines here.")
    xy = np.stack((cols.astype(np.float64), rows.astype(np.float64)), axis=1)
    distances = np.linalg.norm(xy - target, axis=1)
    nearest = int(np.argmin(distances))
    limit = ANCHOR_SNAP * math.hypot(width, height)
    if distances[nearest] > limit:
        raise InputError(
            f"An anchor is more than {ANCHOR_SNAP:.3f} of the image diagonal "
            "from any traced fine line. Mark it directly on a visible line."
        )
    return int(rows[nearest]), int(cols[nearest]), round(float(distances[nearest]), 3)


def ridge_walkable(source_luma, support, controls):
    """Observed dark-line pixels that a vein trace is allowed to follow.

    The fine-line skeleton can lose a few pixels at line junctions because of
    its thinning response.  Tracing is therefore allowed on the broader dark
    ridge as well, but only inside the same supported region: mounting strips
    and other excluded areas stay non-walkable, so gaps remain real gaps.
    """
    window = max(7, int(round(controls["radius"] * 9)))
    baseline = ndimage.median_filter(np.where(support, source_luma, 0.0), size=window, mode="nearest")
    contrast = np.where(support, baseline - source_luma, 0.0)
    ridge_threshold = max(0.025, float(np.percentile(contrast[support], 70)))
    dark = support & (contrast >= ridge_threshold)
    dark = retained_regions(dark, max(4, controls["component_minimum"] // 2))
    return dark


def weighted_trace(walkable, preferred, start, goal):
    """Cheapest 8-connected trace on observed line pixels.

    ``preferred`` (the fine-line centerline) carries a low cost; other dark
    ridge pixels cost more so the trace stays on the thinned vein wherever it
    exists and only leaves it briefly at junction artifacts.  No pixel outside
    ``walkable`` is ever entered.
    """
    import heapq

    if start == goal:
        return [start], 0.0
    height, width = walkable.shape
    best = {start: 0.0}
    previous = {}
    heap = [(0.0, start)]
    settled = set()
    while heap:
        cost, current = heapq.heappop(heap)
        if current in settled:
            continue
        settled.add(current)
        if current == goal:
            break
        row, col = current
        for dy, dx in NEIGHBOR_STEPS:
            neighbor = (row + dy, col + dx)
            if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                continue
            if not walkable[neighbor]:
                continue
            step = math.sqrt(2) if dy and dx else 1.0
            # Centerline pixels are strongly preferred; ridge-only pixels
            # cost several times more, proportional to step length.
            pixel_cost = 1.0 if preferred[neighbor] else 6.0
            candidate = cost + step * pixel_cost
            if candidate < best.get(neighbor, float("inf")):
                best[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(heap, (candidate, neighbor))
    if goal not in best:
        return None, None
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    if len(path) > MAX_TRACE_PX:
        raise InputError("The traced vein segment exceeds the supported length.")
    return path, best[goal]


def component_from(walkable, start):
    """All walkable pixels in the 8-connected component containing ``start``."""
    height, width = walkable.shape
    found = np.zeros((height, width), dtype=bool)
    queue = [start]
    found[start] = True
    head = 0
    while head < len(queue):
        row, col = queue[head]
        head += 1
        for dy, dx in NEIGHBOR_STEPS:
            neighbor = (row + dy, col + dx)
            if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                continue
            if found[neighbor] or not walkable[neighbor]:
                continue
            found[neighbor] = True
            queue.append(neighbor)
    return found


def tip_stump(walkable, centerline, extent, tip_point):
    """Longest centerline-following stump reachable from a vein tip."""
    row, col, _ = pixel_anchor(tip_point, extent, walkable)
    start = (row, col)
    component = component_from(walkable, start)
    rows, cols = np.nonzero(component & centerline)
    if not len(rows):
        rows, cols = np.nonzero(component)
    target_xy = np.stack((cols.astype(np.float64), rows.astype(np.float64)), axis=1)
    origin = np.array([start[1], start[0]], dtype=np.float64)
    far = int(np.argmax(np.linalg.norm(target_xy - origin, axis=1)))
    goal = (int(rows[far]), int(cols[far]))
    path, cost = weighted_trace(walkable, centerline, start, goal)
    if path is None:
        return None
    metrics = path_metrics(path, extent, None)
    metrics["trace_cost"] = round(float(cost), 3)
    return metrics


def path_metrics(path, extent, scale):
    """Cumulative physical/pixel length of a traced pixel path."""
    coords = np.asarray(path, dtype=np.float64)
    steps = np.diff(coords, axis=0)
    px_lengths = np.hypot(steps[:, 0], steps[:, 1])
    total_px = float(px_lengths.sum())
    if scale:
        physical = np.hypot(steps[:, 0] * scale[1], steps[:, 1] * scale[0])
        total = float(physical.sum())
        unit = "mm"
    else:
        total = total_px
        unit = "px"
    width, height = extent
    normalized = [[round((col + 0.5) / width, 6), round((row + 0.5) / height, 6)] for row, col in path]
    return {
        "pixel_count": len(path),
        "trace_length": round(total, 5),
        "trace_length_px": round(total_px, 4),
        "length_unit": unit,
        "points": normalized,
    }


def network_centerlines(frame, linear, area, controls):
    """Re-run the fine-line network pipeline to obtain centerlines + area."""
    source = luminance(linear)
    candidate = area & frame.valid & ~frame.clipped
    if source.size > 3_100_000:
        raise InputError("Vein branch maps support registered planes up to 3.1 megapixels.")
    filtered, valid = gradient_field(source, candidate, controls["radius"])
    fine, support = contrast_suppress(source, candidate, controls["radius"])
    coarse, margin = contrast_suppress(source, candidate, 2 * controls["radius"])
    support &= valid & margin
    if np.count_nonzero(support) < 100:
        raise InputError("Choose a larger area with contiguous, unclipped image support.")
    thin, binary, thresholds = threshold_centerline(filtered, fine, coarse, support, controls)
    return thin, binary, thresholds


def trace_between(walkable, centerline, extent, start_point, end_point):
    starts = {}
    start = goal = None
    for label, point in (("start", start_point), ("end", end_point)):
        row, col, distance = pixel_anchor(point, extent, walkable)
        starts[label] = {"point": [round(point[0], 6), round(point[1], 6)],
                         "snapped": [round((col + 0.5) / extent[0], 6),
                                     round((row + 0.5) / extent[1], 6)],
                         "snap_distance_px": distance}
        if label == "start":
            start = (row, col)
        else:
            goal = (row, col)
    path, cost = weighted_trace(walkable, centerline, start, goal)
    return starts, path, cost


def candidate_trace(walkable, centerline, extent, tip_point, junction_points):
    """Trace one lateral tip to each candidate junction; record both outcomes."""
    options = []
    tip = None
    for index, junction in enumerate(junction_points):
        anchors, path, cost = trace_between(walkable, centerline, extent, tip_point, junction)
        if index == 0:
            tip = anchors["start"]
        option = {
            "junction_index": index,
            "junction_anchor": anchors["end"],
            "connected": path is not None,
        }
        if path is not None:
            option["trace"] = path_metrics(path, extent, None)
            option["trace"]["trace_cost"] = round(float(cost), 3)
        options.append(option)
    return tip, options


def select_junctions(options, preferred):
    """Decide which candidate junction each lateral vein adopts once.

    The user's preferred junction is used whenever it has an observed trace.
    If the preferred route is broken by the mounting strip, the alternative
    is adopted when available.  When both are broken the vein keeps its
    visible stump and the junction is left open.  The two lateral veins never
    share a junction choice unless the observed network forces both to the
    same surviving candidate.
    """
    chosen = []
    for lateral in options:
        picks = lateral["options"]
        preferred_option = next(item for item in picks if item["junction_index"] == preferred)
        alternative = next(item for item in picks if item["junction_index"] != preferred)
        if preferred_option["connected"]:
            adopted, reason = preferred, "preferred junction has a continuous observed line"
        elif alternative["connected"]:
            adopted, reason = alternative["junction_index"], "preferred junction is occluded; alternative observed line used"
        else:
            adopted, reason = None, "both candidate junctions are occluded; vein ends at an open gap"
        chosen.append({"lateral": lateral["lateral"], "adopted_junction": adopted, "reason": reason})
    return chosen


def midrib_shared_stretch(walkable, centerline, extent, midrib_points):
    anchors = []
    pieces = []
    for start_point, end_point in zip(midrib_points, midrib_points[1:]):
        ends, path, _ = trace_between(walkable, centerline, extent, start_point, end_point)
        anchors.extend([ends["start"], ends["end"]])
        pieces.append(path)
    # Deduplicate overlapping anchor records while preserving order.
    unique_anchors = []
    seen = set()
    for anchor in anchors:
        key = tuple(anchor["snapped"])
        if key not in seen:
            seen.add(key)
            unique_anchors.append(anchor)
    return unique_anchors, pieces


def build_branch_map(frame, linear, area, options):
    extent = frame.extent
    network, footprint, thresholds = network_centerlines(frame, linear, area, options)
    support = area & frame.valid & ~frame.clipped
    walkable = ridge_walkable(luminance(linear), support, options)
    # The skeleton is always a subset of walkable pixels and remains the
    # preferred corridor; junction pixels that thinning dropped stay walkable
    # through the ridge mask, but excluded (e.g. mounting strip) pixels do not.
    walkable = walkable | network
    midrib_points = options["midrib"]
    tip_points = options["lateral_tips"]
    junction_points = options["junctions"]
    preferred = options["preferred_junction"]

    midrib_anchors, midrib_pieces = midrib_shared_stretch(walkable, network, extent, midrib_points)
    laterals = []
    for lateral_index, tip in enumerate(tip_points):
        tip_anchor, candidates = candidate_trace(walkable, network, extent, tip, junction_points)
        laterals.append({
            "lateral": lateral_index + 1, "tip_anchor": tip_anchor,
            "options": candidates, "tip_point": tip,
        })
    decisions = select_junctions(laterals, preferred)

    # Shared midrib: one retained trunk, with observed gaps reported.
    midrib_segments = []
    midrib_gaps = []
    for index, path in enumerate(midrib_pieces):
        if path is None:
            midrib_gaps.append({"between_anchors": [index, index + 1], "state": "occluded"})
            continue
        metrics = path_metrics(path, extent, frame.scale)
        metrics["segment"] = index + 1
        midrib_segments.append(metrics)

    retained_lateral = []
    for lateral, decision in zip(laterals, decisions):
        entry = {"lateral": lateral["lateral"], "tip_anchor": lateral["tip_anchor"], **decision}
        adopted = decision["adopted_junction"]
        if adopted is None:
            # No candidate junction is reachable on observed lines. Keep the
            # longest visible stump reachable from the tip so the fine vein
            # is preserved as far as the image supports it, then leave a gap.
            connected = [item for item in lateral["options"] if item["connected"]]
            stump = max((item["trace"] for item in connected), key=lambda trace: trace["pixel_count"], default=None)
            if stump is None:
                stump = tip_stump(walkable, network, extent, lateral["tip_point"])
            entry["stump"] = stump
            entry["open_gap"] = True
        else:
            chosen_option = next(item for item in lateral["options"] if item["junction_index"] == adopted)
            entry["trace"] = chosen_option["trace"]
            entry["junction_anchor"] = chosen_option["junction_anchor"]
            entry["open_gap"] = False
        # Keep the trace the rejected junction would have required, so the
        # overlay can show exactly what the fork decision gives up.
        rejected_options = [item for item in lateral["options"] if item["junction_index"] != adopted]
        entry["rejected"] = [
            {"junction_index": item["junction_index"], "trace": item.get("trace"), "connected": item["connected"]}
            for item in rejected_options
        ]
        retained_lateral.append(entry)

    metrics = {
        "network_thresholds": thresholds,
        "midrib": {
            "anchors": midrib_anchors,
            "segments": midrib_segments,
            "occluded_gaps": midrib_gaps,
            "retained_length": round(sum(item["trace_length"] for item in midrib_segments), 5),
            "length_unit": midrib_segments[0]["length_unit"] if midrib_segments else ("mm" if frame.scale else "px"),
        },
        "laterals": retained_lateral,
        "junction_candidates": [
            {"junction_index": index, "point": [round(p[0], 6), round(p[1], 6)]}
            for index, p in enumerate(junction_points)
        ],
        "preferred_junction": preferred,
        "junction_decisions": decisions,
        "note": (
            "Paths follow observed fine-line centerlines only. Segments hidden by the "
            "mounting strip remain open gaps; no straight interpolation crosses them."
        ),
    }
    overlay = branch_overlay(frame, network, footprint, metrics, options)
    return metrics, overlay, network


def draw_path(draw, path_points, extent, fill, width, gap=False):
    width_px, height_px = extent
    xy = [(x * width_px, y * height_px) for x, y in path_points]
    if gap:
        for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
            draw.line((x1, y1, x2, y2), fill=fill, width=width)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            draw.ellipse((mx - 2, my - 2, mx + 2, my + 2), fill=SHADOW)
        return
    draw.line(xy, fill=SHADOW, width=width + 3, joint="curve")
    draw.line(xy, fill=fill, width=width, joint="curve")


def draw_dashed(draw, path_points, extent, fill, width, dash=9):
    """Dashed polyline used for the traces a rejected junction would need."""
    width_px, height_px = extent
    xy = [(x * width_px, y * height_px) for x, y in path_points]
    remaining = dash
    pen_down = True
    for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
        segment_length = math.hypot(x2 - x1, y2 - y1)
        if segment_length == 0:
            continue
        walked = 0.0
        while walked < segment_length:
            take = min(remaining, segment_length - walked)
            ratio0 = walked / segment_length
            ratio1 = (walked + take) / segment_length
            if pen_down:
                draw.line(
                    (x1 + (x2 - x1) * ratio0, y1 + (y2 - y1) * ratio0,
                     x1 + (x2 - x1) * ratio1, y1 + (y2 - y1) * ratio1),
                    fill=fill, width=width,
                )
            walked += take
            remaining -= take
            if remaining <= 0:
                pen_down = not pen_down
                remaining = dash


def branch_overlay(frame, network, footprint, metrics, options):
    image = base_rgba(frame)
    raw = np.asarray(image, dtype=np.float32).copy()
    footprint_rgb = footprint & (raw[:, :, 3] > 0)
    raw[footprint_rgb, :3] = raw[footprint_rgb, :3] * 0.72 + np.array([0.36, 0.32, 0.26], dtype=np.float32) * 0.28
    image = Image.fromarray(raw.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    stroke = max(2, round(min(width, height) / 300))

    # Faint full-network context.
    enhanced = ndimage.maximum_filter(network, size=2)
    layer = np.array(image)
    context = (layer[enhanced, :3] * 0.55 + np.array([180, 160, 120]) * 0.45).astype(np.uint8)
    layer[enhanced, :3] = context
    image = Image.fromarray(layer)
    draw = ImageDraw.Draw(image)

    for segment in metrics["midrib"]["segments"]:
        draw_path(draw, segment["points"], image.size, MIDRIB_COLOR, stroke + 1)
    for gap in metrics["midrib"]["occluded_gaps"]:
        anchors = metrics["midrib"]["anchors"]
        lo, hi = gap["between_anchors"]
        if lo < len(anchors) and hi < len(anchors):
            draw_path(draw, [anchors[lo]["snapped"], anchors[hi]["snapped"]],
                      image.size, GAP_COLOR, stroke, gap=True)

    for lateral in metrics["laterals"]:
        color = LATERAL_COLORS[(lateral["lateral"] - 1) % len(LATERAL_COLORS)]
        trace = lateral.get("trace") or lateral.get("stump")
        if trace:
            draw_path(draw, trace["points"], image.size, color, stroke)
        # The trace the rejected junction would have used, drawn dashed so
        # the trade-off (including any press-strip-edge line) stays visible.
        for rejected in lateral.get("rejected", []):
            if rejected.get("trace"):
                draw_dashed(draw, rejected["trace"]["points"], image.size, REJECTED_COLOR, stroke + 1, dash=8)
        if lateral.get("open_gap"):
            tip = lateral["tip_anchor"]["snapped"]
            draw.ellipse((tip[0] * width - 5, tip[1] * height - 5,
                          tip[0] * width + 5, tip[1] * height + 5),
                         outline=GAP_COLOR, width=stroke)
        adopted = lateral["adopted_junction"]
        for candidate in metrics["junction_candidates"]:
            idx = candidate["junction_index"]
            x, y = candidate["point"]
            px, py = x * width, y * height
            if idx == adopted:
                draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=color, outline=INK, width=2)
                draw.text((px + 9, py - 8), f"J{idx + 1} used", fill=INK, stroke_width=2, stroke_fill=SHADOW)
            else:
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline=REJECTED_COLOR, width=stroke)
                draw.line((px - 8, py - 8, px + 8, py + 8), fill=REJECTED_COLOR, width=stroke)
                draw.text((px + 9, py - 8), f"J{idx + 1} rejected", fill=REJECTED_COLOR,
                          stroke_width=2, stroke_fill=SHADOW)

    # User anchors.
    for anchor in metrics["midrib"]["anchors"]:
        x, y = anchor["snapped"]
        draw.ellipse((x * width - 4, y * height - 4, x * width + 4, y * height + 4),
                     fill=ANCHOR_COLOR, outline=SHADOW, width=1)
    for lateral in metrics["laterals"]:
        x, y = lateral["tip_anchor"]["snapped"]
        draw.ellipse((x * width - 4, y * height - 4, x * width + 4, y * height + 4),
                     fill=ANCHOR_COLOR, outline=SHADOW, width=1)

    legend_y = 10
    for label, color in (
        ("Shared midrib (one trunk)", MIDRIB_COLOR),
        ("Lateral veins (one junction each)", LATERAL_COLORS[0]),
        ("Open gap under mounting strip", GAP_COLOR),
        ("Rejected junction / trace", REJECTED_COLOR),
        ("Fine-line network context", (180, 160, 120)),
    ):
        draw.rectangle((10, legend_y, 24, legend_y + 12), fill=color, outline=SHADOW)
        draw.text((30, legend_y - 1), label, fill=INK, stroke_width=2, stroke_fill=SHADOW)
        legend_y += 18
    return image


def curve_grid(extent, points, count, band_width):
    """Sample stations along a polyline with a transverse sampling band."""
    width, height = extent
    xy = np.asarray(points, dtype=np.float64) * [width, height] - 0.5
    deltas = np.diff(xy, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    if np.any(seg_lengths < 0.5):
        raise InputError("Follow-vein anchor points need at least one pixel between them.")
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total = cumulative[-1]
    if total < 3:
        raise InputError("The vein path is shorter than three registered pixels.")
    targets = np.linspace(0.0, total, count)
    centers = np.zeros((count, 2), dtype=np.float64)
    tangents = np.zeros((count, 2), dtype=np.float64)
    for i, distance in enumerate(targets):
        segment = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(seg_lengths) - 1)
        segment = max(segment, 0)
        fraction = (distance - cumulative[segment]) / seg_lengths[segment]
        centers[i] = xy[segment] + fraction * deltas[segment]
        tangents[i] = deltas[segment] / seg_lengths[segment]
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    steps = max(int(math.ceil(band_width)), 1)
    transverse = np.linspace(-band_width / 2, band_width / 2, steps) if steps > 1 else np.array([0.0])
    coords_xy = centers[:, None, :] + transverse[None, :, None] * normals[:, None, :]
    row_coords = np.clip(coords_xy[..., 1], 0, height - 1)
    col_coords = np.clip(coords_xy[..., 0], 0, width - 1)
    return (row_coords, col_coords), total, centers, normals


def follow_rows(lab, intensity, counts, distance, unit, centers, extent):
    width, height = extent
    size = len(counts)
    axis = np.linspace(0, distance, size)
    rows = []
    for index in range(size):
        samples = int(counts[index])
        x, y = centers[index]
        rows.append({
            "position": round(float(axis[index]), 5),
            "position_unit": unit,
            "relative_position": [round(float((x + 0.5) / width), 6), round(float((y + 0.5) / height), 6)],
            "samples": samples,
            "lightness": round(float(lab[index, 0]), 4) if samples else None,
            "a": round(float(lab[index, 1]), 4) if samples else None,
            "b": round(float(lab[index, 2]), 4) if samples else None,
            "intensity": round(float(intensity[index, 0]), 5) if samples else None,
            "state": "measured" if samples else "gap",
        })
    return rows


def follow_vein_scan(frame, linear, domain, branch_metrics, lateral_number, samples, band_pixels):
    """Read color along a retained lateral vein, stopping at open gaps."""
    laterals = [item for item in branch_metrics["laterals"] if item["lateral"] == lateral_number]
    if not laterals:
        raise InputError("The chosen lateral vein is not part of this branch map.")
    lateral = laterals[0]
    trace = lateral.get("trace") or lateral.get("stump")
    if trace is None:
        raise InputError("That lateral vein has no observed trace to follow.")
    points = trace["points"]
    coords, total_px, centers, _ = curve_grid(frame.extent, points, samples, band_pixels)
    supported = frame.valid & ~frame.clipped & domain
    lab, counts = measure_line_samples(linear_lab(linear), supported, coords)
    intensity, _ = measure_line_samples(luminance(linear), supported, coords)
    valid = counts > 0
    if np.count_nonzero(valid) < 3:
        raise InputError("The vein path needs at least three supported stations to scan.")

    # A run of unsupported stations is a physical gap (e.g. the mounting
    # strip).  Everything from the first such break onward is reported empty:
    # the scan never jumps across it onto another branch.
    gap_start = None
    invalid = np.flatnonzero(~valid)
    if len(invalid):
        gap_start = int(invalid[0])
    measured_until = gap_start if gap_start is not None else len(valid)
    for index in range(measured_until, len(valid)):
        counts[index] = 0

    if frame.scale:
        scale_x, scale_y = frame.scale
        distance = total_px * math.hypot(scale_x, scale_y)
        unit = "mm"
    else:
        distance = total_px
        unit = "px"
    rows = follow_rows(lab, intensity, counts, distance, unit, centers, frame.extent)
    supported_count = int(np.count_nonzero(counts > 0))
    return {
        "mode": "follow_vein",
        "lateral_number": lateral_number,
        "path_points": points,
        "band_pixels": band_pixels,
        "line_length": round(distance, 5),
        "distance_unit": unit,
        "supported_stations": supported_count,
        "requested_stations": len(counts),
        "gap_station": gap_start,
        "open_gap": lateral.get("open_gap", False),
        "rows": rows,
        "note": (
            "Stations follow the traced curved vein. Reading stops at the first unsupported "
            "gap; later stations stay empty instead of bridging to another line."
        ),
    }
