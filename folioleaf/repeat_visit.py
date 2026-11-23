from dataclasses import replace
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

try:
    from scipy.spatial import QhullError
except ImportError:  # SciPy < 1.6 kept the class under qhull.
    from scipy.spatial.qhull import QhullError

from .art import SHADOW, stored_preview
from .color import corrected_linear, linear_lab
from .geometry import polygon_mask, specimen_frame, validate_polygon
from .model import InputError, collection, dictionary, enumeration, identity, numeric_pair, scalar, string
from .photos import linear_to_srgb, uint8_image


MIN_LANDMARKS = 4
MAX_LANDMARKS = 20
MAX_CIRCLES = 16
HULL_DILATION = 0.045
ANCHOR_REACH = 0.13
JAC_LOW, JAC_HIGH = 0.40, 2.1
FINE_SMOOTH = 3.8
COARSE_SMOOTH = 7.5
BACKGROUND_SMOOTH = 26.0
CHANGE_MIN_PIXELS = 14
MIN_CHANGE_DELTA = 3.0
DAY_ONE = (255, 240, 205)
DAY_TWO = (247, 178, 92)
DARKEN = (96, 205, 138)
LIGHTEN = (240, 196, 96)
HUE_SHIFT = (178, 139, 224)
EXCLUDE = (231, 105, 96)
UNCERTAIN = (226, 222, 205)
NO_EVIDENCE = (110, 116, 112)
EVIDENCE_EDGE = (109, 214, 150)
MISSING_EDGE = (231, 105, 96)


def landmark_fields(payload):
    data = dictionary(payload, "Repeat visit")
    first_id = identity(data.get("first_photo_id"), "First-day photo ID")
    second_id = identity(data.get("second_photo_id"), "Second-day photo ID")
    if first_id == second_id:
        raise InputError("Choose two different registered captures for the two visits.")
    raw_pairs = collection(data.get("landmarks"), "Landmark pairs", minimum=MIN_LANDMARKS, maximum=MAX_LANDMARKS)
    pairs = []
    for index, item in enumerate(raw_pairs, 1):
        entry = dictionary(item, f"Landmark pair {index}")
        pairs.append([
            numeric_pair(entry.get("first"), f"Day-one landmark {index}", lower=0, upper=1),
            numeric_pair(entry.get("second"), f"Day-two landmark {index}", lower=0, upper=1),
        ])
    circles = []
    for index, item in enumerate(collection(data.get("exclusions", []), "Exclusion circles", maximum=MAX_CIRCLES), 1):
        entry = dictionary(item, f"Exclusion {index}")
        day = enumeration(entry.get("day", "first"), ("first", "second"), "Exclusion day")
        circles.append({
            "day": day,
            "center": numeric_pair(entry.get("center"), f"Exclusion {index} center", lower=0, upper=1),
            "radius": scalar(entry.get("radius", 0.03), f"Exclusion {index} radius", lower=0.002, upper=0.18),
        })
    title = string(data.get("title"), "Comparison title", maximum=90)
    return title, first_id, second_id, pairs, circles


def _distinct(points):
    seen = {tuple(round(v, 5) for v in point) for point in points}
    return len(seen) == len(points)


def validate_landmarks(pairs):
    first = [pair[0] for pair in pairs]
    second = [pair[1] for pair in pairs]
    if not _distinct(first) or not _distinct(second):
        raise InputError("Each landmark must sit on a distinct vein crossing or notch; repeated positions cannot anchor a local warp.")
    spread = np.asarray(first, dtype=np.float64)
    if np.ptp(spread[:, 0]) < 0.04 or np.ptp(spread[:, 1]) < 0.04:
        raise InputError("Spread the landmarks across the blade rather than marking one small cluster.")


def fit_tps(pairs):
    """Thin-plate spline mapping day-one normalized positions onto day-two."""
    a = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    b = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    n = len(a)
    pairwise = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
    kernel = np.where(pairwise > 1e-12, pairwise ** 2 * np.log(np.maximum(pairwise, 1e-12)), 0.0)
    regular = 0.01 * float(np.mean(pairwise ** 2))
    block = kernel + regular * np.eye(n)
    system = np.block([
        [block, np.ones((n, 1)), a],
        [np.ones((1, n)), np.zeros((1, 3))],
        [a.T, np.zeros((2, 3))],
    ])
    rhs = np.vstack([b, np.zeros((3, 2))])
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as error:
        raise InputError("The landmarks cannot define a stable local warp. Spread them more evenly.") from error
    if not np.all(np.isfinite(weights)):
        raise InputError("The landmark geometry is degenerate. Remark the corresponding vein crossings.")
    return {"a": a, "w": weights, "regular": regular}


def tps_eval(model, coords, block=260000):
    a, weights = model["a"], model["w"]
    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
    out = np.empty((len(coords), 2), dtype=np.float64)
    for start in range(0, len(coords), block):
        p = coords[start:start + block]
        d = np.linalg.norm(p[:, None, :] - a[None, :, :], axis=2)
        basis = np.where(d > 1e-12, d ** 2 * np.log(np.maximum(d, 1e-12)), 0.0)
        design = np.hstack([basis, np.ones((len(p), 1)), p])
        out[start:start + block] = design @ weights
    return out


def warp_grid(model, extent):
    width, height = extent
    gy, gx = np.meshgrid(
        (np.arange(height, dtype=np.float64) + 0.5) / height,
        (np.arange(width, dtype=np.float64) + 0.5) / width,
        indexing="ij",
    )
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    mapped = tps_eval(model, coords)
    map_x = mapped[:, 0].reshape(height, width)
    map_y = mapped[:, 1].reshape(height, width)
    return map_x, map_y


def jacobian_status(map_x, map_y, inside):
    height, width = map_x.shape
    # Gradients in normalized plane coordinates. Image y points downward, so a
    # one-to-one surface map has determinant -1; folding means the determinant
    # crosses through zero and changes sign, not that it is negative.
    dxdy, dxdx = np.gradient(map_x, 1.0 / height, 1.0 / width)
    dydy, dydx = np.gradient(map_y, 1.0 / height, 1.0 / width)
    det = dxdx * dydy - dxdy * dydx
    scale = np.abs(det)
    # A valid warp keeps one orientation everywhere. Any sign change means a
    # local fold that would swap neighbouring vein cells or bridge a broken vein.
    negatives = np.count_nonzero((det < 0) & inside)
    positives = np.count_nonzero((det > 0) & inside)
    if negatives and positives:
        minority = min(negatives, positives)
        if minority > max(3, 0.002 * (negatives + positives)):
            raise InputError(
                "These landmarks fold the local map (neighbouring vein cells would swap). "
                "Re-check parallel-vein pairs and do not bridge a broken vein to its nearest segment."
            )
    strained = (scale < JAC_LOW) | (scale > JAC_HIGH)
    return scale, strained


def anchor_distance(points, extent):
    """Distance in normalized plane coordinates to the nearest landmark."""
    width, height = extent
    gy, gx = np.meshgrid(
        (np.arange(height, dtype=np.float32) + 0.5) / height,
        (np.arange(width, dtype=np.float32) + 0.5) / width,
        indexing="ij",
    )
    nearest = np.full((height, width), np.inf, dtype=np.float32)
    for x, y in points:
        nearest = np.minimum(nearest, np.hypot(gx - x, gy - y))
    return nearest


def hull_region(points, extent, dilation):
    width, height = extent
    try:
        hull = ConvexHull(points)
    except QhullError as error:
        raise InputError("The landmarks need to span a two-dimensional area of the blade.") from error
    grid = Image.new("1", extent)
    hull_px = [(float(x * width - 0.5), float(y * height - 0.5)) for x, y in points[hull.vertices]]
    ImageDraw.Draw(grid).polygon(hull_px, fill=1)
    inside = np.asarray(grid, dtype=bool)
    steps = max(2, int(round(dilation * min(width, height))))
    grown = ndimage.binary_dilation(inside, iterations=steps)
    return inside, grown


def anchor_distance(points, extent):
    """Distance in normalized plane coordinates to the nearest landmark."""
    width, height = extent
    gy, gx = np.meshgrid(
        (np.arange(height, dtype=np.float32) + 0.5) / height,
        (np.arange(width, dtype=np.float32) + 0.5) / width,
        indexing="ij",
    )
    nearest = np.full((height, width), np.inf, dtype=np.float32)
    for x, y in points:
        nearest = np.minimum(nearest, np.hypot(gx - x, gy - y))
    return nearest


def circle_masks(extent, circles):
    width, height = extent
    masks = {"first": np.zeros((height, width), bool), "second": np.zeros((height, width), bool)}
    for circle in circles:
        grid = Image.new("1", extent)
        x, y = circle["center"]
        r = circle["radius"]
        ImageDraw.Draw(grid).ellipse(
            ((x - r) * width, (y - r) * height, (x + r) * width, (y + r) * height), fill=1,
        )
        masks[circle["day"]] |= np.asarray(grid, dtype=bool)
    return masks


def remap_second(frame_two, map_x, map_y):
    height, width = map_x.shape
    coords = (map_y * height - 0.5, map_x * width - 0.5)
    warped = np.empty((height, width, 3), np.float32)
    for channel in range(3):
        warped[:, :, channel] = ndimage.map_coordinates(
            frame_two.linear[:, :, channel], coords, order=1, mode="constant", cval=0,
        )
    valid = ndimage.map_coordinates(
        frame_two.valid.astype(np.float32), coords, order=1, mode="constant", cval=0,
    ) >= 0.995
    return warped, valid


def blade_mask(linear):
    """Greenness-based blade mask, independent of marked reference patches."""
    r, g, b = linear[:, :, 0], linear[:, :, 1], linear[:, :, 2]
    greenness = g - 0.55 * (r + b)
    threshold = float(np.quantile(greenness, 0.62))
    mask = greenness >= max(threshold, 0.012)
    mask = ndimage.binary_opening(mask, iterations=2)
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    keep = int(np.argmax(sizes)) + 1
    return labels == keep


def evidence_tiers(extent, frame_one, frame_two, model, circles):
    map_x, map_y = warp_grid(model, extent)
    warped_two, two_supported = remap_second(frame_two, map_x, map_y)
    points = model["a"]
    hull_core, hull_grown = hull_region(points, extent, HULL_DILATION)
    _, strained = jacobian_status(map_x, map_y, hull_grown)
    masks = circle_masks(extent, circles)
    excluded = masks["first"] | masks["second"]
    # Evidence tiers describe blade tissue only; paper backing is never a color claim.
    blade = blade_mask(frame_one.linear) & ndimage.binary_dilation(blade_mask(warped_two), iterations=2)
    base_support = frame_one.valid & two_supported & blade
    distance = anchor_distance(points, extent)
    anchored = distance <= ANCHOR_REACH
    # High-contrast ridges (veins, blade rim) cannot be trusted at sub-pixel
    # mismatch; exclude them from strong evidence so edge shift is not a color change.
    gray = np.mean(frame_one.linear, axis=-1)
    ridge = ndimage.gaussian_filter(np.hypot(*np.gradient(ndimage.gaussian_filter(gray, 1.2))), 1.1) > 0.030
    # A rim band is kept out of the color-change claim because the blade edge
    # shifts position as it re-flattens.
    blade_interior = ndimage.binary_erosion(blade, iterations=3)
    rim_band = blade & ~ndimage.binary_erosion(blade, iterations=5)
    no_evidence = excluded | ~base_support
    strong = base_support & blade_interior & hull_grown & anchored & ~strained & ~ridge
    uncertain = base_support & ~no_evidence & ~strong
    if np.count_nonzero(strong) < 64:
        raise InputError("Fewer than 64 well-supported blade pixels remain between the landmarks and exclusions; add landmarks or shrink exclusions.")
    tiers = {"strong": strong, "uncertain": uncertain, "no_evidence": no_evidence, "blade": blade, "rim_band": rim_band}
    return warped_two, two_supported, map_x, map_y, tiers

def _smoothed_lab(lab, sigma):
    return np.stack([ndimage.gaussian_filter(lab[:, :, c], sigma) for c in range(3)], axis=2)


def change_components(lab_one, lab_warped, strong, rim_band):
    # Two scales: fine catches a small new spot, coarse survives smoothing over
    # a fading old mark. Both are coarser than a vein width so edge
    # misregistration never reads as tissue change.
    fine = (_smoothed_lab(lab_one, FINE_SMOOTH), _smoothed_lab(lab_warped, FINE_SMOOTH))
    coarse = (_smoothed_lab(lab_one, COARSE_SMOOTH), _smoothed_lab(lab_warped, COARSE_SMOOTH))
    scale_maps = []
    for first_lab, second_lab in (fine, coarse):
        residual = second_lab - first_lab
        offset = np.median(residual[strong], axis=0)
        local = residual - offset
        magnitude = np.linalg.norm(local, axis=-1)
        background = ndimage.gaussian_filter(magnitude, BACKGROUND_SMOOTH)
        scale_maps.append((local, magnitude - background))
    local_lab, excess_coarse = scale_maps[1]
    excess_fine = scale_maps[0][1]
    excess = np.maximum(excess_fine, excess_coarse)
    noise = float(np.quantile(excess[strong], 0.95))
    robust_median = float(np.median(excess[strong]))
    robust_sd = float(np.median(np.abs(excess[strong] - robust_median))) * 1.4826
    threshold = max(noise, robust_median + 3.0 * robust_sd, MIN_CHANGE_DELTA)
    changed = strong & (excess >= threshold)
    dl = local_lab[:, :, 0]
    chroma_delta = np.abs(local_lab[:, :, 1:]).max(axis=2)
    components = []
    labels, count = ndimage.label(changed)
    for label_id in range(1, count + 1):
        member = labels == label_id
        pixels = int(np.count_nonzero(member))
        if pixels < CHANGE_MIN_PIXELS:
            continue
        # Reject components that mostly hug the moving blade rim: their color
        # difference is edge repositioning, not tissue change.
        if np.count_nonzero(member & rim_band) > 0.45 * pixels:
            continue
        rows, cols = np.nonzero(member)
        median_dl = float(np.median(dl[member]))
        kind = "hue" if abs(median_dl) < 2.5 and float(np.median(chroma_delta[member])) > 3.0 else (
            "lightened" if median_dl > 0 else "darkened")
        components.append({
            "key": f"C{len(components) + 1}",
            "kind": kind,
            "pixels": pixels,
            "center": [round(float(cols.mean()) / strong.shape[1], 5), round(float(rows.mean()) / strong.shape[0], 5)],
            "median_delta_e": round(float(np.median(np.linalg.norm(local_lab, axis=-1)[member])), 3),
            "median_lightness_change": round(median_dl, 3),
        })
    components.sort(key=lambda item: -item["pixels"])
    for index, item in enumerate(components, 1):
        item["key"] = f"C{index}"
    return changed, excess, threshold, components, robust_median, np.median(scale_maps[1][0][strong], axis=0)


def checkerboard(one, warped_two, extent, cells=10):
    width, height = extent
    step = max(6, round(min(width, height) / cells))
    yy, xx = np.ogrid[:height, :width]
    switch = ((xx // step) + (yy // step)) % 2 == 0
    return np.where(switch[:, :, None], one, warped_two)


def blade_outline(draw, mask, extent, color=(236, 230, 206)):
    padded = np.pad(mask, 1)
    rim = (
        (padded[1:-1, 1:-1] & ~padded[:-2, 1:-1])
        | (padded[1:-1, 1:-1] & ~padded[2:, 1:-1])
        | (padded[1:-1, 1:-1] & ~padded[1:-1, :-2])
        | (padded[1:-1, 1:-1] & ~padded[1:-1, 2:])
    )
    rows, cols = np.nonzero(rim)
    for y, x in zip(rows.tolist(), cols.tolist()):
        draw.point((x, y), fill=color)


def _draw_markers(draw, extent, pairs, circles, top_offset=0):
    width, height = extent
    for index, (one, two) in enumerate(pairs, 1):
        for point, color, dy in ((one, DAY_ONE, -16), (two, DAY_TWO, 4)):
            x, y = point[0] * width, top_offset + point[1] * height
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=SHADOW, outline=color, width=2)
            draw.text((x + 7, y + dy), str(index), fill=color, stroke_width=2, stroke_fill=SHADOW)
    for circle in circles:
        x, y = circle["center"]
        r = circle["radius"]
        draw.ellipse(((x - r) * width, (top_offset + y - r) * height,
                      (x + r) * width, (top_offset + y + r) * height),
                     outline=EXCLUDE, width=2)
        draw.line(((x - r) * width, (top_offset + y - r) * height,
                   (x + r) * width, (top_offset + y + r) * height), fill=EXCLUDE, width=1)


def hatch(image, mask, color, gap=9):
    arr = np.asarray(image).copy()
    yy, xx = np.mgrid[0: arr.shape[0], 0: arr.shape[1]]
    stripe = ((xx + yy) % gap < 2) & mask
    arr[stripe, :3] = color
    return Image.fromarray(arr)


def warp_overlay(frame_one, warped_two, tiers, extent, pairs, circles):
    two_view = linear_to_srgb(warped_two)
    # Where day two has no support, keep the day-one pixel instead of a black cell.
    two_view = np.where(frame_one.valid[:, :, None], two_view, frame_one.display())
    blended = checkerboard(frame_one.display(), two_view, extent)
    image = uint8_image(blended, frame_one.valid.astype(np.float32)).convert("RGBA")
    image = hatch(image, tiers["uncertain"] & tiers["blade"], UNCERTAIN, 11)
    image = hatch(image, tiers["no_evidence"] & tiers["blade"], NO_EVIDENCE, 7)
    draw = ImageDraw.Draw(image)
    blade_outline(draw, tiers["blade"], extent)
    _draw_markers(draw, extent, pairs, circles)
    return image


def change_overlay(frame_one, changed, tiers, components, extent):
    base = frame_one.display() * 0.55
    base[changed] = (0.30 * base[changed] + 0.70 * np.array(DARKEN, np.float32) / 255)
    image = uint8_image(base, frame_one.valid.astype(np.float32)).convert("RGBA")
    image = hatch(image, tiers["uncertain"] & tiers["blade"], UNCERTAIN, 11)
    image = hatch(image, tiers["no_evidence"] & tiers["blade"], NO_EVIDENCE, 7)
    draw = ImageDraw.Draw(image)
    blade_outline(draw, tiers["blade"], extent)
    for item in components:
        x, y = item["center"][0] * extent[0], item["center"][1] * extent[1]
        color = LIGHTEN if item["kind"] == "lightened" else HUE_SHIFT if item["kind"] == "hue" else DARKEN
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=2)
        draw.text((x + 8, y - 8), item["key"], fill=color, stroke_width=2, stroke_fill=SHADOW)
    return image


def tradeoff_sheet(frame_one, frame_two, warped_two, tiers, extent, pairs, circles):
    width, height = extent
    gap, head = 18, 34
    sheet = Image.new("RGBA", (width, 2 * height + gap + head), (28, 38, 34, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 8), "Day 1 plane (with day-2 evidence limits)", fill=DAY_ONE)
    panels = []
    for source, day_circles in ((frame_one.display(), [c for c in circles if c["day"] == "first"]),
                                (linear_to_srgb(warped_two), [c for c in circles if c["day"] == "second"])):
        panel = uint8_image(source, frame_one.valid.astype(np.float32)).convert("RGBA")
        panel = hatch(panel, tiers["uncertain"] & tiers["blade"], UNCERTAIN, 11)
        panel = hatch(panel, tiers["no_evidence"] & tiers["blade"], NO_EVIDENCE, 7)
        pdraw = ImageDraw.Draw(panel)
        blade_outline(pdraw, tiers["blade"], extent)
        for circle in day_circles:
            x, y = circle["center"]
            r = circle["radius"]
            pdraw.ellipse(((x - r) * width, (y - r) * height, (x + r) * width, (y + r) * height),
                          outline=EXCLUDE, width=2)
        panels.append(panel)
    sheet.paste(panels[0], (0, head))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, head + height + gap), "Day 2 warped onto day 1 (same local mesh)", fill=DAY_TWO)
    sheet.paste(panels[1], (0, head + height + gap))
    draw = ImageDraw.Draw(sheet)
    for index, (one, two) in enumerate(pairs, 1):
        for point, color, top, dy in ((one, DAY_ONE, head, -16), (two, DAY_TWO, head + height + gap, 4)):
            x, y = point[0] * width, top + point[1] * height
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=SHADOW, outline=color, width=2)
            draw.text((x + 7, y + dy), str(index), fill=color, stroke_width=2, stroke_fill=SHADOW)
    return sheet


def build_repeat(store, tile, payload):
    title, first_id, second_id, pairs, circles = landmark_fields(payload)
    validate_landmarks(pairs)
    frame_one = specimen_frame(store.blobs, tile, first_id)
    frame_two = specimen_frame(store.blobs, tile, second_id)
    if frame_one.extent != frame_two.extent:
        raise InputError("Both captures must be registered on the same survey plane before comparing visits.")
    linear_one, _ = corrected_linear(tile, frame_one, True)
    linear_two, _ = corrected_linear(tile, frame_two, True)
    frame_one = replace(frame_one, linear=linear_one)
    frame_two = replace(frame_two, linear=linear_two)
    model = fit_tps(pairs)
    warped_two, _, map_x, map_y, tiers = evidence_tiers(frame_one.extent, frame_one, frame_two, model, circles)
    lab_one, lab_warped = linear_lab(frame_one.linear), linear_lab(warped_two)
    changed, local_de, threshold, components, robust_median, global_offset = change_components(
        lab_one, lab_warped, tiers["strong"], tiers["rim_band"])
    extent = frame_one.extent
    displacement = [round(float(np.linalg.norm(np.array(b) - np.array(a))), 5) for a, b in pairs]
    scale = float(np.prod(frame_one.scale)) if frame_one.scale else 1.0
    blade_total = int(np.count_nonzero(tiers["blade"]))
    tier_keys = ("strong", "uncertain", "no_evidence")
    counts = {key: int(np.count_nonzero(tiers[key] & tiers["blade"])) for key in tier_keys}
    metrics = {
        "landmark_count": len(pairs),
        "landmark_displacement": displacement,
        "median_displacement": round(float(np.median(displacement)), 5),
        "blade_pixels": blade_total,
        "evidence_pixels": counts,
        "evidence_percent": {key: round(100 * value / max(blade_total, 1), 2) for key, value in counts.items()},
        "robust_delta_e_median": round(robust_median, 3),
        "change_threshold_delta_e": round(threshold, 3),
        "global_lab_offset": np.round(global_offset, 3).tolist(),
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_area": round(np.count_nonzero(changed) * scale, 4),
        "area_unit": "mm²" if frame_one.scale else "px²",
        "darkened_components": sum(c["kind"] == "darkened" for c in components),
        "lightened_components": sum(c["kind"] == "lightened" for c in components),
        "hue_components": sum(c["kind"] == "hue" for c in components),
        "components": components,
        "uncertain_note": "Only blade tissue is judged. Blade pixels outside the landmark envelope, far from an anchor, on a vein ridge, under local strain or inside an exclusion circle are kept uncertain and are never reported as a color change.",
    }
    documents = {
        "title": title,
        "tile_id": tile["id"],
        "first_photo_id": first_id,
        "second_photo_id": second_id,
        "landmarks": pairs,
        "exclusions": circles,
        "regularization": model["regular"],
        "metrics": metrics,
        "extent": list(extent),
    }
    pictures = {
        "overlay": stored_preview(warp_overlay(frame_one, warped_two, tiers, extent, pairs, circles), store.blobs),
        "change": stored_preview(change_overlay(frame_one, changed, tiers, components, extent), store.blobs),
        "tradeoff": stored_preview(
            tradeoff_sheet(frame_one, frame_two, warped_two, tiers, extent, pairs, circles), store.blobs),
    }
    return documents, pictures, {"map_x": map_x, "map_y": map_y, "tiers": tiers, "warped_two": warped_two}


def _edge_support(vertices, extent, mask):
    factor = np.asarray(extent, dtype=np.float64)
    edge_rows = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        length = float(np.linalg.norm((np.array(end) - np.array(start)) * factor))
        count = min(2400, max(8, int(math.ceil(length)) + 1))
        line = np.linspace(start, end, count)
        sampled = ndimage.map_coordinates(
            mask.astype(np.float32), (line[:, 1] * extent[1] - 0.5, line[:, 0] * extent[0] - 0.5),
            order=1, mode="constant", cval=0,
        )
        strong_fraction = float(np.mean(sampled >= 0.995))
        edge_rows.append({
            "index": index + 1,
            "comparable": strong_fraction >= 0.85,
            "strong_support_percent": round(100 * strong_fraction, 2),
        })
    return edge_rows


def compare_region(store, tile, repeat, region, frame_one, frame_two):
    vertices = validate_polygon([list(v) for v in region["vertices"]])
    model = fit_tps(repeat["landmarks"])
    warped_vertices = tps_eval(model, np.asarray(vertices, dtype=np.float64))
    if np.any((warped_vertices < -0.05) | (warped_vertices > 1.05)):
        raise InputError("Part of this region warps beyond the second-day frame; its edge cannot be compared there.")
    warped_vertices = np.clip(warped_vertices, 0.0, 1.0).tolist()
    extent = frame_one.extent
    warped_two, _, _, _, tiers = evidence_tiers(extent, frame_one, frame_two, model, repeat["exclusions"])
    mask_one = polygon_mask(extent, vertices)
    mask_two = polygon_mask(extent, warped_vertices)
    scale = float(np.prod(frame_one.scale)) if frame_one.scale else 1.0
    support_one = mask_one & frame_one.valid
    # Day-two area counts only pixels the warp can actually deliver evidence for.
    support_two = mask_two & (tiers["strong"] | tiers["uncertain"])
    edges_one = _edge_support(vertices, extent, tiers["strong"])
    edges_two = _edge_support(warped_vertices, extent, tiers["strong"])
    shared = int(np.count_nonzero(mask_one & mask_two & tiers["strong"]))
    shape = {
        "points": vertices,
        "warped_points": np.round(warped_vertices, 6).tolist(),
        "day_one_area": round(np.count_nonzero(support_one) * scale, 4),
        "day_two_area": round(np.count_nonzero(support_two) * scale, 4),
        "day_one_raster_px": int(np.count_nonzero(mask_one)),
        "day_two_raster_px": int(np.count_nonzero(mask_two)),
        "shared_strong_px": shared,
        "units": "mm²" if frame_one.scale else "px²",
        "edges_day_one": edges_one,
        "edges_day_two": edges_two,
        "comparable_edge_count": sum(e["comparable"] for e in edges_one) + sum(e["comparable"] for e in edges_two),
        "missing_edge_count": sum(not e["comparable"] for e in edges_one) + sum(not e["comparable"] for e in edges_two),
        "repeat_id": repeat["id"],
    }
    image = region_comparison_image(frame_one, warped_two, tiers, vertices, warped_vertices,
                                    edges_one, edges_two, shape, extent)
    return shape, image


def _panel(frame_linear, valid, vertices, warped_vertices, edges, extent, title, base_vertices):
    image = uint8_image(linear_to_srgb(frame_linear), valid.astype(np.float32)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = extent
    primary = [(x * width, y * height) for x, y in vertices]
    reference = [(x * width, y * height) for x, y in base_vertices]
    draw.line([*reference, reference[0]], fill=(210, 214, 198, 150), width=1)
    for index, (start, end) in enumerate(zip(primary, primary[1:] + primary[:1])):
        color = EVIDENCE_EDGE if edges[index]["comparable"] else MISSING_EDGE
        draw.line((start, end), fill=SHADOW, width=4)
        draw.line((start, end), fill=color, width=2)
    draw.text((6, 6), title, fill=(255, 245, 225), stroke_width=2, stroke_fill=SHADOW)
    return image


def region_comparison_image(frame_one, warped_two, tiers, vertices, warped_vertices,
                            edges_one, edges_two, shape, extent):
    width, height = extent
    gap, head = 18, 30
    sheet = Image.new("RGBA", (width, 2 * height + gap + 2 * head), (28, 38, 34, 255))
    unit_label = shape["units"].replace("²", "2")
    panel_one = _panel(frame_one.linear, frame_one.valid, vertices, warped_vertices,
                       edges_one, extent, f"Day 1 area {shape['day_one_area']} {unit_label}", warped_vertices)
    panel_two = _panel(warped_two, frame_one.valid, warped_vertices, vertices,
                       edges_two, extent, f"Day 2 same mesh area {shape['day_two_area']} {unit_label}", vertices)
    panel_two = hatch(panel_two, tiers["no_evidence"] & tiers["blade"], NO_EVIDENCE, 7)
    sheet.paste(panel_one, (0, head))
    sheet.paste(panel_two, (0, 2 * head + height + gap))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), "Green edge: comparable on both days · red edge: missing evidence / uncertain", fill=UNCERTAIN)
    return sheet
