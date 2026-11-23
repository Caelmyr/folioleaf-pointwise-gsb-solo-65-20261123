"""Batch tone-card correction.

One global, monotone non-decreasing tone curve is fitted from a photographed
gray scale whose patches carry their own (operator-verified) reflectance
readings.  The curve operates on scalar luminance in the linear working image
and scales the three channels together, so it can never reorder the original
brightness, never drives a channel into the capture ceiling, and preserves the
chromaticity / color proportions of every pixel.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from .art import font
from .color import float_list, linear_lab, swatch
from .geometry import polygon_mask, region_domain, specimen_frame
from .model import (
    InputError,
    collection,
    dictionary,
    enumeration,
    identity,
    scalar,
    string,
)
from .photos import linear_to_srgb, luminance, png_bytes, uint8_image

# Card order from dark to light.  The two light grays stay distinct by role.
PATCH_ROLES = ("dark", "mid", "light-a", "light-b")
ROLE_LABELS = {
    "dark": "Dark gray",
    "mid": "Middle gray",
    "light-a": "Light gray A",
    "light-b": "Light gray B",
}
MIN_PATCH_SAMPLES = 40
MIN_UNCLIPPED_FRACTION = 0.70
MIN_LUMINANCE = 0.004
# Within-patch coefficient of spread above which glare/nonuniformity rejects it.
MAX_PATCH_SPREAD = 0.30
# Residual (L* units) inside which a patch counts as "held" rather than
# "compromised by the monotone constraint".
HOLD_TOLERANCE_LSTAR = 1.5
MIN_FIT_POINTS = 2
LUT_LEVELS = 4096
# Curves are only accepted over a bounded gain envelope.
MIN_SLOPE = 0.18
MAX_SLOPE = 4.0


def patch_parameters(payload):
    data = dictionary(payload, "Tone-card request")
    patches = collection(data.get("patches"), "Tone-card patches", minimum=1, maximum=len(PATCH_ROLES))
    parsed = []
    seen = set()
    for item in patches:
        entry = dictionary(item, "Tone-card patch")
        role = enumeration(entry.get("role"), PATCH_ROLES, "Gray-card role")
        if role in seen:
            raise InputError(f"Only one {ROLE_LABELS[role]} patch can be listed.")
        seen.add(role)
        region_id = identity(entry.get("region_id"), "Patch region ID")
        target = scalar(entry.get("target"), f"{ROLE_LABELS[role]} card reading", lower=0.001, upper=0.98)
        parsed.append({"role": role, "region_id": region_id, "target": float(target)})
    note = string(data.get("note", ""), "Tone-card note", maximum=600, required=False)
    return data, parsed, note


def _patch_mask(frame, tile, region_id):
    region = next((item for item in tile["regions"] if item["id"] == region_id), None)
    if region is None:
        raise InputError("A marked tone-card region has been removed. Redraw the gray-scale patches.")
    if region["purpose"] == "ignore":
        raise InputError("An exclusion region cannot be used as a gray-card patch.")
    domain = region_domain(frame.extent, tile, region)
    return polygon_mask(frame.extent, region["vertices"]) & domain, region


def _patch_reading(frame, mask):
    native = int(np.count_nonzero(mask & frame.valid))
    support = mask & frame.valid & ~frame.clipped
    count = int(np.count_nonzero(support))
    clipped = native - count
    rejected = None
    if native < MIN_PATCH_SAMPLES:
        rejected = "too few supported pixels in the marked cell"
    elif count / max(native, 1) < MIN_UNCLIPPED_FRACTION:
        rejected = "the cell is dominated by clipped glare pixels"
    colors = frame.linear[support] if count else np.zeros((1, 3), dtype=np.float32)
    levels = luminance(colors)
    measured = float(np.median(levels)) if count else 0.0
    spread = (
        float((np.quantile(levels, 0.9) - np.quantile(levels, 0.1)) / max(measured, 1e-8))
        if count else np.inf
    )
    if rejected is None and measured < MIN_LUMINANCE:
        rejected = "the cell is too dark for a stable curve reading"
    if rejected is None and spread > MAX_PATCH_SPREAD:
        rejected = "the cell brightness varies too much (glare or a boundary crossing)"
    median_rgb = np.median(colors, axis=0) if count else np.zeros(3, dtype=np.float32)
    return {
        "supported_pixels": count,
        "clipped_pixels": int(clipped),
        "measured_luminance": round(measured, 6),
        "within_patch_spread": round(float(spread), 4),
        "median_rgb": float_list(median_rgb, 6),
        "rejected_reason": rejected,
    }


def collect_patches(frame, tile, requested):
    rows = []
    for entry in requested:
        mask, region = _patch_mask(frame, tile, entry["region_id"])
        reading = _patch_reading(frame, mask)
        rows.append({
            **entry,
            "region_name": region["name"],
            "region_vertices": region["vertices"],
            **reading,
        })
    return rows


def _fit_nodes(patches):
    """Monotone non-decreasing (measured -> card) anchors.

    The card readings already carry the true dark-to-light order.  Anchors are
    sorted by what the scan measured, and ties (e.g. an inversion where two
    cells came out equally bright) are averaged so the fitted table can never
    decrease; the monotone LUT then restores the proper brightness ordering.
    """
    usable = [p for p in patches if p["rejected_reason"] is None]
    if len(usable) < MIN_FIT_POINTS:
        raise InputError("At least two usable gray cells are needed to define a tone curve.")
    pairs = sorted(
        ((float(p["measured_luminance"]), float(p["target"]), p["role"]) for p in usable),
        key=lambda item: item[0],
    )
    xs, ys, roles = [], [], []
    for measured, target, role in pairs:
        if xs and abs(measured - xs[-1]) <= 1e-9:
            xs[-1] = (xs[-1] + measured) / 2
            ys[-1] = (ys[-1] + target) / 2
            roles[-1] = roles[-1] + "+" + role
            continue
        xs.append(measured)
        ys.append(target)
        roles.append(role)
    # Guarantee a strictly non-decreasing target sequence against rounding.
    for index in range(1, len(ys)):
        if ys[index] < ys[index - 1]:
            ys[index] = ys[index - 1]
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), roles


def build_lut(measured, target, roles=None):
    """Fit the single monotone non-decreasing tone curve.

    The weighted least-squares objective in L* lightness expresses the stated
    priority directly: middle gray stays on its card reading first, then both
    light grays (held as two distinct levels), then the dark gray.  Monotone
    ordering and a minimum separation of the two light cells are hard
    constraints, so the curve can never invert the original brightness.  Where a
    scanned inversion makes every target unreachable at once, the anchors meet
    part-way and the affected patches are reported as the accepted error tier.
    """
    measured = np.asarray(measured, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    roles = list(roles or [])
    levels = np.linspace(0.0, 1.0, LUT_LEVELS, dtype=np.float64)
    if len(measured) == 1:
        mapped = levels * (target[0] / max(measured[0], 1e-9))
    else:
        weights = np.array([_role_weight(role) for role in roles], dtype=np.float64)
        if len(weights) != len(measured):
            weights = np.ones(len(measured))
        light_indices = [index for index, role in enumerate(roles) if str(role).startswith("light")]
        min_gap = _min_light_gap(target, light_indices)
        target_lstar = _lstar(np.clip(target, 1e-5, None))

        def objective(values):
            return float(np.sum(weights * (_lstar(np.clip(values, 1e-5, None)) - target_lstar) ** 2))

        constraints = [{"type": "ineq", "fun": lambda values: np.diff(values) - 1e-5}]
        if len(light_indices) == 2:
            lo, hi = sorted(light_indices, key=lambda index: measured[index])
            constraints.append({
                "type": "ineq",
                "fun": lambda values, lo=lo, hi=hi: values[hi] - values[lo] - min_gap,
            })
        bounds = [(max(0.002, node - 0.18), min(0.97, node + 0.18)) for node in target]
        starts = (target.copy(), np.sort(target.copy()), np.linspace(0.05, 0.6, len(target)))
        best = None
        for start in starts:
            estimate = minimize(
                objective, start, method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            if estimate.success and (best is None or estimate.fun < best.fun):
                best = estimate
        if best is None:
            raise InputError("No monotone tone curve could be fitted to these gray cells.")
        fitted = np.maximum.accumulate(best.x)
        order = np.argsort(measured, kind="stable")
        interpolator = PchipInterpolator(measured[order], fitted[order], extrapolate=True)
        mapped = interpolator(levels)
        low_slope = float(np.clip(fitted[order][0] / max(measured[order][0], 1e-9), MIN_SLOPE, MAX_SLOPE))
        mapped[levels < measured[order][0]] = low_slope * levels[levels < measured[order][0]]
        end_slope = float(np.clip(
            (fitted[order][-1] - fitted[order][-2]) / max(measured[order][-1] - measured[order][-2], 1e-9),
            0.0, MAX_SLOPE,
        ))
        tail = levels >= measured[order][-1]
        mapped[tail] = fitted[order][-1] + end_slope * (levels[tail] - measured[order][-1])
    mapped = np.maximum(mapped, 0.0)
    monotone = np.maximum.accumulate(mapped)
    slopes = np.diff(monotone) / (levels[1] - levels[0])
    # A flat (zero-slope) segment is a legal compression; reject only runaway gain.
    if np.max(slopes) > MAX_SLOPE * 1.5:
        raise InputError("The gray cells would require a tone curve outside the supported gain range.")
    return np.round(monotone, 7).astype(np.float32)


def _role_weight(role):
    label = str(role)
    if label == "mid":
        return 8.0
    if label.startswith("light"):
        return 2.0
    return 1.0


def _min_light_gap(target, light_indices):
    """Smallest linear-luminance gap that still reads as two L* steps apart."""
    if len(light_indices) != 2:
        return 0.0
    gap = abs(float(target[light_indices[0]] - target[light_indices[1]]))
    return max(gap * 0.6, 0.012)


def apply_tone_lut(linear, lut):
    """Apply the scalar curve while keeping each pixel's channel ratios."""
    luma = luminance(linear)
    index = np.clip(luma * (LUT_LEVELS - 1), 0, LUT_LEVELS - 1)
    lower = np.floor(index).astype(np.int32)
    upper = np.minimum(lower + 1, LUT_LEVELS - 1)
    fraction = index - lower
    mapped = lut[lower] * (1 - fraction) + lut[upper] * fraction
    scale = np.divide(mapped, np.maximum(luma, 1e-12), where=luma > 1e-12, out=np.zeros_like(luma))
    # No channel may reach the ceiling: ratios are preserved and the common
    # factor is trimmed where necessary instead of clipping a single channel.
    peak = np.max(linear, axis=-1)
    over = peak > 0
    ceiling = np.divide(0.975, peak, where=over, out=np.ones_like(peak))
    scale = np.minimum(scale, ceiling)
    scale = np.where(luma > 1e-12, scale, 0.0)
    return np.clip(linear * scale[..., None], 0.0, 0.98)


def _lstar(luminance_value):
    """L* for achromatic linear luminance, matching color.linear_lab's white point."""
    values = np.clip(np.asarray(luminance_value, dtype=np.float32), 1e-7, None)
    neutral = np.broadcast_to(values[..., None], values.shape + (3,))
    return np.asarray(linear_lab(np.ascontiguousarray(neutral))[..., 0], dtype=np.float64)


def evaluate_patches(patches, lut):
    """Report what the chosen curve delivers at every requested cell."""
    levels = np.linspace(0.0, 1.0, LUT_LEVELS, dtype=np.float64)
    report = []
    for patch in patches:
        row = {key: patch[key] for key in (
            "role", "region_id", "region_name", "region_vertices", "target", "measured_luminance",
            "supported_pixels", "clipped_pixels", "within_patch_spread", "rejected_reason",
        )}
        row["role_label"] = ROLE_LABELS[patch["role"]]
        row["measured_swatch"] = swatch(np.full(3, patch["measured_luminance"], dtype=np.float32))
        if patch["rejected_reason"] is not None:
            row["status"] = "rejected"
            row["fitted_luminance"] = None
            row["fitted_swatch"] = None
            row["target_error_lstar"] = None
            row["held"] = False
        else:
            fitted = float(np.interp(patch["measured_luminance"], levels, lut))
            error = float(abs(_lstar(fitted) - _lstar(patch["target"])))
            row["fitted_luminance"] = round(fitted, 6)
            row["fitted_swatch"] = swatch(np.full(3, fitted, dtype=np.float32))
            row["target_swatch"] = swatch(np.full(3, patch["target"], dtype=np.float32))
            row["target_error_lstar"] = round(error, 3)
            row["held"] = error <= HOLD_TOLERANCE_LSTAR
            row["status"] = "held" if row["held"] else "compromised"
        report.append(row)
    return report


def tone_card_record(store, tile, payload, *, persist):
    data, requested, note = patch_parameters(payload)
    photo_id = identity(data["photo_id"], "Photo ID") if data.get("photo_id") else tile["plane"]["photo_id"]
    frame = specimen_frame(store.blobs, tile, photo_id)
    patches = collect_patches(frame, tile, requested)
    measured, target, used_roles = _fit_nodes(patches)
    lut = build_lut(measured, target, used_roles)
    evaluation = evaluate_patches(patches, lut)
    curve_svg = curve_diagram(evaluation, lut)
    corrected = apply_tone_lut(frame.linear, lut)
    overview = card_overview(frame, evaluation, corrected)
    record = {
        "kind": "tone-card",
        "photo_id": photo_id,
        "note": note,
        "lut": lut.tolist(),
        "lut_levels": LUT_LEVELS,
        "anchors": [
            {"role": role, "measured_luminance": round(float(xv), 6), "card_luminance": round(float(yv), 6)}
            for role, xv, yv in zip(used_roles, measured, target)
        ],
        "patches": evaluation,
        "plane_size": list(frame.extent),
        "plane_corners": tile["plane"]["corners"],
        "curve_svg": curve_svg,
        "hold_tolerance_lstar": HOLD_TOLERANCE_LSTAR,
        "color_rule": "One scalar luminance curve scales RGB together; ratios are preserved and no channel is clipped.",
    }
    preview_digest = store.blobs.put(png_bytes(overview))
    record["overview_digest"] = preview_digest
    if persist:
        def change(live):
            live["tone_card"] = record
        tile = store.edit_tile(tile["id"], tile["version"], change, "tone-card", "Saved batch gray-scale tone curve")
        record = tile["tone_card"]
    return record, overview


def _draw_box(draw, bounds, color, width):
    draw.rectangle(bounds, outline=color, width=width)


def card_overview(frame, patches, corrected):
    """Side-by-side original/corrected survey plane with cell verdicts."""
    before = uint8_image(linear_to_srgb(frame.linear), frame.valid.astype(np.float32)).convert("RGB")
    after = uint8_image(linear_to_srgb(corrected), frame.valid.astype(np.float32)).convert("RGB")
    gap, top_band, bottom_band = 18, 58, 96
    width, height = before.size
    sheet = Image.new("RGB", (width * 2 + gap, height + top_band + bottom_band), (24, 31, 34))
    sheet.paste(before, (0, top_band))
    sheet.paste(after, (width + gap, top_band))
    draw = ImageDraw.Draw(sheet)
    title_font, small_font = font(20), font(13)
    draw.text((8, 12), "Original survey plane", fill=(242, 238, 223), font=title_font)
    draw.text((width + gap + 8, 12), "Single monotone tone curve applied", fill=(242, 238, 223), font=title_font)
    status_color = {"held": (110, 200, 150), "compromised": (235, 178, 94), "rejected": (224, 116, 120)}
    role_tag = {"dark": "D", "mid": "M", "light-a": "A", "light-b": "B"}
    line_w = max(2, round(min(width, height) / 320))
    for patch in patches:
        vertices = patch["region_vertices"]
        points = [(x * width, top_band + y * height) for x, y in vertices]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        color = status_color[patch["status"]]
        for offset in (0, width + gap):
            shifted = (bounds[0] + offset, bounds[1], bounds[2] + offset, bounds[3])
            _draw_box(draw, shifted, color, line_w)
            # A compact role tag inside the cell cannot collide with neighbors.
            draw.text((shifted[0] + 4, shifted[1] + 2), role_tag[patch["role"]],
                      fill=color, font=small_font)
    footer_y = height + top_band + 10
    held = [p for p in patches if p["status"] == "held"]
    compromised = [p for p in patches if p["status"] == "compromised"]
    rejected = [p for p in patches if p["status"] == "rejected"]
    legend = "  ".join(f"{role_tag[p['role']]}={ROLE_LABELS[p['role']]}" for p in patches)
    summary = (
        f"Held within {HOLD_TOLERANCE_LSTAR:g} L*: {', '.join(ROLE_LABELS[p['role']] for p in held) or '—'}   "
        f"Accepted error: {', '.join(ROLE_LABELS[p['role']] for p in compromised) or '—'}   "
        f"Rejected (glare): {', '.join(ROLE_LABELS[p['role']] for p in rejected) or '—'}"
    )
    draw.text((8, footer_y), summary, fill=(210, 214, 200), font=small_font)
    draw.text((8, footer_y + 22), legend, fill=(167, 181, 174), font=small_font)
    draw.text((8, footer_y + 44), "Curve never decreases; equal input luminance maps to one output; RGB ratios retained without channel clipping.",
              fill=(167, 181, 174), font=small_font)
    return sheet


def curve_diagram(patches, lut):
    width, height = 320, 240
    left, top, right, bottom = 38, 16, width - 14, height - 34
    levels = np.linspace(0, 1, LUT_LEVELS)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Fitted monotone tone curve">',
        f'<rect width="{width}" height="{height}" fill="#18201f"/>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = left + fraction * (right - left)
        gy = bottom - fraction * (bottom - top)
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{bottom}" stroke="#33403d" stroke-width="1"/>')
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" stroke="#33403d" stroke-width="1"/>')
    colors = {"held": "#6ec896", "compromised": "#ebb25e", "rejected": "#e07478"}
    used = [p for p in patches if p["rejected_reason"] is None]
    used.sort(key=lambda p: p["measured_luminance"])
    path = []
    for index, value in enumerate(levels[::64]):
        px = left + value * (right - left)
        py = bottom - float(np.interp(value, levels, lut)) * (bottom - top)
        path.append(("M" if index == 0 else "L") + f"{px:.1f},{py:.1f}")
    parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="#dfe7d8" stroke-width="2"/>')
    for patch in used:
        px = left + patch["measured_luminance"] * (right - left)
        py = bottom - patch["target"] * (bottom - top)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colors[patch["status"]]}"/>')
    parts.append('<text x="%d" y="%d" fill="#9fb3a9" font-size="10">measured →</text>' % (left, height - 8))
    parts.append('<text x="6" y="%d" fill="#9fb3a9" font-size="10" transform="rotate(-90 10 %d)">card →</text>' % (top + 40, top + 40))
    parts.append("</svg>")
    return "".join(parts)


def tone_for(tile, frame):
    """Return the stored LUT when it still describes this frame's plane."""
    record = tile.get("tone_card")
    if not record or frame.photo["id"] != record["photo_id"]:
        return None
    if record["plane_size"] != list(frame.extent) or record["plane_corners"] != tile["plane"]["corners"]:
        return None
    region_ids = {region["id"] for region in tile["regions"]}
    if any(patch["region_id"] not in region_ids for patch in record["patches"]):
        return None
    return record


def corrected_with_tone(source_linear, lut):
    """Apply the scalar curve to an alternate (e.g. neutral-corrected) image."""
    return apply_tone_lut(np.asarray(source_linear, dtype=np.float32), np.asarray(lut, dtype=np.float32))


def tone_basis(record):
    if record is None:
        return None
    return {
        "curve": "tone-card-luminance-lut",
        "anchors": record["anchors"],
        "lut_levels": record["lut_levels"],
        "photo_id": record["photo_id"],
    }


def _region_readout(blobs, tile, photo_id, region, lut):
    frame = specimen_frame(blobs, tile, photo_id)
    domain = region_domain(frame.extent, tile, region)
    support = domain & frame.valid & ~frame.clipped
    if int(np.count_nonzero(support)) < MIN_PATCH_SAMPLES:
        return None
    corrected = apply_tone_lut(frame.linear, np.asarray(lut, dtype=np.float32))
    lab = linear_lab(corrected)[support]
    center = np.median(lab, axis=0)
    luma = luminance(corrected)[support]
    return {
        "median_lab": float_list(center, 5),
        "median_luminance": round(float(np.median(luma)), 6),
        "supported_pixels": int(np.count_nonzero(support)),
        "swatch": swatch(np.median(corrected[support], axis=0)),
    }


def _curve_identity(record):
    """Stable identity of a fitted curve (its LUT contents)."""
    if record is None:
        return None
    return {
        "photo_id": record["photo_id"],
        "lut": record["lut"],
        "lut_levels": record["lut_levels"],
    }


def ruler_readouts(store, tile, frame, region, tone, reference):
    """Re-read current and archived reference regions on their tone curves.

    The two readings are only comparable when both specimens share the exact
    same fitted curve.  This never mutates the archived reference record; when
    the archived source image or sampling region cannot be verified, the
    reference side is flagged instead of being forced onto the current ruler.
    """
    if tone is None:
        return None, None
    current_section = None
    if region is not None:
        current = _region_readout(store.blobs, tile, frame.photo["id"], region, tone["lut"])
        if current is not None:
            current_section = {
                **current,
                "region_name": region["name"],
                "curve_photo_id": tone["photo_id"],
            }

    reference_section = None
    ref_source = reference.get("source") or {}
    ref_region = reference.get("region")
    ref_tile_id = (reference.get("specimen") or {}).get("id")
    source_available = False
    same_curve = False
    if ref_tile_id and ref_region and ref_source.get("id"):
        try:
            ref_tile = store.tile(ref_tile_id)
            ref_photo = next((p for p in ref_tile["photos"] if p["id"] == ref_source["id"]), None)
            live_region = next((r for r in ref_tile["regions"] if r["id"] == ref_region["id"]), None)
            if ref_photo is not None and live_region is not None and live_region["vertices"] == ref_region["vertices"]:
                ref_frame = specimen_frame(store.blobs, ref_tile, ref_photo["id"])
                ref_tone = tone_for(ref_tile, ref_frame)
                if ref_tone is not None:
                    same_curve = ref_tone["lut"] == tone["lut"]
                    reading = _region_readout(store.blobs, ref_tile, ref_photo["id"], live_region, ref_tone["lut"])
                    if reading is not None:
                        reference_section = {
                            **reading,
                            "region_name": ref_region["name"],
                            "curve_photo_id": ref_tone["photo_id"],
                            "same_curve": same_curve,
                            "source_available": True,
                            "archived_median_lab": reference["fingerprint"]["median_lab"],
                        }
                        source_available = True
        except (InputError, KeyError):
            source_available = False
    if reference_section is None:
        reference_section = {
            "source_available": source_available,
            "same_curve": False,
            "archived_median_lab": reference["fingerprint"]["median_lab"],
            "archived_swatch": reference["fingerprint"]["swatch"],
            "curve_photo_id": tone["photo_id"],
        }
    return current_section, reference_section
