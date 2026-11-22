"""Measure the visible silhouette of a single mounted leaf."""

import numpy as np
from PIL import Image
from scipy import ndimage

from .art import base_rgba
from .color import linear_lab
from .coverage import MIN_SEPARATION, classify_fraction, reference_region, sample_chip
from .geometry import proper_axis, region_domain
from .model import InputError


CONNECTIVITY = ndimage.generate_binary_structure(2, 1)


def classify_leaf(tile, frame, domain, linear, options):
    available = region_domain(frame.extent, tile)
    lab = linear_lab(linear)
    backing = reference_region(tile, "bare", options["bare_region"])
    leaf = reference_region(tile, "glazed", options["glazed_region"])
    undercoat, _, backing_info = sample_chip(frame, lab, available, backing)
    foliage, _, foliage_info = sample_chip(frame, lab, available, leaf)
    separation = float(np.linalg.norm(foliage - undercoat))
    if separation < MIN_SEPARATION:
        raise InputError("The leaf and backing references look too similar. Mark a clean patch of each.")
    axis = foliage - undercoat
    projection = np.einsum("...i,i->...", lab - undercoat, axis) / (separation ** 2)
    residual = np.linalg.norm(lab - undercoat - projection[..., None] * axis, axis=2)
    fraction = np.clip(projection, 0, 1)
    support = domain & frame.valid & ~frame.clipped
    labels = classify_fraction(
        fraction, residual, support,
        ambiguity=options["uncertainty"], residual_limit=max(8, separation * .52),
    )
    return labels, {"backing": backing_info, "leaf": foliage_info, "separation": round(separation, 3)}


def principal_blade(classes, minimum):
    regions, _ = ndimage.label(classes == 2, structure=CONNECTIVITY)
    pixels = np.bincount(regions.ravel())
    eligible = np.flatnonzero(pixels >= minimum)
    eligible = eligible[eligible > 0]
    if not eligible.size:
        raise InputError("No connected leaf silhouette met the requested size. Check the focus area and reference patches.")
    principal = eligible[np.argmax(pixels[eligible])]
    return regions == principal, len(eligible) - 1


def visible_edge(blade, known_backing, scale):
    # Count only interfaces backed by observed paper and report unresolved interfaces separately.
    pad_leaf = np.pad(blade, 1)
    pad_paper = np.pad(known_backing, 1)
    middle = pad_leaf[1:-1, 1:-1]
    neighbors = (
        (pad_leaf[1:-1, :-2], pad_paper[1:-1, :-2], scale[1]),
        (pad_leaf[1:-1, 2:], pad_paper[1:-1, 2:], scale[1]),
        (pad_leaf[:-2, 1:-1], pad_paper[:-2, 1:-1], scale[0]),
        (pad_leaf[2:, 1:-1], pad_paper[2:, 1:-1], scale[0]),
    )
    visible_pixels = np.zeros(blade.shape, dtype=bool)
    unknown_pixels = np.zeros(blade.shape, dtype=bool)
    measured_length = 0.0
    for other_leaf, paper, length in neighbors:
        exposed = middle & ~other_leaf
        confirmed = exposed & paper
        measured_length += np.count_nonzero(confirmed) * length
        visible_pixels |= confirmed
        unknown_pixels |= exposed & ~paper
    return round(measured_length, 4), visible_pixels, unknown_pixels


def silhouette_overlay(frame, blade, visible, unresolved, unknown):
    overlay = np.asarray(base_rgba(frame)).copy()
    overlay[blade, :3] = np.rint(overlay[blade, :3] * .68 + np.array([104, 183, 117]) * .32).astype(np.uint8)
    overlay[visible, :3] = (109, 233, 150)
    overlay[unresolved, :3] = (244, 170, 79)
    y, x = np.indices(blade.shape, sparse=True)
    overlay[unknown & ((x + y) % 9 < 2), :3] = (233, 208, 170)
    return Image.fromarray(overlay)


def measure_silhouette(tile, frame, domain, linear, options):
    classes, references = classify_leaf(tile, frame, domain, linear, options)
    blade, other = principal_blade(classes, options["minimum_pixels"])
    scale = frame.scale or (1, 1)
    visible_length, visible, unresolved = visible_edge(blade, classes == 0, scale)
    unknown = domain & (classes < 0)
    marked_count = int(np.count_nonzero(blade))
    rows, cols = np.nonzero(blade)
    unit = "mm" if frame.scale else "px"
    readings = {
        "visible_pixels": marked_count,
        "visible_area": round(marked_count * scale[0] * scale[1], 4),
        "units": unit,
        "visible_edge": visible_length,
        "observed_edge_pixels": int(np.count_nonzero(visible)),
        "unconfirmed_edge_pixels": int(np.count_nonzero(unresolved)),
        "mixed_pixels": int(np.count_nonzero(classes == 1)),
        "unobserved_pixels": int(np.count_nonzero(unknown)),
        "other_components": other,
        "scale_bbox": [round((cols.max() - cols.min() + 1) * scale[0], 3),
                       round((rows.max() - rows.min() + 1) * scale[1], 3)],
        "axis": proper_axis(blade, frame.scale),
        "references": references,
        "note": "One largest connected blade is reported. Green edge segments meet visible backing. Amber segments abut ambiguous or unobserved samples and are omitted from the edge length. Length is a pixel-grid estimate.",
    }
    return readings, silhouette_overlay(frame, blade, visible, unresolved, unknown)
