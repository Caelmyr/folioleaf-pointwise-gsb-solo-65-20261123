"""Service layer for fine-band row analyses and their rechecks."""

from .analysis import capture_provenance, selected_capture, specimen_provenance
from .art import base_rgba, stored_preview
from .bandrow import (
    build_row_metrics,
    cell_spectrum,
    reassign_options,
    resolve_regions,
    select_peaks,
    transect_options,
)
from .color import corrected_linear
from .geometry import region_domain, specimen_frame
from .model import ConflictError, InputError, timestamp


def run_band_row(store, tile_id, payload):
    tile = store.tile(tile_id)
    settings = transect_options(payload)
    photo_id = selected_capture(tile, settings["photo_id"])
    regions = resolve_regions(tile, settings["region_ids"])
    frame = specimen_frame(store.blobs, tile, photo_id)
    linear, _ = corrected_linear(tile, frame, True)
    metrics, picture = build_row_metrics(frame, linear, tile, settings, regions)
    record = {
        "tile_id": tile_id,
        "source_version": tile["version"],
        "kind": "band_row",
        "label": "Fine band row",
        "settings": settings,
        "metrics": metrics,
        "basis": capture_basis(frame),
        "support": row_support(frame, tile, regions),
        "provenance": {
            "analyzer": "band-row-1",
            "specimen": specimen_provenance(tile),
            "photo": capture_provenance(frame),
            "cells": [{"cell": index, "region_id": region["id"], "region_name": region["name"]}
                      for index, region in enumerate(regions, start=1)],
        },
        "pictures": {
            "source": stored_preview(base_rgba(frame, linear), store.blobs),
            "overlay": stored_preview(picture, store.blobs),
        },
        "recheck_of": None,
    }
    return store.save_band_row(record)


def capture_basis(frame):
    return {
        "photo_id": frame.photo["id"],
        "photo_role": frame.photo["role"],
        "working_digest": frame.photo["working_digest"],
        "synthetic": frame.photo["synthetic"],
        "scale": list(frame.scale) if frame.scale else None,
    }


def row_support(frame, tile, regions):
    entries = []
    for index, region in enumerate(regions, start=1):
        area = region_domain(frame.extent, tile, region)
        requested = int(area.sum())
        available = int((area & frame.valid).sum())
        entries.append({
            "cell": index,
            "requested_pixels": requested,
            "available_pixels": available,
            "available_percent": round(100 * available / requested, 2) if requested else 0.0,
        })
    return entries


def recheck_band_row(store, identifier, payload):
    """Re-evaluate a saved row on the same photo and original cells.

    The previous record is never modified: the confirmed reassignment produces
    a new revision that links back to the reading it replaces.
    """
    changes = reassign_options(payload)
    previous = store.band_row(identifier)
    if previous["version"] != changes["version"]:
        raise ConflictError("The row reading changed in another operation. Refresh and retry.")
    tile = store.tile(previous["tile_id"])
    settings = previous["settings"]
    photo_id = selected_capture(tile, settings["photo_id"])
    if photo_id != previous["provenance"]["photo"]["id"]:
        raise InputError("The original row photograph is no longer the selected capture.")
    regions = resolve_regions(tile, settings["region_ids"])
    frame = specimen_frame(store.blobs, tile, photo_id)
    if frame.photo["working_digest"] != previous["provenance"]["photo"]["working_digest"]:
        raise InputError("The row photograph changed since the original reading; a recheck needs the same image.")
    linear, _ = corrected_linear(tile, frame, True)
    cell_count = len(regions)
    if not 1 <= changes["cell"] <= cell_count:
        raise InputError("That cell number is outside this row.")
    cell_index = changes["cell"] - 1
    # Re-derive the current cell peaks before accepting a confirmation.
    domain = region_domain(frame.extent, tile, regions[cell_index])
    spectrum_data = cell_spectrum(frame, linear, domain, settings)
    current_peaks = select_peaks(spectrum_data) if spectrum_data else []
    chosen_peak = next((peak for peak in current_peaks if peak["peak"] == changes["peak"]), None)
    if chosen_peak is None:
        raise InputError("The confirmed peak is not visible in that cell on the current image.")
    forced = {"cell": changes["cell"], "family": changes["family"], "peak": changes["peak"],
              "note": changes["note"], "confirmed_at": timestamp()}
    metrics, picture = build_row_metrics(frame, linear, tile, settings, regions,
                                         forced=forced, revision=previous["version"])
    record = {
        "tile_id": tile["id"],
        "source_version": tile["version"],
        "kind": "band_row",
        "label": "Fine band row · recheck",
        "settings": settings,
        "metrics": metrics,
        "basis": capture_basis(frame),
        "support": row_support(frame, tile, regions),
        "provenance": {
            "analyzer": "band-row-1",
            "specimen": specimen_provenance(tile),
            "photo": capture_provenance(frame),
            "cells": list(previous["provenance"]["cells"]),
        },
        "pictures": {
            "source": stored_preview(base_rgba(frame, linear), store.blobs),
            "overlay": stored_preview(picture, store.blobs),
        },
        "recheck_of": previous["id"],
    }
    return store.save_band_row(record)
