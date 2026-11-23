from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage

from .geometry import DEFAULT_CORNERS, canvas_dimensions, homography
from .photos import linear_to_srgb, png_bytes, srgb_to_linear
from .store import document, event, make_id
from .model import timestamp
from .photos import store_capture


DEMO_KEY = "folioleaf.training.v2"
PLANE_SIZE = (620, 472)
PHOTO_SIZE = (766, 598)
DEMO_ROLES = ("diffuse", "parallel", "cross", "rake-left", "rake-right")
REPEAT_SEED = 562
REPEAT_ROLE = "revisit"
POSES = {
    "diffuse": [[0.092, 0.086], [0.911, 0.112], [0.911, 0.912], [0.081, 0.889]],
    "parallel": [[0.122, 0.107], [0.910, 0.084], [0.925, 0.871], [0.101, 0.916]],
    "cross": [[0.079, 0.101], [0.908, 0.119], [0.877, 0.916], [0.081, 0.909]],
    "rake-left": [[0.089, 0.094], [0.894, 0.088], [0.902, 0.905], [0.064, 0.911]],
    "rake-right": [[0.094, 0.095], [0.913, 0.094], [0.926, 0.906], [0.112, 0.893]],
    REPEAT_ROLE: [[0.087, 0.090], [0.905, 0.104], [0.900, 0.908], [0.076, 0.895]],
}
# Stable vein crossings and an edge notch the user can recognize on both days.
# Day-two positions are derived from repeat_drift once that function is defined.
_REPEAT_ANCHORS = [
    (0.286, 0.478), (0.348, 0.462), (0.438, 0.548), (0.522, 0.430), (0.610, 0.520),
    (0.692, 0.455), (0.766, 0.520), (0.568, 0.648), (0.472, 0.612), (0.402, 0.372),
    (0.644, 0.386), (0.386, 0.606), (0.582, 0.352),
]
REPEAT_LANDMARKS = []
# A fold line hides blade tissue in both visits, on different patches.
REPEAT_EXCLUSIONS = [
    {"day": "first", "center": [0.628, 0.585], "radius": 0.034},
    {"day": "second", "center": [0.651, 0.566], "radius": 0.030},
]


@dataclass(frozen=True)
class DemoSpecimen:
    seed: int
    name: str
    body: str
    glaze: str
    firing: str
    batch: str
    colors: tuple


SERIES = (
    DemoSpecimen(173, "Pressed oval · LF07", "Broad oval form", "Dry mounted", "Diffuse survey", "Folio A",
                ((50, 56, 29), (111, 126, 58), (155, 163, 82))),
    DemoSpecimen(284, "Serrated blade · LF12", "Fine serrated form", "Paper strips", "Diffuse survey", "Folio B",
                ((47, 64, 41), (106, 129, 73), (160, 169, 104))),
    DemoSpecimen(391, "Lobed blade · LF16", "Broad lobed form", "Dry mounted", "Repeat survey", "Folio A",
                ((86, 58, 35), (148, 110, 53), (182, 145, 82))),
    DemoSpecimen(475, "Narrow blade · LF21", "Narrow pointed form", "Paper strips", "Repeat survey", "Folio C",
                ((61, 57, 34), (129, 115, 66), (178, 151, 85))),
)


def random_field(generator, size, y_space, x_space):
    height, width = size
    rows, cols = int(np.ceil(height / y_space)) + 4, int(np.ceil(width / x_space)) + 4
    coarse = generator.normal(size=(rows, cols)).astype(np.float32)
    y, x = np.indices(size, dtype=np.float32)
    field = ndimage.map_coordinates(coarse, (y / y_space + 1, x / x_space + 1), order=3)
    return field.astype(np.float32)


@dataclass(frozen=True)
class ProofField:
    diffuse: np.ndarray
    height: np.ndarray
    ink_fraction: np.ndarray
    keyline_mask: np.ndarray
    roughness: np.ndarray
    u: np.ndarray
    v: np.ndarray


def surface_palette(colors, values):
    linear_colors = srgb_to_linear(np.array(colors, dtype=np.float32) / 255)
    stops = (-1.2, 0.05, 1.25)
    return np.stack(
        [np.interp(values, stops, linear_colors[:, channel]) for channel in range(3)],
        axis=2,
    ).astype(np.float32)


def neutral_card(diffuse, roughness, x, y):
    sample = (x >= 0.07) & (x <= 0.165) & (y >= 0.844) & (y <= 0.961)
    border = (x >= 0.063) & (x <= 0.172) & (y >= 0.835) & (y <= 0.968) & ~sample
    diffuse[sample] = np.array([0.194, 0.172, 0.155])
    diffuse[border] *= 0.70
    roughness[sample] = 0.76


def folio_marks(x, y, rng):
    lower_tag = (x > .70) & (x < .94) & (y > .785) & (y < .943)
    lines = ((np.floor(y * 235).astype(int) % 6) == 0) & lower_tag & (x > .717) & (x < .919)
    scale = (x > .48) & (x < .66) & (y > .886) & (y < .938)
    dark_ticks = (np.floor((x - .48) * 225).astype(int) % 2 == 0) & scale
    folio_number = (x > .763) & (x < .875) & (y > .810) & (y < .818)
    tape_1 = (x > .250) & (x < .276) & (y > .372) & (y < .517)
    tape_2 = (x > .613) & (x < .629) & (y > .475) & (y < .635)
    sheet = np.broadcast_to(np.array([.68, .62, .49], dtype=np.float32), (*x.shape, 3)).copy()
    paper = rng.normal(0, .014, x.shape).astype(np.float32)
    warm = (.027 * np.sin(4*x+3*y) + .042*x) * (1 - y)
    sheet += (paper + warm)[..., None]
    sheet[lower_tag | scale] += [.12, .13, .14]
    sheet[lines | dark_ticks | folio_number] = [.14, .15, .12]
    return sheet, tape_1 | tape_2


def leaf_form(specimen, x, y, mottling):
    center = .51 + .027 * np.sin(5 * x) + .017 * x
    length = (x - .201) / .655
    radial = np.clip(length, 0, 1)
    profile = np.maximum(np.sin(np.pi * radial), 0) ** .79
    seed_styles = {173: (.202, .006, 38), 284: (.187, .037, 93),
                   391: (.222, .19, 37), 475: (.119, .035, 57), REPEAT_SEED: (.205, .010, 41)}
    width, serration, wave = seed_styles[specimen.seed]
    left_scale = 1 + serration * np.cos(wave * x + .46)
    right_scale = 1 + serration * np.cos(wave * x - .72)
    reach = width * profile * np.where(y < center, left_scale, right_scale)
    reach = np.maximum(0, reach + .0027 * mottling * profile)
    side = np.abs(y - center)
    blade = (length > 0) & (length < 1) & (side <= reach)
    stem = (x > .142) & (x < .37) & (np.abs(y - center + .08 * np.maximum(.22 - x, 0)) < .0051)
    # The alternating secondary ribs follow the taper and stop at the outer blade.
    chevrons = np.sin((x - .66 * side - 1.8 * side**2) * 98 - 7 * side)
    ribs = np.exp(-17 * np.abs(chevrons)) * blade * np.exp(-4.6 * side)
    spine = np.exp(-((y - center) / .0062)**2) * (blade | stem)
    return blade | stem, ribs, spine, reach, center


def bake_field(specimen):
    width, height = PLANE_SIZE
    shape = height, width
    generator = np.random.default_rng(specimen.seed)
    y, x = np.indices(shape, dtype=np.float32)
    y /= height
    x /= width
    broad = random_field(generator, shape, 47, 68)
    mottling = random_field(generator, shape, 11, 19)
    fibers = generator.normal(size=shape).astype(np.float32)
    sheet, tape = folio_marks(x, y, generator)
    blade, ribs, spine, reach, center = leaf_form(specimen, x, y, mottling)
    surface = .40 * broad + .29 * mottling + .10 * fibers
    surface += .59 * spine + .35 * ribs
    surface -= .31 * np.clip(abs(y - center) / np.maximum(reach, .01), 0, 1)
    leaf = surface_palette(specimen.colors, surface)
    # Isolated material contrast and an edge notch remain distinguishable from the sheet.
    stress = (
        np.exp(-(((x - .433) / .061)**2 + ((y - .593) / .029)**2))
        + .70 * np.exp(-(((x - .66) / .093)**2 + ((y - .422) / .046)**2))
    )
    leaf *= 1 - .28 * np.clip(stress, 0, 1)[..., None]
    for index in range(28):
        cx, cy = generator.uniform((.27, .383), (.766, .622))
        rx, ry = generator.uniform(.002, .008), generator.uniform(.002, .010)
        spot = ((x - cx) / rx)**2 + ((y - cy) / ry)**2 < 1 + .14*mottling
        leaf[spot] *= .54 + .19 * ((index % 4) / 3)
    notch = (((x - .568) / .034)**2 + ((y - .687) / .019)**2 < 1) & (specimen.seed % 2 == 0)
    blade &= ~notch
    diffuse = np.where(blade[..., None], leaf, sheet)
    tape_clip = tape & blade
    diffuse[tape_clip] = .62 * diffuse[tape_clip] + .38 * sheet[tape_clip]
    cross = np.zeros(shape, dtype=bool)
    for cx in (.086, .919):
        for cy in (.171, .797):
            cross |= ((abs(x - cx) < .014) & (abs(y - cy) < .0025))
            cross |= ((abs(x - cx) < .002) & (abs(y - cy) < .019))
    diffuse[cross] = [.055, .059, .052]
    relief = (.032 * spine + .023 * ribs + .022 * broad + .008 * mottling) * blade
    roughness = np.where(blade, np.clip(.66 + .12 * mottling + .11 * ribs, .30, .93), .82)
    neutral_card(diffuse, roughness, x, y)
    return ProofField(np.clip(diffuse, 0, 1), relief,
                      np.where(blade, .51 + .25 * ribs, .05),
                      (tape_clip | cross).astype(np.float32), roughness, x, y)


def gloss_reflection(field, role):
    x, y = field.u, field.v
    hotspot = np.exp(-(((x - 0.64 + 0.17 * y) / 0.21) ** 2 + ((y - 0.38) / 0.25) ** 2))
    elongated = np.exp(-(((x - 0.64 - 0.38 * (y - 0.3)) / 0.075) ** 2 + ((y - 0.49) / 0.31) ** 2))
    low = (1 - field.roughness) * field.ink_fraction * (0.5 + 0.5 * np.sin(8 * x + 13 * y) ** 2)
    intensity = 0.079 * low if role == "diffuse" else 0.34 * low + 0.24 * hotspot + 0.12 * elongated
    if role == "cross":
        intensity = (0.009 + hotspot * 0.018) * low
    return intensity * (1 - field.keyline_mask * 0.58)


def light_field(field, role):
    x, y = field.u, field.v
    dy, dx = np.gradient(0.92 * field.height)
    broad = 0.93 + 0.13 * (1 - y) - 0.08 * (x - 0.5) ** 2
    if role in ("rake-left", "rake-right"):
        direction = 1 if role == "rake-left" else -1
        side = 0.91 + direction * 0.13 * (0.5 - x)
        shading = np.clip(0.66 + (dx * direction - dy * 0.23) * 3.9, 0.26, 1.32)
        rgb = field.diffuse * (broad * side * shading)[:, :, None]
    else:
        side = np.clip(0.98 + dx * 0.38 - dy * 0.31, 0.64, 1.2)
        reflection = gloss_reflection(field, role)
        exposure = 0.88 if role == "cross" else 1.09 if role == "parallel" else 1.0
        rgb = field.diffuse * (broad * side * exposure)[:, :, None]
        rgb += reflection[:, :, None] * np.array([0.69, 0.79, 0.82], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def desk_background(width, height, seed):
    generator = np.random.default_rng(seed + 500)
    noise = random_field(generator, (height, width), 21, 30) * 0.018
    y, x = np.indices((height, width), dtype=np.float32)
    grad = 0.12 + 0.19 * np.exp(-((x / width - 0.24) ** 2 + (y / height - 0.10) ** 2) * 2.4)
    grain = generator.normal(size=(height, width)).astype(np.float32) * 0.008
    neutral = np.maximum(grad + grain + noise, 0.035)
    return neutral[:, :, None] * np.array([0.50, 0.57, 0.58], dtype=np.float32)


def surface_alpha(x, y):
    horizontal = np.abs(x - 0.5) - 0.461
    vertical = np.abs(y - 0.5) - 0.452
    corner = np.hypot(np.maximum(horizontal, 0), np.maximum(vertical, 0))
    nearest = np.minimum(np.maximum(horizontal, vertical), 0)
    sdf = corner + nearest - 0.035
    return np.clip(0.5 - sdf * 410, 0, 1)


def warp_to_photo(array, corners, photo_size, seed):
    width, height = photo_size
    photo_to_tile = np.linalg.inv(homography(DEFAULT_CORNERS, corners))
    photo_y, photo_x = np.indices((height, width), dtype=np.float32)
    photo_y = (photo_y + 0.5) / height
    photo_x = (photo_x + 0.5) / width
    denominator = photo_to_tile[2, 0] * photo_x + photo_to_tile[2, 1] * photo_y + photo_to_tile[2, 2]
    tile_u = (photo_to_tile[0, 0] * photo_x + photo_to_tile[0, 1] * photo_y + photo_to_tile[0, 2]) / denominator
    tile_v = (photo_to_tile[1, 0] * photo_x + photo_to_tile[1, 1] * photo_y + photo_to_tile[1, 2]) / denominator
    tile_h, tile_w = array.shape[:2]
    coords = (tile_v * tile_h - 0.5, tile_u * tile_w - 0.5)
    tile_image = np.stack(
        [ndimage.map_coordinates(array[:, :, component], coords, order=1, mode="nearest") for component in range(3)],
        axis=2,
    )
    alpha = surface_alpha(tile_u, tile_v)
    shadow = ndimage.shift(ndimage.gaussian_filter(alpha, 7), (9, 5), order=1, mode="constant")
    stage = desk_background(width, height, seed)
    stage *= (1 - 0.61 * shadow)[:, :, None]
    rim = np.clip(alpha - ndimage.minimum_filter(alpha, size=8), 0, 1)
    tile_image *= 1 - 0.24 * rim[:, :, None]
    mixed = stage * (1 - alpha[:, :, None]) + tile_image * alpha[:, :, None]
    return linear_to_srgb(mixed)


def render_photo(field, role, specimen):
    linear = light_field(field, role)
    image = warp_to_photo(linear, POSES[role], PHOTO_SIZE, specimen.seed)
    array = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


def rectangle(name, purpose, left, top, right, bottom):
    return {
        "id": make_id(), "name": name, "purpose": purpose,
        "vertices": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


def training_regions():
    return [
        rectangle("Full blade window", "inspection", 0.12, 0.24, 0.88, 0.766),
        rectangle("Leaf color reference", "glazed", 0.42, 0.470, 0.47, 0.499),
        rectangle("Backing color reference", "bare", 0.33, 0.262, 0.39, 0.295),
        rectangle("Neutral card", "neutral", 0.086, 0.858, 0.15, 0.943),
        rectangle("Image scale", "ignore", 0.478, 0.883, 0.663, 0.941),
        rectangle("Central blade window", "inspection", .232, .272, .786, .625),
    ]


def repeat_drift(x, y):
    """Smooth local re-flattening: the blade opens out across its middle, fading
    to nothing at the anchored tip and base so their rims do not scale."""
    growth = np.exp(-(((x - 0.50) / 0.26) ** 2 + ((y - 0.52) / 0.30) ** 2))
    envelope = np.clip((x - 0.24) / 0.10, 0, 1) * np.clip((0.82 - x) / 0.10, 0, 1)
    magnitude = growth * envelope
    return np.stack([0.016 * magnitude, 0.022 * magnitude * np.sin(2.4 * x + 0.7)], axis=-1)


REPEAT_LANDMARKS[:] = [
    ([x, y], [round(float(x + repeat_drift(np.array(x), np.array(y))[0]), 5),
              round(float(y + repeat_drift(np.array(x), np.array(y))[1]), 5)])
    for x, y in _REPEAT_ANCHORS
]


def render_repeat_photo(field, specimen):
    height, width = field.diffuse.shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    nx = (x + 0.5) / width
    ny = (y + 0.5) / height
    # Inverse warp: a day-two pixel at (nx, ny) samples the day-one tissue at
    # (nx, ny) minus the local re-flattening drift. This leaves no stretched
    # edge holes, unlike a forward scatter of the original pixels.
    drift = repeat_drift(nx, ny)
    source_x = np.clip(nx - drift[:, :, 0], 0, 1)
    source_y = np.clip(ny - drift[:, :, 1], 0, 1)
    coords = (source_y * height - 0.5, source_x * width - 0.5)
    moved = np.stack(
        [ndimage.map_coordinates(field.diffuse[:, :, c], coords, order=1, mode="nearest") for c in range(3)],
        axis=2,
    )
    # Day-two tissue now sits at the drifted (forward) positions. The old stress
    # mark fades where it now lands, and a small new dark mark appears beside it.
    old_center = np.array((0.433, 0.593)) + repeat_drift(np.array(0.433), np.array(0.593))
    old = np.exp(-(((nx - old_center[0]) / .062) ** 2 + ((ny - old_center[1]) / .032) ** 2))
    moved += (0.46 * old[..., None] * np.array([0.11, 0.10, 0.045], np.float32))
    fresh_center = np.array((0.495, 0.548))
    fresh = np.exp(-(((nx - fresh_center[0]) / .016) ** 2 + ((ny - fresh_center[1]) / .014) ** 2))
    moved *= 1 - 0.42 * fresh[..., None]
    repeat_field = ProofField(
        np.clip(moved, 0, 1), field.height, field.ink_fraction, field.keyline_mask,
        field.roughness, field.u, field.v,
    )
    # Reuse the diffuse lighting field so visit differences are tissue changes, not relighting.
    linear = light_field(repeat_field, "diffuse")
    image = warp_to_photo(linear, POSES[REPEAT_ROLE], PHOTO_SIZE, specimen.seed)
    array = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


REPEAT_SPECIMEN = DemoSpecimen(REPEAT_SEED, "Re-pressed blade · LF27", "Broad oval form", "Dry mounted",
                               "Three-day revisit", "Folio D",
                               ((56, 52, 30), (118, 122, 57), (162, 158, 84)))


def repeat_training_specimen(store):
    specimen = REPEAT_SPECIMEN
    field = bake_field(specimen)
    photos = []
    for role in DEMO_ROLES:
        photo = store_capture(
            store.blobs, png_bytes(render_photo(field, role, specimen)), role,
            f"{specimen.name.lower().replace(' ', '-')}-{role}.png",
            "Synthetic mounted pressed-leaf form with backing and registration targets.",
            make_id(), synthetic=True,
        )
        photos.append(photo)
    revisit = store_capture(
        store.blobs, png_bytes(render_repeat_photo(field, specimen)), REPEAT_ROLE,
        f"{specimen.name.lower().replace(' ', '-')}-revisit.png",
        "Synthetic re-photograph after three days: the blade re-flattens locally, an old mark fades and a new mark appears.",
        make_id(), synthetic=True,
    )
    photos.append(revisit)
    base = photos[0]
    size = canvas_dimensions(base, POSES["diffuse"], 112, 84)
    regions = training_regions()
    regions.append(rectangle("Watched old mark", "inspection", 0.385, 0.548, 0.472, 0.632))
    return {
        "name": specimen.name,
        "batch": specimen.batch,
        "body": specimen.body,
        "glaze": specimen.glaze,
        "firing": specimen.firing,
        "notes": "Synthetic leaf photographed twice while flattening. Day-one and revisit views share the survey plane; the blade itself moves locally.",
        "width_mm": 112,
        "height_mm": 84,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": POSES["diffuse"], "size": list(size)},
        "regions": regions,
        "registrations": {
            photo["id"]: {"base_photo_id": base["id"], "corners": POSES[photo["role"]], "support_percent": 100}
            for photo in photos[1:]
        },
    }


def training_specimen(store, specimen):
    field = bake_field(specimen)
    photos = []
    for role in DEMO_ROLES:
        image = render_photo(field, role, specimen)
        photo = store_capture(
            store.blobs, png_bytes(image), role,
            f"{specimen.name.lower().replace(' ', '-')}-{role}.png",
            "Synthetic mounted pressed-leaf form with backing, tape, registration targets and color card.",
            make_id(),
            synthetic=True,
        )
        photos.append(photo)
    base = photos[0]
    size = canvas_dimensions(base, POSES["diffuse"], 112, 84)
    return {
        "name": specimen.name,
        "batch": specimen.batch,
        "body": specimen.body,
        "glaze": specimen.glaze,
        "firing": specimen.firing,
        "notes": "Synthetic mounted leaf. The vein pattern and the backing are generated in a common physical plane across each view.",
        "width_mm": 112,
        "height_mm": 84,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": POSES["diffuse"], "size": list(size)},
        "regions": training_regions(),
        "registrations": {
            photo["id"]: {"base_photo_id": base["id"], "corners": POSES[photo["role"]], "support_percent": 100}
            for photo in photos[1:]
        },
    }


def load_training(store):
    existing = store.seeded_tiles(DEMO_KEY)
    if existing:
        return existing, False
    rows = [training_specimen(store, specimen) for specimen in SERIES]
    repeat_row = repeat_training_specimen(store)
    with store.transaction(write=True) as conn:
        for index, row in enumerate(rows):
            rows[index] = store._insert_tile(conn, row, "Loaded synthetic training material")
        repeat_tile = store._insert_tile(conn, repeat_row, "Loaded synthetic training material")
        documents, pictures, _ = build_seeded_repeat(store, repeat_tile)
        record = {**documents, "id": make_id(), "created_at": timestamp(), "version": 1, "pictures": pictures}
        conn.execute(
            "INSERT INTO repeats(id,tile_id,document,created_at) VALUES (?,?,?,?)",
            (record["id"], record["tile_id"], document(record), record["created_at"]),
        )
        event(conn, repeat_tile["id"], "repeat", f"Saved repeat-visit comparison: {record['title']}")
        stored = [tile["id"] for tile in [*rows, repeat_tile]]
        conn.execute("INSERT INTO settings(key,value) VALUES (?,?)", (DEMO_KEY, document(stored)))
    rows.append(repeat_tile)
    return rows, True


def build_seeded_repeat(store, tile):
    first_id = tile["plane"]["photo_id"]
    second_id = next(photo["id"] for photo in tile["photos"] if photo["role"] == REPEAT_ROLE)
    payload = {
        "title": "Three-day flattening revisit",
        "first_photo_id": first_id,
        "second_photo_id": second_id,
        "landmarks": [{"first": one, "second": two} for one, two in REPEAT_LANDMARKS],
        "exclusions": REPEAT_EXCLUSIONS,
    }
    from .repeat_visit import build_repeat
    return build_repeat(store, tile, payload)


@lru_cache(maxsize=1)
def preview_photo():
    specimen = SERIES[0]
    rendered = render_photo(bake_field(specimen), "diffuse", specimen)
    return png_bytes(rendered)
