# FolioLeaf

FolioLeaf is a local image workbench for photographed, mounted leaves. It keeps source exposures beside registered views and retained measurements. It estimates the visible blade and edges against a photographed backing. It does not reconstruct edges hidden by folds, overlap or the edge of a photo.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19984`. Set `FOLIOLEAF_PORT` to select another local port. The named volume holds source photos, measurements and evidence. With Python 3.11 and the pinned dependencies installed, run `./start.sh`.

## Inspect a leaf sheet

1. Load sample leaves or create a specimen and import your own exposures. In **Captures**, mark the four corners of the survey plane, set its physical width and height, and register repeat views using the same landmarks. The sample sheets are synthetic and include paper backing, mounts, an exposure card and several lighting conditions.
2. In **Sampling regions**, mark a blade area, a leaf reference and a backing reference. In **Measure**, run **Leaf silhouette**. Green marks parts of the largest connected blade edge with observed backing. Amber marks an uncertain edge. The printed area measures visible blade pixels and the edge length is a grid approximation. For shape widths, use **Leaf width along axis** along two points on the blade with suitable backing and leaf references.
3. **Repeat visit** aligns a later re-photograph of the same flattening leaf onto the first day. Mark matched vein crossings and edge notches on both registered captures, and circle tissue hidden on either day. A thin-plate spline follows the same blade tissue locally; a Jacobian fold check rejects landmark sets that would swap two parallel veins or bridge a broken vein to its nearest segment. Blade pixels are split into strong evidence, uncertain (outside the landmark envelope, far from an anchor, on a vein rim or under local strain) and no evidence; only strong-evidence blade tissue is reported as a color change, with lightened, darkened and hue components listed. The trade-off sheet keeps every alignment choice visible rather than stretching an old mark to keep a region complete. From a saved comparison, open any day-one region in **Geometry** to measure the same physical area on both days, with comparable edges and missing-evidence edges marked separately.
4. You can also run color variation, blade color coverage, dark or pale fleck counts, spot review, opposed raking light, polarized glare, fine-line networks, fine band spacing, a line scan, mount seam scan, and survey margin color. Use **Geometry** to save a traced length or area. Use **Specimen comparisons** to compare retained measurements.
5. In **Evidence**, pin measurements and download an evidence ZIP, optionally including original photos. **Received evidence** reads a provided evidence ZIP and checks its declared members and digests.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. Each specimen has up to ten captures and 24 regions. The registered plane is limited to 1,800 pixels on its longest side. The features are projections in that plane. Measurements cannot establish species identity or physical thickness. One connected, distinguishable blade is measured at a time.

FolioLeaf reuses the photographed-surface registration, image analysis, local measurement archive and evidence engine from the local CoreSection project. It adds a leaf-sheet scene generator, the measured single-blade silhouette with observed/ambiguous edges, and a mounting workflow using leaf and backing references.
