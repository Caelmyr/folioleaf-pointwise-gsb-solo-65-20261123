# FolioLeaf

FolioLeaf is a local image workbench for photographed, mounted leaves. It keeps source exposures beside registered views and retained measurements. It estimates the visible blade and edges against a photographed backing. It does not reconstruct edges hidden by folds, overlap or the edge of a photo.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19984`. Set `FOLIOLEAF_PORT` to select another local port. The named volume holds source photos, measurements and evidence. With Python 3.11 and the pinned dependencies installed, run `./start.sh`.

## Inspect a leaf sheet

1. Load sample leaves or create a specimen and import your own exposures. In **Captures**, mark the four corners of the survey plane, set its physical width and height, and register repeat views using the same landmarks. The sample sheets are synthetic and include paper backing, mounts, an exposure card and several lighting conditions.
2. In **Sampling regions**, mark a blade area, a leaf reference and a backing reference. In **Measure**, run **Leaf silhouette**. Green marks parts of the largest connected blade edge with observed backing. Amber marks an uncertain edge. The printed area measures visible blade pixels and the edge length is a grid approximation. For shape widths, use **Leaf width along axis** along two points on the blade with suitable backing and leaf references.
3. You can also run color variation, blade color coverage, dark or pale fleck counts, spot review, opposed raking light, polarized glare, fine-line networks, fine band spacing, a line scan, mount seam scan, and survey margin color. Use **Geometry** to save a traced length or area. Use **Specimen comparisons** to compare retained measurements.
4. **Vein branch map** builds a measured fork on top of the fine-line network. Mark a trusted midrib stretch, two non-adjacent lateral vein tips, and the two candidate junctions near the mounting strip where a real line passes. Each lateral vein is traced on observed lines and adopts exactly one junction; the shared midrib keeps one trunk and anything hidden under the mounting strip stays an open gap. The chosen junction and the trace the rejected junction would have needed are both drawn on the overlay. In **Line scan**, choose *Follow vein* and pick a saved branch map to read color and distance along a lateral vein's actual curved trace. Reading follows the curved vein and stops at the first gap — stations beyond it stay empty, so the scan never bridges in a straight line onto another branch. The ordinary straight line scan remains available.
4. In **Evidence**, pin measurements and download an evidence ZIP, optionally including original photos. **Received evidence** reads a provided evidence ZIP and checks its declared members and digests.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. Each specimen has up to ten captures and 24 regions. The registered plane is limited to 1,800 pixels on its longest side. The features are projections in that plane. Measurements cannot establish species identity or physical thickness. One connected, distinguishable blade is measured at a time.

FolioLeaf reuses the photographed-surface registration, image analysis, local measurement archive and evidence engine from the local CoreSection project. It adds a leaf-sheet scene generator, the measured single-blade silhouette with observed/ambiguous edges, and a mounting workflow using leaf and backing references.
