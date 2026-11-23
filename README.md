# FolioLeaf

FolioLeaf is a local image workbench for photographed, mounted leaves. It keeps source exposures beside registered views and retained measurements. It estimates the visible blade and edges against a photographed backing. It does not reconstruct edges hidden by folds, overlap or the edge of a photo.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19984`. Set `FOLIOLEAF_PORT` to select another local port. The named volume holds source photos, measurements and evidence. With Python 3.11 and the pinned dependencies installed, run `./start.sh`.

## Inspect a leaf sheet

1. Load sample leaves or create a specimen and import your own exposures. In **Captures**, mark the four corners of the survey plane, set its physical width and height, and register repeat views using the same landmarks. The sample sheets are synthetic and include paper backing, mounts, an exposure card and several lighting conditions.
2. In **Sampling regions**, mark a blade area, a leaf reference and a backing reference. In **Measure**, run **Leaf silhouette**. Green marks parts of the largest connected blade edge with observed backing. Amber marks an uncertain edge. The printed area measures visible blade pixels and the edge length is a grid approximation. For shape widths, use **Leaf width along axis** along two points on the blade with suitable backing and leaf references.
3. You can also run color variation, blade color coverage, dark or pale fleck counts, spot review, opposed raking light, polarized glare, fine-line networks, fine band spacing, a line scan, mount seam scan, and survey margin color. Use **Geometry** to save a traced length or area. Use **Specimen comparisons** to compare retained measurements.
4. In **Evidence**, pin measurements and download an evidence ZIP, optionally including original photos. **Received evidence** reads a provided evidence ZIP and checks its declared members and digests.

## Batch gray-scale tone curve

When a scan lifts a middle gray above the adjacent light grays, mark one sampling region per usable cell of a photographed gray scale (dark, middle, and the two light grays), enter each cell's paper-card reading, and leave any glare-damaged cell unused. FolioLeaf fits one global, monotone non-decreasing luminance curve that scales the three RGB channels together. It therefore never inverts the original bright/dark order, maps equal input luminance to one output, keeps each pixel's color proportions, and trims the common factor instead of clipping a channel to the ceiling. The preview beside the original image reports which cells are held on their card reading, which represent the accepted error tier the monotone constraint makes unavoidable, and which were rejected for glare.

In **Reference match**, the current image and the approved reference's original sampling region can each be re-read on the same fitted curve. The reference's archived reading is never overwritten. When the archived source image or sampling region cannot be verified, or the two specimens do not share the same curve, the result is explicitly marked as not comparable on this ruler.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. Each specimen has up to ten captures and 24 regions. The registered plane is limited to 1,800 pixels on its longest side. The features are projections in that plane. Measurements cannot establish species identity or physical thickness. One connected, distinguishable blade is measured at a time.

FolioLeaf reuses the photographed-surface registration, image analysis, local measurement archive and evidence engine from the local CoreSection project. It adds a leaf-sheet scene generator, the measured single-blade silhouette with observed/ambiguous edges, and a mounting workflow using leaf and backing references.
