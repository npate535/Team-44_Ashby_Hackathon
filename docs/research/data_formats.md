# Data formats research: BACH, THUNDER split, patient files, Quilt-1M, PathGen, PathMMU

Researched 2026-10-17. I checked BACH by reading the Zenodo zip's central directory with HTTP range reads, then pulling
small members: the CSV, all 10 XMLs, one TIFF header, and the full `A03.svs` (166 MB) for its TIFF directories. The
SVS was deleted afterwards and nothing was committed. Pinned code: THUNDER `MICS-Lab/thunder@94d1df26` (bach.py,
2025-11-11) and eva `kaiko-ai/eva@baf69f47` (bach.py, 2025-04-07). [V] = verified here. [U] = unverified or inferred.

## 1. Official BACH release (Zenodo 3632035, CC BY-NC-ND 4.0)

Record API: https://zenodo.org/api/records/3632035 [V]
| file | bytes | md5 |
|---|---|---|
| `ICIAR2018_BACH_Challenge.zip` | 10,419,148,291 | 8ae1801334aa943c44627c1eef3631b2 |
| `ICIAR2018_BACH_Challenge_TestDataset.zip` (unlabeled test set; not needed) | 2,964,635,883 | 193704e3411f75bdd98d5cb93a2e56c8 |
| `README.txt`, `LICENSE.txt` (citation + licence text only) | 1,771 / 746 | |

Download URL used by THUNDER: `https://zenodo.org/api/records/3632035/files/ICIAR2018_BACH_Challenge.zip/content`
(supports HTTP Range, returns 206). eva uses `https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip`.
Verify with `md5sum`. Required citations are the three papers listed in README.txt, including
Aresta et al., MedIA 2019, doi:10.1016/j.media.2019.05.010.

### 1.1 Zip layout [V] (497 entries; deflate)
```
ICIAR2018_BACH_Challenge/
  Photos/microscopy_ground_truth.csv        # 7,200 B, no header, CRLF: "n001.tif,Normal" (400 rows)
  Photos/{Benign,InSitu,Invasive,Normal}/   # exactly 100 .tif each, plus junk Thumbs.db/.DS_Store
     Benign/b001.tif..b100.tif  InSitu/is001..is100.tif  Invasive/iv001..iv100.tif  Normal/n001..n100.tif
  Photos/.DS_Store
  WSI/01.svs..20.svs                        # 20 unannotated WSIs (no XML)
  WSI/A01.svs..A10.svs + A01.xml..A10.xml   # the 10 annotated WSIs (xml 80 KB-1.3 MB)
  WSI/gt_thumbnails/A01.png..A10.png        # rendered GT masks (A01: 1958x1317 RGB)
  WSI/thumbnails/{01..20,A01..A10}_thumb.png
  WSI/.DS_Store
```
- The zip contains **no patient xlsx** (see §3). Filter photos by `name.endswith(".tif")`.
- CSV order runs Normal, Benign, InSitu, Invasive. Every row agrees with its folder [V]. The folder is the label;
  the CSV is a cross-check.
- gt_thumbnail colours [V]: black = unannotated, red (255,0,0) = Benign, green = InSitu, blue = Invasive. These come
  from the official reader below. Use them as a visual or IoU check of our rasterizer.

### 1.2 Microscopy TIFF gotcha [V]
- All 400 files are 2048x1536, uncompressed RGB, 19 strips, ResolutionUnit = inch, XRes 96 dpi. That dpi value is
  meaningless; the true scale is **0.42 µm/px** (challenge site, THUNDER `mpp: 0.42`).
- **381 files are 16 bits/sample** (18,874,748 B). **19 are 8-bit** (9,437,492 B): b091-b094, is081-is084,
  iv077-iv087. The 16-bit samples checked equal `v8*257`, so the high byte is lossless.
- `PIL.Image.open(p).convert("RGB")` decodes both kinds to 8-bit RGB. Pillow reports mode `RGB` with rawmode
  `RGB;16L`. I verified this on Pillow 12.2 with a synthetic 16-bit RGB TIFF.
- `tifffile` or `cv2.IMREAD_UNCHANGED` return uint16 for most files. Assert `im.mode == "RGB"` and
  `im.size == (2048, 1536)`.

### 1.3 WSIs [V for A03 + XML; U for the other slides]
- Official spec: Leica SCN400 scans, `.svs`, about 200-250 MB each, pixel scale **0.467 µm/px**, sizes vary
  (e.g. 42113x62625) (https://iciar2018-challenge.grand-challenge.org/Dataset/).
- Every XML root says `MicronsPerPixel="0.466667"`.
- `A03.svs` measured from its TIFF IFDs (Aperio Image Library v12.4.0, classic TIFF):
  - IFD0: 57345x38945, tiles 240x240, **compression 33005** (Aperio JPEG2000 RGB, "J2K/KDU Q=30").
  - IFD1: 1024x695 JPEG thumbnail.
  - IFD2: 14336x9736 (downsample 4). IFD3: 3584x2434 (downsample 16).
  - No label or macro image. OpenSlide should report 3 levels with downsamples of about [1, 4, 16]. [U: OpenSlide
    not run]
- **MPP conflict:** the SVS ImageDescription says `AppMag = 20,000000|MPP = 0,500000` (comma decimals), but the XML
  and the organizers say 0.4667.
  - Do not trust `openslide.mpp-x`; the comma may make it absent or wrong [U].
  - Read `MicronsPerPixel` from each XML. Keep `wsi.mpp: 0.467` in config, with fallback 0.4667 for unannotated slides.
  - If 0.5 were true, physical tile sizes would be about 7% off. Record this as a limitation.
- `aperio.OriginalWidth/OriginalHeight` are swapped (38945/57345). Use `slide.dimensions`.
- XML vertices are **level-0 pixel coordinates, X = column, Y = row, origin top-left**. This is [V]: A03's
  X range 6920-46425 only fits the 57345 axis, and Y range 8112-37528 fits under 38945. The official reader doc
  says the same.
- JP2K needs OpenSlide built with OpenJPEG. That is standard in OpenSlide ≥3.4 and in the `openslide-bin` wheels
  [U]. Smoke-test with `read_region` on A03 before the full run.
- WSI patient file (§3): patient k owns `A0k.svs` (labeled) plus two unlabeled slides,
  `{2k-1, 2k}.svs` → 01,02→1; 03,04→2; …; 19,20→10. A10 is patient 10.
  - So **patient-disjoint WSI folds = one annotated WSI per fold**.
  - WSI patient IDs are 1-10 and microscopy IDs are 11-39. That suggests one shared namespace and no overlap [U].

### 1.4 XML schema: Aperio ImageScope XML (not ASAP) [V]
```xml
<Annotations MicronsPerPixel="0.466667">
 <Annotation Id="1" Name="" Type="4" LineColor="65280" ...>   <!-- exactly one Annotation per file -->
  <Attributes/>
  <Regions><RegionAttributeHeaders/>
   <Region Id="1" Type="0" Zoom=".." Length=".." Area=".." LengthMicrons=".." AreaMicrons=".."
           Text="Invasive carcinoma" NegativeROA="0" ...>      <!-- Type 0 = freehand polygon; all NegativeROA=0 -->
    <Attributes/>          <!-- A01 only: <Attribute Name="1" Id="0" Value="Benign"/> and Text="" -->
    <Vertices><Vertex X="5106" Y="5349" Z="0"/>…</Vertices>  <!-- integer px; polygons implicitly closed; ≥83 vertices -->
```
Region labels per file [V]. The raw strings are inconsistent, so normalize them by substring:
| WSI | Benign | InSitu | Invasive | raw strings |
|---|---|---|---|---|
| A01 | 28 | 4 | 8 | `Attribute@Value`: "Benign", "Carcinoma in situ", "Invasive carcinoma" (Text empty) |
| A02 | 6 | 8 | 3 | "Benign", "Carcinoma in situ", "Carcinoma invasive" |
| A03 | 0 | 8 | 2 | "Carcinoma in situ", "Invasive carcinoma" |
| A04 | 0 | 0 | 1 | "Invasive carcinoma" (one region, 1947 vertices) |
| A05 | 10 | 2 | 3 | "Benign", "In situ carcinoma", "Invasive carcinoma" |
| A06 / A07 | 5 / 6 | 0 | 1 / 1 | "Benign", "Invasive carcinoma" |
| A08 | 1 | 7 | 84 | same three strings |
| A09 | 1 | 23 | 5 | same three strings |
| A10 | 0 | 8 | 1 | "In situ carcinoma", "Invasive carcinoma" |

**There is no "Normal" region anywhere.** The site says: "Each image has a corresponding list of labelled
coordinates that enclose benign, in situ carcinoma and invasive carcinoma regions (the remaining tissue is considered
normal and thus is not relevant for performance evaluation)."
Recommendation:
- mask 0 = unannotated, then 1 Benign, 2 InSitu, 3 Invasive.
- Clean tiles come only from 1-3.
- Unannotated tissue becomes `Normal (weak)` only in an ablation, never as clean truth (plan §A9: unannotated ≠
  normal).
- Paint in severity order (Benign → InSitu → Invasive, last wins) so nested or overlapping polygons keep the more
  severe label. Log the overlap pixel counts.

Official reader (`read_xml.py` from the challenge's Dropbox "code" folder,
https://www.dropbox.com/sh/7b5gjddakmm7rlh/AACAwdbT3SSKmInEmFyQAGMAa?dl=1), quoted verbatim [V]:
```python
    regions = root[0][1].findall('Region')
    pixel_spacing = float(root.get('MicronsPerPixel'))
    for r in regions:
        area += [float(r.get('AreaMicrons'))]
        length += [float(r.get('LengthMicrons'))]
        try:
            label = r[0][0].get('Value')
        except:
            label = r.get('Text')
        if 'benign' in label.lower():
            label = 1
        elif 'in situ' in label.lower():
            label = 2
        elif 'invasive' in label.lower():
            label = 3
        ...
        vertices = r[1]
        for v in vertices:
            x = int(v.get('X')); y = int(v.get('Y')); coord += [[x,y]]
# saveImage: colors = [(0,0,0),(255,0,0),(0,255,0),(0,0,255)]; cv2.fillPoly(img, [np.int32(c)], color)  # XML order
```
I found no XML parser for BACH WSIs in eva or THUNDER; both use microscopy only. The official script needs
`scipy.misc` and cv2, so do not import it. Use the stdlib port below.

### 1.5 Implementation (openslide + PIL; Python ≥3.10)
```python
import xml.etree.ElementTree as ET, numpy as np, openslide
from PIL import Image, ImageDraw
CODE = {"Benign": 1, "InSitu": 2, "Invasive": 3}            # 0 = unannotated; names from config
PAINT_ORDER = ("Benign", "InSitu", "Invasive")               # later overwrites earlier

def norm_label(raw: str) -> str:
    t = raw.lower()
    if "benign" in t: return "Benign"
    if "in situ" in t: return "InSitu"
    if "invasive" in t: return "Invasive"
    raise ValueError(f"unknown BACH WSI label {raw!r}")

def parse_bach_xml(path):
    root = ET.parse(path).getroot()
    mpp = float(root.get("MicronsPerPixel"))                 # 0.466667
    regions = []
    for reg in root.iter("Region"):
        a = reg.find("./Attributes/Attribute")               # A01 stores label here; others in Text
        raw = (a.get("Value") if a is not None else "") or reg.get("Text") or ""
        xy = [(float(v.get("X")), float(v.get("Y"))) for v in reg.findall("./Vertices/Vertex")]
        if len(xy) >= 3:
            regions.append({"id": reg.get("Id"), "raw": raw, "label": norm_label(raw), "xy": xy,
                            "area_um2": float(reg.get("AreaMicrons") or 0)})
    return mpp, regions

def label_mask(slide, regions, level):                       # uint8 HxW at `level`
    W0, H0 = slide.dimensions; W, H = slide.level_dimensions[level]
    sx, sy = W / W0, H / H0                                  # exact per-axis scale (14336/57345 != 1/4)
    m = Image.new("L", (W, H), 0); d = ImageDraw.Draw(m)
    for name in PAINT_ORDER:
        for r in regions:
            if r["label"] == name:
                d.polygon([(x * sx, y * sy) for x, y in r["xy"]], fill=CODE[name])
    return np.asarray(m)

def read_rgb(slide, xy0, level, size):                       # RGBA -> RGB on white (transparent = no data)
    t = slide.read_region(xy0, level, size)
    return Image.alpha_composite(Image.new("RGBA", t.size, (255, 255, 255, 255)), t).convert("RGB")

def tissue_mask(slide, level, s_min=20, v_max=235):          # thresholds from config
    hsv = np.asarray(read_rgb(slide, (0, 0), level, slide.level_dimensions[level]).convert("HSV"))
    return (hsv[..., 1] > s_min) & (hsv[..., 2] < v_max)

def labeled_tiles(slide, mask, mask_level, mpp0, tile_px=512, target_mpp=0.42, purity=0.90):
    """Yield (x0, y0, code, purity, PIL tile at target_mpp). Grid in level-0 px, non-overlapping."""
    side0 = int(round(tile_px * target_mpp / mpp0))         # 512 -> 461 px at L0 (eff. 0.4202 µm/px); 256 -> 230
    W0, H0 = slide.dimensions
    sx = mask.shape[1] / W0; sy = mask.shape[0] / H0
    for y0 in range(0, H0 - side0 + 1, side0):
        for x0 in range(0, W0 - side0 + 1, side0):
            win = mask[int(y0 * sy):int((y0 + side0) * sy), int(x0 * sx):int((x0 + side0) * sx)]
            if win.size == 0: continue
            vals, cnt = np.unique(win, return_counts=True)
            k, frac = int(vals[cnt.argmax()]), cnt.max() / win.size
            if k == 0 or frac < purity: continue
            tile = read_rgb(slide, (x0, y0), 0, (side0, side0)).resize((tile_px, tile_px), Image.LANCZOS)
            yield x0, y0, k, float(frac), tile
# usage: s = openslide.OpenSlide("WSI/A03.svs"); mpp, regs = parse_bach_xml("WSI/A03.xml")
#        m1 = label_mask(s, regs, level=1)   # ~4 L0 px/mask px: purity resolution ≈1.9 µm (A03: 14336x9736 = 140 MB)
#        overview / IoU vs gt_thumbnails: label_mask(s, regs, level=s.level_count-1)
```
- Margin windows at 0.42 µm/px: read **1843x1382 level-0 px** (2048·0.42/0.4667) and resize to 2048x1536.
- The 256-px edge strip ≈ 107.5 µm.
- Folds = `{A01..A10}` → patients 1-10. Never place `A0k` and `{2k-1, 2k}.svs` in different folds.

## 2. THUNDER BACH split (`src/thunder/datasets/dataset/bach.py`, `src/thunder/config/dataset/bach.yaml`) [V]
`bach.yaml`:
```yaml
nb_train_samples: 218, nb_val_samples: 50, nb_test_samples: 132, mpp: 0.42, image_sizes: [[2048,1536]]
classes: ["Benign", "InSitu", "Invasive", "Normal"]
class_to_id: {Benign: 0, InSitu: 1, Invasive: 2, Normal: 3}
id_to_classname: {0: breast non-malignant benign tissue, 1: breast malignant in-situ carcinoma,
                  2: breast malignant invasive carcinoma, 3: normal breast tissue}   # "From Quiltnet and KEEP papers"
md5sum: "6b1a33fab172d8d626714978ae08f09f"   # of THUNDER's split dict (paths prefixed "ICIAR2018_BACH_Challenge/Photos/")
```
Listing:
- For each class in that order: `sorted(os.listdir(Photos/<cls>))`, filter `".tif" in s`, `assert len == 100`.
- The global index is `class_idx*100 + (file_number-1)`, so index 0 = b001, 100 = is001, 200 = iv001, 300 = n001.

Ranges are copied from eva and eva's train range is split further:
| split | THUNDER ranges | files | per class B/IS/IV/N |
|---|---|---|---|
| train 218 | (0,15)(25,41)(59,60)(90,110)(120,139)(169,200)(210,240)(258,260)(273,305)(315,345)(368,380)(390,400) | b001-015, b026-041, b060, b091-is010, is021-039, is070-iv040 … | 42/60/59/57 |
| val (=our cal) 50 | (15,25)(110,120)(200,210)(305,315)(380,390) | b016-025, is011-020, iv001-010, n006-015, n081-090 | 10/10/10/20 |
| test 132 | (41,59)(60,90)(139,169)(240,258)(260,273)(345,368) | b042-059, b061-090, is040-069, iv041-058, iv061-073, n046-068 | 48/30/31/23 |

eva (`kaiko-ai/eva` `classification/bach.py`):
- Classes and `class_to_idx` are identical, loaded via torchvision `make_dataset`, which sorts classes and files.
- eva **train** 268 = `(0,41)(59,60)(90,139)(169,240)(258,260)(273,345)(368,400)` = THUNDER train ∪ val.
- eva **val** 132 = THUNDER test, exactly. Our test set is the eva/THUNDER leaderboard set.

Patient semantics:
- `patients = {"Histology": [...145 filenames], "Label": [...], "Patient": [...int]}`. These are parallel lists, one
  row per **identified** image, and they are exactly the xlsx's non-blank Patient cells.
- Identified: Benign b032-b090, InSitu is040-is069, Invasive iv041-iv073, Normal n046-n068.
- 29 patients (IDs 11-39). Patient 14 has 36 images across Benign, InSitu and Invasive; patients cross classes (e.g. 22:
  InSitu+Invasive+Normal).
- Images not listed are unidentified. The organizers say "it is safe to assume that all unidentified images come from
  different patients of the identified ones". So unknowns can never collide with known IDs, and THUNDER only checks
  known IDs.

Result of re-running the assertion here:
- Train known patients = {11,12,13,15,19}, from 13 images: b032-b041, b060, iv059, iv060.
  Patient 19 = b060 + iv059/iv060, which explains the odd single ranges (59,60) and (258,260).
- Val has 0 identified images.
- Test known = {14,16-18,20-39}. All 132 test images are identified.
- All intersections are empty.

Reproduction for `tests/test_split.py`:
```python
def ranges_to_idx(r): return [i for a, b in r for i in range(a, b)]
CLASSES = ["Benign", "InSitu", "Invasive", "Normal"]           # config; sorted() of folder names
files = [f"Photos/{c}/{f}" for c in CLASSES for f in sorted(x for x in os.listdir(root/"Photos"/c) if x.endswith(".tif"))]
assert len(files) == 400 and all(len([f for f in files if f"/{c}/" in f]) == 100 for c in CLASSES)
split = {s: [files[i] for i in ranges_to_idx(cfg.split[s])] for s in ("train", "cal", "test")}
assert sorted(sum(split.values(), [])) == sorted(files)              # partition, sizes 218/50/132
pmap = load_micro_patients(xlsx)                                      # {"b032.tif": 11, ...} 145 entries
P = {s: {pmap[os.path.basename(f)] for f in fs if os.path.basename(f) in pmap} for s, fs in split.items()}
assert not (P["train"] & P["test"]) and not (P["train"] & P["cal"]) and not (P["cal"] & P["test"])
# CV groups inside train: known -> f"p{id}"; unknown -> own group (organizer statement) or 5-file block (conservative)
```

## 3. Patient files (Dropbox, from the challenge Dataset page) [V]
- **Microscopy:** the user's link
  `https://www.dropbox.com/scl/fo/efjnzcjydcjc4ebz1py7n/AF1Dx2gYEZRjo3eWw29IMIE/ICIAR2018_BACH_dataset_micro_patient.xlsx?rlkey=w5ebtg9dlwrk57663chvodzy6&dl=1`
  - Returns HTTP 200 and the raw xlsx: 15,181 B, md5 `7a23a77e4531c06c05d8e08e39498b06`.
  - It is identical to the file in the Dataset page's folder link
    `https://www.dropbox.com/sh/sc7yg21bcs3wr0z/AACiavY0BQPF6GYna9Fkjzola?dl=1` (zip).
  - The scl folder page itself returns "no access"; only the direct file link works.
  - One sheet, `Sheet1`, dimension A1:T101 (column T is empty), in a **wide layout** of 4 column groups.
    Row 1 is a header repeated per group, rows 2-101 are images 001-100:
    `A Histology | B Label | C Patient` (Benign) · `E,F,G` (InSitu) · `I,J,K` (Invasive) · `M,N,O` (Normal).
  - Patient is a numeric cell (11-39) or **blank** when unidentified. Labels are `Benign/InSitu/Invasive/Normal`.
  - The 145 non-blank rows match THUNDER's dict exactly (0 diffs).
  - Do not rely on openpyxl being installed on Delta; this stdlib parser is verified:
```python
import zipfile, re, xml.etree.ElementTree as ET
def load_micro_patients(path):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}; z = zipfile.ZipFile(path)
    ss = ["".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    g = {}
    for c in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{ns['m']}}}c"):
        v = c.find("m:v", ns)
        if v is not None: g[c.get("r")] = ss[int(v.text)] if c.get("t") == "s" else v.text
    out = {}
    for h, l, p in (("A","B","C"), ("E","F","G"), ("I","J","K"), ("M","N","O")):
        for r in range(2, 102):
            if g.get(f"{p}{r}") not in (None, ""): out[g[f"{h}{r}"]] = int(float(g[f"{p}{r}"]))
    assert len(out) == 145; return out
```
- **WSI:** `ICIAR2018_BACH_dataset_WSI_patient.xlsx` (9,927 B, md5 e1e736043692114041d96fd73f7901d3), from
  `https://www.dropbox.com/sh/8un9pwrhtqu99r3/AADX3De88B_iJdDtp359X5jHa?dl=1` (zip).
  - `Sheet1` has columns `WSI filenames | Type | Patient`.
  - Rows `01.svs..20.svs` are `unlabeled` with patients 1,1,2,2,…,10,10. Rows `A01..A10.svs` are `labeled`, patients
    1..10.
- Optional: Dropbox "code" zip → `code/read_xml.py` and `code/read_svs.py` (quoted in §1.4).

## 4. Quilt-1M lookup CSV (`/work/nvme/bfqi/shared/datasets/quilt/quilt_1M_lookup-002.csv`)
Sources:
- https://github.com/wisdomikezogwo/quilt1m: README, including the 2023-10-26 changelog that added `single_wsi`
  and `not_histology`.
- Zenodo 8239942 (restricted; 512-px version).
- There is **no HF dataset card for Quilt-1M**: the `wisdomik` HF datasets are only VQA/LLaVA/MedicalNarratives.
- The column list comes from a published 5-row sample of the CSV (`song83244829/pathphi/temp.csv`) and from loaders
  (`VectorInstitute/mmlearn …/datasets/quilt.py`, `DDVD233/CLIMB`, `ExplainableML/fomo_in_flux`).
- The NCSA guide confirms only `caption, image_path, pathology, magnification`.

Header [V via public sample; confirm with `pd.read_csv(path, nrows=5)` on Delta]:
`<unnamed index → "Unnamed: 0">, caption, image_path, subset, split, pathology, roi_text, noisy_text, corrected_text,
med_umls_ids, magnification, height, width, not_histology, single_wsi`
| column | meaning / format |
|---|---|
| `caption` | Training text (one row per image-caption pair). The same `image_path` repeats once per caption. |
| `image_path` | **Bare filename**, e.g. `YC5jOhcQGes_image_ede69e4d-….jpg` (quilt), `994701482116173824_0.jpg` (openpath) |
| `subset` | `quilt` (YouTube), `openpath` (Twitter), `pubmed`, `laion` (the 4 values in mmlearn) |
| `split` | `train` / `val` (mmlearn) |
| `pathology` | **Python-list string**, e.g. `"['Breast', 'Gynecologic', 'Soft tissue']"`; video-level sub-pathology (corrected 2023-10-26); empty for non-quilt subsets. Parse with `ast.literal_eval`; drop `'Unknown'`. |
| `roi_text` | List string of ROI phrases, e.g. `"['Breast tissue with hyperplasia']"` or `"[]"` |
| `noisy_text` / `corrected_text` | Raw vs LLM-corrected ASR transcript chunk around the frame (long; skip unless needed) |
| `med_umls_ids` | Nested list string `[[{'entity','concept_id' (CUI),'confidence'}…]]` (heavy; skip) |
| `magnification` | Float code (`0.0`, `1.0`, …). The paper's classifier predicts 3 scales {1-10x, >10-20x, >20x}; code→scale mapping assumed 0/1/2 [U] |
| `height`, `width` | Image px (float); empty for openpath |
| `not_histology` | 0/1 manual video flag (1 = projection/drawing/non-histo) → **filter `== 0`** |
| `single_wsi` | 1 = the video covers one WSI, 0 = several |

Files:
- Images are in `images_part_{1..10}/quilt_1m/<image_path>` (official guide). The part number is not in the CSV.
- Build a `basename → part` index once, in a CPU job (≈653k dirents):
  `os.scandir(f"{root}/images_part_{i}/quilt_1m")`.
- Cache the index as CSV. Rows whose file is missing are counted and dropped.

Reading:
- `pd.read_csv(p, usecols=lambda c: c in WANT, dtype=str, keep_default_na=False, chunksize=200_000)` with
  `WANT = {caption, image_path, subset, split, pathology, roi_text, magnification, not_histology, single_wsi}`.
  Captions contain quoted commas and newlines, so use the C engine (the default).
- Breast filter:
  - `any(x.lower().startswith("breast") for x in literal_eval(pathology))` (older CSV versions had
    "Breast pathology");
  - or a caption regex.
- Leakage-safe groups: dedupe to image level first, then group.
```python
VID = re.compile(r"^([A-Za-z0-9_-]{11})_image_")         # YouTube id = 11 chars
def quilt_group(subset, image_path):
    if subset == "quilt" and (m := VID.match(image_path)): return "yt:" + m.group(1)
    if subset == "openpath": return "tw:" + image_path.split("_")[0]   # tweet id; _k = image index
    return f"{subset}:{image_path}"                                     # pubmed/laion: per image [U]
```
Use `GroupShuffleSplit` or `StratifiedGroupKFold` on that group. There is no `video_id` column; the video ID is the
filename prefix (pattern verified on the sample rows).

## 5. PathGen-1.6M (`/projects/bfqi/shared/datasets/pathgen_dataset/`)
Sources: https://github.com/PathFoundation/PathGen-1.6M (README, `Data/PathGen-1.6M_sample.json`) and HF
`jamessyx/PathGen` (gated; files: `PathGen-1.6M.json` 978,614,564 B and README).

JSON format [V on the sample and the R2 head]:
- One top-level **list** of dicts: `{"wsi_id": "TCGA-AA-3844-01Z-00-DX1.<uuid>", "position": ["35136","33344"],
  "caption": "..."}`.
- `position` holds **strings**: level-0 (x, y) top-left of a **672x672** patch read with
  `wsi.read_region((x, y), 0, (672, 672))` (README code).
- The author's public R2 copy (`https://pub-7a38cc906afa44a4a01533c288d0b1af.r2.dev/PathGen-1.6M.json`,
  889,466,384 B) has exactly those 3 keys.
- The README example also shows `"file_id"` (GDC UUID, for `gdc-client download <file_id>`). The larger HF file
  probably adds it [U].
- The README code uses `item['WSI_id']`, so read with `e.get("wsi_id") or e.get("WSI_id")`, and `file_id` optionally.

Patch images:
- They are **not released**. Users must download the TCGA `.svs` files and crop them. Formal-diagnostic `DX1` slides
  are the majority in the sample.
- What Delta's "2k-slide sample" is (raw TCGA WSIs or pre-cropped patches) is **unknown** [U]. `00_inventory.py`
  should record: file extensions and counts, whether names match `TCGA-…DX1.<uuid>.svs` (WSIs) or
  `<wsi_id>_<x>_<y>.png` (README crop naming), and the JSON key set.

Grouping:
- slide = `wsi_id`; patient = `wsi_id[:12]` (TCGA barcode).
- In the 5k sample, 3582 slides = 3582 patients (≤6 rows per slide), and 591/5000 captions (11.8%) contain
  "breast".
- Breast filter: caption regex, optionally a GDC project lookup via `file_id` from the laptop.

Load:
- `json.load` of a ~0.9-1 GB file needs several GB of RAM. Run it in the CPU job only, never on the login node.
- Keep `{wsi_id, x, y, caption}` and write a slim CSV/parquet.

## 6. PathMMU PathCLS (BACH items)
Sources: https://github.com/PathMMU-Benchmark/PathMMU (`data/construct_pathcls.py`, `data/instructions.md`,
`eval/utils/data_utils.py`) and HF `jamessyx/PathMMU` (gated: `data.json`, `images.zip`).

Structure [V from code]:
- `data[category][split]` → list. Categories are `PubMed, SocialPath, Atlas, EduContent, PathCLS`; splits are
  `val, test, test_tiny`.
- PathCLS mixes all its source datasets in one list per split.
- Item keys: `No, img, question, options, answer, explanation` (+ `source_img` for PathCLS/Atlas). Example from
  `instructions.md`:
  `"options": ["A) Hyperkeratosis", …], "answer": "D) Both A and B", "source_img": "books_set/images/<uuid>.png"`.
- The correct letter is `answer[0]`. Option text is `opt.split(")", 1)[1].strip()` (data_utils uses `split(')')[1]`).
- `construct_pathcls.py` copies `source_img` (relative to a root with `BACH/images/`) to `images/<img>`. It converts
  `.tif` to **JPEG** (`img.convert('RGB').save(..., 'JPEG')`): lossy, and `img` is a hash name.
- The paper: PathCLS samples are "reformulate[d] … into a Q&A format and randomly sample[d]". BACH option wording and
  `source_img` names are not public [U].

Mapping (assert exactly one hit, else route to manual review; cross-check with the filename prefix when
`basename(source_img)` matches `^(b|is|iv|n)\d{3}\.tif$`):
```python
def pathcls_bach_class(answer: str) -> str:
    t = answer.split(")", 1)[1].lower() if answer[1:2] == ")" else answer.lower()
    hits = set()
    if re.search(r"in[\s-]?situ|\bdcis\b|non-?invasive", t): hits.add("InSitu")
    elif re.search(r"invasive|infiltrat", t): hits.add("Invasive")
    if re.search(r"benign|non-?malignant", t): hits.add("Benign")
    if re.search(r"\bnormal\b", t): hits.add("Normal")
    if len(hits) != 1: raise ValueError(f"ambiguous BACH answer {answer!r}")
    return hits.pop()
```
- Any PathMMU-BACH image whose source maps to one of our 400 files inherits that file's THUNDER split. Never train
  on test basenames.
- If `source_img` points to BACH **test** photos (`test*.tif`, unlabeled officially), they are outside our 400 files.

## Verified vs unverified
**Verified here:**
- Zenodo file list, sizes, md5s and Range support.
- Full zip layout (497 entries); exact tif names; the CSV's format and agreement with folders.
- The 16-/8-bit TIFF split (by size) and the 8-bit-in-16 encoding (b001 strip 1); Pillow decodes 16-bit RGB TIFF.
- 30 WSIs, with XML only for A01-A10.
- XML schema, per-WSI label strings and counts, A01's Attribute-based labels, `MicronsPerPixel` 0.466667.
- A03 levels, JP2K compression, header `MPP = 0,500000`, swapped `OriginalWidth`, and level-0 XML coordinates.
- The official `read_xml.py` text and colour code; the gt_thumbnail palette.
- THUNDER yaml/py content and ranges; sizes 218/50/132; zero leakage on re-run; the eva ranges.
- The micro xlsx (link, md5, layout, 145 rows identical to THUNDER); WSI patient xlsx contents.
- Quilt columns from a public CSV sample; PathGen keys from the sample and R2 head; PathMMU keys from repo code.

**Unverified:**
- Physical WSI mpp (0.467 vs header 0.5).
- What OpenSlide reports for `mpp-x`, `level_count` and JP2K support in our Delta venv.
- Other WSIs' dimensions.
- Whether all 381 large TIFFs are 8-bit×257.
- Whether WSI and microscopy patient IDs share one namespace.
- Quilt `magnification` code→scale mapping; `pubmed`/`laion` filename patterns; exact headers of the Delta copy
  `-002.csv`.
- Which PathGen JSON variant is on Delta and whether it has `file_id`; what the "2k-slide sample" contains.
- PathMMU BACH option wording and `source_img` naming. Our HF access to `data.json` is still pending.

Nothing was run on Delta (constraint: research agents never SSH). `00_inventory.py` should confirm every [U] item.
