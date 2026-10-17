# Cells backends: exact recipes (research, 2026-10-17)

Scope: segment and classify nuclei (PanNuke, 5 types) on about 1,200 BACH tiles. Each tile is 512 px at the measured mpp
(about 0.42) and is pre-resized to 0.25 µm/px (about 860 px). The run is one Delta A100 job of at most 1 h, with offline
compute nodes. Base environment: `pytorch-conda/2.8` (Py 3.11.13, torch 2.8.0+cu128, numpy 2.2.6, timm 1.0.19).
Feeds CELLS_SPEC §5 and BUILD_PLAN Task 8.

Sources actually read (the version or commit is pinned in each link):
- TIAToolbox 2.1.3 wheel from PyPI, released 2026-07-21. File `tiatoolbox-2.1.3-py3-none-any.whl`, sha256 `0c68c4c0c63fcf95aa7e29179c6b92f94435835569d387cc9e7514f891d96dc4`.
  The source was read from it: `engine/{engine_abc,semantic_segmentor,multi_task_segmentor,nucleus_instance_segmentor}.py`,
  `architecture/{__init__,hovernet,utils}.py`, `models/dataset/dataset_abc.py`, `wsicore/wsireader/base.py`,
  `data/pretrained_model.yaml`, `__init__.py`. Metadata from https://pypi.org/pypi/tiatoolbox/2.1.3/json
- Notebook 08 at tag v2.1.3 (commit 0d1495f726aa73118da3e2b61d70ea50784505f7):
  https://github.com/TissueImageAnalytics/tiatoolbox/blob/v2.1.3/examples/08-nucleus-instance-segmentation.ipynb
- Weights: https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights at repo commit
  `ea909bdcce3aa409bf436c09b4495cca90a8e99b`. The repo is not gated.
- vqdang/hover_net `master` @ `67e2ce5e3f1a64a2ece77ad1c24233653a9e0901` (2023-10-27). Read: README, `run_infer.py`,
  `run_tile.sh`, `type_info.json`, `infer/{base,tile}.py`, `models/hovernet/post_proc.py`, `run_utils/utils.py`.
- HoVer-NeXt https://github.com/digitalpathologybern/hover_next_inference `main` @ `0e99d535fd64640b5486ca9ee4f3864fa96ba83c`
  (2026-08-26). Read: README and `main.py` arguments.

**Bottom line:** TIAToolbox 2.1.3 runs in a `--system-site-packages` venv without replacing the module's torch or numpy.
No declared range excludes torch 2.8.0 or numpy 2.2.6. The call that works in 2.1.3 is
`MultiTaskSegmentor(...).run(pngs, patch_mode=False, save_dir=..., input_resolutions=[{"units":"baseline","resolution":1.0}], auto_get_mask=False)`.
It writes one zarr per tile with arrays `box/centroid/contours/prob/type`. The wall-clock bottleneck is CPU
post-processing, not the GPU, so shard 4 processes on the one GPU.

---

## 1. Primary: TIAToolbox 2.1.3 `hovernet_fast-pannuke`

### 1.1 Dependency check against the module (from 2.1.3 `requires_dist`)
| tiatoolbox pin | Module value | Result |
|---|---|---|
| `python >=3.11,<3.15` | 3.11.13 | OK |
| `torch >=2.5,<2.13`, `torchvision >=0.15,<0.28` | 2.8.0, module's tv | OK. pip's default only-if-needed strategy keeps them. **Pin anyway**: the newest torchvision (0.27.1) requires `torch==2.12.1` |
| `numpy >=2.0,<2.4.4` | 2.2.6 | OK. No in-range dependency needs numpy >2.2.6: pandas 3.0.2 ≥1.26, scipy 1.17.0 ≥1.26.4, numba 0.65.1 <2.5, opencv 4.14 ≥2, zarr 3.1.6 ≥2.0 |
| `timm >=1.0.3,<1.0.28` | 1.0.19 | OK, pin |
| `transformers >=4.51.1,<5.14.1`, `huggingface_hub >=0.33.3,<1.24` | module's (unknown) | `import tiatoolbox.models` runs `from transformers import SamModel, SamProcessor` (sam.py), so transformers **must import**. Pin both only if they are in range (script §5). Otherwise the venv gets transformers 5.x and hub ≥1.5, which is fine inside this venv only |
| `scikit-image >=0.26.0,<0.26.1` (that is, ==0.26.0), `zarr >=3.0.8,<3.2`, `dask >=2026.1.2,<2026.7.1`, `s3fs` (pulls a matching `fsspec`) | probably older | Installed into the venv and shadows the module's copies (venv first on `sys.path`). pip may warn about module packages such as `datasets` and fsspec. That is harmless unless the cells venv imports them |
| `opencv-python <5` | module may ship cv2 | The venv copy wins. Both import as `cv2` |
| Hard dependencies `jupyterlab, sphinx, bokeh, umap-learn, numba, wsidicom, openslide-bin, SimpleITK, glymur, imagecodecs` | – | Pure bloat, a few hundred MB, but installed as wheels. `openslide-bin` ships libopenslide, so no system libraries are needed |

Conclusion: nothing conflicts with numpy 2.2.6 or torch 2.8. The constraints file pins the torch stack and numpy (and
timm/transformers/hub when in range) so the result is deterministic.

### 1.2 Weights (offline)
- File: `hovernet_fast-pannuke.pth` in `TIACentre/TIAToolbox_pretrained_weights`. Size 151,109,291 B,
  sha256 `789b2be8bdaf532627f584a009f83b3ba06e04c0e55c1635252ff12785ef57e8`.
- Behaviour in 2.1.3: `get_pretrained_model(name, weights)` calls `fetch_pretrained_weights` (an `hf_hub_download` into
  `~/.tiatoolbox/models`) **only when `weights is None`**. Passing `weights="/abs/path/hovernet_fast-pannuke.pth"` means
  no network access. It runs `torch.load(weights, map_location="cpu")` (weights_only default; plain state_dict) then
  `load_state_dict(strict=True)`.
- License: TIAToolbox code is BSD-3-Clause, and its LICENSE notes the weights are licensed differently. The weights are
  PanNuke-derived, so CC BY-NC-SA 4.0 (hover_net README). Non-commercial hackathon use is fine; cite them.

### 1.3 API facts (2.1.3 source)
- **Deprecation:** `NucleusInstanceSegmentor` is a thin subclass of `MultiTaskSegmentor` with the same constructor.
  It has been deprecated since 2.1.0 and emits a `DeprecationWarning` plus a logger warning. Notebook 08 at v2.1.3
  still uses it. Use `from tiatoolbox.models import MultiTaskSegmentor`. The 1.x API is gone in 2.x: there is no
  `predict(..., mode="tile")` and no `.dat`/joblib outputs. Use `run()`.
- **torch.compile:** `EngineABC.__init__` calls `compile_model(model, mode=rcParam["torch_compile_mode"])`. The default
  is `"default"`, and an A100 (capability (8,0)) qualifies, so it **would compile** (triton/gcc on the node, plus
  recompiles when the tail batch has a different size). To disable it, run
  `from tiatoolbox import rcParam; rcParam["torch_compile_mode"] = "disable"` **before constructing** the engine.
- **Registry:** `HoVerNet(num_types=6, mode="fast")`. ioconfig is `IOInstanceSegmentorConfig`: input/output 0.25 mpp,
  `patch_input_shape [256,256]`, `patch_output_shape [164,164]`, `stride [164,164]`, `tile_shape [1024,1024]`,
  `margin 128`.
- **Classes** (`nuc_type_dict`): `0 Background, 1 Neoplastic, 2 Inflammatory, 3 Connective, 4 Dead, 5 Non-Neoplastic
  Epithelial`. These are the same ids as vqdang's `type_info.json` (`nolabe, neopla, inflam, connec, necros, no-neo`).
- **Tile mode = `patch_mode=False` with PNG paths.** `.png` opens as `VirtualWSIReader`. Use
  `input_resolutions=[{"units":"baseline","resolution":1.0}]`, as notebook 08 does, so the pre-resized image is read
  at native resolution. Equivalent alternative: `wsireader_kwargs={"mpp": (0.25, 0.25), "power": 40}` with the default
  ioconfig. Set `auto_get_mask=False` to segment the whole tile. Tiles ≤1024 px take the in-memory `_process_full_wsi`
  path; tiles over 1024 px switch to dask tile-merging.
- **`save_dir` is required** in WSI mode (`OSError` otherwise). `prepare_engines_save_dir` calls
  `save_dir.mkdir(parents=True)` **without `exist_ok`**. An existing dir is a `FileExistsError` unless
  `overwrite=True`, which `rmtree`s it. Use a fresh scratch dir per chunk. Passing `save_dir` forces
  `output_type="zarr"` ("dict" is downgraded with a log line). `"annotationstore"` (.db) and `"qupath"` (.json) are
  the other options.
- **Output:** `run()` returns `{Path(input_png): save_dir/"<png_stem>.zarr"}`. **Stems must be unique** (use
  `<case_id>_<tile_idx>.png`). Zarr v3 group, one array per field, row i = nucleus i:
  `box (N,4) [x_min,y_min,x_max,y_max]`, `centroid (N,2) [x,y]` float, `contours (N,Lmax,2) [x,y]` int32 **padded
  with `np.iinfo(int32).min`** (`pad_contours`, zarr v3 has no object dtype), `prob (N,)`, `type (N,)`.
  Coordinates are pixels of the input PNG (scale factor 1 at baseline). With zero nuclei the arrays are empty, so
  guard for that.
- **`prob` is not a softmax.** The `tp` head is argmaxed per pixel in `infer_batch`. `prob` is the instance's majority
  type pixel fraction, and when the majority is type 0 the second type is used. vqdang's `type_prob` is the same.
  Label it "type vote fraction" in the UI and docs.
- Engine knobs: constructor `batch_size` (default 8), `num_workers` (DataLoader; `persistent_workers=num_workers>0`
  is re-created **per image**, so use `0` for 860-px tiles), `device="cuda"`, `verbose`.

### 1.4 Run code (core of `pipeline/04b_cells.py`, TIAToolbox backend)
```python
import os, shutil, uuid
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import zarr
from PIL import Image
from tiatoolbox import rcParam
rcParam["torch_compile_mode"] = "disable"          # must precede engine construction
from tiatoolbox.models import MultiTaskSegmentor   # not the deprecated NucleusInstanceSegmentor
from cells_convert import nuclei_records, render_overlay, write_tile_json   # §4

def resize_to_model(src: Path, dst: Path, mpp_src: float, mpp_model: float) -> dict:
    im = Image.open(src).convert("RGB")           # drop alpha: a 4-ch PNG makes VirtualWSIReader use "feature" mode
    s = mpp_src / mpp_model                        # measured mpp (inventory.json) / 0.25, e.g. 1.68
    size = (round(im.width * s), round(im.height * s))
    im.resize(size, Image.Resampling.BICUBIC).save(dst)
    return {"scale": size[0] / im.width,           # model px per original px
            "mpp_model": mpp_src * im.width / size[0],   # exact px size of resized tile (~0.25)
            "hw_model": (size[1], size[0]), "tile_px": [im.width, im.height]}

engine = MultiTaskSegmentor(model="hovernet_fast-pannuke", weights=str(WEIGHTS),   # local file, no HF call
                            batch_size=16, num_workers=0, device="cuda", verbose=False)

def segment_chunk(chunk, scratch: Path, cells_dir: Path):   # chunk: <=50 rows of cell_tiles.csv
    tmp_in = scratch / f"in_{uuid.uuid4().hex[:8]}"; tmp_in.mkdir(parents=True)
    pngs, geo = [], {}
    for t in chunk:
        p = tmp_in / f"{t.case_id}_{t.tile_idx}.png"         # unique stem -> unique <stem>.zarr
        geo[p] = resize_to_model(Path(t.path), p, t.mpp_src, MODEL_MPP); pngs.append(p)
    save_dir = scratch / f"tia_{uuid.uuid4().hex[:8]}"       # must NOT exist yet
    out = engine.run(images=pngs, patch_mode=False, save_dir=save_dir, output_type="zarr",
                     input_resolutions=[{"units": "baseline", "resolution": 1.0}],
                     auto_get_mask=False, overwrite=False)
    for t, p in zip(chunk, pngs):
        g, gm = zarr.open(str(out[p]), mode="r"), geo[p]
        if "type" in g and g["type"].shape[0] > 0:
            recs, n_untyped = nuclei_records(g["contours"][:], g["centroid"][:], g["type"][:], g["prob"][:],
                                             mpp_model=gm["mpp_model"], scale=gm["scale"], hw_model=gm["hw_model"])
        else:
            recs, n_untyped = [], 0
        d = cells_dir / t.case_id
        render_overlay(Path(t.path), recs, d / f"{t.tile_idx}_overlay.png")
        write_tile_json(d / f"{t.tile_idx}.json", recs, meta={       # JSON written LAST = completion marker
            "case_id": t.case_id, "tile_idx": int(t.tile_idx), "backend": "tiatoolbox==2.1.3",
            "model": "hovernet_fast-pannuke", "weights_sha256": WEIGHTS_SHA256, "mpp_source": t.mpp_src,
            "mpp_model": gm["mpp_model"], "scale": gm["scale"], "tile_px": gm["tile_px"],
            "contour_frame": "original_tile_px", "type_prob_kind": "majority_vote_fraction",
            "n_untyped": n_untyped})
    shutil.rmtree(save_dir); shutil.rmtree(tmp_in)

# Resume + shard: todo = [t for t in tiles[shard::nshards] if not (cells_dir/t.case_id/f"{t.tile_idx}.json").exists()]
# Loop segment_chunk over batches of 50 (a crash loses <=50 tiles). If run() turns out not to be re-entrant
# (unverified), construct a new engine per chunk (~seconds).
```

### 1.5 Throughput on A100 (estimate, **unverified**; measure in `cells_spike`)
- Patches: an 860-px tile with 256-px input, 164-px output and stride 164 gives 6×6 = 36 patches. 1,200 tiles give
  about 43k patches. HoVer-Net-fast fp32 at batch 16–32 is likely ≥150–300 patches/s, so roughly 3–5 min of GPU time
  in total.
- Per image, WSI mode adds CPU work: reader setup (opens the PNG twice), stitching, `_proc_np_hv` watershed on 860²,
  the per-instance contour loop, and the zarr writes for 5 arrays. Expect about 1–2.5 s/tile in one process, so
  20–50 min single-process. That is too close to 1 h.
- Plan: run 4 shards (4 CPUs each, `OMP_NUM_THREADS=4`, batch 16) on the one GPU. That should give about 6–15 min
  wall-clock. GPU memory is roughly ≤6 GB per shard (unverified); a 40 GB A100 is fine.
- Gate: the spike logs seconds per tile. If `1200 × s/tile / nshards > 35 min`, raise the shard count (CPU-bound, ≤16
  CPUs) or reduce tiles.

---

## 2. Fallback: vqdang/hover_net tile mode, PanNuke fast checkpoint

Install on the login node, reusing the cells venv. It already has cv2, scikit-image, scipy, matplotlib, psutil and
tqdm. **Do not** `pip install -r requirements.txt`: it pins numpy 1.19, skimage 0.17 and imgaug.
```bash
git clone https://github.com/vqdang/hover_net.git "$HOME/src/hover_net"
git -C "$HOME/src/hover_net" checkout 67e2ce5e3f1a64a2ece77ad1c24233653a9e0901
# 1) imgaug (NumPy-2 incompatible, unused at inference: `ia` appears only in a comment)
sed -i '/^from imgaug import imgaug as ia$/d' "$HOME/src/hover_net/run_utils/utils.py"
# 2) torch>=2.6 defaults weights_only=True; checkpoint is {"desc": state_dict}; trusted source
sed -i 's|torch.load(self.method\["model_path"\])\["desc"\]|torch.load(self.method["model_path"], map_location="cpu", weights_only=False)["desc"]|' \
  "$HOME/src/hover_net/infer/base.py"
grep -n imgaug "$HOME/src/hover_net/run_utils/utils.py" && echo "PATCH FAILED" || echo "imgaug removed"
pip install docopt==0.6.2 termcolor gdown          # inside the activated cells venv
gdown "https://drive.google.com/uc?id=1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR" \
  -O "$TEAM_DIR/weights/hovernet_fast_pannuke_type_tf2pytorch.tar"   # README "PanNuke checkpoint" (seg+class)
sha256sum "$TEAM_DIR/weights/hovernet_fast_pannuke_type_tf2pytorch.tar" | tee "$TEAM_DIR/weights/hovernet_fast_pannuke.sha256"
```
If Drive refuses (quota or confirm page), download on the laptop and `scp` it to the same path (CELLS_SPEC §11).
Other legacy items: `scipy.ndimage.{filters,measurements,morphology}` are deprecated namespaces that still exist
before SciPy 2.0 (warnings only). `skimage.segmentation.watershed` is fine. `plt.get_cmap` still exists. A scan found
no `np.int/np.bool/np.float` aliases on the tile inference path.

Run it from the repo dir; imports are relative. The input dir must contain **only** images (`glob("<dir>/*")`).
`tile` mode `rm_n_mkdir`s `out/{json,mat,overlay}`, so use a fresh `--output_dir` per chunk:
```bash
cd "$HOME/src/hover_net"
python run_infer.py --gpu="${CUDA_VISIBLE_DEVICES:-0}" --nr_types=6 --type_info_path=type_info.json \
  --batch_size=64 --model_mode=fast --model_path="$TEAM_DIR/weights/hovernet_fast_pannuke_type_tf2pytorch.tar" \
  --nr_inference_workers=8 --nr_post_proc_workers=12 \
  tile --input_dir="$SCR/in_000" --output_dir="$SCR/out_000" --mem_usage=0.1
```
`--model_mode=fast` is **mandatory** for PanNuke, and `--nr_types=6` too. `--gpu` overwrites `CUDA_VISIBLE_DEVICES`,
so pass Slurm's value through. Outputs: `out/json/<stem>.json`, `out/mat/<stem>.mat` and `out/overlay/<stem>.png`
(its own colors, ignore it). The JSON is
`{"mag": null, "nuc": {"<id>": {"bbox": [[rmin,cmin],[rmax,cmax]], "centroid": [x,y], "contour": [[x,y],...],
"type": int, "type_prob": float}}}` in input-PNG px. Convert with the same §4 function:
```python
nuc = list(json.load(open(out_dir / "json" / f"{stem}.json"))["nuc"].values())
recs, n_untyped = nuclei_records([n["contour"] for n in nuc], [n["centroid"] for n in nuc],
                                 [n["type"] for n in nuc], [n["type_prob"] for n in nuc],
                                 mpp_model=gm["mpp_model"], scale=gm["scale"], hw_model=gm["hw_model"])
```
License: the code is MIT and the weights are CC BY-NC-SA 4.0 (README).

## 3. Third: HoVer-NeXt (brief)
`python3 main.py --input "$SCR/in/*.png" --output_root "$SCR/hn/" --cp pannuke_convnextv2_tiny_1 --tta 4
--overlap 0.9375 --inf_workers 8 --pp_workers 15 --pp_tiling 8`. Use `--overlap 0.9375` because the help text says
to use it for 0.25 mpp; the default 0.96875 is for 0.5 mpp. PNG and NPY inputs are supported and the output is
**zarr only**: an instance map, a class lookup with centroids, and .tsv files (see `sample_analysis.ipynb`). To get
contours, run `cv2.findContours` per instance, then apply §4.
The README environment is conda `environment.yml` plus `torch==2.1.1/cu118`. That clashes with the module, so it needs
its own env (untested on torch 2.8). Weights are on Zenodo record 10635618 and download automatically at runtime, so
**pre-download on the login node**: `pannuke_convnextv2_tiny_1.zip`, unzipped next to `main.py`. License is GPL-3.0.
Use it only if §1 and §2 both fail. CellViT stays skipped (BUILD_PLAN).

---

## 4. Shared conversion + overlay (`cells_convert.py`)
This module is backend-agnostic. Tested locally on synthetic shapes with numpy 2.2.6, scikit-image 0.26.0, zarr 3.1
and opencv-headless: refilling the contour with `fillPoly` reproduces the instance mask's regionprops exactly, the
padding rows are stripped, and the edge flag works.
```python
import json, os, tempfile
from pathlib import Path
import cv2, numpy as np
from skimage.measure import regionprops

PANNUKE = {1: "neoplastic", 2: "inflammatory", 3: "connective", 4: "dead", 5: "epithelial"}  # 0 = background
COLORS = {"neoplastic": "#B0335C", "inflammatory": "#4A3B8F", "connective": "#C9A227",
          "dead": "#5A5A5A", "epithelial": "#2E8BC0"}      # CELLS_SPEC §5; move both dicts to config/
PAD = np.iinfo(np.int32).min                                # TIAToolbox pad_contours() fill value

def nuclei_records(contours, centroids, types, probs, *, mpp_model, scale, hw_model, simplify_px=1.0, edge_px=2):
    """Inputs in model-res px (x, y). scale = model px / original px. Returns (records, n_untyped)."""
    H, W = hw_model; recs, n_untyped = [], 0
    for c, cen, t, p in zip(contours, centroids, types, probs):
        c = np.asarray(c); c = c[(c != PAD).all(axis=1)].astype(np.int32)
        if len(c) < 3: continue
        if int(t) not in PANNUKE: n_untyped += 1; continue          # type 0: report count, don't emit
        x0, y0 = c.min(0); x1, y1 = c.max(0)
        m = np.zeros((y1 - y0 + 3, x1 - x0 + 3), np.uint8)
        cv2.fillPoly(m, [(c - [x0 - 1, y0 - 1]).reshape(-1, 1, 2)], 1)
        rp = regionprops(m)[0]
        simp = cv2.approxPolyDP(c.reshape(-1, 1, 2).astype(np.float32), simplify_px, True)
        recs.append({
            "centroid_um": [round(float(cen[0]) * mpp_model, 2), round(float(cen[1]) * mpp_model, 2)],
            "area_um2": round(float(rp.area) * mpp_model ** 2, 2),
            "perimeter_um": round(float(rp.perimeter) * mpp_model, 2),
            "eccentricity": round(float(rp.eccentricity), 4), "solidity": round(float(rp.solidity), 4),
            "type": PANNUKE[int(t)], "type_prob": round(float(p), 4),
            "edge": bool(x0 <= edge_px or y0 <= edge_px or x1 >= W - 1 - edge_px or y1 >= H - 1 - edge_px),
            "contour_px": (simp.reshape(-1, 2) / scale).round(1).tolist()})   # ORIGINAL tile px
    return recs, n_untyped

def atomic_write_bytes(path: Path, data: bytes):            # swap for core/io.py when it exists
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "wb") as f: f.write(data)
    os.replace(tmp, path)

def write_tile_json(path: Path, recs, *, meta: dict):
    atomic_write_bytes(path, json.dumps({**meta, "n_nuclei": len(recs), "nuclei": recs}).encode())

def _bgr(h): h = h.lstrip("#"); return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)

def render_overlay(orig_png: Path, recs, out_png: Path, thickness: int = 1):
    img = cv2.imread(str(orig_png), cv2.IMREAD_COLOR)       # original-resolution tile (BGR)
    for r in recs:
        pts = np.round(np.asarray(r["contour_px"])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, _bgr(COLORS[r["type"]]), thickness, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img); assert ok
    atomic_write_bytes(out_png, buf.tobytes())
# PIL equivalent: ImageDraw.Draw(im).line([*map(tuple, pts), tuple(pts[0])], fill=COLORS[type], width=1)
```
Notes:
- `centroid_um` and `contour_px` are tile-local; 04c only needs relative distances (the TIL radius).
- Exclude nuclei with `edge: true` from the `nuc_*` morphology features (border nuclei are truncated) but keep them in
  counts. Record that rule in `config/cell_features.yaml`.
- `type_prob` is a vote fraction.

---

## 5. Login-node install + weight pre-download (`pipeline/jobs/cells_env.sh`, run by the controller)
```bash
#!/usr/bin/env bash
# LOGIN NODE ONLY: installs + downloads. No inference here.
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail                                   # after module load (module scripts trip `set -u`)
: "${TEAM_DIR:?export TEAM_DIR=/work/nvme/bfqi/teams/<team>/pathqueue}"
VENV=/work/nvme/bfqi/$USER/venvs/cells; W=$TEAM_DIR/weights
mkdir -p "$W" "$(dirname "$VENV")"
python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python - "$VENV/constraints.txt" <<'PY'            # pin what the module actually has
import re, sys
from importlib.metadata import distributions
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name as cn
always = re.compile(r"^(torch|torchvision|torchaudio|triton|numpy|nvidia-.+)$")
if_in_range = {"timm": ">=1.0.3,<1.0.28", "transformers": ">=4.51.1,<5.14.1", "huggingface-hub": ">=0.33.3,<1.24.0"}
pins = {}
for d in distributions():
    n = cn(d.metadata["Name"] or "")
    if always.match(n) or (n in if_in_range and d.version in SpecifierSet(if_in_range[n])):
        pins[n] = f"{n}=={d.version}"
open(sys.argv[1], "w").write("\n".join(sorted(pins.values())) + "\n"); print(*sorted(pins.values()), sep="\n")
PY
PIP_CONSTRAINT="$VENV/constraints.txt" python -m pip install "tiatoolbox==2.1.3" "docopt==0.6.2" termcolor gdown
python -m pip check || echo "pip check: review above (conflicts with module-only packages are expected)"
python - <<'PY'
import numpy, torch, timm, transformers, skimage, zarr, dask, cv2, tiatoolbox
import tiatoolbox.models                            # pulls transformers.SamModel, openslide, zarr, dask
from tiatoolbox import rcParam
print({"torch": torch.__version__, "numpy": numpy.__version__, "timm": timm.__version__,
       "transformers": transformers.__version__, "tiatoolbox": tiatoolbox.__version__,
       "skimage": skimage.__version__, "zarr": zarr.__version__, "cv2": cv2.__version__,
       "compile_default": rcParam["torch_compile_mode"]})
assert torch.__version__.startswith("2.8.0") and numpy.__version__ == "2.2.6", "module torch/numpy were replaced!"
PY
python - "$W" <<'PY'                                # public, ungated: no token needed
import hashlib, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download("TIACentre/TIAToolbox_pretrained_weights", "hovernet_fast-pannuke.pth",
                    revision="ea909bdcce3aa409bf436c09b4495cca90a8e99b", local_dir=sys.argv[1])
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
assert h == "789b2be8bdaf532627f584a009f83b3ba06e04c0e55c1635252ff12785ef57e8", h
print("weights OK", p)
PY
# Fallback assets (§2), only if the controller wants them ready too: clone+patch hover_net, gdown checkpoint.
```

## 6. Job-side run script skeleton (`pipeline/jobs/cells.sbatch` body lines)
Header, set by the author: `--account=bfqi-delta-gpu --partition=gpuA100x4 --gpus-per-node=1 --cpus-per-task=16
--mem=60G --time=01:00:00`. Optionally add `--deadline=<Sun 07:59 or 17:59 CDT>` so Slurm drops the job instead of
letting it cross a boundary. A human submits it.
```bash
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail
source /work/nvme/bfqi/$USER/venvs/cells/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
SCR=${TMPDIR:-/tmp}/cells_${SLURM_JOB_ID}; mkdir -p "$SCR" logs; export MPLCONFIGDIR=$SCR/mpl
cd "$HOME/pathqueue"                                 # repo checkout (code in home)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,numpy,tiatoolbox,skimage,zarr;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'numpy',numpy.__version__,'tia',tiatoolbox.__version__,'skimage',skimage.__version__,'zarr',zarr.__version__)"
python pipeline/04a_select_tiles.py                  # in-job OOF selection -> cell_tiles.csv (seconds)
NS=${CELLS_SHARDS:-4}; pids=()
for i in $(seq 0 $((NS-1))); do
  python pipeline/04b_cells.py --backend tiatoolbox --shard "$i" --nshards "$NS" --scratch "$SCR" \
    ${CELLS_LIMIT:+--limit "$CELLS_LIMIT"} > "logs/cells_${SLURM_JOB_ID}_${i}.log" 2>&1 & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
python pipeline/04b_cells.py --verify                # every selected tile has .json + _overlay.png; else exit 1
rm -rf "$SCR"; exit $fail
```
For `cells_spike`, use the same body with `--time=00:20:00`, `CELLS_SHARDS=1`, `--limit 5`, and tiles picked from
different classes. Log s/tile for the §1.5 gate.

---

## Verified vs unverified
**Verified (read from source or metadata today, or executed locally):**
- PyPI: tiatoolbox 2.1.3 (2026-07-21). The full `requires_dist` ranges are quoted in §1.1. The latest in-range
  versions of every dependency were resolved from PyPI; none requires numpy >2.2.6 or a torch other than 2.8.
- 2.1.3 source:
  - The `rcParam["torch_compile_mode"]` key with default `"default"`; `"disable"` returns the model unchanged; A100
    (8,0) compiles by default.
  - `NucleusInstanceSegmentor` was deprecated in 2.1.0 (it subclasses `MultiTaskSegmentor`), and there is no
    `predict()`.
  - The `weights=` path skips the HF download.
  - The registry entry, ioconfig and class map.
  - `save_dir` behaviour: mkdir without exist_ok, forced zarr.
  - Output naming `<stem>.zarr`, the zarr keys, and the int32-min contour padding.
  - `prob` is a vote fraction.
  - `.png` opens as `VirtualWSIReader` with `mpp`/`power` kwargs.
  - `import tiatoolbox.models` imports `transformers`.
- Notebook 08 @ v2.1.3 uses exactly `run(images=[png], save_dir=..., patch_mode=False, input_resolutions=[baseline 1.0],
  auto_get_mask=False)` and reads `box/centroid/contours/prob/type` from zarr.
- HF weights: file name, size 151,109,291, sha256 789b2be8…, repo commit ea909bdc…, not gated.
- vqdang @ 67e2ce5e:
  - The imgaug import line (`run_utils/utils.py:9`; `ia` is otherwise only in a comment).
  - The `torch.load(...)["desc"]` line in `infer/base.py:65`.
  - The CLI flags in `run_tile.sh`, `type_info.json`, the JSON schema, and the `rm_n_mkdir` of the output subdirs.
  - The `--gpu` → `CUDA_VISIBLE_DEVICES` override.
  - The Drive ID for the PanNuke checkpoint (README).
  - Weights license CC BY-NC-SA 4.0.
- HoVer-NeXt: README and `main.py` flags, Zenodo IDs, GPL-3.0, and the torch 2.1.1 env.
- §4 conversion and overlay code was run locally on synthetic shapes (numpy 2.2.6, skimage 0.26.0, zarr 3.1). The
  results matched the true-mask regionprops exactly.

**Unverified (needs the login-node install and the 5-tile spike):**
- Actual pip resolution on Delta. The module's installed transformers, huggingface_hub, scipy, zarr, dask and cv2
  versions are unknown, so the conditional pins and the `pip check` output are unknown too. TIAToolbox may still fail
  at runtime on torch 2.8 even though 2.8 is inside its declared range.
- Whether `engine.run()` is re-entrant across chunks (the fallback is a new engine per chunk).
- Every throughput and GPU-memory number in §1.5, and whether 4 shards share the GPU cleanly.
- vqdang on numpy 2.2, skimage 0.26 and recent SciPy beyond the scanned import and API issues. Whether its checkpoint
  needs `weights_only=False` (the patch is safe either way). Whether `gdown` works from a Delta login node.
- HoVer-NeXt PNG handling of mpp and torch 2.8 compatibility. Whether TIAToolbox's `.pth` equals vqdang's PanNuke
  checkpoint (not assumed).
- Login-node internet access to PyPI, HF, GitHub and Drive (assumed from organizer docs).
- Segmentation quality on BACH at 0.42→0.25 rescale. This is judged only by the spike overlays (CELLS_SPEC §2), not
  claimed.
