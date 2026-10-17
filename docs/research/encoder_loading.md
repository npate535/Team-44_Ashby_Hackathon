# Encoder loading research: CONCH, MUSK, UNI2-h, Virchow2, H-optimus-0, Phikon-v2, PathGen-CLIP

Researched 2026-10-17 for offline embedding of BACH breast H&E tiles on Delta A100. Target environment: Python 3.11.13, torch 2.8.0+cu128,
timm 1.0.19, transformers 4.56.1 (which pins huggingface_hub >=0.34,<1.0), open_clip_torch, and `pip --user` installs of CONCH and MUSK.
Weights load from the default HF cache with `HF_HUB_OFFLINE=1`. Tile inputs are 512 px (CONCH/MUSK) and 256 px (the 224-input encoders) at about 0.42 µm/px.

## Sources read (URL @ revision)
- CONCH code: github.com/mahmoodlab/CONCH @ `141cc09` (2025-03-26). Files: `conch/open_clip_custom/{coca_model,vision_tower,factory,custom_tokenizer,transform,constants}.py`,
  `model_configs/conch_ViT-B-16.json`, `pyproject.toml`, `MANIFEST.in`, `notebooks/zeroshot_classification_example_starter.ipynb`.
  HF card: huggingface.co/MahmoodLab/CONCH (repo sha `f9ca9f8`, 2024-05-05).
- MUSK code: github.com/lilab-stanford/MUSK @ `714b666` (2025-10-26). Files: `musk/modeling.py`, `musk/utils.py`, `pyproject.toml`, `requirements.txt`,
  `musk/torchscale/component/flash_attention.py`, `benchmarks/clip_benchmark/metrics/{zeroshot_classification,linear_probe}.py`.
  HF card: huggingface.co/xiangjx/musk (sha `de1ffed`, 2025-01-19).
- HF cards: MahmoodLab/UNI2-h (sha `d517a8d`, 2025-01-16); paige-ai/Virchow2 (sha `3158645`, 2024-10-26);
  bioptimus/H-optimus-0 (sha `b145cc1`, 2025-12-12); owkin/phikon-v2 (sha `2ae989a`, 2026-01-19; read the raw card, config.json and preprocessor_config.json);
  jamessyx/PathGen-CLIP (sha `fb80c40`, 2025-04-22); jamessyx/PathGen-CLIP-L (sha `feb8cbf`); jamessyx/pathgenclip-vit-large-patch14-hf (sha `33af829`).
  Also read the github.com/PathFoundation/PathGen-1.6M README (main) and arxiv.org/html/2407.00203.
- Cross-check: github.com/mahmoodlab/TRIDENT `trident/patch_encoder_models/load.py` @ `bdeeb74` (2026-10-12). It is independent working loaders for all 7 models.
- Library internals: timm v1.0.19 `_hub.py`, `_builder.py` and `_factory.py`; huggingface_hub v0.35.0 `errors.py`; transformers v4.56.1 `setup.py`;
  open_clip v3.3.0 `factory.py` and the `model_configs/` listing.
- File lists and sizes come from `https://huggingface.co/api/models/<repo>?blobs=true` (readable without access). Gated card text was read from the rendered HF page.

## Summary table
| Model | Files to pre-download (repo: files) | Download | Input (our tile -> model) | Output dim | Precision | License / gate |
|---|---|---|---|---|---|---|
| CONCH | `MahmoodLab/CONCH`: `pytorch_model.bin`, `meta.yaml` | 0.80 GB | 512 -> 448 (about 0.48 µm/px) | probe 512, align 512 | fp32 (TRIDENT); trained fp16-AMP | CC-BY-NC-ND-4.0; gated (auto, institutional email) |
| MUSK | `xiangjx/musk`: `model.safetensors`, `tokenizer.spm` | 1.35 GB | 512 -> 384 (about 0.56 µm/px) | probe 2048 (ms_aug), align 1024 | fp16 (card + TRIDENT) | CC-BY-NC-ND-4.0; gated (MANUAL) |
| UNI2-h | `MahmoodLab/UNI2-h`: `config.json`, `pytorch_model.bin` | 2.73 GB | 256 -> 224 | 1536 (CLS) | bf16 autocast (TRIDENT; trained bf16) | CC-BY-NC-ND-4.0; gated (auto) |
| Virchow2 | `paige-ai/Virchow2`: `config.json`, `model.safetensors` | 2.53 GB | 256 -> 224 | 2560 (CLS + mean patch) | fp16 autocast (card) | CC-BY-NC-ND-4.0 + extra terms; gated (auto) |
| H-optimus-0 | `bioptimus/H-optimus-0`: `config.json`, `pytorch_model.bin` | 4.54 GB | 256 -> 224 (card: 0.5 µm/px) | 1536 | fp16 autocast (card) | Apache-2.0; gated (accept form) |
| Phikon-v2 | `owkin/phikon-v2`: `config.json`, `model.safetensors`, `preprocessor_config.json` | 1.21 GB | 256 -> 224 (20x) | 1024 (CLS) | fp32 (TRIDENT); trained fp16 | Owkin non-commercial (LICENSE.pdf); NOT gated |
| PathGen-CLIP | `jamessyx/PathGen-CLIP`: `pathgenclip.pt` | 0.60 GB | 256 -> 224 | 512 (image and text) | fp16 autocast (card) | HF metadata: cc-by-2.0; gated (auto) |
| (opt) PathGen-CLIP-L | `jamessyx/PathGen-CLIP-L`: `pathgen-clip-l.pt` | 1.71 GB | -> 336 | 768 | fp16 autocast | CC-BY-NC-4.0; NOT gated |

Download totals: about 13.76 GB for the 7 core models (13,756,113,845 bytes), or 15.47 GB with PathGen-CLIP-L. That fits the 100 GB `/u` home quota.
Do not fetch these files, which are not needed:
- `xiangjx/musk/image_tokenizer.pth` (2.53 GB, pretraining only).
- `paige-ai/Virchow2/pytorch_model.bin` (2.53 GB duplicate). timm prefers `model.safetensors`.

Version check: every model's minimum is satisfied by our environment.

| Model | Minimum required | Satisfied? |
|---|---|---|
| CONCH | timm>=0.9.8, torch>=2.0.1 | yes |
| MUSK | its requirements.txt pins timm==0.9.8 and torch==2.0.1, but these are not install deps | yes. timm 1.0.19 still ships the `timm.models.registry` / `timm.models.layers` shims it imports (FutureWarning only; verified in v1.0.19 source) |
| UNI2-h | timm>=0.9.8 | yes |
| Virchow2 | timm>=0.9.11 | yes |
| H-optimus-0 | timm 1.x | yes |
| Phikon-v2 | Dinov2Model (transformers >=4.31; config saved with 4.40.2) | yes |
| PathGen-CLIP | open_clip with local-file `pretrained` | yes, supported in v3.3.0 |

## Common offline rules (apply to every loader)
```bash
module load pytorch-conda/2.8; unset PYTHONNOUSERSITE        # unset after EVERY module load
export HF_HOME="$HOME/.cache/huggingface"                   # MUST be identical at download time and in the sbatch job
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```
- Download with the default revision (`main`). A pinned `revision=<sha>` does not write `refs/main`, and later `hf-hub:` loads (which ask for `main`) then fail offline.
  For reproducibility, record the resolved commit from the snapshot path instead (the script below writes a manifest).
- timm `hf-hub:` loads first try `model.safetensors`. Offline, a missing file raises `LocalEntryNotFoundError`. That is a subclass of `EntryNotFoundError`
  (huggingface_hub 0.35.0 `errors.py` L275), which timm catches (`_hub.py` L239), so the loader falls back to `pytorch_model.bin`.
  As a result, UNI2-h and H-optimus-0 load offline with only `config.json` + `.bin`.
- timm 1.0.19 loads hf-hub `.bin` files with `torch.load(weights_only=True)` (`_builder.py` L226). This works for these plain state dicts.
- Fallback if hf-hub resolution misbehaves: `timm.create_model("local-dir:/abs/path/to/snapshot", pretrained=True, **kwargs)`. timm 1.0.19 supports the
  `local-dir:` scheme; the directory must contain `config.json` plus the weights.
- Cache directories are named after the repo id and are case-sensitive: `MahmoodLab/CONCH` and `MahmoodLab/conch` are different directories (the CONCH card uses lowercase).
  Pre-download with the same casing the loader uses, or pass local file paths (preferred; used below for CONCH, MUSK and PathGen).
- Use `torch.autocast("cuda", dtype=...)`. `torch.cuda.amp.autocast()`, which appears on the PathGen card, is deprecated in torch 2.8.
- Cache each embedding file with: repo id, commit sha, head (`probe` or `align`), preprocessing string, input px, µm/px and dtype. Never mix models in one space.

---
## 1. CONCH (MahmoodLab/CONCH): both heads from ONE trunk pass
**What the source shows** (`coca_model.py` L207-223, `vision_tower.py`):
- `encode_image(proj_contrast=True)` calls `_encode_image`, which runs `self.visual(images)` and then normalizes.
  `VisualModel.forward` computes `trunk(x)`, then `ln_contrast(attn_pool_contrast(x)[:,0])`, then `@ proj_contrast`. It ALSO runs the 256-query `attn_pool_caption`, which is wasted work for embedding.
- `encode_image(proj_contrast=False)` calls `visual.forward_no_head(x, normalize)`: `trunk(x)`, then `ln_contrast(attn_pool_contrast(x)[:,0])`, then optional normalize.
- So the probe output is exactly the align output before `@ proj_contrast` and normalization. Both heads can come from one trunk pass, which is also cheaper than `encode_image(proj_contrast=True)`.
- The probe output is 512-d (the attentional-pool `d_model` is `embed_dim=512`). It is NOT the 768-d ViT CLS token.
- `trunk` is a timm `VisionTransformer` with `forward = forward_features` (`dynamic_img_size=True`), `image_size=448`, patch 16, so `trunk(x)` returns `[B, 1+784, 768]`.
```python
import os, torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

ckpt = hf_hub_download("MahmoodLab/CONCH", "pytorch_model.bin")          # cache hit offline
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt, device="cuda")
model.eval()
# preprocess (transform.py + factory.py): Resize(448, BICUBIC) -> CenterCrop(448) -> convert RGB -> ToTensor
#   -> Normalize(OPENAI mean (0.48145466,0.4578275,0.40821073), std (0.26862954,0.26130258,0.27577711))

@torch.inference_mode()
def conch_embed(x):                                    # x: [B,3,448,448] float32 cuda
    v = model.visual
    tok = v.trunk(x, **v.trunk_kwargs)                 # [B,785,768]
    probe = v.ln_contrast(v.attn_pool_contrast(tok)[:, 0])   # [B,512] == encode_image(x, proj_contrast=False, normalize=False)
    align = F.normalize(probe @ v.proj_contrast, dim=-1)     # [B,512] == encode_image(x, proj_contrast=True,  normalize=True)
    return probe.float(), align.float()

# one-time parity check (do this in the smoke test):
# p_ref = model.encode_image(x, proj_contrast=False, normalize=False); a_ref = model.encode_image(x)  # defaults True/True
# assert torch.allclose(probe, p_ref, atol=1e-4) and torch.allclose(align, a_ref, atol=1e-4)

# Text (README + starter notebook)
tokenizer = get_tokenizer()          # PreTrainedTokenizerFast from bundled conch_byte_level_bpe_uncased.json (no network)
tokens = tokenize(texts=prompts, tokenizer=tokenizer).to("cuda")   # [N,128]: 127 max tokens + pad placeholder for <cls>
with torch.inference_mode():
    t = model.encode_text(tokens)                     # normalize=True, embed_cls=True by default -> [N,512] unit vectors
    logits = model.logit_scale.exp() * align @ t.T    # notebook: (img @ txt.T * model.logit_scale.exp()).softmax(-1)
```
- Magnification and input: config `image_size=448`. A 512 px crop at 0.42 µm/px resizes to 448, about 0.48 µm/px (near 20x). Card use cases: linear/kNN probe and MIL use the `proj_contrast=False` features; retrieval and zero-shot use the projected, normalized features.
- Precision: the card says training used fp16 automatic mixed precision. TRIDENT runs CONCH in fp32. The ViT-B at 448 is cheap enough on an A100 to stay fp32. Use bf16 autocast only after a cosine-similarity parity check (≥0.999) against fp32.
- Pre-download: `pytorch_model.bin` (802 MB). Add `meta.yaml` (37 B) only if you use the `"hf_hub:MahmoodLab/CONCH"` string path, which downloads both. A local-path load needs only the `.bin`.
- Gotchas:
  - `load_checkpoint` uses `load_state_dict(strict=False)`, so a wrong file loads silently. Keep the parity and zero-shot sanity check.
  - Decoder weights are removed from the public checkpoint; this is irrelevant for `encode_*`.
  - `read_state_dict` uses `torch.load` with the torch 2.8 default (`weights_only=True`). The organizers' `models.py` runs this same path on `pytorch-conda/2.8` for the hosted CONCH server (inferred to work; not run by me).
  - Tokenizer and config JSONs are shipped through `MANIFEST.in`, so `pip install git+` includes them.
  - Importing the tokenizer sets `TOKENIZERS_PARALLELISM=false`.

## 2. MUSK (xiangjx/musk): both heads from ONE call
**What the source shows** (`modeling.py` `MUSK.forward`, `utils.py` `MultiScaleForward`):
- `ms_aug=True` computes `MultiScaleForward(scales=[1,2])`:
  - scale 1 = `F.interpolate(x.float(), 384, bicubic)`. A same-size bicubic resample is an identity (all sample points fall on pixel centres, cubic weights [0,1,0,0]), then the CLS token is taken.
  - scale 2 = upsample to 768, split chessboard-style into 4×384 crops, then the mean of their CLS tokens.
  - Output is `cat([CLS@1x, mean CLS@2x])` = 2048-d. Each pass calls `model(..., ms_aug=False, with_head=False, out_norm=False)`.
- With `with_head=True`, the code itself does `vision_head(vision_cls[:, :1024])`.
- So `vision_head(probe[:, :1024])` equals the `ms_aug=False, with_head=True` path up to numerics. Align = 1024-d (`vision_head` is Linear(1024,1024, bias=False)).
- Cost: one `ms_aug=True` call is 5 forward passes per image, versus 1 pass for align alone.
```python
import torch, torch.nn.functional as F, timm
from huggingface_hub import hf_hub_download
from musk import utils as musk_utils, modeling  # noqa: F401  (registers 'musk_large_patch16_384' with timm)
from transformers import XLMRobertaTokenizer
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD   # (0.5,0.5,0.5)/(0.5,0.5,0.5)

weights = hf_hub_download("xiangjx/musk", "model.safetensors")   # LOCAL PATH: avoids MUSK's force_download into ~/.cache/model.safetensors
spm     = hf_hub_download("xiangjx/musk", "tokenizer.spm")       # pip install does NOT ship musk/models/tokenizer.spm
model = timm.create_model("musk_large_patch16_384")
musk_utils.load_model_and_may_interpolate(weights, model, "model|module", "")
model = model.to("cuda", dtype=torch.float16).eval()
tokenizer = XLMRobertaTokenizer(spm)
preprocess = transforms.Compose([                                 # card, verbatim (interpolation=3 == BICUBIC)
    transforms.Resize(384, interpolation=3, antialias=True),
    transforms.CenterCrop((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)])

@torch.inference_mode()
def musk_embed(x):                                                 # x: [B,3,384,384]
    x = x.to("cuda", dtype=torch.float16)
    probe = model(image=x, with_head=False, out_norm=False, ms_aug=True, return_global=True)[0]  # [B,2048] (card's probe setting)
    align = F.normalize(model.vision_head(probe[:, :1024]), dim=-1)                            # [B,1024]
    return probe.float(), align.float()
# parity: a_ref = model(image=x, with_head=True, out_norm=True, ms_aug=False, return_global=True)[0]; allclose(align, a_ref, atol=2e-3) in fp16

@torch.inference_mode()
def musk_text(prompts):            # same as organizers' models.py; MUSK benchmark: "head must be used for zero-shot task"
    tok = [musk_utils.xlm_tokenizer(p, tokenizer, max_len=100) for p in prompts]
    ids = torch.tensor([a for a, _ in tok], device="cuda"); pad = torch.tensor([b for _, b in tok], device="cuda")
    return model(text_description=ids, padding_mask=pad, with_head=True, out_norm=True,
                 ms_aug=False, return_global=True)[1].float()                                # [N,1024]
# zero-shot: logits = model.logit_scale.exp().float() * align @ text.T
```
- Organizers' `models.py` handling (correct, keep it):
  - Weights come from `hf_hub_download("xiangjx/musk","model.safetensors")` and are passed as a local path.
  - The tokenizer falls back to `tokenizer.spm` from the HF repo because pip installs only `.py` files (no MANIFEST/package-data in MUSK's pyproject).
  - fp16 is used on CUDA. Image and text both use `with_head=True, out_norm=True, ms_aug=False`, i.e. align only; `/embed` returns 1024-d. Its comment says "3.7 GB": that is the whole repo; `model.safetensors` itself is 1.35 GB.
- The card's text example uses `with_head=False` (out_norm=True). That is wrong for image-text matching: use `with_head=True` for both modalities.
  MUSK's own linear-probe benchmark used `out_norm=True, with_head=False, ms_aug` (the card says `out_norm=False`). Either works if features are standardized before the probe.
- Magnification and input: 384 input. A 512 px crop at 0.42 µm/px resizes to 384, about 0.56 µm/px. ms_aug adds a 2x zoomed view.
- Precision: fp16. The weights are stored F16 in safetensors, and attention softmax runs in fp32. bf16 is not documented by the authors.
- Gotchas:
  - (a) Never call `load_model_and_may_interpolate("hf_hub:xiangjx/musk", ...)` offline: it uses `force_download=True` into the generic `~/.cache/model.safetensors`.
    It also silently loads ANY file already sitting at that path.
  - (b) MUSK declares no install deps. Check `einops`, `fairscale`, `sentencepiece` and `safetensors` are importable.
    `flash_attn`, `xformers` and `apex` are optional (their imports are guarded by try/except).
  - (c) Missing or unexpected keys are only printed, not raised. Grep the job log.
  - (d) Repo is MANUAL-gated: each user needs individual approval.

## 3. UNI2-h (MahmoodLab/UNI2-h)
```python
import timm, torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
timm_kwargs = {                                   # card, verbatim
    'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24, 'init_values': 1e-5,
    'embed_dim': 1536, 'mlp_ratio': 2.66667*2, 'num_classes': 0, 'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True}
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).eval().to("cuda")
card_hub_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))  # card (hf-hub path); print it once
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),   # card manual path + TRIDENT
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    emb = model(x).float()                        # [B,1536] class token
```
- Local-weights alternative from the card: `timm.create_model('vit_giant_patch14_224', pretrained=False, **timm_kwargs)`, then `load_state_dict(torch.load(path, map_location='cpu'), strict=True)`.
- Transform caveat: I could not read the gated `config.json`. If its `crop_pct` is below 1 (timm default 0.875), `create_transform` would Resize(256) then CenterCrop(224) a 256 px tile.
  That crops the tile instead of resizing it, changing µm/px. Use the explicit Resize(224) transform above for consistent magnification across encoders, and log `resolve_data_config(...)`.
- Precision: the card says training was "bf16 using PyTorch-FSDP mixed-precision"; TRIDENT uses bf16. Keep bf16, not fp16.

## 4. Virchow2 (paige-ai/Virchow2)
```python
import timm, torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                          mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).eval().to("cuda")   # card, verbatim
transforms_ = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(x)                              # [B,261,1280]: 0=CLS, 1..4 = register tokens, 5: = 256 patch tokens
class_token = output[:, 0]
patch_tokens = output[:, 5:]                       # card: "tokens 1-4 are register tokens so we ignore those"
embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1).float()   # [B,2560]
```
- The card notes the output is fp32 under autocast because the final op is a LayerNorm.
- Kwargs TRIDENT uses explicitly (they mirror the config): `img_size=224, init_values=1e-5, num_classes=0, reg_tokens=4, mlp_ratio=5.3375, global_pool="", dynamic_img_size=True`,
  plus SwiGLUPacked/SiLU. Arch `vit_huge_patch14_224` (for `local-dir:` or `strict=True` local loads).
- TRIDENT transform: Resize(224, BICUBIC), ToTensor, ImageNet mean/std. Resize 256 px tiles to 224: with `dynamic_img_size`, a 256 input would silently run at 18×18 patches and a different µm/px.
- Magnification: trained on mixed 5x/10x/20x/40x tiles (2.0/1.0/0.5/0.25 µm/px). A ~0.48 µm/px input is in-distribution.
- License gotcha: the extra terms forbid use "as a medical device, clinical support, diagnostic tool ... including RUO/IUO". Frame any Virchow2 result as academic research only.
- Download only `model.safetensors`; the repo also has a duplicate 2.5 GB `pytorch_model.bin`.

## 5. H-optimus-0 (bioptimus/H-optimus-0)
```python
import timm, torch
from torchvision import transforms
model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                          init_values=1e-5, dynamic_img_size=False).eval().to("cuda")      # card, verbatim
transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),       # ADDED: card assumes tiles are already 224 px @ 0.5 µm/px
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517))])   # card
with torch.autocast(device_type="cuda", dtype=torch.float16), torch.inference_mode():
    features = model(x).float()                    # [B,1536]  (card asserts (1,1536))
```
- With `dynamic_img_size=False`, input must be exactly 224×224.
- 1.1B params; the fp32 `.bin` is 4.54 GB and the repo has no safetensors. timm falls back to `.bin` offline (see the common rules). Card recommends fp16 autocast.
- TRIDENT's local route: `vit_giant_patch14_reg4_dinov2` with `num_classes=0, global_pool="token", init_values=1e-5`, then `strict=True`.
- The card was updated 2025-12-12. The current card transform has no Resize; earlier copies differ.

## 6. Phikon-v2 (owkin/phikon-v2): not gated
```python
import torch
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")      # BitImageProcessor
model = AutoModel.from_pretrained("owkin/phikon-v2").eval().to("cuda")  # Dinov2Model, ViT-L/16, 24 layers, no registers
inputs = processor(images=list_of_pil_rgb, return_tensors="pt")         # card uses processor(image, return_tensors="pt")
with torch.inference_mode():
    outputs = model(pixel_values=inputs["pixel_values"].to("cuda"))
    features = outputs.last_hidden_state[:, 0, :].float()               # [B,1024] CLS (card)
```
- `preprocessor_config.json` (read directly): `do_convert_rgb`, resize `shortest_edge=224` with `resample=3` (BICUBIC), center crop 224×224, rescale 1/255, ImageNet mean/std.
  DataLoader-friendly equivalent: `Resize(224, BICUBIC) -> CenterCrop(224) -> ToTensor -> Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))`.
  TRIDENT uses the default bilinear resize; prefer bicubic to match the processor.
- Card: trained on 224 px tiles at 20x, fp16 FSDP; TRIDENT runs fp32. fp32 is fine for a ViT-L.
- transformers 4.56 may warn about a slow image processor. That is harmless, or avoid it by using the torchvision transform.

## 7. PathGen-CLIP (jamessyx/PathGen-CLIP)
- Repo contents: exactly one weights file, `pathgenclip.pt` (598,593,445 B), the same size as the public mirror `pub-7a38cc906afa44a4a01533c288d0b1af.r2.dev/pathgenclip.pt`.
  I inspected it with a static, non-executing pickle walk of the first 4 MB:
  - plain `OrderedDict` state dict, no `module.` prefix, FloatStorage only (so it loads with open_clip's `weights_only=True`);
  - `visual.conv1.weight [768,3,16,16]`, `visual.positional_embedding [197,768]`, `visual.proj [768,512]`, `positional_embedding [77,512]`, `token_embedding [49408,512]`, 12+12 blocks.
  This matches open_clip **ViT-B-16 at 224**. The ViT-L variant is a SEPARATE repo, `jamessyx/PathGen-CLIP-L` (`pathgen-clip-l.pt`), checked the same way:
  `visual.positional_embedding [577,1024]`, `conv1 [1024,3,14,14]`, embed 768. That is **ViT-L-14-336**, not ViT-B-16.
- The organizers' spec `openclip:ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt` is CORRECT. `models.py` parses it to repo `jamessyx/PathGen-CLIP`, file `pathgenclip.pt`, arch `ViT-B-16`.
  The card's commented PathGen-CLIP-L line (`'ViT-B-16'` with `pathgen-clip-l.pt`) is wrong and would fail on shape mismatch.
  For L use `ViT-L-14-336` (or `ViT-L-14-336-quickgelu`; see the caveat below).
```python
import torch, open_clip
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("jamessyx/PathGen-CLIP", "pathgenclip.pt")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained=ckpt)   # card
model = model.eval().to("cuda")
tokenizer = open_clip.get_tokenizer("ViT-B-16")      # bundled BPE vocab, no network; context 77
# preprocess: Resize(224, BICUBIC) -> CenterCrop(224) -> RGB -> ToTensor -> Normalize(OPENAI mean/std)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    img = model.encode_image(x);                      img = img / img.norm(dim=-1, keepdim=True)   # [B,512]
    txt = model.encode_text(tokenizer(prompts).to("cuda")); txt = txt / txt.norm(dim=-1, keepdim=True)
probs = (100.0 * img.float() @ txt.float().T).softmax(dim=-1)   # card uses a fixed 100; model.logit_scale.exp() also exists
```
- QuickGELU caveat (unverified): open_clip `ViT-B-16` uses nn.GELU, while OpenAI-initialised CLIPs use QuickGELU. The HF conversion of PathGen-CLIP-L declares `hidden_act: quick_gelu`.
  For the B-16 model, follow the card (`ViT-B-16`). As a quick check, A/B zero-shot accuracy on about 100 labeled tiles against
  `create_model_and_transforms("ViT-B-16-quickgelu", pretrained=ckpt)` (that config exists in open_clip); report whichever is used.
- Published context (not ours): the PathGen paper reports BACH zero-shot accuracy of 59.5% for PathGen-CLIP, with PathGen-CLIP-L about 12 points higher.

---
## Pre-download script (Delta LOGIN node; network on; relies on an existing `hf auth login`; never prints the token)
Save it as e.g. `~/team44/predownload_encoders.sh` (home dir, not a shared folder) and run `bash ~/team44/predownload_encoders.sh`.
To include PathGen-CLIP-L, run `WITH_PATHGEN_L=1 bash ...`. The script only does I/O, which is allowed on login nodes.
```bash
#!/usr/bin/env bash
set -euo pipefail
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE                      # downloads need the network
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"          # job must use the SAME HF_HOME
python3 -I - "${WITH_PATHGEN_L:-0}" <<'PY'
import json, os, sys
from huggingface_hub import hf_hub_download, get_token, try_to_load_from_cache
from huggingface_hub.errors import GatedRepoError
if not get_token():
    sys.exit("No Hugging Face token found. Run `hf auth login` first (the token is never printed).")
FILES = {
    "MahmoodLab/CONCH":      ["pytorch_model.bin", "meta.yaml"],
    "xiangjx/musk":          ["model.safetensors", "tokenizer.spm"],
    "MahmoodLab/UNI2-h":     ["config.json", "pytorch_model.bin"],
    "paige-ai/Virchow2":     ["config.json", "model.safetensors"],
    "bioptimus/H-optimus-0": ["config.json", "pytorch_model.bin"],
    "owkin/phikon-v2":       ["config.json", "model.safetensors", "preprocessor_config.json"],
    "jamessyx/PathGen-CLIP": ["pathgenclip.pt"],
}
if sys.argv[1] == "1":
    FILES["jamessyx/PathGen-CLIP-L"] = ["pathgen-clip-l.pt"]
manifest, failed = {}, []
for repo, names in FILES.items():
    for fn in names:
        try:
            p = hf_hub_download(repo_id=repo, filename=fn)      # default revision 'main' -> writes refs/main
            commit = p.split("/snapshots/")[1].split("/")[0]
            manifest.setdefault(repo, {})[fn] = {"commit": commit, "bytes": os.path.getsize(p)}
            print(f"OK    {repo}/{fn}  {os.path.getsize(p)/1e9:.2f} GB  @{commit[:7]}")
        except GatedRepoError:
            failed.append(f"{repo}/{fn}")
            print(f"GATED {repo}: request/accept access on huggingface.co with THIS account, then rerun")
        except Exception as e:                                     # HF errors carry no token
            failed.append(f"{repo}/{fn}")
            print(f"FAIL  {repo}/{fn}: {type(e).__name__}: {str(e)[:160]}")
# offline-readiness check: every file must now resolve from the cache alone
missing = [f"{r}/{f}" for r, fs in FILES.items() for f in fs if not isinstance(try_to_load_from_cache(r, f), str)]
out = os.path.join(os.environ["HF_HOME"], "team44_encoder_manifest.json")
with open(out, "w") as fh:
    json.dump(manifest, fh, indent=1)
print(f"manifest -> {out}; missing from cache: {missing or 'none'}")
sys.exit(1 if (failed or missing) else 0)
PY
python3 -c "import einops, fairscale, sentencepiece, safetensors, open_clip, conch, musk, timm, transformers; \
print('deps OK  timm', timm.__version__, ' transformers', transformers.__version__, ' open_clip', open_clip.__version__)"
```

## Smoke-test checklist (first GPU job, about 64 tiles, before scaling)
1. Run with `HF_HUB_OFFLINE=1` and assert output shapes:

   | Model | Expected shape |
   |---|---|
   | CONCH | (B,512) ×2 |
   | MUSK | (B,2048) + (B,1024) |
   | UNI2-h | (B,1536) |
   | Virchow2 | (B,2560) |
   | H-optimus-0 | (B,1536) |
   | Phikon-v2 | (B,1024) |
   | PathGen-CLIP | (B,512) |

2. Parity: CONCH one-pass vs `encode_image` (both heads); MUSK one-call align vs `ms_aug=False` align. Also check the norms of align vectors are about 1.
3. Grep the logs for MUSK "Weights ... not initialized" / "not used" and timm missing-key warnings. CONCH's strict=False hides these, so also compare against
   the hosted CONCH/MUSK `/embed` output on the same tiles (expect cosine about 1, given identical preprocessing).
4. Run a mixed-precision vs fp32 cosine check on 16 tiles for any model run below fp32. Record dtype in the metadata.

## Verified vs unverified
**Verified** (read the source, card or API directly on 2026-10-17):
- CONCH: the probe/align split happens after `ln_contrast` (code); both heads are 512-d; the tokenizer and preprocessing pipeline; OpenAI mean/std; `strict=False`; the tokenizer ships with pip.
- MUSK: `ms_aug` with `with_head` applies `vision_head` to `[:, :1024]` (code); scale 1 is a same-size bicubic resample; MUSK declares no install deps;
  `tokenizer.spm` is not pip-installed; the hf_hub path force-downloads to `~/.cache/model.safetensors`.
- All model-card loading snippets, kwargs, normalization constants and output dims quoted above.
- HF file lists, sizes, gate types and license metadata for all 8 repos.
- PathGen-CLIP and PathGen-CLIP-L architectures, from checkpoint tensor shapes (static pickle parse, nothing executed). The organizers' PathGen spec parses correctly.
- timm 1.0.19: hf-hub safetensors fallback, `weights_only=True`, `local-dir:` scheme, and the deprecated-import shims.
  huggingface_hub 0.35.0: `LocalEntryNotFoundError` subclasses `EntryNotFoundError`. transformers 4.56.1 pins hub <1.0 and timm <=1.0.19.

**Unverified** (needs the Delta smoke test or gated access):
- Nothing was run. No weights were loaded and no GPU was used.
- Gated `config.json` contents for UNI2-h, Virchow2 and H-optimus-0 (crop_pct/interpolation; inferred from cards and TRIDENT).
- Whether CONCH's `.bin` loads under torch 2.8's `weights_only=True` default. This is inferred from the organizers running the same code on the same module.
- Numerical tolerance of the one-pass parities (expected exact or near-exact).
- `einops` availability in `pytorch-conda/2.8`.
- The open_clip version installed on Delta.
- The QuickGELU vs GELU choice for PathGen-CLIP.
- Whether this account has been approved for UNI2-h, Virchow2 and H-optimus-0. The brief only reports approval for CONCH, MUSK and PathGen-CLIP.
- bf16 safety for CONCH and Phikon-v2.
