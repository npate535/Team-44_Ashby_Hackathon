"""Foundation-model encoders for tile embedding and prompt encoding (BUILD_PLAN Task 3, §A6-§A8; Agent 2).

One loader per encoder in config/encoders.yaml, each following its model card and docs/research/encoder_loading.md:

    conch       official adapter (resources/official/models.py); ONE trunk pass feeds both heads:
                align = proj_contrast + normalize (512, == hosted /embed), probe = pre-projection (512)
    musk        official adapter; ONE ms_aug call gives probe (2048) and align = normalize(vision_head(probe[:, :1024]))
    pathgenclip official open_clip adapter (ViT-B-16 + local pathgenclip.pt); probe = encode_image, align = normalized
    uni2h, hoptimus0, virchow2   timm hf-hub loaders with the card kwargs; probe head only
    phikonv2    transformers AutoModel CLS; probe head only
    test        the organizers' deterministic `test` adapter (CPU, no weights): lets the whole engine run in tests

The CONCH, MUSK and PathGen-CLIP loaders reuse the organizers' adapters unchanged (import, never edit): the model
object is read from the adapter's closure, so weights, preprocessing and precision are exactly the adapter's, and
`Encoder.reference` returns the adapter's own output for the align == adapter parity check in 02_embed.

    from pipeline import encoders
    ecfg = encoders.load_encoders_config()
    enc = encoders.load_encoder("conch", ecfg, device="cuda")   # EncoderUnavailable if weights are not cached
    heads = enc.embed(batch)                                    # {"align": [B,512], "probe": [B,512]} float32
    text = enc.encode_texts(["an H&E image of tumor"])          # [N,512] unit vectors (text-aligned encoders)

Delta-portable: Python >= 3.10, the pytorch-conda/2.8 packages plus the organizer installs (open_clip, CONCH, MUSK).
torch, timm, transformers and huggingface_hub are imported lazily so tests on a laptop only need numpy + Pillow.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ENCODERS_YAML = REPO_ROOT / "config" / "encoders.yaml"
OFFICIAL_MODELS = REPO_ROOT / "resources" / "official" / "models.py"
PROMPTS_NCT = REPO_ROOT / "resources" / "official" / "prompts_nct.json"

LOADERS = ("official_conch", "official_musk", "official_openclip", "official_test", "timm_hub", "hf_automodel")
AUTOCAST = (None, "fp16", "bf16")
INTERPOLATIONS = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}
_SPEC_KEYS = ("loader", "model_card", "license", "input_px", "resize", "crop", "interpolation", "mean", "std",
              "native_grid", "heads", "autocast", "text_aligned")
_TIMM_SYMBOLS = ("SwiGLUPacked", "SiLU")


class EncoderUnavailable(RuntimeError):
    """The encoder's weights are not in the local HF cache (gated access not granted / not downloaded yet). The smoke
    job skips such encoders with --skip-unavailable; every other failure (imports included) is a real error."""


# ---- config ------------------------------------------------------------------------------------------------------
def load_encoders_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config/encoders.yaml and validate it strictly (a missing key raises, never defaults)."""
    path = Path(path) if path is not None else ENCODERS_YAML
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    problems = encoders_config_problems(data)
    if problems:
        raise ValueError(f"{path} is invalid:\n- " + "\n- ".join(problems))
    return data


def encoders_config_problems(data: Any) -> list[str]:
    if not isinstance(data, Mapping):
        return ["encoders.yaml: top level must be a mapping"]
    problems = [f"missing top-level key {k}" for k in ("encoders", "engine", "checks", "hosted_check", "smoke")
                if k not in data]
    for name, spec in (data.get("encoders") or {}).items():
        if not isinstance(spec, Mapping):
            problems.append(f"encoders.{name} must be a mapping")
            continue
        problems += [f"encoders.{name}: missing key {k}" for k in _SPEC_KEYS if k not in spec]
        if spec.get("loader") not in LOADERS:
            problems.append(f"encoders.{name}.loader must be one of {list(LOADERS)}, got {spec.get('loader')!r}")
        if spec.get("autocast") not in AUTOCAST:
            problems.append(f"encoders.{name}.autocast must be one of {list(AUTOCAST)}")
        if spec.get("interpolation") not in INTERPOLATIONS:
            problems.append(f"encoders.{name}.interpolation must be one of {sorted(INTERPOLATIONS)}")
        heads = spec.get("heads")
        if not isinstance(heads, Mapping) or not heads or any(h not in ("align", "probe") for h in heads):
            problems.append(f"encoders.{name}.heads must map align/probe to {{dim, normalized}}")
        else:
            for head, h in heads.items():
                if not (isinstance(h, Mapping) and isinstance(h.get("dim"), int) and h["dim"] > 0
                        and isinstance(h.get("normalized"), bool)):
                    problems.append(f"encoders.{name}.heads.{head} needs an int dim > 0 and a bool normalized")
                elif head == "align" and not h["normalized"]:
                    problems.append(f"encoders.{name}.heads.align must be normalized")
        if spec.get("text_aligned") and "align" not in (heads or {}):
            problems.append(f"encoders.{name}: text_aligned encoders need an align head")
        if spec.get("text_aligned") and not isinstance(spec.get("nct_floor"), (int, float)):
            problems.append(f"encoders.{name}: text_aligned encoders need a numeric nct_floor")
        if spec.get("loader", "").startswith("official_") and "spec" not in spec:
            problems.append(f"encoders.{name}: official loaders need the resolve_adapter spec")
        if spec.get("loader") in ("timm_hub", "hf_automodel") and ("repo" not in spec or "output" not in spec):
            problems.append(f"encoders.{name}: {spec.get('loader')} needs repo and output")
    return problems


def encoder_spec(ecfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        return ecfg["encoders"][name]
    except KeyError:
        raise ValueError(f"unknown encoder {name!r}; config/encoders.yaml defines {sorted(ecfg['encoders'])}") from None


def check_native_grids(ecfg: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    """settings.encoders.native_grid and encoders.yaml must agree for every encoder named in settings."""
    bad = {name: (grid, ecfg["encoders"].get(name, {}).get("native_grid"))
           for name, grid in cfg["encoders"]["native_grid"].items()
           if ecfg["encoders"].get(name, {}).get("native_grid") != grid}
    if bad:
        raise ValueError(f"native grids disagree (settings, encoders.yaml): {bad}")


def batch_size(ecfg: Mapping[str, Any], name: str) -> int:
    sizes = ecfg["engine"]["batch_size"]
    return int(sizes.get(name, sizes["default"]))


# ---- image transforms -------------------------------------------------------------------------------------------------
def dihedral(image: Image.Image, view: int) -> Image.Image:
    """Dihedral view 0-7 (core.schemas EMBED_MANIFEST): rotate 90° counter-clockwise (view % 4) times, then mirror
    left-right when view >= 4. View 0 is the identity."""
    if isinstance(view, bool) or not isinstance(view, (int, np.integer)) or not 0 <= int(view) <= 7:
        raise ValueError(f"view must be an integer 0-7, got {view!r}")
    view = int(view)
    turns = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)[view % 4]
    out = image.transpose(turns) if turns is not None else image
    return out.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if view >= 4 else out


def crop_box(image: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Crop (x, y, w, h); (0, 0, image width, image height) is the whole image. Out-of-bounds boxes raise."""
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > image.width or y + h > image.height:
        raise ValueError(f"crop box {(x, y, w, h)} is outside the {image.width}x{image.height} image")
    if (x, y, w, h) == (0, 0, image.width, image.height):
        return image
    return image.crop((x, y, x + w, y + h))


@dataclass(frozen=True)
class Preprocess:
    """Card preprocessing for the non-adapter encoders: Resize(shorter side -> resize) -> CenterCrop(crop) ->
    ToTensor -> Normalize(mean, std). On PIL inputs torchvision's Resize calls PIL's resize, so this matches the
    cards' torchvision pipelines without needing torchvision."""

    resize: int
    crop: int
    interpolation: str
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __call__(self, image: Image.Image) -> Any:
        import torch

        image = image.convert("RGB")
        w, h = image.size
        if w <= h:
            size = (self.resize, int(self.resize * h / w))
        else:
            size = (int(self.resize * w / h), self.resize)
        if size != (w, h):
            image = image.resize(size, INTERPOLATIONS[self.interpolation])
        w, h = image.size
        left, top = int(round((w - self.crop) / 2.0)), int(round((h - self.crop) / 2.0))
        if (w, h) != (self.crop, self.crop):
            image = image.crop((left, top, left + self.crop, top + self.crop))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Preprocess:
        return cls(int(spec["resize"]), int(spec["crop"]), str(spec["interpolation"]),
                   tuple(float(v) for v in spec["mean"]), tuple(float(v) for v in spec["std"]))


def preprocess_info(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {"resize": spec["resize"], "crop": spec["crop"], "interpolation": spec["interpolation"],
            "mean": list(spec["mean"]), "std": list(spec["std"]), "input_px": spec["input_px"]}


# ---- the encoder object -----------------------------------------------------------------------------------------------
@dataclass
class Encoder:
    """A loaded encoder. `forward(x)` maps a preprocessed batch on `device` to {head: tensor [B, D]}; `reference(x)`
    (official adapters only) returns the adapter's own outputs {head: ndarray} for the parity check."""

    name: str
    spec: Mapping[str, Any]
    device: str
    preprocess: Callable[[Image.Image], Any]
    forward: Callable[[Any], dict[str, Any]]
    info: dict[str, Any]
    reference: Callable[[Any], dict[str, np.ndarray]] | None = None
    text_fn: Callable[[list[str]], np.ndarray] | None = None
    notes: list[str] = field(default_factory=list)
    set_one_pass: Callable[[bool], None] | None = None  # parity fallback: switch to the adapter's own two calls

    @property
    def heads(self) -> dict[str, dict[str, Any]]:
        return {h: dict(v) for h, v in self.spec["heads"].items()}

    @property
    def text_aligned(self) -> bool:
        return bool(self.spec["text_aligned"]) and self.text_fn is not None

    def autocast_context(self, enabled: bool = True) -> Any:
        dtype = self.spec["autocast"]
        if not enabled or dtype is None or not self.device.startswith("cuda"):
            return contextlib.nullcontext()
        import torch

        return torch.autocast("cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[dtype])

    def embed(self, batch: Any, *, use_autocast: bool = True) -> dict[str, np.ndarray]:
        """{head: float32 ndarray [B, D]}; raises ValueError on a head/dimension mismatch."""
        import torch

        with torch.inference_mode(), self.autocast_context(use_autocast):
            out = self.forward(batch.to(self.device, non_blocking=True))
        arrays = {h: v.float().cpu().numpy() if hasattr(v, "cpu") else np.asarray(v, dtype=np.float32)
                  for h, v in out.items()}
        check_head_dims(self.name, self.spec, arrays, n=int(batch.shape[0]))
        return arrays

    def encode_texts(self, texts: Sequence[str], batch: int = 64) -> np.ndarray:
        if self.text_fn is None:
            raise ValueError(f"encoder {self.name!r} has no text encoder")
        parts = [np.asarray(self.text_fn(list(texts[i:i + batch])), dtype=np.float32)
                 for i in range(0, len(texts), batch)]
        return np.concatenate(parts) if parts else np.zeros((0, self.spec["heads"]["align"]["dim"]), np.float32)


def check_head_dims(name: str, spec: Mapping[str, Any], arrays: Mapping[str, np.ndarray], n: int) -> None:
    want = {h: int(v["dim"]) for h, v in spec["heads"].items()}
    if set(arrays) != set(want):
        raise ValueError(f"{name}: heads {sorted(arrays)} != expected {sorted(want)}")
    for head, arr in arrays.items():
        if arr.shape != (n, want[head]):
            raise ValueError(f"{name}/{head}: output shape {arr.shape}, expected ({n}, {want[head]})")


# ---- head math (module functions so tests can drive them with fake modules) -------------------------------------------
def conch_heads(model: Any, x: Any, *, one_pass: bool = True) -> dict[str, Any]:
    """CONCH align + probe. One pass: trunk -> attn_pool_contrast -> ln_contrast = probe (== encode_image(
    proj_contrast=False, normalize=False)); align = normalize(probe @ proj_contrast) (== encode_image())."""
    import torch.nn.functional as F

    if one_pass:
        v = model.visual
        tokens = v.trunk(x, **(getattr(v, "trunk_kwargs", None) or {}))
        probe = v.ln_contrast(v.attn_pool_contrast(tokens)[:, 0])
        align = F.normalize((probe @ v.proj_contrast).float(), dim=-1)
    else:
        probe = model.encode_image(x, proj_contrast=False, normalize=False)
        align = model.encode_image(x, proj_contrast=True, normalize=True)
    return {"align": align, "probe": probe}


def musk_heads(model: Any, x: Any, *, align_dim: int, dtype: Any = None, one_pass: bool = True) -> dict[str, Any]:
    """MUSK probe = ms_aug CLS features (2048, card probe setting); align = normalize(vision_head(probe[:, :1024]))
    (the code's own with_head path), else a second with_head/out_norm call when one_pass is off."""
    import torch.nn.functional as F

    if dtype is not None:
        x = x.to(dtype=dtype)
    probe = model(image=x, with_head=False, out_norm=False, ms_aug=True, return_global=True)[0]
    if one_pass:
        align = F.normalize(model.vision_head(probe[:, :align_dim]).float(), dim=-1)
    else:
        align = model(image=x, with_head=True, out_norm=True, ms_aug=False, return_global=True)[0]
    return {"align": align, "probe": probe}


def clip_heads(model: Any, x: Any) -> dict[str, Any]:
    """open_clip: probe = encode_image (projected, unnormalized), align = its L2-normalized copy (adapter output)."""
    import torch.nn.functional as F

    probe = model.encode_image(x)
    return {"align": F.normalize(probe.float(), dim=-1), "probe": probe}


def token_heads(out: Any, output: str, register_tokens: int = 0) -> dict[str, Any]:
    """timm / transformers outputs: 'cls' = pooled [B, D] (or token 0 of [B, T, D]); 'cls_meanpatch' = CLS ⊕ mean of
    the patch tokens after the register tokens (Virchow2 card)."""
    import torch

    if output == "cls":
        return {"probe": out if out.ndim == 2 else out[:, 0]}
    if output == "cls_meanpatch":
        if out.ndim != 3:
            raise ValueError(f"cls_meanpatch needs token outputs [B, T, D], got shape {tuple(out.shape)}")
        return {"probe": torch.cat([out[:, 0], out[:, 1 + register_tokens:].mean(1)], dim=-1)}
    raise ValueError(f"unknown output kind {output!r}")


# ---- official adapters -------------------------------------------------------------------------------------------------
_OFFICIAL: Any = None


def official_models() -> Any:
    """resources/official/models.py imported under a private name (it is not a package; kept unmodified)."""
    global _OFFICIAL
    if _OFFICIAL is None:
        spec = importlib.util.spec_from_file_location("pathqueue_official_models", OFFICIAL_MODELS)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {OFFICIAL_MODELS}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _OFFICIAL = module
    return _OFFICIAL


def closure_value(fn: Callable, name: str) -> Any:
    """A free variable of a (possibly decorated) closure, e.g. the `model` an official adapter's encode_images
    closes over. Raises when the adapter code no longer has it (the organizers changed models.py)."""
    nonlocals = inspect.getclosurevars(inspect.unwrap(fn)).nonlocals
    if name not in nonlocals:
        raise RuntimeError(f"official adapter function {fn.__name__} no longer closes over {name!r}; "
                           f"found {sorted(nonlocals)}. Update pipeline/encoders.py for the new models.py.")
    return nonlocals[name]


def _l2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def _test_text_fn(dim: int) -> Callable[[list[str]], np.ndarray]:
    """Deterministic text stand-in for the `test` encoder. The official test adapter seeds from Python's salted
    hash(), which changes every process; tests need the same vector for the same text in every run."""

    def encode(texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
            rows.append(np.random.default_rng(seed).standard_normal(dim))
        return _l2(np.asarray(rows, dtype=np.float32).reshape(len(texts), dim))

    return encode


def _load_official(name: str, spec: Mapping[str, Any], device: str) -> Encoder:
    import torch

    mod = official_models()
    try:
        adapter = mod.resolve_adapter(spec["spec"], device=device)
    except ImportError as exc:  # a broken install is a real failure, never "unavailable"
        raise RuntimeError(f"{name}: import failed ({exc}); run pipeline/jobs/enc_env.sh") from exc
    loader = spec["loader"]
    one_pass = bool(spec.get("one_pass", True))
    reference: Callable[[Any], dict[str, np.ndarray]]
    text_fn = adapter.encode_texts
    notes: list[str] = []
    state = {"one_pass": one_pass}
    info = encoder_info(name, spec, device, revision=_revision(spec))
    info["one_pass"] = one_pass

    def set_one_pass(flag: bool) -> None:
        state["one_pass"] = bool(flag)
        info["one_pass"] = bool(flag)
    if loader == "official_conch":
        model = closure_value(adapter.encode_images, "model")

        def forward(x: Any) -> dict[str, Any]:
            if state["one_pass"]:
                try:
                    return conch_heads(model, x, one_pass=True)
                except (AttributeError, TypeError, IndexError) as exc:  # CONCH internals changed: use its own API
                    notes.append(f"conch one-pass path failed ({type(exc).__name__}: {exc}); "
                                 "switched to the two encode_image calls")
                    set_one_pass(False)
            return conch_heads(model, x, one_pass=False)

        def reference(x: Any) -> dict[str, np.ndarray]:
            with torch.inference_mode():
                probe = model.encode_image(x.to(device), proj_contrast=False, normalize=False)
            return {"align": adapter.encode_images(x), "probe": probe.float().cpu().numpy()}
    elif loader == "official_musk":
        model = closure_value(adapter.encode_images, "model")
        dtype = closure_value(adapter.encode_images, "dtype")
        align_dim = int(spec["heads"]["align"]["dim"])

        def forward(x: Any) -> dict[str, Any]:
            if state["one_pass"]:
                try:
                    return musk_heads(model, x, align_dim=align_dim, dtype=dtype, one_pass=True)
                except (AttributeError, TypeError, IndexError) as exc:  # no vision_head: use the with_head call
                    notes.append(f"musk one-pass path failed ({type(exc).__name__}: {exc}); "
                                 "switched to a second with_head call")
                    set_one_pass(False)
            return musk_heads(model, x, align_dim=align_dim, dtype=dtype, one_pass=False)

        def reference(x: Any) -> dict[str, np.ndarray]:
            return {"align": adapter.encode_images(x)}
    elif loader == "official_openclip":
        model = closure_value(adapter.encode_images, "model")

        def forward(x: Any) -> dict[str, Any]:
            return clip_heads(model, x)

        def reference(x: Any) -> dict[str, np.ndarray]:
            return {"align": adapter.encode_images(x)}
    elif loader == "official_test":
        def forward(x: Any) -> dict[str, Any]:
            probe = torch.from_numpy(np.asarray(adapter.encode_images(x.cpu()), dtype=np.float32))
            return {"align": torch.from_numpy(_l2(probe.numpy())), "probe": probe}

        def reference(x: Any) -> dict[str, np.ndarray]:
            return {"align": _l2(adapter.encode_images(x.cpu()))}

        text_fn = _test_text_fn(int(spec["heads"]["align"]["dim"]))
        notes.append("official test adapter: meaningless deterministic embeddings, never for results")
    else:  # pragma: no cover - guarded by encoders_config_problems
        raise ValueError(loader)
    enc = Encoder(name, spec, device, adapter.preprocess, forward, info, reference, text_fn, notes)
    if loader in ("official_conch", "official_musk"):
        enc.set_one_pass = set_one_pass
    return enc


def _timm_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
    import timm
    import torch

    symbols = {"SwiGLUPacked": timm.layers.SwiGLUPacked, "SiLU": torch.nn.SiLU}
    return {k: symbols[v] if isinstance(v, str) and v in _TIMM_SYMBOLS else v
            for k, v in dict(spec.get("timm_kwargs") or {}).items()}


def add_lora(model: Any, r: int, alpha: float) -> list[Any]:
    """Wrap every attention qkv and proj Linear of a timm ViT with a rank-r LoRA adapter (pipeline/05c_finetune.py).
    lora_b starts at zero, so the wrapped model is initially identical to the base. Returns the adapter parameters."""
    import math

    import torch

    class LoRALinear(torch.nn.Module):
        def __init__(self, base: Any) -> None:
            super().__init__()
            self.base, self.scale = base, float(alpha) / int(r)
            self.lora_a = torch.nn.Parameter(torch.empty(int(r), base.in_features))
            self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, int(r)))
            torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        def forward(self, x: Any) -> Any:
            return self.base(x) + (x @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype) * self.scale

    params = []
    for block in model.blocks:
        for attr in ("qkv", "proj"):
            wrapped = LoRALinear(getattr(block.attn, attr))
            setattr(block.attn, attr, wrapped)
            params += [wrapped.lora_a, wrapped.lora_b]
    return params


def lora_state(model: Any) -> dict[str, Any]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_" in k}


def lora_weights_path(spec: Mapping[str, Any]) -> Path:
    return Path(str(spec["lora"]["weights"])).expanduser()


def _load_timm(name: str, spec: Mapping[str, Any], device: str) -> Encoder:
    import timm

    model = timm.create_model(f"hf-hub:{spec['repo']}", pretrained=True, **_timm_kwargs(spec))
    if spec.get("lora"):                       # fine-tuned variant (uni2h_ft): base weights + private LoRA adapters
        import torch

        add_lora(model, int(spec["lora"]["r"]), float(spec["lora"]["alpha"]))
        state = torch.load(lora_weights_path(spec), map_location="cpu")
        expected = set(lora_state(model))
        if set(state["lora"]) != expected:
            raise ValueError(f"{name}: LoRA weights do not match the model ({len(state['lora'])} vs {len(expected)} tensors)")
        model.load_state_dict(state["lora"], strict=False)
    model = model.eval().to(device)
    output, registers = spec["output"], int(spec.get("register_tokens", 0))

    def forward(x: Any) -> dict[str, Any]:
        return token_heads(model(x), output, registers)

    enc = Encoder(name, spec, device, Preprocess.from_spec(spec), forward, {})
    enc.info = encoder_info(name, spec, device, revision=_revision(spec))
    enc.info["timm_pretrained_cfg"] = {k: v for k, v in dict(getattr(model, "pretrained_cfg", {}) or {}).items()
                                       if k in ("input_size", "interpolation", "crop_pct", "mean", "std")}
    return enc


def _load_hf(name: str, spec: Mapping[str, Any], device: str) -> Encoder:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(spec["repo"]).eval().to(device)
    output = spec["output"]

    def forward(x: Any) -> dict[str, Any]:
        return token_heads(model(pixel_values=x).last_hidden_state, output)

    enc = Encoder(name, spec, device, Preprocess.from_spec(spec), forward, {})
    enc.info = encoder_info(name, spec, device, revision=_revision(spec))
    return enc


# ---- availability, revision and metadata ---------------------------------------------------------------------------------
def _hf_repo_file(spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    if spec["loader"] == "official_test":
        return []
    repo = spec.get("repo")
    return [(repo, f) for f in spec.get("files") or []] if repo else []


def missing_files(spec: Mapping[str, Any]) -> list[str]:
    """Files of the encoder that are not in the local HF cache (offline readiness)."""
    pairs = _hf_repo_file(spec)
    if not pairs:
        return []
    from huggingface_hub import try_to_load_from_cache

    return [f"{repo}/{fn}" for repo, fn in pairs if not isinstance(try_to_load_from_cache(repo, fn), str)]


def _revision(spec: Mapping[str, Any]) -> str:
    """Resolved HF commit from the cached snapshot path ('.../snapshots/<sha>/<file>')."""
    pairs = _hf_repo_file(spec)
    if not pairs:
        return "n/a"
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(*pairs[0])
    if isinstance(path, str) and "/snapshots/" in path:
        return path.split("/snapshots/")[1].split("/")[0]
    return "unknown"


def encoder_info(name: str, spec: Mapping[str, Any], device: str, *, revision: str) -> dict[str, Any]:
    if spec["loader"] == "official_musk" and device.startswith("cuda"):
        dtype = "fp16"
    else:
        dtype = (spec["autocast"] or "fp32") if device.startswith("cuda") else "fp32"
    return {
        "encoder": name, "repo": spec.get("repo"), "model_card": spec["model_card"], "license": spec["license"],
        "revision": revision, "loader": spec["loader"], "spec": spec.get("spec"), "device": device,
        "compute_dtype": dtype, "preprocess": preprocess_info(spec),
        "heads": {h: {"dim": int(v["dim"]), "normalized": bool(v["normalized"])} for h, v in spec["heads"].items()},
        "text_aligned": bool(spec["text_aligned"]), "testing_only": bool(spec.get("testing_only", False)),
    }


def load_encoder(name: str, ecfg: Mapping[str, Any], device: str = "cuda") -> Encoder:
    """Load one encoder exactly per its card. Raises EncoderUnavailable when its files are not cached."""
    spec = encoder_spec(ecfg, name)
    if spec["loader"] != "official_test":
        missing = missing_files(spec)
        if spec.get("lora") and not lora_weights_path(spec).is_file():
            missing.append(str(lora_weights_path(spec)))
        if missing:
            raise EncoderUnavailable(f"{name}: not in the HF cache: {missing}. Run pipeline/jobs/hf_download.sh "
                                     f"on the login node (HF access must be granted for this account).")
    loader = spec["loader"]
    if loader.startswith("official_"):
        return _load_official(name, spec, device)
    if loader == "timm_hub":
        return _load_timm(name, spec, device)
    return _load_hf(name, spec, device)
