"""pipeline.encoders: config validation, dihedral views, card preprocessing, head bookkeeping with fake modules."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from core import config
from pipeline import encoders as E


@pytest.fixture
def ecfg():
    return E.load_encoders_config()


def test_shipped_encoders_yaml_is_valid_and_matches_settings(ecfg):
    cfg = config.load_settings()
    E.check_native_grids(ecfg, cfg)
    assert set(cfg["encoders"]["priority"]) <= set(ecfg["encoders"])
    for name in cfg["encoders"]["text_aligned"]:
        spec = ecfg["encoders"][name]
        assert spec["text_aligned"] and "align" in spec["heads"] and spec["nct_floor"] > 1 / 9
    dims = {n: {h: v["dim"] for h, v in s["heads"].items()} for n, s in ecfg["encoders"].items()}
    assert dims["conch"] == {"align": 512, "probe": 512}
    assert dims["musk"] == {"align": 1024, "probe": 2048}
    assert dims["virchow2"] == {"probe": 2560} and dims["uni2h"] == {"probe": 1536}
    assert dims["hoptimus0"] == {"probe": 1536} and dims["phikonv2"] == {"probe": 1024}
    assert dims["pathgenclip"] == {"align": 512, "probe": 512}


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["encoders"]["conch"].pop("mean"), "missing key mean"),
    (lambda d: d["encoders"]["conch"].update(loader="magic"), "loader must be one of"),
    (lambda d: d["encoders"]["musk"]["heads"]["align"].update(normalized=False), "align must be normalized"),
    (lambda d: d["encoders"]["conch"].pop("nct_floor"), "nct_floor"),
    (lambda d: d.pop("checks"), "missing top-level key checks"),
])
def test_encoders_yaml_defects_are_reported(ecfg, mutate, message):
    data = copy.deepcopy(ecfg)
    mutate(data)
    assert any(message in p for p in E.encoders_config_problems(data))


def test_native_grid_disagreement_raises(ecfg):
    cfg = config.load_settings()
    cfg["encoders"]["native_grid"]["conch"] = "g256"
    with pytest.raises(ValueError, match="native grids disagree"):
        E.check_native_grids(ecfg, cfg)


def _asym() -> tuple[Image.Image, np.ndarray]:
    arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) * 10
    return Image.fromarray(arr), arr


@pytest.mark.parametrize("view", range(8))
def test_dihedral_view_is_ccw_rotation_then_mirror(view):
    image, arr = _asym()
    want = np.rot90(arr, k=view % 4, axes=(0, 1))  # numpy rot90 with k > 0 is counter-clockwise
    if view >= 4:
        want = want[:, ::-1]
    np.testing.assert_array_equal(np.asarray(E.dihedral(image, view)), want)


@pytest.mark.parametrize("bad", [-1, 8, 1.5, True])
def test_dihedral_rejects_bad_views(bad):
    with pytest.raises(ValueError):
        E.dihedral(_asym()[0], bad)


def test_crop_box_whole_image_and_bounds():
    image = Image.new("RGB", (40, 30))
    assert E.crop_box(image, 0, 0, 40, 30) is image
    assert E.crop_box(image, 8, 4, 16, 10).size == (16, 10)
    with pytest.raises(ValueError, match="outside"):
        E.crop_box(image, 30, 0, 16, 10)


def test_card_preprocess_resizes_a_256_tile_to_224_without_cropping_tissue(ecfg):
    pre = E.Preprocess.from_spec(ecfg["encoders"]["uni2h"])
    image = Image.new("RGB", (256, 256), (255, 128, 0))
    x = pre(image)
    assert tuple(x.shape) == (3, 224, 224) and x.dtype == torch.float32
    mean, std = np.array(pre.mean), np.array(pre.std)
    np.testing.assert_allclose(x[:, 100, 100].numpy(), (np.array([1.0, 128 / 255, 0.0]) - mean) / std, atol=1e-5)


def test_card_preprocess_resizes_the_shorter_side_then_center_crops(ecfg):
    pre = E.Preprocess.from_spec(ecfg["encoders"]["phikonv2"])
    arr = np.zeros((224, 448, 3), dtype=np.uint8)
    arr[:, 112:336] = 255  # bright centre band, the part a 224 center crop keeps
    x = pre(Image.fromarray(arr))
    assert tuple(x.shape) == (3, 224, 224)
    assert x[0].min() > 1.5  # only the white band survives


def test_preprocess_info_is_recorded(ecfg):
    info = E.preprocess_info(ecfg["encoders"]["conch"])
    assert info == {"resize": 448, "crop": 448, "interpolation": "bicubic", "mean": ecfg["encoders"]["conch"]["mean"],
                    "std": ecfg["encoders"]["conch"]["std"], "input_px": 448}


# ---- head math with fake modules --------------------------------------------------------------------------------
class FakeConchVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w_tok = torch.randn(3, 6, generator=g)
        self.pool = torch.nn.Linear(6, 5)
        self.ln_contrast = torch.nn.LayerNorm(5)
        self.proj_contrast = torch.randn(5, 4, generator=g)
        self.trunk_kwargs = {}

    def trunk(self, x):
        return x.flatten(2).transpose(1, 2) @ self.w_tok[:3]  # [B, T, 6] tokens

    def attn_pool_contrast(self, tokens):
        return self.pool(tokens).mean(1, keepdim=True).repeat(1, 3, 1)


class FakeConch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = FakeConchVisual()

    def encode_image(self, x, proj_contrast=True, normalize=True):
        v = self.visual
        feats = v.ln_contrast(v.attn_pool_contrast(v.trunk(x))[:, 0])
        if proj_contrast:
            feats = feats @ v.proj_contrast
        return F.normalize(feats, dim=-1) if normalize else feats


def test_conch_one_pass_equals_the_two_encode_image_calls():
    model, x = FakeConch(), torch.rand(2, 3, 4, 4)
    one = E.conch_heads(model, x, one_pass=True)
    two = E.conch_heads(model, x, one_pass=False)
    torch.testing.assert_close(one["probe"], model.encode_image(x, proj_contrast=False, normalize=False))
    torch.testing.assert_close(one["align"], model.encode_image(x))
    torch.testing.assert_close(one["align"], two["align"])


class FakeMusk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_head = torch.nn.Linear(3, 3, bias=False)

    def forward(self, image, with_head, out_norm, ms_aug, return_global):
        base = image.flatten(1)[:, :3]
        if ms_aug:
            return (torch.cat([base, base * 2], dim=1), None)  # [B, 6] = CLS@1x ⊕ CLS@2x
        out = self.vision_head(base) if with_head else base
        return (F.normalize(out, dim=-1) if out_norm else out, None)


def test_musk_align_from_the_probe_call_equals_the_with_head_call():
    model, x = FakeMusk(), torch.rand(2, 3, 2, 2)
    one = E.musk_heads(model, x, align_dim=3, one_pass=True)
    two = E.musk_heads(model, x, align_dim=3, one_pass=False)
    assert tuple(one["probe"].shape) == (2, 6)
    torch.testing.assert_close(one["align"], two["align"])
    torch.testing.assert_close(one["align"].norm(dim=-1), torch.ones(2))


def test_clip_heads_normalize_only_the_align_head():
    class FakeClip:
        def encode_image(self, x):
            return x.flatten(1)[:, :4] * 3

    heads = E.clip_heads(FakeClip(), torch.rand(2, 3, 2, 2))
    torch.testing.assert_close(heads["align"], F.normalize(heads["probe"], dim=-1))


def test_token_heads_cls_and_virchow_cls_meanpatch():
    tokens = torch.arange(2 * 7 * 2, dtype=torch.float32).reshape(2, 7, 2)
    torch.testing.assert_close(E.token_heads(tokens, "cls")["probe"], tokens[:, 0])
    pooled = torch.rand(2, 5)
    torch.testing.assert_close(E.token_heads(pooled, "cls")["probe"], pooled)
    cat = E.token_heads(tokens, "cls_meanpatch", register_tokens=4)["probe"]
    torch.testing.assert_close(cat, torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=-1))
    with pytest.raises(ValueError, match="token outputs"):
        E.token_heads(pooled, "cls_meanpatch", 4)


def test_closure_value_sees_through_the_no_grad_decorator():
    model = object()

    def make():
        @torch.no_grad()
        def encode_images(batch):
            return model, batch

        return encode_images

    assert E.closure_value(make(), "model") is model
    with pytest.raises(RuntimeError, match="no longer closes over"):
        E.closure_value(make(), "tokenizer")


def test_check_head_dims_rejects_missing_heads_and_shapes(ecfg):
    spec = ecfg["encoders"]["conch"]
    with pytest.raises(ValueError, match="heads"):
        E.check_head_dims("conch", spec, {"align": np.zeros((2, 512))}, n=2)
    with pytest.raises(ValueError, match="expected"):
        E.check_head_dims("conch", spec, {"align": np.zeros((2, 512)), "probe": np.zeros((2, 768))}, n=2)


def test_official_test_adapter_wrapper_runs_on_cpu(ecfg, monkeypatch):
    monkeypatch.setenv("PATHOLOGY_TEST_ADAPTER_DELAY_S", "0")
    enc = E.load_encoder("test", ecfg, device="cpu")
    batch = torch.stack([enc.preprocess(Image.new("RGB", (16, 16), c)) for c in [(255, 0, 0), (10, 200, 30)]])
    out = enc.embed(batch)
    assert out["align"].shape == (2, 3) and out["probe"].shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(out["align"], axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(out["align"], enc.reference(batch)["align"], atol=1e-6)
    t1, t2 = enc.encode_texts(["a", "b", "a"]), enc.encode_texts(["a"])
    np.testing.assert_allclose(t1[0], t1[2])
    np.testing.assert_allclose(t1[0], t2[0])
    assert enc.info["testing_only"] and enc.info["revision"] == "n/a" and enc.text_aligned


def test_unknown_encoder_is_rejected(ecfg):
    with pytest.raises(ValueError, match="unknown encoder"):
        E.encoder_spec(ecfg, "uni3")
