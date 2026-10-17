"""pipeline.degrade: determinism, size/mode, severity monotonicity, input untouched, unknown type/severity raise."""

from __future__ import annotations

import io as _io

import numpy as np
import pytest
from PIL import Image

from core import config
from pipeline import degrade

CFG = config.load_settings()


def _pattern(w: int = 96, h: int = 64, seed: int = 0) -> Image.Image:
    """Generated non-tissue test image: random pastel blocks plus fine stripes (has high-frequency detail)."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            arr[y:y + 8, x:x + 8] = rng.integers(60, 230, size=3)
    arr[:, ::3] = (arr[:, ::3] * 0.6).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float64)


def _laplacian_var(img: Image.Image) -> float:
    g = _arr(img.convert("L"))
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def _mae(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(_arr(a) - _arr(b)).mean())


@pytest.mark.parametrize("kind", CFG["degrade"]["types"])
@pytest.mark.parametrize("severity", CFG["degrade"]["severities"])
def test_apply_keeps_size_mode_and_input(kind, severity):
    img = _pattern()
    before = _arr(img).copy()
    out = degrade.apply(img, kind, severity, 7, CFG)
    assert out is not img
    assert out.mode == "RGB" and out.size == img.size
    np.testing.assert_array_equal(_arr(img), before)


@pytest.mark.parametrize("kind", CFG["degrade"]["types"])
def test_apply_is_deterministic_per_seed(kind):
    img = _pattern()
    a = degrade.apply(img, kind, 2, 123, CFG)
    b = degrade.apply(img, kind, 2, 123, CFG)
    np.testing.assert_array_equal(_arr(a), _arr(b))


def test_stain_shift_depends_on_seed_and_changes_colors():
    img = _pattern()
    a = degrade.stain_shift(img, 0.2, 0.06, seed=1)
    b = degrade.stain_shift(img, 0.2, 0.06, seed=2)
    assert _mae(a, b) > 0.5
    assert _mae(a, img) > 0.5


def test_stain_shift_zero_jitter_is_near_identity():
    img = _pattern()
    out = degrade.stain_shift(img, 0.0, 0.0, seed=3)
    assert _mae(out, img) < 1.0


def test_accepts_non_rgb_input():
    img = _pattern().convert("RGBA")
    out = degrade.apply(img, "blur", 1, 0, CFG)
    assert out.mode == "RGB" and out.size == img.size


def test_blur_loses_detail_monotonically():
    img = _pattern(128, 96)
    lv = [_laplacian_var(degrade.apply(img, "blur", s, 0, CFG)) for s in CFG["degrade"]["severities"]]
    assert _laplacian_var(img) > lv[0] > lv[1] > lv[2]


def test_resample_loses_detail_monotonically():
    img = _pattern(128, 96)
    errs = [_mae(degrade.apply(img, "resample", s, 0, CFG), img) for s in CFG["degrade"]["severities"]]
    assert 0 < errs[0] < errs[1] < errs[2]


def test_jpeg_error_grows_with_severity():
    img = _pattern(128, 96)
    errs = [_mae(degrade.apply(img, "jpeg", s, 0, CFG), img) for s in CFG["degrade"]["severities"]]
    assert 0 < errs[0] < errs[1] < errs[2]


def test_stain_shift_magnitude_grows_with_severity_on_average():
    img = _pattern(128, 96)
    mean_err = []
    for s in CFG["degrade"]["severities"]:
        mean_err.append(np.mean([_mae(degrade.apply(img, "stain", s, seed, CFG), img) for seed in range(12)]))
    assert mean_err[0] < mean_err[1] < mean_err[2]


def test_severity_params_read_from_config():
    assert degrade.severity_params(CFG, "blur", 2) == {"sigma_px": CFG["degrade"]["params"]["blur"]["sigma_px"][1]}
    assert degrade.severity_params(CFG, "stain", 3) == {
        "alpha": CFG["degrade"]["params"]["stain"]["alpha"][2],
        "beta": CFG["degrade"]["params"]["stain"]["beta"][2],
    }


@pytest.mark.parametrize("kind,severity", [("smudge", 1), ("blur", 0), ("blur", 4), ("jpeg", -1), ("blur", True)])
def test_unknown_type_or_severity_raises(kind, severity):
    with pytest.raises(ValueError):
        degrade.severity_params(CFG, kind, severity)
    with pytest.raises(ValueError):
        degrade.apply(_pattern(), kind, severity, 0, CFG)


def test_jpeg_output_is_decoded_rgb_not_a_lazy_file():
    out = degrade.jpeg_compress(_pattern(), 30)
    buf = _io.BytesIO()
    out.save(buf, format="PNG")  # must not depend on a closed in-memory JPEG buffer
    assert out.mode == "RGB"


def test_down_up_sample_handles_tiny_images():
    img = _pattern(10, 6)
    out = degrade.down_up_sample(img, 8)
    assert out.size == img.size and out.mode == "RGB"
