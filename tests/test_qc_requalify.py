"""Tissue mask value test (qc.brightness_max) and pipeline/01q_requalify_qc tile mapping."""

import importlib

import numpy as np
import pandas as pd
from PIL import Image

from core import config, conformal
from pipeline import common

step01q = importlib.import_module("pipeline.01q_requalify_qc")
CFG = config.load_settings()


def _cfg(brightness_max):
    return {**CFG, "qc": {**CFG["qc"], "saturation_min": 0.07, "brightness_max": brightness_max}}


def test_value_cut_rejects_pale_tissue_and_is_disabled_at_one():
    pale_eosin = (255, 190, 225)          # HSV value 1.0, saturation ~0.25: real tissue
    glass = (236, 236, 238)               # near-zero saturation: background
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), pale_eosin)
    img.putpixel((1, 0), glass)
    assert common.tissue_mask(img, _cfg(0.92)).tolist() == [[False, False]]
    assert common.tissue_mask(img, _cfg(1.0)).tolist() == [[True, False]]


def test_tile_fracs_maps_source_boxes_onto_display_pixels():
    mask = np.zeros((6, 8), dtype=bool)
    mask[:4, :4] = True                   # the first 512-px tile (4x4 display px) is tissue
    tiles = pd.DataFrame({"x": [0, 512], "y": [0, 0], "size": [512, 512]})
    fr = step01q.tile_fracs(mask, tiles, scale_x=8 / 1024, scale_y=6 / 768)
    assert fr.tolist() == [1.0, 0.0]


def test_nonempty_sets_adds_only_the_argmax_to_empty_rows():
    probs = np.array([[0.1, 0.6, 0.3], [0.5, 0.4, 0.1]])
    sets = np.array([[False, False, False], [True, True, False]])
    out = conformal.nonempty_sets(sets, probs)
    assert out.tolist() == [[False, True, False], [True, True, False]]
    assert sets[0].sum() == 0             # input untouched


def test_tier_set_source_and_name():
    base = {"conformal": {"method": "aps", "mode": "marginal", "nonempty": True}}
    assert conformal.tier_set_source(base) == "set_aps"
    assert conformal.tier_set_name(base) == "aps_marginal_nonempty"
    lac = {"conformal": {"method": "lac", "mode": "mondrian"}}
    assert conformal.tier_set_source(lac) == "set_mondrian"
    assert conformal.tier_set_name(lac) == "lac_mondrian"
