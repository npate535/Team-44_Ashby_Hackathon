"""Fact sheet (eval/fact_sheet.py): units, splits, disjointness, per-class rates, errors, case examples, rendering."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval import fact_sheet as fs

CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]


def _cases() -> pd.DataFrame:
    rows = []
    for i in range(12):
        split = ["train", "calibration", "test"][i % 3]
        rows.append({"case_id": f"BACH_{i:04d}", "label": CLASSES[i % 4], "split": split,
                     "patient_id": "7" if i < 3 else "", "group": "p7" if i < 3 else f"b{i}"})
    return pd.DataFrame(rows)


def test_split_table_counts_and_patients():
    table = fs.split_table(_cases(), CLASSES)
    assert sum(s["n_images"] for s in table.values()) == 12
    assert table["train"]["n_images"] == 4
    assert table["train"]["identified_patients"] == 1
    assert table["test"]["per_class"]["Invasive"] == sum(1 for i in range(12) if i % 3 == 2 and i % 4 == 3)


def test_disjointness_flags_shared_group():
    cases = _cases()
    shared = fs.disjointness(cases)
    # p7 holds cases 0, 1, 2, which fall in three different splits.
    assert shared["groups_in_more_than_one_split"] == ["p7"]
    assert shared["ok"] is False
    clean = cases.assign(group=[f"g{i}" for i in range(12)], patient_id="")
    assert fs.disjointness(clean)["ok"] is True


def test_per_class_rates_from_confusion():
    counts = [[20, 3, 0, 0], [9, 39, 0, 0], [2, 1, 24, 3], [0, 0, 0, 31]]
    rates = fs.per_class_rates(counts, CLASSES)
    assert rates["Invasive"]["sensitivity"] == pytest.approx(1.0)
    assert rates["Invasive"]["correct"] == 31 and rates["Invasive"]["n"] == 31
    # Specificity for Invasive: non-invasive images not called invasive = (132 - 31 - 3) / (132 - 31).
    assert rates["Invasive"]["specificity"] == pytest.approx(98 / 101)
    assert rates["InSitu"]["sensitivity"] == pytest.approx(24 / 30)


def test_errors_list_only_misreads():
    preds = pd.DataFrame({"case_id": ["A", "B", "C"], "top": ["Invasive", "Benign", "Normal"],
                          "pcal__Normal": [0.0, 0.1, 0.8], "pcal__Benign": [0.0, 0.8, 0.1],
                          "pcal__InSitu": [0.2, 0.1, 0.05], "pcal__Invasive": [0.8, 0.0, 0.05],
                          "set_tier": ["['Invasive']", "['Benign']", "['Normal','Benign']"]})
    truth = pd.Series({"A": "InSitu", "B": "Benign", "C": "Benign"})
    errs = fs.error_rows(preds, truth, CLASSES)
    assert [e["case_id"] for e in errs] == ["A", "C"]
    assert errs[0]["truth"] == "InSitu" and errs[0]["predicted"] == "Invasive"
    assert errs[0]["p_predicted"] == pytest.approx(0.8)


def test_case_example_keeps_every_class_probability():
    case = {"case_id": "BACH_0087", "lane": "biopsy",
            "prediction": {"classes": CLASSES, "probs": [0.1491, 0.0166, 0.4272, 0.4072],
                           "conformal_set": ["Normal", "InSitu", "Invasive"]},
            "flag": {"tier": 2, "reasons": ["aggressive_in_conformal_set"]},
            "eval_only": {"true_label": "InSitu"},
            "agent": {"orders": [{"type": "stain", "items": ["p63", "calponin"]}]},
            "contrast_case": {"case_id": "BACH_0070", "label": "InSitu", "differs_most_in": ["necrosis"]}}
    ex = fs.case_example(case)
    assert sum(ex["probs_percent"].values()) == pytest.approx(100.0, abs=0.2)
    assert ex["probs_percent"]["Benign"] == pytest.approx(1.7, abs=0.05)
    assert ex["true_label"] == "InSitu" and ex["tier"] == 2
    assert ex["drafted_orders"] == ["p63 + calponin"]


def test_render_md_states_units_and_conventions():
    sheet = {"bach": {"n_images": 400, "per_class": {c: 100 for c in CLASSES}, "image_px": [2048, 1536],
                      "mpp": 0.42, "regions_per_image": 12, "grid": "3 x 4", "region_px": 512, "views": 1,
                      "n_regions": 4800, "n_region_views": 4800, "wsi": {"n_svs": 30, "n_annotated": 10}},
             "auroc_conventions": {"invasive_vs_rest": {"value": 0.999, "definition": "binary"}}}
    md = fs.render_md(sheet)
    assert "not whole-slide images" in md
    assert "4,800" in md
    assert "invasive_vs_rest" not in md  # rendered with readable names
    assert "reads each region once" in md


def test_exact_at_boundary_uses_the_true_denominator():
    iv = fs.exact_at_boundary(1.0, [1.0, 1.0], 31, 31)
    assert iv["k"] == 31 and iv["n"] == 31
    assert iv["lo"] == pytest.approx(0.888, abs=5e-4) and iv["hi"] == 1.0
    assert fs.exact_text(iv) == "31 of 31 (exact 95% CI 88.8–100%)"
    assert fs.exact_text(fs.exact_at_boundary(0.0, [0.0, 0.0], 0, 71)) == "0 of 71 (exact 95% CI 0–5.1%)"
    assert fs.exact_at_boundary(0.97, [0.936, 1.0], 98, 101) is None      # not on the boundary
    assert fs.exact_at_boundary(1.0, [0.9, 1.0], 31, 31) is None          # bootstrap did not collapse
    with pytest.raises(ValueError):
        fs.exact_at_boundary(1.0, [1.0, 1.0], 31, 132)                    # 31 of 132 is not 100%


def test_percentages_round_half_up_like_the_docs():
    assert fs._pct(39 / 48) == "81.3%"
    assert fs._pct(20 / 23) == "87.0%"
    assert fs._pc_short(1.0) == "100" and fs._pc_short(0.05061) == "5.1"


def test_render_md_replaces_collapsed_intervals():
    counts = [[20, 3, 0, 0], [9, 39, 0, 0], [2, 1, 24, 3], [0, 0, 0, 31]]
    sens = {"value": 1.0, "ci": [1.0, 1.0], "k": 31, "n": 31,
            "exact_ci": fs.exact_at_boundary(1.0, [1.0, 1.0], 31, 31)}
    other = {"value": 0.97, "ci": [0.936, 1.0], "k": 98, "n": 101, "exact_ci": None}
    sheet = {"test": {"model": "m", "auroc_conventions": {"invasive_vs_rest": {"value": 1.0, "ci": [1.0, 1.0]}},
                      "macro_f1": {"value": 0.86, "ci": [0.68, 0.93]},
                      "balanced_accuracy": {"value": 0.87, "ci": [0.68, 0.94]},
                      "accuracy": {"value": 114 / 132, "ci": [0.76, 0.94], "k": 114, "n": 132},
                      "invasive_sensitivity": sens, "invasive_specificity": other,
                      "confusion": {"classes": CLASSES, "counts": counts},
                      "per_class": fs.per_class_rates(counts, CLASSES), "errors": []},
             "external_test": {"n_patients": 25, "n_slides": 65, "n_crops": 353, "protocol": "p",
                               "metrics": {"frozen_tumor_vs_normal_auroc": {"value": 1.0, "ci": [1.0, 1.0], "n": 42,
                                                                            "collapsed": True, "n_pos": 25,
                                                                            "n_neg": 17},
                                           "frozen_normal_specificity": {
                                               "value": 1.0, "ci": [1.0, 1.0], "n": 17, "k": 17,
                                               "exact_ci": fs.exact_at_boundary(1.0, [1.0, 1.0], 17, 17)}}}}
    md = fs.render_md(sheet)
    assert "31 of 31 (exact 95% CI 88.8–100%)" in md
    assert "17 of 17 (exact 95% CI 80.5–100%)" in md
    assert "all positives ranked above all negatives" in md
    assert "all 25 frozen tumor slides ranked above all 17 frozen normal slides" in md
    assert "1.000–1.000" not in md and "[1.000, 1.000]" not in md and "100.0%, 100.0%" not in md


DATA = Path(os.environ.get("PATHQUEUE_DATA_ROOT", "/nonexistent"))


@pytest.mark.skipif(not (DATA / "cases.csv").is_file(), reason="real data root not available")
def test_real_data_fact_sheet():
    root = Path(__file__).resolve().parents[1]
    art = Path(os.environ.get("PATHQUEUE_ARTIFACTS_ROOT", root / "artifacts"))
    sheet = fs.build_fact_sheet(DATA, art)
    splits = sheet["splits"]
    assert [splits[s]["n_images"] for s in ("train", "calibration", "test")] == [218, 50, 132]
    assert sheet["disjointness"]["ok"] is True
    conf = np.array(sheet["test"]["confusion"]["counts"])
    assert conf.sum() == 132 and conf[3, 3] == 31
    sens = sheet["test"]["invasive_sensitivity"]
    assert (sens["k"], sens["n"]) == (31, 31)                    # denominator: invasive images, not all 132
    assert sens["exact_ci"]["lo"] == pytest.approx(0.888, abs=5e-4)
    assert (sheet["test"]["invasive_specificity"]["k"], sheet["test"]["invasive_specificity"]["n"]) == (98, 101)
    assert sheet["bach"]["n_regions"] == 4800
    assert sheet["production_features"]["views"] == 1  # uni2h bach512 was embedded at view 0 only
