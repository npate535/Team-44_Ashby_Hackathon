"""eval/deferral.py: risk-coverage mechanics on hand-made reads, the read join, and a run on the eval fixture
(docs/VALIDATION_ROUND3.md section 2)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from core import config, io, schemas
from eval import deferral as DF
from tests import downstream_fixture as D

CFG = copy.deepcopy(config.load_settings())
CFG["bootstrap"]["n"] = 50
CLASSES = list(CFG["classes"])
FRACTIONS = [round(0.05 * i, 2) for i in range(11)]


def test_perfect_ranking_removes_errors_first():
    correct = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    key = np.where(correct, 1.0, 0.0)                     # errors carry the lowest score -> deferred first
    rows = DF.risk_coverage(correct, key, [0.0, 0.1, 0.2, 0.3])
    assert [r["kept"] for r in rows] == [10, 9, 8, 7]
    assert rows[0]["accuracy"] == pytest.approx(0.8)
    assert rows[1]["accuracy"] == pytest.approx(8 / 9) and rows[1]["errors_deferred_share"] == pytest.approx(0.5)
    assert rows[2]["accuracy"] == 1.0 and rows[2]["errors_deferred_share"] == 1.0
    assert rows[3]["accuracy"] == 1.0                     # past the errors, accuracy stays perfect


def test_random_ranking_is_flat_on_average():
    rng = np.random.default_rng(0)
    correct = rng.random(400) < 0.7
    rows = DF.random_risk_coverage(correct, FRACTIONS, {}, n_orders=500, seed=20261018)
    overall = correct.mean()
    for r in rows:
        assert r["accuracy"] == pytest.approx(overall, abs=0.01)
        assert r["errors_deferred_share"] == pytest.approx(r["fraction"], abs=0.01)
    assert rows == DF.random_risk_coverage(correct, FRACTIONS, {}, n_orders=500, seed=20261018)   # seeded


def test_fraction_zero_matches_overall_accuracy():
    correct = np.array([1, 0, 1, 1, 0, 1, 1], dtype=bool)
    flags = {"invasive_missed": np.array([0, 1, 0, 0, 0, 0, 0], dtype=bool)}
    row = DF.risk_coverage(correct, np.arange(7.0), [0.0], flags)[0]
    assert row["kept"] == 7 and row["accuracy"] == pytest.approx(correct.mean())
    assert row["errors_deferred_share"] == 0.0 and row["invasive_missed_kept"] == 1
    # no errors at all: the share of errors deferred is undefined (null in JSON), never 0 or 1
    assert math.isnan(DF.risk_coverage(np.ones(5, bool), np.arange(5.0), [0.2])[0]["errors_deferred_share"])


def test_ties_are_deterministic():
    # equal scores: the read that comes first (read-ID order upstream) is deferred first
    correct = np.array([0, 1, 1, 1], dtype=bool)
    rows = DF.risk_coverage(correct, np.zeros(4), [0.25])
    assert rows[0]["kept"] == 3 and rows[0]["accuracy"] == 1.0
    rows = DF.risk_coverage(correct[::-1].copy(), np.zeros(4), [0.25])
    assert rows[0]["accuracy"] == pytest.approx(2 / 3)
    # deferred count is floor(fraction * n): 20% of 1716 reads defers 343
    assert DF.n_deferred(1716, 0.2) == 343 and DF.n_deferred(132, 0.2) == 26 and DF.n_deferred(100, 0.35) == 35
    # set-size ranking: largest set first, then lowest top-class probability, then read ID
    keys = DF.ranking_keys(pd.DataFrame({"reliability": [0.5, 0.5, 0.1], "top_prob": [0.9, 0.6, 0.6],
                                         "set_size": [2, 2, 1]}))
    assert list(np.argsort(keys["set_size"], kind="stable")) == [1, 0, 2]
    assert list(np.argsort(keys["reliability"], kind="stable")) == [2, 0, 1]
    assert list(np.argsort(keys["top_prob"], kind="stable")) == [1, 2, 0]


def _frames():
    cases = pd.DataFrame({"case_id": ["A", "B", "T"], "label": ["Invasive", "InSitu", "Normal"],
                          "split": ["test", "test", "train"], "group": ["g1", "g2", "g3"]})

    def pred(cid, top, model="m512"):
        p = {c: 0.1 for c in CLASSES}
        p[top] = 0.7
        return {"case_id": cid, "split": "test", "model": model, "top": top, "set_tier_size": 1,
                **{schemas.cal_prob_col(c): v for c, v in p.items()}}

    preds = pd.DataFrame([pred("A", "Benign"), pred("B", "Invasive"), pred("A_FSIM_blur1", "Invasive", "m256"),
                          pred("COMP_000", "Invasive", "m256")])
    rel = pd.DataFrame({"case_id": ["A", "B", "A_FSIM_blur1", "COMP_000"],
                        "source_case_id": ["A", "B", "A", "A"],
                        "variant": ["clean", "clean", "blur1", "composite"], "split": "test",
                        "reliability": [0.9, 0.2, 0.5, 0.1]})
    return cases, preds, rel


def test_composites_excluded():
    cases, preds, rel = _frames()
    reads = DF.build_reads(cases, preds, rel, CFG)
    assert list(reads["read_id"]) == ["A", "A_FSIM_blur1", "B"]            # sorted by read ID, composite gone
    assert "COMP_000" not in set(reads["read_id"])
    assert list(reads["truth"]) == ["Invasive", "Invasive", "InSitu"]      # a variant takes its source case's label
    assert list(reads["clean"]) == [True, False, True]
    assert list(reads["correct"]) == [False, True, False]
    assert list(reads["invasive_missed"]) == [True, False, False]
    assert list(reads["insitu_overcalled"]) == [False, False, True]
    assert list(reads["group"]) == ["g1", "g1", "g2"]


def test_unmatched_reads_fail_loudly():
    cases, preds, rel = _frames()
    with pytest.raises(ValueError, match="missing from predictions.csv"):
        DF.build_reads(cases, preds[preds["case_id"] != "A_FSIM_blur1"], rel, CFG)
    with pytest.raises(ValueError, match="missing from reliability.csv"):
        DF.build_reads(cases, preds, rel[rel["case_id"] != "A_FSIM_blur1"], CFG)
    with pytest.raises(ValueError, match="no clean read"):
        DF.build_reads(cases, preds[preds["case_id"] != "B"], rel[rel["case_id"] != "B"], CFG)


def write_deferral_root(root, cfg):
    """The eval fixture plus what the deferral join needs: the tier-set size, and a reliability row for the fixture's
    one simulated degraded read. Returns the fixture IDs."""
    ids = D.write_eval_root(root, cfg)
    preds = io.read_csv(schemas.predictions_csv(root))
    preds["set_tier_size"] = preds["set_size"].clip(lower=1)
    io.write_csv(schemas.predictions_csv(root), preds)
    rel = io.read_csv(schemas.reliability_csv(root))
    vid = ids["variant"]
    src = vid.split("_FSIM_")[0]
    extra = dict(rel[rel["case_id"] == src].iloc[0], case_id=vid, variant="blur3", reliability=0.05)
    io.write_csv(schemas.reliability_csv(root), pd.concat([rel, pd.DataFrame([extra])], ignore_index=True))
    return ids


def test_run_on_the_eval_fixture(tmp_path):
    root = tmp_path / "data"
    write_deferral_root(root, CFG)
    out = DF.run(D.make_ctx(CFG, root, tmp_path / "art", step="eval_deferral"))
    d = io.read_json(out)
    assert d["name"] == "deferral" and d["synthetic"] is True and d["fractions"] == FRACTIONS
    clean, allr = d["populations"]["clean_test"], d["populations"]["all_test_reads"]
    assert clean["n_reads"] == 48 and allr["n_reads"] == 49
    for pop in (clean, allr):
        assert set(pop["rankings"]) == {"reliability", "top_prob", "set_size", "random"}
        for rows in pop["rankings"].values():
            assert [r["fraction"] for r in rows] == FRACTIONS
            assert set(rows[0]) == {"fraction", "kept", "accuracy", "errors_deferred_share", "invasive_missed_kept",
                                    "insitu_overcalled_kept"}
    h = d["headline"]
    assert h["population"] == "all_test_reads" and h["fraction"] == 0.2
    assert h["accuracy_all"]["estimate"] == pytest.approx(allr["rankings"]["reliability"][0]["accuracy"])
    assert h["accuracy_kept"]["estimate"] == pytest.approx(allr["rankings"]["reliability"][4]["accuracy"])
    assert h["accuracy_random_20"] == pytest.approx(allr["rankings"]["random"][4]["accuracy"])
    assert "NaN" not in out.read_text()


def test_results_md_section_quotes_the_headline_and_every_statistic(tmp_path):
    from eval import results_md as R
    root = tmp_path / "data"
    write_deferral_root(root, CFG)
    d = io.read_json(DF.run(D.make_ctx(CFG, root, tmp_path / "art", step="eval_deferral")))
    text = "\n".join(R.section_deferral(d))
    h = d["headline"]
    assert text.startswith("## Deferral") and "eval/deferral.json" in text and "VALIDATION_ROUND3" in text
    assert R.fmt(h["accuracy_kept"], pct=True) in text and R.fmt(h["accuracy_all"], pct=True) in text
    assert f"{100 * h['accuracy_random_20']:.1f}%" in text and "simulated" in text
    for pop in ("Clean test images", "All test reads"):
        assert pop in text
    assert text.count("| 50% |") == 2 and "nan" not in text.lower()
    md, info = R.build(tmp_path / "art", CFG)
    assert "## Deferral" in md and "deferral" in info["present"]
