"""eval/figures.py renders every available figure (PNG), skips missing files, labels synthetic data."""

from __future__ import annotations

import copy

from PIL import Image

from core import config, io, schemas
from eval import benchmark_thunder, figures, margin_eval, metrics, queue_sim
from tests import downstream_fixture as D

CFG = copy.deepcopy(config.load_settings())
CFG["bootstrap"]["n"] = 100


def test_figures_for_every_eval_file(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    ctx = D.make_ctx(CFG, tmp_path / "data", tmp_path / "art")
    metrics.run(ctx)
    queue_sim.run(ctx, reps=20)
    margin_eval.run(ctx)
    benchmark_thunder.run(ctx)
    # a foreign eval file with an unknown shape gets the generic dot + CI view
    io.write_json(schemas.eval_json(tmp_path / "art", "concepts"),
                  {"name": "concepts", "synthetic": True, "lymphocytes": {"auroc": {"estimate": 0.8, "lo": 0.7, "hi": 0.9}}})
    result = figures.run(ctx)
    assert set(result["written"]) == {"calibration", "conformal", "metrics", "thunder", "queue_sim", "margin", "concepts"}
    for path in result["written"].values():
        with Image.open(path) as img:
            assert img.format == "PNG" and img.width > 600
    assert result["skipped"]["reliability"] == "no eval file"


def test_generic_extractor_finds_ci_blocks():
    data = {"a": {"estimate": 1.0, "lo": 0.5, "hi": 1.5}, "b": [{"x": {"estimate": 2, "lo": 1, "hi": 3}}], "c": 4}
    assert [lab for lab, _ in figures.ci_blocks(data)] == ["a", "b[0] · x"]


def test_no_eval_files_writes_nothing(tmp_path):
    result = figures.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"))
    assert result["written"] == {}


def test_results_md_quotes_every_number_with_its_source(tmp_path):
    from eval import results_md

    D.write_eval_root(tmp_path / "data", CFG)
    ctx = D.make_ctx(CFG, tmp_path / "data", tmp_path / "art")
    metrics.run(ctx)
    queue_sim.run(ctx, reps=20)
    margin_eval.run(ctx)
    benchmark_thunder.run(ctx)
    io.write_json(schemas.eval_json(tmp_path / "art", "spotcheck"),
                  {"created_utc": "x", "synthetic": True, "reviewer_kind": "ai", "reviews": [], "summary": {}})
    out = results_md.run(ctx)
    assert out == tmp_path / "art" / "RESULTS_synthetic.md"          # synthetic never overwrites docs/RESULTS.md
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# PathQueue results") and "SYNTHETIC DATA" in text
    for src in ("eval/metrics.json", "eval/calibration.json", "eval/conformal.json", "eval/thunder.json",
                "eval/queue_sim.json", "eval/margin.json"):
        assert src in text
    assert "same split, different protocol" in text and "AI-reviewed, not a pathologist" in text
    assert "delay imposed on unflagged cases" in text and "## Limitations" in text
    assert "[" in text and "groups" in text


def test_results_md_post_hoc_note_is_built_from_conformal_json():
    from eval import results_md

    def block(est):
        return {"estimate": est, "lo": est, "hi": est, "n": 10}

    lac = {"coverage": block(0.8), "mean_set_size": block(1.2), "empty_set_rate": 0.1,
           "set_size_histogram": {"0": 1, "1": 6, "2": 3, "3": 0, "4": 0}, "per_class": {}}
    tier = {**lac, "empty_set_rate": 0.0, "set_size_histogram": {"0": 0, "1": 7, "2": 3, "3": 0, "4": 0}}
    c = {"alpha": 0.1, "target_coverage": 0.9, "drives_tiers": "lac_mondrian", "finite_sample": {"n_calibration": 77},
         "methods": {"lac_marginal": lac, "lac_mondrian": tier}}
    text = "\n".join(results_md.section_conformal(c))
    assert "multi-class sets for 30.0% of cases" in text and "no differential" not in text
    assert "77-image calibration" in text and "50-image" not in text
    assert "Romano" not in text and "argmax" not in text                 # not APS, not nonempty
    lac["set_size_histogram"] = {"0": 1, "1": 9, "2": 0, "3": 0, "4": 0}
    c["drives_tiers"] = "aps_marginal_nonempty"
    c["methods"] = {"lac_marginal": lac, "aps_marginal_nonempty": tier}
    text = "\n".join(results_md.section_conformal(c))
    assert "no multi-class sets" in text and "Romano" in text and "argmax class added" in text


def test_results_md_number_formatting():
    from eval import results_md as R

    assert R.fmt({"estimate": 0.998, "lo": 0.994, "hi": 1.0, "n": 132}) == "0.998 [0.994, 1.000] (n 132)"
    assert R.fmt({"estimate": 0.61, "lo": 0.5, "hi": 0.7}) == "0.61 [0.50, 0.70]"
    assert R.fmt({"estimate": -0.001, "lo": -0.004, "hi": 0.002}) == "0.00 [0.00, 0.00]"      # no signed zero
    assert R.fmt({"estimate": -0.0001, "lo": -0.01, "hi": 0.0}, pct=True) == "0.0% [-1.0%, 0.0%]"
    assert R.fmt({"estimate": 0.999823, "lo": 0.847377, "hi": 0.999893}, digits=5) == "0.99982 [0.84738, 0.99989]"
    assert R.sim_fmt({"mean": -0.01, "lo": -0.04, "hi": 0.0}) == "0.0 [0.0, 0.0]"


def test_results_md_margin_threshold_queue_and_foreign_sections():
    from eval import results_md as R

    def blk(e):
        return {"estimate": e, "lo": e, "hi": e, "n": 4}

    m = {"threshold": 0.999823, "threshold_ci": {"lo": 0.847377, "hi": 0.999893, "ci": 0.95, "n": 30, "n_groups": 6},
         "n_benign_cal_images": 30, "notes": [],
         "benign_fpr": {"pooled_per_margin": blk(0.0), "any_margin": blk(0.0), "target_max_fpr": 0.05,
                        "n_scored": 1, "n_benign_test": 1},
         "echo": {"p_at_least_one_clear_margin": blk(0.5), "spearman_max_margin_vs_risk": blk(0.1)}}
    text = "\n".join(R.section_margin(m))
    assert "threshold P = 0.99982 [0.84738, 0.99989] (n 30, 6 groups)" in text and "{" not in text
    none = {"mean": None, "lo": None, "hi": None, "n_reps": 0}
    some = {"mean": 3.0, "lo": 1.0, "hi": 5.0}
    q = {"reps": 2, "policies": {}, "notes": [], "assumptions": {},
         "results": {"tiers": {k: some for k in ("aggressive_mean", "aggressive_p90", "non_flagged_mean", "overtime")}
                     | {"vs_fifo": {"aggressive_mean": some, "non_flagged_mean": some, "false_negative_mean": none}}}}
    text = "\n".join(R.section_queue(q))
    assert "n/a min" not in text and "false negatives: not available" in text
    data = {"name": "x", "git_sha": None, **{f"k{i}": blk(0.1 * (i % 10)) for i in range(55)}}
    lines = R.section_foreign("x", data)
    assert sum(line.startswith("- ") for line in lines) == 55                  # no silent truncation


def test_results_md_reviewers_and_git_sha_come_from_the_files(tmp_path):
    from eval import results_md as R

    art = tmp_path / "art"
    reviews = [{"item_id": "concept:tubules:1", "reviewer": "Claude Opus 5.5 (AI)", "reviewer_kind": "ai",
                "verdict": "acceptable"}] * 2 + [{"item_id": "cells:x:1", "reviewer": "Codex UI check (AI)",
                                                  "reviewer_kind": "ai", "verdict": "unsure"}]
    io.write_json(schemas.eval_json(art, "spotcheck"), {"created_utc": "x", "git_sha": None, "synthetic": True,
                                                         "reviewer_kind": "ai", "reviews": reviews, "summary": {}})
    io.write_json(schemas.eval_json(art, "retrieval"), {"name": "retrieval", "git_sha": "abc123", "synthetic": True})
    text, _ = R.build(art, CFG)
    assert "reviews by Claude Opus 5.5 (AI) × 2; Codex UI check (AI) × 1" in text and "(Claude)," not in text
    generated = next(line for line in text.split("\n") if line.startswith("Generated"))
    assert generated.endswith("(git abc123).")


def test_results_md_finds_plain_marginal_aps_as_the_tier_set():
    from eval import results_md as R

    blk = {"estimate": 0.9, "lo": 0.8, "hi": 0.95, "n": 10}
    aps = {"coverage": blk, "mean_set_size": blk, "empty_set_rate": 0.0, "set_size_histogram": {"1": 10},
           "per_class": {"Invasive": {"estimate": 0.97, "lo": 0.9, "hi": 1.0, "n": 3}}}
    lac = {**aps, "per_class": {}}
    c = {"alpha": 0.1, "target_coverage": 0.9, "drives_tiers": "aps_marginal", "finite_sample": {},
         "methods": {"lac_marginal": lac, "aps": aps}}
    assert "Per-class coverage of the tier set (aps_marginal): Invasive 97.0%" in "\n".join(R.section_conformal(c))
