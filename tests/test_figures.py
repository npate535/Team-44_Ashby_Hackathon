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
    assert "same split, different protocol" in text and "AI-reviewed (Claude), not a pathologist" in text
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
