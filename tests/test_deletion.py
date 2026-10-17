"""eval/deletion_test.py: in-distribution re-aggregation, paired cluster bootstrap, cells version."""

import importlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from core import calibration, cli, config, schemas
from tests.a3_synthetic import after_05, clone_root

deletion = importlib.import_module("eval.deletion_test")
CFG = config.load_settings()
CLASSES = list(CFG["classes"])


@pytest.fixture(scope="module")
def run_del(tmp_path_factory):
    root0, art0, cfg_path = after_05()
    work = tmp_path_factory.mktemp("del")
    root, art = work / "data", work / "art"
    clone_root(root0, root)
    shutil.copytree(art0, art)
    assert deletion.main(["--data-root", str(root), "--artifacts-root", str(art), "--config", str(cfg_path)]) == 0
    return root, art, cfg_path


def ctx_for(root, art, cfg_path):
    parser = cli.step_parser("eval.deletion_test", "x")
    return cli.context(parser.parse_args(["--data-root", str(root), "--artifacts-root", str(art),
                                          "--config", str(cfg_path)]))


def test_full_read_reproduces_05_probabilities(run_del):
    """With nothing removed, the re-aggregation path must give exactly 05's uncalibrated model probabilities."""
    root, art, cfg_path = run_del
    ctx = ctx_for(root, art, cfg_path)
    choice = json.loads(schemas.eval_json(art, "model_choice").read_text())
    pred = pd.read_csv(schemas.predictions_csv(root), dtype={"case_id": str}).set_index("case_id")
    cases = pd.read_csv(schemas.cases_csv(root), dtype={"case_id": str})
    test = sorted(cases[cases.split == "test"].case_id)[:15]
    tiles = deletion.rm.read_tiles(root)
    for model in {choice["production"], choice["model_a"]}:
        reader = deletion.Reader(model, ctx, choice, tiles, test)
        for cid in test:
            p = calibration.softmax(reader.logits(cid, set())[None, :], 1.0)[0]
            np.testing.assert_allclose(p, pred.loc[cid, [schemas.prob_col(model, c) for c in CLASSES]].to_numpy(float),
                                       atol=1e-6)


def test_removing_tiles_changes_embedding_reads(run_del):
    root, art, cfg_path = run_del
    ctx = ctx_for(root, art, cfg_path)
    choice = json.loads(schemas.eval_json(art, "model_choice").read_text())
    cases = pd.read_csv(schemas.cases_csv(root), dtype={"case_id": str})
    cid = sorted(cases[cases.split == "test"].case_id)[0]
    reader = deletion.Reader(choice["model_a"], ctx, choice, deletion.rm.read_tiles(root), [cid])
    avail = reader.qc_tiles(cid)
    assert reader.risk(cid, set(avail[:3].tolist())) != reader.risk(cid, set())
    with pytest.raises(ValueError):
        reader.logits(cid, set(avail.tolist()))


def test_deletion_json(run_del):
    _, art, _ = run_del
    doc = json.loads(schemas.eval_json(art, "deletion").read_text())
    for key in ("name", "created_utc", "config_hash", "git_sha", "synthetic"):
        assert key in doc
    choice = json.loads(schemas.eval_json(art, "model_choice").read_text())
    assert set(doc["models"]) == {choice["production"], choice["model_a"]}
    a = doc["models"][choice["model_a"]]
    s = a["all_eligible"]
    assert s["n"] > 50 and s["n_groups"] > 10
    for key in ("mean_drop_random", "mean_drop_risk", "risk_minus_random"):
        assert s[key]["lo"] <= s[key]["estimate"] <= s[key]["hi"]
    # the planted heatmap is faithful for the black-box model on synthetic data
    assert a["predicted_aggressive"]["risk_minus_random"]["lo"] > 0
    prod = doc["models"][choice["production"]]
    if prod.get("kind") == "concept":
        assert "circular" in prod["caveat"]
    elif "skipped" in prod:
        assert "cell-only" in prod["skipped"]
    cells = doc["cells"]
    assert cells.get("model") in ("cb_text_cells", "cb_cells") and cells["top_minus_random"]["estimate"] > 0


def test_top_concept_picks_largest_positive_sum():
    contrib = pd.DataFrame({"case_id": ["a"] * 4, "concept": ["x", "x", "y", "z"], "source": ["text"] * 3 + ["cell"],
                            "contribution": [0.2, 0.3, 0.4, 9.0]})
    assert deletion.top_concept(contrib, "a") == "x"
    neg = contrib.assign(contribution=[-1.0, -1.0, -0.5, 9.0])
    assert deletion.top_concept(neg, "a") is None
