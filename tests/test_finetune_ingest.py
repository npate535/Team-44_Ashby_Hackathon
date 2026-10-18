"""pipeline/05c_finetune.py ingest (laptop) on the small synthetic root: fine-tune outputs become an evaluated model."""

import importlib
import json
import shutil

import numpy as np
import pandas as pd

from core import config, schemas
from pipeline import riskmodels as rm
from tests.a3_synthetic import after_05, clone_root

step05c = importlib.import_module("pipeline.05c_finetune")


def _fake_outputs(root, cfg, rng):
    classes = list(cfg["classes"])
    cases = rm.read_cases(root)
    train = cases[cases.split == "train"].sort_values("case_id").reset_index(drop=True)
    fold = np.full(len(train), -1)
    for k, held in enumerate(rm.cv_splits(train, cfg)[0]):
        fold[held] = k
    y = train.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    logits = rng.normal(0, 1, (len(train), len(classes)))
    logits[np.arange(len(train)), y] += 2.0                       # informative, so the temperature fit is sane
    oof = pd.DataFrame(logits, columns=classes)
    oof.insert(0, "fold", fold)
    oof.insert(0, "case_id", train.case_id)
    ev = cases[cases.split.isin(["calibration", "test"])].sort_values("case_id")
    fin = pd.DataFrame(rng.normal(0, 1, (len(ev), len(classes))), columns=classes)
    fin.insert(0, "split", ev.split.to_numpy())
    fin.insert(0, "case_id", ev.case_id.to_numpy())
    d = root / "finetune"
    d.mkdir(exist_ok=True)
    oof.to_csv(d / "cv_oof.csv", index=False)
    fin.to_csv(d / "final_logits.csv", index=False)
    return train, ev


def test_ingest_adds_an_evaluated_model_and_never_changes_production(tmp_path):
    root0, art0, cfg_path = after_05()
    root, art = clone_root(root0, tmp_path / "data"), tmp_path / "art"
    shutil.copytree(art0, art)
    cfg = config.load_settings(cfg_path)
    train, ev = _fake_outputs(root, cfg, np.random.default_rng(0))
    before = pd.read_csv(schemas.predictions_csv(root), keep_default_na=False, na_values=[""])
    assert step05c.main(["--data-root", str(root), "--artifacts-root", str(art), "--config", str(cfg_path)]) == 0
    after = pd.read_csv(schemas.predictions_csv(root), keep_default_na=False, na_values=[""])
    schemas.validate_frame(schemas.PREDICTIONS, after)
    cols = [schemas.prob_col("uni2h_ft", c) for c in cfg["classes"]]
    pd.testing.assert_frame_equal(after[before.columns], before)       # production columns untouched
    clean = after[after.case_id.isin(set(train.case_id) | set(ev.case_id))]
    np.testing.assert_allclose(clean[cols].sum(1), 1.0, atol=1e-9)
    assert after.loc[~after.case_id.isin(clean.case_id), cols].isna().all().all()   # no variant reads
    out = json.loads(schemas.eval_json(art, "finetune").read_text())
    assert out["model"] == "uni2h_ft" and out["adopted_as_production"] is False
    assert "uni2h_ft" in out["cv_repeat0"] and out["rule_choice"] in out["cv_repeat0"]


def test_ingest_without_outputs_is_a_no_op(tmp_path):
    root0, art0, cfg_path = after_05()
    root, art = clone_root(root0, tmp_path / "data"), tmp_path / "art"
    shutil.copytree(art0, art)
    assert step05c.main(["--data-root", str(root), "--artifacts-root", str(art), "--config", str(cfg_path)]) == 0
    assert not schemas.eval_json(art, "finetune").exists()
