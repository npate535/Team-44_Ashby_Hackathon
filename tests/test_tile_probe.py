"""pipeline.tile_probe: out-of-fold microscopy-only tile risk (CELLS_SPEC §4, plan §6.6)."""

import numpy as np
import pandas as pd
import pytest

from core import schemas, stats
from core.config import load_settings
from pipeline import tile_probe

CFG = load_settings()
CLASSES = list(CFG["classes"])
AGG = CFG["aggressive_class"]


def fake_world(n_train=60, n_cal=20, n_test=20, tiles=6, dim=24, signal=2.0, seed=0, group_size=3):
    """Cases (with multi-case groups), a bach512-like index and embeddings with a planted class direction."""
    rng = np.random.default_rng(seed)
    rows, n = [], n_train + n_cal + n_test
    splits = ["train"] * n_train + ["calibration"] * n_cal + ["test"] * n_test
    for i in range(n):
        rows.append({"case_id": f"C{i:03d}", "label": CLASSES[i % 4], "split": splits[i],
                     "group": f"g{i // group_size}"})
    cases = pd.DataFrame(rows)
    dirs = rng.normal(size=(4, dim))
    idx_rows, emb = [], []
    for _, c in cases.iterrows():
        offset = rng.normal(scale=0.5, size=dim)
        for t in range(tiles):
            idx_rows.append({"emb_row": len(idx_rows), "item_id": f"bach512:{c.case_id}:{t}", "set": "bach512",
                             "case_id": c.case_id, "split": c.split, "grid": "g512", "tile_idx": t, "view": -1,
                             "n_views": 8, "degrade_type": "", "degrade_severity": 0, "label": "", "group": ""})
            emb.append(signal * dirs[CLASSES.index(c.label)] + offset + rng.normal(size=dim))
    return np.asarray(emb, dtype=np.float32), pd.DataFrame(idx_rows), cases


def run(emb, index, cases, **kw):
    kw.setdefault("probe_name", "conch_probe_oof")
    kw.setdefault("folds", int(CFG["cells"]["oof_folds"]))
    return tile_probe.oof_tile_risk(emb, index, cases, CFG, **kw)


def test_output_schema_and_fold_semantics():
    emb, index, cases = fake_world()
    out = run(emb, index, cases)
    schemas.validate_frame(schemas.TILE_RISK_OOF, out)
    assert list(out.columns) == ["case_id", "split", "grid", "tile_idx", "risk", "probe", "fold"]
    assert len(out) == len(index)
    assert (out["probe"] == "conch_probe_oof").all()
    assert out["risk"].between(0, 1).all()
    assert (out.loc[out.split == "train", "fold"] >= 0).all()
    assert (out.loc[out.split != "train", "fold"] == -1).all()
    assert set(out.loc[out.split == "train", "fold"]) == set(range(5))
    # one fold per case
    assert out[out.split == "train"].groupby("case_id")["fold"].nunique().max() == 1


def test_planted_signal_gives_high_oof_auroc():
    emb, index, cases = fake_world()
    out = run(emb, index, cases).merge(cases[["case_id", "label"]], on="case_id")
    for split in ("train", "test"):
        part = out[out.split == split]
        assert stats.auroc((part.label == AGG).astype(int).to_numpy(), part.risk.to_numpy()) > 0.9


def test_oof_rows_never_scored_by_a_model_that_saw_their_case():
    """Pure per-case noise in high dimension: a model that saw a case memorizes it (in-sample AUROC ~1);
    honest out-of-fold scores carry no label information (AUROC ~0.5)."""
    emb, index, cases = fake_world(n_train=80, tiles=4, dim=400, signal=0.0, seed=3, group_size=1)
    cfg = {**CFG, "tile_probe": {**CFG["tile_probe"], "C": 100.0}}
    out = tile_probe.oof_tile_risk(emb, index, cases, cfg, probe_name="p", folds=5)
    out = out.merge(cases[["case_id", "label"]], on="case_id")
    train = out[out.split == "train"]
    oof_auc = stats.auroc((train.label == AGG).astype(int).to_numpy(), train.risk.to_numpy())
    # in-sample comparison: score train tiles with the full-train fit (fold = -1 rule applied to train)
    full = tile_probe.fit_predict(emb[index.split == "train"], index[index.split == "train"].case_id.map(
        cases.set_index("case_id").label).to_numpy(), emb[index.split == "train"], cfg)
    in_auc = stats.auroc((index[index.split == "train"].case_id.map(cases.set_index("case_id").label) == AGG)
                         .astype(int).to_numpy(), full)
    assert in_auc > 0.95
    assert abs(oof_auc - 0.5) < 0.15


def test_case_folds_never_split_a_group_and_are_stratified():
    _, _, cases = fake_world(n_train=100, group_size=4)
    train = cases[cases.split == "train"]
    folds = tile_probe.case_folds(train, 5, seed=1)
    assert set(folds.index) == set(train.case_id)
    by_group = train.assign(fold=train.case_id.map(folds)).groupby("group")["fold"].nunique()
    assert by_group.max() == 1
    assert set(folds) == set(range(5))
    counts = train.assign(fold=train.case_id.map(folds)).groupby(["fold", "label"]).size().unstack(fill_value=0)
    assert (counts > 0).all().all()                           # every class in every fold


def test_case_folds_seeded_and_too_few_groups_raises():
    _, _, cases = fake_world(n_train=40)
    train = cases[cases.split == "train"]
    pd.testing.assert_series_equal(tile_probe.case_folds(train, 5, 2), tile_probe.case_folds(train, 5, 2))
    with pytest.raises(ValueError):
        tile_probe.case_folds(train.head(6), 5, 2)            # 2 groups < 5 folds


def test_deterministic():
    emb, index, cases = fake_world()
    pd.testing.assert_frame_equal(run(emb, index, cases), run(emb, index, cases))


def test_qc_filter_scores_only_passing_tiles():
    emb, index, cases = fake_world()
    tiles = index[["case_id", "grid", "tile_idx"]].copy()
    tiles["qc_pass"] = (tiles.tile_idx != 0).astype(int)
    out = run(emb, index, cases, qc_tiles=tiles)
    assert len(out) == int(tiles.qc_pass.sum())
    assert (out.tile_idx != 0).all()


def test_qc_filter_changes_training_set():
    emb, index, cases = fake_world()
    tiles = index[["case_id", "grid", "tile_idx"]].copy()
    tiles["qc_pass"] = 1
    a = run(emb, index, cases, qc_tiles=tiles)
    tiles.loc[tiles.tile_idx == 0, "qc_pass"] = 0
    b = run(emb, index, cases, qc_tiles=tiles)
    merged = a.merge(b, on=["case_id", "grid", "tile_idx"], suffixes=("_a", "_b"))
    assert not np.allclose(merged.risk_a, merged.risk_b)


def test_index_rows_of_unknown_cases_or_misalignment_raise():
    emb, index, cases = fake_world()
    with pytest.raises(ValueError, match="rows"):
        run(emb[:-1], index, cases)
    with pytest.raises(ValueError, match="unknown"):
        run(emb, index, cases[cases.case_id != "C000"])
    dup = pd.concat([index, index.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        run(np.vstack([emb, emb[:1]]), dup, cases)
    with pytest.raises(ValueError, match="folds"):
        run(emb, index, cases, folds=1)


def test_index_order_does_not_matter():
    emb, index, cases = fake_world()
    perm = np.random.default_rng(0).permutation(len(index))
    a = run(emb, index, cases)
    b = run(emb[perm], index.iloc[perm].reset_index(drop=True), cases)
    pd.testing.assert_frame_equal(a, b)


def test_missing_settings_key_raises():
    emb, index, cases = fake_world()
    cfg = {**CFG, "tile_probe": {k: v for k, v in CFG["tile_probe"].items() if k != "C"}}
    with pytest.raises(KeyError):
        tile_probe.oof_tile_risk(emb, index, cases, cfg, probe_name="p", folds=5)


def test_frozen_signature():
    import inspect
    sig = inspect.signature(tile_probe.oof_tile_risk)
    assert list(sig.parameters) == ["emb", "index", "cases", "cfg", "probe_name", "folds", "qc_tiles"]
    assert [p.name for p in sig.parameters.values() if p.kind is inspect.Parameter.KEYWORD_ONLY] == \
        ["probe_name", "folds", "qc_tiles"]
    assert sig.parameters["qc_tiles"].default is None


def test_module_is_delta_portable():
    """No pydantic / fastapi / core.contract / core.tiers imports (Agent 1's 04a imports this inside a Slurm job)."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(tile_probe.__file__).read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {f"{n.module}.{a.name}" if n.module else a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
              for a in n.names}
    banned = ("pydantic", "fastapi", "core.contract", "core.tiers", "torch")
    assert not [m for m in names if m.startswith(banned)]
