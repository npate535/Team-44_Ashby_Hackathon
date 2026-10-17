"""eval.cells_eval: pending sections for missing inputs, direction checks (incl. wrong-way reporting), convergent
validity on planted correlations, review sheet without labels, Agent 3 file pass-through, disabled flag."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from core import io, schemas
from pipeline import cells_convert as cc
from tests import synthetic_tree as st

ev = importlib.import_module("eval.cells_eval")


def _build(tmp_path, n_per_class=6):
    cfg_path, cfg = st.small_settings(tmp_path, **{"bootstrap.n": 200})
    data, art = tmp_path / "data", tmp_path / "art"
    specs = []
    k = 1
    for split, n in (("train", 2), ("test", n_per_class)):
        for cls in cfg["classes"]:
            for _ in range(n):
                specs.append((f"BACH_{k:04d}", cls, split, "biopsy"))
                k += 1
    cases = st.write_cases_csv(data, cfg, specs)
    cases["group"] = [f"g{i // 2}" for i in range(len(cases))]  # pairs of cases share a pseudo-patient
    cases.to_csv(data / "cases.csv", index=False)
    fcfg = cc.load_feature_config(cfg)
    ids = [f["id"] for f in fcfg["features"]]
    rng = np.random.default_rng(0)
    case_rows, tile_rows, sel_rows, z_rows = [], [], [], []
    for c in cases.itertuples(index=False):
        inv = 1.0 if c.label == "Invasive" else 0.0
        base = {f: float(rng.uniform(0.2, 0.4)) for f in ids}
        base["neo_fraction"] += 0.4 * inv          # up, as expected
        base["epi_fraction"] += 0.4 * inv          # down expected -> wrong way
        row = {"case_id": c.case_id, "split": c.split, "n_tiles": 2, "n_nuclei": 100, "n_neoplastic": 40,
               "insufficient_tumor_nuclei": 0}
        for f in ids:
            row.update({f: base[f], f"{f}__missing": 0, f"{f}__ref_pct": 0.5})
        case_rows.append(row)
        for rank, t in enumerate((1, 2), start=1):
            ov = schemas.cell_overlay_png(data, c.case_id, t)
            ov.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (200, 100, 150)).save(ov)
            cv = float(rng.normal())
            tile_rows.append({"case_id": c.case_id, "tile_idx": t, "n_nuclei": 50,
                              **{f"n_{ty}": 10 for ty in cfg["cells"]["types"]},
                              "overlay": ov.relative_to(data).as_posix(), **base, "nuc_area_cv": cv,
                              "infl_fraction": float(rng.uniform())})
            sel_rows.append({"case_id": c.case_id, "tile_idx": t, "split": c.split, "rank": rank, "tile_risk": 0.5,
                             "path": f"tiles/g512/{c.case_id}/0_{t}.png", "n_tiles": 2})
            z_rows.append((c.case_id, t, cv))
    pd.DataFrame(case_rows).to_csv(schemas.cell_features_csv(data), index=False)
    pd.DataFrame(tile_rows).to_csv(schemas.cell_features_csv(data, per_tile=True), index=False)
    pd.DataFrame(sel_rows).to_csv(schemas.cell_tiles_csv(data), index=False)
    return cfg_path, cfg, data, art, z_rows


def _run(cfg_path, data, art):
    assert ev.main(["--config", str(cfg_path), "--data-root", str(data), "--artifacts-root", str(art)]) == 0
    return json.loads(schemas.eval_json(art, "cells").read_text())


def test_pending_sections_and_direction_checks(tmp_path):
    cfg_path, cfg, data, art, _ = _build(tmp_path)
    out = _run(cfg_path, data, art)
    for k in schemas.EVAL_ENVELOPE:
        assert k in out
    assert out["name"] == "cells" and out["status"] == "ok" and out["split"] == "test"
    assert out["convergent_validity"]["status"] == "pending"
    assert out["ablation"]["status"] == "pending" and out["faithfulness"]["status"] == "pending"
    dc = out["direction_checks"]
    assert dc["features"]["neo_fraction"]["verdict"] == "as_expected"
    assert dc["features"]["epi_fraction"]["verdict"] == "wrong_way" and "epi_fraction" in dc["wrong_way"]
    assert dc["features"]["infl_fraction"]["verdict"] == "context"
    m = dc["features"]["neo_fraction"]["class_means"]["Invasive"]
    assert m["lo"] <= m["estimate"] <= m["hi"] and m["n_groups"] == 3  # 6 test cases in pseudo-patient pairs
    assert (art / "eval" / "cells_direction.png").is_file() and out["figures"] == ["eval/cells_direction.png"]


def test_review_sheet_hides_labels(tmp_path):
    cfg_path, cfg, data, art, _ = _build(tmp_path)
    out = _run(cfg_path, data, art)
    rs = out["review_sheet"]
    assert rs["status"] == "ok" and rs["n_items"] == cfg["cells"]["review_per_class"] * len(cfg["classes"])
    sheet = pd.read_csv(art / "eval" / "cells_review.csv", dtype=str, keep_default_na=False)
    assert list(sheet.columns) == ["item_id", "case_id", "tile_idx", "overlay", "verdict", "note"]
    page = (art / "eval" / "cells_review.html").read_text()
    for cls in cfg["classes"]:
        assert cls not in page and cls not in (art / "eval" / "cells_review.csv").read_text()
    assert page.count("data:image/png;base64,") == rs["n_items"]
    cases = pd.read_csv(data / "cases.csv", dtype=str).set_index("case_id")
    labels = cases.loc[sheet["case_id"], "label"]
    assert labels.value_counts().to_dict() == {c: 5 for c in cfg["classes"]}
    assert labels.iloc[:5].nunique() > 1  # shuffled: classes are not blocked in order


def test_convergent_validity_and_agent3_files(tmp_path):
    cfg_path, cfg, data, art, z_rows = _build(tmp_path)
    ids = cfg["concepts"]["ids"]
    zmat = np.zeros((len(z_rows), len(ids)), np.float32)
    rng = np.random.default_rng(1)
    for i, (_, _, cv) in enumerate(z_rows):
        zmat[i, ids.index("pleomorphism")] = cv + 0.1 * rng.normal()   # planted agreement
        zmat[i, ids.index("lymphocytes")] = rng.normal()               # unrelated
    io.save_npz(schemas.concepts_tile_npz(data), case_id=np.array([r[0] for r in z_rows]),
                tile_idx=np.array([r[1] for r in z_rows], np.int32), grid=np.array("g512"),
                concept_ids=np.array(ids), z=zmat, z_text=zmat, z_cav=zmat * np.nan, spread=zmat * 0,
                encoder=np.array("conch"), synthetic=np.array(True))
    (art / "eval").mkdir(parents=True, exist_ok=True)
    io.write_json(schemas.eval_json(art, "model_choice"), {"candidates": [
        {"model": "cb_text", "cv_log_loss": 0.9}, {"model": "cb_text_cells", "cv_log_loss": 0.85},
        {"model": "cb_cells", "cv_log_loss": 1.0}, {"model": "conch_probe_lr", "cv_log_loss": 0.7}]})
    io.write_json(schemas.eval_json(art, "deletion"), {"concepts": {"x": 1}, "cell_features": {"mean_drop": 0.1}})
    out = _run(cfg_path, data, art)
    cvd = out["convergent_validity"]
    assert cvd["status"] == "ok"
    assert cvd["pairs"]["nuc_area_cv~pleomorphism"]["tile"]["estimate"] > 0.8
    assert abs(cvd["pairs"]["infl_fraction~lymphocytes"]["tile"]["estimate"]) < 0.5
    assert {r["model"] for r in out["ablation"]["rows"]} == {"cb_text", "cb_text_cells", "cb_cells"}
    assert out["faithfulness"]["cell_features"] == {"mean_drop": 0.1}


def test_missing_features_pending_and_disabled(tmp_path):
    cfg_path, _ = st.small_settings(tmp_path)
    st.write_cases_csv(tmp_path / "data", st.small_settings(tmp_path)[1], [("A_0001", "Normal", "test", "biopsy")])
    out = _run(cfg_path, tmp_path / "data", tmp_path / "art")
    assert out["status"] == "pending"
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    off, _ = st.small_settings(off_dir, **{"cells.enabled": False})
    assert ev.main(["--config", str(off), "--data-root", str(tmp_path / "data"), "--artifacts-root",
                    str(off_dir / "art")]) == 0
    assert json.loads(schemas.eval_json(off_dir / "art", "cells").read_text())["status"] == "disabled"
