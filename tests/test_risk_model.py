"""pipeline/05_risk_model.py end to end on the small synthetic root (with test-only fake concept features)."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

from core import config, conformal, contract, margin, schemas
from tests.a3_synthetic import after_05

step05 = importlib.import_module("pipeline.05_risk_model")
CFG = config.load_settings()
CLASSES = list(CFG["classes"])


@pytest.fixture(scope="module")
def run05():
    root, art, cfg_path = after_05()
    return root, art, config.load_settings(cfg_path)


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"case_id": str})


def test_predictions_schema_rows_and_probabilities(run05):
    root, _, cfg = run05
    df = read(schemas.predictions_csv(root))
    schemas.validate_frame(schemas.PREDICTIONS, df)
    cases = read(schemas.cases_csv(root))
    clean = df[~df.case_id.str.contains(r"_FSIM_|_COMP_|^COMP_")]
    assert set(clean[clean.split != "train"].case_id) == set(cases[cases.split.isin(["calibration", "test"])].case_id)
    assert set(clean[clean.split == "train"].case_id) == set(cases[cases.split == "train"].case_id)
    pcal = df[[schemas.cal_prob_col(c) for c in CLASSES]].to_numpy()
    np.testing.assert_allclose(pcal.sum(1), 1.0, atol=1e-9)
    np.testing.assert_allclose(df.risk, df[schemas.cal_prob_col(CFG["aggressive_class"])])
    assert (df.top == np.array(CLASSES)[pcal.argmax(1)]).all()
    # every degraded (deg512) and composite variant is predicted
    deg = read(schemas.emb_index_csv(root, "conch", "deg512", "probe"))
    want = {schemas.variant_case_id(c, t, int(s)) for c, t, s in zip(deg.case_id, deg.degrade_type, deg.degrade_severity)}
    assert want <= set(df.case_id)
    comp = read(schemas.composites_csv(root))
    assert set(comp.case_id) <= set(df.case_id)
    # every evaluated model has uncalibrated columns (A, B and the MUSK agreement model)
    choice = json.loads(schemas.eval_json(run05[1], "model_choice").read_text())
    for model in {choice["model_a"], choice["model_b"], "musk_lr"}:
        assert all(schemas.prob_col(model, c) in df.columns for c in CLASSES)
    assert df.agree_top.notna().any()


def test_sets_match_qhat_and_lac_definition(run05):
    root, _, _ = run05
    df = read(schemas.predictions_csv(root))
    for row in df.sample(40, random_state=0).itertuples():
        p = np.array([getattr(row, schemas.cal_prob_col(c)) for c in CLASSES])
        mask = conformal.lac_sets(p[None, :], row.qhat_lac)[0]
        assert schemas.parse_set(row.set_lac) == conformal.set_labels(mask, CLASSES)
        assert row.set_size == mask.sum()


def test_conformal_qhat_uses_calibration_only(run05):
    root, art, cfg = run05
    df = read(schemas.predictions_csv(root))
    choice = json.loads(schemas.eval_json(art, "model_choice").read_text())
    prod = choice["production"]
    cases = read(schemas.cases_csv(root)).set_index("case_id")
    cal = df[(df.split == "calibration") & (df.model == prod)]
    y = np.array([CLASSES.index(cases.label[c]) for c in cal.case_id])
    p = cal[[schemas.cal_prob_col(c) for c in CLASSES]].to_numpy()
    assert conformal.fit_lac(p, y, cfg["alpha"]) == pytest.approx(choice["conformal"][prod]["qhat_lac"])
    assert choice["protocol"]["test_split_used"] is False


def test_model_choice_records_cv_selection(run05):
    _, art, cfg = run05
    choice = json.loads(schemas.eval_json(art, "model_choice").read_text())
    for key in ("name", "created_utc", "config_hash", "git_sha", "synthetic"):
        assert key in choice
    assert choice["synthetic"] is True
    assert choice["production"] in {choice["model_a"], choice["model_b"]}
    assert choice["a_vs_b"]["auroc_margin"] == cfg["model_choice"]["auroc_margin"]
    names = {r["model"] for r in choice["candidates"]}
    assert {"conch_probe_lr", "musk_lr", "uni2h_lr", "hoptimus0_lr", "cb_text", "cb_text_cells", "cb_cells",
            "fusion_top3", "abmil"} <= names
    for fam in ("black_box", "concept"):
        info = choice["family_choice"][fam]
        assert info["chosen"] in info["eligible"]
    m = choice["cv_metrics"][choice["production"]]["auroc_invasive"]
    assert m["lo"] <= m["estimate"] <= m["hi"] and m["n_groups"] > 10
    assert set(choice["ablation_concepts_cells"]) == {"cb_text", "cb_text_cells", "cb_cells"}
    assert 0.05 <= min(choice["temperatures"].values()) and max(choice["temperatures"].values()) <= 20


def test_planted_signal_recovered_out_of_fold(run05):
    _, art, _ = run05
    choice = json.loads(schemas.eval_json(art, "model_choice").read_text())
    assert choice["cv_metrics"]["conch_probe_lr"]["auroc_invasive"]["estimate"] > 0.85
    assert choice["cv_metrics"]["cb_text"]["auroc_invasive"]["estimate"] > 0.8


def test_contributions_sum_and_schema(run05):
    root, art, _ = run05
    con = read(schemas.contributions_csv(root))
    schemas.validate_frame(schemas.CONTRIBUTIONS, con)
    choice = json.loads(schemas.eval_json(art, "model_choice").read_text())
    assert choice["model_b"] in set(con.model) and set(con.model) <= {choice["model_b"]} | set(choice["variants"]["fallback_order"])
    assert set(con.split) <= {"calibration", "test"}
    assert con.case_id.str.contains("_FSIM_").any()
    per_case = con.groupby("case_id").contribution.sum()
    assert per_case.std() > 0


def test_tile_risk_and_oof(run05):
    root, _, _ = run05
    tr = read(schemas.tile_risk_csv(root))
    schemas.validate_frame(schemas.TILE_RISK, tr)
    assert set(tr.split) <= {"calibration", "test"} and (tr.fold == -1).all()
    assert tr.case_id.str.contains("_FSIM_").any() and (tr.case_id.str.contains("_COMP_") | tr.case_id.str.startswith("COMP_")).any()
    oof = read(schemas.tile_risk_csv(root, oof=True))
    schemas.validate_frame(schemas.TILE_RISK_OOF, oof)
    assert (oof[oof.split == "train"].fold >= 0).all()


def test_margins_json_covers_reads_and_validates(run05):
    root, _, cfg = run05
    doc = json.loads(schemas.margins_json(root).read_text())
    assert not schemas.MARGINS_JSON.problems(doc)
    assert doc["synthetic"] is True and doc["coarse"] is False
    cases = read(schemas.cases_csv(root))
    need = set(cases[cases.split.isin(["calibration", "test"])].case_id)
    need |= set(read(schemas.composites_csv(root)).case_id)
    need |= set(read(schemas.wsi_windows_csv(root)).window_id)
    assert need <= set(doc["cases"])
    surg = cases[(cases.split == "test") & (cases.lane == "surgical")]
    sev = cfg["degrade"]["display_severity"]
    for cid in surg.case_id:
        for t in cfg["degrade"]["types"]:
            assert schemas.variant_case_id(cid, t, sev) in doc["cases"]
    assert doc["n_benign_cal_images"] == int(((cases.split == "calibration") & cases.label.isin(cfg["benign_classes"])).sum())
    assert doc["threshold_ci"]["lo"] <= doc["threshold"] <= doc["threshold_ci"]["hi"]
    assert doc["benign_cal_fpr"] <= cfg["margin"]["max_fpr"]
    for cid in list(doc["cases"])[:30]:
        rec = doc["cases"][cid]
        call, involved = margin.suggest_call(rec["per_margin"], doc["threshold"], False)
        contract.Margin.model_validate({"suggested_call": call, "involved": involved, "per_margin": rec["per_margin"],
                                        "threshold": doc["threshold"], "segments": rec["segments"]})


def test_margin_probe_localizes_on_composites(run05):
    """Benign-half (exact_negative) margins of composites are mostly clear; carcinoma-side ones are involved more often
    (a carcinoma lesion need not reach the carcinoma-side edge)."""
    root, _, cfg = run05
    doc = json.loads(schemas.margins_json(root).read_text())
    comp = read(schemas.composites_csv(root))
    neg, pos = [], []
    for r in comp.itertuples():
        pm = doc["cases"][r.case_id]["per_margin"]
        for m in schemas.MARGINS:
            truth = getattr(r, f"truth_{m}")
            (neg if truth == "exact_negative" else pos if truth == "tumor_side" else []).append(pm[m] > doc["threshold"])
    assert np.mean(neg) <= 0.25                          # an echo would call these involved whenever the case is
    assert np.mean(pos) >= np.mean(neg) + 0.25           # carcinoma; a localizer leaves them clear


def test_rerun_skips_and_view0_file(run05):
    root, art, _ = run05
    v0 = read(root / "predictions_view0.csv")
    assert set(v0.split) == {"calibration", "test"} and (v0.read == "view0").all() or (v0.read != "view0").all()
    before = schemas.predictions_csv(root).stat().st_mtime_ns
    assert step05.main(["--data-root", str(root), "--artifacts-root", str(art)]) == 0
    assert schemas.predictions_csv(root).stat().st_mtime_ns == before


def test_tier_set_follows_configured_method_and_is_nonempty(run05):
    root, _, cfg = run05
    df = read(schemas.predictions_csv(root))
    source = conformal.tier_set_source(cfg)
    nonempty = bool(cfg["conformal"].get("nonempty", False))
    for row in df.itertuples():
        base = schemas.parse_set(getattr(row, source))
        tier = schemas.parse_set(row.set_tier)
        assert row.set_tier_size == len(tier)
        assert set(base) <= set(tier)
        if base or not nonempty:
            assert tier == base
        else:
            assert tier == [row.top]
