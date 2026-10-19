"""pipeline/07_retrieval.py on the synthetic root after 05."""

import importlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from core import config, schemas
from tests.a3_synthetic import after_05, clone_root

step07 = importlib.import_module("pipeline.07_retrieval")


# AI spot-check summary written before 07 runs: >= 0.75 acceptable passes (tubules sits exactly on the cut-off),
# below fails; concepts absent from the summary, or with nothing reviewed, have no spot check (null, not reliable).
SPOT = {"tubules": {"n": 20, "acceptable": 15}, "pleomorphism": {"n": 20, "acceptable": 14},
        "invasion": {"n": 20, "acceptable": 19}, "duct_confined": {"n": 20, "acceptable": 3},
        "normal_lobules": {"n": 20, "acceptable": 0}, "mitoses": {"n": 0, "acceptable": 0}}
RELIABLE = {"tubules": True, "pleomorphism": False, "invasion": True, "duct_confined": False,
            "normal_lobules": False, "mitoses": False, "benign_stroma_fat": False, "lymphocytes": False,
            "necrosis": False}


def write_spotcheck(art):
    path = schemas.eval_json(art, "spotcheck")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": "spotcheck", "reviewer_kind": "ai", "reviews": [],
                                "summary": {"concept": SPOT, "cells": {"n": 0, "acceptable": 0}}}))


def run_on_copy(work, mutate=None):
    root0, art0, cfg_path = after_05()
    root, art = work / "data", work / "art"
    clone_root(root0, root)
    shutil.copytree(art0, art)
    write_spotcheck(art)
    if mutate is not None:
        mutate(root)
    assert step07.main(["--data-root", str(root), "--artifacts-root", str(art), "--config", str(cfg_path)]) == 0
    return root, art, config.load_settings(cfg_path)


@pytest.fixture(scope="module")
def run07(tmp_path_factory):
    return run_on_copy(tmp_path_factory.mktemp("r07"))


def train_z(root, cfg):
    """case_features c__top<k>, z-scored on the train library cases (07's ranking scale)."""
    k = int(cfg["concepts"]["top_k"])
    cols = [f"{c}__top{k}" for c in cfg["concepts"]["ids"]]
    feats = pd.read_csv(schemas.case_features_csv(root), dtype={"case_id": str}).set_index("case_id")[cols]
    cases = pd.read_csv(schemas.cases_csv(root), dtype={"case_id": str})
    train = feats.loc[[i for i in cases[cases.split == "train"].case_id if i in feats.index]]
    return (feats - train.mean()) / train.std(ddof=0).replace(0, 1.0)


def test_retrieval_json_contract(run07):
    root, _, cfg = run07
    doc = json.loads(schemas.retrieval_json(root).read_text())
    assert not schemas.RETRIEVAL.problems(doc) and doc["synthetic"] is True
    cases = pd.read_csv(schemas.cases_csv(root), dtype={"case_id": str})
    pred = pd.read_csv(schemas.predictions_csv(root), dtype={"case_id": str})
    want = set(pred[pred.split != "train"].case_id)
    assert want == set(doc["cases"])
    train = set(cases[cases.split == "train"].case_id)
    top = dict(zip(pred.case_id, pred.top))
    label = dict(zip(cases.case_id, cases.label))
    for cid, rec in list(doc["cases"].items())[:200]:
        assert len(rec["neighbors"]) == cfg["retrieval"]["k"] and rec["of"] == cfg["retrieval"]["k"]
        assert all(n["case_id"] in train and n["label"] == label[n["case_id"]] for n in rec["neighbors"])
        sims = [n["similarity"] for n in rec["neighbors"]]
        assert sims == sorted(sims, reverse=True)
        assert rec["predicted"] == top[cid]
        assert rec["agree"] == sum(n["label"] == rec["predicted"] for n in rec["neighbors"])
        assert rec["neighbor_agreement"] == pytest.approx(rec["agree"] / rec["of"])
        c = rec["contrast"]
        assert c["label"] != rec["predicted"] and c["case_id"] in train
        assert len(c["differs_most_in"]) == cfg["retrieval"]["differs_most_in_k"]
        assert set(c["differs_most_in"]) <= set(cfg["concepts"]["ids"])
        # contrast = the nearest library case with a different label
        assert c["similarity"] <= sims[0] + 1e-9


def test_precision_at_k_eval(run07):
    _, art, _ = run07
    ev = json.loads(schemas.eval_json(art, "retrieval").read_text())
    p = ev["precision_at_k_test"]
    assert p["lo"] <= p["estimate"] <= p["hi"] and p["estimate"] > 0.4 and p["n_groups"] > 10


# ---- contrast detail (judge feedback T4) -------------------------------------------------------------------------------
DETAIL_KEYS = {"concept", "query_z", "contrast_z", "delta", "spotcheck", "reliable", "query_region", "contrast_region"}


def test_contrast_detail_matches_differs_most_in(run07):
    root, _, cfg = run07
    doc = json.loads(schemas.retrieval_json(root).read_text())
    z = train_z(root, cfg)
    ids = list(cfg["concepts"]["ids"])
    k = int(cfg["concepts"]["top_k"])
    tiles = np.load(schemas.concepts_tile_npz(root))
    n_detail = n_region = 0
    for cid, rec in doc["cases"].items():
        c = rec["contrast"]
        detail = c["detail"]
        assert detail is not None, cid                                  # every synthetic read has features
        n_detail += 1
        assert [d["concept"] for d in detail] == c["differs_most_in"]   # same concepts, same order
        delta_all = (z.loc[cid] - z.loc[c["case_id"]]).to_numpy()
        ranked = [ids[t] for t in np.argsort(-np.abs(delta_all), kind="stable")[:cfg["retrieval"]["differs_most_in_k"]]]
        assert ranked == c["differs_most_in"]
        for d in detail:
            assert set(d) == DETAIL_KEYS
            col = f"{d['concept']}__top{k}"
            qz, rz = float(z.loc[cid, col]), float(z.loc[c["case_id"], col])
            assert d["query_z"] == round(qz, 2) and d["contrast_z"] == round(rz, 2)
            assert d["delta"] == round(qz - rz, 2)
            assert abs(d["delta"]) == pytest.approx(round(abs(qz - rz), 2))   # |delta| is the ranking key
            for who, region in ((cid, d["query_region"]), (c["case_id"], d["contrast_region"])):
                rows = tiles["case_id"] == who
                if not rows.any():
                    assert region is None                               # no per-tile scores for this read
                    continue
                ci = ids.index(d["concept"])
                zz, idx = tiles["z"][rows, ci], tiles["tile_idx"][rows]
                assert region == int(idx[np.lexsort((idx, -zz))[0]])    # highest tile score, lowest idx on ties
                n_region += 1
    assert n_detail == len(doc["cases"]) and n_region > 0


def test_contrast_detail_marks_failed_spotcheck(run07):
    root, _, _ = run07
    doc = json.loads(schemas.retrieval_json(root).read_text())
    seen = {True: 0, False: 0}
    for rec in doc["cases"].values():
        for d in rec["contrast"]["detail"]:
            spot = SPOT.get(d["concept"])
            if spot is None or spot["n"] == 0:
                assert d["spotcheck"] is None
            else:
                assert d["spotcheck"] == {"acceptable": spot["acceptable"], "reviewed": spot["n"]}
            assert d["reliable"] is RELIABLE[d["concept"]], d
            seen[d["reliable"]] += 1
    assert seen[True] > 0 and seen[False] > 0


def test_contrast_detail_unit_rules():
    z = pd.DataFrame({"a__top3": [2.0, -1.0], "b__top3": [0.5, 0.4], "c__top3": [-1.25, 1.0]}, index=["q", "r"])
    tiles = {"q": (np.array([3, 1, 2]), np.array([[0.1, 0.0, 2.0], [0.9, 0.0, 2.0], [0.9, 0.0, -1.0]])),
             "r": (np.array([0, 5]), np.array([[0.2, 0.3, 0.1], [0.7, 0.3, 0.4]]))}
    spot = {"a": {"acceptable": 0, "reviewed": 20}, "c": {"acceptable": 18, "reviewed": 20}}
    dz, detail = step07.contrast_detail("q", "r", z, ["a", "b", "c"], 2, tiles, spot)
    assert dz == ["a", "c"]
    assert detail == [
        {"concept": "a", "query_z": 2.0, "contrast_z": -1.0, "delta": 3.0, "spotcheck": {"acceptable": 0, "reviewed": 20},
         "reliable": False, "query_region": 1, "contrast_region": 5},
        {"concept": "c", "query_z": -1.25, "contrast_z": 1.0, "delta": -2.25,
         "spotcheck": {"acceptable": 18, "reviewed": 20}, "reliable": True, "query_region": 1, "contrast_region": 5},
    ]
    assert step07.contrast_detail("q", "missing", z, ["a", "b", "c"], 2, tiles, spot) == ([], None)
    assert step07.contrast_detail("q", "r", None, ["a", "b", "c"], 2, tiles, spot) == ([], None)
    _, no_tiles = step07.contrast_detail("q", "r", z, ["a", "b", "c"], 2, {}, {})
    assert [(d["query_region"], d["contrast_region"], d["spotcheck"], d["reliable"]) for d in no_tiles] == \
        [(None, None, None, False)] * 2


def test_contrast_detail_null_when_features_missing(tmp_path):
    def drop_variants(root):                                            # degraded / composite reads lose features
        path = schemas.case_features_csv(root)
        df = pd.read_csv(path, dtype={"case_id": str})
        clean = set(pd.read_csv(schemas.cases_csv(root), dtype={"case_id": str}).case_id)
        df[df.case_id.isin(clean)].to_csv(path, index=False)

    root, _, _ = run_on_copy(tmp_path, drop_variants)
    doc = json.loads(schemas.retrieval_json(root).read_text())
    clean = set(pd.read_csv(schemas.cases_csv(root), dtype={"case_id": str}).case_id)
    variants = [cid for cid in doc["cases"] if cid not in clean]
    assert variants and len(variants) < len(doc["cases"])
    for cid, rec in doc["cases"].items():
        c = rec["contrast"]
        assert c is not None
        if cid in variants:
            assert c["detail"] is None and c["differs_most_in"] == []   # never zeros
        else:
            assert c["detail"] is not None and len(c["detail"]) == len(c["differs_most_in"]) > 0
