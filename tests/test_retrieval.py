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


@pytest.fixture(scope="module")
def run07(tmp_path_factory):
    root0, art0, cfg_path = after_05()
    work = tmp_path_factory.mktemp("r07")
    root, art = work / "data", work / "art"
    clone_root(root0, root)
    shutil.copytree(art0, art)
    assert step07.main(["--data-root", str(root), "--artifacts-root", str(art), "--config", str(cfg_path)]) == 0
    return root, art, config.load_settings(cfg_path)


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
