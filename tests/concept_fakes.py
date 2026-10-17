"""Schema-exact fake data roots with planted concept structure for 04_concepts / 04_cav tests (Agent 2).

Embeddings live in a small D-dim space: each concept c has a unit direction u_c and each class a direction v_k.
Prompt embeddings: concept positives ~ u_c, contrasts ~ -u_c, class prompts ~ v_k. Tile embeddings of a case of
class k ~ v_k + planted concept directions (Invasive tiles carry +u_invasion, InSitu +u_duct_confined, Normal
+u_normal_lobules, ...). NCT tiles of class LYM ~ u_lymphocytes etc. Mined Quilt images: positives ~ u_c, negatives
~ -u_c, grouped by fake video. All arrays are random numbers: no image of anything.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core import config, schemas

text = importlib.import_module("pipeline.03_text")

D = 24
CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]
PLANT = {"Invasive": ["invasion", "pleomorphism", "necrosis"], "InSitu": ["duct_confined", "pleomorphism"],
         "Normal": ["normal_lobules", "tubules", "benign_stroma_fat"], "Benign": ["benign_stroma_fat", "tubules"]}
NCT_PLANT = {"LYM": "lymphocytes", "DEB": "necrosis", "TUM": "pleomorphism", "STR": "benign_stroma_fat",
             "ADI": "benign_stroma_fat"}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Fake:
    def __init__(self, root: Path, *, seed: int = 0, per_class: int = 8, bootstrap_n: int = 30):
        self.root = root
        self.rng = np.random.default_rng(seed)
        self.cfg = config.load_settings()
        self.ids = list(self.cfg["concepts"]["ids"])
        basis = np.linalg.qr(self.rng.standard_normal((D, D)))[0]
        self.u = {cid: basis[i] for i, cid in enumerate(self.ids)}
        self.v = {cls: basis[len(self.ids) + i] for i, cls in enumerate(CLASSES)}
        self.per_class = per_class
        settings = yaml.safe_load(config.SETTINGS_PATH.read_text())
        settings["bootstrap"]["n"] = bootstrap_n
        self.settings_path = root / "settings_test.yaml"
        root.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(yaml.safe_dump(settings))

    # ---- tables ----
    def cases(self) -> pd.DataFrame:
        rows, n = [], 0
        for cls in CLASSES:
            for i in range(self.per_class):
                split = "train" if i < self.per_class // 2 else ("calibration" if i < self.per_class * 5 // 8 else "test")
                rows.append({"case_id": f"BACH_{n:04d}", "source": "synthetic", "source_file": f"{cls}/{i}.tif",
                             "source_index": n, "image_path": f"/x/{n}.tif", "label": cls, "split": split,
                             "patient_id": "", "group": f"b{n // 2}", "lane": "biopsy", "width": 2048, "height": 1536,
                             "display_path": f"display/BACH_{n:04d}.jpg"})
                n += 1
        df = pd.DataFrame(rows)
        schemas.validate_frame(schemas.CASES, df)
        return df

    def tiles(self, cases: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for case_id in cases["case_id"]:
            for t in range(12):
                r, c = divmod(t, 4)
                rows.append({"case_id": case_id, "grid": "g512", "tile_idx": t, "row": r, "col": c, "x": c * 512,
                             "y": r * 512, "size": 512, "tissue_frac": 0.1 if t == 11 else 0.9,
                             "qc_pass": 0 if t == 11 else 1, "path": f"tiles/g512/{case_id}/{r}_{c}.png"})
        df = pd.DataFrame(rows)
        schemas.validate_frame(schemas.TILES, df)
        return df

    def tile_vec(self, cls: str, strength: float = 1.2, noise: float = 0.35) -> np.ndarray:
        vec = self.v[cls] + noise * self.rng.standard_normal(D)
        for cid in PLANT[cls]:
            vec = vec + strength * self.u[cid] * self.rng.uniform(0.3, 1.0)
        return _unit(vec)

    def write_index(self, encoder: str, set_name: str, rows: list[dict[str, Any]], emb: np.ndarray) -> None:
        df = pd.DataFrame(rows)
        df.insert(0, "emb_row", np.arange(len(df)))
        for col in [c.name for c in schemas.EMB_INDEX.columns]:
            if col not in df:
                df[col] = {"view": -1, "n_views": 1, "degrade_severity": 0, "tile_idx": -1}.get(col, "")
        df = df[[c.name for c in schemas.EMB_INDEX.columns]]
        schemas.validate_frame(schemas.EMB_INDEX, df)
        out = schemas.emb_npy(self.root, encoder, set_name, "align")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, emb.astype(np.float16))
        df.to_csv(schemas.emb_index_csv(self.root, encoder, set_name, "align"), index=False)

    def text_npz(self, encoder: str, bank: schemas.ConceptBank) -> None:
        ids, texts = text.prompt_table(bank)
        rows = []
        for pid in ids:
            parts = pid.split(":")
            if parts[0] == "concept":
                base = self.u[parts[1]] * (1.0 if parts[2] == "pos" else -1.0)
            elif parts[0] == "class":
                base = self.v[parts[1]]
            else:
                base = self.u.get(NCT_PLANT.get(parts[1], ""), self.rng.standard_normal(D) * 0.1)
            rows.append(_unit(base + 0.2 * self.rng.standard_normal(D)))
        path = schemas.text_npz(self.root, encoder)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=np.asarray(ids, dtype=str), texts=np.asarray(texts, dtype=str),
                 emb=np.asarray(rows, np.float32), model=np.asarray(encoder), synthetic=np.asarray(True))

    def build(self, *, encoders: tuple[str, ...] = ("conch",), deg: bool = True, comp: bool = True,
              nct: bool = True, quilt: bool = False) -> dict[str, Any]:
        cases = self.cases()
        tiles = self.tiles(cases)
        cases.to_csv(schemas.cases_csv(self.root), index=False)
        tiles.to_csv(schemas.tiles_csv(self.root), index=False)
        bank = config.load_concept_bank(cfg=self.cfg)
        label = dict(zip(cases["case_id"], cases["label"]))
        split = dict(zip(cases["case_id"], cases["split"]))
        for enc in encoders:
            self.text_npz(enc, bank)
            rows, emb = [], []
            for case_id in cases["case_id"]:
                for t in range(12):
                    rows.append({"item_id": f"bach512:{case_id}:{t}", "set": "bach512", "case_id": case_id,
                                 "split": split[case_id], "grid": "g512", "tile_idx": t, "n_views": 8})
                    emb.append(self.tile_vec(label[case_id]))
            self.write_index(enc, "bach512", rows, np.asarray(emb))
        test_cases = cases[cases["split"] == "test"]["case_id"].tolist()
        if deg:
            rows, emb = [], []
            for case_id in test_cases[:3]:
                for dtype, sev in (("blur", 3), ("jpeg", 1)):
                    for t in range(12):
                        var_id = schemas.variant_case_id(case_id, dtype, sev)  # 01_build_cases convention
                        rows.append({"item_id": f"deg512:{var_id}:{t}:{dtype}{sev}", "set": "deg512",
                                     "case_id": var_id, "split": "test", "grid": "g512", "tile_idx": t,
                                     "degrade_type": dtype, "degrade_severity": sev})
                        emb.append(self.tile_vec(label[case_id], strength=0.6, noise=0.8))
            self.write_index("conch", "deg512", rows, np.asarray(emb))
        if comp:
            carcinoma = [c for c in test_cases if label[c] in ("Invasive", "InSitu")][0]
            benign = [c for c in test_cases if label[c] in ("Normal", "Benign")][0]
            comp_id = schemas.composite_case_id(carcinoma, 0)
            pd.DataFrame([{"case_id": comp_id, "carcinoma_case_id": carcinoma, "benign_case_id": benign,
                           "orientation": 0, "carcinoma_side": "left", "width": 2048, "height": 1536,
                           "path": f"composites/{comp_id}.png", "display_path": f"display/{comp_id}.jpg",
                           **{f"truth_{m}": "mixed" for m in schemas.MARGINS}}]).to_csv(
                schemas.composites_csv(self.root), index=False)
            rows = [{"item_id": f"comp512:{comp_id}:{t}", "set": "comp512", "case_id": comp_id, "split": "",
                     "grid": "g512", "tile_idx": t} for t in range(12)]
            emb = [self.tile_vec(label[carcinoma] if t % 4 < 2 else label[benign]) for t in range(12)]
            self.write_index("conch", "comp512", rows, np.asarray(emb))
        if nct:
            rows, emb = [], []
            for cls in self.cfg["nct"]["classes"]:
                for i in range(20):
                    base = self.u[NCT_PLANT[cls]] if cls in NCT_PLANT else self.rng.standard_normal(D) * 0.3
                    rows.append({"item_id": f"nct:{cls}-{i}", "set": "nct", "case_id": f"{cls}-{i}", "label": cls})
                    emb.append(_unit(base + 0.6 * self.rng.standard_normal(D)))
            self.write_index("conch", "nct", rows, np.asarray(emb))
        if quilt:
            self.quilt()
        return {"cases": cases, "tiles": tiles, "bank": bank}

    def quilt(self, *, concepts: tuple[str, ...] | None = None, n: int = 40, n_groups: int = 8,
              separable: bool = True) -> None:
        rows, emb = [], []
        for cid in concepts or self.ids:
            for polarity, sign in (("pos", 1.0), ("neg", -1.0)):
                for i in range(n):
                    rows.append({"item_id": f"quilt:{cid}-{polarity}-{i}", "set": "quilt",
                                 "case_id": f"q-{cid}-{polarity}-{i}", "label": f"{cid}:{polarity}",
                                 "group": f"video{i % n_groups}"})
                    signal = sign * self.u[cid] if separable else 0.0
                    emb.append(_unit(signal + 0.8 * self.rng.standard_normal(D)))
        self.write_index("conch", "quilt", rows, np.asarray(emb))


def run(module: Any, fake: Fake, *extra: str) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = module.main(["--config", str(fake.settings_path), "--data-root", str(fake.root),
                        "--artifacts-root", str(fake.root / "artifacts"), *extra], log=lines.append)
    return code, lines
