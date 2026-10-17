"""Shared risk-model code for 05_risk_model, 05b_abmil, 06_reliability, 07_retrieval and eval/deletion_test (Agent 3).

Feature assembly (case level, WSI-free, §A9):
  * Embedding candidates: mean of a case's QC-passing tile embeddings (tiles.csv qc_pass; a case with no passing
    tile falls back to all its tiles and is listed in `qc_fallback`). Clean reads use the per-item TTA means;
    train augmentation uses the `_views` files (one pooled row per (case, view), training folds only, §A7); clean
    view-0 reads (ruling M9) come from the `_views` files at view 0. Degraded items (deg512/deg256) are pooled per
    variant_case_id with the SOURCE case's QC mask; composites (comp512/comp256) pool all tiles.
  * Concept bottlenecks: case_features.csv concept columns (core.schemas.concept_feature_columns), cell features
    from cell_features.csv (value + __missing indicator; NaN imputed with the training-fold median).

Models: LinearModel (standardization + NaN-median imputation + multinomial L2 logistic regression, fit inside each
training fold) and ABMIL (gated-attention MIL, torch, CPU, seeded). Both export/import plain npz weights; sklearn
objects are never pickled (§A22).

Selection (§A3): patient-grouped K-fold CV inside train repeated R times (settings split.cv; folds from
pipeline.tile_probe.case_folds with seed split.cv.seed + repeat). Score = mean held-out log-loss over the K x R
folds, each fold temperature-scaled with a T fit on the other folds' out-of-fold logits (so every candidate,
including the late fusion, is scored calibrated and out-of-sample); SE = SD of those fold log-losses / sqrt(K) (repeats reduce noise but are not independent folds). One-SE
rule: among configs within one SE of the best, the simplest wins (settings risk_model.complexity order, then the
smaller C). The A-vs-B rule (plan §6.6, settings model_choice.auroc_margin / bacc_margin) compares the chosen
black box A with the chosen concept bottleneck B on repeat-averaged out-of-fold predictions.

Laptop code (Python 3.13, uv): numpy, pandas, scikit-learn, torch (ABMIL only, imported lazily).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core import calibration, io, schemas, stats
from pipeline.tile_probe import case_folds

TILE_KEY = ["case_id", "grid", "tile_idx"]
DEG_SETS = {"bach512": "deg512", "bach256": "deg256"}
COMP_SETS = {"bach512": "comp512", "bach256": "comp256"}


# ---- reading --------------------------------------------------------------------------------------------------------------
def read_csv(path: str | Path) -> pd.DataFrame:
    """CSV with ID-like columns kept as strings ('' stays missing)."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False,
                       dtype={"case_id": str, "patient_id": str, "group": str, "label": str, "split": str,
                              "item_id": str, "degrade_type": str, "source_case_id": str})


def read_cases(root: str | Path) -> pd.DataFrame:
    cases = read_csv(schemas.cases_csv(root))
    schemas.validate_frame(schemas.CASES, cases)
    return cases.sort_values("case_id").reset_index(drop=True)


def read_tiles(root: str | Path) -> pd.DataFrame:
    tiles = read_csv(schemas.tiles_csv(root))
    schemas.validate_frame(schemas.TILES, tiles)
    tiles["qc_pass"] = tiles["qc_pass"].astype(str).str.lower().isin(["1", "true", "1.0"])
    return tiles


def emb_exists(root: str | Path, encoder: str, set_name: str, head: str, *, views: bool = False) -> bool:
    return (schemas.emb_npy(root, encoder, set_name, head, views=views).is_file()
            and schemas.emb_index_csv(root, encoder, set_name, head, views=views).is_file())


def load_emb(root: str | Path, encoder: str, set_name: str, head: str, *, views: bool = False
             ) -> tuple[np.ndarray, pd.DataFrame]:
    x = np.load(schemas.emb_npy(root, encoder, set_name, head, views=views)).astype(np.float32)
    index = read_csv(schemas.emb_index_csv(root, encoder, set_name, head, views=views))
    schemas.validate_frame(schemas.EMB_INDEX, index)
    if x.ndim != 2 or len(index) != x.shape[0]:
        raise ValueError(f"{encoder}/{set_name}_{head}{'_views' if views else ''}: {x.shape} rows vs {len(index)} index rows")
    if not (index["emb_row"].to_numpy() == np.arange(len(index))).all():
        raise ValueError(f"{encoder}/{set_name}_{head}: emb_row is not 0..N-1 in order")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{encoder}/{set_name}_{head}: non-finite embeddings")
    return x, index


def read_ids(index: pd.DataFrame) -> np.ndarray:
    """Vectorized read_id over an EMB_INDEX frame (variant_case_id for degraded rows, else case_id)."""
    case = index["case_id"].astype(str)
    dtype = index["degrade_type"].fillna("").astype(str)
    sev = index["degrade_severity"].fillna(0).astype(int).astype(str)
    bad = sorted(set(dtype[dtype != ""]) - set(schemas.DEGRADE_TYPES))
    if bad:
        raise ValueError(f"unknown degrade types {bad}")
    return np.where(dtype == "", case, case + "_FSIM_" + dtype + sev).astype(object)


def read_id(row_case: Any, dtype: Any, sev: Any) -> str:
    if isinstance(dtype, str) and dtype:
        return schemas.variant_case_id(str(row_case), dtype, int(sev))
    return str(row_case)


def jsonable(obj: Any) -> Any:
    """Copy of obj with non-finite floats as None (JSON has no NaN; core.io.write_json refuses it) and numpy
    scalars as Python numbers. An undefined metric is reported as null, never dropped."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


# ---- pooling ----------------------------------------------------------------------------------------------------------------
@dataclass
class Pooled:
    """Pooled case-level rows: keys has read_id, source_case_id, variant, view; x is aligned (n, D)."""

    keys: pd.DataFrame
    x: np.ndarray
    qc_fallback: list[str] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.x, index=pd.Index(self.keys["read_id"], name="case_id"))


def pool_tiles(x: np.ndarray, index: pd.DataFrame, qc: pd.DataFrame | None, *, by_view: bool = False,
               composite: bool = False) -> Pooled:
    """Mean of QC-passing tile rows per read (variant_case_id for degraded rows), optionally per view.

    qc: tiles.csv rows (case_id, grid, tile_idx, qc_pass); None or composite=True keeps every tile. Reads whose
    tiles all fail QC fall back to every tile (recorded in qc_fallback).
    """
    idx = index[["case_id", "grid", "tile_idx", "degrade_type", "degrade_severity", "view"]].copy()
    idx["degrade_type"] = idx["degrade_type"].fillna("").astype(str)
    idx["read_id"] = read_ids(idx)
    if qc is not None and not composite:
        q = qc[TILE_KEY + ["qc_pass"]].copy()
        q["tile_idx"] = q["tile_idx"].astype(int)
        idx["tile_idx"] = idx["tile_idx"].astype(int)
        idx = idx.merge(q, on=TILE_KEY, how="left", validate="many_to_one")
        if idx["qc_pass"].isna().any():
            missing = idx.loc[idx["qc_pass"].isna(), "case_id"].unique()[:5]
            raise ValueError(f"embedding tiles missing from tiles.csv, e.g. cases {list(missing)}")
        keep = idx["qc_pass"].astype(bool).to_numpy()
    else:
        keep = np.ones(len(idx), dtype=bool)
    gkey = (idx["read_id"].astype(str) + "|" + idx["view"].astype(int).map("{:02d}".format)) if by_view else idx["read_id"]
    codes, uniques = pd.factorize(gkey, sort=True)
    has_keep = np.zeros(len(uniques), dtype=bool)
    np.logical_or.at(has_keep, codes, keep)
    use = keep | ~has_keep[codes]
    fallback = sorted(set(idx.loc[~has_keep[codes], "read_id"]))
    sums = np.zeros((len(uniques), x.shape[1]), dtype=np.float64)
    counts = np.zeros(len(uniques))
    np.add.at(sums, codes[use], x[use])
    np.add.at(counts, codes[use], 1)
    first = pd.DataFrame({"code": codes, "read_id": idx["read_id"], "source_case_id": idx["case_id"].astype(str),
                          "variant": [f"{d}{int(s)}" if d else "clean" for d, s in
                                      zip(idx["degrade_type"], idx["degrade_severity"].fillna(0))],
                          "view": idx["view"].astype(int)}).drop_duplicates("code").sort_values("code")
    if composite:
        first["variant"] = "composite"
    keys = first.drop(columns="code").reset_index(drop=True)
    return Pooled(keys, (sums / counts[:, None]).astype(np.float32), fallback)


# ---- feature sets -----------------------------------------------------------------------------------------------------------
@dataclass
class FeatureSet:
    """Case-level features for one candidate. clean: TTA reads of every case (index = case_id); aug: training
    augmentation rows (index = case_id, repeated per view) or None; variants: degraded/composite reads (index =
    variant read id); view0: clean view-0 reads (ruling M9) or None. names: feature names."""

    name: str
    kind: str                       # embedding | concept
    clean: pd.DataFrame
    names: list[str]
    aug: pd.DataFrame | None = None
    variants: pd.DataFrame | None = None
    view0: pd.DataFrame | None = None
    source: dict[str, Any] = field(default_factory=dict)
    qc_fallback: list[str] = field(default_factory=list)


def embedding_features(root: str | Path, name: str, spec: Mapping[str, str], tiles: pd.DataFrame, *,
                       use_views: bool) -> FeatureSet | None:
    """Mean-pooled QC tile embeddings of one encoder/set/head (None when the clean set is missing)."""
    enc, set_name, head = spec["encoder"], spec["set"], spec["head"]
    if not emb_exists(root, enc, set_name, head):
        return None
    x, index = load_emb(root, enc, set_name, head)
    clean = pool_tiles(x, index, tiles)
    names = [f"{enc}_{head}_{j}" for j in range(x.shape[1])]
    fs = FeatureSet(name, "embedding", _named(clean.frame(), names), names,
                    source={"encoder": enc, "set": set_name, "head": head}, qc_fallback=clean.qc_fallback)
    if emb_exists(root, enc, set_name, head, views=True):
        xv, iv = load_emb(root, enc, set_name, head, views=True)
        views = pool_tiles(xv, iv, tiles, by_view=True)
        frame = views.frame()
        if use_views:
            fs.aug = _named(frame, names)
        v0 = views.keys["view"].to_numpy() == 0
        fs.view0 = _named(pd.DataFrame(views.x[v0], index=pd.Index(views.keys.loc[v0, "read_id"], name="case_id")),
                          names)
    parts = []
    deg = DEG_SETS.get(set_name)
    if deg and emb_exists(root, enc, deg, head):
        xd, idd = load_emb(root, enc, deg, head)
        parts.append(pool_tiles(xd, idd, tiles).frame())
    comp = COMP_SETS.get(set_name)
    if comp and emb_exists(root, enc, comp, head):
        xc, ic = load_emb(root, enc, comp, head)
        parts.append(pool_tiles(xc, ic, None, composite=True).frame())
    if parts:
        fs.variants = _named(pd.concat(parts), names)
    return fs


def _named(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = list(names)
    return frame


CASE_FEATURES_VIEW0 = "case_features_view0.csv"   # optional (requested from Agent 2): clean view-0 concept features


def concept_frame(root: str | Path, cfg: Mapping[str, Any], *, view0: bool = False) -> pd.DataFrame | None:
    """case_features.csv indexed by case_id (None when absent); view0=True reads the optional clean view-0 file."""
    path = Path(root) / CASE_FEATURES_VIEW0 if view0 else schemas.case_features_csv(root)
    if not path.is_file():
        return None
    df = read_csv(path)
    schemas.validate_frame(schemas.CASE_FEATURES, df, extra_required=schemas.concept_feature_columns(cfg))
    return df.set_index("case_id")


def cell_frame(root: str | Path, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str]] | None:
    path = schemas.cell_features_csv(root)
    if not bool(cfg["cells"]["enabled"]) or not path.is_file():
        return None
    df = read_csv(path)
    schemas.validate_frame(schemas.CELL_FEATURES, df)
    feats = [c[: -len("__missing")] for c in df.columns if c.endswith("__missing") and c[: -len("__missing")] in df]
    cols = [col for f in feats for col in (f, f"{f}__missing")]
    return df.set_index("case_id")[cols].astype(float), feats


def concept_features(name: str, concepts: pd.DataFrame | None, cells: tuple[pd.DataFrame, list[str]] | None,
                     cfg: Mapping[str, Any], cases: pd.DataFrame, concepts_view0: pd.DataFrame | None = None
                     ) -> FeatureSet | None:
    """cb_text (B), cb_text_cells (B+C) or cb_cells (C); None when an input is missing."""
    use_text = name in ("cb_text", "cb_text_cells")
    use_cells = name in ("cb_cells", "cb_text_cells")
    if (use_text and concepts is None) or (use_cells and cells is None):
        return None
    frames, names = [], []
    if use_text:
        cols = schemas.concept_feature_columns(cfg)
        frames.append(concepts[cols].astype(float))
        names += cols
    if use_cells:
        cf, _ = cells
        frames.append(cf)
        names += list(cf.columns)
    joined = pd.concat(frames, axis=1, join="outer") if len(frames) > 1 else frames[0]
    real = set(cases["case_id"])
    clean = joined.loc[[i for i in joined.index if i in real]]
    missing = sorted(real - set(clean.index))
    if missing and not use_cells:
        raise ValueError(f"case_features.csv lacks {len(missing)} cases, e.g. {missing[:3]}")
    if missing:                                          # cells absent for a case: every cell feature missing
        clean = pd.concat([clean, pd.DataFrame(np.nan, index=pd.Index(missing, name="case_id"), columns=names)])
        for col in [c for c in names if c.endswith("__missing")]:
            clean.loc[missing, col] = 1.0
    variants = joined.loc[[i for i in joined.index if i not in real]] if not use_cells else None
    if variants is not None and variants.empty:
        variants = None
    view0 = None
    if use_text and concepts_view0 is not None:
        v = concepts_view0[schemas.concept_feature_columns(cfg)].astype(float)
        if use_cells:
            v = v.join(cells[0], how="left")
        view0 = v[names]
    return FeatureSet(name, "concept", clean[names], names, variants=variants, view0=view0,
                      source={"case_features": use_text, "cell_features": use_cells})


# ---- linear model -----------------------------------------------------------------------------------------------------------
class LinearModel:
    """NaN-median imputation + standardization (fit on training rows) + multinomial L2 logistic regression."""

    def __init__(self, classes: Sequence[str], C: float, max_iter: int, class_weight: str | None):
        self.classes = list(classes)
        self.C, self.max_iter, self.class_weight = float(C), int(max_iter), class_weight
        self.median = self.mean = self.scale = self.coef = self.intercept = None
        self.names: list[str] = []

    def fit(self, x: np.ndarray, y: np.ndarray, names: Sequence[str] = ()) -> "LinearModel":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        present = sorted(set(y.tolist()))
        if present != list(range(len(self.classes))):
            raise ValueError(f"training rows must contain every class; got class indices {present}")
        med = np.nanmedian(np.where(np.isnan(x).all(0, keepdims=True), 0.0, x), axis=0)
        self.median = np.where(np.isnan(med), 0.0, med)
        xi = np.where(np.isnan(x), self.median, x)
        self.mean = xi.mean(0)
        sd = xi.std(0)
        self.scale = np.where(sd > 1e-12, sd, 1.0)
        clf = LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight=self.class_weight)
        clf.fit((xi - self.mean) / self.scale, y)
        self.coef, self.intercept = clf.coef_.copy(), clf.intercept_.copy()
        self.names = list(names)
        return self

    def standardize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (np.where(np.isnan(x), self.median, x) - self.mean) / self.scale

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.standardize(x) @ self.coef.T + self.intercept

    def to_npz(self, path: str | Path, **meta: Any) -> Path:
        return io.save_npz(path, kind=np.asarray("linear"), classes=np.asarray(self.classes), C=np.asarray(self.C),
                           median=self.median, mean=self.mean, scale=self.scale, coef=self.coef,
                           intercept=self.intercept, names=np.asarray(self.names, dtype=str),
                           **{k: np.asarray(v) for k, v in meta.items()})

    @classmethod
    def from_npz(cls, path: str | Path) -> "LinearModel":
        z = io.load_npz(path)
        if str(z["kind"]) != "linear":
            raise ValueError(f"{path} is not a linear model export")
        m = cls(list(z["classes"]), float(z["C"]), 1, None)
        m.median, m.mean, m.scale = z["median"], z["mean"], z["scale"]
        m.coef, m.intercept, m.names = z["coef"], z["intercept"], list(z["names"])
        return m


def centered_contributions(model: LinearModel, x: np.ndarray, aggressive: int) -> np.ndarray:
    """(W_aggressive - mean_k W_k) * x_std per feature: signed push toward (+) / away from (-) the aggressive class."""
    w = model.coef[aggressive] - model.coef.mean(axis=0)
    return model.standardize(x) * w[None, :]


# ---- ABMIL ------------------------------------------------------------------------------------------------------------------
class ABMIL:
    """Gated-attention MIL (Ilse et al., ICML 2018) over a case's tile embeddings; CPU, seeded, npz export."""

    def __init__(self, classes: Sequence[str], params: Mapping[str, Any]):
        self.classes = list(classes)
        self.params = {k: params[k] for k in ("hidden", "attn_dim", "dropout", "epochs", "lr", "weight_decay", "seed")}
        self.state: dict[str, np.ndarray] | None = None
        self.mean = self.scale = None

    def _net(self, dim: int):
        import torch
        from torch import nn

        p = self.params
        n_classes = len(self.classes)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Sequential(nn.Linear(dim, p["hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]))
                self.att_v = nn.Linear(p["hidden"], p["attn_dim"])
                self.att_u = nn.Linear(p["hidden"], p["attn_dim"])
                self.att_w = nn.Linear(p["attn_dim"], 1)
                self.head = nn.Linear(p["hidden"], n_classes)

            def forward(self, x: "torch.Tensor", mask: "torch.Tensor"):
                h = self.embed(x)                                           # (B, T, H)
                a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
                a = a.masked_fill(~mask, float("-inf"))
                att = torch.softmax(a, dim=1)
                z = (att.unsqueeze(-1) * h).sum(1)
                return self.head(z), att

        return Net()

    @staticmethod
    def _pad(bags: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        t = max(len(b) for b in bags)
        x = np.zeros((len(bags), t, bags[0].shape[1]), dtype=np.float32)
        mask = np.zeros((len(bags), t), dtype=bool)
        for i, b in enumerate(bags):
            x[i, : len(b)] = b
            mask[i, : len(b)] = True
        return x, mask

    def fit(self, bags: Sequence[np.ndarray], y: np.ndarray) -> "ABMIL":
        import torch

        if any(len(b) == 0 for b in bags):
            raise ValueError("ABMIL bags must contain at least one tile")
        threads = torch.get_num_threads()
        torch.set_num_threads(1)                  # tiny nets: one thread is faster and fully deterministic
        try:
            return self._fit(bags, y)
        finally:
            torch.set_num_threads(threads)

    def _fit(self, bags: Sequence[np.ndarray], y: np.ndarray) -> "ABMIL":
        import torch

        torch.manual_seed(int(self.params["seed"]))
        rng = np.random.default_rng(int(self.params["seed"]))
        allx = np.concatenate(bags)
        self.mean = allx.mean(0)
        sd = allx.std(0)
        self.scale = np.where(sd > 1e-12, sd, 1.0).astype(np.float32)
        x, mask = self._pad([(b - self.mean) / self.scale for b in bags])
        net = self._net(x.shape[2])
        opt = torch.optim.Adam(net.parameters(), lr=float(self.params["lr"]),
                               weight_decay=float(self.params["weight_decay"]))
        xt, mt, yt = torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(np.asarray(y, dtype=np.int64))
        batch = 32
        net.train()
        for _ in range(int(self.params["epochs"])):
            order = rng.permutation(len(x))
            for s in range(0, len(order), batch):
                b = torch.from_numpy(order[s:s + batch])
                logits, _ = net(xt[b], mt[b])
                loss = torch.nn.functional.cross_entropy(logits, yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.state = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        return self

    def predict(self, bags: Sequence[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        """(logits (n, C), per-bag attention arrays summing to 1)."""
        import torch

        x, mask = self._pad([(b - self.mean) / self.scale for b in bags])
        net = self._net(x.shape[2])
        net.load_state_dict({k: torch.from_numpy(v) for k, v in self.state.items()})
        net.eval()
        with torch.no_grad():
            logits, att = net(torch.from_numpy(x), torch.from_numpy(mask))
        att = att.numpy()
        return logits.numpy().astype(np.float64), [att[i, : len(b)] for i, b in enumerate(bags)]

    def to_npz(self, path: str | Path, **meta: Any) -> Path:
        arrays = {f"w__{k}": v for k, v in self.state.items()}
        return io.save_npz(path, kind=np.asarray("abmil"), classes=np.asarray(self.classes), mean=self.mean,
                           scale=self.scale, params=np.asarray(repr(sorted(self.params.items()))), **arrays,
                           **{k: np.asarray(v) for k, v in meta.items()})

    @classmethod
    def from_npz(cls, path: str | Path, params: Mapping[str, Any]) -> "ABMIL":
        z = io.load_npz(path)
        if str(z["kind"]) != "abmil":
            raise ValueError(f"{path} is not an ABMIL export")
        m = cls(list(z["classes"]), params)
        m.mean, m.scale = z["mean"], z["scale"]
        m.state = {k[3:]: v for k, v in z.items() if k.startswith("w__")}
        return m


def tile_bags(x: np.ndarray, index: pd.DataFrame, qc: pd.DataFrame | None, *, by_view: bool = False,
              composite: bool = False) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Per read (and view): (read ids, bags of QC tile embeddings, matching tile_idx arrays). QC-less reads keep
    every tile, as in pool_tiles."""
    idx = index[["case_id", "grid", "tile_idx", "degrade_type", "degrade_severity", "view"]].copy()
    idx["degrade_type"] = idx["degrade_type"].fillna("").astype(str)
    idx["read_id"] = read_ids(idx)
    idx["tile_idx"] = idx["tile_idx"].astype(int)
    if qc is not None and not composite:
        q = qc[TILE_KEY + ["qc_pass"]].copy()
        q["tile_idx"] = q["tile_idx"].astype(int)
        idx = idx.merge(q, on=TILE_KEY, how="left", validate="many_to_one")
        idx["qc_pass"] = idx["qc_pass"].fillna(False).astype(bool)
    else:
        idx["qc_pass"] = True
    idx["_row"] = np.arange(len(idx))
    keys = ["read_id", "view"] if by_view else ["read_id"]
    ids, bags, tiles = [], [], []
    for key, part in idx.groupby(keys, sort=True):
        use = part[part["qc_pass"]] if part["qc_pass"].any() else part
        ids.append(key[0] if isinstance(key, tuple) else key)
        bags.append(x[use["_row"].to_numpy()])
        tiles.append(use["tile_idx"].to_numpy())
    return ids, bags, tiles


# ---- CV and selection -----------------------------------------------------------------------------------------------------
@dataclass
class CVResult:
    name: str
    family: str                  # black_box | concept
    config: dict[str, Any]
    oof_logits: np.ndarray       # (R, n_train, C)
    fold_losses: list[float]
    complexity: int
    n_features: int

    @property
    def mean_loss(self) -> float:
        return float(np.mean(self.fold_losses))

    def se(self, n_folds: int) -> float:
        return float(np.std(self.fold_losses, ddof=1) / math.sqrt(n_folds)) if len(self.fold_losses) > 1 else 0.0

    def oof_probs(self) -> np.ndarray:
        """Repeat-averaged out-of-fold probabilities (n_train, C)."""
        return np.mean([calibration.softmax(l, 1.0) for l in self.oof_logits], axis=0)


def cv_splits(train_cases: pd.DataFrame, cfg: Mapping[str, Any]) -> list[list[np.ndarray]]:
    """For each repeat r: a list of K held-out position arrays over train_cases (sorted by case_id)."""
    cv = cfg["split"]["cv"]
    train_cases = train_cases.sort_values("case_id").reset_index(drop=True)
    out = []
    for r in range(int(cv["repeats"])):
        folds = case_folds(train_cases, int(cv["folds"]), int(cv["seed"]) + r)
        f = train_cases["case_id"].map(folds).to_numpy()
        out.append([np.flatnonzero(f == k) for k in range(int(cv["folds"]))])
    return out


def fold_calibrated_losses(oof: np.ndarray, labels: np.ndarray, splits: list[list[np.ndarray]],
                           bounds: Sequence[float] | None) -> list[float]:
    """Held-out log-loss per (repeat, fold). With bounds, each fold's logits are first temperature-scaled with a T fit
    on the OTHER folds' out-of-fold logits of the same repeat (out-of-sample, like production's temperature); without
    bounds, T = 1."""
    losses = []
    n = oof.shape[1]
    for r, folds in enumerate(splits):
        for held in folds:
            t = 1.0
            if bounds is not None:
                rest = np.setdiff1d(np.arange(n), held)
                t = calibration.fit_temperature(oof[r, rest], labels[rest], bounds)
            losses.append(calibration.nll(calibration.softmax(oof[r, held], t), labels[held]))
    return losses


def run_cv(name: str, family: str, config: Mapping[str, Any], n_train: int, labels: np.ndarray,
           splits: list[list[np.ndarray]], fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
           complexity: int, n_features: int, n_classes: int, bounds: Sequence[float] | None = None) -> CVResult:
    """fit_predict(train_pos, held_pos) -> held-out logits. Returns OOF logits for every repeat and the fold losses
    (fold-calibrated when temperature bounds are given; see fold_calibrated_losses)."""
    oof = np.full((len(splits), n_train, n_classes), np.nan)
    for r, folds in enumerate(splits):
        for held in folds:
            fit = np.setdiff1d(np.arange(n_train), held)
            oof[r, held] = fit_predict(fit, held)
    if np.isnan(oof).any():
        raise RuntimeError(f"{name}: some train cases were never held out")
    return CVResult(name, family, dict(config), oof, fold_calibrated_losses(oof, labels, splits, bounds),
                    complexity, n_features)


def one_se_choice(results: Sequence[CVResult], n_folds: int) -> tuple[CVResult, dict[str, Any]]:
    """Simplest config within one SE of the best mean log-loss (complexity rank, then smaller C)."""
    if not results:
        raise ValueError("no CV results to choose from")
    best = min(results, key=lambda r: r.mean_loss)
    limit = best.mean_loss + best.se(n_folds)
    eligible = [r for r in results if r.mean_loss <= limit + 1e-12]
    chosen = min(eligible, key=lambda r: (r.complexity, float(r.config.get("C", 0.0)), r.mean_loss))
    return chosen, {"best": _cfg_label(best), "best_loss": best.mean_loss, "best_se": best.se(n_folds),
                    "limit": limit, "eligible": [_cfg_label(r) for r in eligible], "chosen": _cfg_label(chosen),
                    "rule": "one_se: simplest (settings risk_model.complexity, then smaller C) within 1 SE of the best"}


def _cfg_label(r: CVResult) -> str:
    c = r.config.get("C")
    return f"{r.name}" + (f"@C={c:g}" if c is not None else "")


def cv_metrics(result: CVResult, labels: np.ndarray, groups: np.ndarray, cfg: Mapping[str, Any],
               aggressive: int) -> dict[str, Any]:
    """Repeat-averaged OOF metrics with cluster-bootstrap CIs over train groups."""
    probs = result.oof_probs()
    k = probs.shape[1]
    boot = cfg["bootstrap"]
    y_agg = (labels == aggressive).astype(int)

    def ci(metric: Callable[..., float], arrays: Sequence[np.ndarray]) -> dict[str, Any]:
        return stats.bootstrap_ci(metric, arrays, groups, None, int(boot["n"]), float(boot["ci"]), int(boot["seed"]))

    return {
        "log_loss": ci(lambda y, p: calibration.nll(p, y), [labels, probs]),
        "auroc_invasive": ci(stats.auroc, [y_agg, probs[:, aggressive]]),
        "balanced_accuracy": ci(lambda y, p: stats.balanced_accuracy(y, p.argmax(1), k), [labels, probs]),
        "macro_f1": ci(lambda y, p: stats.macro_f1(y, p.argmax(1), k), [labels, probs]),
        "macro_auroc": ci(lambda y, p: stats.macro_auroc(y, p, k), [labels, probs]),
    }


def a_vs_b(metrics_a: Mapping[str, Any], metrics_b: Mapping[str, Any], auroc_margin: float, bacc_margin: float
           ) -> tuple[str, dict[str, Any]]:
    """Plan §6.6: B drives iff AUROC(A) - AUROC(B) <= auroc_margin and BAcc(A) - BAcc(B) <= bacc_margin."""
    d_auc = metrics_a["auroc_invasive"]["estimate"] - metrics_b["auroc_invasive"]["estimate"]
    d_bacc = metrics_a["balanced_accuracy"]["estimate"] - metrics_b["balanced_accuracy"]["estimate"]
    b_drives = d_auc <= auroc_margin and d_bacc <= bacc_margin
    return ("B" if b_drives else "A"), {"auroc_gap": float(d_auc), "bacc_gap": float(d_bacc),
                                        "auroc_margin": float(auroc_margin), "bacc_margin": float(bacc_margin),
                                        "b_drives": bool(b_drives)}
