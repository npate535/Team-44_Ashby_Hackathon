"""Deletion test: are the explanations faithful to the model? (plan §12, BUILD_PLAN Task 11, CELLS_SPEC §8.5; Agent 3)

    python -m eval.deletion_test [--data-root D] [--artifacts-root A] [--limit N] [--force]

For the production model and Model A (eval/model_choice.json), on clean TEST cases:
  * concept-ranked: remove the case's top-3 QC tiles for its top concept (the concept with the largest positive
    summed contribution in contributions.csv, ranked by that concept's tile z in concepts_tile.npz);
  * risk-ranked: remove its top-3 tiles by the production heatmap (tile_risk.csv);
  * contribution-ranked (mean-pooled linear models only): remove its top-3 regions by exact contribution
    (pipeline/05d_tile_contrib.py: the region's additive term of the model's centered calibrated logit);
  * random: remove 3 random QC tiles, deletion.n_random times (seeded), averaged.
Every removal is re-aggregated IN DISTRIBUTION through the model's own feature path (pipeline.riskmodels): pooled
embeddings are recomputed from the remaining tiles (a removed 512-px tile removes its four 256-px sub-tiles for
224-native encoders), ABMIL bags lose those tiles, concept features c__top<k> / c__frac are recomputed from the
remaining tiles' z. Risk = calibrated P(aggressive class) with the model's temperature; drop = full - removed (the full
read uses the same recomputation path). Every drop is also reported in logit units: the drop in the centered calibrated
logit (logit_aggressive - mean_k logit_k) / T, computed from the re-pooled logits (never from log of a probability, so
it stays finite when P is ~1). Paired per-case cluster bootstrap (patient / pseudo-patient groups) of the concept-,
risk- and contribution-vs-random drop differences in both units. Headline subset: cases the model calls Invasive; all eligible
test cases are reported too. When a concept bottleneck is tested, the concept-ranked removal partly defines its own
top-k features (BUILD_PLAN: "not on B") — this is stated next to its numbers.
Cells version (cells enabled and a cell-feature model is production or B): set the case's top cell feature to the
benign-reference median (train Normal/Benign) vs a random other cell feature (exact average over the others).
Writes eval/deletion.json. Whatever comes out is reported, including a null result.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

import importlib

from core import calibration, cli, io, schemas, stats
from pipeline import riskmodels as rm

tc = importlib.import_module("pipeline.05d_tile_contrib")

NAME = "deletion"
ARMS = ("concept", "risk", "contribution")


class Reader:
    """Re-aggregating risk function risk(case_id, removed g512 tile set) for one exported model."""

    def __init__(self, name: str, ctx: cli.StepContext, choice: Mapping[str, Any], tiles: pd.DataFrame,
                 case_ids: Sequence[str]):
        cfg, root = ctx.cfg, ctx.data_root
        self.name, self.cfg = name, cfg
        self.classes = list(cfg["classes"])
        self.agg = self.classes.index(cfg["aggressive_class"])
        self.temperature = float(choice["temperatures"][name])
        self.models_dir = root / cfg["risk_model"]["models_dir"]
        self.g512 = cfg["tiles"]["grids"]["g512"]
        self.g256 = cfg["tiles"]["grids"]["g256"]
        self.kind = self._kind(name, choice)
        wanted = set(case_ids)
        if self.kind == "lr_embedding":
            self._load_embedding(root, cfg["risk_model"]["embedding_candidates"][name], tiles, wanted)
            self.model = rm.LinearModel.from_npz(self.models_dir / f"{name}.npz")
        elif self.kind == "abmil":
            z = io.load_npz(self.models_dir / "abmil.npz")
            self._load_embedding(root, {"encoder": str(z["encoder"]), "set": str(z["set"]), "head": str(z["head"])},
                                 tiles, wanted)
            self.model = rm.ABMIL.from_npz(self.models_dir / "abmil.npz", cfg["abmil"])
        elif self.kind == "fusion":
            comps = next(r["config"]["components"] for r in choice["candidates"] if r["model"] == name)
            self.components = [Reader(c, ctx, choice, tiles, case_ids) for c in comps]
        else:
            self._load_concepts(root, tiles, wanted)
            self.model = rm.LinearModel.from_npz(self.models_dir / f"{name}.npz")

    @staticmethod
    def _kind(name: str, choice: Mapping[str, Any]) -> str:
        if name == "abmil":
            return "abmil"
        if name.startswith("fusion"):
            return "fusion"
        if name in ("cb_text", "cb_text_cells", "cb_cells"):
            return "concept"
        return "lr_embedding"

    def _load_embedding(self, root, spec: Mapping[str, str], tiles: pd.DataFrame, wanted: set[str]) -> None:
        x, idx = rm.load_emb(root, spec["encoder"], spec["set"], spec["head"])
        selected = idx.case_id.isin(wanted).to_numpy()
        idx = idx.loc[selected].reset_index(drop=True)
        if idx.empty:
            raise ValueError("no embedding rows available for the requested deletion cases")
        self.grid = str(idx.grid.iloc[0])
        # Use the production QC rule, including its all-tiles fallback when no tile passes.
        ids, bags, tile_ids = rm.tile_bags(x[selected], idx, tiles)
        self.tiles = {cid: (t, bag) for cid, t, bag in zip(ids, tile_ids, bags)}

    def _load_concepts(self, root, tiles: pd.DataFrame, wanted: set[str]) -> None:
        cfg = self.cfg
        names = list(self.model_names())
        self.use_text = any("__top" in n or n.endswith("__frac") for n in names)
        self.cells = None
        if any(not ("__top" in n or n.endswith("__frac")) for n in names):
            cf = rm.cell_frame(root, cfg)
            self.cells = cf[0] if cf else None
        self.feature_names = names
        if self.use_text:
            z = io.load_npz(schemas.concepts_tile_npz(root))
            self.concept_ids = [str(c) for c in z["concept_ids"]]
            case = np.asarray(z["case_id"]).astype(str)
            self.tile_z: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for cid in wanted:
                m = case == cid
                self.tile_z[cid] = (np.asarray(z["tile_idx"])[m].astype(int), np.asarray(z["z"])[m].astype(float))

    def model_names(self) -> list[str]:
        return list(io.load_npz(self.models_dir / f"{self.name}.npz")["names"])

    def qc_tiles(self, cid: str) -> np.ndarray:
        """g512 tile indices available for removal (the model's own tile list, mapped to g512)."""
        if self.kind == "fusion":
            return self.components[0].qc_tiles(cid)
        if self.kind == "concept":
            if self.use_text:
                return self.tile_z[cid][0]
            raise ValueError("cell-only model has no tiles to remove")
        t, _ = self.tiles[cid]
        return np.unique(self._to512(t))

    def _to512(self, t: np.ndarray) -> np.ndarray:
        if self.grid == "g512":
            return t
        r, c = np.divmod(t, self.g256["cols"])
        return (r // 2) * self.g512["cols"] + (c // 2)

    def risk(self, cid: str, removed: set[int]) -> float:
        return self.read(cid, removed)[0]

    def read(self, cid: str, removed: set[int]) -> tuple[float, float]:
        """(calibrated P(aggressive), centered calibrated logit (logit_agg - mean_k logit_k) / T) from one logit read."""
        lg = self.logits(cid, removed)
        risk = float(calibration.softmax(lg[None, :], self.temperature)[0, self.agg])
        return risk, float((lg[self.agg] - lg.mean()) / self.temperature)

    def contribution_ranking(self, cid: str) -> list[int]:
        """g512 regions by exact contribution (largest first, ties by index); mean-pooled linear models only."""
        t, x = self.tiles[cid]
        terms, _ = tc.read_contributions(self.model, x, self._to512(t), self.temperature, self.agg)
        return sorted((int(i) for i in terms.index), key=lambda i: (-terms[i], i))

    def logits(self, cid: str, removed: set[int]) -> np.ndarray:
        """Uncalibrated logits (fusion: log of the mean of the components' calibrated probabilities, as in 05)."""
        if self.kind == "fusion":
            probs = np.mean([calibration.softmax(c.logits(cid, removed)[None, :], c.temperature)[0]
                             for c in self.components], axis=0)
            return np.log(np.clip(probs, 1e-12, 1.0))
        if self.kind in ("lr_embedding", "abmil"):
            t, x = self.tiles[cid]
            keep = ~np.isin(self._to512(t), list(removed))
            if not keep.any():
                raise ValueError(f"{cid}: every tile removed")
            if self.kind == "abmil":
                return self.model.predict([x[keep]])[0][0]
            return self.model.logits(x[keep].mean(0, keepdims=True))[0]
        return self.model.logits(self.concept_vector(cid, removed)[None, :])[0]

    def concept_vector(self, cid: str, removed: set[int], override: Mapping[str, float] | None = None) -> np.ndarray:
        cfg = self.cfg
        k = int(cfg["concepts"]["top_k"])
        thr = float(cfg["concepts"]["z_threshold"])
        vals: dict[str, float] = {}
        if self.use_text:
            t, z = self.tile_z[cid]
            keep = ~np.isin(t, list(removed))
            zk = z[keep]
            for j, c in enumerate(self.concept_ids):
                col = np.sort(zk[:, j])[::-1]
                vals[f"{c}__top{k}"] = float(col[:k].mean())
                vals[f"{c}__frac"] = float((col > thr).mean())
        if self.cells is not None:
            row = self.cells.loc[cid] if cid in self.cells.index else None
            for n in self.feature_names:
                if n not in vals:
                    vals[n] = float(row[n]) if row is not None and n in row.index else np.nan
        if override:
            vals.update(override)
        return np.array([vals[n] for n in self.feature_names], dtype=float)


def top_concept(contrib: pd.DataFrame, cid: str) -> str | None:
    part = contrib[(contrib.case_id == cid) & (contrib.source != "cell")]
    if part.empty:
        return None
    s = part.groupby("concept").contribution.sum()
    return str(s.idxmax()) if s.max() > 0 else None


def summarize(drops: pd.DataFrame, groups: Mapping[str, str], cfg: Mapping[str, Any]) -> dict[str, Any]:
    boot = cfg["bootstrap"]
    out: dict[str, Any] = {"n": int(len(drops)), "n_groups": int(len({groups[c] for c in drops.index}))}
    if len(drops) < 2:
        out["note"] = "too few eligible cases"
        return out
    g = np.array([groups[c] for c in drops.index])

    for col in [c for c in drops.columns if c.startswith("drop_")]:
        ok = drops[col].notna().to_numpy()
        if ok.sum() >= 2:
            out[f"mean_{col}"] = stats.bootstrap_ci(lambda v: float(np.mean(v)), [drops[col].to_numpy()[ok]], g[ok],
                                                    None, int(boot["n"]), float(boot["ci"]), int(boot["seed"]))
    for m in ARMS:
        for unit, suffix in (("", ""), ("logit_", "_logit")):
            col = f"drop_{unit}{m}"
            if col not in drops:
                continue
            ok = drops[col].notna().to_numpy()
            if not unit:
                out[f"n_{m}"] = int(ok.sum())
            if ok.sum() >= 2:
                diff = (drops[col] - drops[f"drop_{unit}random"]).to_numpy()[ok]
                out[f"{m}_minus_random{suffix}"] = stats.bootstrap_ci(lambda v: float(np.mean(v)), [diff], g[ok], None,
                                                                      int(boot["n"]), float(boot["ci"]),
                                                                      int(boot["seed"]))
    return out


def run(ctx: cli.StepContext) -> dict[str, Any]:
    cfg, root = ctx.cfg, ctx.data_root
    dcfg = cfg["deletion"]
    n_del, n_rand = int(dcfg["n_tiles"]), int(dcfg["n_random"])
    choice = io.read_json(schemas.eval_json(ctx.artifacts_root, "model_choice"))
    cases = rm.read_cases(root)
    tiles = rm.read_tiles(root)
    groups = dict(zip(cases.case_id, cases.group))
    test_ids = sorted(cases[cases.split == "test"].case_id)
    if ctx.limit is not None:
        test_ids = cli.limit_case_ids(test_ids, ctx.limit)
    pred = rm.read_csv(schemas.predictions_csv(root)).set_index("case_id")
    contrib = rm.read_csv(schemas.contributions_csv(root)) if schemas.contributions_csv(root).is_file() else None
    tile_risk = rm.read_csv(schemas.tile_risk_csv(root)) if schemas.tile_risk_csv(root).is_file() else None
    models = []
    for m in (choice["production"], choice.get("model_a")):
        if m and m not in models:
            models.append(m)
    doc: dict[str, Any] = {"models": {}, "settings": dict(dcfg)}
    for name in models:
        reader = Reader(name, ctx, choice, tiles, test_ids)
        if reader.kind == "concept" and not reader.use_text:
            doc["models"][name] = {"skipped": "cell-only model: no tiles to remove (see cells section)"}
            continue
        rng = np.random.default_rng(int(dcfg["seed"]))
        rows = []
        for cid in test_ids:
            avail = reader.qc_tiles(cid)
            if len(avail) < n_del + int(dcfg["min_remaining"]):
                continue
            full, full_logit = reader.read(cid, set())
            row = {"case_id": cid, "risk_full": full,
                   "predicted_aggressive": int(np.argmax(reader.logits(cid, set())) == reader.agg)}

            def drop(arm: str, removed: set[int] | None) -> None:
                if removed is None:
                    row[f"drop_{arm}"] = row[f"drop_logit_{arm}"] = np.nan
                    return
                risk, logit = reader.read(cid, removed)
                row[f"drop_{arm}"], row[f"drop_logit_{arm}"] = full - risk, full_logit - logit

            rand = [reader.read(cid, set(rng.choice(avail, size=n_del, replace=False).tolist())) for _ in range(n_rand)]
            row["drop_random"] = float(np.mean([full - r for r, _ in rand]))
            row["drop_logit_random"] = float(np.mean([full_logit - lg for _, lg in rand]))
            if tile_risk is not None:
                tr = tile_risk[(tile_risk.case_id == cid) & (tile_risk.grid == "g512") & tile_risk.tile_idx.isin(avail)]
                top = tr.sort_values(["risk", "tile_idx"], ascending=[False, True]).tile_idx.head(n_del).astype(int)
                drop("risk", set(top.tolist()) if len(top) == n_del else None)
            concept = top_concept(contrib, cid) if contrib is not None and len(contrib) else None
            row["top_concept"] = concept
            ranked = None
            if concept is not None and schemas.concepts_tile_npz(root).is_file():
                zt = _concept_tiles(root, cid, concept)
                ranked = [t for t in zt if t in set(avail.tolist())][:n_del]
            drop("concept", set(ranked) if ranked and len(ranked) == n_del else None)
            if reader.kind == "lr_embedding":
                drop("contribution", set(reader.contribution_ranking(cid)[:n_del]))
            rows.append(row)
        drops = pd.DataFrame(rows).set_index("case_id")
        res: dict[str, Any] = {"kind": reader.kind, "n_eligible": int(len(drops))}
        if len(drops):
            sub = drops[drops.predicted_aggressive == 1]
            res["predicted_aggressive"] = summarize(sub.drop(columns=["top_concept"]), groups, cfg)
            res["all_eligible"] = summarize(drops.drop(columns=["top_concept"]), groups, cfg)
            res["top_concept_counts"] = drops.top_concept.value_counts().to_dict()
        if reader.kind != "lr_embedding":
            res["contribution_arm"] = (f"not available: a {reader.kind} model has no exact additive decomposition over "
                                       "regions")
        if reader.kind == "concept":
            res["caveat"] = ("concept bottleneck: the concept-ranked removal changes the very top-k features the model "
                             "reads, so its concept drop is partly circular (BUILD_PLAN Task 11)"
                             + ("; its cell features are held fixed when tiles are removed (cells come from the "
                                "selected top-risk tiles and are not re-segmented)" if reader.cells is not None else ""))
        doc["models"][name] = res
    doc["cells"] = cells_version(ctx, choice, test_ids, groups)
    return doc


_TILE_Z: dict[str, Any] = {}


def _concept_tiles(root, cid: str, concept: str) -> list[int]:
    """g512 tile indices of a case ordered by the concept's tile z (descending, ties by tile index)."""
    if "z" not in _TILE_Z or _TILE_Z.get("root") != str(root):
        z = io.load_npz(schemas.concepts_tile_npz(root))
        _TILE_Z.update(root=str(root), z=z, case=np.asarray(z["case_id"]).astype(str),
                       ids=[str(c) for c in z["concept_ids"]])
    z = _TILE_Z["z"]
    m = _TILE_Z["case"] == cid
    j = _TILE_Z["ids"].index(concept)
    t = np.asarray(z["tile_idx"])[m].astype(int)
    v = np.asarray(z["z"])[m][:, j]
    order = np.lexsort((t, -v))
    return t[order].tolist()


def cells_version(ctx: cli.StepContext, choice: Mapping[str, Any], test_ids: Sequence[str],
                  groups: Mapping[str, str]) -> dict[str, Any]:
    """CELLS_SPEC §8.5: top cell feature -> benign-reference median vs a random other cell feature."""
    cfg, root = ctx.cfg, ctx.data_root
    if not bool(cfg["cells"]["enabled"]):
        return {"skipped": "cells.enabled is false"}
    model = next((m for m in (choice["production"], choice.get("model_b")) if m in ("cb_text_cells", "cb_cells")), None)
    if model is None:
        return {"skipped": "no cell-feature model is production or B"}
    reader = Reader(model, ctx, choice, rm.read_tiles(root), test_ids)
    cf = rm.cell_frame(root, cfg)
    if cf is None:
        return {"skipped": "cell_features.csv missing"}
    frame, feats = cf
    cases = rm.read_cases(root).set_index("case_id")
    ref_ids = [c for c in cases.index if cases.split[c] == "train" and cases.label[c] in cfg["cells"]["reference_classes"]]
    medians = frame.loc[[c for c in ref_ids if c in frame.index], feats].median()
    w = reader.model.coef[reader.agg] - reader.model.coef.mean(0)
    names = reader.feature_names
    rows = []
    for cid in test_ids:
        if cid not in frame.index:
            continue
        x = reader.concept_vector(cid, set())
        contrib = reader.model.standardize(x[None, :])[0] * w
        cand = [(contrib[names.index(f)], f) for f in feats if f in names and not np.isnan(frame.loc[cid, f])]
        if len(cand) < 2:
            continue
        cand.sort(reverse=True)
        top_c, top_f = cand[0]
        if top_c <= 0:
            continue
        full = reader.risk(cid, set())

        def reset(f: str) -> float:
            v = reader.concept_vector(cid, set(), {f: float(medians[f]), f"{f}__missing": 0.0})
            return float(calibration.softmax(reader.model.logits(v[None, :]), reader.temperature)[0, reader.agg])

        rows.append({"case_id": cid, "drop_top": full - reset(top_f),
                     "drop_random": float(np.mean([full - reset(f) for _, f in cand[1:]])), "feature": top_f})
    if not rows:
        return {"model": model, "skipped": "no test case with a positive top cell-feature contribution"}
    d = pd.DataFrame(rows).set_index("case_id")
    boot = cfg["bootstrap"]
    g = np.array([groups[c] for c in d.index])

    def ci(v: np.ndarray) -> dict[str, Any]:
        return stats.bootstrap_ci(lambda a: float(np.mean(a)), [v], g, None, int(boot["n"]), float(boot["ci"]),
                                  int(boot["seed"]))

    return {"model": model, "n": int(len(d)), "n_groups": int(len(set(g))),
            "mean_drop_top": ci(d.drop_top.to_numpy()), "mean_drop_random": ci(d.drop_random.to_numpy()),
            "top_minus_random": ci((d.drop_top - d.drop_random).to_numpy()),
            "top_feature_counts": d.feature.value_counts().to_dict(),
            "reference": "median over train Normal/Benign cases (cells.reference_classes)"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("eval.deletion_test", "Deletion-test faithfulness of explanations (plan §12).")
    ctx = cli.context(parser.parse_args(argv))
    out = schemas.eval_json(ctx.artifacts_root, NAME)
    if out.is_file() and not ctx.force:
        print("eval.deletion_test: output exists; use --force to recompute")
        return 0
    need = [schemas.eval_json(ctx.artifacts_root, "model_choice"), schemas.predictions_csv(ctx.data_root)]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        print(f"eval.deletion_test: run pipeline.05_risk_model first; missing {missing}", file=sys.stderr)
        return 2
    body = run(ctx)
    prov = io.provenance(ctx.cfg, synthetic=ctx.synthetic)
    doc = {"name": NAME, "created_utc": prov["created_utc"], "config_hash": prov["config_hash"],
           "git_sha": prov["git_sha"], "synthetic": bool(ctx.synthetic), **body,
           "method": __doc__.split("\n\n")[1].strip()}
    io.write_json(out, rm.jsonable(doc))
    print(f"eval.deletion_test: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
