"""Step 05: risk models, CV-only selection, temperature, conformal sets, contributions, tile probes, margin probe.

    python -m pipeline.05_risk_model [--data-root D] [--artifacts-root A] [--limit N] [--force] [--no-abmil]

Plan §6.6, CELLS_SPEC §7.1, BUILD_PLAN §A3/§A5/§A8/§A9/§A12. Laptop step; reads the data root, never the test labels
for any choice. Everything below is WSI-free except the explicitly labeled ablations.

1. Candidates (pre-registered, settings model_choice): mean-pooled L2-LR per embedding candidate
   (risk_model.embedding_candidates), `abmil` on the best single encoder, `fusion_top3` (average of the top-k
   encoders' temperature-calibrated probabilities; never mixed embeddings), and the concept bottlenecks cb_text (B),
   cb_text_cells (B+C), cb_cells (C). A candidate whose inputs are missing is skipped and recorded, never faked.
2. Selection by patient-grouped CV inside train (riskmodels: K folds x R repeats, one-SE rule, simplest wins),
   separately for the black-box family (A) and the concept family (B); then the plan §6.6 A-vs-B rule on CV
   out-of-fold AUROC(Invasive) / BAcc decides what drives predictions (settings model_choice.production overrides).
3. Temperature per model from its out-of-fold train logits (all repeats, §A3); marginal LAC conformal on the
   calibration split at settings.alpha (APS with conformal.seed for the fit and conformal.seed + 1 for the sets,
   and Mondrian, both secondary); risk = calibrated P(aggressive class).
4. Outputs (data root): predictions.csv (calibration/test + every degraded and composite variant + out-of-fold train
   rows), contributions.csv (concept bottleneck B), tile_risk.csv (+ tile_risk_oof.csv when 04a has not written
   it), margins.json, models/<name>.npz (exported weights), predictions_view0.csv (private to 06: clean view-0
   reads of calibration/test cases, ruling M9). Artifacts root: eval/model_choice.json. The test split is only
   predicted, never scored, here.

Variant reads use the production model when it has features for the variant, else the first model of
risk_model.variant_fallback that has them (each row's `model`, `temperature` and `qhat_lac` say which). No coverage
claim is made for variants.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import calibration, cli, conformal, io, margin, schemas, stats
from pipeline import riskmodels as rm
from pipeline import tile_probe

STEP = "05_risk_model"
SPLITS_OUT = ("calibration", "test")


# ---- candidates ---------------------------------------------------------------------------------------------------------
class LRCandidate:
    """Mean-pooled (or concept-feature) L2-LR candidate over a FeatureSet."""

    kind = "lr"

    def __init__(self, name: str, family: str, fs: rm.FeatureSet, cfg: Mapping[str, Any], complexity: int):
        self.name, self.family, self.fs, self.cfg, self.complexity = name, family, fs, cfg, complexity
        self.results: list[rm.CVResult] = []
        self.chosen: rm.CVResult | None = None
        self.model: rm.LinearModel | None = None
        self.temperature = 1.0

    def _rows(self, ids: Sequence[str], labels: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
        src = self.fs.aug if self.fs.aug is not None else self.fs.clean
        part = src.loc[list(ids)]
        return part.to_numpy(np.float64), np.array([labels[i] for i in part.index])

    def _fit(self, ids: Sequence[str], labels: Mapping[str, int], C: float) -> rm.LinearModel:
        x, y = self._rows(ids, labels)
        rmc = self.cfg["risk_model"]
        return rm.LinearModel(self.cfg["classes"], C, int(rmc["max_iter"]), rmc["class_weight"]).fit(x, y, self.fs.names)

    def run_cv(self, train_ids: list[str], labels: Mapping[str, int], y: np.ndarray, splits) -> None:
        for C in self.cfg["risk_model"]["C_grid"]:
            def fit_predict(fit: np.ndarray, held: np.ndarray, C: float = float(C)) -> np.ndarray:
                m = self._fit([train_ids[i] for i in fit], labels, C)
                return m.logits(self.fs.clean.loc[[train_ids[i] for i in held]].to_numpy(np.float64))
            self.results.append(rm.run_cv(self.name, self.family, {"C": float(C)}, len(train_ids), y, splits,
                                          fit_predict, self.complexity, len(self.fs.names), len(self.cfg["classes"])))

    def fit_full(self, train_ids: list[str], labels: Mapping[str, int]) -> None:
        self.model = self._fit(train_ids, labels, float(self.chosen.config["C"]))

    def reads(self) -> dict[str, pd.DataFrame]:
        out = {"clean": self.fs.clean}
        if self.fs.variants is not None:
            out["variants"] = self.fs.variants
        if self.fs.view0 is not None:
            out["view0"] = self.fs.view0
        return out

    def logits(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.logits(frame.to_numpy(np.float64))

    def export(self, path: Path) -> None:
        self.model.to_npz(path, model=self.name, family=self.family, temperature=self.temperature,
                          source=repr(self.fs.source))


class ABMILCandidate:
    """Gated-attention MIL on one encoder's QC tile embeddings (train augmentation: per-view bags)."""

    kind = "abmil"

    def __init__(self, name: str, root: Path, spec: Mapping[str, str], tiles: pd.DataFrame, cfg: Mapping[str, Any],
                 complexity: int):
        self.name, self.family, self.cfg, self.complexity = name, "black_box", cfg, complexity
        self.spec = dict(spec)
        enc, set_name, head = spec["encoder"], spec["set"], spec["head"]
        x, idx = rm.load_emb(root, enc, set_name, head)
        ids, bags, tl = rm.tile_bags(x, idx, tiles)
        self.bags = dict(zip(ids, bags))
        self.bag_tiles = dict(zip(ids, tl))
        self.grid = str(idx["grid"].iloc[0])
        self.view_bags: dict[str, list[np.ndarray]] = {}
        self.view0: dict[str, np.ndarray] = {}
        if rm.emb_exists(root, enc, set_name, head, views=True) and bool(cfg["risk_model"]["use_views"]):
            xv, iv = rm.load_emb(root, enc, set_name, head, views=True)
            vid, vb, _ = rm.tile_bags(xv, iv, tiles, by_view=True)
            views = iv.drop_duplicates(["case_id", "view"]).sort_values(["case_id", "view"])
            for rid, bag in zip(vid, vb):
                self.view_bags.setdefault(rid, []).append(bag)
            first = rm.tile_bags(xv[iv["view"].to_numpy() == 0], iv[iv["view"] == 0].reset_index(drop=True), tiles)
            self.view0 = dict(zip(first[0], first[1]))
            del views
        self.variant_bags: dict[str, np.ndarray] = {}
        self.variant_tiles: dict[str, np.ndarray] = {}
        for vset, comp in ((rm.DEG_SETS.get(set_name), False), (rm.COMP_SETS.get(set_name), True)):
            if vset and rm.emb_exists(root, enc, vset, head):
                xd, idd = rm.load_emb(root, enc, vset, head)
                vi, vb, vt = rm.tile_bags(xd, idd, tiles, composite=comp)
                self.variant_bags.update(zip(vi, vb))
                self.variant_tiles.update(zip(vi, vt))
        self.results: list[rm.CVResult] = []
        self.chosen: rm.CVResult | None = None
        self.model: rm.ABMIL | None = None
        self.temperature = 1.0

    def _train(self, ids: Sequence[str], labels: Mapping[str, int]) -> rm.ABMIL:
        bags, y = [], []
        for i in ids:
            for b in self.view_bags.get(i, [self.bags[i]]):
                bags.append(b)
                y.append(labels[i])
        return rm.ABMIL(self.cfg["classes"], self.cfg["abmil"]).fit(bags, np.array(y))

    def run_cv(self, train_ids: list[str], labels: Mapping[str, int], y: np.ndarray, splits) -> None:
        def fit_predict(fit: np.ndarray, held: np.ndarray) -> np.ndarray:
            m = self._train([train_ids[i] for i in fit], labels)
            return m.predict([self.bags[train_ids[i]] for i in held])[0]
        self.results.append(rm.run_cv(self.name, self.family, {"encoder": self.spec["encoder"]}, len(train_ids), y,
                                      splits, fit_predict, self.complexity, 0, len(self.cfg["classes"])))

    def fit_full(self, train_ids: list[str], labels: Mapping[str, int]) -> None:
        self.model = self._train(train_ids, labels)

    def reads(self) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, Any] = {"clean": self.bags}
        if self.variant_bags:
            out["variants"] = self.variant_bags
        if self.view0:
            out["view0"] = self.view0
        return out

    def logits_bags(self, bags: Mapping[str, np.ndarray]) -> pd.DataFrame:
        ids = sorted(bags)
        logits, _ = self.model.predict([bags[i] for i in ids])
        return pd.DataFrame(logits, index=pd.Index(ids, name="case_id"))

    def export(self, path: Path) -> None:
        self.model.to_npz(path, model=self.name, temperature=self.temperature, encoder=self.spec["encoder"],
                          set=self.spec["set"], head=self.spec["head"])


class FusionCandidate:
    """Late fusion: mean of the components' temperature-calibrated probabilities (logits = log of the mean)."""

    kind = "fusion"

    def __init__(self, name: str, components: list[LRCandidate], complexity: int, cfg: Mapping[str, Any]):
        self.name, self.family, self.components, self.complexity, self.cfg = name, "black_box", components, complexity, cfg
        self.results: list[rm.CVResult] = []
        self.chosen: rm.CVResult | None = None
        self.temperature = 1.0

    def run_cv(self, train_ids: list[str], labels: Mapping[str, int], y: np.ndarray, splits) -> None:
        temps = [calibration.fit_temperature(c.chosen.oof_logits.reshape(-1, c.chosen.oof_logits.shape[-1]),
                                             np.tile(y, c.chosen.oof_logits.shape[0]),
                                             self.cfg["calibration"]["temperature_bounds"]) for c in self.components]
        reps = []
        for r in range(len(splits)):
            probs = np.mean([calibration.softmax(c.chosen.oof_logits[r], t) for c, t in zip(self.components, temps)], 0)
            reps.append(np.log(np.clip(probs, 1e-12, 1.0)))
        oof = np.stack(reps)
        losses = [calibration.nll(calibration.softmax(oof[r, held], 1.0), y[held])
                  for r, folds in enumerate(splits) for held in folds]
        self.results.append(rm.CVResult(self.name, self.family, {"components": [c.name for c in self.components]},
                                        oof, losses, self.complexity, 0))

    def fit_full(self, train_ids: list[str], labels: Mapping[str, int]) -> None:
        pass                                              # components are fit (and calibrated) on their own

    def fused(self, per_component: list[pd.DataFrame]) -> pd.DataFrame:
        common = sorted(set.intersection(*[set(f.index) for f in per_component]))
        probs = np.mean([calibration.softmax(f.loc[common].to_numpy(), c.temperature)
                         for f, c in zip(per_component, self.components)], axis=0)
        return pd.DataFrame(np.log(np.clip(probs, 1e-12, 1.0)), index=pd.Index(common, name="case_id"))


# ---- helpers ---------------------------------------------------------------------------------------------------------------
def cand_logits(c: Any, read: str) -> pd.DataFrame | None:
    """Uncalibrated logits of a fitted candidate for one read kind (clean | variants | view0), indexed by read id."""
    if c.kind == "lr":
        frame = c.reads().get(read)
        return None if frame is None else pd.DataFrame(c.logits(frame), index=frame.index)
    if c.kind == "abmil":
        bags = c.reads().get(read)
        return None if not bags else c.logits_bags(bags)
    parts = [cand_logits(comp, read) for comp in c.components]
    if any(p is None for p in parts):
        return None
    return c.fused(parts)


def envelope(name: str, cfg: Mapping[str, Any], synthetic: bool) -> dict[str, Any]:
    prov = io.provenance(cfg, synthetic=synthetic)
    return {"name": name, "created_utc": prov["created_utc"], "config_hash": prov["config_hash"],
            "git_sha": prov["git_sha"], "synthetic": bool(synthetic)}


def _cfg_row(r: rm.CVResult, n_folds: int) -> dict[str, Any]:
    return {"model": r.name, "family": r.family, "config": r.config, "complexity": r.complexity,
            "n_features": r.n_features, "cv_log_loss": r.mean_loss, "cv_log_loss_se": r.se(n_folds),
            "n_fold_scores": len(r.fold_losses)}


# ---- the step ---------------------------------------------------------------------------------------------------------------
class RiskStep:
    def __init__(self, ctx: cli.StepContext, use_abmil: bool):
        self.ctx, self.cfg, self.root = ctx, ctx.cfg, ctx.data_root
        self.use_abmil = use_abmil
        self.classes = list(self.cfg["classes"])
        self.agg = self.classes.index(self.cfg["aggressive_class"])
        self.cases = rm.read_cases(self.root)
        self.tiles = rm.read_tiles(self.root)
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        self.labels = {r.case_id: self.cidx[r.label] for r in self.cases.itertuples()}
        self.split_of = dict(zip(self.cases.case_id, self.cases.split))
        self.group_of = dict(zip(self.cases.case_id, self.cases.group))
        train = self.cases[self.cases.split == "train"].sort_values("case_id")
        self.train_ids = train.case_id.tolist()
        self.y_train = np.array([self.labels[i] for i in self.train_ids])
        self.groups_train = train.group.to_numpy()
        self.splits = rm.cv_splits(train, self.cfg)
        self.n_folds = int(self.cfg["split"]["cv"]["folds"])
        self.skipped: dict[str, str] = {}
        self.complexity = {n: i for i, n in enumerate(self.cfg["risk_model"]["complexity"])}

    # -- candidates ---------------------------------------------------------------------------------------------
    def build_candidates(self) -> list[Any]:
        cfg, out = self.cfg, []
        mc, rmc = cfg["model_choice"], cfg["risk_model"]
        specs = rmc["embedding_candidates"]
        for name in list(mc["candidates"]):
            if name in specs:
                fs = rm.embedding_features(self.root, name, specs[name], self.tiles, use_views=bool(rmc["use_views"]))
                if fs is None:
                    self.skipped[name] = f"embeddings missing: emb/{specs[name]['encoder']}/{specs[name]['set']}_{specs[name]['head']}"
                    continue
                out.append(LRCandidate(name, "black_box", fs, cfg, self._rank(name)))
            elif name not in ("abmil", "fusion_top3"):
                self.skipped[name] = "no specification in risk_model.embedding_candidates"
        concepts = rm.concept_frame(self.root, cfg)
        cells = rm.cell_frame(self.root, cfg)
        for name in mc["concept_models"]:
            fs = rm.concept_features(name, concepts, cells, cfg, self.cases)
            if fs is None:
                why = []
                if name in ("cb_text", "cb_text_cells") and concepts is None:
                    why.append("case_features.csv missing (04_concepts not run)")
                if name in ("cb_cells", "cb_text_cells") and cells is None:
                    why.append("cell_features.csv missing or cells.enabled false")
                self.skipped[name] = "; ".join(why)
                continue
            out.append(LRCandidate(name, "concept", fs, cfg, self._rank(name)))
        return out

    def _rank(self, name: str) -> int:
        if name not in self.complexity:
            raise KeyError(f"candidate {name!r} is missing from settings risk_model.complexity")
        return self.complexity[name]

    def run(self) -> None:
        cfg = self.cfg
        cands = self.build_candidates()
        if not cands:
            raise SystemExit("05_risk_model: no candidate has its inputs; run the embedding/concept steps first")
        for c in cands:
            c.run_cv(self.train_ids, self.labels, self.y_train, self.splits)
            c.chosen, _ = rm.one_se_choice(c.results, self.n_folds)
        singles = sorted([c for c in cands if c.family == "black_box"], key=lambda c: c.chosen.mean_loss)
        if "abmil" in cfg["model_choice"]["candidates"]:
            if not self.use_abmil:
                self.skipped["abmil"] = "disabled with --no-abmil"
            elif not singles:
                self.skipped["abmil"] = "no single-encoder embedding candidate available"
            else:
                best = singles[0]
                ab = ABMILCandidate("abmil", self.root, best.fs.source, self.tiles, cfg, self._rank("abmil"))
                ab.run_cv(self.train_ids, self.labels, self.y_train, self.splits)
                ab.chosen = ab.results[0]
                cands.append(ab)
        if "fusion_top3" in cfg["model_choice"]["candidates"]:
            k = int(cfg["risk_model"]["fusion_k"])
            by_encoder: dict[str, LRCandidate] = {}
            for c in singles:
                by_encoder.setdefault(c.fs.source["encoder"], c)
            comps = list(by_encoder.values())[:k]
            if len(comps) < 2:
                self.skipped["fusion_top3"] = f"needs >= 2 encoders, found {len(comps)}"
            else:
                fu = FusionCandidate("fusion_top3", comps, self._rank("fusion_top3"), cfg)
                fu.run_cv(self.train_ids, self.labels, self.y_train, self.splits)
                fu.chosen = fu.results[0]
                cands.append(fu)
        self.cands = {c.name: c for c in cands}
        self.select()
        self.fit_and_calibrate()
        self.write_predictions()
        self.write_contributions()
        self.write_tile_probes()
        self.write_margins()
        self.write_model_choice()

    # -- selection ----------------------------------------------------------------------------------------------
    def select(self) -> None:
        cfg = self.cfg
        mc = cfg["model_choice"]
        self.metrics = {n: rm.cv_metrics(c.chosen, self.y_train, self.groups_train, cfg, self.agg)
                        for n, c in self.cands.items()}
        fam = {f: [c for c in self.cands.values() if c.family == f] for f in ("black_box", "concept")}
        self.family_choice: dict[str, Any] = {}
        for f, members in fam.items():
            if members:
                chosen, info = rm.one_se_choice([c.chosen for c in members], self.n_folds)
                self.family_choice[f] = {"model": chosen.name, **info}
        a = self.family_choice.get("black_box", {}).get("model")
        b = self.family_choice.get("concept", {}).get("model")
        self.model_a, self.model_b = a, b
        self.ab_rule: dict[str, Any] = {}
        if mc["production"]:
            if mc["production"] not in self.cands:
                raise SystemExit(f"settings model_choice.production = {mc['production']!r} has no inputs on this root")
            self.production, self.production_reason = mc["production"], "settings model_choice.production (master)"
        elif a and b:
            winner, self.ab_rule = rm.a_vs_b(self.metrics[a], self.metrics[b], float(mc["auroc_margin"]),
                                             float(mc["bacc_margin"]))
            self.production = b if winner == "B" else a
            self.production_reason = (f"A-vs-B rule (CV out-of-fold): {'B drives' if winner == 'B' else 'A drives; B shown as associated features'}")
        else:
            self.production = a or b
            self.production_reason = "only one model family available"
        self.agreement = mc["agreement_model"] if mc["agreement_model"] in self.cands else None

    # -- fitting / calibration -------------------------------------------------------------------------------------
    def fit_and_calibrate(self) -> None:
        cfg = self.cfg
        bounds = cfg["calibration"]["temperature_bounds"]
        models_dir = self.root / cfg["risk_model"]["models_dir"]
        self.logit_tables: dict[str, dict[str, pd.DataFrame]] = {}
        for name, c in self.cands.items():
            if c.kind == "fusion":
                continue
            c.fit_full(self.train_ids, self.labels)
            oof = c.chosen.oof_logits
            c.temperature = calibration.fit_temperature(oof.reshape(-1, oof.shape[-1]), np.tile(self.y_train, oof.shape[0]),
                                                        bounds)
            c.export(models_dir / f"{name}.npz")
        for name, c in self.cands.items():
            if c.kind == "fusion":
                oof = c.chosen.oof_logits
                c.temperature = calibration.fit_temperature(oof.reshape(-1, oof.shape[-1]),
                                                            np.tile(self.y_train, oof.shape[0]), bounds)
        for name, c in self.cands.items():
            self.logit_tables[name] = {r: t for r in ("clean", "variants", "view0")
                                       if (t := cand_logits(c, r)) is not None}
        # conformal per model on the calibration split (production drives; others serve variant fallbacks)
        alpha = float(cfg["alpha"])
        seed = int(cfg["conformal"]["seed"])
        cal_ids = sorted(self.cases[self.cases.split == "calibration"].case_id)
        y_cal = np.array([self.labels[i] for i in cal_ids])
        self.conf: dict[str, dict[str, Any]] = {}
        for name, c in self.cands.items():
            clean = self.logit_tables[name]["clean"]
            p_cal = calibration.softmax(clean.loc[cal_ids].to_numpy(), c.temperature)
            self.conf[name] = {
                "qhat_lac": conformal.fit_lac(p_cal, y_cal, alpha),
                "qhat_aps": conformal.fit_aps(p_cal, y_cal, alpha, seed, bool(cfg["conformal"]["aps_randomized"])),
                "qhat_mondrian": conformal.mondrian_quantiles(p_cal, y_cal, alpha).tolist(),
                "temperature": float(c.temperature), "n_cal": len(cal_ids),
            }

    # -- predictions -------------------------------------------------------------------------------------------------
    def _read_meta(self, rid: str) -> tuple[str, str]:
        """(split, source case id) of a read id (clean case, degraded variant or composite)."""
        if rid in self.split_of:
            return self.split_of[rid], rid
        if "_FSIM_" in rid:
            src = rid.split("_FSIM_")[0]
            return self.split_of[src], src
        if "_COMP_" in rid:
            src = rid.split("_COMP_")[0]
            return self.split_of[src], src
        raise ValueError(f"cannot place read id {rid!r}")

    def _limit_ids(self) -> set[str] | None:
        if self.ctx.limit is None:
            return None
        pool = self.cases[self.cases.split.isin(SPLITS_OUT)].case_id
        return set(cli.limit_case_ids(pool, self.ctx.limit))

    def rows_for(self, rid: str, model: str, logits: np.ndarray, split: str) -> dict[str, Any]:
        cfg = self.cfg
        c = self.cands[model]
        conf = self.conf[model]
        pcal = calibration.softmax(logits[None, :], c.temperature)[0]
        mask = conformal.lac_sets(pcal[None, :], conf["qhat_lac"])[0]
        members = conformal.set_labels(mask, self.classes)
        mond = conformal.mondrian_sets(pcal[None, :], np.array(conf["qhat_mondrian"]))[0]
        row = {"case_id": rid, "split": split, "model": model, "temperature": float(c.temperature),
               "set_lac": schemas.format_set(members, self.classes), "qhat_lac": float(conf["qhat_lac"]),
               "set_size": int(mask.sum()), "risk": float(pcal[self.agg]), "top": self.classes[int(pcal.argmax())],
               "set_mondrian": schemas.format_set(conformal.set_labels(mond, self.classes), self.classes)}
        row.update({schemas.cal_prob_col(k): float(v) for k, v in zip(self.classes, pcal)})
        return row

    def write_predictions(self) -> None:
        cfg = self.cfg
        prod = self.production
        keep = self._limit_ids()
        tables = self.logit_tables
        rows: list[dict[str, Any]] = []
        aps_inputs: list[tuple[int, str, np.ndarray]] = []
        fallback = [m for m in cfg["risk_model"]["variant_fallback"] if m in self.cands and m != prod]
        self.missing_variants: list[str] = []
        clean = tables[prod]["clean"]
        for rid in clean.index:
            split = self.split_of[rid]
            if split not in SPLITS_OUT or (keep is not None and rid not in keep):
                continue
            rows.append(self.rows_for(rid, prod, clean.loc[rid].to_numpy(), split))
        variant_ids = sorted(set().union(*[set(t["variants"].index) for t in tables.values() if "variants" in t]))
        for rid in variant_ids:
            split, src = self._read_meta(rid)
            if keep is not None and src not in keep:
                continue
            for model in [prod] + fallback:
                t = tables[model].get("variants")
                if t is not None and rid in t.index:
                    rows.append(self.rows_for(rid, model, t.loc[rid].to_numpy(), split))
                    break
            else:
                self.missing_variants.append(rid)
        # out-of-fold train rows (calibration diagnostics; split = train)
        c = self.cands[prod]
        oof_cal = np.mean([calibration.softmax(l, c.temperature) for l in c.chosen.oof_logits], axis=0)
        for i, rid in enumerate(self.train_ids):
            row = self.rows_for(rid, prod, np.log(np.clip(oof_cal[i], 1e-12, 1)), "train")
            row["temperature"] = float(c.temperature)       # probabilities above are already calibrated OOF means
            rows.append(row)
        df = pd.DataFrame(rows)
        # randomized APS: one call on the whole matrix per model, fit seed for q-hat, seed + 1 for the sets
        seed = int(cfg["conformal"]["seed"])
        df["set_aps"] = ""
        for model, part in df.groupby("model"):
            p = part[[schemas.cal_prob_col(k) for k in self.classes]].to_numpy()
            sets = conformal.aps_sets(p, self.conf[model]["qhat_aps"], seed + 1, bool(cfg["conformal"]["aps_randomized"]))
            df.loc[part.index, "set_aps"] = [schemas.format_set(conformal.set_labels(s, self.classes), self.classes)
                                             for s in sets]
        # uncalibrated probabilities of every evaluated model; OOF means for train rows
        for model, c in self.cands.items():
            cols = [schemas.prob_col(model, k) for k in self.classes]
            vals = np.full((len(df), len(self.classes)), np.nan)
            oof = c.chosen.oof_probs()
            pos = {rid: i for i, rid in enumerate(self.train_ids)}
            for j, (rid, split) in enumerate(zip(df.case_id, df.split)):
                if split == "train":
                    vals[j] = oof[pos[rid]]
                    continue
                for kind in ("clean", "variants"):
                    t = self.logit_tables[model].get(kind)
                    if t is not None and rid in t.index:
                        vals[j] = calibration.softmax(t.loc[rid].to_numpy()[None, :], 1.0)[0]
                        break
            df[cols] = vals
        if self.agreement:
            acol = [schemas.prob_col(self.agreement, k) for k in self.classes]
            top_agree = np.array(self.classes)[np.nan_to_num(df[acol].to_numpy(), nan=-1).argmax(1)]
            has = df[acol].notna().all(axis=1)
            df["agree_top"] = np.where(has, (top_agree == df["top"]).astype(int), np.nan)
            df["agree_top"] = df["agree_top"].astype("Int64")
        else:
            df["agree_top"] = pd.array([pd.NA] * len(df), dtype="Int64")
        lead = [c.name for c in schemas.PREDICTIONS.columns]
        df = df[lead + [c for c in df.columns if c not in lead]].sort_values(["split", "case_id"]).reset_index(drop=True)
        schemas.validate_frame(schemas.PREDICTIONS, df)
        io.write_csv(schemas.predictions_csv(self.root), df)
        self.predictions = df
        self.write_view0()

    def write_view0(self) -> None:
        """Private file for 06 (ruling M9): clean view-0 reads of calibration/test cases by the production model."""
        prod = self.production
        t = self.logit_tables[prod].get("view0")
        if t is None:
            t = self.logit_tables[prod]["clean"]
            source = "tta (no view-0 features for the production model)"
        else:
            source = "view0"
        rows = []
        agree = self.logit_tables.get(self.agreement, {}).get("view0") if self.agreement else None
        for rid in t.index:
            split = self.split_of.get(rid)
            if split not in SPLITS_OUT:
                continue
            row = self.rows_for(rid, prod, t.loc[rid].to_numpy(), split)
            row["read"] = source
            if agree is not None and rid in agree.index:
                row["agree_top"] = int(self.classes[int(np.argmax(agree.loc[rid].to_numpy()))] == row["top"])
            rows.append(row)
        df = pd.DataFrame(rows)
        io.write_csv(self.root / "predictions_view0.csv", df)

    # -- contributions -------------------------------------------------------------------------------------------------
    def write_contributions(self) -> None:
        b = self.model_b
        self.contrib_model = b
        if b is None:
            io.write_csv(schemas.contributions_csv(self.root), pd.DataFrame(columns=[c.name for c in schemas.CONTRIBUTIONS.columns]))
            return
        c = self.cands[b]
        keep = self._limit_ids()
        frames = [c.fs.clean.loc[[i for i in c.fs.clean.index if self.split_of.get(i) in SPLITS_OUT]]]
        if c.fs.variants is not None:
            frames.append(c.fs.variants)
        x = pd.concat(frames)
        if keep is not None:
            x = x.loc[[i for i in x.index if self._read_meta(i)[1] in keep]]
        contrib = rm.centered_contributions(c.model, x.to_numpy(np.float64), self.agg)
        std = c.model.standardize(x.to_numpy(np.float64))
        concepts = set(self.cfg["concepts"]["ids"])
        rows = []
        for i, rid in enumerate(x.index):
            split = self._read_meta(rid)[0]
            for j, feat in enumerate(c.fs.names):
                base = feat.split("__")[0]
                rows.append({"case_id": rid, "split": split, "model": b, "feature": feat, "concept": base,
                             "source": "text" if base in concepts else "cell", "value": float(x.iloc[i, j]),
                             "value_std": float(std[i, j]), "contribution": float(contrib[i, j])})
        df = pd.DataFrame(rows)
        schemas.validate_frame(schemas.CONTRIBUTIONS, df)
        io.write_csv(schemas.contributions_csv(self.root), df)

    # -- tile probes ------------------------------------------------------------------------------------------------------
    def write_tile_probes(self) -> None:
        cfg = self.cfg
        spec = cfg["risk_model"]["heatmap_probe"]
        self.tile_probe_info: dict[str, Any] = {"probe": spec["name"]}
        if not rm.emb_exists(self.root, spec["encoder"], spec["set"], spec["head"]):
            self.tile_probe_info["skipped"] = "heatmap probe embeddings missing"
            return
        x, idx = rm.load_emb(self.root, spec["encoder"], spec["set"], spec["head"])
        oof_path = schemas.tile_risk_csv(self.root, oof=True)
        oof = tile_probe.oof_tile_risk(x, idx, self.cases, cfg, probe_name=spec["name"],
                                       folds=int(cfg["cells"]["oof_folds"]), qc_tiles=self.tiles)
        if not oof_path.is_file() or self.ctx.force:
            io.write_csv(oof_path, oof)
            self.tile_probe_info["tile_risk_oof"] = "written by 05 (04a had not written it)"
        else:
            self.tile_probe_info["tile_risk_oof"] = "kept (written earlier, e.g. by 04a)"
        # full-train fit for variants (fold = -1)
        qc = self.tiles[self.tiles.qc_pass][rm.TILE_KEY]
        tr = idx.drop(columns=["label"]).merge(qc, on=rm.TILE_KEY).merge(
            self.cases[["case_id", "label", "split"]].rename(columns={"split": "s"}), on="case_id")
        tr = tr[tr.s == "train"]
        x_train, y_train = x[tr.emb_row.to_numpy()], tr.label.to_numpy()
        heat = oof[oof.split.isin(SPLITS_OUT)].copy()
        wsi = self.wsi_tile_ablation(x_train, y_train, heat, spec)
        use_wsi = bool(wsi.get("drives_heatmap"))
        parts = [heat if not use_wsi else wsi["heat"]]
        for vset in (rm.DEG_SETS.get(spec["set"]), rm.COMP_SETS.get(spec["set"])):
            if not vset or not rm.emb_exists(self.root, spec["encoder"], vset, spec["head"]):
                continue
            xv, iv = rm.load_emb(self.root, spec["encoder"], vset, spec["head"])
            fit_x, fit_y = (wsi["x"], wsi["y"]) if use_wsi else (x_train, y_train)
            risk = tile_probe.fit_predict(fit_x, fit_y, xv, cfg)
            rid = list(rm.read_ids(iv))
            split = [self._read_meta(r)[0] for r in rid]
            parts.append(pd.DataFrame({"case_id": rid, "split": split, "grid": iv.grid, "tile_idx": iv.tile_idx.astype(int),
                                       "risk": risk, "probe": wsi["name"] if use_wsi else spec["name"], "fold": -1}))
        out = pd.concat(parts, ignore_index=True)
        keep = self._limit_ids()
        if keep is not None:
            out = out[[self._read_meta(r)[1] in keep for r in out.case_id]]
        out = out.sort_values(rm.TILE_KEY).reset_index(drop=True)
        schemas.validate_frame(schemas.TILE_RISK, out)
        io.write_csv(schemas.tile_risk_csv(self.root), out)
        self.tile_probe_info.update({k: v for k, v in wsi.items() if k not in ("heat", "x", "y")})

    def wsi_tile_ablation(self, x_train: np.ndarray, y_train: np.ndarray, heat: pd.DataFrame,
                          spec: Mapping[str, str]) -> dict[str, Any]:
        """WSI-enhanced tile probe (ablation, §A9): microscopy train tiles + labeled WSI tiles. Drives the heatmap only
        if its calibration-split case AUROC (mean of the top-3 tile risks, Invasive vs rest) beats the microscopy-only
        probe; both are also scored on held-out WSI tiles (patient-disjoint folds)."""
        cfg = self.cfg
        wset = cfg["risk_model"]["wsi_probe_set"]
        name = f"{spec['encoder']}_wsi_probe"
        if not rm.emb_exists(self.root, spec["encoder"], wset, spec["head"]):
            return {"wsi_ablation": f"skipped: emb/{spec['encoder']}/{wset}_{spec['head']} missing", "name": name}
        xw, iw = rm.load_emb(self.root, spec["encoder"], wset, spec["head"])
        ok = iw.label.isin(self.classes).to_numpy()
        xw, iw = xw[ok], iw[ok].reset_index(drop=True)
        x_all, y_all = np.vstack([x_train, xw]), np.concatenate([y_train, iw.label.to_numpy()])
        cal = heat[heat.split == "calibration"]
        xc, ic = rm.load_emb(self.root, spec["encoder"], spec["set"], spec["head"])
        cal_idx = ic.reset_index().merge(cal[rm.TILE_KEY], on=rm.TILE_KEY)
        risk_w = tile_probe.fit_predict(x_all, y_all, xc[cal_idx["index"].to_numpy()], cfg)
        heat_w = cal_idx[rm.TILE_KEY].assign(risk=risk_w)

        def case_auc(frame: pd.DataFrame) -> float:
            k = int(self.cfg["concepts"]["top_k"])
            top = frame.groupby("case_id").risk.apply(lambda s: float(np.sort(s.to_numpy())[::-1][:k].mean()))
            y = np.array([int(self.labels[c] == self.agg) for c in top.index])
            return stats.auroc(y, top.to_numpy())

        auc_m, auc_w = case_auc(cal), case_auc(heat_w)
        # held-out WSI tiles, folds by patient group
        groups = sorted(iw.group.astype(str).unique())
        nf = min(int(cfg["wsi"]["folds"]), len(groups))
        fold_of = {g: i % nf for i, g in enumerate(groups)}
        f = iw.group.astype(str).map(fold_of).to_numpy()
        held_w = np.zeros(len(iw))
        for k in range(nf):
            m = f == k
            held_w[m] = tile_probe.fit_predict(np.vstack([x_train, xw[~m]]),
                                               np.concatenate([y_train, iw.label.to_numpy()[~m]]), xw[m], cfg)
        held_m = tile_probe.fit_predict(x_train, y_train, xw, cfg)
        yw = (iw.label == self.cfg["aggressive_class"]).astype(int).to_numpy()
        drives = bool(auc_w > auc_m)
        out = {"name": name, "wsi_set": wset, "n_wsi_tiles": int(len(iw)),
               "calibration_case_auroc": {"microscopy_only": auc_m, "wsi_enhanced": auc_w},
               "heldout_wsi_tile_auroc": {"microscopy_only": stats.auroc(yw, held_m), "wsi_enhanced": stats.auroc(yw, held_w)},
               "drives_heatmap": drives, "rule": "WSI-enhanced drives the heatmap only if its calibration case AUROC is higher"}
        if drives:
            full = self.cases[self.cases.split.isin(SPLITS_OUT)][["case_id"]]
            all_idx = ic.reset_index().merge(heat[rm.TILE_KEY], on=rm.TILE_KEY)
            risk = tile_probe.fit_predict(x_all, y_all, xc[all_idx["index"].to_numpy()], cfg)
            out["heat"] = all_idx[rm.TILE_KEY].assign(split=[self.split_of[c] for c in all_idx.case_id], risk=risk,
                                                      probe=name, fold=-1)[["case_id", "split", "grid", "tile_idx", "risk", "probe", "fold"]]
            out["x"], out["y"] = x_all, y_all
            del full
        return out

    # -- margins ----------------------------------------------------------------------------------------------------------
    def margin_source(self) -> tuple[dict[str, str], bool]:
        for enc in self.cfg["encoders"]["margin_probe_encoders"]:
            if rm.emb_exists(self.root, enc, "bach256", "probe"):
                return {"encoder": enc, "set": "bach256", "head": "probe"}, False
        fb = self.cfg["margin"]["probe_fallback"]
        if rm.emb_exists(self.root, fb["encoder"], fb["set"], fb["head"]):
            return dict(fb), True
        return {}, True

    def write_margins(self) -> None:
        cfg = self.cfg
        mcfg = cfg["margin"]
        src, coarse = self.margin_source()
        self.margin_info: dict[str, Any] = {}
        if not src:
            self.margin_info["skipped"] = "no margin probe embeddings"
            return
        enc, set_name, head = src["encoder"], src["set"], src["head"]
        x, idx = rm.load_emb(self.root, enc, set_name, head)
        grid = cfg["tiles"]["grids"][str(idx.grid.iloc[0])]
        carc = set(cfg["carcinoma_classes"])
        qc = self.tiles[self.tiles.qc_pass][rm.TILE_KEY]
        tr = idx.reset_index().merge(qc, on=rm.TILE_KEY)
        tr = tr[tr.split == "train"]
        y = np.array([int(self.cases.set_index("case_id").label[c] in carc) for c in tr.case_id])
        probe = BinaryProbe(float(mcfg["probe_C"]), int(mcfg["probe_max_iter"])).fit(x[tr["index"].to_numpy()], y)
        seg_px = int(mcfg["segment_px"])
        names = mcfg["names"]

        def score_reads(xs: np.ndarray, ix: pd.DataFrame, sizes: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
            p = probe.predict(xs)
            ix = ix.assign(p=p, read=rm.read_ids(ix))
            out = {}
            for rid, part in ix.groupby("read"):
                w, h = sizes[rid]
                segs = margin.strip_segments(int(w), int(h), seg_px, names)
                lookup = dict(zip(part.tile_idx.astype(int), part.p))
                ps = []
                for s in segs:
                    t = (s["y"] // grid["size"]) * grid["cols"] + (s["x"] // grid["size"])
                    if t not in lookup:
                        raise ValueError(f"{rid}: tile {t} for margin segment ({s['x']}, {s['y']}) is missing")
                    ps.append(float(lookup[t]))
                out[rid] = {"width": int(w), "height": int(h),
                            "segments": [{**s, "p": round(pv, 6)} for s, pv in zip(segs, ps)],
                            "per_margin": margin.per_margin_scores(segs, [round(pv, 6) for pv in ps])}
            return out

        case_sizes = {r.case_id: (r.width, r.height) for r in self.cases.itertuples()}
        reads: dict[str, Any] = {}
        clean = idx[idx.split.isin(SPLITS_OUT)]
        reads.update(score_reads(x[clean.index.to_numpy()], clean, case_sizes))
        missing: list[str] = []
        comp_set = rm.COMP_SETS.get(set_name)
        comp_csv = schemas.composites_csv(self.root)
        if comp_set and comp_csv.is_file() and rm.emb_exists(self.root, enc, comp_set, head):
            comp = rm.read_csv(comp_csv)
            xc, ic = rm.load_emb(self.root, enc, comp_set, head)
            reads.update(score_reads(xc, ic, {r.case_id: (r.width, r.height) for r in comp.itertuples()}))
        deg_set = rm.DEG_SETS.get(set_name)
        if deg_set and rm.emb_exists(self.root, enc, deg_set, head):
            xd, idd = rm.load_emb(self.root, enc, deg_set, head)
            sizes = {r: case_sizes[c] for r, c in zip(rm.read_ids(idd), idd.case_id)}
            reads.update(score_reads(xd, idd, sizes))
        # degraded variants that have predictions but no fine-scale embeddings
        preds = set(self.predictions.case_id[self.predictions.case_id.str.contains("_FSIM_")])
        missing = sorted(preds - set(reads))
        win_csv = schemas.wsi_windows_csv(self.root)
        windows_eval: dict[str, Any] = {}
        if win_csv.is_file() and rm.emb_exists(self.root, enc, "wsi_margin256", head) and set_name == "bach256":
            win = rm.read_csv(win_csv).set_index("window_id")
            xw, iw = rm.load_emb(self.root, enc, "wsi_margin256", head)
            reads.update(score_reads(xw, iw, {w: (r.width, r.height) for w, r in win.iterrows()}))
            windows_eval = self.wsi_margin_ablation(x, idx, tr, y, xw, iw, win, score_reads)
        keep = self._limit_ids()
        if keep is not None:
            reads = {k: v for k, v in reads.items() if k not in self.split_of or k in keep}
        benign = sorted(c for c in self.cases[(self.cases.split == "calibration")
                                              & self.cases.label.isin(cfg["benign_classes"])].case_id if c in reads)
        if not benign:
            raise SystemExit("05_risk_model: no benign calibration image to set the margin threshold")
        values = np.array([[reads[c]["per_margin"][m] for m in schemas.MARGINS] for c in benign])
        max_fpr = float(mcfg["max_fpr"])
        threshold = margin.fit_threshold(values, max_fpr)
        boot = cfg["bootstrap"]
        groups = np.array([self.group_of[c] for c in benign])
        ci = stats.bootstrap_ci(lambda v: margin.fit_threshold(v, max_fpr), [values], groups, None, int(boot["n"]),
                                float(boot["ci"]), int(boot["seed"]))
        probe_name = f"{enc}_{'g512_coarse' if coarse else 'g256'}_carcinoma_probe"
        doc = {
            "probe": probe_name, "encoder": enc, "set": set_name, "coarse": bool(coarse),
            "coarse_note": ("no 224-native encoder embeddings: CONCH 512-px tiles score each 256-px segment "
                            "(two segments share a tile), documented as coarser") if coarse else None,
            "training": "WSI-free: train-split QC tiles with propagated case labels (InSitu or Invasive = carcinoma)",
            "strip_px": int(mcfg["strip_px"]), "segment_px": seg_px, "max_fpr": max_fpr, "threshold": float(threshold),
            "threshold_ci": {k: ci[k] for k in ("lo", "hi", "ci", "n_boot", "n_valid", "n", "n_groups")},
            "n_benign_cal_images": len(benign), "benign_cal_fpr": float(np.mean(values > threshold)),
            "cases": reads, "missing_variants": missing, "wsi_windows_ablation": windows_eval,
            **envelope("margins", cfg, self.ctx.synthetic),
        }
        doc.pop("name")
        if not schemas.MARGINS_JSON.problems(doc) == []:
            raise ValueError(schemas.MARGINS_JSON.problems(doc))
        io.write_json(schemas.margins_json(self.root), doc)
        self.margin_info = {k: doc[k] for k in ("probe", "coarse", "threshold", "threshold_ci", "n_benign_cal_images",
                                                "benign_cal_fpr")}
        self.margin_info["n_reads"] = len(reads)
        self.margin_info["missing_variants"] = len(missing)
        self.margin_info["wsi_windows_ablation"] = windows_eval

    def wsi_margin_ablation(self, x, idx, tr, y, xw, iw, win, score_reads) -> dict[str, Any]:
        """Per-margin AUROC on held-out WSI windows: WSI-free probe vs a probe that also trains on labeled wsi256
        tiles from other WSI folds (ablation; windows are scored by the WSI-free probe in production)."""
        mcfg = self.cfg["margin"]
        out: dict[str, Any] = {}
        truth = []
        free = score_reads(xw, iw, {w: (r.width, r.height) for w, r in win.iterrows()})
        for wid, r in win.iterrows():
            for m in schemas.MARGINS:
                truth.append((wid, m, int(r[f"truth_{m}"] == "tumor"), free[wid]["per_margin"][m], int(r["fold"])))
        t = pd.DataFrame(truth, columns=["window", "margin", "y", "p_free", "fold"])
        out["n_windows"] = int(win.shape[0])
        out["wsi_free_margin_auroc"] = stats.auroc(t.y.to_numpy(), t.p_free.to_numpy())
        enc = self.margin_source()[0]["encoder"]
        if rm.emb_exists(self.root, enc, "wsi256", "probe"):
            xl, il = rm.load_emb(self.root, enc, "wsi256", "probe")
            ok = il.label.isin(self.classes).to_numpy()
            xl, il = xl[ok], il[ok].reset_index(drop=True)
            yl = il.label.isin(self.cfg["carcinoma_classes"]).astype(int).to_numpy()
            fold_of = dict(zip(win.patient_id.astype(str), win.fold.astype(int)))
            lf = il.group.astype(str).map(fold_of).fillna(-1).astype(int).to_numpy()
            p_wsi = np.zeros(len(t))
            for k in sorted(t.fold.unique()):
                use = lf != k
                pr = BinaryProbe(float(mcfg["probe_C"]), int(mcfg["probe_max_iter"])).fit(
                    np.vstack([x[tr["index"].to_numpy()], xl[use]]), np.concatenate([y, yl[use]]))
                wins = win[win.fold == k].index
                sub = iw[iw.case_id.isin(wins)]
                ps = pr.predict(xw[sub.index.to_numpy()])
                cols = self.cfg["tiles"]["grids"]["g256"]["cols"]
                rows_ = self.cfg["tiles"]["grids"]["g256"]["rows"]
                names = self.cfg["margin"]["names"]
                for wid, part in sub.assign(p=ps).groupby("case_id"):
                    grid_p = np.zeros((rows_, cols))
                    for ti, pv in zip(part.tile_idx.astype(int), part.p):
                        grid_p[divmod(ti, cols)] = pv
                    edge = {"top": grid_p[0].max(), "bottom": grid_p[-1].max(), "left": grid_p[:, 0].max(),
                            "right": grid_p[:, -1].max()}
                    for e, m in names.items():
                        p_wsi[(t.window == wid) & (t.margin == m)] = edge[e]
            out["wsi_label_margin_auroc"] = stats.auroc(t.y.to_numpy(), p_wsi)
        out["note"] = "ablation only; WSI windows are not microscopy test images (scanner shift, imperfect annotations)"
        return out

    # -- model choice JSON ---------------------------------------------------------------------------------------------------
    def write_model_choice(self) -> None:
        cfg = self.cfg
        table = []
        for name, c in self.cands.items():
            for r in c.results:
                row = _cfg_row(r, self.n_folds)
                row["chosen_within_candidate"] = r is c.chosen
                table.append(row)
        ablation = {n: self.metrics[n] for n in cfg["model_choice"]["concept_models"] if n in self.metrics}
        doc = {
            **envelope("model_choice", cfg, self.ctx.synthetic),
            "protocol": {"selection": "patient-grouped CV inside train only (BUILD_PLAN §A3)",
                         "folds": self.n_folds, "repeats": int(cfg["split"]["cv"]["repeats"]),
                         "fold_seed_base": int(cfg["split"]["cv"]["seed"]), "group_by": cfg["split"]["cv"]["group_by"],
                         "cv_metric": cfg["model_choice"]["cv_metric"], "rule": cfg["model_choice"]["rule"],
                         "se_definition": "SD of the K x R held-out fold log-losses / sqrt(K)",
                         "temperature": "per model from its out-of-fold train logits (all repeats)",
                         "test_split_used": False, "n_train": len(self.train_ids),
                         "n_train_groups": int(len(set(self.groups_train)))},
            "candidates": table,
            "cv_metrics": self.metrics,
            "family_choice": self.family_choice,
            "model_a": self.model_a, "model_b": self.model_b,
            "a_vs_b": self.ab_rule,
            "production": self.production, "production_reason": self.production_reason,
            "agreement_model": self.agreement,
            "contributions_model": self.contrib_model,
            "skipped": self.skipped,
            "temperatures": {n: float(c.temperature) for n, c in self.cands.items()},
            "conformal": self.conf,
            "ablation_concepts_cells": ablation,
            "tile_probe": self.tile_probe_info,
            "margin_probe": self.margin_info,
            "variants": {"n_predicted": int((self.predictions.case_id.str.contains("_FSIM_|_COMP_")).sum()),
                         "missing": self.missing_variants,
                         "fallback_order": cfg["risk_model"]["variant_fallback"],
                         "note": "variants are predicted by the frozen pipeline; no coverage claim on variants"},
            "qc_fallback_cases": sorted({i for c in self.cands.values() if getattr(c, "fs", None) for i in c.fs.qc_fallback}),
        }
        io.write_json(schemas.eval_json(self.ctx.artifacts_root, "model_choice"), doc)


class BinaryProbe:
    """Standardized binary L2 logistic regression (margin carcinoma probe)."""

    def __init__(self, C: float, max_iter: int):
        self.C, self.max_iter = C, max_iter

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BinaryProbe":
        from sklearn.linear_model import LogisticRegression

        if len(set(np.asarray(y).tolist())) < 2:
            raise ValueError("margin probe needs carcinoma and non-carcinoma training tiles")
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(0)
        sd = x.std(0)
        self.scale = np.where(sd > 1e-12, sd, 1.0)
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter).fit((x - self.mean) / self.scale, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba((np.asarray(x, dtype=np.float64) - self.mean) / self.scale)[:, 1]


def outputs(root: Path, art: Path) -> list[Path]:
    return [schemas.predictions_csv(root), schemas.contributions_csv(root), schemas.tile_risk_csv(root),
            schemas.margins_json(root), schemas.eval_json(art, "model_choice"), root / "predictions_view0.csv"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Risk models, CV selection, calibration, conformal, tile/margin probes.")
    parser.add_argument("--no-abmil", action="store_true", help="skip the ABMIL candidate (cut order §I item 4)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    outs = outputs(ctx.data_root, ctx.artifacts_root)
    if io.outputs_exist(*outs) and not ctx.force:
        print(f"{STEP}: outputs exist; use --force to recompute")
        return 0
    if not schemas.cases_csv(ctx.data_root).is_file():
        print(f"{STEP}: {schemas.cases_csv(ctx.data_root)} missing", file=sys.stderr)
        return 2
    step = RiskStep(ctx, use_abmil=not args.no_abmil)
    step.run()
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=ctx.cfg,
                  inputs=[schemas.cases_csv(ctx.data_root), schemas.tiles_csv(ctx.data_root), "emb/", "case_features.csv"],
                  outputs=outs, args={"limit": ctx.limit, "force": ctx.force, "no_abmil": args.no_abmil},
                  synthetic=ctx.synthetic, extra={"production": step.production, "skipped": step.skipped})
    print(f"{STEP}: production = {step.production} ({step.production_reason}); A = {step.model_a}, B = {step.model_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
