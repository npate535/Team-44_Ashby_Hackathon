"""Worklist queue simulation: FIFO vs PathQueue tiers vs tiers + aging guard (plan §12, BUILD_PLAN Task 11).

    python -m eval.queue_sim [--reps N]

Model (every assumption is written to queue_sim.json `assumptions`):
  * One simulated day per replication: Poisson arrivals at settings queue_sim.arrivals_per_hour for
    queue_sim.day_hours; each arrival is a real TEST case drawn uniformly with replacement (clean cases only), with
    its real read: lane, calibrated probabilities, LAC set, risk, reliability flag and QC tile count.
  * One reader, fixed read time queue_sim.read_time_min per case, non-preemptive; the reader works past the end of
    the day until the queue is empty (overtime is reported).
  * Sensitivity: one extra run 'tiers_aging_<name>' per settings queue_sim.aging_scenarios entry, with the aging
    targets replaced (the configured 24/48 h targets exceed an 8 h day, so the configured guard never fires; the
    promoted_fraction metric shows how many reads were promoted). Labelled as illustrative, never the configured system.
  * Policies (settings queue_sim.policies): fifo = arrival order; tiers = core.tiers.sort_key with now=None (static
    tiers incl. the Invasive safety rule); tiers_aging = core.tiers.sort_key with now = the reader clock (aging guard,
    settings aging). Every policy sees the SAME arrival stream in a replication (common random numbers), so policy
    differences are paired.
  * Ground truth (cases.csv label) is used only to score waits: "aggressive" = true settings aggressive_class.
Outputs (minutes): time to first look for truly aggressive cases (mean / median / P90), mean wait per true class, per
base tier, for non-flagged (tier 4) cases, for false negatives (truly aggressive cases the read did not put in tier
<= 2), surgical cases, the paired difference vs FIFO (the delay a policy imposes on non-flagged cases and on false
negatives — reported honestly, Annarumma 2019), overtime and maximum wait. Each value is the mean over replications
with a 95% interval of the replication distribution (simulation uncertainty only, not test-set sampling).
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from core import cli, io, schemas, tiers
from eval.metrics import load_cases, load_predictions, write_eval

STEP = "eval_queue_sim"


def test_cases(cfg: Mapping[str, Any], root) -> list[dict[str, Any]]:
    """Static Case-shaped dicts (what core.tiers reads) for every clean test case, plus truth for scoring."""
    cases = load_cases(cfg, root)
    preds = load_predictions(cfg, root).set_index("case_id")
    rel_path = schemas.reliability_csv(root)
    if not rel_path.is_file():
        raise FileNotFoundError(f"missing input reliability.csv: {rel_path}")
    rel = io.read_csv(rel_path).set_index("case_id")
    tiles = io.read_csv(schemas.tiles_csv(root))
    tiles = tiles[tiles["grid"] == cfg["tiles"]["contract_grid"]]
    qc = tiles.groupby("case_id")["qc_pass"].apply(lambda s: int(pd.to_numeric(s).sum()))
    classes = list(cfg["classes"])
    out = []
    for _, row in cases[cases["split"] == "test"].sort_values("case_id").iterrows():
        cid = row["case_id"]
        for what, table in (("predictions.csv", preds), ("reliability.csv", rel)):
            if cid not in table.index:
                raise ValueError(f"test case {cid} is missing from {what}")
        if cid not in qc.index:
            raise ValueError(f"test case {cid} is missing from tiles.csv")
        p = preds.loc[cid]
        flag = rel.loc[cid, "reliability_flag"]
        if pd.isna(flag):
            raise ValueError(f"test case {cid}: blank reliability_flag")
        case = {"case_id": cid, "lane": row["lane"], "arrived_offset_s": 0.0,
                "prediction": {"classes": classes, "probs": [float(p[schemas.cal_prob_col(c)]) for c in classes],
                               "conformal_set": schemas.parse_set(p["set_lac"]), "risk_score": float(p["risk"])},
                "quality": {"reliability_flag": bool(int(flag)), "qc_tiles_kept": int(qc[cid])}}
        tier, reasons = tiers.assign_tier(case, cfg)
        out.append({"case": case, "truth": row["label"], "tier": tier, "reasons": reasons})
    if not out:
        raise ValueError("no test cases")
    return out


def simulate_day(pool: Sequence[Mapping[str, Any]], arrivals: Sequence[tuple[float, int]], policy: str,
                 cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Waits for one day. arrivals: (time_min, pool index) sorted by time. Returns one record per arrival."""
    read = float(cfg["queue_sim"]["read_time_min"])
    items = []
    for j, (t, k) in enumerate(arrivals):
        base = pool[k]
        case = {**base["case"], "case_id": f"{base['case']['case_id']}#{j:04d}", "arrived_offset_s": t * 60.0}
        items.append({"j": j, "t": t, "case": case, "pool": k})
    if policy == "tiers":
        static = {it["j"]: tiers.sort_key(it["case"], cfg, None) for it in items}
    queue: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    clock, nxt = 0.0, 0
    while nxt < len(items) or queue:
        while nxt < len(items) and items[nxt]["t"] <= clock:
            queue.append(items[nxt])
            nxt += 1
        if not queue:
            clock = items[nxt]["t"]
            continue
        if policy == "fifo":
            pick = min(queue, key=lambda it: (it["t"], it["j"]))
        elif policy == "tiers":
            pick = min(queue, key=lambda it: static[it["j"]])
        elif policy == "tiers_aging":
            now = clock * 60.0
            pick = min(queue, key=lambda it: tiers.sort_key(it["case"], cfg, now))
        else:
            raise ValueError(f"unknown queue policy {policy!r}")
        queue.remove(pick)
        promoted = policy == "tiers_aging" and tiers.AGING_PROMOTED in tiers.assign_tier(pick["case"], cfg, clock * 60.0)[1]
        out.append({"pool": pick["pool"], "arrival": pick["t"], "start": clock, "wait": clock - pick["t"],
                    "promoted": promoted})
        clock += read
    return out


def _stat(values: Sequence[float], fn) -> float:
    return float(fn(np.asarray(values))) if len(values) else math.nan


def day_metrics(records: Sequence[Mapping[str, Any]], pool: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
                ) -> dict[str, float]:
    aggressive = cfg["aggressive_class"]
    day_end = float(cfg["queue_sim"]["day_hours"]) * 60.0
    read = float(cfg["queue_sim"]["read_time_min"])
    waits = {k: [] for k in ("aggressive", "non_flagged", "false_negative", "surgical", "all")}
    per_class: dict[str, list[float]] = {c: [] for c in cfg["classes"]}
    per_tier: dict[int, list[float]] = {t: [] for t in (1, 2, 3, 4)}
    for r in records:
        item = pool[r["pool"]]
        w = r["wait"]
        waits["all"].append(w)
        per_class[item["truth"]].append(w)
        per_tier[item["tier"]].append(w)
        if item["truth"] == aggressive:
            waits["aggressive"].append(w)
            if item["tier"] > 2:
                waits["false_negative"].append(w)
        if item["tier"] == 4:
            waits["non_flagged"].append(w)
        if item["case"]["lane"] == "surgical":
            waits["surgical"].append(w)
    out = {
        "aggressive_mean": _stat(waits["aggressive"], np.mean),
        "aggressive_median": _stat(waits["aggressive"], np.median),
        "aggressive_p90": _stat(waits["aggressive"], lambda a: np.percentile(a, 90)),
        "non_flagged_mean": _stat(waits["non_flagged"], np.mean),
        "false_negative_mean": _stat(waits["false_negative"], np.mean),
        "surgical_mean": _stat(waits["surgical"], np.mean),
        "all_mean": _stat(waits["all"], np.mean),
        "max_wait": _stat(waits["all"], np.max),
        "overtime": max(0.0, (max((r["start"] for r in records), default=0.0) + read) - day_end) if records else 0.0,
        "n_arrivals": float(len(records)),
        "promoted_fraction": (sum(bool(r.get("promoted")) for r in records) / len(records)) if records else math.nan,
    }
    out.update({f"class_{c}_mean": _stat(v, np.mean) for c, v in per_class.items()})
    out.update({f"tier_{t}_mean": _stat(v, np.mean) for t, v in per_tier.items()})
    return out


def summarize(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if not len(arr):
        return {"mean": None, "lo": None, "hi": None, "n_reps": 0}
    return {"mean": float(arr.mean()), "lo": float(np.percentile(arr, 2.5)), "hi": float(np.percentile(arr, 97.5)),
            "n_reps": int(len(arr))}


def policy_runs(cfg: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """Run name -> (simulate_day policy, settings to use): the configured policies, plus one sensitivity run
    'tiers_aging_<scenario>' per settings queue_sim.aging_scenarios entry (aging targets replaced, aging enabled)."""
    runs: dict[str, tuple[str, Mapping[str, Any]]] = {p: (p, cfg) for p in cfg["queue_sim"]["policies"]}
    for name, targets in (cfg["queue_sim"].get("aging_scenarios") or {}).items():
        alt = {**cfg, "aging": {**cfg["aging"], "enabled": True, "turnaround_target_min": dict(targets)}}
        runs[f"tiers_aging_{name}"] = ("tiers_aging", alt)
    return runs


def simulate(pool: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], reps: int) -> dict[str, Any]:
    q = cfg["queue_sim"]
    rate, hours, seed = float(q["arrivals_per_hour"]), float(q["day_hours"]), int(q["seed"])
    runs = policy_runs(cfg)
    policies = list(runs)
    per_policy: dict[str, list[dict[str, float]]] = {p: [] for p in policies}
    rng = random.Random(seed)
    for _ in range(reps):
        t, arrivals = 0.0, []
        while True:
            t += rng.expovariate(rate / 60.0)
            if t > hours * 60.0:
                break
            arrivals.append((t, rng.randrange(len(pool))))
        for name, (policy, run_cfg) in runs.items():
            per_policy[name].append(day_metrics(simulate_day(pool, arrivals, policy, run_cfg), pool, run_cfg))
    keys = list(per_policy[policies[0]][0]) if reps else []
    results = {p: {k: summarize([d[k] for d in days]) for k in keys} for p, days in per_policy.items()}
    if "fifo" in per_policy:
        for p in policies:
            if p == "fifo":
                continue
            results[p]["vs_fifo"] = {
                k: summarize([d[k] - f[k] for d, f in zip(per_policy[p], per_policy["fifo"])])
                for k in ("aggressive_mean", "aggressive_p90", "non_flagged_mean", "false_negative_mean",
                          "surgical_mean", "all_mean")}
    return results


def run(ctx: cli.StepContext, reps: int | None = None) -> Any:
    cfg = ctx.cfg
    reps = int(cfg["queue_sim"]["reps"]) if reps is None else int(reps)
    pool = test_cases(cfg, ctx.data_root)
    q = cfg["queue_sim"]
    utilization = float(q["arrivals_per_hour"]) * float(q["read_time_min"]) / 60.0
    tier_counts = {str(t): sum(p["tier"] == t for p in pool) for t in (1, 2, 3, 4)}
    aggressive = cfg["aggressive_class"]
    payload = {
        "split": "test", "n_cases": len(pool), "reps": reps,
        "policies": {p: cfg["eval"]["queue_policies"].get(p, p) for p in policy_runs(cfg)},
        "assumptions": {
            "arrivals": f"Poisson, {q['arrivals_per_hour']} per hour for {q['day_hours']} h; cases drawn uniformly "
                        "with replacement from the clean test split",
            "reader": f"one reader, fixed {q['read_time_min']} min per case, non-preemptive, works overtime until empty",
            "utilization": round(utilization, 3),
            "tiers": "core.tiers (surgical 1; Invasive top or in the LAC set 2; set size != 1, reliability flag or "
                     "insufficient tissue 3; else 4)",
            "aging": dict(cfg["aging"]),
            "common_random_numbers": True, "seed": int(q["seed"]),
            "interval": "2.5-97.5 percentiles over simulated days (simulation uncertainty only)",
            "truth_use": "true labels only score the waits; they never drive the queue",
        },
        "test_case_tiers": tier_counts,
        "test_case_truth": {c: sum(p["truth"] == c for p in pool) for c in cfg["classes"]},
        "false_negatives_in_pool": sum(p["truth"] == aggressive and p["tier"] > 2 for p in pool),
        "results": simulate(pool, cfg, reps),
        "notes": ["Prioritizing flagged cases delays non-flagged ones: the vs_fifo rows report that cost (and the "
                  "delay of false negatives) next to the benefit (cf. Annarumma 2019).",
                  "A simulation with stated assumptions, not a measured workflow outcome."],
    }
    res = payload["results"]
    for name, block in res.items():
        frac = (block.get("promoted_fraction") or {}).get("mean")
        if name.startswith("tiers_aging") and frac is not None:
            payload["aging_promotions"] = {**payload.get("aging_promotions", {}), name: frac}
    if (res.get("tiers_aging", {}).get("promoted_fraction") or {}).get("mean") == 0:
        payload["notes"].append(
            "At the configured aging targets (" + ", ".join(f"tier {t}: {m} min" for t, m in
                                                             cfg["aging"]["turnaround_target_min"].items())
            + f") no case waits long enough to be promoted within a {q['day_hours']} h day, so tiers and tiers_aging "
              "coincide: the aging guard has NO measured effect here and must not be claimed to protect unflagged cases.")
    for name in res:
        if name.startswith("tiers_aging_") and "tiers" in res:
            frac = (res[name].get("promoted_fraction") or {}).get("mean")
            nf, base = (res[name].get("non_flagged_mean") or {}).get("mean"), (res["tiers"].get("non_flagged_mean") or {}).get("mean")
            effect = (f": {100 * frac:.1f}% of reads promoted; unflagged mean wait {base:.1f} -> {nf:.1f} min vs plain tiers"
                      if None not in (frac, nf, base) else "")
            payload["notes"].append(f"{name} is a sensitivity run with illustrative day-scale aging targets "
                                    f"(settings queue_sim.aging_scenarios), not the configured system{effect}.")
    out = write_eval(ctx.artifacts_root, "queue_sim", payload, cfg, ctx.synthetic)
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=cfg,
                  inputs=[schemas.cases_csv(ctx.data_root), schemas.predictions_csv(ctx.data_root),
                          schemas.reliability_csv(ctx.data_root), schemas.tiles_csv(ctx.data_root)],
                  outputs=[out], args={"reps": reps}, synthetic=ctx.synthetic)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("eval.queue_sim", "FIFO vs tiers vs tiers+aging queue simulation.")
    parser.add_argument("--reps", type=int, default=None, help="replications (default settings queue_sim.reps)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    try:
        out = run(ctx, args.reps)
    except (FileNotFoundError, ValueError) as exc:
        print(f"eval.queue_sim: {exc}", file=sys.stderr)
        return 1
    print(f"eval.queue_sim: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
