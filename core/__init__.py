"""PathQueue shared pure logic, imported by pipeline, eval, agent and backend (BUILD_PLAN §A15).

Modules and where they may run:

    core.config       settings.yaml loader/validator, data/artifacts roots, mpp, config hash   Delta + laptop
    core.io           atomic writes, JSON/CSV/npy/npz helpers, meta.json writer               Delta + laptop
    core.schemas      column/shape specs + path helpers for every inter-task file              Delta + laptop
    core.contract     pydantic models of the Case JSON contract (plan §7 + additions)          laptop only
    core.tiers        worklist tier assignment + sort key (plan §1, BUILD_PLAN §A5/§A21)       laptop only
    core.conformal    split-conformal sets (LAC / APS / Mondrian)          [Task 4 implements]  laptop only
    core.calibration  temperature scaling, ECE, Brier, NLL                 [Task 4 implements]  laptop only
    core.stats        bootstrap CIs, AUROC, macro-F1, balanced accuracy    [Task 4 implements]  laptop only
    core.margin       surgical margin strip geometry, threshold, call      [Task 4 implements]  laptop only

Portability rule: "Delta + laptop" modules must run on Python >= 3.10 with only the packages of
the Delta `pytorch-conda/2.8` module (stdlib, numpy, pandas, PyYAML, Pillow). They must not import
pydantic, fastapi or anything else that is not guaranteed there; tests/test_config.py enforces this.
This package therefore imports nothing at package level.

Run pipeline steps from the repo root as modules (``python -m pipeline.<step>``) so ``import core``
resolves without an installed environment (Delta uses the conda module, not uv).
"""
