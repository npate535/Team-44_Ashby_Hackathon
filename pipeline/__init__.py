"""PathQueue offline pipeline (plan §6; steps 00-08 plus helpers).

Delta-side steps (00-03, wsi_extract, mine_captions, 04b, degrade) must run on Python >= 3.10 with
only the `pytorch-conda/2.8` packages; laptop steps (04-08) run under the uv environment.
Run steps from the repo root as modules: ``python -m pipeline.<step> --limit 5``.
"""
