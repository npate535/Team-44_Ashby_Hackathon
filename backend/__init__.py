"""PathQueue demo backend (BUILD_PLAN Task 6): FastAPI + SQLite + arrival simulator, run locally on a laptop.

    backend.db         SQLite store: case loading/validation, agent + decision state machine, routing, events
    backend.simulator  simulator clock, arrivals, surgical-case injection
    backend.app        HTTP API (docs/API.md) and the static frontend at /
    backend.dev_cases  synthetic, contract-valid dev/e2e cases (generated non-tissue media)

Research prototype on public, de-identified data. Not for clinical use. Ground truth (`eval_only`) is dropped at load
time and never stored or served.
"""
