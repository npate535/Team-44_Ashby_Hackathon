# PathQueue task runner (laptop). Controller-owned shared file: tasks request target changes in their status report.
# Delta-side steps run inside Slurm jobs (pipeline/jobs/*.sbatch), never from here.
#
#   make setup                      uv environment (Python 3.13)
#   make test                       full pytest suite (no network)
#   make types fixtures             regenerate frontend/src/types.ts and fixtures/ after a contract change
#   make synthetic                  data_synthetic/ (schema-exact fake pipeline inputs, synthetic: true)
#   make analyze DATA=data_synthetic   laptop steps 04-08 + eval on a data root (skips steps not built yet)
#   make demo                       backend + teammate frontend on artifacts/ (CASES=fixtures for the fixture demo)
#   make demo-fresh                 same, in the background, on a fresh workspace (old DB archived)
#   make e2e                        Playwright end-to-end demo script (tests/e2e; skipped by `make test` unless PQ_E2E=1)

PY        ?= uv run python
DATA      ?= data
ART       ?= artifacts
CASES     ?=
HOST      ?= 127.0.0.1
PORT      ?= 8000
STEP_ARGS ?=

export PATHQUEUE_SITE ?= local

# Run `python -m <pkg>.<step>` only when the module exists (keeps main green while tasks land).
define run_step
	@if [ -f $(1)/$(2).py ]; then \
		echo ">> $(1).$(2)"; \
		PATHQUEUE_DATA_ROOT=$(DATA) PATHQUEUE_ARTIFACTS_ROOT=$(ART) $(PY) -m $(1).$(2) $(STEP_ARGS) || exit $$?; \
	else echo "-- skip $(1).$(2) (not built yet)"; fi
endef

.PHONY: setup test types types-check fixtures synthetic analyze pipeline-laptop eval demo demo-fresh e2e clean-synthetic

setup:
	uv sync

test:
	uv run pytest -q

types:
	$(PY) scripts/gen_types_ts.py

types-check:
	$(PY) scripts/gen_types_ts.py --check

fixtures:
	$(PY) scripts/make_fixtures.py

synthetic:
	@if [ -f pipeline/make_synthetic.py ]; then \
		$(PY) -m pipeline.make_synthetic --out data_synthetic --with-cells; \
	else echo "-- pipeline/make_synthetic.py not built yet"; fi

# Laptop analytics in dependency order (BUILD_PLAN §B). Each step is idempotent/resumable and reads the roots
# from PATHQUEUE_DATA_ROOT / PATHQUEUE_ARTIFACTS_ROOT. Pass extra flags with STEP_ARGS="--limit 5".
pipeline-laptop:
	$(call run_step,pipeline,01q_requalify_qc)
	$(call run_step,pipeline,04_cav)
	$(call run_step,pipeline,04_concepts)
	$(call run_step,pipeline,04c_cell_features)
	$(call run_step,pipeline,05_risk_model)
	$(call run_step,pipeline,05b_abmil)
	$(call run_step,pipeline,06_reliability)
	$(call run_step,pipeline,07_retrieval)
	$(call run_step,pipeline,08_build_case_json)

eval:
	$(call run_step,eval,metrics)
	$(call run_step,eval,deletion_test)
	$(call run_step,eval,queue_sim)
	$(call run_step,eval,margin_eval)
	$(call run_step,eval,cells_eval)
	$(call run_step,eval,benchmark_thunder)
	$(call run_step,eval,figures)
	$(call run_step,eval,analytics)
	$(call run_step,eval,results_md)

analyze: pipeline-laptop eval

demo:
	@if [ -f backend/app.py ]; then \
		PATHQUEUE_ARTIFACTS_ROOT=$(ART) PATHQUEUE_CASES_DIR=$(CASES) uv run uvicorn backend.app:app --host $(HOST) --port $(PORT); \
	else echo "-- backend/app.py not built yet"; fi

# Fresh replay for a presentation: archives (never deletes) the workspace DB, restarts the server in the background.
demo-fresh:
	PORT=$(PORT) ART=$(ART) HOST=$(HOST) scripts/demo_fresh.sh

e2e:
	@if [ -d tests/e2e ]; then PQ_E2E=1 uv run pytest -q tests/e2e; else echo "-- tests/e2e not built yet"; fi

clean-synthetic:
	rm -rf data_synthetic
