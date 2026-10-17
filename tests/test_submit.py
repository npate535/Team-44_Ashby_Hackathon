"""pipeline/jobs/submit.sh refuses every forbidden submission (fake sbatch/squeue/date shims on PATH)."""

from __future__ import annotations

import datetime as dt
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "pipeline" / "jobs" / "submit.sh"
CDT = dt.timezone(dt.timedelta(hours=-5))

GPU_SCRIPT = """#!/bin/bash
#SBATCH --job-name=pq-{name}
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={out}/%x-%j.out
{extra}
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail
"""
CPU_SCRIPT = """#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time={time}
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail
"""


def epoch(day: int, hh: int, mm: int) -> int:
    return int(dt.datetime(2026, 10, day, hh, mm, tzinfo=CDT).timestamp())


SAT_2100 = epoch(17, 21, 0)


class Env:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.jobs = tmp / "code" / "pipeline" / "jobs"
        self.jobs.mkdir(parents=True)
        shutil.copy(SUBMIT, self.jobs / "submit.sh")
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.squeue = tmp / "squeue.txt"
        self.squeue.write_text("")
        self.sbatch_calls = tmp / "sbatch_calls.txt"
        self.now = SAT_2100
        self.testonly_rc = 0
        self.jobid = "777"
        self._shim("date", 'if [ "${1:-}" = "+%s" ]; then echo "$FAKE_NOW"; else echo 2026-10-18T02:00:00Z; fi')
        self._shim("squeue", 'cat "$FAKE_SQUEUE"')
        self._shim("sbatch", 'echo "$* env:${SBATCH_RESERVATION:-}${SBATCH_TIMELIMIT:-}" >> "$FAKE_SBATCH_CALLS"\n'
                             'case " $* " in *" --test-only "*) echo "sbatch: Job 1 to start at 2027-03-01T00:00:00 '
                             'using 1 processors" >&2; exit "$FAKE_TESTONLY_RC" ;; esac\necho "$FAKE_JOBID"')
        self.gpu("smoke", time="00:20:00")
        self.gpu("embed_core")
        self.gpu("embed_extra")
        self.gpu("embed_aux")
        self.cpu("text_cpu")

    def _shim(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text(f"#!/bin/bash\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def gpu(self, name, *, time="01:00:00", gpus=1, cpus=16, mem="60G", extra=""):
        (self.jobs / f"{name}.sbatch").write_text(GPU_SCRIPT.format(name=name, time=time, gpus=gpus, cpus=cpus,
                                                                    mem=mem, extra=extra, out=self.tmp / "slurm"))

    def cpu(self, name, *, time="00:30:00", account="bfqi-delta-cpu"):
        (self.jobs / f"{name}.sbatch").write_text(CPU_SCRIPT.format(time=time, account=account))

    @property
    def log(self) -> Path:
        return self.tmp / "home" / "pathqueue" / "logs" / "submissions.log"

    def run(self, *args: str) -> subprocess.CompletedProcess:
        env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "FAKE_NOW": str(self.now),
               "FAKE_SQUEUE": str(self.squeue), "FAKE_SBATCH_CALLS": str(self.sbatch_calls),
               "FAKE_TESTONLY_RC": str(self.testonly_rc), "FAKE_JOBID": self.jobid, "USER": "rbansal3",
               "HOME": str(self.tmp / "home"), "SBATCH_RESERVATION": "sup-30434-17", "SBATCH_TIMELIMIT": "06:00:00"}
        return subprocess.run(["bash", str(self.jobs / "submit.sh"), *args], capture_output=True, text=True, env=env,
                              timeout=30)

    def submitted(self) -> list[str]:
        if not self.sbatch_calls.exists():
            return []
        return [l for l in self.sbatch_calls.read_text().splitlines() if "--test-only" not in l]

    def log_rows(self) -> list[list[str]]:
        return [l.split("\t") for l in self.log.read_text().splitlines()] if self.log.exists() else []


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def test_submits_and_logs_with_the_overnight_deadline(env):
    res = env.run("smoke")
    assert res.returncode == 0, res.stderr
    assert "SUBMITTED smoke as job 777" in res.stdout
    (call,) = env.submitted()
    assert call.startswith("--parsable --deadline=2026-10-18T07:59:00 ") and call.endswith("smoke.sbatch env:")
    (row,) = env.log_rows()
    assert row[1:6] == ["submitted", "smoke", "777", "1", "20"]
    assert "test_only" in row[8] and "one_gpu" in row[8] and "gpu_cap" in row[8]


def test_far_future_test_only_estimate_is_not_an_error(env):
    res = env.run("embed_core")
    assert res.returncode == 0 and "ignores reservations" in res.stdout


@pytest.mark.parametrize("job", ["rm_rf", "../smoke", "smoke.sbatch", ""])
def test_jobs_outside_the_allow_list_are_refused(env, job):
    res = env.run(job) if job else env.run()
    assert res.returncode != 0 and env.submitted() == []


def test_allowed_job_without_a_script_is_refused(env):
    res = env.run("cells")
    assert res.returncode == 1 and "does not exist" in res.stderr


@pytest.mark.parametrize("kwargs, message", [
    ({"time": "02:00:00"}, "exceeds 01:00:00"),
    ({"gpus": 2}, "exactly --gpus-per-node=1"),
    ({"cpus": 32}, "exceeds 16"),
    ({"mem": "120G"}, "exceeds 60G"),
    ({"extra": "#SBATCH --reservation=sup-30434-17"}, "--reservation is not allowed"),
])
def test_gpu_resource_limits(env, kwargs, message):
    env.gpu("embed_core", **kwargs)
    res = env.run("embed_core")
    assert res.returncode == 1 and message in res.stderr and env.submitted() == []
    assert env.log_rows()[-1][1] == "refused"


def test_cpu_job_with_the_gpu_account_is_refused(env):
    env.cpu("text_cpu", account="bfqi-delta-gpu")
    assert "bfqi-delta-cpu" in env.run("text_cpu").stderr


@pytest.mark.parametrize("now, job, ok, message", [
    (epoch(18, 6, 58), "smoke", True, ""),                       # 06:58 + 20 min ends 07:18
    (epoch(18, 7, 0), "embed_core", False, "crosses Sun 07:59"),  # 07:00 + 60 min = 08:00 > 07:59
    (epoch(18, 7, 45), "smoke", False, "crosses Sun 07:59"),
    (epoch(18, 7, 59), "smoke", False, "07:59-08:00 gap"),
    (epoch(18, 8, 0), "embed_core", True, ""),
    (epoch(18, 17, 30), "embed_core", False, "crosses Sun 18:00"),
    (epoch(18, 17, 30), "text_cpu", True, ""),                   # 30 min CPU job ends at 18:00
    (epoch(18, 18, 0), "text_cpu", False, "ended"),
])
def test_event_windows(env, now, job, ok, message):
    env.now = now
    res = env.run(job)
    assert (res.returncode == 0) == ok, res.stderr
    if ok:
        want = "2026-10-18T07:59:00" if now < epoch(18, 7, 59) else "2026-10-18T18:00:00"
        assert f"--deadline={want}" in env.submitted()[0]
    else:
        assert message in res.stderr and env.submitted() == []


def test_gpu_job_refused_while_another_gpu_job_is_queued(env):
    env.squeue.write_text("22956300|gpuA100x4|RUNNING|(null)\n22956284|cpu|RUNNING|(null)\n")
    res = env.run("embed_core")
    assert res.returncode == 1 and "already running/pending" in res.stderr


def test_cpu_job_is_allowed_while_a_gpu_job_runs(env):
    env.squeue.write_text("22956300|gpuA100x4|RUNNING|(null)\n")
    assert env.run("text_cpu").returncode == 0


def test_chain_after_the_tail_adds_the_dependency(env):
    env.squeue.write_text("100|gpuA100x4|RUNNING|(null)\n101|gpuA100x4|PENDING|afterok:100(unfulfilled)\n")
    res = env.run("embed_aux", "--after", "101")
    assert res.returncode == 0, res.stderr
    assert "--dependency=afterok:101" in env.submitted()[0]
    assert env.log_rows()[-1][6] == "101"


def test_chain_after_a_non_tail_job_is_refused(env):
    env.squeue.write_text("100|gpuA100x4|RUNNING|(null)\n101|gpuA100x4|PENDING|afterok:100(unfulfilled)\n")
    res = env.run("embed_aux", "--after", "100")
    assert res.returncode == 1 and "tail" in res.stderr


def test_after_an_unknown_job_while_gpu_busy_is_refused(env):
    env.squeue.write_text("100|gpuA100x4|RUNNING|(null)\n")
    res = env.run("embed_extra", "--after", "999")
    assert res.returncode == 1 and "not one of our queued GPU jobs" in res.stderr


def test_two_independent_gpu_jobs_already_queued_is_refused(env):
    env.squeue.write_text("100|gpuA100x4|RUNNING|(null)\n101|gpuA100x4|PENDING|(null)\n")
    res = env.run("embed_extra", "--after", "101")
    assert res.returncode == 1 and "one dependency chain" in res.stderr


def test_non_numeric_after_is_refused(env):
    assert "numeric" in env.run("smoke", "--after", "12;rm").stderr


def test_cumulative_gpu_cap_counts_only_submitted_gpu_jobs(env):
    for i in range(6):
        env.jobid = str(900 + i)
        assert env.run("embed_core").returncode == 0  # 6 x 60 = 360 min
    assert env.run("text_cpu").returncode == 0           # CPU time is not GPU time
    res = env.run("embed_extra")                         # 360 + 60 = 420: still allowed
    assert res.returncode == 0
    res = env.run("smoke")                               # 420 + 20 > 420
    assert res.returncode == 1 and "cap 420" in res.stderr
    statuses = [r[1] for r in env.log_rows()]
    assert statuses.count("submitted") == 8 and statuses[-1] == "refused"


def test_failed_test_only_refuses(env):
    env.testonly_rc = 1
    res = env.run("smoke")
    assert res.returncode == 1 and "--test-only rejected" in res.stderr and env.submitted() == []


def test_print_only_submits_nothing_and_logs_nothing(env):
    res = env.run("embed_core", "--print-only")
    assert res.returncode == 0 and "PRINT-ONLY: sbatch --parsable --deadline=2026-10-18T07:59:00" in res.stdout
    assert env.submitted() == [] and env.log_rows() == []


def test_unknown_flags_are_refused(env):
    res = env.run("smoke", "--force")
    assert res.returncode != 0 and env.submitted() == []


def test_script_never_loops_or_resubmits():
    text = SUBMIT.read_text()
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert code.count("$(sbatch --parsable") == 1  # exactly one real submission call
    assert "until " not in code and "while true" not in code


def test_repo_job_scripts_pass_the_guard(env):
    """Every GPU/CPU sbatch Agent 2 owns satisfies the directive checks (print-only, in the overnight block)."""
    for name in ("smoke", "embed_core", "embed_extra", "embed_aux", "text_cpu"):
        src = REPO / "pipeline" / "jobs" / f"{name}.sbatch"
        if not src.exists():
            continue
        shutil.copy(src, env.jobs / f"{name}.sbatch")
        env.now = epoch(17, 21, 0)
        res = env.run(name, "--print-only")
        assert res.returncode == 0, (name, res.stderr)


OWN_SBATCH = ("smoke", "embed_core", "embed_extra", "embed_aux", "text_cpu")
OWN_SHELL = ("submit.sh", "enc_env.sh", "hf_download.sh")


@pytest.mark.parametrize("name", OWN_SBATCH)
def test_job_scripts_follow_the_delta_template(name):
    path = REPO / "pipeline" / "jobs" / f"{name}.sbatch"
    if not path.exists():
        pytest.skip(f"{name}.sbatch not written yet")
    lines = path.read_text().splitlines()
    body = [l for l in lines[1:] if not l.startswith("#")]
    assert body[:3] == ["module load pytorch-conda/2.8", "unset PYTHONNOUSERSITE", "set -euo pipefail"]
    text = path.read_text()
    assert "--reservation" not in text.replace("no --reservation", "")
    assert "#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out" in lines
    assert 'cd "${PQ_CODE_DIR:-$HOME/pathqueue}"' in text
    if "python -m pipeline." in text:
        assert "HF_HUB_OFFLINE=1" in text and 'HF_HOME="$HOME/.cache/huggingface"' in text
    for later in [i for i, l in enumerate(body) if l.startswith("module load")][1:]:
        assert body[later + 1] == "unset PYTHONNOUSERSITE"
    assert "sbatch" not in "\n".join(body)  # jobs never submit jobs


@pytest.mark.parametrize("name", OWN_SBATCH + OWN_SHELL)
def test_shell_syntax(name):
    path = REPO / "pipeline" / "jobs" / (name if "." in name else f"{name}.sbatch")
    if not path.exists():
        pytest.skip(f"{path.name} not written yet")
    res = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_no_secrets_or_tokens_in_job_scripts():
    for path in (REPO / "pipeline" / "jobs").glob("*"):
        text = path.read_text()
        assert "hf_" not in text.replace("hf_download", "").replace("hf_hub", ""), path
        assert "API_KEY=" not in text and "Bearer " not in text, path


def test_job_encoder_lists_match_settings():
    from core import config

    cfg = config.load_settings()
    extra = (REPO / "pipeline" / "jobs" / "embed_extra.sbatch").read_text()
    margin = ",".join(cfg["encoders"]["margin_probe_encoders"])
    assert f"--sets bach256,bach512,deg256 --max-views bach512=1 --encoders {margin}" in extra
    core = (REPO / "pipeline" / "jobs" / "embed_core.sbatch").read_text()
    assert f"--encoders {cfg['encoders']['primary']}" in core and f"--encoders {cfg['encoders']['agreement']}" in core
    for name in ("embed_core", "text_cpu"):
        body = (REPO / "pipeline" / "jobs" / f"{name}.sbatch").read_text()
        assert "--encoders " + ",".join(cfg["encoders"]["text_aligned"]) in body


@pytest.mark.parametrize("extra, message", [
    ("#SBATCH --array=0-3", "--array is not allowed"),
    ("#SBATCH --exclusive", "--exclusive is not allowed"),
    ("#SBATCH --gres=gpu:4", "--gres is not allowed"),
    ("#SBATCH --gpus=2", "--gpus is not allowed"),
    ("#SBATCH -G 2", "-G / -a is not allowed"),
    ("#SBATCH --mem-per-cpu=8G", "--mem-per-cpu is not allowed"),
    ("#SBATCH --time=06:00:00", "--time appears 2 times"),
    ("#SBATCH --nodes=2", "--nodes appears 2 times"),
])
def test_resource_escapes_are_refused(env, extra, message):
    env.gpu("embed_core", extra=extra)
    res = env.run("embed_core")
    assert res.returncode == 1 and message in res.stderr and env.submitted() == []


def test_whole_node_memory_is_refused(env):
    env.gpu("embed_core", mem="0G")
    assert "whole node" in env.run("embed_core").stderr


def test_sbatch_input_environment_overrides_are_cleared(env):
    assert env.run("smoke").returncode == 0
    assert all(call.endswith("env:") for call in env.sbatch_calls.read_text().splitlines())


def test_concurrent_submission_lock(env):
    lock = env.log.parent / ".submit.lock"
    lock.mkdir(parents=True)
    res = env.run("smoke")
    assert res.returncode == 1 and "another submission" in res.stderr
    lock.rmdir()
    assert env.run("smoke").returncode == 0 and not lock.exists()


def test_team_users_are_all_checked(env):
    calls = env.tmp / "squeue_args.txt"
    env._shim("squeue", f'echo "$*" >> {calls}\ncat "$FAKE_SQUEUE"')
    os.environ["PQ_TEAM_USERS"] = "rbansal3,teammate2"
    try:
        assert env.run("smoke").returncode == 0
    finally:
        del os.environ["PQ_TEAM_USERS"]
    assert "-u rbansal3,teammate2" in calls.read_text()


def test_output_folder_is_created_before_submission(env):
    assert env.run("smoke").returncode == 0
    assert (env.tmp / "slurm").is_dir()
