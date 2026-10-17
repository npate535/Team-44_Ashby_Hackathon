"""Shared step CLI conventions (core/cli.py)."""

import ast
import sys
from pathlib import Path

import pytest

from core import cli, config

REPO = Path(__file__).resolve().parents[1]


def test_base_flags_and_roots(tmp_path):
    parser = cli.step_parser("pipeline.demo_step", "demo")
    parser.add_argument("--encoder", default="conch")
    args = parser.parse_args(["--data-root", str(tmp_path / "d"), "--artifacts-root", str(tmp_path / "a"),
                              "--limit", "5", "--force", "--encoder", "musk"])
    ctx = cli.context(args, env={})
    assert ctx.step == "pipeline.demo_step"
    assert ctx.data_root == tmp_path / "d" and ctx.artifacts_root == tmp_path / "a"
    assert ctx.limit == 5 and ctx.force and ctx.args.encoder == "musk"
    assert ctx.cfg["aggressive_class"] == "Invasive"
    assert ctx.synthetic is False


def test_environment_roots_apply_when_flags_are_absent(tmp_path):
    args = cli.step_parser("pipeline.demo_step", "demo").parse_args([])
    env = {config.ENV_DATA_ROOT: str(tmp_path / "env_data"), config.ENV_SITE: "local"}
    ctx = cli.context(args, env=env)
    assert ctx.data_root == tmp_path / "env_data" and ctx.limit is None and not ctx.force


def test_synthetic_marker(tmp_path):
    (tmp_path / cli.SYNTHETIC_MARKER).write_text("{}")
    args = cli.step_parser("pipeline.demo_step", "demo").parse_args(["--data-root", str(tmp_path)])
    assert cli.context(args, env={}).synthetic is True


def test_limit_case_ids_sorts_dedups_and_truncates():
    assert cli.limit_case_ids(["BACH_0003", "BACH_0001", "BACH_0003", "BACH_0002"], 2) == ["BACH_0001", "BACH_0002"]
    assert cli.limit_case_ids(["b", "a"], None) == ["a", "b"]


def test_limit_must_be_positive():
    with pytest.raises(SystemExit):
        cli.step_parser("pipeline.demo_step", "demo").parse_args(["--limit", "0"])


def test_cli_is_delta_portable():
    tree = ast.parse((REPO / "core" / "cli.py").read_text(), feature_version=(3, 10))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported - set(sys.stdlib_module_names) - {"core", "__future__"} == set()
