"""Load and validate config/settings.yaml; resolve data/artifacts roots; derived settings helpers.

Usage:
    from core import config
    cfg = config.load_settings()            # plain nested dict, validated, "{team_dir}" expanded
    root = config.data_root(cfg)            # pipeline files (core.schemas path helpers take this root)
    art = config.artifacts_root(cfg)        # app-facing files (cases/, images/, cells/, eval/)
    mpp = config.effective_mpp(cfg)         # measured BACH µm/px from inventory.json, else dataset.mpp

Roots:
    data root       $PATHQUEUE_DATA_ROOT if set, else paths.delta.data_root on Delta, else ./data
    artifacts root  $PATHQUEUE_ARTIFACTS_ROOT if set, else paths.delta.artifacts_root on Delta, else ./artifacts
    Relative paths (in settings or env) resolve against the repo root, never the current directory.
    The site is $PATHQUEUE_SITE (delta | local) if set, else "delta" when paths.delta.work_root exists.
    Empty environment variables count as unset.

Portable: Python >= 3.10, stdlib + PyYAML only (runs inside Delta jobs).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from core import schemas

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"

ENV_DATA_ROOT = "PATHQUEUE_DATA_ROOT"
ENV_ARTIFACTS_ROOT = "PATHQUEUE_ARTIFACTS_ROOT"
ENV_SITE = "PATHQUEUE_SITE"
SITES = ("delta", "local")
TEAM_DIR_PLACEHOLDER = "{team_dir}"

CONFORMAL_METHODS = ("lac", "aps")
CONFORMAL_MODES = ("marginal", "mondrian")
CELL_BACKENDS = ("tiatoolbox", "hovernet", "hovernext")
MARGIN_SIDES = ("top", "bottom", "left", "right")


class ConfigError(ValueError):
    """Invalid settings; `problems` lists every issue found (not just the first)."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("invalid settings:\n- " + "\n- ".join(self.problems))


# ---- loading -----------------------------------------------------------------------------------------
def load_settings(path: str | Path | None = None, *, validate: bool = True) -> dict[str, Any]:
    """Read settings.yaml (default config/settings.yaml), expand "{team_dir}", validate, return a dict."""
    path = Path(path) if path is not None else SETTINGS_PATH
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError([f"{path} must contain a mapping at the top level"])
    team_dir = (raw.get("team") or {}).get("dir")
    if not isinstance(team_dir, str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", team_dir):
        raise ConfigError(["team.dir must be a simple folder name (letters, digits, _ or -)"])
    cfg = _expand(raw, team_dir)
    if validate:
        validate_settings(cfg)
    return cfg


def _expand(obj: Any, team_dir: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(TEAM_DIR_PLACEHOLDER, team_dir)
    if isinstance(obj, list):
        return [_expand(v, team_dir) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand(v, team_dir) for k, v in obj.items()}
    return obj


def load_concept_bank(path: str | Path | None = None, cfg: Mapping[str, Any] | None = None) -> schemas.ConceptBank:
    """Read and validate config/concepts.yaml against the ruling-R1 schema (core.schemas)."""
    cfg = load_settings() if cfg is None else cfg
    path = Path(path) if path is not None else resolve_repo_path(cfg["concepts"]["file"])
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return schemas.validate_concept_bank(data, cfg)


# ---- validation --------------------------------------------------------------------------------------
_MISSING = object()


def _get(cfg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def settings_problems(cfg: Mapping[str, Any]) -> list[str]:
    """Return every problem found in a settings mapping (empty list = valid)."""
    problems: list[str] = []

    def need(dotted: str) -> Any:
        value = _get(cfg, dotted)
        if value is _MISSING:
            problems.append(f"missing key {dotted}")
        return value

    def check(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    def number_in(dotted: str, lo: float, hi: float, *, lo_open: bool = False, hi_open: bool = False) -> None:
        value = need(dotted)
        if value is _MISSING:
            return
        ok = _is_num(value) and (value > lo if lo_open else value >= lo) and (value < hi if hi_open else value <= hi)
        lo_b, hi_b = ("(" if lo_open else "["), (")" if hi_open else "]")
        check(ok, f"{dotted} must be a number in {lo_b}{lo}, {hi}{hi_b}, got {value!r}")

    def positive_int(dotted: str) -> None:
        value = need(dotted)
        if value is not _MISSING:
            check(_is_int(value) and value >= 1, f"{dotted} must be a positive integer, got {value!r}")

    def one_of(dotted: str, allowed: tuple[str, ...]) -> None:
        value = need(dotted)
        if value is not _MISSING:
            check(value in allowed, f"{dotted} must be one of {list(allowed)}, got {value!r}")

    # classes
    classes = need("classes")
    class_set: set[str] = set()
    if classes is not _MISSING:
        if isinstance(classes, list) and classes and all(isinstance(c, str) and c for c in classes) \
                and len(set(classes)) == len(classes):
            class_set = set(classes)
        else:
            problems.append("classes must be a non-empty list of unique names")

    def classes_subset(dotted: str, values: Any) -> None:
        if not class_set or values is _MISSING:
            return
        unknown = [v for v in values if v not in class_set] if isinstance(values, list) else [values]
        check(not unknown, f"{dotted} lists names that are not classes: {unknown}")

    aggressive = need("aggressive_class")
    if class_set and aggressive is not _MISSING:
        check(aggressive in class_set, f"aggressive_class {aggressive!r} is not one of the classes {classes}")
    for dotted in ("carcinoma_classes", "benign_classes", "cells.reference_classes"):
        classes_subset(dotted, need(dotted))
    order = need("split.class_order")
    if class_set and order is not _MISSING:
        check(isinstance(order, list) and sorted(order) == sorted(class_set) and len(order) == len(class_set),
              f"split.class_order must be a permutation of the classes, got {order!r}")
    cover = need("lanes.surgical_must_cover")
    if cover is not _MISSING:
        if isinstance(cover, list) and all(isinstance(g, list) and g for g in cover):
            for group in cover:
                classes_subset("lanes.surgical_must_cover", group)
        else:
            problems.append("lanes.surgical_must_cover must be a list of non-empty class lists")

    # team / dataset / qc / tiles
    team_dir = need("team.dir")
    if team_dir is not _MISSING:
        check(isinstance(team_dir, str) and bool(re.fullmatch(r"[A-Za-z0-9_\-]+", team_dir)),
              f"team.dir must be a simple folder name, got {team_dir!r}")
    number_in("dataset.mpp", 0, 100, lo_open=True)
    number_in("qc.saturation_min", 0, 1)
    number_in("qc.brightness_max", 0, 1)
    number_in("qc.min_tissue_frac", 0, 1)
    min_tiles = need("qc.min_tiles_kept")
    if min_tiles is not _MISSING:
        check(_is_int(min_tiles) and min_tiles >= 0, f"qc.min_tiles_kept must be an integer >= 0, got {min_tiles!r}")
    grids = need("tiles.grids")
    grid_names: set[str] = set()
    if grids is not _MISSING:
        if isinstance(grids, Mapping) and grids:
            grid_names = set(grids)
            for name, spec in grids.items():
                size = spec.get("size") if isinstance(spec, Mapping) else None
                check(_is_int(size) and size > 0, f"tiles.grids.{name}.size must be a positive integer")
        else:
            problems.append("tiles.grids must be a non-empty mapping")
    contract_grid = need("tiles.contract_grid")
    if grid_names and contract_grid is not _MISSING:
        check(contract_grid in grid_names, f"tiles.contract_grid {contract_grid!r} is not a defined grid")
    number_in("lanes.surgical_fraction", 0, 1)

    # split
    one_of("split.scheme", ("thunder",))
    _split_problems(cfg, problems)

    # conformal / calibration / bootstrap
    number_in("alpha", 0, 1, lo_open=True, hi_open=True)
    one_of("conformal.method", CONFORMAL_METHODS)
    one_of("conformal.mode", CONFORMAL_MODES)
    positive_int("calibration.ece_bins")
    positive_int("bootstrap.n")
    number_in("bootstrap.ci", 0, 1, lo_open=True, hi_open=True)

    # concepts
    _concept_problems(cfg, problems)

    # encoders
    priority = need("encoders.priority")
    if priority is not _MISSING:
        if isinstance(priority, list) and priority:
            for dotted in ("encoders.primary", "encoders.agreement"):
                value = need(dotted)
                if value is not _MISSING:
                    check(value in priority, f"{dotted} {value!r} is not in encoders.priority")
        else:
            problems.append("encoders.priority must be a non-empty list")
    native = need("encoders.native_grid")
    if grid_names and isinstance(native, Mapping):
        bad = {k: v for k, v in native.items() if v not in grid_names}
        check(not bad, f"encoders.native_grid uses undefined grids: {bad}")

    # reliability / retrieval / model choice
    positive_int("reliability.k")
    number_in("reliability.target_error_recall", 0, 1, lo_open=True)
    positive_int("retrieval.k")
    number_in("model_choice.auroc_margin", 0, 1)
    number_in("model_choice.bacc_margin", 0, 1)

    # degradations
    types = need("degrade.types")
    severities = need("degrade.severities")
    params = need("degrade.params")
    if isinstance(types, list) and isinstance(severities, list) and isinstance(params, Mapping):
        for kind in types:
            kind_params = params.get(kind)
            if not isinstance(kind_params, Mapping) or not kind_params:
                problems.append(f"degrade.params.{kind} is missing")
                continue
            for name, values in kind_params.items():
                check(isinstance(values, list) and len(values) == len(severities),
                      f"degrade.params.{kind}.{name} needs one value per severity {severities}")

    # margin / timer / aging
    names = need("margin.names")
    if names is not _MISSING:
        check(isinstance(names, Mapping) and sorted(names) == sorted(MARGIN_SIDES)
              and sorted(names.values()) == sorted(schemas.MARGINS),
              f"margin.names must map {list(MARGIN_SIDES)} onto {list(schemas.MARGINS)}")
    positive_int("margin.strip_px")
    positive_int("margin.segment_px")
    number_in("margin.max_fpr", 0, 1, lo_open=True, hi_open=True)
    amber, red = need("margin.timer.amber_min"), need("margin.timer.red_min")
    if amber is not _MISSING and red is not _MISSING:
        check(_is_num(amber) and _is_num(red) and 0 < amber < red,
              f"margin.timer needs 0 < amber_min < red_min, got amber {amber!r}, red {red!r}")
    enabled = need("aging.enabled")
    if enabled is not _MISSING:
        check(isinstance(enabled, bool), "aging.enabled must be true or false")
    number_in("aging.promote_at_fraction", 0, 1, lo_open=True)
    targets = need("aging.turnaround_target_min")
    if targets is not _MISSING:
        ok = isinstance(targets, Mapping) and all(
            _is_num(targets.get(t, targets.get(str(t)))) and targets.get(t, targets.get(str(t))) > 0
            for t in (1, 2, 3, 4)
        )
        check(ok, "aging.turnaround_target_min needs a positive number of minutes for tiers 1-4")

    # cells
    cells_enabled = need("cells.enabled")
    if cells_enabled is not _MISSING:
        check(isinstance(cells_enabled, bool), "cells.enabled must be true or false")
    one_of("cells.backend", CELL_BACKENDS)
    number_in("cells.model_mpp", 0, 100, lo_open=True)
    positive_int("cells.tiles_per_case")
    extremes = need("cells.findings_percentile_extremes")
    if extremes is not _MISSING:
        check(isinstance(extremes, list) and len(extremes) == 2 and all(_is_num(v) for v in extremes)
              and 0 <= extremes[0] < extremes[1] <= 1,
              f"cells.findings_percentile_extremes must be [lo, hi] with 0 <= lo < hi <= 1, got {extremes!r}")

    # llm / sim
    prefs = need("llm.model_preference")
    if prefs is not _MISSING:
        check(isinstance(prefs, list) and bool(prefs) and all(isinstance(p, str) and p for p in prefs),
              "llm.model_preference must be a non-empty list of model ids")
    number_in("sim.speed", 0, 1e6, lo_open=True)
    return problems


def _split_problems(cfg: Mapping[str, Any], problems: list[str]) -> None:
    n_images = _get(cfg, "dataset.n_images")
    ranges = _get(cfg, "split.ranges")
    expected = _get(cfg, "split.expected_counts")
    if ranges is _MISSING or not isinstance(ranges, Mapping):
        problems.append("missing key split.ranges")
        return
    if sorted(ranges) != sorted(schemas.SPLITS):
        problems.append(f"split.ranges must define exactly {list(schemas.SPLITS)}, got {sorted(ranges)}")
    owner: dict[int, str] = {}
    overlaps: set[tuple[str, str]] = set()
    for split, spans in ranges.items():
        for span in spans if isinstance(spans, list) else []:
            if not (isinstance(span, list) and len(span) == 2 and all(_is_int(v) for v in span) and 0 <= span[0] < span[1]):
                problems.append(f"split.ranges.{split}: {span!r} is not a half-open [start, end) pair")
                continue
            for index in range(span[0], span[1]):
                if index in owner:
                    overlaps.add((owner[index], split))
                else:
                    owner[index] = split
    if overlaps:
        problems.append(f"split.ranges overlap between {sorted(overlaps)}")
    if _is_int(n_images):
        unassigned = [i for i in range(n_images) if i not in owner]
        if unassigned:
            problems.append(f"split.ranges: {len(unassigned)} image indices not assigned (first {unassigned[:5]})")
        beyond = sorted(i for i in owner if i >= n_images)
        if beyond:
            problems.append(f"split.ranges index beyond dataset.n_images={n_images}: {beyond[:5]}")
    else:
        problems.append("dataset.n_images must be an integer")
    counts = Counter(owner.values())
    if isinstance(expected, Mapping):
        for split, want in expected.items():
            if counts.get(split, 0) != want:
                problems.append(f"split {split} has {counts.get(split, 0)} images, expected_counts says {want}")
    else:
        problems.append("missing key split.expected_counts")


def _concept_problems(cfg: Mapping[str, Any], problems: list[str]) -> None:
    ids = _get(cfg, "concepts.ids")
    groups = _get(cfg, "concepts.groups")
    if not isinstance(ids, list) or not ids:
        problems.append("concepts.ids must be a non-empty list")
        return
    if len(set(ids)) != len(ids):
        problems.append("concepts.ids must be unique")
    if not isinstance(groups, Mapping) or not groups:
        problems.append("concepts.groups must be a non-empty mapping")
        return
    grouped: list[str] = []
    for name, spec in groups.items():
        members = spec.get("ids") if isinstance(spec, Mapping) else None
        if not isinstance(members, list):
            problems.append(f"concepts.groups.{name}.ids must be a list")
            continue
        grouped.extend(members)
    repeated = sorted({c for c in grouped if grouped.count(c) > 1})
    if repeated:
        problems.append(f"concepts.groups put {repeated} in more than one group")
    uncovered = [c for c in ids if c not in grouped]
    if uncovered:
        problems.append(f"concepts.groups do not cover {uncovered}")
    unknown = [c for c in grouped if c not in ids]
    if unknown:
        problems.append(f"concepts.groups list ids missing from concepts.ids: {unknown}")
    for key in ("positives", "contrasts"):
        bounds = _get(cfg, f"concepts.{key}")
        if not (isinstance(bounds, list) and len(bounds) == 2 and all(_is_int(b) for b in bounds) and 1 <= bounds[0] <= bounds[1]):
            problems.append(f"concepts.{key} must be [min, max] phrasing counts with 1 <= min <= max")
    top_k = _get(cfg, "concepts.top_k")
    if not (_is_int(top_k) and top_k >= 1):
        problems.append("concepts.top_k must be a positive integer")
    if not _is_num(_get(cfg, "concepts.z_threshold")):
        problems.append("concepts.z_threshold must be a number")


def validate_settings(cfg: Mapping[str, Any]) -> None:
    """Raise ConfigError listing every problem; return None when the settings are valid."""
    problems = settings_problems(cfg)
    if problems:
        raise ConfigError(problems)


# ---- derived helpers -------------------------------------------------------------------------------------
def config_hash(cfg: Mapping[str, Any]) -> str:
    """sha256 hex digest of the canonical JSON of the settings (key order does not matter)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_repo_path(path: str | Path) -> Path:
    """Expand ~; relative paths resolve against the repo root."""
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else REPO_ROOT / p


def _env(env: Mapping[str, str] | None, name: str) -> str:
    source = os.environ if env is None else env
    return (source.get(name) or "").strip()


def detect_site(cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> str:
    """'delta' or 'local': $PATHQUEUE_SITE if set, else 'delta' when paths.delta.work_root exists."""
    site = _env(env, ENV_SITE)
    if site:
        if site not in SITES:
            raise ValueError(f"{ENV_SITE} must be one of {SITES}, got {site!r}")
        return site
    return "delta" if Path(cfg["paths"]["delta"]["work_root"]).is_dir() else "local"


def data_root(cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> Path:
    """Root of pipeline files (cases.csv, tiles.csv, emb/, ...). See the module docstring."""
    override = _env(env, ENV_DATA_ROOT)
    if override:
        return resolve_repo_path(override)
    if detect_site(cfg, env) == "delta":
        return Path(cfg["paths"]["delta"]["data_root"])
    return resolve_repo_path(cfg["paths"]["local"]["data_root"])


def artifacts_root(cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> Path:
    """Root of app-facing files (cases/, images/, cells/, eval/). See the module docstring."""
    override = _env(env, ENV_ARTIFACTS_ROOT)
    if override:
        return resolve_repo_path(override)
    if detect_site(cfg, env) == "delta":
        return Path(cfg["paths"]["delta"]["artifacts_root"])
    return resolve_repo_path(cfg["paths"]["local"]["artifacts_root"])


def fixtures_root(cfg: Mapping[str, Any]) -> Path:
    """Committed synthetic fixtures: case JSONs at <root>/<case_id>.json, media refs resolve against <root>."""
    return resolve_repo_path(cfg["paths"]["local"]["fixtures"])


def effective_mpp(cfg: Mapping[str, Any], inventory_path: str | Path | None = None) -> float:
    """BACH µm/px: inventory.json "measured_mpp" when present and not null, else settings dataset.mpp.

    A missing inventory file or a null value falls back to the default; a non-positive or non-numeric
    measured value is a corrupt inventory and raises ValueError (never silently replaced).
    """
    path = Path(inventory_path) if inventory_path is not None else schemas.inventory_json(data_root(cfg))
    default = float(cfg["dataset"]["mpp"])
    if not path.is_file():
        return default
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle).get(schemas.INVENTORY_MPP_KEY)
    if value is None:
        return default
    if not _is_num(value) or value <= 0:
        raise ValueError(f"{path}: {schemas.INVENTORY_MPP_KEY} must be a positive number, got {value!r}")
    return float(value)


def class_index(cfg: Mapping[str, Any]) -> dict[str, int]:
    """Class name -> column index in every probability vector."""
    return {name: i for i, name in enumerate(cfg["classes"])}


def aggressive_index(cfg: Mapping[str, Any]) -> int:
    return list(cfg["classes"]).index(cfg["aggressive_class"])


def concept_ids(cfg: Mapping[str, Any]) -> list[str]:
    return list(cfg["concepts"]["ids"])


def split_assignments(cfg: Mapping[str, Any]) -> list[str]:
    """Split name for each index 0..n_images-1 of the class-folder-sorted, file-name-sorted image list."""
    n_images = int(cfg["dataset"]["n_images"])
    out: list[str | None] = [None] * n_images
    for split, spans in cfg["split"]["ranges"].items():
        for start, end in spans:
            for index in range(start, end):
                out[index] = split
    missing = [i for i, s in enumerate(out) if s is None]
    if missing:
        raise ValueError(f"split.ranges leave indices unassigned: {missing[:5]}")
    return [s for s in out if s is not None]
