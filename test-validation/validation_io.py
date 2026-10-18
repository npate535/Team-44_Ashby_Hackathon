"""Shared output helpers for tests/validation/.

Every strategy writes through a RunContext so that CSVs and JPG charts land in
one predictable tree, carry run metadata, and are listed in a checksummed
manifest:

    results/<run_id>/
        csv/      tidy, long-format CSVs (UTF-8)
        charts/   .jpg figures (200 dpi, quality 95, white background)
        manifest.csv

Rules enforced here:
  * Empty results are an error. Missing evaluations stay unavailable rather
    than becoming placeholder files (matches the README's policy).
  * Synthetic workspaces cannot produce real-data results.
  * Simulated outputs (Strategy 6) are watermarked on every chart.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLOCKED_PATH_MARKERS = ("data_synthetic", "artifacts/qa")
JPG_DPI = 200
JPG_QUALITY = 95
FLOAT_FMT = "%.6f"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class RunContext:
    out_root: Path = Path("results")
    seed: int = 0
    simulated: bool = False          # True for Strategy 6
    data_root: str = "data"          # checked against blocked markers
    run_id: str = field(default_factory=lambda: dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    commit: str = field(default_factory=_git_commit)
    _manifest: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if any(m in str(self.data_root) for m in BLOCKED_PATH_MARKERS):
            raise ValueError(
                f"Refusing to write validation results from synthetic data: {self.data_root}"
            )
        self.out_root = Path(self.out_root)
        self.csv_dir = self.out_root / self.run_id / "csv"
        self.chart_dir = self.out_root / self.run_id / "charts"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.chart_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ CSV
    def write_csv(
        self,
        name: str,
        df: pd.DataFrame,
        required_cols: Sequence[str] | None = None,
    ) -> Path:
        """Write a tidy CSV with run metadata columns prepended."""
        if df is None or len(df) == 0:
            raise ValueError(f"{name}: empty result; not writing a placeholder file")
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{name}: missing required columns {missing}")
        out = df.copy()
        meta = {"run_id": self.run_id, "commit": self.commit, "seed": self.seed}
        for i, (k, v) in enumerate(meta.items()):
            if k not in out.columns:
                out.insert(i, k, v)
        if self.simulated and "simulated" not in out.columns:
            out["simulated"] = True
        path = self.csv_dir / (name if name.endswith(".csv") else f"{name}.csv")
        out.to_csv(path, index=False, float_format=FLOAT_FMT, encoding="utf-8")
        self._record(path, "csv", len(out))
        return path

    # ------------------------------------------------------------------ JPG
    def save_jpg(self, fig: plt.Figure, name: str, title: str | None = None) -> Path:
        """Save a figure as .jpg with a provenance footer (and SIMULATED mark)."""
        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold")
        footer = f"run {self.run_id} | commit {self.commit} | seed {self.seed}"
        fig.text(0.99, 0.005, footer, ha="right", va="bottom", fontsize=6, color="#666666")
        if self.simulated:
            fig.text(
                0.5, 0.5, "SIMULATED - NOT A CLINICAL CLAIM", ha="center", va="center",
                fontsize=28, color="#999999", alpha=0.18, rotation=25,
            )
        fig.patch.set_facecolor("white")
        path = self.chart_dir / (name if name.endswith(".jpg") else f"{name}.jpg")
        fig.savefig(
            path, format="jpg", dpi=JPG_DPI, bbox_inches="tight",
            pil_kwargs={"quality": JPG_QUALITY},
        )
        plt.close(fig)
        self._record(path, "jpg", None)
        return path

    # ------------------------------------------------------------- manifest
    def _record(self, path: Path, kind: str, rows: int | None) -> None:
        self._manifest.append(
            {
                "file": str(path.relative_to(self.out_root / self.run_id)),
                "kind": kind,
                "rows": rows if rows is not None else "",
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    def write_manifest(self) -> Path:
        path = self.out_root / self.run_id / "manifest.csv"
        pd.DataFrame(self._manifest).to_csv(path, index=False)
        return path


# =========================================================== chart helpers
def forest_plot(
    df: pd.DataFrame, label: str, est: str, lo: str, hi: str,
    ref: float = 0.0, xlabel: str = "Difference (95% CI)",
) -> plt.Figure:
    """Paired-difference forest plot. Rows whose CI excludes `ref` are bold."""
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(df) + 1.5))
    y = np.arange(len(df))[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        sig = not (r[lo] <= ref <= r[hi])
        ax.errorbar(
            r[est], yi, xerr=[[r[est] - r[lo]], [r[hi] - r[est]]],
            fmt="o", color="#1f4e79" if sig else "#7f7f7f", capsize=3,
        )
    ax.axvline(ref, color="black", lw=0.8, ls="--")
    ax.set_yticks(y, df[label])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    return fig


def bar_ci(
    df: pd.DataFrame, x: str, y: str, lo: str | None = None, hi: str | None = None,
    hue: str | None = None, ylabel: str | None = None,
) -> plt.Figure:
    """Grouped bar chart with optional CI whiskers."""
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = list(df[hue].unique()) if hue else [None]
    cats = list(df[x].unique())
    w = 0.8 / len(groups)
    for i, g in enumerate(groups):
        sub = df if g is None else df[df[hue] == g]
        sub = sub.set_index(x).reindex(cats)
        pos = np.arange(len(cats)) + i * w
        err = None
        if lo and hi:
            err = [sub[y] - sub[lo], sub[hi] - sub[y]]
        ax.bar(pos, sub[y], w, yerr=err, capsize=3, label=g)
    ax.set_xticks(np.arange(len(cats)) + w * (len(groups) - 1) / 2, cats, rotation=20)
    ax.set_ylabel(ylabel or y)
    if hue:
        ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    return fig


def line_ci(
    df: pd.DataFrame, x: str, y: str, hue: str,
    lo: str | None = None, hi: str | None = None,
    xlabel: str | None = None, ylabel: str | None = None,
    hline: float | None = None,
) -> plt.Figure:
    """Line chart per group with shaded CI band (degradation, label efficiency)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, sub in df.groupby(hue):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o", label=str(g))
        if lo and hi:
            ax.fill_between(sub[x], sub[lo], sub[hi], alpha=0.2)
    if hline is not None:
        ax.axhline(hline, color="black", ls="--", lw=0.8)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    return fig


def hist_by_group(
    df: pd.DataFrame, value: str, hue: str, vline: float | None = None,
    xlabel: str | None = None,
) -> plt.Figure:
    """Overlaid histograms per group (e.g., coverage across resplits)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, sub in df.groupby(hue):
        ax.hist(sub[value], bins=30, alpha=0.5, label=str(g))
    if vline is not None:
        ax.axvline(vline, color="black", ls="--", lw=1, label="target 1-alpha")
    ax.set_xlabel(xlabel or value)
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    return fig


def confusion_heatmap(cm: np.ndarray, labels: Iterable[str]) -> plt.Figure:
    labels = list(labels)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig
