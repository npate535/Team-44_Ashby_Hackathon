"""Hosted CONCH/MUSK /embed equivalence check on a few tiles (BUILD_PLAN §A6, §E step 5; Agent 2).

    python -m pipeline.hosted_check --data-root /work/nvme/bfqi/teams/team44/pathqueue/smoke [--n 5]

Login-node safe: an HTTP client only (urllib, numpy, pandas, Pillow; never torch, never a model load). The hosted
servers are reachable only from inside Delta and run until Sun Oct 18 18:00 CDT. URLs and API keys come from the
environment or settings.hosted.env_file (mode 0600, KEY=VALUE lines); they are never printed or logged.

For each encoder (conch, musk): take the first n items of our job's per-item align embeddings for the
hosted_check.set (NCT whole tiles at view 0, so the per-item row is the un-augmented embedding), re-read the same
pixels (manifest path + crop box), send them as lossless PNG base64 to POST <url>/embed in batches of at most
settings.hosted.batch_size with retry + exponential backoff, and report the cosine between the hosted and our
embeddings (expect ~1). Exit 0 when every cosine >= hosted_check.min_cosine, else 1.
"""

from __future__ import annotations

import base64
import io as _io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from core import cli, schemas
from pipeline import encoders as E

STEP = "pipeline.hosted_check"


class HostedError(RuntimeError):
    pass


def read_env_file(path: str | Path) -> dict[str, str]:
    """KEY=VALUE lines ('export ' prefix and quotes allowed, '#' comments). Missing file -> {}."""
    path = Path(os.path.expanduser(str(path)))
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def endpoint(cfg: Mapping[str, Any], name: str, env: Mapping[str, str]) -> tuple[str, str]:
    """(url, key) for a hosted encoder from the environment, else the env file. Raises if unset (never prints)."""
    spec = cfg["hosted"][name]
    url, key = env.get(spec["url_env"], ""), env.get(spec["key_env"], "")
    if not url or not key:
        raise HostedError(f"{name}: {spec['url_env']} / {spec['key_env']} not set (environment or "
                          f"{cfg['hosted']['env_file']})")
    return url.rstrip("/"), key


def post_json(url: str, payload: Mapping[str, Any], key: str, timeout: float) -> Any:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed_hosted(url: str, key: str, images_b64: Sequence[str], *, batch: int, retries: int, backoff: float,
                 timeout: float, post: Callable[..., Any] = post_json, sleep: Callable[[float], None] = time.sleep
                 ) -> np.ndarray:
    out: list[list[float]] = []
    for start in range(0, len(images_b64), batch):
        chunk = list(images_b64[start:start + batch])
        for attempt in range(retries + 1):
            try:
                result = post(f"{url}/embed", {"images": chunk}, key, timeout)
                embs = result["embeddings"]
                if len(embs) != len(chunk):
                    raise HostedError(f"server returned {len(embs)} embeddings for {len(chunk)} images")
                out.extend(embs)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError) as exc:
                status = getattr(exc, "code", None)
                if attempt == retries or status in (400, 401, 403, 404):
                    raise HostedError(f"/embed failed after {attempt + 1} attempt(s): {type(exc).__name__} "
                                      f"{status or ''}".strip()) from None
                sleep(backoff * (2 ** attempt))
    return np.asarray(out, dtype=np.float32)


def tile_png_b64(root: Path, row: Mapping[str, Any]) -> str:
    with Image.open(schemas.resolve(root, str(row["path"]))) as handle:
        image = handle.convert("RGB")
    tile = E.crop_box(image, int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]))
    buf = _io.BytesIO()
    tile.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def pick_items(root: Path, encoder: str, set_name: str, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    """First n per-item rows with exactly one view (view 0, clean), joined with their manifest crop."""
    idx = pd.read_csv(schemas.emb_index_csv(root, encoder, set_name, "align"), keep_default_na=False,
                      dtype={"item_id": str, "degrade_type": str})
    arr = np.load(schemas.emb_npy(root, encoder, set_name, "align")).astype(np.float32)
    manifest = pd.read_csv(schemas.embed_manifest_csv(root), keep_default_na=False, dtype={"item_id": str, "path": str})
    view0 = manifest[manifest["view"].astype(int) == 0].drop_duplicates("item_id").set_index("item_id")
    ok = idx[(idx["n_views"] == 1) & (idx["degrade_type"] == "") & idx["item_id"].isin(view0.index)]
    ok = ok.sort_values("item_id").head(n)
    if ok.empty:
        raise HostedError(f"no single-view clean items in {encoder}/{set_name}")
    rows = view0.loc[ok["item_id"], ["path", "x", "y", "w", "h"]].reset_index()
    return rows, arr[ok["emb_row"].to_numpy()]


def check(root: Path, cfg: Mapping[str, Any], ecfg: Mapping[str, Any], encoders: Sequence[str], n: int,
          env: Mapping[str, str], post: Callable[..., Any] = post_json, log: Callable[[str], None] = print) -> bool:
    hc = ecfg["hosted_check"]
    all_ok = True
    for name in encoders:
        url, key = endpoint(cfg, name, env)
        rows, ours = pick_items(root, name, hc["set"], n)
        images = [tile_png_b64(root, r) for r in rows.to_dict("records")]
        hosted = embed_hosted(url, key, images, batch=int(cfg["hosted"]["batch_size"]), retries=int(hc["retries"]),
                              backoff=float(hc["backoff_s"]), timeout=float(hc["timeout_s"]), post=post)
        if hosted.shape != ours.shape:
            log(f"{name}: hosted shape {hosted.shape} != ours {ours.shape}")
            all_ok = False
            continue
        cos = (hosted * ours).sum(1) / np.maximum(np.linalg.norm(hosted, axis=1) * np.linalg.norm(ours, axis=1), 1e-12)
        ok = bool(cos.min() >= float(hc["min_cosine"]))
        all_ok &= ok
        log(f"{name}: {'PASS' if ok else 'FAIL'} hosted vs job align cosine over {len(cos)} tiles: "
            f"min {cos.min():.5f} mean {cos.mean():.5f} (need >= {hc['min_cosine']}); "
            f"items {rows['item_id'].tolist()}")
    return all_ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser(STEP, "Hosted /embed vs job align embeddings on a few tiles (login-node safe).")
    parser.add_argument("--encoders", default="conch,musk")
    parser.add_argument("--n", type=int, default=None, help="tiles per encoder (default hosted_check.n_tiles)")
    parser.add_argument("--encoders-config", default=None)
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    ecfg = E.load_encoders_config(args.encoders_config)
    env = {**read_env_file(ctx.cfg["hosted"]["env_file"]), **{k: v for k, v in os.environ.items() if v}}
    names = [s for s in args.encoders.split(",") if s]
    try:
        ok = check(ctx.data_root, ctx.cfg, ecfg, names, args.n or int(ecfg["hosted_check"]["n_tiles"]), env)
    except HostedError as exc:
        print(f"hosted_check: ERROR {exc}")
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
