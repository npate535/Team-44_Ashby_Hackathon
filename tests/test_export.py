"""pipeline.export: what is packed (and never packed), sha256 manifest, verified unpack, unsafe archives."""

from __future__ import annotations

import io as _io
import json
import tarfile

import numpy as np
import pytest

from pipeline import export as X


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    monkeypatch.setenv("PATHQUEUE_SITE", "local")


def _tree(root):
    files = {
        "cases.csv": "case_id\nBACH_0001\n", "tiles.csv": "a\n", "inventory.json": "{}", "tile_risk_oof.csv": "x\n",
        "cell_features.csv": "c\n", "embed_manifest.csv": "big\n",
        "emb/conch/bach512_align_index.csv": "i\n", "emb/conch/bach512_align_views_index.csv": "v\n",
        "emb/conch/meta.json": "{}", "emb/musk/nct_probe_index.csv": "m\n",
        "meta/02_embed.json": "{}", "display/BACH_0001.jpg": "jpg", "cells/BACH_0001/3.json": "{}",
        "tiles/g512/BACH_0001/0_0.png": "png", "smoke/emb/conch/nct_align.npy": "s", "logs/x.out": "log",
        "emb/conch/.chunks/bach512/00000_abc.npz": "chunk",
    }
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
    for rel in ("emb/conch/bach512_align.npy", "emb/conch/bach512_align_views.npy", "emb/musk/nct_probe.npy"):
        np.save(root / rel, np.ones((4, 3), np.float16))
    (root / "text").mkdir()
    np.savez(root / "text" / "conch.npz", ids=np.array(["a"]))


def test_collect_packs_per_item_arrays_and_laptop_inputs_only(tmp_path):
    _tree(tmp_path)
    rel = {p.as_posix() for p in X.collect(tmp_path)}
    assert {"cases.csv", "tiles.csv", "inventory.json", "tile_risk_oof.csv", "cell_features.csv",
            "emb/conch/bach512_align.npy", "emb/conch/bach512_align_index.csv", "emb/conch/meta.json",
            "emb/musk/nct_probe.npy", "emb/musk/nct_probe_index.csv", "text/conch.npz", "meta/02_embed.json", "display/BACH_0001.jpg",
            "cells/BACH_0001/3.json"} == rel
    with_views = {p.as_posix() for p in X.collect(tmp_path, with_views=True, with_manifest=True, encoders=["conch"])}
    assert "emb/conch/bach512_align_views.npy" in with_views and "embed_manifest.csv" in with_views
    assert not any(p.startswith("emb/musk") for p in with_views)


def test_dry_run_prints_the_size_and_writes_nothing(tmp_path):
    _tree(tmp_path)
    lines = []
    assert X.main(["--data-root", str(tmp_path), "--dry-run"], log=lines.append) == 0
    assert "14 files" in lines[0] and "MB uncompressed" in lines[0]
    assert not (tmp_path / "exports").exists()


def test_pack_then_unpack_round_trips_with_verified_hashes(tmp_path):
    src, dest = tmp_path / "delta", tmp_path / "laptop"
    src.mkdir()
    _tree(src)
    out = tmp_path / "x.tgz"
    assert X.main(["--data-root", str(src), "--out", str(out)], log=lambda s: None) == 0
    with tarfile.open(out) as tar:
        names = tar.getnames()
        manifest = json.load(tar.extractfile(X.MANIFEST_NAME))
    assert names[:2] == [X.MANIFEST_NAME, X.SUMS_NAME] and manifest["n_files"] == 14
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])
    lines = []
    assert X.main(["--data-root", str(dest), "--unpack", str(out)], log=lines.append) == 0
    assert "sha256 verified" in lines[-1]
    np.testing.assert_array_equal(np.load(dest / "emb/conch/bach512_align.npy"), np.ones((4, 3), np.float16))
    assert (dest / "display/BACH_0001.jpg").read_text() == "jpg"
    assert not list(dest.glob(".unpack_*")) and (dest / "meta" / "export_unpack.json").is_file()


def _tar_with(path, members, manifest=None):
    with tarfile.open(path, "w:gz") as tar:
        if manifest is not None:
            data = json.dumps(manifest).encode()
            info = tarfile.TarInfo(X.MANIFEST_NAME)
            info.size = len(data)
            tar.addfile(info, _io.BytesIO(data))
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, _io.BytesIO(content))


def test_tampered_archive_is_rejected_and_dest_left_unchanged(tmp_path):
    manifest = {"n_files": 1, "total_bytes": 3, "files": [{"path": "cases.csv", "bytes": 3,
                                                            "sha256": X.hashlib.sha256(b"abc").hexdigest()}]}
    bad = tmp_path / "bad.tgz"
    _tar_with(bad, [("cases.csv", b"abd")], manifest)
    dest = tmp_path / "dest"
    lines = []
    assert X.main(["--data-root", str(dest), "--unpack", str(bad)], log=lines.append) == 2
    assert "mismatch" in lines[-1] and not (dest / "cases.csv").exists()


@pytest.mark.parametrize("name", ["../evil.csv", "/abs/evil.csv", "emb/../../evil"])
def test_unsafe_member_paths_are_refused(tmp_path, name):
    bad = tmp_path / "bad.tgz"
    _tar_with(bad, [(name, b"x")], {"n_files": 0, "total_bytes": 0, "files": []})
    lines = []
    assert X.main(["--data-root", str(tmp_path / "d"), "--unpack", str(bad)], log=lines.append) == 2
    assert "unsafe" in lines[-1]


def test_extra_files_not_in_the_manifest_are_refused(tmp_path):
    bad = tmp_path / "bad.tgz"
    _tar_with(bad, [("planted.py", b"x")], {"n_files": 0, "total_bytes": 0, "files": []})
    lines = []
    assert X.main(["--data-root", str(tmp_path / "d"), "--unpack", str(bad)], log=lines.append) == 2
    assert "differ from the manifest" in lines[-1]
