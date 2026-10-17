"""02_embed end to end on CPU with the organizers' `test` adapter (views, chunking, resume, TTA, alignment, meta)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFilter

from core import config, io, schemas
from pipeline import encoders as E
from tests.embed_helpers import build_tree, embed, expected_test_embedding, noise_image, run_embed, make_encoders_yaml


@pytest.fixture(autouse=True)
def _fast_test_adapter(monkeypatch):
    monkeypatch.setenv("PATHOLOGY_TEST_ADAPTER_DELAY_S", "0")
    monkeypatch.setenv("PATHQUEUE_SITE", "local")


@pytest.fixture
def fake_degrade(monkeypatch):
    """Agent 1 implements pipeline.degrade; until then a deterministic stand-in with the frozen signature."""
    calls = []

    def apply(image, degrade_type, severity, seed, cfg):
        calls.append((degrade_type, severity, seed))
        return image.filter(ImageFilter.GaussianBlur(severity)) if degrade_type == "blur" else image.rotate(0)

    monkeypatch.setattr("pipeline.degrade.apply", apply)
    return calls


@pytest.fixture
def cfg():
    return config.load_settings()


def _load(root, set_name, head, views=False):
    arr = np.load(schemas.emb_npy(root, "test", set_name, head, views=views))
    idx = pd.read_csv(schemas.emb_index_csv(root, "test", set_name, head, views=views), keep_default_na=False,
                      dtype={"item_id": str, "split": str, "degrade_type": str, "label": str})
    return arr, idx


def test_end_to_end_tta_means_views_and_alignment(tmp_path, fake_degrade, cfg):
    manifest = build_tree(tmp_path)
    code, log = run_embed(tmp_path, make_encoders_yaml(tmp_path))
    assert code == 0, log
    for set_name in ("bach512", "nct", "deg512"):
        for head in ("align", "probe"):
            arr, idx = _load(tmp_path, set_name, head)
            sub = manifest[manifest["set"] == set_name]
            assert arr.dtype == np.float16 and arr.shape == (sub["item_id"].nunique(), 3)
            assert idx["emb_row"].tolist() == list(range(len(idx)))
            assert idx["item_id"].is_unique and set(idx["item_id"]) == set(sub["item_id"])
            assert (idx["view"] == -1).all()
            schemas.validate_frame(schemas.EMB_INDEX, idx)
    align, idx = _load(tmp_path, "bach512", "align")
    probe, _ = _load(tmp_path, "bach512", "probe")
    np.testing.assert_allclose(np.linalg.norm(align.astype(np.float32), axis=1), 1.0, atol=2e-3)
    # row i of the npy is item idx.item_id[i]: probe = raw mean over views, align = renormalized mean of unit views
    for i, item in enumerate(idx["item_id"]):
        rows = manifest[manifest["item_id"] == item]
        per_view = np.stack([expected_test_embedding(tmp_path, r) for _, r in rows.iterrows()])
        assert idx.loc[i, "n_views"] == len(rows)
        np.testing.assert_allclose(probe[i], per_view.mean(0), atol=2e-3)
        unit = per_view / np.linalg.norm(per_view, axis=1, keepdims=True)
        want = unit.mean(0) / np.linalg.norm(unit.mean(0))
        np.testing.assert_allclose(align[i].astype(np.float32), want, atol=2e-3)
    assert set(idx.loc[idx["split"] == "train", "n_views"]) == {3}
    assert set(idx.loc[idx["split"] != "train", "n_views"]) == {2}


def test_views_file_covers_every_split_and_matches_the_per_item_means(tmp_path, fake_degrade):
    manifest = build_tree(tmp_path)
    assert run_embed(tmp_path, make_encoders_yaml(tmp_path))[0] == 0
    varr, vidx = _load(tmp_path, "bach512", "probe", views=True)
    bach = manifest[manifest["set"] == "bach512"]
    assert len(vidx) == len(bach) == varr.shape[0]  # ruling M9: every split, one row per (item, view)
    assert set(vidx["split"]) == {"train", "calibration", "test"} and (vidx["n_views"] == 1).all()
    assert set(vidx.loc[vidx["split"] == "train", "view"]) == {0, 1, 2}
    assert set(vidx.loc[vidx["split"] == "test", "view"]) == {0, 1}
    arr, idx = _load(tmp_path, "bach512", "probe")
    for item, group in vidx.groupby("item_id"):
        row = idx.index[idx["item_id"] == item][0]
        np.testing.assert_allclose(arr[row], varr[group["emb_row"]].astype(np.float32).mean(0), atol=2e-3)
    assert not schemas.emb_npy(tmp_path, "test", "nct", "probe", views=True).exists()  # one view only


def test_degraded_rows_call_degrade_apply_with_manifest_params(tmp_path, fake_degrade):
    build_tree(tmp_path)
    assert run_embed(tmp_path, make_encoders_yaml(tmp_path), "--sets", "deg512")[0] == 0
    assert sorted(fake_degrade) == [("blur", 2, 7), ("jpeg", 2, 7)]
    _, idx = _load(tmp_path, "deg512", "align")
    assert set(idx["degrade_type"]) == {"blur", "jpeg"} and set(idx["degrade_severity"]) == {2}


def test_meta_json_matches_the_schema_and_records_checks(tmp_path, fake_degrade):
    build_tree(tmp_path)
    assert run_embed(tmp_path, make_encoders_yaml(tmp_path))[0] == 0
    meta = io.read_json(schemas.emb_meta_json(tmp_path, "test"))
    assert schemas.EMB_META.problems(meta) == []
    assert meta["dtype"] == "float16" and meta["heads"]["align"] == {"dim": 3, "normalized": True}
    assert meta["sets"]["bach512"]["n_items"] == 16 and meta["sets"]["bach512"]["n_views"] == 3
    assert meta["sets"]["bach512"]["views_file"] is True and meta["sets"]["nct"]["views_file"] is False
    assert meta["checks"]["bach512"]["parity_min_cosine"]["align"] > 0.999
    assert meta["checks"]["nct"]["nct_zero_shot"]["status"] == "skipped"  # 6 items < nct_min_items
    assert meta["testing_only"] is True and meta["synthetic"] is False
    assert schemas.STEP_META.problems(io.read_json(schemas.step_meta_json(tmp_path, "02_embed"))) == []


def test_chunked_run_resumes_after_an_interrupt(tmp_path, fake_degrade, monkeypatch):
    manifest = build_tree(tmp_path)
    ecfg = make_encoders_yaml(tmp_path, chunk_rows=5)
    n_bach = int((manifest["set"] == "bach512").sum())  # 40 rows -> 8 chunks of 5
    real_embed = E.Encoder.embed
    seen = {"rows": 0}

    def flaky_embed(self, batch, **kw):
        if seen["rows"] >= 15:
            raise KeyboardInterrupt("simulated job kill")
        seen["rows"] += int(batch.shape[0])
        return real_embed(self, batch, **kw)

    monkeypatch.setattr(E.Encoder, "embed", flaky_embed)
    with pytest.raises(KeyboardInterrupt):
        run_embed(tmp_path, ecfg, "--sets", "bach512")
    chunks = sorted((schemas.emb_dir(tmp_path, "test") / ".chunks" / "bach512").glob("*.npz"))
    assert len(chunks) == 3 and not schemas.emb_npy(tmp_path, "test", "bach512", "align").exists()

    counted = {"rows": 0}

    def counting_embed(self, batch, **kw):
        counted["rows"] += int(batch.shape[0])
        return real_embed(self, batch, **kw)

    monkeypatch.setattr(E.Encoder, "embed", counting_embed)
    code, _ = run_embed(tmp_path, ecfg, "--sets", "bach512")
    assert code == 0
    assert counted["rows"] == n_bach - 15  # only the 5 chunks the killed run did not save
    full = tmp_path / "fresh"
    build_tree(full)
    monkeypatch.setattr(E.Encoder, "embed", real_embed)
    assert run_embed(full, make_encoders_yaml(tmp_path, chunk_rows=5), "--sets", "bach512")[0] == 0
    np.testing.assert_array_equal(_load(tmp_path, "bach512", "probe")[0], _load(full, "bach512", "probe")[0])
    assert not (schemas.emb_dir(tmp_path, "test") / ".chunks" / "bach512").exists()  # cleaned after finalize


def test_finished_sets_are_skipped_and_force_recomputes(tmp_path, fake_degrade, monkeypatch):
    build_tree(tmp_path)
    ecfg = make_encoders_yaml(tmp_path)
    assert run_embed(tmp_path, ecfg)[0] == 0
    loads = []
    real = E.load_encoder
    monkeypatch.setattr(E, "load_encoder", lambda *a, **k: loads.append(a) or real(*a, **k))
    code, log = run_embed(tmp_path, ecfg)
    assert code == 0 and loads == [] and any("already embedded" in line for line in log)
    code, _ = run_embed(tmp_path, ecfg, "--sets", "nct", "--force")
    assert code == 0 and len(loads) == 1


def test_limit_selects_first_cases_and_never_shrinks_a_finished_set(tmp_path, fake_degrade):
    build_tree(tmp_path)
    ecfg = make_encoders_yaml(tmp_path)
    assert run_embed(tmp_path, ecfg, "--sets", "bach512", "--limit", "2")[0] == 0
    _, idx = _load(tmp_path, "bach512", "align")
    assert sorted(set(idx["case_id"])) == ["BACH_0001", "BACH_0002"]
    assert run_embed(tmp_path, ecfg, "--sets", "bach512")[0] == 0
    assert len(_load(tmp_path, "bach512", "align")[1]) == 16
    code, log = run_embed(tmp_path, ecfg, "--sets", "bach512", "--limit", "1")
    assert code == 0 and len(_load(tmp_path, "bach512", "align")[1]) == 16


def test_nct_zero_shot_below_the_floor_aborts(tmp_path, fake_degrade):
    build_tree(tmp_path)
    code, log = run_embed(tmp_path, make_encoders_yaml(tmp_path, nct_min_items=1, nct_floor=1.01), "--sets", "nct")
    assert code == 2
    assert any("zero-shot" in line and "misaligned" in line for line in log)
    meta = io.read_json(schemas.emb_meta_json(tmp_path, "test"))
    assert meta["checks"]["nct"]["nct_zero_shot"]["status"] == "fail"


def test_wrong_head_dimension_aborts(tmp_path, fake_degrade):
    build_tree(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        run_embed(tmp_path, make_encoders_yaml(tmp_path, heads={"align": {"dim": 4, "normalized": True},
                                                                "probe": {"dim": 3, "normalized": False}}))


def test_parity_failure_aborts_with_a_clear_message(tmp_path, fake_degrade, monkeypatch):
    build_tree(tmp_path)
    real = E._load_official

    def broken(name, spec, device):
        enc = real(name, spec, device)
        enc.reference = lambda x: {"align": -enc.embed(x)["align"]}
        return enc

    monkeypatch.setattr(E, "_load_official", broken)
    code, log = run_embed(tmp_path, make_encoders_yaml(tmp_path), "--sets", "nct")
    assert code == 2 and any("official adapter" in line for line in log)


def test_nan_embeddings_abort(tmp_path, fake_degrade, monkeypatch):
    build_tree(tmp_path)
    real_embed = E.Encoder.embed

    def nan_embed(self, batch, **kw):
        out = real_embed(self, batch, **kw)
        out["probe"][0, 0] = np.nan
        return out

    monkeypatch.setattr(E.Encoder, "embed", nan_embed)
    monkeypatch.setattr(embed, "run_parity_check", lambda enc, b, out, m, log: (out, {}))
    code, log = run_embed(tmp_path, make_encoders_yaml(tmp_path), "--sets", "nct")
    assert code == 2 and any("NaN" in line for line in log)


def test_unreadable_image_is_fatal(tmp_path, fake_degrade):
    build_tree(tmp_path)
    (tmp_path / "nct" / "ADI" / "ADI-T0.png").write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match="cannot read image"):
        run_embed(tmp_path, make_encoders_yaml(tmp_path), "--sets", "nct")


def test_missing_manifest_and_unknown_sets_exit_non_zero(tmp_path):
    ecfg = make_encoders_yaml(tmp_path)
    assert run_embed(tmp_path, ecfg)[0] == 2
    build_tree(tmp_path)
    assert run_embed(tmp_path, ecfg, "--sets", "bogus")[0] == 2


def test_manifest_with_bad_views_is_rejected(tmp_path):
    df = build_tree(tmp_path)
    df.loc[0, "view"] = 9
    df.to_csv(schemas.embed_manifest_csv(tmp_path), index=False)
    with pytest.raises(ValueError, match="view"):
        embed.read_manifest(schemas.embed_manifest_csv(tmp_path))


def test_smoke_manifest_needs_no_agent1_outputs(tmp_path, cfg):
    nct = tmp_path / "nct"
    for k, cls in enumerate(cfg["nct"]["classes"]):
        for j in range(3):
            noise_image(nct / cls / f"{cls}-X{j}.tif", 24, 24, seed=10 * k + j)
    bach = tmp_path / "bach"
    noise_image(bach / "Benign" / "b001.tif", 1100, 600, seed=1)
    root = tmp_path / "smoke"
    lines = []
    code = embed.main(["--data-root", str(root), "--smoke-manifest", "--nct-dir", str(nct), "--bach-dir", str(bach)],
                      log=lines.append)
    assert code == 0, lines
    df = embed.read_manifest(schemas.embed_manifest_csv(root))
    assert (df["set"] == "nct").sum() == 27 and set(df.loc[df["set"] == "nct", "label"]) == set(cfg["nct"]["classes"])
    bach_rows = df[df["set"] == "bach512"]
    assert len(bach_rows) == 2 * 1 * 2  # 1x2 tiles of 512 fit in 1100x600, 2 views each
    assert set(bach_rows["view"]) == {0, 1} and set(bach_rows["split"]) == {"test"}
    # degraded rows appear once pipeline.degrade is implemented (Agent 1); until then the gap is noted
    assert (df["set"] == "deg512").any() or any("degrade" in line for line in lines)


# ---- hosted /embed equivalence check (HTTP faked; never a network call) ------------------------------------------
from pipeline import hosted_check as H  # noqa: E402


def _hosted_cfg(cfg):
    cfg = dict(cfg)
    cfg["hosted"] = {**cfg["hosted"], "test": {"url_env": "TEST_URL", "key_env": "TEST_KEY"}}
    return cfg


def test_hosted_check_passes_when_the_server_matches_our_embeddings(tmp_path, fake_degrade, cfg):
    build_tree(tmp_path)
    assert run_embed(tmp_path, make_encoders_yaml(tmp_path), "--sets", "nct")[0] == 0
    ecfg = E.load_encoders_config(make_encoders_yaml(tmp_path))
    arr, idx = _load(tmp_path, "nct", "align")
    by_png = {}
    manifest = embed.read_manifest(schemas.embed_manifest_csv(tmp_path))
    for _, row in manifest[manifest["set"] == "nct"].iterrows():
        by_png[H.tile_png_b64(tmp_path, row)] = arr[idx.index[idx["item_id"] == row["item_id"]][0]].astype(float)
    sent = []

    def post(url, payload, key, timeout):
        sent.append((url, key, len(payload["images"])))
        return {"embeddings": [by_png[b].tolist() for b in payload["images"]]}

    lines = []
    env = {"TEST_URL": "http://hosted.example/", "TEST_KEY": "k"}
    assert H.check(tmp_path, _hosted_cfg(cfg), ecfg, ["test"], 5, env, post=post, log=lines.append)
    assert sent == [("http://hosted.example/embed", "k", 5)] and "PASS" in lines[0] and "k" not in lines[0]

    def wrong(url, payload, key, timeout):
        return {"embeddings": [[1.0, -1.0, 0.0]] * len(payload["images"])}

    assert not H.check(tmp_path, _hosted_cfg(cfg), ecfg, ["test"], 5, env, post=wrong, log=lines.append)


def test_hosted_retries_with_backoff_then_gives_up_on_auth_errors():
    import urllib.error

    calls, sleeps = [], []

    def flaky(url, payload, key, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("busy")
        return {"embeddings": [[1.0]] * len(payload["images"])}

    out = H.embed_hosted("u", "k", ["a"] * 40, batch=32, retries=4, backoff=2.0, timeout=1, post=flaky,
                         sleep=sleeps.append)
    assert out.shape == (40, 1) and sleeps == [2.0, 4.0] and len(calls) == 4  # 2 failures, then 2 batches

    def unauthorized(url, payload, key, timeout):
        raise urllib.error.HTTPError(url, 401, "no", {}, None)

    with pytest.raises(H.HostedError, match="401") as err:
        H.embed_hosted("u", "secret-key", ["a"], batch=32, retries=4, backoff=1, timeout=1, post=unauthorized,
                       sleep=sleeps.append)
    assert "secret-key" not in str(err.value)


def test_env_file_parsing_and_missing_endpoints(tmp_path, cfg):
    path = tmp_path / "servers.env"
    path.write_text("# comment\nexport CONCH_URL='http://x:8000'\nCONCH_API_KEY=\"abc\"\nBROKEN\n")
    env = H.read_env_file(path)
    assert env == {"CONCH_URL": "http://x:8000", "CONCH_API_KEY": "abc"}
    assert H.endpoint(cfg, "conch", env) == ("http://x:8000", "abc")
    with pytest.raises(H.HostedError, match="MUSK_URL") as err:
        H.endpoint(cfg, "musk", env)
    assert "abc" not in str(err.value)
    assert H.read_env_file(tmp_path / "absent.env") == {}


def test_max_views_keeps_view_zero_only(tmp_path, fake_degrade):
    build_tree(tmp_path)
    assert run_embed(tmp_path, make_encoders_yaml(tmp_path), "--sets", "bach512", "--max-views", "bach512=1")[0] == 0
    _, idx = _load(tmp_path, "bach512", "probe")
    assert len(idx) == 16 and set(idx["n_views"]) == {1}
    assert not schemas.emb_npy(tmp_path, "test", "bach512", "probe", views=True).exists()
    with pytest.raises(ValueError, match="max-views"):
        embed.parse_max_views("bach512=0")


def test_a_failed_zero_shot_set_is_never_treated_as_done(tmp_path, fake_degrade):
    build_tree(tmp_path)
    ecfg = make_encoders_yaml(tmp_path, nct_min_items=1, nct_floor=1.01)
    assert run_embed(tmp_path, ecfg, "--sets", "nct")[0] == 2
    code, log = run_embed(tmp_path, ecfg, "--sets", "nct")  # no --force: must re-run the check, not skip
    assert code == 2 and not any("already embedded" in line for line in log)


def test_changed_manifest_geometry_recomputes_the_set(tmp_path, fake_degrade):
    df = build_tree(tmp_path)
    ecfg = make_encoders_yaml(tmp_path)
    assert run_embed(tmp_path, ecfg, "--sets", "bach512")[0] == 0
    before = _load(tmp_path, "bach512", "probe")[0].copy()
    df.loc[df["set"] == "bach512", "x"] = 0  # every tile now crops the left column
    df.to_csv(schemas.embed_manifest_csv(tmp_path), index=False)
    code, log = run_embed(tmp_path, ecfg, "--sets", "bach512")
    assert code == 0 and not any("already embedded" in line for line in log)
    assert not np.array_equal(before, _load(tmp_path, "bach512", "probe")[0])


def test_gpu_allocation_without_cuda_aborts_instead_of_running_on_cpu(tmp_path, monkeypatch):
    build_tree(tmp_path)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    lines = []
    code = embed.main(["--data-root", str(tmp_path), "--encoders", "test", "--encoders-config",
                       str(make_encoders_yaml(tmp_path))], log=lines.append)
    assert code == 2 and "no CUDA device" in lines[-1]
    with pytest.raises(embed.CheckFailed):
        embed.resolve_device("cuda:0")


def test_import_errors_are_failures_not_unavailable(tmp_path, fake_degrade, monkeypatch):
    build_tree(tmp_path)

    def broken(*a, **k):
        raise ImportError("No module named 'fairscale'")

    monkeypatch.setattr(E.official_models(), "resolve_adapter", broken)
    code, log = run_embed(tmp_path, make_encoders_yaml(tmp_path), "--sets", "nct", "--skip-unavailable",
                          "--keep-going")
    assert code == 2 and any("ERROR test" in line and "fairscale" in line for line in log)


def test_all_encoders_unavailable_exits_non_zero(tmp_path, fake_degrade, monkeypatch):
    build_tree(tmp_path)

    def unavailable(name, ecfg, device="cuda"):
        raise E.EncoderUnavailable(f"{name}: not in the HF cache")

    monkeypatch.setattr(E, "load_encoder", unavailable)
    code, log = run_embed(tmp_path, make_encoders_yaml(tmp_path), "--sets", "nct", "--skip-unavailable")
    assert code == 2 and any("no encoder was available" in line for line in log)


def test_dataloader_workers_fork_on_linux_and_run_in_process_elsewhere(monkeypatch):
    monkeypatch.setattr(embed.sys, "platform", "linux")
    n, ctx = embed.loader_workers(14)
    assert n == 14 and ctx.get_start_method() == "fork"
    monkeypatch.setattr(embed.sys, "platform", "darwin")
    assert embed.loader_workers(14) == (0, None)
    assert embed.loader_workers(0) == (0, None)
