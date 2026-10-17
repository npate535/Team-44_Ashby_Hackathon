"""Playwright end-to-end fixtures (Agent 5). Skipped unless PQ_E2E=1 and Chromium is installed (`make e2e`;
install once with `uv run playwright install chromium`). Starts the backend on a free port over a fresh
backend.dev_cases root (synthetic, generated non-tissue media) and saves screenshots to docs/screenshots/."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from core import config

REPO = config.REPO_ROOT
SHOTS = REPO / "docs" / "screenshots"
VIEWPORTS = {"desktop": {"width": 1280, "height": 900}, "phone": {"width": 390, "height": 844}}

if os.environ.get("PQ_E2E") != "1":
    pytest.skip("browser e2e tests are opt-in: run `make e2e` (PQ_E2E=1)", allow_module_level=True)
sync_api = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as res:
        return json.loads(res.read())


def _post(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, method="POST", data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as res:
        return json.loads(res.read())


@pytest.fixture(scope="session")
def dev_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("e2e") / "dev"
    subprocess.run([sys.executable, "-m", "backend.dev_cases", "--out", str(root), "--with-eval",
                    "--mean-interarrival-s", "6", "--fill", "4"], cwd=REPO, check=True, capture_output=True)
    return root


@pytest.fixture(scope="session")
def roles(dev_root) -> dict[str, str]:
    """role -> case_id from the dev files' eval_only.role (test-only knowledge, never sent to the UI)."""
    return {json.loads(p.read_text())["eval_only"]["role"]: p.stem for p in (dev_root / "cases").glob("*.json")}


@pytest.fixture(scope="session")
def server(dev_root):
    port = _free_port()
    env = {**os.environ, "PATHQUEUE_ARTIFACTS_ROOT": str(dev_root), "PATHQUEUE_CASES_DIR": ""}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1",
                             "--port", str(port)], cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if _get(base + "/api/health")["loaded"]:
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("backend did not start: " + proc.stdout.read().decode()[-2000:])
    yield base
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture()
def api(server):
    class Api:
        base = server

        def get(self, path):
            return _get(server + path)

        def post(self, path, body=None):
            return _post(server + path, body)
    return Api()


@pytest.fixture(scope="session")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - any launch failure means "no browser installed"
            pytest.skip(f"Chromium not available ({exc.__class__.__name__}); run `uv run playwright install chromium`")
        yield b
        b.close()


class Page:
    """A page plus everything the leak checks need: every /api JSON body and console/page errors."""

    def __init__(self, page, base):
        self.page, self.base = page, base
        self.api_bodies: list[tuple[str, str]] = []
        self.errors: list[str] = []
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: self.errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.on("response", self._response)

    def _response(self, res):
        if "/api/" in res.url or res.url.endswith(".json"):
            try:
                self.api_bodies.append((res.url, res.text()))
            except Exception:  # noqa: BLE001 - redirects/aborted bodies have no text
                pass

    def shot(self, name: str) -> None:
        SHOTS.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)

    def assert_no_ground_truth(self) -> None:
        for url, body in self.api_bodies:
            assert "eval_only" not in body and "true_label" not in body, url
        html = self.page.content()
        assert "eval_only" not in html and "true_label" not in html

    def assert_no_horizontal_scroll(self) -> None:
        assert self.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")


@pytest.fixture(params=["desktop", "phone"])
def page(request, browser, server, api):
    api.post("/api/sim/reset")
    scheme = "dark" if request.param == "phone" else "light"
    ctx = browser.new_context(viewport=VIEWPORTS[request.param], color_scheme=scheme)
    pg = ctx.new_page()
    pg.set_default_timeout(15000)
    wrapped = Page(pg, server)
    wrapped.name = f"{request.param}-{scheme}"
    yield wrapped
    ctx.close()
    assert wrapped.errors == [], wrapped.errors
