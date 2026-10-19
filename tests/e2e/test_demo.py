"""plan §14 demo script end to end in a real browser, at desktop (light) and 390 px (dark)."""

from __future__ import annotations

import re

from playwright.sync_api import expect

from core import config

REPLAY_S = config.load_settings()["sim"]["max_replay_wall_s"] or 120

ROW = "[data-testid=worklist-row]"


def tid(name: str) -> str:
    return f"[data-testid={name}]"


def start_and_wait_for_biopsies(pg, api, roles) -> None:
    pg.goto(pg_url(pg))
    expect(pg.locator(tid("state-empty"))).to_be_visible()
    pg.click(tid("sim-start"))
    hero = roles["hero_biopsy"]
    # the replay window bounds the last arrival (settings sim.max_replay_wall_s)
    expect(pg.locator(f"{ROW}[data-case-id='{hero}']")).to_be_visible(timeout=(REPLAY_S + 20) * 1000)


def pg_url(pg) -> str:
    return pg._base + "/"


def test_demo_script(page, api, roles):
    pg = page.page
    pg._base = page.base
    hero = roles["hero_biopsy"]

    # 1. Start the simulator; biopsies arrive; the hero (arriving last) sits in tier 2.
    start_and_wait_for_biopsies(pg, api, roles)
    hero_row = pg.locator(f"{ROW}[data-case-id='{hero}']")
    expect(hero_row).to_have_attribute("data-tier", "2")
    # every live reason is on the row; the Flags column shows only those the Suggested column does not already say
    expect(hero_row).to_have_attribute("data-reasons", re.compile(r"\baggressive_in_conformal_set\b"))
    expect(hero_row.locator(tid("orders-drafted"))).to_be_visible()
    expect(pg.locator(tid("queue-stats"))).to_be_visible()
    expect(pg.locator(tid("open-next"))).to_be_enabled()
    expect(pg.locator(tid("held-surgical"))).to_be_visible()
    ids_ai = pg.locator(ROW).evaluate_all("rows => rows.map(r => r.dataset.caseId)")
    pg.click(tid("order-arrival"))
    expect(pg.locator(tid("order-arrival"))).to_have_attribute("aria-pressed", "true")
    pg.wait_for_function("() => document.querySelector('[data-testid=worklist-row]:last-of-type')")
    ids_arrival = pg.locator(ROW).evaluate_all("rows => rows.map(r => r.dataset.caseId)")
    assert ids_arrival[-1] == hero                       # arrived last
    assert ids_ai.index(hero) < ids_ai.index(roles["routine_normal"])   # but read before routine cases
    pg.click(tid("order-ai"))
    page.assert_no_horizontal_scroll()
    page.shot(f"01-worklist-{page.name}")

    # 2. Open the hero: heatmap, Why, conformal set, drafted orders; Order tests; Sign out → oncologist inbox.
    pg.locator(f"{ROW}[data-case-id='{hero}']").click()
    view = pg.locator(f"{tid('case-view')}[data-lane=biopsy]")
    expect(view).to_be_visible()
    expect(pg.locator(tid("heatmap-img"))).to_be_visible()
    pg.wait_for_function("() => document.querySelector('[data-testid=heatmap-img]').naturalWidth > 0")
    pg.select_option(tid("heatmap-select"), "invasion")
    pg.click(tid("why-bar-invasion"))
    expect(pg.locator("rect.hl")).to_have_count(2)
    expect(pg.locator(tid("src-measured")).first).to_be_visible()
    expect(pg.locator(tid("src-estimated")).first).to_be_visible()
    assert pg.locator(tid("conformal-chip")).all_inner_texts() == ["In situ", "Invasive"]   # display labels, never raw ids
    orders = " ".join(pg.locator(tid("agent-order")).all_inner_texts())
    for item in ("p63", "calponin", "ER", "PR", "HER2", "Ki-67"):
        assert item in orders
    expect(pg.locator(tid("summary-generator"))).to_be_visible()
    expect(pg.locator(tid("action-signout"))).to_be_disabled()
    page.shot(f"02-biopsy-{page.name}")
    pg.click(tid("action-approve"))                          # irreversible actions confirm inline first
    expect(pg.locator(tid("confirm-box"))).to_contain_text("p63")
    expect(pg.locator(tid("agent-status"))).to_have_attribute("data-status", "drafted")
    pg.click(tid("confirm-action"))
    expect(pg.locator(tid("agent-status"))).to_have_attribute("data-status", "approved")
    expect(pg.locator(tid("decision-feedback"))).to_contain_text("approved in Sepia")
    pg.click(tid("action-signout"))
    pg.click(tid("confirm-action"))
    expect(pg.locator(tid("verdict"))).to_contain_text("Signed out")
    pg.click(tid("nav-oncologist"))
    expect(pg.locator(f"{tid('oncologist-item')}[data-case-id='{hero}']")).to_be_visible()
    page.shot(f"03-oncologist-{page.name}")

    # 3. Inject a surgical case: top of the list with a live timer; tumor at the superior margin; timer stops.
    pg.click(tid("nav-worklist"))
    pg.click(tid("sim-inject-surgical"))
    first = pg.locator(ROW).first
    expect(first).to_have_attribute("data-lane", "surgical")
    expect(first).to_have_attribute("data-tier", "1")
    timer = first.locator("[data-wait] b")
    before = timer.inner_text()
    pg.wait_for_timeout(2200)
    assert timer.inner_text() != before
    first.click()
    expect(pg.locator(f"{tid('case-view')}[data-lane=surgical]")).to_be_visible()
    expect(pg.locator(tid("sim-banner"))).to_be_visible()
    expect(pg.locator(tid("margin-2d-note"))).to_be_visible()
    assert pg.locator(f"{tid('margin-seg')}[data-margin=superior][data-involved=true]").count() >= 1
    expect(pg.locator(tid("agent-guard"))).to_be_visible()
    page.shot(f"04-surgical-{page.name}")
    pg.click(tid("call-tumor"))
    expect(pg.locator(tid("margin-pick-superior"))).to_be_checked()
    pg.click(tid("call-confirm"))
    expect(pg.locator(tid("verdict"))).to_contain_text("superior")
    expect(pg.locator(tid("surgical-timer"))).to_have_attribute("data-state", "stopped")
    pg.click(tid("nav-surgeon"))
    expect(pg.locator(tid("surgeon-result")).first).to_contain_text("superior")
    page.shot(f"05-or-board-{page.name}")

    # 4. Inject a degraded case: reliability flag → Defer to permanent.
    pg.click(tid("nav-worklist"))
    pg.click(tid("sim-inject-degraded"))
    pg.locator(f"{ROW}[data-lane=surgical]").first.click()
    expect(pg.locator(f".flag{tid('reliability-flag')}")).to_contain_text("defer to permanent")
    page.shot(f"06-degraded-{page.name}")
    pg.click(tid("call-defer"))
    pg.click(tid("confirm-action"))
    expect(pg.locator(tid("verdict"))).to_contain_text("defer")
    pg.click(tid("nav-surgeon"))
    expect(pg.locator(tid("surgeon-result")).filter(has_text=re.compile("defer", re.I))).to_have_count(1)

    # composite margin case
    pg.click(tid("nav-worklist"))
    pg.click(tid("sim-inject-composite"))
    pg.locator(f"{ROW}[data-lane=surgical]").first.click()
    expect(pg.locator(tid("sim-banner"))).to_contain_text("Composite")
    assert pg.locator(f"{tid('margin-seg')}[data-involved=true]").count() >= 1

    # 5. Metrics page renders (synthetic dev eval files) with charts and limitations.
    pg.click(tid("nav-metrics"))
    for section in ("metrics-queue-sim", "metrics-calibration", "metrics-concepts", "metrics-reliability"):
        key = section.removeprefix("metrics-").replace("-", "_")
        # a chart when the eval file exists, otherwise its explicit pending state
        expect(pg.locator(f"{tid(section)} svg.chart, {tid('metrics-missing-' + key)}").first).to_be_visible()
    expect(pg.locator(tid("metrics-synthetic"))).to_be_visible()
    expect(pg.locator(tid("limitations"))).to_be_visible()
    page.assert_no_horizontal_scroll()
    page.shot(f"07-metrics-{page.name}")
    page.assert_no_ground_truth()


def test_cells_tab_and_finding_refs(page, api, roles):
    pg = page.page
    pg._base = page.base
    start_and_wait_for_biopsies(pg, api, roles)
    pg.locator(f"{ROW}[data-case-id='{roles['hero_biopsy']}']").click()
    target = pg.locator(f"{tid('finding-ref')}[data-kind=cells]")
    expect(target).to_be_visible()
    target.click()
    expect(pg.locator(tid("cells-tab"))).to_be_visible()
    expect(pg.locator(f"{tid('cells-legend')} {tid('cells-legend-item')}")).to_have_count(5)
    img = pg.locator(tid("cells-overlay-img")).first
    src = img.get_attribute("src")
    pg.click(tid("cells-overlay-toggle"))
    expect(pg.locator(tid("cells-overlay-img"))).to_have_count(0)
    pg.click(tid("cells-overlay-toggle"))
    assert pg.locator(tid("cells-overlay-img")).first.get_attribute("src") == src
    expect(pg.locator(tid("cells-feature-nuc_area_cv"))).to_contain_text("measured")
    page.shot(f"08-cells-{page.name}")
    # a case without cell measurements shows the disabled state
    pg.goto(page.base + f"/#/case/{roles['routine_normal']}")
    pg.click(tid("tab-cells"))
    expect(pg.locator(tid("cells-disabled"))).to_be_visible()
    page.assert_no_ground_truth()


def test_edit_orders_and_spotcheck(page, api, roles):
    pg = page.page
    pg._base = page.base
    start_and_wait_for_biopsies(pg, api, roles)
    pg.locator(f"{ROW}[data-case-id='{roles['hero_biopsy']}']").click()
    pg.click(tid("action-edit"))
    boxes = pg.locator(tid("edit-item"))
    assert boxes.count() == 6
    boxes.nth(0).uncheck()                                   # drop p63
    pg.wait_for_timeout(2300)                                # a poll must not clobber the edit form
    assert not boxes.nth(0).is_checked()
    pg.click(tid("action-edit-save"))
    expect(pg.locator(tid("agent-status"))).to_have_attribute("data-status", "edited_and_approved")
    assert "p63" not in pg.locator(f"{tid('agent-order')}[data-status=edited]").first.inner_text()

    pg.click(tid("nav-spotcheck"))
    expect(pg.locator(tid("spot-item")).first).to_be_visible()
    pg.locator(tid("spot-item")).nth(0).locator(tid("spot-verdict-ok")).check()
    pg.locator(tid("spot-item")).nth(1).locator(tid("spot-verdict-not_ok")).check()
    pg.fill(tid("spot-reviewer"), "e2e reviewer")
    pg.select_option(tid("spot-kind"), "ai")
    with pg.expect_request(lambda r: r.url.endswith("/api/spotcheck") and r.method == "POST") as req:
        pg.click(tid("spot-submit"))
    body = req.value.post_data_json
    assert body["reviewer_kind"] == "ai" and len(body["reviews"]) == 2
    expect(pg.locator(tid("spot-result"))).to_contain_text("Recorded 2")
    page.assert_no_ground_truth()


def test_backend_down_state(page, api, roles):
    pg = page.page
    pg.goto(page.base + "/")
    expect(pg.locator(tid("state-empty"))).to_be_visible()
    pg.route("**/api/**", lambda route: route.abort())
    expect(pg.locator(tid("state-offline"))).to_be_visible(timeout=12000)
    pg.unroute("**/api/**")
    expect(pg.locator(tid("state-offline"))).to_be_hidden(timeout=12000)
    page.errors.clear()       # aborted requests log network errors by design


def test_mock_mode_runs_without_the_api(page, api, roles):
    pg = page.page
    pg.route("**/api/**", lambda route: route.abort())      # prove mock mode never needs the backend API
    pg.goto(page.base + "/?mock=1")
    expect(pg.locator(tid("banner-mock"))).to_be_visible()
    pg.click(tid("sim-start"))
    expect(pg.locator(ROW).first).to_be_visible(timeout=20000)
    pg.click(tid("sim-inject-surgical"))
    expect(pg.locator(ROW).first).to_have_attribute("data-lane", "surgical")
    page.assert_no_ground_truth()


def test_queue_search_and_status_filters(page, api, roles):
    pg = page.page
    pg._base = page.base
    start_and_wait_for_biopsies(pg, api, roles)
    hero = roles['hero_biopsy']
    total = pg.locator('#queue-count').inner_text()
    pg.click('[data-lane-filter=surgical]')
    expect(pg.locator('#queue-count')).to_have_text(total)
    pg.click('[data-lane-filter=all]')
    pg.fill('#case-search', hero)
    expect(pg.locator(ROW)).to_have_count(1)
    expect(pg.locator(ROW)).to_have_attribute('data-case-id', hero)
    pg.wait_for_timeout(2300)
    expect(pg.locator('#case-search')).to_have_value(hero)
    assert pg.locator('#case-search').evaluate('(el) => document.activeElement === el')
    pg.fill('#case-search', '<no such case>')
    expect(pg.locator(tid('state-empty'))).to_contain_text('No matching cases')
    pg.click('[data-clear-search]')
    expect(pg.locator('#case-search')).to_have_value('')
    assert pg.locator(ROW).count() > 1
    pg.click('[data-status-filter=done]')
    expect(pg.locator(ROW)).to_have_count(0)
    pg.click('[data-clear-search]')
    pg.click('[data-status-filter=open]')
    assert pg.locator(ROW).count() > 1
    page.assert_no_horizontal_scroll()


def test_evidence_chart_style_and_navigation(page, api, roles):
    pg = page.page
    pg._base = page.base
    start_and_wait_for_biopsies(pg, api, roles)
    pg.locator(f"{ROW}[data-case-id='{roles['hero_biopsy']}']").press('Enter')
    expect(pg.locator(tid('case-view'))).to_be_visible()
    # Icon strokes must not leak into the charts' small coordinate systems.
    assert pg.locator('.why-row svg rect').first.evaluate('(el) => getComputedStyle(el).stroke') == 'none'
    expect(pg.locator('#breadcrumb')).to_have_text('Case review')
    expect(pg.locator(tid('nav-worklist'))).to_have_attribute('aria-current', 'page')
    pg.click(tid('nav-surgeon'))
    expect(pg.locator('#breadcrumb')).to_have_text('Operating rooms')
    expect(pg.locator(tid('nav-surgeon'))).to_have_attribute('aria-current', 'page')
    pg.click(tid('nav-worklist'))
    expect(pg.locator('#page-title')).to_have_text('Worklist')
    page.assert_no_horizontal_scroll()


def test_replay_pause_resume_and_reset_guard(page, api, roles):
    pg = page.page
    pg._base = page.base
    start_and_wait_for_biopsies(pg, api, roles)
    pg.click(tid('sim-pause'))
    expect(pg.locator(tid('sim-clock'))).to_contain_text('paused')
    expect(pg.locator(tid('sim-pause'))).to_have_text('Resume replay')
    pg.click(tid('sim-pause'))
    expect(pg.locator(tid('sim-clock'))).to_contain_text('running')
    count = pg.locator(ROW).count()
    pg.click(tid('sim-reset'))
    expect(pg.locator('#reset-dialog')).to_be_visible()
    pg.get_by_role('button', name='Keep reviewing').click()
    expect(pg.locator('#reset-dialog')).to_be_hidden()
    assert pg.locator(ROW).count() == count
    pg.click(tid('sim-reset'))
    pg.click('[data-reset-confirm]')
    expect(pg.locator('#reset-dialog')).to_be_hidden()
    expect(pg.locator(tid('state-empty'))).to_be_visible()


# ---- workflow shell: keyboard, next case, routing feedback, OR board, inbox, persistence --------------------------------
def focused_case(pg) -> str | None:
    return pg.evaluate("document.activeElement && document.activeElement.dataset.caseId || null")


def test_keyboard_next_case_and_routing_feedback(page, api, roles):
    pg = page.page
    pg._base = page.base
    start_and_wait_for_biopsies(pg, api, roles)
    hero = roles["hero_biopsy"]
    ids = pg.locator(ROW).evaluate_all("rows => rows.map(r => r.dataset.caseId)")
    pg.keyboard.press("/")
    assert pg.evaluate("document.activeElement.id") == "case-search"
    pg.evaluate("document.activeElement.blur()")
    pg.keyboard.press("j")
    assert focused_case(pg) == ids[0]
    pg.keyboard.press("j")
    assert focused_case(pg) == ids[1]
    pg.keyboard.press("k")
    assert focused_case(pg) == ids[0]
    pg.wait_for_timeout(2300)                                   # a poll re-render keeps the row focus
    assert focused_case(pg) == ids[0]
    pg.keyboard.press("Enter")
    expect(pg.locator(f"{tid('case-view')}[data-case-id='{ids[0]}']")).to_be_visible()
    expect(pg.locator(tid("casebar"))).to_contain_text("Case 1 of")
    pg.keyboard.press("n")                                      # next case in priority order
    expect(pg.locator(f"{tid('case-view')}[data-case-id='{ids[1]}']")).to_be_visible()

    # act on the hero: orders placed, sign-out routed, the shell offers the next case and counts the unread result
    pg.goto(page.base + f"/#/case/{hero}")
    expect(pg.locator(f"{tid('case-view')}[data-case-id='{hero}']")).to_be_visible()
    pg.click(tid("action-approve"))
    pg.click(tid("confirm-action"))                         # irreversible actions confirm inline first
    expect(pg.locator(tid("toast")).filter(has_text="Orders placed")).to_be_visible()
    pg.click(tid("action-signout"))
    pg.click(tid("confirm-action"))
    expect(pg.locator(tid("casebar-done"))).to_contain_text("Signed out")
    expect(pg.locator(tid("toast")).filter(has_text="signed out")).to_be_visible()
    expect(pg.locator(tid("inbox-unread"))).to_have_text("1")
    nxt = pg.locator(tid("casebar-next"))
    expect(nxt).to_be_focused()
    pg.keyboard.press("Enter")
    expect(pg.locator(tid("case-view"))).not_to_have_attribute("data-case-id", hero)
    page.shot(f"09-next-case-{page.name}")

    # inbox report card: unread until read, then the badge clears
    pg.click(tid("nav-oncologist"))
    item = pg.locator(f"{tid('oncologist-item')}[data-case-id='{hero}']")
    expect(item).to_have_attribute("data-unread", "true")
    expect(item.locator(tid("inbox-category"))).not_to_be_empty()
    expect(item.locator(tid("inbox-signed-out"))).to_contain_text("Signed out")
    assert item.locator(tid("inbox-order")).count() >= 1
    item.locator("[data-inbox-read]").click()
    expect(item).to_have_attribute("data-unread", "false")
    expect(pg.locator(tid("inbox-unread"))).to_be_hidden()
    page.assert_no_horizontal_scroll()
    page.shot(f"10-inbox-{page.name}")

    # reload: decisions persist (server state), and the read state persists (per viewer)
    pg.reload()
    expect(pg.locator(f"{tid('oncologist-item')}[data-case-id='{hero}']")).to_have_attribute("data-unread", "false")
    pg.click(tid("nav-worklist"))
    expect(pg.locator(f"{ROW}[data-case-id='{hero}']")).to_have_attribute("data-done", "true")
    page.assert_no_ground_truth()


def test_or_board_runs_then_stops_and_highlights_the_call(page, api, roles):
    pg = page.page
    pg._base = page.base
    pg.goto(page.base + "/#/surgeon")
    expect(pg.locator(tid("surgeon-empty"))).to_be_visible()
    expect(pg.locator(tid("or-room"))).to_have_count(api.get("/api/config")["sim"]["or_rooms"])
    pg.click(tid("sim-inject-surgical"))
    card = pg.locator(f"{tid('or-card')}").first
    expect(card.locator(tid("or-awaiting"))).to_be_visible()
    timer = card.locator(tid("or-timer"))
    before = timer.inner_text()
    pg.wait_for_timeout(2200)
    assert timer.inner_text() != before                          # live while awaiting the call
    expect(pg.locator(tid("nav-surgeon")).locator("#or-count")).to_have_text("1")
    case_id = card.get_attribute("data-case-id")
    card.get_by_role("link", name="Open case").click()
    expect(pg.locator(f"{tid('case-view')}[data-lane=surgical]")).to_be_visible()
    pg.click(tid("call-clear"))
    pg.click(tid("confirm-action"))
    expect(pg.locator(tid("casebar-done"))).to_contain_text("Call sent")
    expect(pg.locator(tid("toast")).filter(has_text="Margin call sent")).to_be_visible()
    pg.click(tid("nav-surgeon"))
    called = pg.locator(f"{tid('or-card')}[data-case-id='{case_id}']")
    expect(called.locator(tid("surgeon-result"))).to_have_attribute("data-call", "clear")
    expect(called.locator(tid("or-new-call"))).to_be_visible()
    stopped = called.locator(tid("or-turnaround")).inner_text()
    pg.wait_for_timeout(2200)
    assert called.locator(tid("or-turnaround")).inner_text() == stopped   # timer stopped at the call
    expect(pg.locator("#or-count")).to_be_hidden()
    page.assert_no_horizontal_scroll()
    page.shot(f"11-or-board-call-{page.name}")


def test_worklist_filters_composition_and_priority_popover(page, api, roles):
    pg = page.page
    pg._base = page.base
    start_and_wait_for_biopsies(pg, api, roles)
    total = pg.locator(ROW).count()
    pg.click("summary[aria-label='How cases are prioritized']")
    expect(pg.locator(tid("order-note"))).to_contain_text("Safety rule")
    pg.keyboard.press("Escape")
    expect(pg.locator(tid("order-note"))).to_be_hidden()
    pg.click(tid("tier-filter-2"))
    tiers = set(pg.locator(ROW).evaluate_all("rows => rows.map(r => r.dataset.tier)"))
    assert tiers == {"2"}
    pg.click(tid("tier-filter-2"))
    assert pg.locator(ROW).count() == total
    pg.check(tid("orders-only"))
    assert pg.locator(ROW).count() == pg.locator(f"{ROW} {tid('orders-drafted')}").count() > 0
    pg.uncheck(tid("orders-only"))
    pg.select_option(tid("flag-filter"), "any")
    assert pg.locator(ROW).count() <= total
    pg.select_option(tid("flag-filter"), "all")
    assert pg.locator(ROW).count() == total
    page.assert_no_horizontal_scroll()


def test_restart_preserves_the_workflow(browser, dev_root, roles, tmp_path):
    """Decisions, routing and the replay clock survive a backend restart (separate server + copied workspace)."""
    import shutil
    import socket
    import subprocess
    import sys
    import time

    from tests.e2e.conftest import REPO, _get, _post

    root = tmp_path / "ws"
    shutil.copytree(dev_root, root, ignore=shutil.ignore_patterns("*.sqlite*"))

    def serve():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = {**__import__("os").environ, "PATHQUEUE_ARTIFACTS_ROOT": str(root), "PATHQUEUE_CASES_DIR": ""}
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port",
                                 str(port)], cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{port}"
        for _ in range(150):
            try:
                if _get(base + "/api/health")["loaded"]:
                    return proc, base
            except OSError:
                time.sleep(0.1)
        proc.kill()
        raise RuntimeError("backend did not start")

    hero = roles["hero_biopsy"]
    proc, base = serve()
    try:
        _post(base + "/api/sim/start")
        for _ in range((REPLAY_S + 20) * 4):
            if any(i["case_id"] == hero for i in _get(base + "/api/worklist")["items"]):
                break
            time.sleep(0.25)
        _post(base + f"/api/cases/{hero}/decision", {"action": "approve_orders"})
        _post(base + f"/api/cases/{hero}/decision", {"action": "sign_out"})
        surgical = _post(base + "/api/sim/inject_surgical", {"variant": "normal"})["injected"]
        _post(base + f"/api/cases/{surgical}/decision", {"action": "margin_call", "payload": {"call": "defer"}})
        before = _get(base + "/api/health")["sim"]["sim_time_s"]
        time.sleep(0.5)                                   # the replay keeps running until the process stops
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    proc, base = serve()
    try:
        assert _get(base + "/api/health")["sim"]["sim_time_s"] >= before
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(base + "/#/worklist")
        expect(pg.locator(f"{ROW}[data-case-id='{hero}']")).to_have_attribute("data-done", "true")
        expect(pg.locator(f"{ROW}[data-case-id='{surgical}']")).to_have_attribute("data-done", "true")
        expect(pg.locator(tid("sim-pause"))).to_have_text("Resume replay")      # restored paused, never auto-run
        pg.click(tid("nav-oncologist"))
        expect(pg.locator(f"{tid('oncologist-item')}[data-case-id='{hero}']")).to_be_visible()
        pg.click(tid("nav-surgeon"))
        expect(pg.locator(f"{tid('or-card')}[data-case-id='{surgical}'] {tid('surgeon-result')}")).to_have_attribute(
            "data-call", "defer")
        assert errors == []
        ctx.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
