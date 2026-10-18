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
    expect(hero_row.locator(tid("reason-aggressive_in_conformal_set"))).to_be_visible()
    expect(hero_row.locator(tid("orders-drafted"))).to_be_visible()
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
    assert pg.locator(tid("conformal-chip")).all_inner_texts() == ["InSitu", "Invasive"]
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
    expect(pg.locator(tid("decision-feedback"))).to_contain_text("ordered")
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
    timer = first.locator("[data-timer]")
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
    expect(pg.locator('#page-title')).to_have_text('Every case. A clearer next step.')
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
