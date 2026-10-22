"""Shell robustness in a real browser (desktop light, 390 px dark): odd addresses, and decisions whose reply is lost."""

from __future__ import annotations

from playwright.sync_api import expect


def tid(name: str) -> str:
    return f"[data-testid={name}]"


def test_refresh_uses_one_workspace_request_and_suspends_while_hidden(page, api):
    pg = page.page
    requests = []
    pg.on("request", lambda request: requests.append(request.url) if "/api/" in request.url else None)
    with pg.expect_response(lambda response: "/api/worklist?" in response.url):
        pg.goto(page.base + "/")
    pg.wait_for_timeout(500)  # finish the initial refresh before changing visibility
    assert any("include_context=true" in url for url in requests)
    assert not any("/api/health" in url or "/api/routed" in url for url in requests)
    pg.evaluate("""() => {
        Object.defineProperty(document, 'hidden', {configurable: true, get: () => true});
        document.dispatchEvent(new Event('visibilitychange'));
    }""")
    before = len(requests)
    pg.wait_for_timeout(4500)  # more than two normal refresh intervals
    assert len(requests) == before
    with pg.expect_response(lambda response: "/api/worklist?" in response.url and response.status == 200):
        pg.evaluate("""() => {
            Object.defineProperty(document, 'hidden', {configurable: true, get: () => false});
            document.dispatchEvent(new Event('visibilitychange'));
        }""")


def test_odd_case_addresses_still_start_the_app(page, api):
    pg = page.page
    # a malformed % escape (typed or truncated link) used to throw in the router, and the app never finished starting
    for bad in ("%E0%A4%A", "BACH_0225%"):
        pg.goto(page.base + f"/#/case/{bad}")
        expect(pg.locator(tid("case-not-found"))).to_be_visible()
    # an external specimen under the worklist address points to its own page
    pg.goto(page.base + "/#/case/TCGA-A7-A4SD_ffpe_tumor_00")
    expect(pg.locator(tid("not-found-tcga"))).to_have_attribute("href", "#/tcga/TCGA-A7-A4SD_ffpe_tumor_00")
    # a held simulated composite arrives only through its OR control
    pg.goto(page.base + "/#/case/COMP_999")
    expect(pg.locator(tid("case-not-found"))).to_contain_text("+ Composite")
    # the unknown ids answer 404 by design (logged by the browser); no script error may have happened
    assert not [e for e in page.errors if e.startswith("pageerror")], page.errors
    page.errors.clear()


def _open_frozen_section(pg, page, api) -> str:
    case_id = api.post("/api/sim/inject_surgical", {"variant": "normal"})["injected"]
    pg.goto(page.base + f"/#/case/{case_id}")
    expect(pg.locator(f"{tid('case-view')}[data-lane=surgical]")).to_be_visible()
    return case_id


def test_a_decision_whose_reply_is_lost_is_checked_before_it_is_called_failed(page, api):
    pg = page.page

    # the request reaches Sepia but the reply never arrives: the case is read again and the call is confirmed
    def reply_lost(route):
        route.fetch()
        route.abort("connectionreset")

    _open_frozen_section(pg, page, api)
    pg.route("**/api/cases/*/decision", reply_lost)
    pg.click(tid("call-clear"))
    pg.click(tid("confirm-action"))
    expect(pg.locator(tid("toast")).filter(has_text="Margin call sent")).to_be_visible()
    expect(pg.locator(tid("casebar-done"))).to_contain_text("Call sent")
    expect(pg.locator(tid("toast")).filter(has_text="not reachable")).to_have_count(0)
    pg.unroute("**/api/cases/*/decision")

    # the request never leaves: said plainly, and the case stays open for the decision
    _open_frozen_section(pg, page, api)
    pg.route("**/api/cases/*/decision", lambda route: route.abort("internetdisconnected"))
    pg.click(tid("call-clear"))
    pg.click(tid("confirm-action"))
    expect(pg.locator(tid("toast")).filter(has_text="Sepia is not reachable; your decision was not sent")).to_be_visible()
    expect(pg.locator(tid("casebar-done"))).to_have_count(0)
    pg.unroute("**/api/cases/*/decision")
    # aborted requests log network errors by design; no script error may have happened
    assert not [e for e in page.errors if e.startswith("pageerror")], page.errors
    page.errors.clear()


def test_a_held_replay_clock_says_so(page, api):
    pg = page.page

    # sim.held (docs/API.md): every scheduled case is in and the replay clock stands at its hold
    def held(route):
        res = route.fetch()
        body = res.json()
        if body.get("sim"):
            body["sim"].update(held=True, hold_at_s=body["sim"]["sim_time_s"], running=True)
        route.fulfill(response=res, json=body)

    pg.route("**/api/worklist*", held)
    pg.goto(page.base + "/")
    expect(pg.locator(tid("sim-clock"))).to_contain_text("Replay complete; clock held")
    expect(pg.locator(tid("sim-clock"))).to_contain_text("simulator clock held")
    expect(pg.locator(tid("replay-toggle"))).to_contain_text("Replay complete")
    pg.unroute("**/api/worklist*")
