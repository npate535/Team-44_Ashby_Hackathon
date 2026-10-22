"""Capture the real Sepia UI for the launch video: 2x stills, a replay time-lapse and element rectangles.

    .venv/bin/python video/launch/capture/capture.py http://127.0.0.1:8123 video/launch/public/captures

Run it against a throwaway server on a fresh workspace database (it starts the replay, approves orders, signs out
and sends margin calls). Every image is a real screenshot; meta.json records the CSS-pixel rectangle of each element
the video points at, so highlights and the cursor land on real controls. Captures contain BACH-derived imagery and
stay out of Git.
"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

base, out = sys.argv[1].rstrip('/'), Path(sys.argv[2])
RESUME = '--resume' in sys.argv      # skip the replay time-lapse; reuse meta.json from the first pass
out.mkdir(parents=True, exist_ok=True)
(out / 'lapse').mkdir(exist_ok=True)
VW, VH, DPR = 1600, 1000, 2
T = lambda s: f'[data-testid="{s}"]'
meta = {'viewport': [VW, VH], 'dpr': DPR, 'shots': {}}
if RESUME:
    meta = json.load(open(out / 'meta.json'))

RECT_JS = """(sels) => {
  const r = {};
  for (const [name, sel0, all] of sels) {
    const [sel, needle] = sel0.split('|');
    let els = [...document.querySelectorAll(sel)];
    if (needle) els = els.filter(e => (e.innerText || '').includes(needle));
    if (!all) els = els.slice(0, 1);
    const rs = els.map(e => { const b = e.getBoundingClientRect();
      return {x: b.x, y: b.y + window.scrollY, w: b.width, h: b.height, text: (e.innerText || '').slice(0, 80)}; });
    r[name] = all ? rs : (rs[0] || null);
  }
  return {rects: r, scrollY: window.scrollY, pageH: document.documentElement.scrollHeight};
}"""


def rects(pg, sels):
    return pg.evaluate(RECT_JS, [[n, s, a] for n, s, a in sels])


def shot(pg, name, sels=(), full=False, settle=900):
    pg.wait_for_timeout(settle)
    pg.screenshot(path=str(out / f'{name}.png'), full_page=full)
    info = rects(pg, sels)
    info['full'] = full
    meta['shots'][name] = info
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    print('shot', name, 'scrollY', info['scrollY'], 'pageH', info['pageH'], flush=True)
    return info


def scroll_to(pg, sel, offset=120):
    pg.evaluate(f"""() => {{ const e = document.querySelector('{sel}'); if (!e) return;
      window.scrollTo({{top: e.getBoundingClientRect().top + window.scrollY - {offset}, behavior: 'instant'}}); }}""")
    pg.wait_for_timeout(500)


def open_toast(pg):
    pg.locator('#toast .toast-act').last.click()
    pg.wait_for_timeout(1800)
    return pg.url.split('/case/')[-1]


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={'width': VW, 'height': VH}, device_scale_factor=DPR, color_scheme='dark')
    pg = ctx.new_page()
    pg.goto(base + '/#/worklist')
    pg.wait_for_selector(T('sim-start') + ':not([disabled])' if not RESUME else T('worklist'), timeout=20000)
    pg.wait_for_timeout(1200)

    # 1. Replay time-lapse: biopsies arrive and the queue sorts itself; a composite frozen section lands on top.
    if not RESUME:
        shot(pg, 'wl_empty', [('start', T('sim-start'), False)])
        pg.click(T('sim-start'))
        t0, frames, sent_frozen, hero_at = time.time(), [], False, None
        while time.time() - t0 < 140:
            f = out / 'lapse' / f'{len(frames):04d}.jpg'
            pg.screenshot(path=str(f), type='jpeg', quality=90)
            frames.append({'f': f.name, 't': round(time.time() - t0, 3)})
            if hero_at is None and pg.locator('text=BACH_0167').count():
                hero_at = time.time() - t0
            if hero_at is not None and not sent_frozen and time.time() - t0 > hero_at + 2.0:
                frames[-1]['event'] = 'composite'
                pg.click(T('sim-inject-composite')); sent_frozen = True
            if sent_frozen and time.time() - t0 > hero_at + 6.5:
                break
        meta['lapse'] = {'frames': frames, 'hero_at': hero_at}
        json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
        print('lapse frames', len(frames), 'hero at', hero_at, flush=True)
        frozen_id = open_toast(pg) if pg.locator('#toast .toast-act').count() else None
        meta['frozen_id'] = frozen_id
        pg.goto(base + '/#/worklist'); pg.wait_for_timeout(1500)
        wl = [('tiers', T('tier-group'), True), ('rows', T('worklist-row'), True), ('stats', T('queue-stats'), False),
              ('hero', T('worklist-row') + '|BACH_0167', False), ('header', 'header', False),
              ('nav', 'nav', False), ('frozen_btns', T('sim-inject-composite'), False)]
        shot(pg, 'wl_final', wl)
        shot(pg, 'wl_full', wl, full=True)


    frozen_id = meta.get('frozen_id')

    # 2. Hero case BACH_0167, heatmap at a readable opacity (the viewer's own control).
    pg.goto(base + '/#/case/BACH_0167'); pg.wait_for_selector(T('heatmap-img')); pg.wait_for_timeout(1500)
    pg.evaluate("""() => { const i = document.querySelector('[data-testid=heatmap-opacity]'); i.value = '0.75';
      i.dispatchEvent(new Event('input', {bubbles: true})); i.dispatchEvent(new Event('change', {bubbles: true})); }""")
    case = [('title', T('case-title'), False), ('summary', T('summary'), False), ('agent', T('agent-panel'), False),
            ('orders', T('agent-order'), True), ('approve', T('action-approve'), False), ('heatmap', T('heatmap'), False),
            ('heatmap_img', T('heatmap-img'), False), ('tiles', T('hm-tile'), True), ('how_sure', T('how-sure'), False),
            ('set', T('conformal-set'), False), ('set_chips', T('set-chip'), True), ('set_callout', T('set-callout'), False),
            ('agreement', T('agreement-checks'), False), ('why', T('why'), False), ('why_regions', T('why-regions'), False),
            ('spot_failed', T('why-bar-mitoses'), False), ('spot_tag', T('spot-mitoses'), False),
            ('why_rows', '.why-row', True), ('region_rows', '.why-region', True), ('similar', T('similar-case'), True),
            ('contrast', T('contrast-case'), False), ('risk', T('risk'), False), ('headbox', '.cv-head .kv, .cv-head dl', False),
            ('casehead', '.cv-head', False), ('reliability', T('reliability-ok'), False)]
    shot(pg, 'case_top', case)
    shot(pg, 'case_full', case, full=True)
    pg.locator(T('heatmap')).screenshot(path=str(out / 'heatmap.png'))
    for k in (5, 9, 1):
        rg = pg.locator(f'[data-testid=heatmap] .rg[data-rg="{k}"]')
        if rg.count():
            rg.first.hover(); pg.wait_for_timeout(700)
            pg.locator(T('heatmap')).screenshot(path=str(out / f'heatmap_hover{k}.png'))
    pg.mouse.move(5, 5)
    scroll_to(pg, T('why'), 90)
    shot(pg, 'case_why', case)
    scroll_to(pg, T('agreement-checks'), 300)
    shot(pg, 'case_agree', case)
    scroll_to(pg, T('similar-case'), 160)
    shot(pg, 'case_similar', case)

    # 3. Approve the drafted orders, then sign out.
    pg.evaluate('window.scrollTo(0, 0)'); pg.wait_for_timeout(400)
    flow = case + [('confirm', T('confirm-action'), False), ('confirm_box', T('confirm-box'), False),
                   ('signout', T('action-signout'), False), ('approval', T('approval'), False)]
    shot(pg, 'flow_0', flow)
    pg.click(T('action-approve')); shot(pg, 'flow_1', flow)
    pg.click(T('confirm-action')); shot(pg, 'flow_2', flow, settle=1400)
    pg.click(T('action-signout')); shot(pg, 'flow_3', flow)
    pg.click(T('confirm-action')); shot(pg, 'flow_4', flow, settle=1600)

    # 4. Oncology inbox.
    pg.goto(base + '/#/oncologist'); pg.wait_for_timeout(1800)
    shot(pg, 'inbox', [('item', T('oncologist-item'), True), ('signout', T('inbox-signout'), True),
                       ('support', T('inbox-decision-support'), True), ('orders', T('inbox-order'), True)])

    # 5. Frozen section: the OR board before the call, the margin call, the board after.
    surg = [('timer', T('surgical-timer'), False), ('margin', T('margin-bars'), False), ('img', T('surgical-img'), False),
            ('suggested', T('suggested-call'), False), ('tumor', T('call-tumor'), False), ('clear', T('call-clear'), False),
            ('defer', T('call-defer'), False), ('confirm', T('call-confirm'), False), ('picks', T('margin-pick'), True),
            ('title', T('case-title'), False), ('flag', T('reliability-flag'), False), ('diagram', T('specimen-diagram'), False)]
    orb = [('cards', T('or-card'), True), ('timer', T('or-timer'), True), ('awaiting', T('or-awaiting'), True),
           ('turnaround', T('or-turnaround'), True), ('suggestion', T('or-model-suggestion'), True), ('result', T('surgeon-result'), True)]
    pg.goto(base + '/#/surgeon'); pg.wait_for_timeout(2500)
    shot(pg, 'or_before', orb)
    if frozen_id:
        pg.goto(base + f'/#/case/{frozen_id}'); pg.wait_for_timeout(2200)
        shot(pg, 'frozen_top', surg)
        shot(pg, 'frozen_full', surg, full=True)
        scroll_to(pg, T('call-tumor'), 420)
        shot(pg, 'frozen_call0', surg)
        pg.click(T('call-tumor')); shot(pg, 'frozen_call1', surg)
        pg.click(T('call-confirm')); shot(pg, 'frozen_call2', surg, settle=1500)
        pg.goto(base + '/#/surgeon'); pg.wait_for_timeout(2500)
        shot(pg, 'or_after', orb)

    # 6. Degraded frozen section: the reliability flag says defer.
    pg.goto(base + '/#/worklist'); pg.wait_for_timeout(1200)
    pg.click(T('sim-inject-degraded')); pg.wait_for_timeout(1500)
    deg_id = open_toast(pg)
    meta['degraded_id'] = deg_id
    shot(pg, 'degraded_top', surg)
    shot(pg, 'degraded_full', surg, full=True)
    if pg.locator(T('reliability-flag')).count():
        scroll_to(pg, T('reliability-flag'), 260)
        shot(pg, 'degraded_flag', surg)

    # 7. Evaluation and external test.
    pg.goto(base + '/#/metrics'); pg.wait_for_timeout(3000)
    met = [('headline', T('metrics-headline'), False), ('tiles', '.mtile, .metric-tile, [data-testid=metrics-headline] > *', True)]
    shot(pg, 'metrics', met)
    shot(pg, 'metrics_full', met, full=True)
    pg.goto(base + '/#/tcga'); pg.wait_for_timeout(3000)
    tc = [('external', T('tcga-external'), False), ('claim', T('tcga-claim'), False), ('table', T('tcga-table'), False),
          ('cohort', T('tcga-cohort'), False)]
    shot(pg, 'tcga', tc)
    shot(pg, 'tcga_full', tc, full=True)
    b.close()
json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
print('done')
