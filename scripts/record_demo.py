"""Record the plan §14 demo on the real artifacts (local backup video; contains BACH-derived imagery, never commit).

    .venv/bin/python scripts/record_demo.py http://127.0.0.1:<port> outputs/demo-video   # run against a throwaway server
                                                                                      # on a cloned artifact root
"""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

base, out = sys.argv[1].rstrip('/'), Path(sys.argv[2])
T = lambda s: f'[data-testid="{s}"]'


def pause(pg, ms=1600):
    pg.wait_for_timeout(ms)


def scroll_to_text(pg, text, ms=1800):
    pg.get_by_text(text, exact=False).first.scroll_into_view_if_needed()
    pause(pg, ms)


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={'width': 1440, 'height': 900}, record_video_dir=str(out),
                        record_video_size={'width': 1440, 'height': 900})
    pg = ctx.new_page()
    pg.goto(base + '/#/worklist'); pause(pg, 2500)
    pg.click(T('sim-start')); pause(pg, 2000)
    for _ in range(80):                                   # biopsies arrive; the hero arrives last
        if pg.locator('text=BACH_0167').count():
            break
        pause(pg, 1200)
    pause(pg, 2500)
    pg.goto(base + '/#/case/BACH_0167'); pause(pg, 3000)
    pg.mouse.move(560, 600); pause(pg, 1500); pg.mouse.move(700, 650); pause(pg, 1500)
    scroll_to_text(pg, 'Grounded summary', 2500)
    scroll_to_text(pg, 'features associated with invasive carcinoma', 3500)
    scroll_to_text(pg, 'Prediction and confidence', 3000)
    pg.evaluate('window.scrollTo({top: 0, behavior: "smooth"})'); pause(pg, 1500)
    pg.click(T('action-approve')); pause(pg, 1500)
    pg.click(T('confirm-action')); pause(pg, 2200)
    pg.click(T('action-signout')); pause(pg, 1500)
    pg.click(T('confirm-action')); pause(pg, 2500)
    pg.goto(base + '/#/oncologist'); pause(pg, 3500)
    pg.goto(base + '/#/worklist'); pause(pg, 1200)
    pg.get_by_role('button', name='Send a composite simulated frozen section').click(); pause(pg, 2500)
    pg.goto(base + '/#/case/COMP_026'); pause(pg, 3500)
    scroll_to_text(pg, 'Margin call', 2000)
    pg.click(T('call-tumor')); pause(pg, 1800)
    pg.click(T('call-confirm')); pause(pg, 2200)
    pg.goto(base + '/#/surgeon'); pause(pg, 3500)
    pg.get_by_role('button', name='Send a degraded simulated frozen section').click(); pause(pg, 2500)
    pg.goto(base + '/#/case/BACH_0301_FSIM_resample3'); pause(pg, 3000)
    scroll_to_text(pg, 'Reliability flag raised', 3500)
    pg.click(T('call-defer')); pause(pg, 1500)
    pg.click(T('confirm-action')); pause(pg, 2500)
    pg.goto(base + '/#/surgeon'); pause(pg, 3000)
    pg.goto(base + '/#/metrics'); pause(pg, 3500)
    for y in range(0, 9000, 900):
        pg.evaluate(f'window.scrollTo({{top: {y}, behavior: "smooth"}})'); pause(pg, 1400)
    video = pg.video.path()
    ctx.close(); b.close()
    final = out / 'pathqueue_demo.webm'
    Path(video).replace(final)
    print('saved', final)
