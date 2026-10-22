"""Stage assets for the Remotion project (all outputs under public/ stay out of Git; src/data holds small JSON).

    .venv/bin/python video/launch/prep.py <narration_dir>

- narration clips and timings from audio/gen_narration.py
- fonts and the Sepia mark from the app
- 256-px thumbnails of the 132 held-out BACH test images with their reference labels and the production model's call
- the capture rectangles (public/captures/meta.json)
"""
import glob, json, shutil, subprocess, sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PUB, DATA = HERE / 'public', HERE / 'src' / 'data'
vo = Path(sys.argv[1])
for d in (PUB / 'audio' / 'vo', PUB / 'fonts', PUB / 'brand', PUB / 'thumbs', DATA):
    d.mkdir(parents=True, exist_ok=True)

secs = json.load(open(vo / 'sections.json'))
for s in secs['sections']:
    shutil.copy(vo / f"{s['id']}.wav", PUB / 'audio' / 'vo' / f"{s['id']}.wav")
json.dump(secs, open(DATA / 'sections.json', 'w'), indent=1)

app = ROOT / 'team44-triage-frontend' / 'app'
for f in ('fraunces-var.woff2', 'bricolage-grotesque-var.woff2'):
    shutil.copy(app / 'fonts' / f, PUB / 'fonts' / f)
shutil.copy(app / 'brand' / 'sepia-mark-256.png', PUB / 'brand' / 'sepia-mark-256.png')

import csv
labels = {r['case_id']: r for r in csv.DictReader(open(ROOT / 'data' / 'cases.csv'))}
test = []
for r in csv.DictReader(open(ROOT / 'data' / 'predictions.csv')):
    if r['split'] != 'test' or r['model'] != 'uni2h512_lr' or r['case_id'] not in labels:
        continue
    cid = r['case_id']
    test.append({'id': cid, 'label': labels[cid]['label'], 'top': r['top'], 'p_inv': round(float(r['pcal__Invasive']), 4)})
    t = PUB / 'thumbs' / f"{cid}.jpg"
    if not t.exists():
        src = ROOT / 'artifacts' / 'images' / f'{cid}.jpg'
        if not src.exists(): src = ROOT / 'data' / labels[cid]['display_path']
        im = Image.open(src).convert('RGB')
        im.thumbnail((320, 240)); im.save(t, quality=86)
inv = [x for x in test if x['label'] == 'Invasive']
non = [x for x in test if x['label'] != 'Invasive']
summary = {'n': len(test), 'invasive': len(inv), 'invasive_called': sum(x['top'] == 'Invasive' for x in inv),
           'non_invasive': len(non), 'non_invasive_not_called': sum(x['top'] != 'Invasive' for x in non)}
print('test split', summary)
assert summary == {'n': 132, 'invasive': 31, 'invasive_called': 31, 'non_invasive': 101, 'non_invasive_not_called': 98}, summary
json.dump({'summary': summary, 'cases': test}, open(DATA / 'testset.json', 'w'), indent=0)

meta = PUB / 'captures' / 'meta.json'
if meta.exists():
    shutil.copy(meta, DATA / 'meta.json')
print('staged')

# The hero case's exact region terms (their sum plus the constant is the model's invasive score).
hero = json.load(open(ROOT / 'artifacts' / 'cases' / 'BACH_0167.json'))
json.dump({'case_id': hero['case_id'], 'classes': hero['prediction']['classes'], 'probs': hero['prediction']['probs'],
           'set': hero['prediction']['conformal_set'], 'constant': hero['prediction']['attribution']['constant'],
           'tiles': [{'idx': t['idx'], 'row': t['row'], 'col': t['col'], 'c': t['contribution']} for t in hero['tiles']],
           'grid': hero['image']['grid']}, open(DATA / 'hero.json', 'w'), indent=1)
print('hero staged')
# Narration gain: +8 dB with a limiter so the voice sits about 15 LU above the score under it (see Soundtrack.tsx).
for s in secs['sections']:
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(vo / f"{s['id']}.wav"), '-af',
                    'volume=8dB,alimiter=limit=0.89:attack=2:release=60:level=disabled', '-ar', '48000',
                    str(PUB / 'audio' / 'vo' / f"{s['id']}.wav")], check=True)
