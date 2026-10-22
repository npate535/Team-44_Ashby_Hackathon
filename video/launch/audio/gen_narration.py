"""Generate the narration as one ElevenLabs take with character timestamps, then cut it into per-section clips.

    python audio/gen_narration.py <voice_id> <out_dir>

Reads ELEVENLABS_API_KEY from ~/.config/ashby-2026/elevenlabs.env (never printed). Writes take.mp3, take.json
(alignment), sections.json (start/end per section) and one <id>.wav per section.
"""
import base64, json, os, re, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
voice, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
key = next(l.split('=', 1)[1].strip() for l in open(Path.home() / '.config/ashby-2026/elevenlabs.env') if l.startswith('ELEVENLABS_API_KEY='))
spec = json.load(open(HERE / 'narration.json'))
SEP = '\n\n'
text = SEP.join(s['text'] for s in spec['sections'])

if not (out / 'take.json').exists():
    body = json.dumps({'text': text, 'model_id': spec['model_id'],
                       'voice_settings': {'stability': 0.5, 'similarity_boost': 0.8, 'style': 0.35, 'use_speaker_boost': True},
                       'seed': int(os.environ.get('SEED', '44'))}).encode()
    req = urllib.request.Request(f'https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_192',
                                 data=body, headers={'xi-api-key': key, 'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    (out / 'take.mp3').write_bytes(base64.b64decode(d['audio_base64']))
    json.dump({'alignment': d['alignment'], 'normalized_alignment': d.get('normalized_alignment'), 'text': text},
              open(out / 'take.json', 'w'))

d = json.load(open(out / 'take.json'))
al = d['alignment']
chars, st, en = al['characters'], al['character_start_times_seconds'], al['character_end_times_seconds']
src = ''.join(chars)
assert src == text, 'alignment text differs from request'

# Section spans in the request text, then spoken start/end from the first/last non-tag, non-space character.
spans, pos = [], 0
for s in spec['sections']:
    i = text.index(s['text'], pos); spans.append((i, i + len(s['text']))); pos = i + len(s['text'])

def spoken(i0, i1):
    t = text[i0:i1]
    mask = [True] * len(t)
    for m in re.finditer(r'\[[^\]]*\]', t):
        for k in range(m.start(), m.end()): mask[k] = False
    idx = [i0 + k for k, c in enumerate(t) if mask[k] and c.isalnum()]
    return st[idx[0]], en[idx[-1]], idx

dur = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(out / 'take.mp3')]))
secs = []
for (i0, i1), s in zip(spans, spec['sections']):
    a, b, idx = spoken(i0, i1)
    # words with times, for on-screen sync
    words, cur = [], None
    for k in range(i0, i1):
        c = text[k]
        if k in set(idx) or (cur and c in "'-"):
            if cur is None: cur = {'w': '', 't0': st[k]}
            cur['w'] += c; cur['t1'] = en[k]
        elif cur is not None:
            words.append(cur); cur = None
    if cur: words.append(cur)
    secs.append({'id': s['id'], 'start': a, 'end': b, 'words': words})

# cut points: midway in the silence between sections; clips keep a little natural breath on each side
for j, s in enumerate(secs):
    lo = 0.0 if j == 0 else (secs[j - 1]['end'] + s['start']) / 2
    hi = dur if j == len(secs) - 1 else (s['end'] + secs[j + 1]['start']) / 2
    lo = max(lo, s['start'] - 0.12); hi = min(hi, s['end'] + 0.35)
    s['clip_start'], s['clip_end'] = lo, hi
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{lo:.3f}', '-to', f'{hi:.3f}', '-i', str(out / 'take.mp3'),
                    '-af', 'afade=t=in:d=0.04,areverse,afade=t=in:d=0.08,areverse', '-ar', '48000', str(out / f"{s['id']}.wav")], check=True)
json.dump({'voice': voice, 'model_id': spec['model_id'], 'take_duration': dur, 'sections': secs}, open(out / 'sections.json', 'w'), indent=1)
for s in secs:
    print(f"{s['id']:9s} {s['start']:6.2f}-{s['end']:6.2f}  speech {s['end']-s['start']:5.2f}s  clip {s['clip_end']-s['clip_start']:5.2f}s")
print(f'take {dur:.2f}s, speech sum {sum(s["end"]-s["start"] for s in secs):.2f}s')
