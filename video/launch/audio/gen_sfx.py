"""Generate the sound-effect kit with ElevenLabs sound generation.   python audio/gen_sfx.py public/audio/sfx"""
import json, sys, urllib.request, urllib.error, concurrent.futures as cf
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
key = next(l.split('=', 1)[1].strip() for l in open(Path.home() / '.config/ashby-2026/elevenlabs.env') if l.startswith('ELEVENLABS_API_KEY='))
KIT = {
    'whoosh_soft':  (0.9, 'soft airy whoosh, smooth swipe transition for a modern product video, clean, warm, no hiss'),
    'whoosh_deep':  (1.4, 'deep smooth cinematic whoosh pass-by, velvety low air movement, subtle, premium'),
    'swoosh_up':    (0.7, 'quick rising swoosh, sleek and light, modern interface transition'),
    'glass_swipe':  (1.0, 'glassy airy swipe transition with a soft sparkle tail, smooth, elegant'),
    'click':        (0.3, 'single soft mouse click, crisp modern trackpad tap, very short, clean'),
    'pop':          (0.4, 'soft pleasant UI pop, rounded bubbly blip, short notification'),
    'tick':         (0.25, 'tiny clean digital tick, short, subtle interface sound'),
    'riser':        (3.5, 'smooth cinematic riser building tension, airy synth swell, rising pitch, ends cleanly'),
    'impact':       (3.0, 'deep soft cinematic impact boom with a long shimmering reverb tail, premium logo reveal'),
    'shimmer':      (2.0, 'light magical shimmer, glassy chime sparkle, subtle and warm'),
    'stamp':        (0.6, 'solid satisfying stamp thud, short and weighty, clean'),
    'count':        (1.6, 'fast soft digital counting ticks, numbers rolling up, subtle interface'),
    'chime':        (1.2, 'gentle positive confirmation chime, two rising notes, modern app success'),
    'clock':        (3.0, 'close-miked mechanical clock ticking, crisp clear ticks, two ticks per second, dry studio recording'),
    'sub_hit':      (1.5, 'clean low sub bass hit, short, round, cinematic'),
    'reverse_swell':(2.0, 'reverse cymbal swell rising into a hit, smooth, airy'),
    'alert':        (0.9, 'soft warning tone, two descending notes, calm medical interface alert, not harsh'),
    'paper_slide':  (0.7, 'smooth card slide on a table, soft quick friction, light'),
    'pad_tail':     (9.0, 'warm soft ambient synth pad, sustained airy chord with gentle shimmer, slowly fading, calm cinematic ending'),
}

def gen(item):
    name, (dur, prompt) = item
    p = out / f'{name}.mp3'
    if p.exists(): return name, 'cached'
    body = json.dumps({'text': prompt, 'duration_seconds': max(dur, 0.5), 'prompt_influence': 0.55}).encode()
    req = urllib.request.Request('https://api.elevenlabs.io/v1/sound-generation?output_format=mp3_44100_192', data=body,
                                 headers={'xi-api-key': key, 'Content-Type': 'application/json'})
    try:
        p.write_bytes(urllib.request.urlopen(req, timeout=180).read())
    except urllib.error.HTTPError as e:
        return name, f'HTTP {e.code} {e.read()[:200]}'
    return name, 'ok'

with cf.ThreadPoolExecutor(4) as ex:
    for name, st in ex.map(gen, KIT.items()): print(name, st, flush=True)
