"""Generate the score with Eleven Music from a section plan timed to the cut.   python audio/gen_music.py public/audio/music"""
import json, sys, urllib.request, urllib.error
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
key = next(l.split('=', 1)[1].strip() for l in open(Path.home() / '.config/ashby-2026/elevenlabs.env') if l.startswith('ELEVENLABS_API_KEY='))
G = ['instrumental', 'modern tech product launch film score', 'premium and minimal', 'warm analog synths', 'felt piano',
     'clean sub bass', '100 BPM', 'leaves room for voice-over', 'polished mix']
NEG = ['vocals', 'singing', 'choir', 'lyrics', 'aggressive EDM drop', 'dubstep', 'distorted guitar', 'cheesy corporate ukulele']
plan = {'positive_global_styles': G, 'negative_global_styles': NEG, 'sections': [
    {'section_name': 'Title', 'duration_ms': 4000, 'lines': [],
     'positive_local_styles': ['airy swell', 'reverse cymbal rising', 'soft deep impact on the last beat'], 'negative_local_styles': ['drums']},
    {'section_name': 'Problem', 'duration_ms': 8200, 'lines': [],
     'positive_local_styles': ['sparse and tense', 'low muted pulse', 'single held synth note', 'quiet ticking hi-hat'], 'negative_local_styles': ['melody', 'full drums']},
    {'section_name': 'Reveal', 'duration_ms': 14200, 'lines': [],
     'positive_local_styles': ['warm synth arpeggio enters', 'steady pulse', 'optimistic', 'soft kick'], 'negative_local_styles': ['loud']},
    {'section_name': 'Deep dive', 'duration_ms': 30600, 'lines': [],
     'positive_local_styles': ['focused and curious', 'evolving arpeggio', 'light percussion', 'subtle tension and release'], 'negative_local_styles': ['busy lead melody']},
    {'section_name': 'Workflow', 'duration_ms': 13000, 'lines': [],
     'positive_local_styles': ['more momentum', 'tighter rhythm', 'building'], 'negative_local_styles': ['drop']},
    {'section_name': 'Results', 'duration_ms': 15000, 'lines': [],
     'positive_local_styles': ['lift', 'confident and full', 'crisp drums', 'wide pads', 'uplifting'], 'negative_local_styles': ['harsh']},
    {'section_name': 'Outro', 'duration_ms': 12000, 'lines': [],
     'positive_local_styles': ['resolve', 'gentle felt piano', 'pads fading out', 'calm ending'], 'negative_local_styles': ['drums']},
]}
for name, body in [('score_plan', {'composition_plan': plan, 'model_id': 'music_v1'}),
                   ('score_prompt', {'prompt': 'Instrumental modern tech product launch film score, premium and minimal, warm analog synths, felt piano, '
                                     'clean sub bass, 100 BPM. 0-4s airy swell into a soft impact; 4-12s sparse and tense; 12s a warm arpeggio '
                                     'enters, optimistic steady pulse; builds through the middle; 70s lifts into a confident full section with crisp '
                                     'drums; 85-97s resolves with gentle piano and fading pads. No vocals.', 'music_length_ms': 97000, 'model_id': 'music_v1'})]:
    p = out / f'{name}.mp3'
    if p.exists(): print(name, 'cached'); continue
    req = urllib.request.Request('https://api.elevenlabs.io/v1/music?output_format=mp3_44100_192', data=json.dumps(body).encode(),
                                 headers={'xi-api-key': key, 'Content-Type': 'application/json'})
    try:
        p.write_bytes(urllib.request.urlopen(req, timeout=600).read()); print(name, 'ok', flush=True)
    except urllib.error.HTTPError as e:
        print(name, 'HTTP', e.code, e.read()[:400], flush=True)
