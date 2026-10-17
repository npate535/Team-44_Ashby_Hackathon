#!/usr/bin/env python3
"""Call an organizer model from the Delta network without exposing API keys.

--health is read-only. Otherwise embeds images, or classifies with --prompts.
"""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', choices=['conch', 'musk', 'quiltnet', 'plip'], required=True)
p.add_argument('--health', action='store_true')
p.add_argument('--remote-images', action='store_true', help='Image paths are on Delta instead of this Mac')
p.add_argument('--prompts', type=Path, help='JSON label-to-description mapping for /classify')
p.add_argument('images', nargs='*')
a = p.parse_args()
if not a.health and not a.images:
    p.error('provide image paths or --health')
if a.health and (a.images or a.prompts):
    p.error('--health cannot be combined with inference inputs')
secrets = Path.home() / '.config/ashby-2026/model-servers.env'
if not secrets.is_file():
    sys.exit('Private model credentials are not configured on this machine.')
values = {}
for line in secrets.read_text().splitlines():
    words = shlex.split(line)
    if words and words[0] == 'export':
        words = words[1:]
    if len(words) == 1 and '=' in words[0]:
        k, v = words[0].split('=', 1)
        values[k] = v
prefix = a.model.upper()
url = values.get(prefix + '_URL')
key = values.get(prefix + '_API_KEY')
if not url or (not a.health and not key):
    sys.exit('Selected model endpoint or credential is missing.')
payload = {'url': url, 'health': a.health}
if not a.health:
    payload['key'] = key
    if a.remote_images:
        payload['paths'] = a.images
    else:
        payload['images'] = [base64.b64encode(Path(i).read_bytes()).decode('ascii') for i in a.images]
    if a.prompts:
        prompts = json.loads(a.prompts.read_text())
        if not isinstance(prompts, dict) or not prompts or not all(isinstance(k, str) and isinstance(v, str) for k, v in prompts.items()):
            sys.exit('Prompts must be a nonempty JSON object mapping string labels to descriptions.')
        payload['prompts'] = prompts
remote_code = r'''
import base64, json, sys, urllib.request, urllib.error
from pathlib import Path
d = json.load(sys.stdin)
try:
    if d['health']:
        req = urllib.request.Request(d['url'].rstrip('/') + '/health')
    else:
        images = d.get('images')
        if images is None:
            images = [base64.b64encode(Path(p).read_bytes()).decode('ascii') for p in d['paths']]
        body = {'images': images}
        endpoint = '/embed'
        if 'prompts' in d:
            body['prompts'] = d['prompts']
            endpoint = '/classify'
        req = urllib.request.Request(d['url'].rstrip('/') + endpoint, data=json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + d['key'], 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        print(response.read().decode())
except urllib.error.HTTPError as e:
    sys.exit('Model HTTP error ' + str(e.code) + '; check current endpoint/key and model permissions.')
except (OSError, urllib.error.URLError) as e:
    sys.exit('Model endpoint or input file unavailable: ' + str(e.reason if hasattr(e, 'reason') else e))
'''
helper = Path(__file__).resolve().parent / 'delta.sh'
result = subprocess.run([str(helper), shlex.join(['python3', '-c', remote_code])], input=json.dumps(payload), text=True)
sys.exit(result.returncode)
