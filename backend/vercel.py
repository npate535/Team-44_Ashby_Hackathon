"""Vercel entrypoint: CDN media and a durable, serialized demo workspace in Redis.

Each API request holds a Redis lease and restores the latest SQLite snapshot.
The response is sent only after an atomic lease-checked save, so concurrent
function instances cannot acknowledge conflicting review decisions.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid
import zlib

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.app import create_app
from core import config

ROOT = config.REPO_ROOT
WORK = Path('/tmp/sepia-workspace')
KEY = os.environ.get('SEPIA_WORKSPACE_KEY', 'sepia:workspace:v4')
LEASE_MS = 480_000
SAVE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[2], ARGV[2])
redis.call('DEL', KEYS[1])
return 1
"""
RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""


class Workspace:
    def __init__(self):
        self.local_lock = asyncio.Lock()
        self.inner = None
        self.redis = None

    async def command(self, *args):
        if self.redis is None:
            url = os.environ.get('UPSTASH_REDIS_REST_URL') or os.environ['KV_REST_API_URL']
            token = os.environ.get('UPSTASH_REDIS_REST_TOKEN') or os.environ['KV_REST_API_TOKEN']
            self.redis = httpx.AsyncClient(base_url=url, headers={'Authorization': f'Bearer {token}'}, timeout=30)
        response = await self.redis.post('/', json=list(args))
        response.raise_for_status()
        payload = response.json()
        if payload.get('error'):
            raise RuntimeError('Workspace storage command failed')
        return payload.get('result')

    def initialize(self):
        if self.inner is not None:
            return
        # Only mutable metadata is copied. Tissue images are immutable CDN assets.
        if WORK.exists():
            shutil.rmtree(WORK)
        shutil.copytree(ROOT / 'artifacts', WORK)
        os.environ.setdefault('PATHQUEUE_CURSOR_CLI', '/tmp/sepia-cursor-runtime/cursor-agent')
        os.environ.setdefault('PATHQUEUE_CURSOR_ARCHIVE', str(ROOT / 'cursor-runtime.tar.gz'))
        os.environ.setdefault('PATHQUEUE_CURSOR_HOME', '/tmp/sepia-home')
        os.environ.setdefault('XDG_CACHE_HOME', '/tmp/sepia-cache')
        self.inner = create_app(artifacts_root=WORK, db_path=WORK / 'pathqueue.sqlite', env=os.environ)
        self.backend = self.inner.state.backend
        self.backend.store._conn.execute('PRAGMA journal_mode=DELETE')
        self.case_payloads = [(row['public_json'], row['case_id'])
                              for row in self.backend.store._conn.execute('SELECT case_id, public_json FROM cases')]
        self.evidence_hash = hashlib.sha256(json.dumps(sorted(self.case_payloads)).encode()).hexdigest()
        self.original_files = {str(p.relative_to(WORK)): p.read_text()
                               for p in (WORK / 'interpretations').rglob('*.json')}
        review = WORK / 'eval/spotcheck.json'
        if review.exists():
            self.original_files['eval/spotcheck.json'] = review.read_text()
        manifest = set(json.loads((ROOT / 'media-manifest.json').read_text()))
        original = self.backend.artifact_status

        def artifact_status():
            result = original()
            missing = [asset for asset in result['missing'] if asset not in manifest]
            return {**result, 'missing': missing, 'available': result['referenced'] - len(missing)}

        self.backend.artifact_status = artifact_status
        self.seed_snapshot = self.snapshot()

    def restore(self, snapshot):
        if not snapshot:
            snapshot = self.seed_snapshot
        data = json.loads(zlib.decompress(base64.b64decode(snapshot)))
        if data.get('evidence_hash', self.evidence_hash) != self.evidence_hash:
            raise RuntimeError('Case evidence differs from the saved workspace; retain the existing state and migrate explicitly')
        conn = self.backend.store._conn
        conn.deserialize(base64.b64decode(data['sqlite']))
        conn.executemany('UPDATE cases SET public_json=? WHERE case_id=?', self.case_payloads)
        clock = self.backend.sim.clock_sim
        saved = data['clock']
        clock.running = saved['running']
        clock.speed = saved['speed']
        clock._base = saved['sim_time_s']
        if clock.running:
            clock._base += max(0, time.time() - saved['wall']) * clock.speed
            clock._started = self.backend.sim.clock.mono()
        else:
            clock._started = None
        current_files = {str(p.relative_to(WORK)): p for p in (WORK / 'interpretations').rglob('*.json')}
        current_files['eval/spotcheck.json'] = WORK / 'eval/spotcheck.json'
        for name, target in current_files.items():
            if name not in data.get('files', {}):
                if name in self.original_files:
                    if target.read_text() != self.original_files[name]:
                        target.write_text(self.original_files[name])
                elif target.exists():
                    target.unlink()
        for name, contents in data.get('files', {}).items():
            target = WORK / name
            if not target.resolve().is_relative_to(WORK.resolve()):
                raise RuntimeError('Invalid workspace file')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)

    def snapshot(self):
        sim = self.backend.sim
        # Persist mutable reviews and generated interpretations together with decisions.
        files = {'eval/spotcheck.json': (WORK / 'eval/spotcheck.json').read_text()} if (WORK / 'eval/spotcheck.json').exists() else {}
        files.update({str(p.relative_to(WORK)): p.read_text() for p in (WORK / 'interpretations').rglob('*.json')})
        files = {name: text for name, text in files.items() if self.original_files.get(name) != text}
        # Case evidence is immutable and already bundled. Persist only live state,
        # removing duplicate evidence from a temporary backup before compression.
        with sqlite3.connect(':memory:', isolation_level=None) as compact:
            self.backend.store._conn.backup(compact)
            compact.execute("UPDATE cases SET public_json=''")
            compact.execute('VACUUM')
            sqlite = base64.b64encode(compact.serialize()).decode()
        data = {'sqlite': sqlite, 'evidence_hash': self.evidence_hash,
                'clock': {'running': sim.clock_sim.running, 'speed': sim.clock_sim.speed,
                          'sim_time_s': sim.sim_now(), 'wall': time.time()}, 'files': files}
        return base64.b64encode(zlib.compress(json.dumps(data, separators=(',', ':')).encode(), 3)).decode()

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return
        async with self.local_lock:
            owner = uuid.uuid4().hex
            acquired = False
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if await self.command('SET', KEY + ':lock', owner, 'NX', 'PX', LEASE_MS):
                        acquired = True
                        break
                    await asyncio.sleep(0.1)
                if not acquired:
                    await JSONResponse({'detail': 'Workspace is busy; please retry.'}, status_code=503)(scope, receive, send)
                    return
                self.initialize()
                self.restore(await self.command('GET', KEY))
                messages = []

                async def buffer(message):
                    messages.append(message)

                await self.inner(scope, receive, buffer)
                value = self.snapshot()
                if not await self.command('EVAL', SAVE, 2, KEY + ':lock', KEY, owner, value):
                    raise RuntimeError('Workspace lease expired before save')
                acquired = False
                for message in messages:
                    await send(message)
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Durable workspace request failed')
                await JSONResponse({'detail': 'Workspace storage is unavailable; please retry.'}, status_code=503)(scope, receive, send)
            finally:
                if acquired:
                    await self.command('EVAL', RELEASE, 1, KEY + ':lock', owner)


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount('/', Workspace())
