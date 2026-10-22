"""Level-match the sound-effect kit: each effect's loudest 50 ms is brought to -14 dBFS RMS, then limited at -1.5 dBFS.

    python audio/normalize_sfx.py public/audio/sfx public/audio/sfx_n
"""
import subprocess, sys
from pathlib import Path
import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for f in sorted(src.glob('*.mp3')):
    x = np.frombuffer(subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(f), '-ac', '1', '-ar', '48000', '-f', 'f32le', '-']), dtype=np.float32)
    fr = 2400
    rms = max(np.sqrt((x[i:i + fr] ** 2).mean()) for i in range(0, max(1, len(x) - fr), fr))
    gain = -14 - 20 * np.log10(rms + 1e-9)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(f), '-af', f'volume={gain:.2f}dB,alimiter=limit=0.84:attack=1:release=40:level=disabled',
                    '-ar', '48000', '-ac', '2', str(dst / f'{f.stem}.wav')], check=True)
    print(f'{f.stem:14s} gain {gain:+6.1f} dB')
