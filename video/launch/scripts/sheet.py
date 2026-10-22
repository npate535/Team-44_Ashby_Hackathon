"""Tile review stills into one image:  python scripts/sheet.py out/stills out/sheet.jpg [cols]"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
src, dst = Path(sys.argv[1]), sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 2
files = sorted(src.glob('t*.jpg'))
w, h = 960, 540
rows = (len(files) + cols - 1) // cols
sheet = Image.new('RGB', (cols * w, rows * (h + 26)), 'black')
d = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    im = Image.open(f).resize((w, h))
    x, y = (i % cols) * w, (i // cols) * (h + 26)
    sheet.paste(im, (x, y + 26)); d.text((x + 8, y + 7), f.stem[1:] + ' s', fill='white')
sheet.save(dst, quality=85)
print(sheet.size)
