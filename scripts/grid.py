"""grid.py in.mp4 out.png cols t1 t2 ... -> contact sheet of half-size frames at the given times [s]"""
import sys

import av
from PIL import Image

path, out, cols, times = sys.argv[1], sys.argv[2], int(sys.argv[3]), sorted(float(x) for x in sys.argv[4:])
fr, want = [], list(times)
with av.open(path) as c:
    for f in c.decode(video=0):
        if want and f.time >= want[0] - 1e-6:
            fr.append(f.to_image().resize((f.width // 2, f.height // 2)))
            want.pop(0)
        if not want:
            break
w, h = fr[0].size
rows = (len(fr) + cols - 1) // cols
g = Image.new("RGB", (w * cols, h * rows))
for i, im in enumerate(fr):
    g.paste(im, ((i % cols) * w, (i // cols) * h))
g.save(out)
print(out, len(fr))
