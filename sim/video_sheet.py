"""Make a contact sheet of frames at given times from an MP4. Usage: python video_sheet.py in.mp4 out.png 1.0 2.5 ..."""
import sys

import av
import numpy as np
from PIL import Image

path, out, times = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:]]
frames, want = [], sorted(times)
with av.open(path) as c:
    for f in c.decode(video=0):
        if want and f.time is not None and f.time >= want[0] - 1e-6:
            frames.append(f.to_ndarray(format="rgb24"))
            want.pop(0)
Image.fromarray(np.concatenate(frames, axis=0)).save(out)
print("saved", out, len(frames), "frames")
