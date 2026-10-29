"""Minimal MP4 writer (PyAV, H.264)."""
import av
import numpy as np


class VideoWriter:
    def __init__(self, path, fps, width, height):
        self.c = av.open(str(path), mode="w")
        self.s = self.c.add_stream("libx264", rate=fps)
        self.s.width, self.s.height, self.s.pix_fmt = width, height, "yuv420p"
        self.s.options = {"crf": "20"}

    def write(self, rgb: np.ndarray):
        for pkt in self.s.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            self.c.mux(pkt)

    def close(self):
        for pkt in self.s.encode():
            self.c.mux(pkt)
        self.c.close()
