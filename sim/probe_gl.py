"""Print the OpenGL renderer GLFW gets on this display (debug helper for WSLg viewer issues)."""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
import glfw
from OpenGL import GL

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
w = glfw.create_window(64, 64, "probe", None, None)
glfw.make_context_current(w)
for k in ("VENDOR", "RENDERER", "VERSION"):
    v = GL.glGetString(getattr(GL, f"GL_{k}"))
    print(f"{k:8s}", v.decode() if v else None)
glfw.terminate()
