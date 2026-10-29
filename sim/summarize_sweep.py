"""Summarize run_demo.py JSON results in a directory. Usage: python summarize_sweep.py DIR"""
import json
import sys
from pathlib import Path

rs = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("demo_seed*.json"))]
ok = [r for r in rs if r["success"]]
print(f"success {len(ok)}/{len(rs)}  skewered {sum(r['skewered'] for r in rs)}/{len(rs)}")
for k in ("max_pen_apple_robot_mm", "max_pen_robot_table_mm", "max_ik_err_mm"):
    v = [r[k] for r in ok]
    if v:
        print(f"[successful eps] {k:24s} max={max(v):6.2f} mean={sum(v) / len(v):6.2f}")
for r in rs:
    if not r["success"]:
        print("FAIL", r["seed"], r["scene"]["apple_xy"], "aborted:", r.get("aborted"), "skewered", r["skewered"],
              "ik", r["max_ik_err_mm"], "trace_end", r.get("insert_trace_dxy_depth_mm", [])[-1:])
