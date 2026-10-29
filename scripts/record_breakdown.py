"""Breakdown of the v2 recording logs: what the successful (kept) episodes contain."""
import glob
import json
import os

rs = []
for f in glob.glob(os.path.expanduser("~/apple_peel/outputs/record_v2/shard*.log")):
    rs += [json.loads(l) for l in open(f) if l.startswith('{"seed"')]
ok = [r for r in rs if r["success"]]
print(f"tried {len(rs)}  kept {len(ok)}  frames kept {sum(r['frames'] for r in ok)}")
print("kept with apple drop+repick:", sum(r["drops"] > 0 for r in ok))
print("kept with skewer miss+retry:", sum(r.get("misses", 0) > 0 for r in ok))
print("kept with peeler drop+repick:", sum(r["peeler_drops"] > 0 for r in ok))
print("kept with none of these:", sum(r["drops"] == 0 and r.get("misses", 0) == 0 and r["peeler_drops"] == 0 for r in ok))
if ok:
    print("peeled mean %.3f min %.3f" % (sum(r["peeled"] for r in ok) / len(ok), min(r["peeled"] for r in ok)))
