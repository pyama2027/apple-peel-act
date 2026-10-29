#!/bin/bash
# expert_batch.sh FIRST COUNT -> run the full-diversity peel expert on seeds FIRST.. (no video, parallel) + summary
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/apple_peel/sim
out=~/apple_peel/outputs/batch_$1; rm -rf $out; mkdir -p $out
for ((s=$1; s<$1+$2; s++)); do
  python run_demo.py --task peel --no-video --seed $s --out $out > /dev/null 2> $out/err_$s.txt &
done
wait
python - "$out" <<'EOF'
import glob, json, sys
rs = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/*.json"))]
for r in rs:
    print(r["seed"], "OK" if r["success"] else "NG", r["aborted"], "peeled", r.get("peeled"), "s", round(r["frames"] / 30),
          "miss", r.get("missed_skewer"), "apple_drop", r.get("drops"), "peeler_drop", r.get("peeler_drop") is not None,
          "rises", r.get("peeler_rise_mm"), "poff", r["scene"].get("peeler_offset"))
print("success", sum(r["success"] for r in rs), "/", len(rs))
EOF
grep -l Traceback $out/err_*.txt | head -3
