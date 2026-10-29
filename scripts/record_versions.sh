#!/bin/bash
# Record exact commits of third-party repos for reproducibility.
cd "$(dirname "$0")/../third_party"
for r in SO-ARM100 lerobot; do
  echo "$r $(git -C "$r" rev-parse HEAD) $(git -C "$r" log -1 --format=%cs)"
done > VERSIONS.txt
cat VERSIONS.txt
