"""Markdown table (Japanese) of the evaluation results for every checkpoint.
Usage: python eval_table.py [v2|v1|v3]   (v1 = v1 policy in the V1 sim, outputs/eval/act_peel_v1sim_<ckpt>;
v3 = the improved V2 in sim_v3, outputs/eval/act_peel_v3_<ckpt>)"""
import json
import sys
from pathlib import Path

ROOT = Path.home() / "apple_peel/outputs/eval"
VER = sys.argv[1] if len(sys.argv) > 1 else "v2"
PREFIX = {"v2": "act_peel_v2_", "v1": "act_peel_v1sim_", "v3": "act_peel_v3_"}[VER]
CKPTS = [(ck, PREFIX + ck) for ck in ("020000", "040000", "060000", "080000", "100000")]
FPS = 30


def sec(f):
    return "-" if f is None else f"{f / FPS:.0f}秒"


TITLE = {"v2": "ACT v2 評価結果（V2シミュレータ", "v1": "ACT v1 評価結果（V1シミュレータ",
         "v3": "ACT v3（V2改良版）評価結果（V3シミュレータ、最大6分"}[VER]
print(f"# {TITLE}、seed 5000〜5005、学習に使っていない配置）\n")
print("| チェックポイント | seed | 判定 | 刺せた時刻 | むけた割合 | 90%むけた時刻 | 動画 |")
print("|---|---|---|---|---|---|---|")
summary = []
for ck, d in CKPTS:
    n_ok, n_done = 0, 0
    for s in range(5000, 5006):
        f = ROOT / d / f"eval_seed{s}.json"
        if not f.exists():
            print(f"| {ck} | {s} | (未評価) | | | | |")
            continue
        r = json.loads(f.read_text())
        ok = r["success_frame"] is not None
        n_ok += ok
        n_done += 1
        print(f"| {ck} | {s} | {'成功' if ok else '失敗'} | {sec(r['skewered_frame']) if r['skewered'] else '刺せず'} | "
              f"{r['peeled'] * 100:.1f}% | {sec(r['peeled_90_frame'])} | {ck}\\eval_seed{s}.mp4 |")
    summary.append((ck, n_ok, n_done))
print("\n## まとめ\n")
for ck, n, done in summary:
    print(f"- {ck} ステップ：" + (f"成功 {n} / {done}" if done else "未評価"))
