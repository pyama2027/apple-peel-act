# ロボットアームによるりんごの皮むき：模倣学習（ACT）で長い作業手順を学ぶ — コード一式（V2まで）

MuJoCo シミュレーションの中で、SO-101（5自由度アーム＋グリッパ）に「りんごを針に刺す → T字ピーラーで皮をむく → りんごを回す」を4面分行わせるお手本動作（エキスパート）を作り、LeRobot の ACT で模倣学習させたコードです。

## フォルダ構成

| フォルダ | 内容 |
|---|---|
| `sim/` | **V2**（最終版）のシミュレーション・お手本動作・録画・評価のコード |
| `sim_v1/` | **V1**（失敗からの回復なし、柵なし）を再現するためのコード |
| `sim/models/so101/` | ロボットのモデル（SO-ARM100 由来、LICENSE 同梱）と作業シーン `peel_scene.xml` |
| `scripts/` | 録画の並列実行、学習、チェックポイント評価、結果表の作成に使ったシェルスクリプト |

## 主なファイル（`sim/`）

| ファイル | 役割 |
|---|---|
| `build_peel_scene.py` | 作業シーン（机・柵・針・りんご・ピーラー台）を生成して `peel_scene.xml` を書き出す |
| `design_finger_ext.py` | 大きなりんごをつかむための指の延長部品の寸法を決める |
| `task.py` | シミュレーション環境（観測・行動・刺さり判定・皮むき判定） |
| `ik.py` | 逆運動学 |
| `expert.py` | お手本動作：りんごをつかんで針に刺す（落下・刺し損ねからの回復を含む） |
| `expert_peel.py` | お手本動作：ピーラーを取る・むく・戻す・りんごを回す（ピーラー落下からの回復を含む） |
| `run_demo.py` | お手本を1回実行して動画を作る（配置や失敗パターンのランダム化もここ） |
| `record_dataset.py` / `merge_shards.py` | LeRobot 形式のデータセットを並列で録画し、1つに結合する |
| `eval_policy.py` | 学習した ACT をシミュレーションで実行して評価する（動画付き） |
| `make_final_video.py` | 評価動画を倍速にして提出用動画を作る |
| `probe_*.py` / `check_*.py` | 届く姿勢・握り方・刃の当て方などを調べるために使った検証用スクリプト |

## 動作環境

- WSL2 Ubuntu 22.04、GPU RTX 4060（学習）、Python 3.12（conda 環境 `lerobot`）
- LeRobot v0.6.1-188（commit d3793b7e）、MuJoCo 3.17.0
- 描画は `LIBGL_ALWAYS_SOFTWARE=1` を指定（WSL で並列描画を速くするため）

## 使い方（例）

```bash
conda activate lerobot
export LIBGL_ALWAYS_SOFTWARE=1
cd sim

# シーン生成
python build_peel_scene.py

# お手本を1回実行して動画を確認
python run_demo.py --task peel --seed 3102

# データセット録画（10並列 x 5エピソード、その後に結合）
for i in $(seq 0 9); do
  python record_dataset.py --task peel --episodes 5 --max-tries 15 --seed0 4000 --shard $i --num-shards 10 &
done; wait
python merge_shards.py --task peel

# 学習（ACT、バッチ8、10万ステップ、2万ステップごとに保存）
cd ..
lerobot-train --dataset.repo_id=local/so101_peel_sim --dataset.root=$PWD/data/so101_peel_sim \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --steps=100000 --save_freq=20000 --num_workers=8 \
  --output_dir=outputs/train/act_peel_v2 --job_name=act_peel_v2 --wandb.enable=false

# 評価（学習に使っていない配置 seed 5000〜5005、動画付き）
cd sim
python eval_policy.py --policy ../outputs/train/act_peel_v2/checkpoints/080000/pretrained_model \
  --seeds 5001 --max-frames 7500 --video --out ../outputs/eval/act_peel_v2_080000
```

データセット（約1.3 GB）と学習済みモデルは容量が大きいため含めていません。上の手順で再生成できます。
