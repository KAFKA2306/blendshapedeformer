# blendshapedeformer — BlendShape転送の検証基準と研究構想

**リポジトリ:** https://github.com/KAFKA2306/blendshapedeformer

VRChatアバターのViseme・表情BlendShape生成を研究するリポジトリです。現在は、機械学習や異種トポロジ転送ではなく、**同一トポロジ・同一頂点順のメッシュ間で頂点オフセットを移す決定論的ベースライン**を実装しています。

## 現在実行できる処理

`src/blendshape_transfer.py`は次を行います。

1. 基準メッシュ、変形済みメッシュ、転送先基準メッシュを`(N, 3)`配列として検証
2. NaN、無限値、頂点数不一致を拒否
3. 面配列がある場合は、面数・頂点インデックス・順序の完全一致を検証
4. `source_shape - source_base`から頂点オフセットを計算
5. 同じオフセットを`target_base`へ適用
6. 最大移動量、平均移動量、トポロジSHA-256を記録
7. NPZとJSONメタデータを出力

この処理は機械学習ではなく、異なるトポロジへは使用できません。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Windowsでは仮想環境の有効化コマンドを環境に合わせて変更してください。

## 入力形式

NPZへ次の配列を保存します。

```text
source_base   float[N, 3]
source_shape  float[N, 3]
target_base   float[N, 3]
source_faces  int[M, K]  # 推奨・CLIでは検証対象
target_faces  int[M, K]
```

例:

```python
import numpy as np

np.savez_compressed(
    "input.npz",
    source_base=source_base,
    source_shape=source_shape,
    target_base=target_base,
    source_faces=source_faces,
    target_faces=target_faces,
)
```

実行:

```bash
python -m src.blendshape_transfer \
  input.npz output.npz \
  --confirmed-same-topology \
  --max-displacement 0.05
```

`--max-displacement`の単位は入力座標と同じです。メートル、センチメートルなどをコードが自動判定しないため、アセットの単位に合わせて明示してください。

出力:

```text
output.npz
  target_shape
  offsets
  target_faces  # 入力に存在した場合

output.npz.json
  method
  vertex_count
  topology_sha256
  maximum_displacement
  mean_displacement
  generated_at_utc
```

## API

```python
from src.blendshape_transfer import transfer_same_topology

target_shape, offsets, metadata = transfer_same_topology(
    source_base,
    source_shape,
    target_base,
    source_faces=source_faces,
    target_faces=target_faces,
    max_displacement=0.05,
)
```

面配列を渡さない場合、呼出側が頂点順・面順を別手段で検証した上で、`confirmed_same_topology=True`を明示する必要があります。

## この実装が保証しないこと

- 異なる頂点数・面構成への転送
- 顔ランドマークの自動対応
- Blender Shape Keyへの直接書込み
- Unity FBXインポート
- VRChat Viseme名への割当
- 歯、舌、口内、まぶたとの干渉回避
- 見た目の自然さ
- 販売品質や公式対応

同一トポロジでも、基準形状の比率・骨格・法線・口内構造が違えば、同じオフセットで自然な表情になるとは限りません。

## 将来の研究対象

異なるトポロジへ進む場合は、単純な頂点番号対応を使用できません。少なくとも次が必要です。

- 顔領域のランドマーク対応
- 表面上の対応点推定
- UV、距離場、局所座標による特徴表現
- 口内、歯、舌、まぶたの構造差への対応
- 学習・検証・OOSテストの分離
- 頂点誤差だけでなく、口唇閉鎖やViseme認識の評価
- Blender、Unity、VRChat内での実動作検証

## 必要な将来成果物

```text
data/
  manifest.json
models/
  model.onnx
  model-card.md
training/
  dataset.py
  train.py
  evaluate.py
blender-addon/
unity/
tests/
```

現時点では、学習データ、学習済みモデル、訓練コード、Blenderアドオン、Unity統合パッケージは未実装です。

## 安全・権利上の注意

- 購入アバターのメッシュやShape Keyを再配布しないでください
- アバターごとの利用規約と機械学習利用可否を確認してください
- 自動生成結果を公式対応・完全互換として販売しないでください
- 顔表現は視覚品質だけでなく、リップシンクやアクセシビリティにも影響します

**README最終監査:** 2026-08-02
