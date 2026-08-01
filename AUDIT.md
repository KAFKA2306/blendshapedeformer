# blendshapedeformer 実装監査

**監査日:** 2026-08-02

## 結論

このリポジトリは、VRChat向けBlendShape自動生成の設計・構想資料です。次の実行可能成果物は確認できませんでした。

- 学習データのmanifest
- 訓練・検証・OOS評価コード
- 学習済みモデルまたはONNXモデル
- モデルカード
- Blenderアドオン実装
- Unity Editor拡張
- VRChat SDK統合
- 自動テスト
- 精度・処理時間・互換性の再現可能な証拠

README中のPython・C#断片は設計例であり、リポジトリ内でimport、build、実行できる製品コードではありません。

## 採用禁止の主張

証拠が追加されるまで、次を仕様・実績として扱いません。

- 完全自動生成
- トポロジ非依存
- 数分で生成
- 品質保証
- 100体で事前学習
- 6時間以内で新アバターへ適応
- 手動制作より高品質
- VRChat完全互換

## 実装開始前の必須条件

1. 学習対象アバター・Shape Keyの利用許諾を記録する
2. データセットをtrain / validation / frozen testへ分離する
3. 同一トポロジと異なるトポロジの問題を別タスクにする
4. 頂点順、座標系、単位、法線、UV、顔部位対応を検証する
5. ベースラインとして線形補間、ランドマーク、RBF等を実装する
6. モデル出力へ有限値、最大変位、自己交差、左右対称性のゲートを置く
7. 未学習アバターでOOS評価する
8. Blender保存・再読込、FBX出力、Unity読込、VRChat実動作を別々に確認する
9. モデル、データ、プロンプト、設定、結果へハッシュを付ける
10. 人間による表情自然さとViseme可読性を評価する

## 推奨ディレクトリ契約

```text
data/manifest.json
training/dataset.py
training/train.py
training/evaluate.py
models/model.onnx
models/model-card.md
blender-addon/
unity/Editor/
tests/
evidence/
```

上記が揃うまで、他リポジトリの正式衣装・アバター制作パイプラインへ依存関係として追加しません。
