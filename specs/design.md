# design.md（ドラフト）

> 本設計書は requirements.md に定義した WHY/WHAT を実現するための **HOW** を示します。タスク分割や品質保証手順は tasks.md／test-plan.md に委譲します。

## システム全体像

```
┌──── Blender Add-on ───┐          ┌──── Unity Editor Tool ──┐
│  UI & Operator        │          │  Avatar Integration     │
│  ├─ Mesh Validator    │          │  ├─ BlendShape Merger   │
│  ├─ Inference Client  │──gRPC──▶ │  ├─ Viseme Mapper       │
│  └─ QA / Auto-Fix     │          │  └─ FX Animator Builder │
└────────┬──────────────┘          └──────────┬──────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────────────────────────────────────┐
│    Core Inference Server (ONNX Runtime)        │
│  ├─ Model Loader & Cache                       │
│  ├─ Vertex Offset Predictor                    │
│  ├─ Post-Process (Clamp & Smooth)              │
│  └─ REST/gRPC API (localhost ONLY)             │
└──────────────┬─────────────────────────────────┘
               │
               ▼
        GPU / CPU (TensorRT or fallback)
```

### 技術スタック
- Python3.11（Inference Server）
- ONNX Runtime + TensorRT / OpenVINO
- gRPC (Blender⇔Server)
- C# Editor拡張（Unity 2019.4LTS/2022.3LTS）
- Blender 3.0+ アドオン (bpy)

## モジュール設計

### 1. Core Inference Server

| サブモジュール | 責務 | 主要クラス / ファイル |
| --- | --- | --- |
| ModelManager | ONNXモデルのロード／ホットスワップ | `model_manager.py` |
| Predictor | バッチ推論・量子化モード切替 | `predictor.py` |
| PostProcessor | 極端変形のクランプ・Laplace平滑化 | `postprocess.py` |
| API Server | gRPCサーバー・ヘルスチェック | `server.py` |

*非同期キュー*でリクエストを処理し、1GPU上で最大4並列まで同時推論。

### 2. Blender Add-on

| サブモジュール | UIパネル | コアロジック |
| --- | --- | --- |
| `panel.py` | 生成対象選択・強度スライダー・進捗バー | — |
| `operator_generate.py` | Viseme/Emotion一括生成 | gRPC呼び出し |
| `mesh_validator.py` | 頂点数・ボーン数・UV有無チェック | — |
| `qa_autofix.py` | 破綻頂点検出→クランプ | Laplacian Smooth |

エラーは `report()` で Blender UI に即時表示し、中断ポイントを明示。

### 3. Unity Editor Tool

| サブモジュール | 機能 | 主要クラス |
| --- | --- | --- |
| FBXImporter | BlendShape付きFBXを読み込み | `FbxBlendShapeImporter.cs` |
| AvatarDescriptorSetup | ViewPosition/EyeLook自動設定 | `AvatarDescriptorUtil.cs` |
| VisemeMapper | 15種Viseme自動マッピング | `VisemeAutoMapper.cs` |
| FXAnimatorBuilder | BlendShape→Param駆動FXレイヤー生成 | `FxAnimatorBuilder.cs` |
| PerformanceOptimizer | ポリゴン/テクスチャ警告 | `PerformanceAudit.cs` |

Animatorは **1BlendShape=1State** の軽量構成。生成済みコントローラを Assets/Generated 以下へ保存。

## データフロー

1. ユーザーが Blender で「Generate BlendShapes」クリック  
2. `operator_generate.py` が gRPC で Core Server へ要求送信  
3. Predictor が頂点オフセット配列を返却  
4. Blender は Shape Key を作成→`qa_autofix.py` で品質検査  
5. 完成メッシュを FBX 出力（自動）  
6. Unity で「Auto BlendShape Integration」実行  
7. EditorTool が FBX を取り込み、Descriptor/FxLayer を自動構築  
8. ユーザーは VRChat SDK3 の Build & Test を実行

## エラーハンドリング方針

| 障害 | 検出レイヤー | 対処 |
| --- | --- | --- |
| Core Server未起動 | Blender | 自動でローカル起動試行→失敗時ダイアログ |
| 推論時間>3s | Predictor | ログ警告＋次ロットでバッチ縮小 |
| 破綻頂点>2% | QA | 自動クランプ、閾値超過で「要手動確認」タグ |
| SDK Upload警告 | Unity Tool | PerformanceOptimizer が警告一覧をConsole出力 |

## セキュリティ

- Inference Serverは `127.0.0.1:50051` のみバインド。外部アクセス不可。
- ONNXモデルは AES256 で暗号化、起動時に一時復号。
- Auto-Update は 署名付きZIP＋SHA-256検証、失敗時ロールバック。

## 拡張ポイント

1. ModelManager は **Strategyパターン**で `TensorRTStrategy` と `OpenVinoStrategy` を実装。将来 Apple Silicon 用 `CoreMLStrategy` 追加可。  
2. Predictor の前段に **Plug-in フィルタ**（例: 顔半径正規化）を挿入できるよう `IPreprocessHook` インターフェイスを用意。

## 開発ガイドライン

- Python: PEP8 + mypy strict。Blackで整形、pytest100％緑。
- C#: Unity Coding Standard + Rider Inspect 0警告。
- PRルール: **1モジュール=1PR**, 必ず `tasks.md` の Issue を reference。

## ファイル配置

```
/backend/
  model_manager.py
  predictor.py
  postprocess.py
  server.py
/blender_addon/
  __init__.py
  panel.py
  operator_generate.py
  mesh_validator.py
  qa_autofix.py
/unity_tool/
  Editor/
    FbxBlendShapeImporter.cs
    VisemeAutoMapper.cs
    AvatarDescriptorUtil.cs
    FxAnimatorBuilder.cs
    PerformanceAudit.cs
/docs/
  requirements.md
  design.md   ← 本ファイル
  tasks.md
  test-plan.md
  operations.md
```

## 追跡すべき主要指標（テレメトリ）

| 指標 | 目標値 | 収集箇所 |
| --- | --- | --- |
| 平均推論時間/Viseme | ≤3s | Predictor |
| 平均頂点誤差 | ≤5mm | QA |
| 破綻頂点率 | ≤2% | QA |
| SDK Upload警告数 | 0 | Unity Tool |
| 月間クラッシュ回数 | <2 | SystemMonitor |
