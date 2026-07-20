# 変更履歴

[English](CHANGELOG.en.md) | 日本語

本プロジェクトの主要な変更を記録する。
バージョン付けは [Semantic Versioning](https://semver.org/spec/v2.0.0.html) に従う。

---

## [0.1.0] - 2026-07-20

初回公開リリース。本推論基盤が元々開発されていた [yu_ai_manager](https://github.com/eauesque/yu_ai_manager)
から切り出し、当該アプリケーションに依存せず他プロジェクトからも Hailo-10H NPU を
利用できるよう、独立したサービスとして公開する。

### Added

- **Hailo-10H NPU 向け HTTP 推論 sidecar** (`yu-hailo-infer`)。対応機能:
  - CLIP 画像 embedding — `POST /v1/infer/clip-image`
  - CLIP テキスト embedding — `POST /v1/infer/clip-text`
  - WD-Tagger タグ推論 — `POST /v1/infer/wd`
  - LLM テキスト生成・tokenize — `POST /v1/infer/llm/generate`(+`/stream`)、`POST /v1/infer/llm/tokenize`
  - VLM テキスト生成 — `POST /v1/infer/vlm/generate`(+`/stream`)。テキストのみ対応で、画像添付チャットは対象外
  - YOLO 物体検出 — `POST /v1/infer/yolo/detect`。NMS・デコード適用済みの最終検出結果を返す
  - 音声文字起こし・tokenize — `POST /v1/infer/speech2text/transcribe`(base64 WAV 入力、transcribe/translate 対応)、`POST /v1/infer/speech2text/tokenize`
- **共有推論エンジン** (`yu-hailo-infer-core`)。ONNX Runtime を用いた WD-Tagger・
  CLIP テキストエンコーダに加え、YOLO 後処理(量子化解除、グリッド/ストライド相対
  デコード、HEF 内蔵 NMS 出力のパース、NMS)を提供する。
- **Bearer token 認証** (`yu-hailo-auth`)。比較は定数時間で行う。token は CLI 引数
  ではなく stdin の起動契約経由で受け渡すため、`/proc/<pid>/cmdline` から漏洩しない。
- **scan root 包含検査**: 呼び出し側が渡すファイルパスは、起動契約で宣言された
  root 配下に解決される場合のみ受け付ける。
- **HailoRT SDK なしでもビルド可能**: SDK ヘッダが存在しない環境ではスタブ shim へ
  フォールバックするため、Hailo ハードウェアを持たない環境でもコンパイルできる。
- **AI エージェント向け文書**: `docs-index.yaml`(コードへのポインタ索引)と
  `docs/ai-reference.yaml`(設定・エンドポイント仕様・既知の癖)。人間向けの長文
  ドキュメントの代わりに、コーディングエージェントに読ませることを想定している。

### Notes

- **非公式**。本プロジェクトは Hailo 社と提携・承認・支援関係にない。HailoRT SDK に
  リンクするのみで、SDK 自体は一切同梱していない。
- **対応デバイス**: Hailo-10H で実機検証済み。Hailo-8 / Hailo-8L 等その他のモデルは
  **未検証**である(HailoRT SDK 層では動作する可能性があるが、動作保証はしない)。
- **推論のみ**を担う。ベクトル検索・インデックス構築・DB 永続化は意図的に対象外とし、
  利用側アプリケーションの責務とする。
- VLM の画像添付チャットと Web 検索 RAG 統合は対象外。いずれも汎用推論ではなく
  アプリケーション層の関心事であるため。

[0.1.0]: https://github.com/eauesque/yu-hailo-infer/releases/tag/v0.1.0
