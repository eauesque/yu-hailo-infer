# yu-hailo-infer
[English](README.en.md) | 日本語

Hailo-10H向けRustネイティブ推論マイクロサービス。HTTP経由でCLIP embedding・WD-Taggerタグ推論・LLM生成・VLMテキスト生成・YOLO物体検出を公開する。

変更履歴: [CHANGELOG.md](CHANGELOG.md)。本プロジェクトは非公式であり、Hailo社との提携関係はない。

## 対応機能

| 機能 | エンドポイント | 状態 |
|---|---|---|
| CLIP画像embedding | `POST /v1/infer/clip-image` | 対応 |
| CLIPテキストembedding | `POST /v1/infer/clip-text` | 対応 |
| WD-Taggerタグ推論 | `POST /v1/infer/wd` | 対応 |
| LLMテキスト生成 | `POST /v1/infer/llm/generate`(+`/stream`) | 対応 |
| LLM tokenize | `POST /v1/infer/llm/tokenize` | 対応 |
| VLMテキスト生成 | `POST /v1/infer/vlm/generate`(+`/stream`) | 対応(テキストのみ、画像添付チャットは非対応) |
| YOLO物体検出 | `POST /v1/infer/yolo/detect` | 対応(NMS/デコード込み、v0.2〜) |
| 音声文字起こし(speech2text) | `POST /v1/infer/speech2text/transcribe`(base64 WAV入力、transcribe/translate対応)、`POST /v1/infer/speech2text/tokenize` | 対応 |

## 対応デバイス

- **Hailo-10H**: 実機検証済み
- Hailo-8 / Hailo-8L 等その他モデル: **未検証**(HailoRT SDKレベルでは動く可能性があるが動作保証なし)

## 依存

- Rust: 1.96.0 でビルド・テストを確認済み。最低対応バージョン(MSRV)は未確定であり、旧toolchainでの検証を行っていないため `rust-version` は宣言していない
- HailoRT SDK(バージョン要記載) — `hailort`共有ライブラリ・ヘッダが必要
- `ort`(ONNX Runtime バインディング) — CLIPテキストエンコーダ・WD-Tagger推論に使用
- `tokenizers`(Apache-2.0) — CLIP/LLMのBPEトークナイズに使用
- 本リポジトリは**推論のみ**を提供する。ベクトル検索・インデックス構築(usearch等)・DB永続化はスコープ外(利用側アプリケーションの責務)

## クイックスタート

```bash
cargo build --release -p yu-hailo-infer

# auth_token / scan_roots / instance_id はCLI引数ではなく、起動時にstdinへ
# JSONで渡す(「起動契約」)。--port はデフォルト18771、--wd-cache-dirは必須。
echo '{"instance_id":"local-dev","scan_roots":["/data/images"],"auth_token":"<token>"}' \
  | ./target/release/yu-hailo-infer --port 8100 --wd-cache-dir /var/cache/yu-infer-wd
```

```bash
curl -X POST http://127.0.0.1:8100/v1/infer/clip-text \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"text": "a photo of a cat"}'
```

## LLMへの説明のさせ方

このリポジトリはAI/LLMコーディングエージェント向けに`docs-index.yaml`(コードファイルへのポインタ索引)を用意している。人間向けの長文ドキュメントの代わりに、エージェントへ以下のように指示することを想定している:

設定・エンドポイント仕様・既知の癖(認証エラーのレスポンス形式差異、デバイス排他制御等)を横断的にまとめたAI向けリファレンスとして`docs/ai-reference.yaml`(英語)も用意している。「このサービスの使い方・設定方法を説明して」といった依頼にはこちらを先に読ませるとよい。

> 「`docs-index.yaml`を読み込み、該当エントリの`path`が指すコードを実際に読んだ上で、〈質問内容〉について説明してください。」

例:
> 「`docs-index.yaml`を読んで、YOLO検出のNMS/デコード実装(`yu-hailo-infer-core/src/yolo_postprocess.rs`)がどう動くか説明して」
> 「`docs-index.yaml`を見て、CLIP画像embeddingのdequantize処理はどこにあるか教えて」

## ライセンス

MIT
