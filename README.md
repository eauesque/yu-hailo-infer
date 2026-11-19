# yu-hailo-infer
[English](README.en.md) | 日本語

Hailo-10H向けRustネイティブ推論マイクロサービス。HTTP経由でCLIP embedding・WD-Taggerタグ推論・LLM生成・VLMテキスト生成・YOLO物体検出を公開する。

変更履歴: [CHANGELOG.md](CHANGELOG.md)。本プロジェクトは非公式であり、Hailo社との提携関係はない。

本crateは`-sys`バインディングではなくサービス本体である。対象はHailo-10Hであり、当該デバイスではC APIの`VStreams`経路が`HAILO_NOT_IMPLEMENTED`を返すため、HailoRTへはC++の`InferModel` APIを独自の小さなシム経由で呼び出している。生のC APIバインディング(Raspberry PiのHailo-8 / AI HAT+向け等)を求める場合は、別プロジェクトである[`hailort-sys`](https://crates.io/crates/hailort-sys)を参照されたい。同crateは本crateより一段下の層に位置する。

## 対応機能

| 機能 | エンドポイント | 状態 |
|---|---|---|
| CLIP画像embedding | `POST /v1/infer/clip-image` | 対応 |
| EmbeddingGemma 2文章embedding | `POST /v1/infer/text-embed` | 実験的 host FP32、256次元 |
| CLIPテキストembedding | `POST /v1/infer/clip-text` | 対応 |
| WD-Taggerタグ推論 | `POST /v1/infer/wd` | ONNX対応、HEFは実機受入検証待ち |
| LLMテキスト生成 | `POST /v1/infer/llm/generate`(+`/stream`) | 対応 |
| LLM tokenize | `POST /v1/infer/llm/tokenize` | 対応 |
| VLMテキスト生成 | `POST /v1/infer/vlm/generate`(+`/stream`) | 対応(テキストのみ、画像添付チャットは非対応) |
| YOLO物体検出 | `POST /v1/infer/yolo/detect` | 対応(NMS/デコード込み、v0.2〜) |
| 音声文字起こし(speech2text) | `POST /v1/infer/speech2text/transcribe`(base64 WAV入力、transcribe/translate対応)、`POST /v1/infer/speech2text/tokenize` | 対応 |

## 対応デバイス

- **Hailo-10H**: 既存機能は実機検証済み。WD HEF推論は実機受入検証まで未検証
- Hailo-8 / Hailo-8L 等その他モデル: **未検証**(HailoRT SDKレベルでは動く可能性があるが動作保証なし)

## 依存

- Rust: **1.88** 以上。`[workspace.package]` に `rust-version = "1.88"` を宣言しており、各クレートが継承する。1.88.0 でのビルド成功と 1.85.0 での失敗を実測して決めた値である(2026-09-03)。下限を決めているのは本リポジトリのコードではなく依存で、cargo は `ort`/`ort-sys` 2.0.0-rc.12 が `rustc 1.88` を要求すると報告する。依存を上下させればこの値も動くため、下げる際は必ず旧 toolchain で `cargo build --workspace` を実測すること
- HailoRT SDK(バージョン要記載) — `hailort`共有ライブラリ・ヘッダが必要
- `ort`(ONNX Runtime バインディング) — CLIPテキストエンコーダ・WD-Tagger推論に使用
- `tokenizers`(Apache-2.0) — CLIP/LLMのBPEトークナイズに使用
- 本リポジトリは**推論のみ**を提供する。ベクトル検索・インデックス構築(usearch等)・DB永続化はスコープ外(利用側アプリケーションの責務)

## クイックスタート

GPU は既存 Cargo feature `cuda` / `rocm` / `directml` / `coreml` / `openvino` で有効にします。標準 build は CPU です。例: `cargo build --release -p yu-hailo-infer --features cuda`。ROCm / DirectML / OpenVINO は対応 ONNX Runtime の `ORT_DYLIB_PATH` が必要です。EmbeddingGemma 2 は FP32 を保持する CUDA（TF32 無効）/ ROCm / DirectML を使用し、CoreML / OpenVINO の EG2 実行は精度確認まで CPU fallback です。CLIP 等は既存 provider 対応を保持します。

`/healthz` の `text_embed.onnx_execution` は要求と登録 provider を区別します。未読込・推論 lock 中は登録 unknown が返る場合があります。GPU 登録は全 node の GPU 割当を保証しません。GPU 実機の数値 parity と速度は未検証です。

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

## WD-Tagger HEFバックエンド（実機未検証）

`POST /v1/infer/wd` は既存の `path`・`model_id`・閾値・任意の `profile` / `model_subdir` に加え、`backend`（省略時は `"onnx"`、または `"hef"`）、`hef_path`、`hef_sha256`、`timeout_ms` を受け取ります。HEFでは `hef_path` と `hef_sha256`（小文字16進数64桁）が必須で、sidecarに既定のHEFパスはありません。hashの欠落・書式違い・内容不一致は503 `hef_invalid` です。ONNXでは `hef_sha256` を完全に無視し、ログにも出しません。`timeout_ms` はHEFのみで使い、既定値・上限とも120,000 msです。sidecar自身はONNXへ自動切替しません。成功応答は `{"ok":true,"backend_applied":"hef","profile_applied":false,"data":{...}}` の形です。ONNX成功時は `backend_applied:"onnx"`、profile指定時は `profile_applied:true` を返します。

`GET /healthz` の追加フィールド `wd_hef` は、Linuxの実HailoRTビルドでHEF経路を含む場合のみ `true` です。stubビルドでは常に `false` です。既存の `hailo_stub` は維持します。旧応答で `wd_hef` が無ければ呼出側は非対応として扱ってください。health要求ではHEFを開かず、WDキャッシュを使わず、VDeviceを作りません。この値はビルドの能力を示し、モデル配置やデバイスの正常性を保証しません。

対象はViT用のUINT8 HEFです。入力は1本・448×448×3・BGR/NHWC（602,112 byte）、出力は1本・UINT8・語彙件数と同じ要素数とし、`(q - zero_point) * scale` で逆量子化します。前処理はONNXと共通の白pad・Lanczos3を使い、異なるprofileは拒否します。HEFでも `--wd-cache-dir` 内の `model.onnx` と `selected_tags.csv` が必要です。HEFは同一fdを `O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC` で開いてfstatし、通常ファイルかつ256 MiB以下を確認して上限付きで読み込み、検査済みbufferをMemoryViewへ渡します。末端symlinkや特殊ファイルは拒否します。

VDeviceはプロセス全体で1つのstaticを共有し、Hailo処理は共通のdevice threadで直列実行します。WD HEFキャッシュは1件だけ常駐し、要求ごとにopen/fstatします。dev・inode・サイズ・mtime（nsを含む）または要求digestが変われば、旧handleを解放してから読み直します。同じ検査済みfdの上限付き読込内容からSHA256を計算し、要求の32 byte全体と照合してからHailoRTのopen・metadata・configureを呼びます。cache再利用にはfile keyと照合済みdigestの両方の一致が必要です。

`HAILO_COMMUNICATION_CLOSED` を受けるとプロセス全体をHailo session喪失状態にします。以後のHailo要求はHailoRTを呼ばず503（WD HEFは `{"error":"device_error"}`）を返し、待機済み処理も拒否します。CPU ONNX経路は終了待ちの間も使えます。`/healthz` は常に `hailo_session_lost` を含み、通常は `false`、喪失後は `true` と `ok:false` を返します。短い待機の後にHTTP応答を完了させてから **70（`EX_SOFTWARE`）** で終了し、yu-serverの再起動に委ねます。回復には新プロセスが必要です。組込利用でもこの終了動作を保つにはlibraryの `run()` を使ってください。

HEF経路のエラー本文は `{"error":"<code>"}` のみで、path・detail・messageは含めません。

| HTTP | error | 条件 |
|---|---|---|
| 503 | `hef_invalid` | `backend:"hef"` で `hef_path` が無い、または `hef_sha256` が欠落・書式違い・内容不一致 |
| 403 | `path outside allowed roots` | 画像がscan roots外（ONNXと同じ応答） |
| 422 | `image_invalid` | 画像のcanonicalize・decode・前処理に失敗 |
| 503 | `hef_not_found` | HEFが無い・読めない |
| 503 | `hef_invalid` | HEFの種類・サイズ・metadata・profileが不正 |
| 503 | `device_busy` | device待機列または前処理予算が満杯 |
| 503 | `device_error` | HailoRT障害・session喪失・degraded・device thread停止 |
| 503 | `hef_timeout` | sidecarの応答期限切れ |
| 503 | `model_not_downloaded` | 語彙またはmodel.onnxが無い |

既存の認証エラーは401、JSONの抽出・型検証エラーはaxumの応答です。この変更はdevice acceptance testing（実機受入検証）待ちです。この改訂では精度・fps・同時常駐時の挙動・メモリ使用量を実証していません。

## LLMへの説明のさせ方

このリポジトリはAI/LLMコーディングエージェント向けに`docs-index.yaml`(コードファイルへのポインタ索引)を用意している。人間向けの長文ドキュメントの代わりに、エージェントへ以下のように指示することを想定している:

設定・エンドポイント仕様・既知の癖(認証エラーのレスポンス形式差異、デバイス排他制御等)を横断的にまとめたAI向けリファレンスとして`docs/ai-reference.yaml`(英語)も用意している。「このサービスの使い方・設定方法を説明して」といった依頼にはこちらを先に読ませるとよい。

> 「`docs-index.yaml`を読み込み、該当エントリの`path`が指すコードを実際に読んだ上で、〈質問内容〉について説明してください。」

例:
> 「`docs-index.yaml`を読んで、YOLO検出のNMS/デコード実装(`yu-hailo-infer-core/src/yolo_postprocess.rs`)がどう動くか説明して」
> 「`docs-index.yaml`を見て、CLIP画像embeddingのdequantize処理はどこにあるか教えて」

## ライセンス

MIT

## EmbeddingGemma 2

`POST /v1/infer/text-embed` accepts `{"text":"桜の花","task":"query","dim":256}` with the existing bearer authentication. For documents use `task:"document"` and an optional `title`; the server adds the search prefix. The success envelope contains `data.vector`, `dim`, `model`, `revision`, `precision`, and `preprocessing`.

Only pinned FP32 revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0` is supported. Place `tokenizer.json`, `onnx/model.onnx`, and `onnx/model.onnx_data` under `~/.cache/yu_ai_manager/embeddinggemma2/<revision>` or an absolute `model_dir`. Artifact sizes and SHA-256 must match the manifest in `embeddinggemma2.rs`; no automatic download occurs here. Loading uses a private verified snapshot (~1.1 GB additional disk space), removed on graceful cache destruction; forced termination may leave residue. Input text/title is limited to16 KiB combined, then8192 tokens. One active request and four FIFO waiters are allowed; overflow returns503. Pi5, GPU behavior, and production throughput are unverified.
