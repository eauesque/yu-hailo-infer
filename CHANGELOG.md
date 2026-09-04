# 変更履歴

[English](CHANGELOG.en.md) | 日本語

本プロジェクトの主要な変更を記録する。
バージョン付けは [Semantic Versioning](https://semver.org/spec/v2.0.0.html) に従う。

---

## [0.4.0] - 2026-09-03

### Added

- **`GET /healthz` が `hailo_stub` フィールドを返すようになった。** ビルド時に
  HailoRT ヘッダが見つからずスタブシムをリンクした場合は `true`、実機ビルドでは
  `false`。呼び出し側(yu-server 等)がサイドカー生存とHailo実機の稼働を区別できる
  ようにするための恒久修正(yu_ai_manager TODO.md v4.689.47 記載分)。
- **`scripts/hailo_infer_smoke.py` を新設した。** yu-server 経由ではなく
  `yu-hailo-infer` 単体プロセスに対して `/v1/infer/*` を直接叩く実機スモーク
  テスト。DB・JobManager・SSE 等の結合部は検証しない(それは呼び出し側の
  `yu_ai_manager/scripts/hailo_realhw_smoke.py` の役目)。Hailo-10H 実機 +
  `~/hailo_models/` 配下の HEF 一式で全15項目 PASS を確認済み(CLIP画像
  embedding・WD-Tagger・YOLO検出・LLM生成・VLMストリーミング生成・音声文字起こし)。
  CLIPテキスト(ONNXモデル)・WD-Taggerはモデル未配置の場合SKIPとして報告する。

### Changed

- **MSRV を `1.88` と実測し、`[workspace.package]` に `rust-version` を宣言した。**
  各クレートは `rust-version.workspace = true` で継承する。従来は「1.96.0 で
  確認済み、MSRV 未確定」と記すのみで `rust-version` を持たなかったため、
  利用者は自分の toolchain で動くかを試すまで判らなかった。
  - 1.88.0 でビルド成功、1.85.0 で失敗することを実測して決めた。
  - **下限を決めているのは本リポジトリのコードではなく依存**である。cargo は
    `ort`/`ort-sys` 2.0.0-rc.12 が `rustc 1.88` を要求すると報告する。依存を
    上下させればこの値も動くため、下げる際は旧 toolchain で実測すること。
  - 宣言が有効であることは注入で確かめた。一時的に `1.99` へ上げると、cargo は
    依存ではなく**当リポジトリのパッケージ名を挙げて**拒否する
    (`yu-hailo-auth@0.3.1 requires rustc 1.99`)。継承が効かなければこの検査は
    通ってしまい、MSRV は誰も強制しない約束になっていた。

## [0.3.1] - 2026-08-30

### Fixed

- **`/_internal/scan-roots-changed` が古い世代の通知で新しい scan roots を上書きし得た問題
  を修正。** yu-server は書込順に単調増加する `generation` を払っているが、受信側はこれを
  読まず毎回無条件に上書きしていた。yu-server 側の送信ロックは自ら開始した送信しか順序化
  できず、5 秒の timeout で諦めた送信は後から到着して新しい状態を潰し得る。受信側で
  `generation` を記憶し、既に適用した値以下の要求を落とすようにした（応答は
  `{"ok":true,"applied":false,"stale":true}`）。`generation` は roots と同一の
  `RwLock` 下に置き、比較と適用が他要求と交錯しないようにしてある。`generation` を持たない
  要求（本フィールド以前の yu-server）は従来どおり無条件に適用し、記憶した世代も動かさない。

---

## [0.3.0] - 2026-08-16

Hailo-10H 実機で検証済み。`/v1/infer/llm/generate/stream` にネイティブ tool 呼出し対応
（HailoRT genai `LLMGenerator::write(messages, tools)`）を配線した。

### Added

- **LLM ストリーミング生成に `tools` フィールドを追加。** リクエスト JSON に OpenAI 形式の
  tool 定義配列（`{"name":...,"description":...,"parameters":...}`）を渡せるようにした。
  `shim.cpp`/`shim.h`/`shim_stub.cpp`/`llm.rs` の C ABI・Rust バインディングに
  `tools_json`/`tools_count` 引数を追加し、HailoRT SDK の `write(prompt_json_strings,
  tools_json_strings)` へ橋渡しする。既存呼出し元は空配列を渡すだけで従来どおり動作する
  （後方互換）。`MAX_LLM_TOOLS`（64件）・合計バイト数（`MAX_PROMPT_BYTES` と同じ上限）の
  検証を追加。
- 実機（Qwen3-1.7B-Instruct.hef）で tool 定義を渡した生成を検証。モデルは
  `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>` 形式（Qwen 独自の
  function-calling 構文）で応答することを確認した。

## [0.2.0] - 2026-08-08

Hailo-10H 実機で検証済み。主眼はモデル常駐で、これは性能改善ではなく、機能が成立する
唯一の形である —— HailoRT 5.3.0 は解放しても CMA を返さないため、請求毎にモデルを
作って捨てる従来の形は 1 請求あたり約 59 MiB を永久に失い、512 MiB のプールでは実質
「1 boot につき 1 リクエスト」しか成立していなかった。

### Added

- **プロセス寿命の単一 `VDevice`。** 従来は種別ごとに 4 箇所で `VDevice::create_shared()`
  を呼んでいた。Hailo-10H は物理デバイスが 1 つで、同一プロセスに 2 つ作ると
  `HAILO_OUT_OF_PHYSICAL_DEVICES(74)` になる。さらに実測では、**同じ group_id でも
  別インスタンス上に作ったモデルは `InferModel.run()` で失敗する**。解放はしない ——
  `VDevice.release()` は CMA を返さないので、解放は何も達成しない。
- **`vdevice_group_id` を起動契約で受け取る。** 省略時は環境変数
  `HAILO_VDEVICE_GROUP_ID` → `"YU_SHARED"` の順に落ちる。これにより、同じ group_id を
  使う別プロセス（yu_ai_manager の Python 拡張など）とデバイスを共有できる。実機で
  確認済み: Python 側が LLM を保持したまま、sidecar の CLIP が動作する。
- **モデルハンドルの常駐。** 鍵は create 引数の全体。InferModel 級（YOLO/CLIP）は別 HEF
  でも併存でき、GenAI 級（LLM/VLM）は同時 1 つで、別 HEF の要求には現在載っている HEF を
  添えて 409 を返す。`Speech2Text` は `clear_context` を持たないため常駐させない。
- **専有デバイススレッド。** モデルハンドルは `!Send` なので、グローバル mutex を
  ハンドルを所有するスレッドへ置き換えた。閉包と結果だけがスレッドを跨ぐため、
  `unsafe impl Send` は 1 つも使っていない。各作業単位は `catch_unwind` で囲む。

### Fixed

- **常駐が chat を 2 ターン目で壊すところだった。** HailoRT は `system` role を文脈が
  空のときのみ受け付ける。呼出側は毎ターン system を送るため、ハンドルを使い回すと
  `System role messages can only be provided on the first prompt` で失敗する。
  ⟹ **生成のたびに `clear_context()` を呼ぶ。** 実機で差分測定により確認済み ——
  この 1 行を外すとターン 2 が `HAILO_INVALID_OPERATION(6)` で落ち、戻すと通る。
- **メディア前処理の資源を上限で縛った。** 画像・音声のデコードが最悪時に確保する
  作業領域をあらかじめ予約し、超える要求は待たせずに拒否する。
- **HailoRT SDK の無い環境で自前のゲートが通らなかった**のを修正し、CI を追加した。

### Changed

- **2 つの shim 実装に共有宣言ヘッダ（`src/hailort/shim.h`）を導入した。** `build.rs` は
  SDK の有無で `shim.cpp` / `shim_stub.cpp` を切り替えるが、`extern "C"` は arity を
  記号名に含めないため、片方だけ変更しても両環境でコンパイル・リンク・試験が通り、
  実行時に沈黙する未定義動作になっていた。**ただしこれが閉じるのは C++ 同士の面のみで、
  Rust の `ffi.rs` は手写しのままである。**

### Known limitations

- **推論中に約 14 MB/分の CMA leak がある**（load/unload とは独立の別経路）。常駐でも
  消えず、30 分以上の連続セッションは Pi の再起動を挟まないと安定しない。
- CMA は Pi 本体の再起動でしか回収されない。プロセス終了でも戻らないことを実測で確認済み。

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
