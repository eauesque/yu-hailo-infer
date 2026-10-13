# TODO

- Measure sentence-only output retrieval on Windows GPU with the same corpus and provider; CPU parity does not establish GPU parity or an end-to-end speedup.

- Validate EmbeddingGemma 2 FP32 memory use, quality, and throughput on Raspberry Pi 5 before claiming support. Current adoption is experimental host-only.
- A crash or forced process termination can leave a private `yu-eg2-*` temporary snapshot (~1.04 GiB). Normal cache/router destruction removes it; no automatic cleanup of other processes' snapshots is performed.
