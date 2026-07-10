# Design Document 001: MLX Metal Port

## Summary

Port the autoresearch pretraining harness from single-GPU CUDA/PyTorch to Apple Silicon via MLX. Preserve the two-file contract (`prepare.py` = frozen fixed metric; `train.py` = agent-editable) and the `val_bpb` evaluation semantics, replacing every CUDA/Flash-Attention-3/`torch.compile`/custom-optimizer surface with first-party MLX primitives. Result: an M-series-only fork that boots on a Mac with unified memory, runs the same 5-minute experiment loop, and hands the agent a lean, minimal `train.py` to iterate on.

## Motivation

**Use cases:**

- A researcher with an Apple Silicon Mac (M-series, unified memory; target reference M5 Pro 48 GB) can run overnight autoresearch loops without renting an H100.
- The autoresearch agent gets a Mac-native harness with a smaller, more honest surface — fewer black-box kernels, simpler optimizer, so the agent's edits are on comprehensible code.
- Extends the "notable forks" ecosystem in the upstream README with a clean, first-party-MLX reference implementation.

**Expected outcomes:**

- `uv sync && uv run prepare.py && uv run train.py` completes end-to-end on macOS with an M-series GPU, no CUDA/torch on the system.
- A 5-minute training run finishes, prints the standard summary block (`val_bpb`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`, `depth`), and does not OOM at the shipped defaults.
- `train.py` is the only file the agent edits; `prepare.py` remains the pinned metric.
- Zero PyTorch, zero CUDA, zero HF `kernels` dependencies in `pyproject.toml`.

## Guide-level explanation

### Setup

- macOS on Apple Silicon (M1 or newer), Python 3.10+, `uv`.
- No CUDA, no `torch`, no HF `kernels` runtime.
- Cache at `~/.cache/autoresearch/` used identically for data shards and tokenizer artifacts.

### Usage examples

Unchanged from the upstream README, minus the "NVIDIA GPU" requirement:

```bash
uv sync
uv run prepare.py     # one-time download + BPE tokenizer training
uv run train.py       # single 5-minute experiment on the Mac GPU
```

Agent workflow (`program.md`) is unchanged in shape — same experiment loop, same `results.tsv` schema, same "edit `train.py` only" rule.

### Error handling

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: [metal::Device] Not enough memory` (or process killed) | Defaults still too large for local unified-memory budget | Lower `DEVICE_BATCH_SIZE` first, then `DEPTH`, then `TOTAL_BATCH_SIZE` (keep power-of-two) in `train.py`. |
| Training runs but `mfu_percent` looks nonsensical | Peak-flops constant is a rough M-series estimate, not exact for the local chip | Adjust the `M_SERIES_BF16_PEAK_FLOPS` constant at the top of `train.py`. |
| `prepare.py` fails to download shards | Network / HuggingFace rate limits | Same as upstream — retry, honors `--download-workers`. |
| Tokenizer round-trip assertion fails after upstream merge | `token_bytes` file format drift (this fork uses `.npy`, upstream uses `.pt`) | Re-run `uv run prepare.py` to regenerate the tokenizer artifacts. |
| `import mlx.core` fails | Non-Apple-Silicon host or an old Python | Confirm `uname -m` reports `arm64` and Python is 3.10+. |

## Reference-level explanation

### Architecture

```mermaid
flowchart LR
    subgraph prep [prepare.py — frozen contract]
        DL[shard downloader] --> TOK[BPE tokenizer trainer]
        TOK --> TBY[token_bytes.npy]
        PARQ[(parquet shards)] --> DOC[document iterator]
        DOC --> PACK[BOS-aligned best-fit packer]
        PACK --> MXARR[mx.array batches]
        EVAL[evaluate_bpb]
    end
    subgraph train [train.py — agent playground]
        CFG[hyperparameters] --> MODEL[GPT (mlx.nn)]
        MODEL --> ATTN[mx.fast.scaled_dot_product_attention causal]
        MODEL --> ROPE[mlx.nn.RoPE / mx.fast.rope]
        MODEL --> RMS[mlx.nn.RMSNorm]
        OPT[MultiOptimizer: Muon + AdamW]
        LOOP[5-min training loop]
        MXARR --> LOOP
        MODEL --> LOOP
        OPT --> LOOP
        LOOP -->|final| EVAL
        LOOP --> SUM[summary block]
    end
```

### Implementation details

**Two-file contract — preserved.** `prepare.py` stays the fixed frozen surface (constants, data, tokenizer, dataloader, `evaluate_bpb`); `train.py` is the only file the agent edits. All CUDA-isms in `prepare.py` are replaced by MLX/numpy equivalents so the frozen contract remains device-appropriate for the fork.

**GPT model in `mlx.nn`.** Same architecture as upstream: token embedding (bf16), N transformer blocks with pre-RMSNorm, GQA-ready attention (`n_kv_head == n_head` by default) with RoPE on Q/K and RMSNorm on Q/K post-rope, ResFormer value residual with input-dependent per-head sigmoid gate on the first 32 channels, ReLU² MLP with 4× expansion, per-layer scalar residual mixers (`resid_lambdas`, `x0_lambdas`), tied-free `lm_head`, `softcap=15` tanh on logits, fp32 cross-entropy. **Sliding-window `SSSL` dropped** — attention is uniformly causal (`mask="causal"`).

**Attention.** `mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, mask="causal")`. Query/key layout `[B, N, T, D]` per MLX convention. GQA handled natively by MLX (no manual repeat-interleave).

**Optimizer.** `mlx.optimizers.MultiOptimizer` with two children:
- `Muon(learning_rate=matrix_lr, momentum=0.95, weight_decay=matrix_wd, nesterov=True, ns_steps=5)` — routed to transformer-block matrix parameters (paths under `transformer.h.*.attn.*` and `transformer.h.*.mlp.*` whose weight is 2-D, matching stock Muon's guidance).
- `AdamW(learning_rate=…, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)` — routed to embeddings (`transformer.wte`, `value_embeds.*`), `lm_head`, and 1-D scalars (`resid_lambdas`, `x0_lambdas`). AdamW-only paths carry an additional `dmodel_lr_scale = (model_dim / 768) ** -0.5` factor on their base LR, matching the upstream scaling intent.

Filters use MLX `MultiOptimizer` predicates over the flattened parameter tree path + array shape. No custom optimizer subclass. No polar-express NS, no NorMuon, no cautious weight decay, no per-shape stacked step — those are optimizer *experiments* the autoresearch agent can rediscover.

**Precision.** Parameters and activations in `mx.bfloat16`; loss/logits promoted to `mx.float32` before cross-entropy and softcap; RoPE cos/sin tables in `mx.bfloat16`. No autocast context — MLX handles dtype propagation through explicit casts at the boundaries.

**Compilation.** A single `mx.compile`d step function that takes `(x, y, model_state, optimizer_state)`-shaped inputs, returns `(loss, new_model_state, new_optimizer_state)`. Compilation captures the model + optimizer state trees. `shapeless=False` (constant `[B, T]`).

**Gradient accumulation.** Loop `grad_accum_steps` microsteps: `loss_i, grads_i = value_and_grad_fn(model, x_i, y_i)`, tree-add into an accumulator, divide by `grad_accum_steps`, then `optimizer.update(model, grads)` + `mx.eval(loss, model.parameters(), optimizer.state)`.

**Dataloader.** `prepare.py` rewritten: same BOS-aligned best-fit packing algorithm producing 100% token utilization, but the pinned-CPU / non-blocking-GPU double buffer is gone (unified memory). Each yield converts the numpy row buffer to `mx.array` (`inputs`, `targets`) with dtype `mx.int32`. Same `epoch` counter semantics.

**Tokenizer artifact.** `train_tokenizer` in `prepare.py` writes `token_bytes.npy` (numpy `int32` array) instead of `token_bytes.pt`. `get_token_bytes(...)` reads via `np.load` and returns an `mx.array`. This removes the last torch dependency.

**Timing.** `mx.eval(loss)` after each microstep to synchronize; wall clock via `time.time()` around fwd+bwd+update; step timing excludes the first 10 steps (JIT + first-batch warm) just like upstream.

**MFU / peak memory.** `mx.get_peak_memory()` for bytes; a top-of-file `M_SERIES_BF16_PEAK_FLOPS` constant (default ~1.5e13, easily overridden; documented as informational). MFU is reported for continuity with the upstream summary block, not used to gate anything.

**Seeding.** `mx.random.seed(42)` + Python `random.seed(42)` + `np.random.seed(42)`.

**Baseline defaults for M5 Pro 48 GB** (locked in `train.py`, all agent-editable):
- `DEPTH = 6`
- `DEVICE_BATCH_SIZE = 32`
- `TOTAL_BATCH_SIZE = 2**17`
- `WINDOW_PATTERN` removed
- `MAX_SEQ_LEN` unchanged in `prepare.py` (2048) — preserves the metric shape
- All other hyperparameters (LRs, betas, WD, warmup/warmdown ratios) start at the upstream values

These are conservative bootstrap defaults chosen so the first run boots without OOM on a 48 GB machine. Phase 3 of the plan validates and tightens them.

### File layout

```
prepare.py            # rewritten: numpy/mlx-backed dataloader, token_bytes.npy, same public API
train.py              # rewritten: mlx.nn GPT, MultiOptimizer, mx.compile step, same summary block
pyproject.toml        # deps swap: mlx replaces torch + kernels; PyTorch index removed
README.md             # updated: MacOS/M-series requirements, remove H100 wording
program.md            # updated: "single Apple Silicon GPU", drop CUDA-specific hints
.planning/architecture.md  # already written in Phase A
```

No new packages, no new directories. Everything else (`analysis.ipynb`, `.gitignore`, `.python-version`, `progress.png`) is unchanged.

### Testing strategy

There is no test framework in this repo and none will be added — matches upstream's deliberate minimalism. Verification is empirical:

1. **Compile smoke test** — `uv run python -c "import train"` succeeds, model constructs, one microstep runs without exception.
2. **Prepare smoke test** — `uv run prepare.py --num-shards 2` completes, `~/.cache/autoresearch/tokenizer/token_bytes.npy` exists, sanity round-trip passes.
3. **Full 5-minute run** — `uv run train.py > run.log 2>&1` completes; `grep "^val_bpb:" run.log` yields a finite positive number; `grep "^peak_vram_mb:" run.log` yields a value comfortably under 48000.
4. **Baseline stability** — repeat run (2 or 3 attempts) shows loss curve descending monotonically past step ~50, not NaN, not exploding past the 100 fast-fail cap.
5. **Contract check** — `prepare.py` public surface (`MAX_SEQ_LEN`, `TIME_BUDGET`, `Tokenizer`, `make_dataloader`, `evaluate_bpb`) unchanged in name and semantics.
