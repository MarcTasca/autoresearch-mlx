---
id: PLAN-001
status: approved
---

# Execution Plan 001: MLX Metal Port

## Objective

Rewrite `train.py` and `prepare.py` on top of MLX so the autoresearch harness runs end-to-end on Apple Silicon (M-series, unified memory) with a first-party-only dependency set, preserving the two-file contract and the `val_bpb` metric, and completes a 5-minute experiment on an M5 Pro 48 GB without OOM.

## Checklist

### Phase 1 — Dependency + repo cutover

- [x] Rewrite `pyproject.toml`: drop `torch`, `kernels`, the `[tool.uv.sources]` torch pin, and the `[[tool.uv.index]] pytorch-cu128` block; add `mlx` (>=0.32) as the sole ML runtime; keep `numpy`, `pyarrow`, `pandas`, `tiktoken`, `rustbpe`, `requests`, `matplotlib`.
- [x] Regenerate `uv.lock` (`uv sync`) and confirm the lockfile no longer references PyTorch or CUDA wheels.
- [x] Confirm `.python-version` (3.10) still satisfies MLX; bump only if MLX requires it.

### Phase 2 — Rewrite `prepare.py` (frozen contract, MLX-native internals)

- [x] Remove all torch imports and CUDA-specific code paths (`pin_memory=True`, `device="cuda"`, `.copy_(non_blocking=True)`, `torch.load`, `torch.save`, `torch.no_grad`, `torch.tensor`).
- [x] Keep the public surface identical in name and semantics: `MAX_SEQ_LEN`, `TIME_BUDGET`, `EVAL_TOKENS`, `Tokenizer` (with `from_directory`, `get_vocab_size`, `get_bos_token_id`, `encode`, `decode`), `make_dataloader(tokenizer, B, T, split)`, `evaluate_bpb(model, tokenizer, batch_size)`, plus the `__main__` CLI (`--num-shards`, `--download-workers`).
- [x] Rewrite `train_tokenizer` to persist token byte lengths as `token_bytes.npy` (numpy `int32`) instead of `token_bytes.pt`.
- [x] Rewrite `get_token_bytes` to `np.load` the `.npy` and return an `mx.array` (default device); drop the `device=` argument (unified memory).
- [x] Rewrite `make_dataloader` to keep the BOS-aligned best-fit packing algorithm and 100% token utilization, but yield `(inputs, targets, epoch)` where `inputs`/`targets` are `mx.array` `int32` of shape `(B, T)`. Use a single numpy scratch buffer; no CPU↔GPU staging.
- [x] Rewrite `evaluate_bpb` to use `mx.eval` and `.item()` on MLX arrays; keep the exact math (nats / (log 2 × bytes), special-token bytes-0 mask), and iterate `EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)` steps against the pinned validation shard.

### Phase 3 — Rewrite `train.py` (agent-editable playground)

- [x] Strip CUDA/kernels imports (`kernels`, `PYTORCH_ALLOC_CONF`, `torch.cuda.*`, `torch.amp.autocast`, `fa3.*`, `torch.compile`, `H100_BF16_PEAK_FLOPS`).
- [x] Reimplement `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` on top of `mlx.nn` with the same architectural behavior described in the DD (RoPE, GQA, RMSNorm on Q/K, ResFormer value residual with per-head sigmoid gate on the first 32 input channels, ReLU² MLP, per-layer resid+x0 scalars, softcap tanh on fp32 logits, bf16 embeddings). Use `mx.fast.scaled_dot_product_attention(..., mask="causal")` — no sliding window.
- [x] Remove `WINDOW_PATTERN`, `_compute_window_sizes`, and any window-related plumbing.
- [x] Reimplement weight init in an `init_weights` method that matches upstream distributions (uniform `±√3/√n_embd` for attention/MLP matrices, `zeros_` for the two residual projections, `normal_(std=1.0)` for token embedding, `normal_(std=0.001)` for `lm_head`, `zeros_` for value-embed gates, `fill_(1.0)/(0.1)` for `resid_lambdas`/`x0_lambdas`), then cast embeddings to bf16.
- [x] Build the optimizer as `MultiOptimizer([Muon(...), AdamW(...)], filters=[matrix_predicate])`: matrix predicate = 2-D parameters under the transformer-block path; everything else falls through to AdamW. Apply the `(model_dim / 768) ** -0.5` LR scaling to the AdamW branch's base LR only.
- [x] Wrap the fwd + `value_and_grad` + optimizer-update sequence in a single `mx.compile`d `step_fn` (capture model + optimizer state trees) and call it inside the microstep loop.
- [x] Implement gradient accumulation as tree-sum of per-microstep grads, divide by `grad_accum_steps`, then a single `optimizer.update` + `mx.eval(loss, model.parameters(), optimizer.state)`.
- [x] Port the schedules verbatim (`get_lr_multiplier`, `get_muon_momentum`, `get_weight_decay`), driving MLX optimizer hyperparameters by writing to `optimizer.learning_rate` / per-child attributes each step.
- [x] Port the training loop: prefetch first batch, per-step timing via `time.time()` bracketed by `mx.eval`, skip the first 10 steps for warm-up, fast-fail on NaN or loss > 100, keep the `\r` progress line format with the same tokens.
- [x] Print the final summary block with the exact keys and formatting used upstream: `val_bpb`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`, `depth`. `peak_vram_mb` sourced from `mx.get_peak_memory() / 1024 / 1024`; `mfu_percent` computed against a top-of-file `M_SERIES_BF16_PEAK_FLOPS` constant.
- [x] Seed with `mx.random.seed(42)` + `random.seed(42)` + `np.random.seed(42)`; drop the CUDA-side seeding.
- [x] Lock the M5-Pro-48 GB baseline defaults specified in the DD (`DEPTH=6`, `DEVICE_BATCH_SIZE=32`, `TOTAL_BATCH_SIZE=2**17`); keep all other hyperparameters at their upstream values.

### Phase 4 — Verification (empirical, no test framework)

- [x] `uv sync` succeeds on macOS/arm64; the lockfile contains `mlx` and no `torch`.
- [x] `uv run prepare.py --num-shards 2` completes; `~/.cache/autoresearch/tokenizer/token_bytes.npy` exists; the internal tokenizer round-trip assertion passes.
- [x] `uv run python -c "import train"` runs without exception (module-level construction of model, optimizer, dataloader all succeed).
- [x] `uv run train.py > run.log 2>&1` completes within the wall-clock budget + startup, does not OOM, and `grep "^val_bpb:\|^peak_vram_mb:\|^num_steps:" run.log` returns finite values with `peak_vram_mb` well under 48000 and `num_steps > 20`.
- [x] Re-run the training script a second time; loss curve descends past step ~50 without NaN and the final `val_bpb` is within ~5% of the first run (sanity, not benchmark).
- [x] Tune the baseline defaults down if any of the above hits OOM, then re-verify; the shipped defaults must be the ones that pass this phase.

### Phase 5 — Documentation

- [x] Update `README.md`: replace the H100/NVIDIA prerequisites and platform-support paragraph with an Apple Silicon (M-series, unified memory) requirement; update Quick Start (no CUDA), keep the run commands identical; note this fork is Mac-only and drop the "Notable forks" pointer to itself.
- [x] Update `program.md`: replace "single GPU" H100-flavored wording with "single Apple Silicon GPU"; drop any residual CUDA hints; leave the experiment loop, results.tsv schema, and NEVER-STOP semantics untouched.
- [x] Update `.planning/architecture.md` to reflect the post-port reality (MLX runtime, no torch, no kernels).

## Key areas affected

| Area / Module | Change |
|---|---|
| `pyproject.toml` + `uv.lock` | Drop `torch`, `kernels`, PyTorch index; add `mlx`. |
| `prepare.py` | Rewrite dataloader + tokenizer artifact I/O + `evaluate_bpb` on numpy/MLX; preserve public API and constants. |
| `train.py` | Rewrite GPT, optimizer wiring, step function, training loop on MLX; drop `SSSL`; drop custom `MuonAdamW`. |
| `README.md`, `program.md` | Refresh platform/runtime prose for M-series. |
| `.planning/architecture.md` | Sync to post-port state. |

## Verification steps

1. `uv sync` — succeeds on macOS/arm64; lockfile references `mlx`, no `torch`, no CUDA wheels.
2. `uv run prepare.py --num-shards 2` — completes; `~/.cache/autoresearch/tokenizer/token_bytes.npy` present; tokenizer sanity round-trip passes.
3. `uv run python -c "import train"` — module import + top-level construction succeeds without exception.
4. `uv run train.py > run.log 2>&1` — completes; `grep "^val_bpb:" run.log` yields a finite number; `grep "^peak_vram_mb:" run.log` reports < 48000; `grep "^num_steps:" run.log` reports > 20.
5. Repeat step 4 — no NaN, loss descends past step 50, `val_bpb` within ~5% of the previous run.
6. `grep -R "torch\|cuda\|fa3\|kernels\|pin_memory\|H100" prepare.py train.py` — returns nothing.

## Review Issues

<!-- Populated by the code reviewer agent during the review loop -->

| # | Severity | File | Description | Status |
|---|----------|------|-------------|--------|

<!-- All six iteration-1 review issues resolved in iteration 2 (see below). -->

## Resolved Review Issues

<!-- Issues moved here once addressed by the coder and confirmed by reviewer -->

| # | Severity | File | Description | Resolution |
|---|----------|------|-------------|------------|
| 1 | should-fix | train.py | Compiled `step_fn` scope narrower than plan/DD — only fwd + `value_and_grad` inside `mx.compile`; accumulation, scale, and `optimizer.update` ran in Python. | Added `apply_fn = mx.compile(inputs=[model.state, optimizer.state], outputs=…)` that fuses the inverse-accum scale with `optimizer.update`; training loop now calls `apply_fn(accum_grads)` per step. Tree-sum accumulation stays in Python (needed for grad accumulation), matching DD §Compilation. |
| 2 | should-fix | train.py | AdamW hyperparameter collapse: all non-Muon params routed through a single AdamW at `EMBEDDING_LR * dmodel_lr_scale`, betas `(0.8, 0.95)`, losing upstream's 4 per-group LRs/betas. | Split into 4 AdamW instances via `MultiOptimizer([muon, adamw_lm_head, adamw_embeds, adamw_resid, adamw_x0], filters=[...])`: `lm_head` at `UNEMBEDDING_LR*dmodel_scale`, `wte+value_embeds` at `EMBEDDING_LR*dmodel_scale`, `resid_lambdas` at `SCALAR_LR*0.01`, `x0_lambdas` at `SCALAR_LR` with betas `(0.96, 0.95)`. `dmodel_lr_scale` applied only to embedding-family LRs, matching upstream `setup_optimizer`. |
| 3 | should-fix | train.py:8 | Dead `HF_HUB_DISABLE_PROGRESS_BARS` env var — no huggingface_hub/datasets consumer. | Removed both the env-var assignment and the now-unused `import os` (grep confirms `os` had no other references). |
| 4 | nit | train.py:21 | Unused `tree_unflatten` import from `mlx.utils`. | Dropped from the import list; `tree_flatten` and `tree_map` remain (still used). |
| 5 | nit | train.py:401 | `t_start_training = time.time()` assigned but never read. | Deleted the line; summary block keeps its nine keys unchanged. |
| 6 | nit | train.py:74,98-99 | RoPE cos/sin end up fp32 because block Linear weights stayed fp32, promoting Q/K back up. | Cast all block matrices (attn c_q/c_k/c_v/c_proj/ve_gate + MLP c_fc/c_proj) to bf16 at end of `init_weights`, matching the existing `wte`/`value_embeds` pattern; added a bf16 cast at `Block.__call__` entry so fp32 residual promotions from the lambda mixers don't re-upcast the block's input. RoPE now sees bf16 Q/K. Verified: MFU jumped from 40.9% (iter 1) to 66.4% at DEPTH=6. |
