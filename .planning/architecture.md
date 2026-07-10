# Architecture

## Language & Framework
- Python **3.10+** (`.python-version` pins 3.10, `pyproject.toml` requires `>=3.10`).
- Managed by **uv** (single-tool policy: `uv sync`, `uv run …`; no `pip`/`venv`).
- Single-GPU **MLX** LLM pretraining harness. macOS / Apple Silicon (M-series, unified memory) only. No PyTorch, no CUDA, no Hugging Face `kernels` at any point.

## Key Dependencies (pyproject.toml)
- `mlx>=0.32` — the entire ML runtime (`mlx.core`, `mlx.nn`, `mlx.optimizers`, `mx.fast.*`).
- `tiktoken>=0.11.0`, `rustbpe>=0.1.0` — tokenizer runtime + trainer.
- `pyarrow>=21.0.0`, `pandas>=2.3.3`, `numpy>=2.2.6` — parquet dataloader.
- `requests>=2.32.0` — dataset shard downloader.
- `matplotlib>=3.10.8` — for `analysis.ipynb`.

## Build & Test Tooling
- No formatter/linter config, no test suite, no CI. Bare-bones by design.
- Verification is empirical: `uv run prepare.py` (one-time data + tokenizer), `uv run train.py > run.log 2>&1` (one experiment), then `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
- Data + tokenizer live in `~/.cache/autoresearch/` (not in repo).

## Project Structure
```
prepare.py       — fixed constants, data download, BPE tokenizer training, dataloader, evaluate_bpb (read-only for the agent)
train.py         — GPT model + MultiOptimizer(Muon, AdamW) + training loop (only file the agent edits)
program.md       — agent skill/instructions
analysis.ipynb   — post-hoc results inspection
pyproject.toml   — deps (mlx only, no torch, no CUDA index)
.gitignore       — hides results/, worktrees/, queue/, dev/, results.tsv, CLAUDE.md, AGENTS.md
README.md        — user-facing overview (Apple Silicon requirements)
progress.png     — teaser chart
```

## Conventions & Patterns
- **Two-file contract**: `prepare.py` (frozen fixed metric + data path) vs. `train.py` (agent playground).
- **Fixed 5-min wall-clock time budget** (`TIME_BUDGET = 300`); comparable across compute platforms.
- **Metric**: `val_bpb` (bits per byte) — vocab-size-independent, sums nats over target-byte lengths on a pinned validation shard (shard 6542).
- **Hyperparameters as top-level constants** in `train.py` (no CLI flags). Agents flip values inline.
- Model uses **RoPE (`nn.RoPE`), n_kv_head == n_head today (GQA-ready), ReLU² MLP, RMSNorm pre-norm, per-layer resid+x0 scalars, ResFormer value-embed residual, pure causal attention** (no sliding window; the upstream `SSSL` pattern is dropped in this fork for a first-party-only `mx.fast.scaled_dot_product_attention(mask="causal")` path).
- Optimizer is `mlx.optimizers.MultiOptimizer([Muon, AdamW], filters=[...])`: **stock Muon** for 2-D transformer-block matrices, **stock AdamW** for embeddings / lm_head / value embeddings / per-layer scalars. No custom optimizer, no polar-express NS, no NorMuon, no cautious weight decay, no per-shape stacked step — those are deliberately left as optimizer *experiments* the agent can rediscover.
- Precision: **bfloat16 embeddings**, logits promoted to fp32 for softcap + cross-entropy. Softcap = 15 tanh on logits.
- Data pipeline: **BOS-aligned best-fit packing** to 100% token utilization, single numpy scratch row buffer, direct materialization as `mx.array` (**unified memory — no pinned-host / non-blocking copies**).
- Tokenizer artifact: `token_bytes.npy` (numpy int32) rather than a torch `.pt`.
- MFU reported against a top-of-file `M_SERIES_BF16_PEAK_FLOPS` constant (default ~1.5e13, informational).

## Platform Reality
- README calls the fork Apple-Silicon-only; upstream (NVIDIA / H100 / CUDA) lives at `karpathy/autoresearch`.
- No CUDA call sites, no `torch.*`, no `kernels.get_kernel`, no `pin_memory`, no `expandable_segments` env.
- Peak memory via `mx.get_peak_memory()`. Seeding via `mx.random.seed(42)` + `random.seed(42)` + `np.random.seed(42)`.

## MLX Surface Used (verified against 0.32.0)
- `mlx.core` + `mlx.nn`: `Embedding`, `Linear`, `RoPE`, `mx.fast.rms_norm`, `mx.fast.scaled_dot_product_attention(q, k, v, scale=…, mask="causal")` in `[B, N, T, D]` layout, `mx.compile(inputs=state, outputs=state)`, `mx.eval`, `nn.value_and_grad`, `nn.losses.cross_entropy`, `mx.bfloat16`, `Module.astype`.
- `mlx.optimizers`: `Muon` (`momentum`, `weight_decay`, `nesterov`, `ns_steps`), `AdamW` (`betas`, `eps`, `weight_decay`), `MultiOptimizer(optimizers, filters=[...])`.
- Memory: `mx.get_peak_memory()`, `mx.reset_peak_memory()`.
