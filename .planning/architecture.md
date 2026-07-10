# Architecture

## Language & Framework
- Python **3.10+** (`.python-version` pins 3.10, `pyproject.toml` requires `>=3.10`).
- Managed by **uv** (single-tool policy: `uv sync`, `uv run …`; no `pip`/`venv`).
- Single-GPU **PyTorch 2.9.1** LLM pretraining harness. CUDA-only today; wheel index pinned to `pytorch-cu128`.

## Key Dependencies (pyproject.toml)
- `torch==2.9.1` (cu128 wheels)
- `kernels>=0.11.7` — Hugging Face `kernels` loader used to pull Flash Attention 3 kernels (`varunneal/flash-attention-3` on Hopper, `kernels-community/flash-attn3` elsewhere).
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
train.py         — GPT model + MuonAdamW optimizer + training loop (only file the agent edits)
program.md       — agent skill/instructions
analysis.ipynb   — post-hoc results inspection
pyproject.toml   — deps
.gitignore       — hides results/, worktrees/, queue/, dev/, results.tsv, CLAUDE.md, AGENTS.md
README.md        — user-facing overview
progress.png     — teaser chart
```

## Conventions & Patterns
- **Two-file contract**: `prepare.py` (frozen fixed metric + data path) vs. `train.py` (agent playground).
- **Fixed 5-min wall-clock time budget** (`TIME_BUDGET = 300`); comparable across compute platforms.
- **Metric**: `val_bpb` (bits per byte) — vocab-size-independent, sums nats over target-byte lengths on a pinned validation shard (shard 6542).
- **Hyperparameters as top-level constants** in `train.py` (no CLI flags). Agents flip values inline.
- Model uses **RoPE, GQA-ready but n_kv_head=n_head today, ReLU² MLP, RMSNorm pre-norm, per-layer resid+x0 scalars, ResFormer value-embed residual, sliding-window pattern "SSSL"** (S=half context, L=full).
- Optimizer is a **custom `MuonAdamW`** that fuses Newton–Schulz polar-express orthogonalization + NorMuon variance reduction + cautious weight decay in a `@torch.compile`d step, with AdamW for embeddings / lm_head / scalars.
- Precision: **bfloat16 autocast + fp32 loss**, `torch.set_float32_matmul_precision("high")`, Flash Attention 3 kernels, `torch.compile(model, dynamic=False)`.
- Data pipeline: **BOS-aligned best-fit packing** to 100% token utilization, pinned-CPU → non-blocking GPU copy.
- MFU reported against `H100_BF16_PEAK_FLOPS = 989.5e12`.

## Platform Reality
- README explicitly calls the code NVIDIA/H100 only, and points to community forks for MacOS (`miolini/autoresearch-macos`, `trevin-creator/autoresearch-mlx`), Windows RTX, and AMD ROCm.
- CUDA-specific call sites: `torch.cuda.get_device_capability`, `torch.cuda.synchronize`, `torch.cuda.manual_seed`, `torch.cuda.max_memory_allocated`, `torch.amp.autocast(device_type="cuda", …)`, `device="cuda"`, `pin_memory=True`, `expandable_segments` env, Flash Attention 3 kernels via `kernels`.

## MLX Target (verified against docs 0.32.0)
- `mlx.core` + `mlx.nn` cover: `Embedding`, `Linear` (bias optional), `RMSNorm`, `RoPE`, `mx.fast.scaled_dot_product_attention(q, k, v, scale=…, mask="causal" | array)`, GQA (uneven `N_q` vs `N_kv`), bfloat16 dtype on Metal, `mx.compile(shapeless=…)`, `mx.eval`, `mx.value_and_grad` / `nn.value_and_grad`.
- `mlx.optimizers` provides `Muon` (Newton–Schulz, `momentum`, `weight_decay`, `nesterov`, `ns_steps`), `AdamW`, `MultiOptimizer(optimizers, filters=[…])` for per-parameter-group routing, and schedulers (`cosine_decay`, `linear_schedule`, `join_schedules`).
- **No built-in sliding-window mask** in SDPA — must build a boolean mask, or drop `SSSL` in favor of pure `L` (baseline for smaller compute per README).
- **No NorMuon / polar-express variant in stock MLX Muon** — trade off exact optimizer fidelity for the simpler, first-party path.
- Memory reporting: `mx.get_peak_memory()`, `mx.reset_peak_memory()`, `mx.metal.device_info()`.
- MLX uses **unified memory** — no pinned-host / non-blocking copies needed; feed `mx.array` directly from numpy.
