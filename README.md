# autoresearch (MLX / Apple Silicon fork)

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. This fork rewrites the training harness on top of **MLX** so it runs on a single **Apple Silicon** GPU (unified memory, M-series) — no PyTorch, no CUDA, no Hugging Face `kernels`. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obviously very hackable.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW via `MultiOptimizer`), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single Apple Silicon Mac (M1 or newer, unified memory; reference target: **M5 Pro, 48 GB**), Python 3.10+, [uv](https://docs.astral.sh/uv/). macOS/arm64 only — this fork has no CPU/CUDA fallback by design.

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (installs mlx, no torch, no CUDA wheels)
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies (mlx only, no torch, no kernels)
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond MLX and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.
- **First-party MLX only.** No custom Metal kernels, no Newton–Schulz variants, no sliding-window attention plumbing. Uses stock `Muon` + `AdamW` via `MultiOptimizer` and `mx.fast.scaled_dot_product_attention` with a causal mask.

## Platform support

This fork requires a single **Apple Silicon** GPU (M-series). It is deliberately Mac-only: no CPU fallback, no CUDA path. If you want the NVIDIA / H100 harness, use the upstream `karpathy/autoresearch` repo directly.

A few tuning knobs to consider on smaller Macs (or if you hit `[metal::Device] Not enough memory`):

1. Lower `DEVICE_BATCH_SIZE` in `train.py` first (halve it until the run boots).
2. Then lower `DEPTH` (fewer transformer layers).
3. Then lower `TOTAL_BATCH_SIZE` (keep it a power of two).
4. For much narrower data, consider a lower-entropy dataset like [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean), a smaller `vocab_size` in `prepare.py`, or a shorter `MAX_SEQ_LEN` (again in `prepare.py`, note this changes the metric shape).
5. `EVAL_TOKENS` in `prepare.py` also drives eval cost — trim it if the final validation phase eats too much wall clock on your machine.

The `M_SERIES_BF16_PEAK_FLOPS` constant at the top of `train.py` is a rough estimate for `mfu_percent`; override it for your specific chip if you want a more accurate number.

## Notable forks

- Upstream (NVIDIA / H100 / CUDA): [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
