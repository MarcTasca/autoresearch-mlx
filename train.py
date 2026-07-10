"""
Autoresearch pretraining script — Apple Silicon / MLX fork.
Single Mac GPU (unified memory), single file. Runs a 5-minute experiment.
Usage: uv run train.py
"""


import gc
import math
import random
import time
from dataclasses import dataclass, asdict
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Platform constant (informational, drives MFU%)
# ---------------------------------------------------------------------------
# Rough BF16 peak for M-series GPUs. This is informational — MFU is reported
# for continuity with the upstream summary block, not used to gate anything.
# Override for your specific chip if you want a more accurate MFU number.
M_SERIES_BF16_PEAK_FLOPS = 1.5e13

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def rms_norm(x):
    """RMSNorm along the last axis, no learnable scale."""
    return mx.fast.rms_norm(x, weight=None, eps=1e-6)


def has_ve(layer_idx, n_layer):
    """Layers with a Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim ** -0.5

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels = 32
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=10000)
        self.has_value_embed = has_ve(layer_idx, config.n_layer)
        if self.has_value_embed:
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def __call__(self, x, ve):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # ResFormer value residual: input-dependent per-head sigmoid gate on
        # first 32 channels, applied to the pre-transpose value tensor.
        if self.has_value_embed and ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, -1) * ve

        # Move to [B, N, T, D] for SDPA (MLX convention).
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # RoPE on Q/K along the sequence axis (-2), then RMSNorm on Q/K.
        q = self.rope(q)
        k = self.rope(k)
        q = rms_norm(q)
        k = rms_norm(k)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(B, T, self.n_embd)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nn.relu(x)
        x = x * x  # ReLU² activation
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve):
        # Cast to bf16 at the block boundary so RoPE cos/sin, Q/K/V, and the
        # attention/MLP matmuls all execute in bf16 (matches upstream autocast).
        x = x.astype(mx.bfloat16)
        x = x + self.attn(rms_norm(x), ve)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # Explicit sub-module container so parameter-tree paths are stable.
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer.h = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones(config.n_layer)
        self.x0_lambdas = mx.zeros(config.n_layer)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        # Value embeddings only on the layers that use them (dict-keyed by index).
        # Non-numeric keys avoid mlx.utils.tree_unflatten treating them as list
        # indices — which breaks MultiOptimizer's sparse tree_map through
        # dict-of-modules with numeric keys.
        self.value_embeds = {
            f"ve_{i}": nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        }

    def init_weights(self):
        cfg = self.config
        n_embd = cfg.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5  # uniform bound

        # Embedding + unembedding
        self.transformer.wte.weight = mx.random.normal(
            shape=self.transformer.wte.weight.shape, loc=0.0, scale=1.0)
        self.lm_head.weight = mx.random.normal(
            shape=self.lm_head.weight.shape, loc=0.0, scale=0.001)

        # Transformer block matrices
        for block in self.transformer.h:
            block.attn.c_q.weight = mx.random.uniform(
                low=-s, high=s, shape=block.attn.c_q.weight.shape)
            block.attn.c_k.weight = mx.random.uniform(
                low=-s, high=s, shape=block.attn.c_k.weight.shape)
            block.attn.c_v.weight = mx.random.uniform(
                low=-s, high=s, shape=block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros(block.attn.c_proj.weight.shape)
            block.mlp.c_fc.weight = mx.random.uniform(
                low=-s, high=s, shape=block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros(block.mlp.c_proj.weight.shape)
            if block.attn.has_value_embed:
                block.attn.ve_gate.weight = mx.zeros(block.attn.ve_gate.weight.shape)

        # Per-layer scalar mixers
        self.resid_lambdas = mx.ones(cfg.n_layer)
        self.x0_lambdas = mx.full(shape=(cfg.n_layer,), vals=0.1)

        # Value embeddings share the same uniform-init distribution as block matrices
        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(low=-s, high=s, shape=ve.weight.shape)

        # Cast bf16-eligible weights (embeddings + block matrices) to bf16 to
        # match upstream to(dtype=bf16); matmul dtype promotion then flows bf16
        # through Q/K into RoPE for cos/sin table computation.
        self.transformer.wte.weight = self.transformer.wte.weight.astype(mx.bfloat16)
        for ve in self.value_embeds.values():
            ve.weight = ve.weight.astype(mx.bfloat16)
        for block in self.transformer.h:
            block.attn.c_q.weight = block.attn.c_q.weight.astype(mx.bfloat16)
            block.attn.c_k.weight = block.attn.c_k.weight.astype(mx.bfloat16)
            block.attn.c_v.weight = block.attn.c_v.weight.astype(mx.bfloat16)
            block.attn.c_proj.weight = block.attn.c_proj.weight.astype(mx.bfloat16)
            if block.attn.has_value_embed:
                block.attn.ve_gate.weight = block.attn.ve_gate.weight.astype(mx.bfloat16)
            block.mlp.c_fc.weight = block.mlp.c_fc.weight.astype(mx.bfloat16)
            block.mlp.c_proj.weight = block.mlp.c_proj.weight.astype(mx.bfloat16)

    def num_scaling_params(self):
        wte = int(self.transformer.wte.weight.size)
        value_embeds = sum(int(ve.weight.size) for ve in self.value_embeds.values())
        lm_head = int(self.lm_head.weight.size)
        transformer_matrices = sum(
            int(v.size) for k, v in tree_flatten({"h": self.transformer.h}))
        scalars = int(self.resid_lambdas.size) + int(self.x0_lambdas.size)
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars,
            'total': total,
        }

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward). Pure causal attention."""
        cfg = self.config
        total_params = self.num_scaling_params()['total']
        value_embeds_numel = sum(
            int(ve.weight.size) for ve in self.value_embeds.values())
        exclude = (int(self.transformer.wte.weight.size) + value_embeds_numel
                   + int(self.resid_lambdas.size) + int(self.x0_lambdas.size))
        # Per-layer attention flops on full causal window
        h, q_dim, t = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.sequence_len
        attn_flops = cfg.n_layer * 12 * h * q_dim * t
        return 6 * (total_params - exclude) + attn_flops

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape

        x = self.transformer.wte(idx)               # bf16 embedding lookup
        x = rms_norm(x)
        x0 = x

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            key = f"ve_{i}"
            ve = self.value_embeds[key](idx) if key in self.value_embeds else None
            x = block(x, ve)

        x = rms_norm(x)

        # Logits + softcap in fp32 for numerical stability
        logits = self.lm_head(x).astype(mx.float32)
        softcap = 15.0
        logits = softcap * mx.tanh(logits / softcap)

        if targets is None:
            return logits

        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction=reduction,
        )
        return loss

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention

# Optimization
TOTAL_BATCH_SIZE = 2**17  # ~131K tokens per optimizer step (M-series baseline)
EMBEDDING_LR = 0.6      # base AdamW LR (routed to embeddings, lm_head, scalars)
MATRIX_LR = 0.04        # Muon LR for 2-D matrix params (attn / mlp)
WEIGHT_DECAY = 0.2      # Muon weight decay (annealed toward 0 over the run)
ADAM_BETAS = (0.8, 0.95) # AdamW betas
UNEMBEDDING_LR = 0.004  # AdamW LR for lm_head (upstream default)
SCALAR_LR = 0.5         # AdamW LR for per-layer scalars (upstream default)
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size (M5 Pro 48 GB baseline)
DEPTH = 6               # number of transformer layers
DEVICE_BATCH_SIZE = 32  # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup: seed, tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)
random.seed(42)
np.random.seed(42)
mx.reset_peak_memory()

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth: int) -> GPTConfig:
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# ---------------------------------------------------------------------------
# Optimizer: MultiOptimizer with Muon (matrices) + four AdamW groups matching
# the upstream per-group hyperparameters (lm_head / embeddings / resid / x0).
# ---------------------------------------------------------------------------

model_dim = config.n_embd
# 1/sqrt(dmodel) LR scaling applied to embedding-family AdamW LRs only
# (lm_head, wte, value_embeds); scalar-group LRs are left untouched.
dmodel_lr_scale = (model_dim / 768) ** -0.5
print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

lm_head_base_lr   = UNEMBEDDING_LR * dmodel_lr_scale
embedding_base_lr = EMBEDDING_LR * dmodel_lr_scale
resid_base_lr     = SCALAR_LR * 0.01
x0_base_lr        = SCALAR_LR

# Routing predicates. First matching predicate wins; the last optimizer in the
# MultiOptimizer list is the fallback (no filter). Order matters.
def is_muon_matrix(path: str, param: mx.array) -> bool:
    """Route 2-D matrix params under transformer.h.* to Muon."""
    return param.ndim == 2 and path.startswith("transformer.h.")

def is_lm_head(path: str, param: mx.array) -> bool:
    return path == "lm_head.weight"

def is_embedding(path: str, param: mx.array) -> bool:
    return path == "transformer.wte.weight" or path.startswith("value_embeds.")

def is_resid_lambda(path: str, param: mx.array) -> bool:
    return path == "resid_lambdas"

muon = optim.Muon(
    learning_rate=MATRIX_LR,
    momentum=0.95,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
    ns_steps=5,
)
adamw_lm_head = optim.AdamW(
    learning_rate=lm_head_base_lr,
    betas=list(ADAM_BETAS),
    eps=1e-10,
    weight_decay=0.0,
)
adamw_embeds = optim.AdamW(
    learning_rate=embedding_base_lr,
    betas=list(ADAM_BETAS),
    eps=1e-10,
    weight_decay=0.0,
)
adamw_resid = optim.AdamW(
    learning_rate=resid_base_lr,
    betas=list(ADAM_BETAS),
    eps=1e-10,
    weight_decay=0.0,
)
# x0_lambdas is the fallback group; upstream uses a distinct beta1=0.96.
adamw_x0 = optim.AdamW(
    learning_rate=x0_base_lr,
    betas=[0.96, 0.95],
    eps=1e-10,
    weight_decay=0.0,
)
optimizer = optim.MultiOptimizer(
    [muon, adamw_lm_head, adamw_embeds, adamw_resid, adamw_x0],
    filters=[is_muon_matrix, is_lm_head, is_embedding, is_resid_lambda],
)
optimizer.init(model.trainable_parameters())

# Snapshot initial LRs so the schedule multiplier scales the *base* LR each step.
lr_schedule_targets = [
    (muon,          MATRIX_LR),
    (adamw_lm_head, lm_head_base_lr),
    (adamw_embeds,  embedding_base_lr),
    (adamw_resid,   resid_base_lr),
    (adamw_x0,      x0_base_lr),
]

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Schedules (all based on progress = training_time / TIME_BUDGET)
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress: float) -> float:
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Compiled fwd + value_and_grad step (per microbatch)
# ---------------------------------------------------------------------------

def _loss_fn(model, x, y):
    return model(x, y)

loss_and_grad_fn = nn.value_and_grad(model, _loss_fn)

# Capture model + random state so mx.compile doesn't retrace across steps.
_micro_state = [model.state, mx.random.state]

@partial(mx.compile, inputs=_micro_state, outputs=_micro_state)
def micro_step(x, y):
    return loss_and_grad_fn(model, x, y)

# Compile the accumulator-scale + optimizer.update pass over model + optimizer
# state; keeps tree-sum accumulation in Python (needed for grad accumulation)
# but hands the scale-and-apply step to MLX end-to-end.
_apply_state = [model.state, optimizer.state]

@partial(mx.compile, inputs=_apply_state, outputs=_apply_state)
def apply_fn(accum_grads):
    scaled = tree_map(lambda g: g * inv_accum, accum_grads)
    optimizer.update(model, scaled)

def _tree_add(a, b):
    return tree_map(lambda u, v: u + v, a, b)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

smooth_train_loss = 0.0
total_training_time = 0.0
step = 0

inv_accum = 1.0 / grad_accum_steps

while True:
    mx.eval(model.parameters())  # ensure a clean sync before timing
    t0 = time.time()

    accum_grads = None
    losses = []
    for _ in range(grad_accum_steps):
        loss, grads = micro_step(x, y)
        # Materialize per-microstep results so intermediate activations don't
        # pile up in the compute graph (peak-memory savings on unified memory).
        if accum_grads is None:
            mx.eval(loss, grads)
            accum_grads = grads
        else:
            accum_grads = _tree_add(accum_grads, grads)
            mx.eval(loss, accum_grads)
        losses.append(loss)
        x, y, epoch = next(train_loader)
    last_loss = losses[-1]

    # Progress-driven schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for opt_ref, base_lr in lr_schedule_targets:
        opt_ref.learning_rate = base_lr * lrm
    muon.momentum = get_muon_momentum(step)
    muon.weight_decay = get_weight_decay(progress)

    apply_fn(accum_grads)
    mx.eval(last_loss, model.parameters(), optimizer.state)

    train_loss_f = float(last_loss.item())

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / M_SERIES_BF16_PEAK_FLOPS if dt > 0 else 0
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    # GC management (Python's GC causes stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
steady_state_mfu = (100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10)
                    / total_training_time / M_SERIES_BF16_PEAK_FLOPS
                    if total_training_time > 0 else 0)
peak_vram_mb = mx.get_peak_memory() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
