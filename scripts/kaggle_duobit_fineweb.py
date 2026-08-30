#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DUOBIT-EST vs FP32 / FP16 on FineWeb-EDU  --  single-file DDP experiment (v2)
============================================================================

Trains the SAME transformer architecture under several weight regimes on the
SAME token stream, then reports quality, memory and throughput side by side:

  1) DUOBIT-EST  : native 2-bit (n_levels=4, rho=1/3) linear maps stored as
                   discrete codes + per-group scales, NO FP32 master weights.
                   Training residual carried in a quantized error-feedback
                   accumulator (QPEFA, int8 stochastic-rounded). Adam moments
                   compressed: block-wise int8 first moment + Adafactor-style
                   factored second moment. Custom CUDA kernels accelerate the
                   quantization math on the GPU (PyTorch fallback otherwise).

  2) FP32        : identical architecture, ordinary nn.Linear + AdamW.

  3) FP16        : identical architecture, ordinary nn.Linear + AdamW under
                   torch.autocast(float16) with a dynamic GradScaler, i.e. the
                   standard mixed-precision recipe (FP32 master weights).

Same model size, same initialization seed, same data (FineWeb-EDU streaming,
GPT-2 tokenizer), same number of steps / batches / tokens => same compute.

v2 changes over v1 (the tuning pass)
------------------------------------
v1 recomputed the group scale from the *dequantized* weight:
`s* = sum(w_i c_i) / sum(c_i^2)` with `w_i = s c_i` returns `s` exactly, so the
"EMA scale update" was an algebraic fixed point and the scales stayed frozen at
their initialization for the whole run (measured: scale_mean moved by 2e-7
relative over 3000 steps). The only continuous degree of freedom in the method
was therefore never trained. v2 fixes that and the issues around it:

  * `--learn-scales`  : group scales get their own Adam step from the exact
                        analytic gradient  dL/ds_g = sum_{i in g} g_i * c_i.
                        Scales are already part of the persistent 2-bit
                        representation (32/G bits per weight, kept at
                        inference), so this does NOT reintroduce master
                        weights; it adds 2*32/G bits/weight of optimizer state.
  * `--scale-relative-lr` : the discrete step is measured in units of the group
                        scale (dw = lr * s_g * Adam_dir) so one learning rate is
                        commensurate with the codebook gap in every layer.
  * `--pefa-clip 1.0` : the post-quantization residual obeys |e| <= 0.5*gap =
                        0.333*s, so clipping the accumulator at 4*s spent 12x of
                        the int8 residual grid on unreachable values. Tightening
                        the clip multiplies the effective residual resolution.
  * `--duobit-wd 0.0` : decoupled weight decay is meaningless for codes whose
                        magnitude is carried by s_g; it only biased transitions.
  * a fused CUDA kernel for the whole QPEFA/transition path and removal of the
    per-layer, per-step `.item()` synchronisations (v1 issued ~150 device syncs
    per step), which is where most of the v1 optimizer time went.

Outputs (in --out-dir), per run name:
  logs_<name>.jsonl        per-step training logs (rank 0)
  step_logs_<name>.csv     the same as CSV
  val_curve_<name>.json    validation curve
  summary_<name>.json      final summary + memory accounting
  all_metrics.json, comparison.csv
  summary_report.md        full written report
  figures/*.png            every visualization
  dashboard.png            combined dashboard

Method reference: DuoBIT-EST (Pratyush Bhardwaj, 2026) --
"Error-Compensated Stochastic Transition Training of Language Models at Two
Bits and Below Without Master Weights". This script re-implements the method
in one file (codebook, QPEFA, compressed Adam, DuobitLinear, DuobitAdam)
faithfully to the reference package in the DuoBIT repository.
"""
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import csv
import gc
import importlib
import json
import math
import queue
import random
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as tmp_mp

__version__ = "1.0.0"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DUOBIT-EST vs FP32/FP16 on FineWeb-EDU (DDP)")
    # experiment
    p.add_argument("--mode", default="both",
                   choices=["both", "all", "duobit", "fp32", "fp16"],
                   help="'both' = duobit+fp32, 'all' = duobit+fp32+fp16")
    p.add_argument("--runs-json", default=None,
                   help="JSON list of run specs, e.g. "
                        "'[{\"name\":\"A\",\"mode\":\"duobit\",\"learn_scales\":0}]'. "
                        "Every key other than 'name' overrides an argparse dest for "
                        "that run only. Overrides --mode when given.")
    p.add_argument("--steps", type=int, default=3000, help="optimizer steps per run")
    p.add_argument("--batch-size", type=int, default=8, help="sequences per GPU")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--eval-freq", type=int, default=250)
    p.add_argument("--log-freq", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default=None, help="default: /kaggle/working or ./duobit_out")
    p.add_argument("--max-gpus", type=int, default=0, help="0 = use all visible GPUs")
    p.add_argument("--smoke", action="store_true", help="tiny config for local verification")
    p.add_argument("--no-kernels", action="store_true", help="force PyTorch fallback ops")
    # model
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1408)
    p.add_argument("--dropout", type=float, default=0.0)
    # duobit
    p.add_argument("--n-levels", type=int, default=4, choices=[2, 3, 4],
                   help="4 = 2-bit {-1,-rho,+rho,+1}, 3 = ternary, 2 = binary")
    p.add_argument("--rho", type=float, default=1.0 / 3.0)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--qpefa-bits", type=int, default=8)
    p.add_argument("--moment-bits", type=int, default=8)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--duobit-lr", type=float, default=8e-3)
    p.add_argument("--pefa-clip", type=float, default=1.0,
                   help="QPEFA accumulator clip in units of the group scale "
                        "(v1 default was 4.0; the reachable residual is 0.333*s)")
    p.add_argument("--scale-update-freq", type=int, default=0,
                   help="legacy MSE-EMA rescale period; 0 disables it (it is a "
                        "fixed point on dequantized weights and does nothing)")
    p.add_argument("--scale-ema", type=float, default=0.0)
    # v2: trainable group scales
    p.add_argument("--learn-scales", type=int, default=1, choices=[0, 1],
                   help="train the per-group scales with their own Adam step "
                        "from the analytic gradient sum_i g_i c_i")
    p.add_argument("--scale-lr", type=float, default=1e-3,
                   help="relative LR for the group scales: ds = -lr * s * Adam_dir")
    p.add_argument("--scale-lr-abs", type=int, default=0, choices=[0, 1],
                   help="use an absolute (not scale-relative) LR for the scales")
    p.add_argument("--scale-min-frac", type=float, default=0.05,
                   help="lower clamp on a group scale as a fraction of its init")
    p.add_argument("--scale-relative-lr", type=int, default=1, choices=[0, 1],
                   help="measure the discrete step in units of s_g: dw = lr*s_g*dir")
    p.add_argument("--duobit-wd", type=float, default=0.0,
                   help="weight decay applied inside the discrete update "
                        "(v1 used --wd here; 0 is the v2 default)")
    p.add_argument("--scale-init", default="var", choices=["var", "mse"],
                   help="group-scale fit at init: 'mse' minimises ||w - s C[z]|| "
                        "(v1, shrinks weight std by ~4.4%% per layer); 'var' "
                        "reproduces the intended per-group std")
    p.add_argument("--init-std", type=float, default=0.0,
                   help="0 = kaiming_uniform (v1); >0 = normal(0,std) init with "
                        "1/sqrt(2*n_layers) scaling on the residual projections")
    # optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", type=int, default=0, choices=[0, 1],
                   help="run this mode under torch.autocast(float16); the fp16 "
                        "mode turns it on automatically")
    # data
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-100BT")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--shuffle-buffer", type=int, default=1024,
                   help="doc-level shuffle buffer size")
    p.add_argument("--val-tokens", type=int, default=131072, help="val tokens per rank")
    p.add_argument("--prefetch", type=int, default=8, help="prefetched batches in queue")
    p.add_argument("--sample-tokens", type=int, default=48, help="greedy sample length")
    return p


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Tiny end-to-end configuration for local CPU verification."""
    args.steps = 10
    args.batch_size = 2
    args.seq_len = 128
    args.d_model = 64
    args.n_layers = 2
    args.n_heads = 4
    args.d_ff = 192
    args.group_size = 32
    args.eval_freq = 5
    args.log_freq = 1
    args.warmup_steps = 2
    args.val_tokens = 4096
    args.shuffle_buffer = 64
    args.prefetch = 4
    args.sample_tokens = 24
    return args


# ----------------------------------------------------------------------------
# CUDA kernels for DuoBIT math (with pure-PyTorch fallback)
# ----------------------------------------------------------------------------
_CPP_SRC = r"""
#include <torch/extension.h>
void duobit_dequant(torch::Tensor codes, torch::Tensor scales, torch::Tensor levels,
                    torch::Tensor out, int64_t group_size);
void duobit_quant(torch::Tensor w, torch::Tensor scales, torch::Tensor levels,
                  torch::Tensor codes, torch::Tensor wq, torch::Tensor res,
                  int64_t group_size);
void qpefa_round(torch::Tensor e, torch::Tensor scales, torch::Tensor noise,
                 torch::Tensor q, double lam, double qmax, int64_t group_size);
void qpefa_dequant(torch::Tensor q, torch::Tensor scales, torch::Tensor e,
                   double lam, double qmax, int64_t group_size);
void blockwise_quant(torch::Tensor x, torch::Tensor q, torch::Tensor scales,
                     int64_t block, double qmax);
void blockwise_dequant(torch::Tensor q, torch::Tensor scales, torch::Tensor x,
                       int64_t block);
void duobit_fused_update(torch::Tensor codes, torch::Tensor scales,
                         torch::Tensor levels, torch::Tensor m,
                         torch::Tensor v_row, torch::Tensor v_col,
                         torch::Tensor e_q, torch::Tensor stats,
                         int64_t group_size, double d_lr, double eps,
                         double bc1, double bc2, double denom_mean, double wd,
                         double lam, double qmax, int64_t scale_relative,
                         int64_t collect_stats, int64_t seed);
void duobit_scale_grad(torch::Tensor grad, torch::Tensor codes,
                       torch::Tensor levels, torch::Tensor out,
                       int64_t group_size);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

#define TPB 256

__global__ void dequant_kernel(const uint8_t* __restrict__ codes,
                                const float* __restrict__ scales,
                                const float* __restrict__ levels,
                                float* __restrict__ out,
                                int64_t I, int64_t G, int Gs, int64_t total) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t o = idx / I;
    int64_t i = idx - o * I;
    float s = scales[o * G + (i / Gs)];
    out[idx] = levels[codes[idx]] * s;
}

__global__ void quant_kernel(const float* __restrict__ w,
                             const float* __restrict__ scales,
                             const float* __restrict__ levels,
                             uint8_t* __restrict__ codes,
                             float* __restrict__ wq,
                             float* __restrict__ res,
                             int64_t I, int64_t G, int Gs, int L, int64_t total) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t o = idx / I;
    int64_t i = idx - o * I;
    float s_raw = scales[o * G + (i / Gs)];
    float u = w[idx] / (s_raw + 1e-8f);
    int best = 0;
    float bd = fabsf(u - levels[0]);
    for (int k = 1; k < 4; k++) {
        if (k < L) {
            float d = fabsf(u - levels[k]);
            if (d < bd) { bd = d; best = k; }
        }
    }
    codes[idx] = (uint8_t)best;
    float wqv = levels[best] * s_raw;
    if (wq != nullptr) wq[idx] = wqv;
    if (res != nullptr) res[idx] = w[idx] - wqv;
}

__global__ void qpefa_round_kernel(const float* __restrict__ e,
                                   const float* __restrict__ scales,
                                   const float* __restrict__ noise,
                                   int8_t* __restrict__ q,
                                   int64_t I, int64_t G, int Gs,
                                   float lam, float qmax, int64_t total) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t o = idx / I;
    int64_t i = idx - o * I;
    float s = scales[o * G + (i / Gs)];
    float denom = lam * fmaxf(s, 1e-8f);
    float norm = (e[idx] / denom) * qmax;
    if (norm > qmax) norm = qmax;
    if (norm < -qmax) norm = -qmax;
    float v = floorf(norm + noise[idx]);
    if (v > qmax) v = qmax;
    if (v < -qmax) v = -qmax;
    q[idx] = (int8_t)v;
}

__global__ void qpefa_dequant_kernel(const int8_t* __restrict__ q,
                                     const float* __restrict__ scales,
                                     float* __restrict__ e,
                                     int64_t I, int64_t G, int Gs,
                                     float lam, float qmax, int64_t total) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t o = idx / I;
    int64_t i = idx - o * I;
    float s = scales[o * G + (i / Gs)];
    e[idx] = ((float)q[idx] / qmax) * (lam * fmaxf(s, 1e-8f));
}

__global__ void blockwise_quant_kernel(const float* __restrict__ x,
                                       int8_t* __restrict__ q,
                                       float* __restrict__ scales,
                                       int64_t n_blocks, int B, float qmax) {
    int64_t b = blockIdx.x;
    if (b >= n_blocks) return;
    extern __shared__ float sdata[];
    int64_t base = b * (int64_t)B;
    float lm = 0.0f;
    for (int j = threadIdx.x; j < B; j += blockDim.x) {
        float v = fabsf(x[base + j]);
        if (v > lm) lm = v;
    }
    sdata[threadIdx.x] = lm;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            float o = sdata[threadIdx.x + off];
            if (o > sdata[threadIdx.x]) sdata[threadIdx.x] = o;
        }
        __syncthreads();
    }
    float absmax = sdata[0];
    if (absmax < 1e-8f) absmax = 1e-8f;
    float scale = absmax / qmax;
    if (threadIdx.x == 0) scales[b] = scale;
    __syncthreads();
    for (int j = threadIdx.x; j < B; j += blockDim.x) {
        float v = roundf(x[base + j] / scale);
        if (v > qmax) v = qmax;
        if (v < -qmax) v = -qmax;
        q[base + j] = (int8_t)v;
    }
}

__global__ void blockwise_dequant_kernel(const int8_t* __restrict__ q,
                                         const float* __restrict__ scales,
                                         float* __restrict__ x,
                                         int64_t N, int B) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] = ((float)q[i]) * scales[i / B];
}

void duobit_dequant(torch::Tensor codes, torch::Tensor scales, torch::Tensor levels,
                    torch::Tensor out, int64_t group_size) {
    TORCH_CHECK(codes.is_cuda() && scales.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    int64_t O = codes.size(0), I = codes.size(1);
    int64_t G = I / group_size;
    int Gs = (int)group_size;
    int64_t total = codes.numel();
    int blocks = (int)((total + TPB - 1) / TPB);
    auto stream = at::cuda::getCurrentCUDAStream();
    dequant_kernel<<<blocks, TPB, 0, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<float>(), levels.data_ptr<float>(),
        out.data_ptr<float>(), I, G, Gs, total);
}

void duobit_quant(torch::Tensor w, torch::Tensor scales, torch::Tensor levels,
                  torch::Tensor codes, torch::Tensor wq, torch::Tensor res,
                  int64_t group_size) {
    TORCH_CHECK(w.is_cuda(), "tensors must be CUDA");
    int64_t O = w.size(0), I = w.size(1);
    int64_t G = I / group_size;
    int Gs = (int)group_size;
    int L = (int)levels.size(0);
    int64_t total = w.numel();
    int blocks = (int)((total + TPB - 1) / TPB);
    auto stream = at::cuda::getCurrentCUDAStream();
    quant_kernel<<<blocks, TPB, 0, stream>>>(
        w.data_ptr<float>(), scales.data_ptr<float>(), levels.data_ptr<float>(),
        codes.data_ptr<uint8_t>(), wq.data_ptr<float>(), res.data_ptr<float>(),
        I, G, Gs, L, total);
}

void qpefa_round(torch::Tensor e, torch::Tensor scales, torch::Tensor noise,
                 torch::Tensor q, double lam, double qmax, int64_t group_size) {
    TORCH_CHECK(e.is_cuda(), "tensors must be CUDA");
    int64_t O = e.size(0), I = e.size(1);
    int64_t G = I / group_size;
    int Gs = (int)group_size;
    int64_t total = e.numel();
    int blocks = (int)((total + TPB - 1) / TPB);
    auto stream = at::cuda::getCurrentCUDAStream();
    qpefa_round_kernel<<<blocks, TPB, 0, stream>>>(
        e.data_ptr<float>(), scales.data_ptr<float>(), noise.data_ptr<float>(),
        q.data_ptr<int8_t>(), I, G, Gs, (float)lam, (float)qmax, total);
}

void qpefa_dequant(torch::Tensor q, torch::Tensor scales, torch::Tensor e,
                   double lam, double qmax, int64_t group_size) {
    TORCH_CHECK(q.is_cuda(), "tensors must be CUDA");
    int64_t O = q.size(0), I = q.size(1);
    int64_t G = I / group_size;
    int Gs = (int)group_size;
    int64_t total = q.numel();
    int blocks = (int)((total + TPB - 1) / TPB);
    auto stream = at::cuda::getCurrentCUDAStream();
    qpefa_dequant_kernel<<<blocks, TPB, 0, stream>>>(
        q.data_ptr<int8_t>(), scales.data_ptr<float>(), e.data_ptr<float>(),
        I, G, Gs, (float)lam, (float)qmax, total);
}

void blockwise_quant(torch::Tensor x, torch::Tensor q, torch::Tensor scales,
                     int64_t block, double qmax) {
    TORCH_CHECK(x.is_cuda(), "tensors must be CUDA");
    int64_t n = x.numel();
    TORCH_CHECK(n % block == 0, "x must be padded to a multiple of block");
    int64_t nb = n / block;
    int B = (int)block;
    int threads = 1;
    while (threads * 2 <= B && threads < 1024) threads *= 2;
    size_t smem = threads * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    blockwise_quant_kernel<<<(int)nb, threads, smem, stream>>>(
        x.data_ptr<float>(), q.data_ptr<int8_t>(), scales.data_ptr<float>(),
        nb, B, (float)qmax);
}

void blockwise_dequant(torch::Tensor q, torch::Tensor scales, torch::Tensor x,
                       int64_t block) {
    TORCH_CHECK(q.is_cuda(), "tensors must be CUDA");
    int64_t n = q.numel();
    int blocks = (int)((n + TPB - 1) / TPB);
    auto stream = at::cuda::getCurrentCUDAStream();
    blockwise_dequant_kernel<<<blocks, TPB, 0, stream>>>(
        q.data_ptr<int8_t>(), scales.data_ptr<float>(), x.data_ptr<float>(),
        n, (int)block);
}

// ---------------------------------------------------------------------------
// v2: one fused pass for the whole discrete update.
//
// Replaces ~12 separate element-wise passes (outer product, Adam direction,
// dequant, residual dequant, clamp, quantize, residual re-quantize, ...) with
// a single read of {codes, m, e_q} and a single write of {codes, e_q}. The
// stochastic-rounding noise is generated in-register from a splitmix64 hash of
// (index, seed) so it is bit-identical on every DDP rank without transferring
// or materialising a full-size random tensor.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float hash_u01(uint64_t x, uint64_t seed) {
    uint64_t z = x + seed * 0x9E3779B97F4A7C15ULL + 0x165667B19E3779F9ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    // 24 mantissa bits -> [0,1)
    return (float)(z >> 40) * (1.0f / 16777216.0f);
}

__global__ void fused_update_kernel(uint8_t* __restrict__ codes,
                                    const float* __restrict__ scales,
                                    const float* __restrict__ levels,
                                    const float* __restrict__ m,
                                    const float* __restrict__ v_row,
                                    const float* __restrict__ v_col,
                                    int8_t* __restrict__ e_q,
                                    float* __restrict__ stats,
                                    int64_t I, int64_t G, int Gs, int L,
                                    float d_lr, float eps, float bc1, float bc2,
                                    float denom_mean, float wd, float lam,
                                    float qmax, int scale_relative,
                                    int collect_stats, uint64_t seed,
                                    int64_t total) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float l_changed = 0.0f, l_res = 0.0f;
    if (idx < total) {
        int64_t o = idx / I;
        int64_t i = idx - o * I;
        float s = scales[o * G + (i / Gs)];
        uint8_t c_old = codes[idx];
        float w_curr = levels[c_old] * s;

        float mh = m[idx] / bc1;
        float vh = (v_row[o] * v_col[i] / denom_mean) / bc2;
        float step = mh / (sqrtf(fmaxf(vh, 0.0f)) + eps);
        if (wd != 0.0f) step += wd * w_curr;

        float lr_eff = scale_relative ? (d_lr * s) : d_lr;
        float lam_s = lam * fmaxf(s, 1e-8f);

        // QPEFA: e <- clip(e + dw, +-lam*s); quantize (w_curr + e) onto the grid
        float e = ((float)e_q[idx] / qmax) * lam_s - lr_eff * step;
        e = fminf(fmaxf(e, -lam_s), lam_s);
        float wv = w_curr + e;

        float u = wv / (s + 1e-8f);
        int best = 0;
        float bd = fabsf(u - levels[0]);
        for (int k = 1; k < 4; k++) {
            if (k < L) {
                float d = fabsf(u - levels[k]);
                if (d < bd) { bd = d; best = k; }
            }
        }
        float wq = levels[best] * s;
        float e_true = wv - wq;

        float norm = (e_true / lam_s) * qmax;
        norm = fminf(fmaxf(norm, -qmax), qmax);
        float v = floorf(norm + hash_u01((uint64_t)idx, seed));
        v = fminf(fmaxf(v, -qmax), qmax);
        e_q[idx] = (int8_t)v;
        codes[idx] = (uint8_t)best;

        if (collect_stats) {
            l_changed = ((uint8_t)best != c_old) ? 1.0f : 0.0f;
            l_res = fabsf(e_true);
        }
    }
    if (collect_stats) {
        // block reduction -> one atomic per block per statistic
        __shared__ float sc[TPB];
        __shared__ float sr[TPB];
        sc[threadIdx.x] = l_changed;
        sr[threadIdx.x] = l_res;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
                sc[threadIdx.x] += sc[threadIdx.x + off];
                sr[threadIdx.x] += sr[threadIdx.x + off];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(&stats[0], sc[0]);
            atomicAdd(&stats[1], sr[0]);
        }
    }
}

// dL/ds_g = sum_{i in g} grad_i * C[z_i]   (one CUDA block per group)
__global__ void scale_grad_kernel(const float* __restrict__ grad,
                                  const uint8_t* __restrict__ codes,
                                  const float* __restrict__ levels,
                                  float* __restrict__ out,
                                  int64_t I, int64_t G, int Gs,
                                  int64_t n_groups) {
    int64_t gidx = blockIdx.x;
    if (gidx >= n_groups) return;
    int64_t o = gidx / G;
    int64_t g = gidx - o * G;
    int64_t base = o * I + g * (int64_t)Gs;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < Gs; j += blockDim.x)
        acc += grad[base + j] * levels[codes[base + j]];
    extern __shared__ float sm[];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[gidx] = sm[0];
}

void duobit_fused_update(torch::Tensor codes, torch::Tensor scales,
                         torch::Tensor levels, torch::Tensor m,
                         torch::Tensor v_row, torch::Tensor v_col,
                         torch::Tensor e_q, torch::Tensor stats,
                         int64_t group_size, double d_lr, double eps,
                         double bc1, double bc2, double denom_mean, double wd,
                         double lam, double qmax, int64_t scale_relative,
                         int64_t collect_stats, int64_t seed) {
    TORCH_CHECK(codes.is_cuda(), "tensors must be CUDA");
    int64_t I = codes.size(1);
    int64_t G = I / group_size;
    int64_t total = codes.numel();
    int blocks = (int)((total + TPB - 1) / TPB);
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_update_kernel<<<blocks, TPB, 0, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<float>(),
        levels.data_ptr<float>(), m.data_ptr<float>(),
        v_row.data_ptr<float>(), v_col.data_ptr<float>(),
        e_q.data_ptr<int8_t>(), stats.data_ptr<float>(),
        I, G, (int)group_size, (int)levels.size(0),
        (float)d_lr, (float)eps, (float)bc1, (float)bc2, (float)denom_mean,
        (float)wd, (float)lam, (float)qmax, (int)scale_relative,
        (int)collect_stats, (uint64_t)seed, total);
}

void duobit_scale_grad(torch::Tensor grad, torch::Tensor codes,
                       torch::Tensor levels, torch::Tensor out,
                       int64_t group_size) {
    TORCH_CHECK(grad.is_cuda(), "tensors must be CUDA");
    int64_t O = grad.size(0), I = grad.size(1);
    int64_t G = I / group_size;
    int64_t n_groups = O * G;
    int threads = 1;
    while (threads * 2 <= (int)group_size && threads < 1024) threads *= 2;
    size_t smem = threads * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    scale_grad_kernel<<<(int)n_groups, threads, smem, stream>>>(
        grad.data_ptr<float>(), codes.data_ptr<uint8_t>(),
        levels.data_ptr<float>(), out.data_ptr<float>(),
        I, G, (int)group_size, n_groups);
}
"""

_EXT = None
_EXT_TRIED = False


def _get_ext():
    """Compile (once) and validate the CUDA kernels; None => use fallback."""
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True
    if os.environ.get("DUOBIT_NO_KERNELS") == "1":
        print("[duobit] kernel compilation disabled (DUOBIT_NO_KERNELS=1)", flush=True)
        return None
    if not torch.cuda.is_available():
        print("[duobit] no CUDA device visible -> using PyTorch fallback ops", flush=True)
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        cap = torch.cuda.get_device_capability(0)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")
        t0 = time.time()
        ext = load_inline(
            name="duobit_kernels_v2",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=[
                "duobit_dequant", "duobit_quant", "qpefa_round", "qpefa_dequant",
                "blockwise_quant", "blockwise_dequant",
                "duobit_fused_update", "duobit_scale_grad",
            ],
            verbose=False,
        )
        ok = _kernel_parity_check(ext)
        if not ok:
            print("[duobit] kernel parity check FAILED -> falling back to PyTorch ops",
                  flush=True)
            return None
        _EXT = ext
        print(f"[duobit] CUDA kernels compiled+validated in {time.time()-t0:.1f}s "
              f"(sm_{cap[0]}{cap[1]})", flush=True)
    except Exception as e:  # pragma: no cover - depends on toolchain
        print(f"[duobit] CUDA kernel build failed ({type(e).__name__}: {e}) "
              f"-> using PyTorch fallback ops", flush=True)
        _EXT = None
    return _EXT


def _kernel_parity_check(ext) -> bool:
    """Compare kernel results against the PyTorch fallback on random data."""
    try:
        dev = torch.device("cuda")
        g = torch.Generator(device=dev)
        g.manual_seed(1234)
        O, I, Gs = 8, 256, 32
        G = I // Gs
        w = torch.randn(O, I, device=dev, generator=g)
        scales = (torch.rand(O, G, device=dev, generator=g) + 0.25).float()
        levels = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=dev)
        codes = torch.randint(0, 4, (O, I), device=dev, generator=g).to(torch.uint8)

        out_k = torch.zeros(O, I, device=dev)
        ext.duobit_dequant(codes, scales, levels, out_k, Gs)
        out_t = (levels[codes.long()].view(O, G, Gs) * scales.view(O, G, 1)).view(O, I)
        if not torch.allclose(out_k, out_t, atol=1e-6):
            return False

        c_k = torch.zeros(O, I, dtype=torch.uint8, device=dev)
        wq_k = torch.zeros(O, I, device=dev)
        res_k = torch.zeros(O, I, device=dev)
        ext.duobit_quant(w, scales, levels, c_k, wq_k, res_k, Gs)
        u = w.view(O, G, Gs) / (scales.view(O, G, 1) + 1e-8)
        distm = (u.unsqueeze(-1) - levels.view(1, 1, 1, -1)).abs()
        c_t = distm.argmin(-1).to(torch.uint8).view(O, I)
        wq_t = (levels[c_t.long()].view(O, G, Gs) * scales.view(O, G, 1)).view(O, I)
        if not (torch.equal(c_k, c_t) and torch.allclose(wq_k, wq_t, atol=1e-6)
                and torch.allclose(res_k, w - wq_t, atol=1e-6)):
            return False

        lam, bits = 4.0, 8
        qmax = float((1 << (bits - 1)) - 1)
        e = torch.randn(O, I, device=dev, generator=g) * 0.3
        noise = torch.rand(O, I, device=dev, generator=g)
        q_k = torch.zeros(O, I, dtype=torch.int8, device=dev)
        ext.qpefa_round(e, scales, noise, q_k, lam, qmax, Gs)
        max_res = (lam * scales.clamp(min=1e-8)).view(O, G, 1)
        norm = (e.view(O, G, Gs) / max_res).clamp(-1, 1) * qmax
        q_t = torch.floor(norm + noise.view(O, G, Gs)).clamp(-qmax, qmax).to(torch.int8)
        if not torch.equal(q_k, q_t.view(O, I)):
            return False

        e_back = torch.zeros(O, I, device=dev)
        ext.qpefa_dequant(q_k, scales, e_back, lam, qmax, Gs)
        e_t = (q_k.float().view(O, G, Gs) * (max_res / qmax)).view(O, I)
        if not torch.allclose(e_back, e_t, atol=1e-6):
            return False

        B = 128
        x = torch.randn(2048, device=dev, generator=g)
        q_b = torch.zeros(2048, dtype=torch.int8, device=dev)
        s_b = torch.zeros(2048 // B, device=dev)
        ext.blockwise_quant(x, q_b, s_b, B, 127.0)
        blocks = x.view(-1, B)
        s_t = blocks.abs().amax(-1).clamp(min=1e-8) / 127.0
        q_t2 = torch.round(blocks / s_t.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        if not (torch.allclose(s_b, s_t, atol=1e-7) and torch.equal(q_b, q_t2.view(-1))):
            return False

        x_back = torch.zeros(2048, device=dev)
        ext.blockwise_dequant(q_b, s_b, x_back, B)
        x_t2 = (q_b.float() * s_b.repeat_interleave(B))
        if not torch.allclose(x_back, x_t2, atol=1e-6):
            return False

        # v2: scale gradient reduction  dL/ds_g = sum_{i in g} g_i C[z_i]
        gr = torch.randn(O, I, device=dev, generator=g)
        sg_k = torch.zeros(O * G, device=dev)
        ext.duobit_scale_grad(gr, codes, levels, sg_k, Gs)
        sg_t = (gr.view(O, G, Gs) * levels[codes.long()].view(O, G, Gs)).sum(-1)
        if not torch.allclose(sg_k, sg_t.reshape(-1), atol=1e-4, rtol=1e-4):
            return False

        # v2: fused discrete update vs the reference PyTorch expansion
        for scale_rel in (0, 1):
            lam2, qmax2 = 1.0, 127.0
            d_lr, epsv, bc1, bc2, wd = 5e-3, 1e-8, 0.9, 0.999, 0.0
            m = torch.randn(O, I, device=dev, generator=g) * 1e-3
            v_row = torch.rand(O, device=dev, generator=g) * 1e-4
            v_col = torch.rand(I, device=dev, generator=g) * 1e-4
            e_q0 = torch.randint(-127, 128, (O, I), device=dev,
                                 generator=g).to(torch.int8)
            denom_mean = float(v_row.mean().clamp(min=epsv))
            c_k2, eq_k2 = codes.clone(), e_q0.clone()
            stats = torch.zeros(2, device=dev)
            ext.duobit_fused_update(c_k2, scales, levels, m, v_row, v_col, eq_k2,
                                    stats, Gs, d_lr, epsv, bc1, bc2, denom_mean,
                                    wd, lam2, qmax2, scale_rel, 1, 99)
            s3 = scales.view(O, G, 1)
            w_curr = (levels[codes.long()].view(O, G, Gs) * s3).view(O, I)
            v_hat = (torch.outer(v_row, v_col) / denom_mean) / bc2
            step = (m / bc1) / (v_hat.clamp(min=0).sqrt() + epsv)
            lr_eff = (d_lr * s3).expand(O, G, Gs).reshape(O, I) if scale_rel else d_lr
            lam_s = (lam2 * scales.clamp(min=1e-8)).view(O, G, 1)
            e_t2 = (e_q0.float().view(O, G, Gs) / qmax2) * lam_s
            e_t2 = (e_t2.view(O, I) - lr_eff * step).view(O, G, Gs)
            e_t2 = torch.minimum(torch.maximum(e_t2, -lam_s), lam_s)
            wv = w_curr + e_t2.view(O, I)
            u2 = wv.view(O, G, Gs) / (s3 + 1e-8)
            c_t2 = (u2.unsqueeze(-1) - levels.view(1, 1, 1, -1)).abs().argmin(-1)
            c_t2 = c_t2.to(torch.uint8).view(O, I)
            if not torch.equal(c_k2, c_t2):
                print("[duobit] fused-update parity: codes differ "
                      f"({int((c_k2 != c_t2).sum())}/{c_k2.numel()}, "
                      f"scale_relative={scale_rel})", flush=True)
                return False
            n_changed = float((c_t2 != codes).sum())
            if abs(float(stats[0]) - n_changed) > 0.5:
                print("[duobit] fused-update parity: transition count differs",
                      flush=True)
                return False
        return True
    except Exception as e:
        print(f"[duobit] parity check error: {e}", flush=True)
        return False


# ----------------------------------------------------------------------------
# DuoBIT quantization core (kernel-accelerated with identical fallbacks)
# ----------------------------------------------------------------------------
_LEVELS_CACHE: Dict[Tuple[int, float, torch.device], torch.Tensor] = {}


def levels_for(n_levels: int, rho: float, device: torch.device) -> torch.Tensor:
    key = (n_levels, float(rho), device)
    if key not in _LEVELS_CACHE:
        if n_levels == 2:
            vals = [-1.0, 1.0]
        elif n_levels == 3:
            vals = [-1.0, 0.0, 1.0]
        elif n_levels == 4:
            vals = [-1.0, -rho, rho, 1.0]
        else:
            raise ValueError(f"n_levels must be 2, 3, or 4, got {n_levels}")
        _LEVELS_CACHE[key] = torch.tensor(vals, device=device, dtype=torch.float32)
    return _LEVELS_CACHE[key]


def dq_weight(codes: torch.Tensor, scales: torch.Tensor, levels: torch.Tensor,
              group_size: int) -> torch.Tensor:
    """codes [O,I] uint8 + scales [O,G] -> dequantized weight [O,I] fp32."""
    O, I = codes.shape
    ext = _get_ext()
    if ext is not None and codes.is_cuda:
        out = torch.empty(O, I, device=codes.device, dtype=torch.float32)
        ext.duobit_dequant(codes.contiguous(), scales.contiguous(), levels, out, group_size)
        return out
    G = I // group_size
    c = levels[codes.long()].view(O, G, group_size)
    return (c * scales.view(O, G, 1)).view(O, I)


def q_weight(w: torch.Tensor, scales: torch.Tensor, levels: torch.Tensor,
             group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest-neighbour quantization -> (codes, w_q, residual)."""
    O, I = w.shape
    ext = _get_ext()
    if ext is not None and w.is_cuda:
        codes = torch.empty(O, I, device=w.device, dtype=torch.uint8)
        wq = torch.empty(O, I, device=w.device, dtype=torch.float32)
        res = torch.empty(O, I, device=w.device, dtype=torch.float32)
        ext.duobit_quant(w.contiguous(), scales.contiguous(), levels, codes, wq, res,
                         group_size)
        return codes, wq, res
    G = I // group_size
    u = w.view(O, G, group_size) / (scales.view(O, G, 1) + 1e-8)
    distm = (u.unsqueeze(-1) - levels.view(1, 1, 1, -1)).abs()
    codes = distm.argmin(-1).to(torch.uint8).view(O, I)
    wq = (levels[codes.long()].view(O, G, group_size) * scales.view(O, G, 1)).view(O, I)
    return codes, wq, w - wq


def q_residual(e: torch.Tensor, scales: torch.Tensor, lam: float, bits: int,
               gen: Optional[torch.Generator] = None,
               noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Stochastic-round the QPEFA residual onto a signed int grid -> int8 [O,I]."""
    O, I = e.shape
    G = scales.shape[1]
    qmax = float((1 << (bits - 1)) - 1)
    ext = _get_ext()
    if ext is not None and e.is_cuda:
        if noise is None:
            noise = torch.rand(e.shape, device=e.device, generator=gen)
        q = torch.empty(O, I, device=e.device, dtype=torch.int8)
        ext.qpefa_round(e.contiguous(), scales.contiguous(), noise.contiguous(), q,
                        float(lam), qmax, I // G)
        return q
    max_res = (lam * scales.clamp(min=1e-8)).view(O, G, 1)
    norm = (e.view(O, G, I // G) / max_res).clamp(-1, 1) * qmax
    if noise is None:
        noise = torch.rand(norm.shape, device=e.device, generator=gen)
    q = torch.floor(norm + noise).clamp(-qmax, qmax).to(torch.int8)
    return q.view(O, I)


def dq_residual(q: torch.Tensor, scales: torch.Tensor, lam: float,
                bits: int) -> torch.Tensor:
    O, I = q.shape
    G = scales.shape[1]
    qmax = float((1 << (bits - 1)) - 1)
    ext = _get_ext()
    if ext is not None and q.is_cuda:
        e = torch.empty(O, I, device=q.device, dtype=torch.float32)
        ext.qpefa_dequant(q.contiguous(), scales.contiguous(), e, float(lam), qmax,
                          I // G)
        return e
    max_res = (lam * scales.clamp(min=1e-8)).view(O, G, 1)
    return (q.float().view(O, G, I // G) * (max_res / qmax)).view(O, I)


def q_blockwise(x_flat: torch.Tensor, bits: int, block: int
                ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Block-wise int8 quantization of a flat fp32 tensor -> (q, scales, orig_n)."""
    n = x_flat.numel()
    pad = (block - n % block) % block
    xp = F.pad(x_flat, (0, pad)) if pad else x_flat
    xp = xp.contiguous()
    nb = xp.numel() // block
    qmax = float((1 << (bits - 1)) - 1)
    ext = _get_ext()
    if ext is not None and xp.is_cuda:
        q = torch.empty(xp.numel(), device=xp.device, dtype=torch.int8)
        s = torch.empty(nb, device=xp.device, dtype=torch.float32)
        ext.blockwise_quant(xp, q, s, block, qmax)
        return q, s, n
    blocks = xp.view(nb, block)
    s = blocks.abs().amax(-1).clamp(min=1e-8) / qmax
    q = torch.round(blocks / s.unsqueeze(-1)).clamp(-qmax, qmax).to(torch.int8).view(-1)
    return q, s, n


def dq_blockwise(q: torch.Tensor, s: torch.Tensor, block: int, orig_n: int
                 ) -> torch.Tensor:
    ext = _get_ext()
    if ext is not None and q.is_cuda:
        x = torch.empty(q.numel(), device=q.device, dtype=torch.float32)
        ext.blockwise_dequant(q.contiguous(), s.contiguous(), x, block)
        return x[:orig_n]
    return (q.float() * s.repeat_interleave(block))[:orig_n]


def scale_grad(grad: torch.Tensor, codes: torch.Tensor, levels: torch.Tensor,
               group_size: int) -> torch.Tensor:
    """Exact gradient of the loss w.r.t. each group scale.

    With w_i = s_g * C[z_i] and z_i held fixed, dL/ds_g = sum_{i in g} g_i C[z_i].
    Returns [O, G].
    """
    O, I = grad.shape
    G = I // group_size
    ext = _get_ext()
    if ext is not None and grad.is_cuda:
        out = torch.empty(O * G, device=grad.device, dtype=torch.float32)
        ext.duobit_scale_grad(grad.contiguous(), codes.contiguous(), levels, out,
                              group_size)
        return out.view(O, G)
    c = levels[codes.long()].view(O, G, group_size)
    return (grad.view(O, G, group_size) * c).sum(-1)


def fused_discrete_update(codes: torch.Tensor, scales: torch.Tensor,
                          levels: torch.Tensor, m: torch.Tensor,
                          v_row: torch.Tensor, v_col: torch.Tensor,
                          e_q: torch.Tensor, group_size: int, d_lr: float,
                          eps: float, bc1: float, bc2: float, denom_mean: float,
                          wd: float, lam: float, qmax: float,
                          scale_relative: bool, stats: Optional[torch.Tensor],
                          seed: int) -> None:
    """One QPEFA + transition step, in place on `codes` and `e_q`.

    Fused into a single CUDA kernel when available; the PyTorch path below is
    the functional reference (it is what the parity check compares against).
    `stats`, when given, accumulates [n_transitions, sum |residual|] on device
    so no host synchronisation is needed on non-logging steps.
    """
    O, I = codes.shape
    G = I // group_size
    ext = _get_ext()
    if ext is not None and codes.is_cuda:
        ext.duobit_fused_update(
            codes, scales.contiguous(), levels, m.contiguous(),
            v_row.contiguous(), v_col.contiguous(), e_q,
            stats if stats is not None else torch.zeros(2, device=codes.device),
            group_size, float(d_lr), float(eps), float(bc1), float(bc2),
            float(denom_mean), float(wd), float(lam), float(qmax),
            1 if scale_relative else 0, 1 if stats is not None else 0, int(seed))
        return

    s3 = scales.view(O, G, 1)
    w_curr = (levels[codes.long()].view(O, G, group_size) * s3).view(O, I)
    v_hat = (torch.outer(v_row, v_col) / denom_mean) / bc2
    step = (m / bc1) / (v_hat.clamp(min=0).sqrt() + eps)
    if wd != 0.0:
        step = step + wd * w_curr
    lr_eff = (d_lr * s3).expand(O, G, group_size).reshape(O, I) if scale_relative \
        else d_lr
    lam_s = (lam * scales.clamp(min=1e-8)).view(O, G, 1)
    e = (e_q.float().view(O, G, group_size) / qmax) * lam_s
    e = (e.view(O, I) - lr_eff * step).view(O, G, group_size)
    e = torch.minimum(torch.maximum(e, -lam_s), lam_s)
    w_virtual = w_curr + e.view(O, I)
    new_codes, _, e_true = q_weight(w_virtual, scales, levels, group_size)
    if stats is not None:
        stats[0] += (new_codes != codes).sum().float()
        stats[1] += e_true.abs().sum()
    # deterministic, rank-independent stochastic rounding (matches the kernel's
    # splitmix64 hash only in distribution, not bit-for-bit; CPU path only)
    gen = torch.Generator(device=e_true.device)
    gen.manual_seed(int(seed) & 0x7FFFFFFF)
    noise = torch.rand(e_true.shape, device=e_true.device, generator=gen)
    norm = ((e_true.view(O, G, group_size) / lam_s).clamp(-1, 1) * qmax)
    e_q.copy_(torch.floor(norm + noise.view(O, G, group_size))
              .clamp(-qmax, qmax).to(torch.int8).view(O, I))
    codes.copy_(new_codes)


def maxabs_group_scales(w: torch.Tensor, group_size: int) -> torch.Tensor:
    O, I = w.shape
    G = I // group_size
    return w.view(O, G, group_size).abs().amax(-1).clamp(min=1e-5)


def mse_group_scales(w: torch.Tensor, codes: torch.Tensor, levels: torch.Tensor,
                     group_size: int) -> torch.Tensor:
    """Closed-form MSE-optimal scale s* = sum(w_i c_i) / (sum(c_i^2) + eps)."""
    O, I = w.shape
    G = I // group_size
    c = levels[codes.long()].view(O, G, group_size)
    num = (w.view(O, G, group_size) * c).sum(-1)
    den = (c * c).sum(-1) + 1e-8
    return (num / den).clamp(min=1e-5)


def var_group_scales(w: torch.Tensor, codes: torch.Tensor, levels: torch.Tensor,
                     group_size: int) -> torch.Tensor:
    """Variance-preserving group scale.

    The MSE-optimal scale minimises ||w - s C[z]||, which for a random
    initialisation systematically *shrinks* the weight: on kaiming_uniform,
    std(s* C[z]) is 0.956 * std(w) at every fan-in. Composed over 2L linear maps
    that compounds (0.49x the activation scale at L=8, 0.24x at L=16), which
    weakens every residual branch relative to its skip path and shrinks the
    logits before a single step is taken. Re-fitting the scale so each group
    reproduces the intended standard deviation removes that bias, and for a
    random init preserving the forward variance matters more than matching the
    particular draw.
    """
    O, I = w.shape
    G = I // group_size
    s = mse_group_scales(w, codes, levels, group_size)
    c = levels[codes.long()].view(O, G, group_size)
    std_w = w.view(O, G, group_size).std(-1)
    std_q = (c * s.view(O, G, 1)).std(-1)
    ratio = (std_w / std_q.clamp(min=1e-12)).clamp(0.5, 2.0)
    return (s * ratio).clamp(min=1e-5)


# ----------------------------------------------------------------------------
# DuoBIT layer: persistent state = codes + scales only (no master weight)
# ----------------------------------------------------------------------------
class DuobitLinear(nn.Module):
    """Linear layer whose weights exist only as discrete codes + group scales.

    The dequantized matrix is ephemeral: materialized for forward/backward,
    then discarded. During training the ephemeral matrix receives a gradient
    which the DuobitAdam optimizer converts into code transitions via QPEFA.
    """

    def __init__(self, in_features: int, out_features: int, n_levels: int = 4,
                 rho: float = 1.0 / 3.0, group_size: int = 128,
                 init_std: float = 0.0, scale_init: str = "var"):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_levels = n_levels
        self.rho = float(rho)
        self.init_std = float(init_std)
        self.scale_init = str(scale_init)
        n_groups = in_features // group_size
        self.register_buffer("codes",
                             torch.zeros((out_features, in_features), dtype=torch.uint8))
        self.register_buffer("scales", torch.ones((out_features, n_groups),
                                                  dtype=torch.float32))
        # Init scale is kept so a trained scale can be floored relative to it
        # (a group whose scale collapses to 0 can never recover).
        self.register_buffer("scales_init", torch.ones((out_features, n_groups),
                                                       dtype=torch.float32))
        self.ephemeral_w: Optional[torch.Tensor] = None
        self.reset_parameters()

    @property
    def levels(self) -> torch.Tensor:
        return levels_for(self.n_levels, self.rho, self.codes.device)

    def reset_parameters(self):
        init_w = torch.empty((self.out_features, self.in_features))
        if self.init_std > 0:
            init_w.normal_(0.0, self.init_std)
        else:
            nn.init.kaiming_uniform_(init_w, a=math.sqrt(5))
        s = maxabs_group_scales(init_w, self.group_size)
        codes, _, _ = q_weight(init_w, s, self.levels, self.group_size)
        fit = var_group_scales if self.scale_init == "var" else mse_group_scales
        s_fit = fit(init_w, codes, self.levels, self.group_size)
        self.codes.copy_(codes)
        self.scales.copy_(s_fit)
        self.scales_init.copy_(s_fit)

    def dequant(self) -> torch.Tensor:
        return dq_weight(self.codes, self.scales, self.levels, self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequant()
        if self.training:
            w.requires_grad_(True)
            self.ephemeral_w = w
        return F.linear(x, w, None)


# ----------------------------------------------------------------------------
# Model (GPT-style decoder: RMSNorm, RoPE, causal SDPA, SwiGLU)
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def make_linear(args, use_duobit: bool, in_f: int, out_f: int,
                std_scale: float = 1.0):
    """Build one linear map in the regime under test.

    `std_scale` implements the GPT-2 residual-projection rule (1/sqrt(2L) on the
    layers that write back into the residual stream); it only applies when
    --init-std is set, otherwise both regimes keep PyTorch's kaiming_uniform.
    """
    base = float(getattr(args, "init_std", 0.0) or 0.0)
    std = base * std_scale
    if use_duobit:
        return DuobitLinear(in_f, out_f, n_levels=args.n_levels, rho=args.rho,
                            group_size=args.group_size, init_std=std,
                            scale_init=getattr(args, "scale_init", "var"))
    lin = nn.Linear(in_f, out_f, bias=False)
    if std > 0:
        nn.init.normal_(lin.weight, mean=0.0, std=std)
    return lin


class Attention(nn.Module):
    def __init__(self, args, use_duobit: bool):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.head_dim = args.d_model // args.n_heads
        res = 1.0 / math.sqrt(2.0 * max(1, args.n_layers))
        mk = lambda i, o, sc=1.0: make_linear(args, use_duobit, i, o, sc)
        self.q_proj = mk(args.d_model, args.d_model)
        self.k_proj = mk(args.d_model, args.d_model)
        self.v_proj = mk(args.d_model, args.d_model)
        self.out_proj = mk(args.d_model, args.d_model, res)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=args.seq_len + 64)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(s)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, args, use_duobit: bool):
        super().__init__()
        res = 1.0 / math.sqrt(2.0 * max(1, args.n_layers))
        mk = lambda i, o, sc=1.0: make_linear(args, use_duobit, i, o, sc)
        self.gate_proj = mk(args.d_model, args.d_ff)
        self.up_proj = mk(args.d_model, args.d_ff)
        self.down_proj = mk(args.d_ff, args.d_model, res)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, args, use_duobit: bool):
        super().__init__()
        self.norm1 = RMSNorm(args.d_model)
        self.attn = Attention(args, use_duobit)
        self.norm2 = RMSNorm(args.d_model)
        self.ffn = SwiGLUFFN(args, use_duobit)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    """GPT-style decoder with DuobitLinear or dense nn.Linear layers."""

    def __init__(self, args, use_duobit: bool = True):
        super().__init__()
        self.args = args
        self.vocab_size = 50257
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.d_model)
        if float(getattr(args, "init_std", 0.0) or 0.0) > 0:
            nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=args.init_std)
        self.layers = nn.ModuleList([Block(args, use_duobit) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.d_model)
        self.lm_head = make_linear(args, use_duobit, args.d_model, self.vocab_size)
        self.use_duobit = use_duobit

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
        return logits, loss

    def count_parameters(self) -> int:
        total = 0
        for m in self.modules():
            if isinstance(m, DuobitLinear):
                total += m.codes.numel()
            elif isinstance(m, nn.Linear):
                total += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            elif isinstance(m, nn.Embedding):
                total += m.weight.numel()
            elif isinstance(m, RMSNorm):
                total += m.weight.numel()
        return total


# ----------------------------------------------------------------------------
# DuobitAdam: AdamW for standard params + DuoBIT discrete-layer stepper
# ----------------------------------------------------------------------------
class DuobitAdam:
    """DUOBIT-EST optimizer (v2).

    Standard parameters (embeddings, RMSNorm gains) take an ordinary AdamW
    step. Each DuobitLinear layer takes an ephemeral Adam step on its
    dequantized matrix, accumulates the sub-threshold remainder in QPEFA, and
    re-quantizes codes. Persistent state per discrete layer: int8 QPEFA
    residual, block-wise int8 first moment, factored second moment.

    v2 adds a trained group scale. With w_i = s_g C[z_i] and the codes held
    fixed, dL/ds_g = sum_{i in g} g_i C[z_i] is exact, so the scales take their
    own Adam step. The scales are part of the persistent 2-bit representation
    (they are kept at inference and already charged at 32/G bits per weight),
    so this does not reintroduce a master-weight tensor; it adds 2*32/G bits
    per weight of optimizer state (0.5 bits/weight at G=128).

    Determinism: stochastic rounding is a hash of (element index, step seed),
    identical on every DDP rank, so codes stay in sync given the all-reduced
    gradient. `codes_sync_check` verifies this during the run.
    """

    def __init__(self, model: nn.Module, args, device, seed: int):
        self.model = model
        self.args = args
        self.device = device
        self.seed = int(seed)
        self.duobit_layers = [m for m in model.modules() if isinstance(m, DuobitLinear)]
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.lr = args.lr
        self.duobit_lr = args.duobit_lr
        self.scale_lr = float(getattr(args, "scale_lr", 1e-3))
        self.b1, self.b2 = 0.9, 0.999
        self.eps = 1e-8
        self.wd = args.wd
        self.duobit_wd = float(getattr(args, "duobit_wd", 0.0))
        self.learn_scales = bool(int(getattr(args, "learn_scales", 0)))
        self.scale_lr_abs = bool(int(getattr(args, "scale_lr_abs", 0)))
        self.scale_min_frac = float(getattr(args, "scale_min_frac", 0.05))
        self.scale_relative = bool(int(getattr(args, "scale_relative_lr", 0)))
        self.dstate: Dict[int, Dict[str, Any]] = {}
        self.pstate: Dict[int, Dict[str, Any]] = {}
        self.step_count = 0
        self.collect_stats = True
        self.last_transition_frac = 0.0
        self.last_residual_abs_mean = 0.0
        self.last_scale_mean = 0.0
        self.last_scale_ratio = 1.0
        self._stats = torch.zeros(2, device=device, dtype=torch.float32)
        self._stats_weights = 0

    # -- public API ---------------------------------------------------------
    def set_lrs(self, lr: float, duobit_lr: float, scale_lr: Optional[float] = None):
        self.lr = lr
        self.duobit_lr = duobit_lr
        if scale_lr is not None:
            self.scale_lr = scale_lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def clip_ephemeral_grads(self, max_norm: float = 1.0) -> torch.Tensor:
        """Clip the ephemeral gradients and return the norm as a device tensor.

        Returning a tensor (rather than a float) keeps the training loop free of
        per-step host synchronisation.
        """
        ts = [l.ephemeral_w.grad for l in self.duobit_layers
              if l.ephemeral_w is not None and l.ephemeral_w.grad is not None]
        if not ts:
            return torch.zeros((), device=self.device)
        return torch.nn.utils.clip_grad_norm_(ts, max_norm)

    def step(self):
        self.step_count += 1
        self._step_standard_params()
        self._step_duobit_layers()

    def read_stats(self) -> Tuple[float, float]:
        """Drain the device-side counters (one sync; call only when logging)."""
        if self._stats_weights == 0:
            return self.last_transition_frac, self.last_residual_abs_mean
        st = self._stats.tolist()
        self.last_transition_frac = st[0] / self._stats_weights
        self.last_residual_abs_mean = st[1] / self._stats_weights
        self._stats.zero_()
        self._stats_weights = 0
        return self.last_transition_frac, self.last_residual_abs_mean

    def read_scale_stats(self) -> Tuple[float, float]:
        """Mean group scale and mean scale/init ratio (one sync)."""
        if not self.duobit_layers:
            return 0.0, 1.0
        means = torch.stack([l.scales.mean() for l in self.duobit_layers])
        ratios = torch.stack([(l.scales / l.scales_init.clamp(min=1e-12)).mean()
                              for l in self.duobit_layers])
        self.last_scale_mean = float(means.mean())
        self.last_scale_ratio = float(ratios.mean())
        return self.last_scale_mean, self.last_scale_ratio

    # -- internals ----------------------------------------------------------
    def _step_standard_params(self):
        lr, b1, b2, eps, wd = self.lr, self.b1, self.b2, self.eps, self.wd
        for p in self.params:
            if p.grad is None:
                continue
            st = self.pstate.get(id(p))
            if st is None:
                st = {"m": torch.zeros_like(p.data), "v": torch.zeros_like(p.data), "t": 0}
                self.pstate[id(p)] = st
            g = p.grad
            st["t"] += 1
            t = st["t"]
            if wd != 0:
                p.data.mul_(1.0 - lr * wd)
            st["m"].mul_(b1).add_(g, alpha=1.0 - b1)
            st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
            denom = (st["v"].sqrt() / math.sqrt(bc2)).add_(eps)
            p.data.addcdiv_(st["m"], denom, value=-(lr / bc1))

    def _init_layer_state(self, lyr: DuobitLinear) -> Dict[str, Any]:
        a = self.args
        O, I = lyr.codes.shape
        n = lyr.codes.numel()
        dev = lyr.codes.device
        nb = (n + a.block_size - 1) // a.block_size
        st = {
            "t": 0,
            "e_q": torch.zeros(O, I, dtype=torch.int8, device=dev),
            "m_q": torch.zeros(nb * a.block_size, dtype=torch.int8, device=dev),
            "m_s": torch.zeros(nb, dtype=torch.float32, device=dev),
            "m_n": n,
            "v_row": torch.zeros(O, dtype=torch.float32, device=dev),
            "v_col": torch.zeros(I, dtype=torch.float32, device=dev),
        }
        if self.learn_scales:
            st["s_m"] = torch.zeros_like(lyr.scales)
            st["s_v"] = torch.zeros_like(lyr.scales)
        return st

    def _step_scales(self, lyr: DuobitLinear, st: Dict[str, Any],
                     grad: torch.Tensor, t: int) -> None:
        """Adam on the per-group scales from the exact analytic gradient."""
        b1, b2, eps = self.b1, self.b2, self.eps
        gs = scale_grad(grad, lyr.codes, lyr.levels, lyr.group_size)
        st["s_m"].mul_(b1).add_(gs, alpha=1.0 - b1)
        st["s_v"].mul_(b2).addcmul_(gs, gs, value=1.0 - b2)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        direction = (st["s_m"] / bc1) / ((st["s_v"] / bc2).sqrt() + eps)
        if self.scale_lr_abs:
            lyr.scales.add_(direction, alpha=-self.scale_lr)
        else:
            # scale-relative: ds = -lr * s * dir, so a group's magnitude moves
            # by a fixed fraction per step regardless of the layer's fan-in
            lyr.scales.addcmul_(lyr.scales, direction, value=-self.scale_lr)
        torch.maximum(lyr.scales, lyr.scales_init * self.scale_min_frac,
                      out=lyr.scales)

    def _step_duobit_layers(self):
        a = self.args
        qmax = float((1 << (a.qpefa_bits - 1)) - 1)
        collect = self.collect_stats
        for li, lyr in enumerate(self.duobit_layers):
            if lyr.ephemeral_w is None or lyr.ephemeral_w.grad is None:
                continue
            grad = lyr.ephemeral_w.grad
            st = self.dstate.get(id(lyr))
            if st is None:
                st = self._init_layer_state(lyr)
                self.dstate[id(lyr)] = st
            st["t"] += 1
            t = st["t"]
            O, I = lyr.codes.shape
            b1, b2, eps = self.b1, self.b2, self.eps

            # 1) trained group scales (uses the pre-update codes)
            if self.learn_scales:
                self._step_scales(lyr, st, grad, t)

            # 2) compressed Adam moments over the ephemeral matrix
            m = dq_blockwise(st["m_q"], st["m_s"], a.block_size, st["m_n"]).view(O, I)
            m.mul_(b1).add_(grad, alpha=1.0 - b1)
            g2 = grad * grad
            st["v_row"].mul_(b2).add_(g2.mean(dim=1), alpha=1.0 - b2)
            st["v_col"].mul_(b2).add_(g2.mean(dim=0), alpha=1.0 - b2)
            del g2
            # Normalise the row factor on device instead of reading the mean back
            # to the host: v_hat = outer(row, col) / mean(row) is unchanged, and
            # the step stays free of host synchronisation.
            row = st["v_row"].clamp(min=0)
            row = row / row.mean().clamp(min=eps)

            # 3) QPEFA + code transitions, fused into a single pass
            fused_discrete_update(
                lyr.codes, lyr.scales, lyr.levels, m, row,
                st["v_col"].clamp(min=0), st["e_q"], lyr.group_size,
                d_lr=self.duobit_lr, eps=eps, bc1=1.0 - b1 ** t, bc2=1.0 - b2 ** t,
                denom_mean=1.0, wd=self.duobit_wd, lam=a.pefa_clip,
                qmax=qmax, scale_relative=self.scale_relative,
                stats=self._stats if collect else None,
                # distinct per (run seed, step, layer) with no aliasing: layer index
                # is well under the 1024 stride
                seed=self.seed * 1000003 + self.step_count * 1024 + li)
            if collect:
                self._stats_weights += lyr.codes.numel()

            # 4) legacy MSE-EMA rescale (a fixed point on dequantized weights;
            #    kept behind a flag for the v1-reproduction ablation)
            if a.scale_update_freq > 0 and t % a.scale_update_freq == 0 and a.scale_ema > 0:
                w_post = lyr.dequant()
                s_target = mse_group_scales(w_post, lyr.codes, lyr.levels, lyr.group_size)
                lyr.scales.mul_(1.0 - a.scale_ema).add_(s_target, alpha=a.scale_ema)

            st["m_q"], st["m_s"], st["m_n"] = q_blockwise(m.reshape(-1), a.moment_bits,
                                                          a.block_size)
            lyr.ephemeral_w = None


# ----------------------------------------------------------------------------
# Persistent-state memory accounting (packed analytic + as-stored)
# ----------------------------------------------------------------------------
def persistent_state_report(model: nn.Module, args, mode: str,
                            optimizer=None) -> Dict[str, float]:
    """Persistent training-state and inference memory of a built model.

    Accounted per regime, always against the same FP32 + Adam reference:

      duobit : codes at log2(L) bits + FP32 group scales + int8 QPEFA residual
               + block-wise int8 first moment (+ its FP32 block scales) +
               factored second moment (O+I floats) + , when --learn-scales is
               on, two FP32 Adam moments per group.
      fp32   : FP32 weight + FP32 m + FP32 v  = 96 bits/weight.
      fp16   : mixed precision keeps FP32 master weights and FP32 Adam moments,
               so training state is also 96 bits/weight; only the inference
               export halves (16 bits/weight).

    Embeddings and RMSNorm gains stay FP32 in every regime and are added to
    both sides, so the reported totals are whole-model, not linear-only.
    """
    is_duobit = (mode == "duobit")
    learn_scales = bool(int(getattr(args, "learn_scales", 0))) and is_duobit
    n_lin = 0
    lin_train = 0          # persistent training state of the linear maps, bytes
    lin_infer = 0          # packed inference footprint of the linear maps, bytes
    as_stored = 0.0        # actual torch storage bytes (unpacked uint8 codes etc.)
    fp32_lin_train = 0
    fp32_lin_infer = 0
    infer_w_bits = 32.0 if mode == "fp32" else (16.0 if mode == "fp16" else 0.0)

    for m in model.modules():
        if isinstance(m, DuobitLinear):
            n = m.codes.numel()
            O, I = m.codes.shape
            n_groups = I // m.group_size
            n_lin += n
            code_bytes = int(math.ceil(math.log2(m.n_levels) * n / 8.0))
            scale_bytes = 4 * n_groups * O
            opt_bytes = (int(math.ceil(n * args.qpefa_bits / 8.0))          # QPEFA
                         + int(math.ceil(n * args.moment_bits / 8.0))       # int8 m
                         + 4 * ((n + args.block_size - 1) // args.block_size)
                         + 4 * (O + I))                                     # factored v
            if learn_scales:
                # scale Adam m, v and the FP32 init-scale reference that floors
                # them: 3*32/G bits per weight, all dropped at inference
                opt_bytes += 12 * n_groups * O
            lin_train += code_bytes + scale_bytes + opt_bytes
            lin_infer += code_bytes + scale_bytes
            as_stored += n + scale_bytes + (12 * n_groups * O if learn_scales else 0)
            fp32_lin_train += 12 * n
            fp32_lin_infer += 4 * n
        elif isinstance(m, nn.Linear):
            n = m.weight.numel()
            n_lin += n
            lin_train += 12 * n
            lin_infer += int(n * infer_w_bits / 8.0)
            fp32_lin_train += 12 * n
            fp32_lin_infer += 4 * n

    other_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    other_train = 12 * other_params   # FP32 W + m + v for embeddings / norms
    other_infer = 4 * other_params

    if is_duobit and optimizer is not None and getattr(optimizer, "dstate", None):
        for st in optimizer.dstate.values():
            as_stored += st["e_q"].numel()
            as_stored += st["m_q"].numel()
            as_stored += st["m_s"].numel() * 4
            as_stored += (st["v_row"].numel() + st["v_col"].numel()) * 4
            if "s_m" in st:
                as_stored += (st["s_m"].numel() + st["s_v"].numel()) * 4

    total_train = lin_train + other_train
    total_infer = lin_infer + other_infer
    fp32_train = fp32_lin_train + other_train
    fp32_infer = fp32_lin_infer + other_infer
    return {
        "n_params_total": float(n_lin + other_params),
        "n_linear_weights": float(n_lin),
        "n_fp32_params": float(other_params),
        "linear_train_bits_per_weight": (8.0 * lin_train / n_lin) if n_lin else 0.0,
        "linear_infer_bits_per_weight": (8.0 * lin_infer / n_lin) if n_lin else 0.0,
        "persistent_train_mb": total_train / 1048576.0,
        "persistent_train_mb_as_stored": ((as_stored + other_train) / 1048576.0
                                          if is_duobit else total_train / 1048576.0),
        "persistent_train_mb_fp32_ref": fp32_train / 1048576.0,
        "persistent_train_compression_vs_fp32": (fp32_train / total_train)
                                                 if total_train else 0.0,
        "inference_mb": total_infer / 1048576.0,
        "inference_mb_fp32_ref": fp32_infer / 1048576.0,
        "inference_compression_vs_fp32": (fp32_infer / total_infer)
                                          if total_infer else 0.0,
    }


# ----------------------------------------------------------------------------
# Dependencies / tokenizer / data
# ----------------------------------------------------------------------------
def _rss_mb() -> float:
    """Resident set size of this process in MB (for OOM diagnostics)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def _malloc_trim():
    """Return freed heap pages to the OS (glibc only)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_TOK = None


def _ensure_deps():
    needed = []
    for mod, pkg in (("datasets", "datasets"), ("transformers", "transformers"),
                     ("matplotlib", "matplotlib")):
        try:
            importlib.import_module(mod)
        except Exception:
            needed.append(pkg)
    if needed:
        print(f"[deps] installing missing packages: {needed}", flush=True)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])
        except Exception as e:
            print(f"[deps] pip install failed ({e}); continuing", flush=True)
    # datasets must be new enough for parquet-native hub datasets (fineweb-edu)
    try:
        import datasets
        from packaging.version import Version
        if Version(datasets.__version__) < Version("2.16"):
            print(f"[deps] upgrading datasets {datasets.__version__} -> latest", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                                   "datasets"])
    except Exception:
        pass


def get_tokenizer(args):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(args.tokenizer)
    return _TOK


_HUB_CACHE: Dict[str, Tuple[str, List[str]]] = {}


def _hub_parquet_files(repo_id: str, config: str) -> Tuple[str, List[str]]:
    """Resolve (pinned revision, sorted parquet file list) for a hub config.

    The revision is pinned once per process so both experiment runs read the
    exact same files even if the repository advances mid-session.
    """
    key = f"{repo_id}|{config}"
    if key not in _HUB_CACHE:
        from huggingface_hub import HfApi
        api = HfApi()
        sha = api.dataset_info(repo_id).sha
        files = api.list_repo_files(repo_id, repo_type="dataset", revision=sha)
        prefix = "data/" if config == "default" else config.replace("-", "/") + "/"
        parquets = sorted(f for f in files
                          if f.endswith(".parquet") and f.startswith(prefix))
        if not parquets:
            raise RuntimeError(
                f"no parquet files found under '{prefix}' of {repo_id} "
                f"(config '{config}')")
        _HUB_CACHE[key] = (sha, parquets)
    return _HUB_CACHE[key]


class DocStream:
    """Deterministic, per-rank, bounded-memory document stream over a HF
    parquet dataset (or a local .jsonl file).

    Why not `datasets` streaming iteration: on the Kaggle image its
    fsspec/pyarrow buffers retain roughly 10 MB of RSS per document read,
    which OOM-kills 2 DDP workers within minutes. This reader instead:

      * resolves the dataset revision + parquet file list once (pinned),
      * assigns files to ranks round-robin (deterministic, disjoint),
      * downloads each parquet file ON DEMAND to the local HF cache on disk
        (bounded RAM; the second run reuses the cache),
      * reads row-groups with pyarrow (one row-group in RAM at a time),
      * mixes documents with a seeded in-memory shuffle buffer.

    Both experiment runs therefore consume an identical, deterministic token
    stream while RSS stays bounded by ~one row-group + the shuffle buffer.
    The last `world` files of the dataset are held out for validation.
    """

    def __init__(self, args, rank: int, world: int, seed: int, for_val: bool = False):
        self.args = args
        self.rank = rank
        self.world = world
        self.for_val = for_val
        self.buffer_docs = max(16, args.shuffle_buffer)
        self.rng = random.Random(seed * 100003 + rank + (555 if for_val else 0))
        self.is_local = str(args.dataset).endswith(".jsonl")
        self.docs_served = 0
        if self.is_local:
            self.repo_id = None
            self.revision = None
            self.files = [str(args.dataset)]
        else:
            self.repo_id = str(args.dataset)
            self.revision, all_files = _hub_parquet_files(self.repo_id,
                                                          args.dataset_config)
            n = len(all_files)
            if for_val:
                self.files = [all_files[max(0, n - 1 - rank)]]
            else:
                self.files = [f for i, f in enumerate(all_files)
                              if i % world == rank and i < n - world]
            if not self.files:
                raise RuntimeError("empty file assignment for this rank")

    def _iter_parquet_docs(self, fname: str):
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        local = hf_hub_download(repo_id=self.repo_id, filename=fname,
                                repo_type="dataset", revision=self.revision)
        pf = pq.ParquetFile(local)
        names = pf.schema_arrow.names
        col = "text" if "text" in names else names[0]
        for batch in pf.iter_batches(batch_size=512, columns=[col]):
            for t in batch.column(0).to_pylist():
                if t and len(t) > 0:
                    yield t

    def _source_iter(self):
        """Raw per-rank document source (deterministic order)."""
        if self.is_local:
            ri = 0
            for fname in self.files:
                with open(fname, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        keep = (ri % self.world == self.rank)
                        ri += 1
                        if not keep:
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        t = d.get("text") if isinstance(d, dict) else None
                        if t and len(t) > 0:
                            yield t
            return
        for fname in self.files:
            yield from self._iter_parquet_docs(fname)

    def __iter__(self):
        buf: List[str] = []
        rng = self.rng
        for t in self._source_iter():
            if len(buf) < self.buffer_docs:
                buf.append(t)
                continue
            i = rng.randrange(len(buf))
            yield buf[i]
            buf[i] = t
            self.docs_served += 1
        rng.shuffle(buf)
        for t in buf:
            yield t
            self.docs_served += 1


class StreamingTokenLoader:
    """FineWeb-EDU docs (DocStream) -> GPT-2 tokens -> packed (B, T) batches.

    Deterministic given (seed, rank, world_size), so both experiment runs
    consume the identical token stream. A background thread prefetches batches.
    """

    def __init__(self, args, tokenizer, rank: int, world_size: int, seed: int):
        self.args = args
        self.rank = rank
        self.world = world_size
        self.B = args.batch_size
        self.T = args.seq_len
        self.eos = tokenizer.eos_token_id
        self._docs = iter(DocStream(args, rank, world_size, seed))
        self._tok = tokenizer
        self._q: "queue.Queue[torch.Tensor]" = queue.Queue(maxsize=max(1, args.prefetch))
        self._buf: List[torch.Tensor] = []
        self._buf_len = 0
        self._exc: Optional[BaseException] = None
        self._stop = False
        self.docs_seen = 0
        self.tokens_seen = 0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            while not self._stop:
                texts = []
                for _ in range(64):
                    if self._stop:
                        return
                    try:
                        t = next(self._docs)
                    except StopIteration:
                        self._put(None)
                        return
                    texts.append(t)
                if not texts:
                    continue
                enc = self._tok(texts, add_special_tokens=False)["input_ids"]
                for toks in enc:
                    self._buf.append(torch.tensor(toks + [self.eos], dtype=torch.long))
                    self._buf_len += len(toks) + 1
                self.docs_seen += len(texts)
                while self._buf_len >= self.B * self.T:
                    if not self._emit():
                        return
        except BaseException as e:
            self._exc = e
            self._put(None)

    def _emit(self) -> bool:
        need = self.B * self.T
        buf = torch.cat(self._buf)
        batch, rest = buf[:need].view(self.B, self.T), buf[need:]
        self._buf = [rest] if rest.numel() else []
        self._buf_len = rest.numel()
        self.tokens_seen += need
        return self._put(batch)

    def _put(self, item) -> bool:
        while True:
            try:
                self._q.put(item, timeout=0.5)
                return True
            except queue.Full:
                if self._stop:
                    return False

    def next_batch(self) -> torch.Tensor:
        b = self._q.get()
        if b is None:
            if self._exc is not None:
                raise RuntimeError("streaming data worker failed") from self._exc
            raise RuntimeError("data stream exhausted before steps completed")
        return b

    def stop(self):
        self._stop = True
        t = getattr(self, "_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=30)


def build_val_tokens(args, tokenizer, rank: int, world_size: int) -> torch.Tensor:
    """Deterministic held-out validation tokens, per rank, cached to disk.

    Uses the dataset's held-out tail files (disjoint from the training files
    by construction) with a different shuffle seed.
    """
    cache = Path(args.out_dir) / f"val_tokens_rank{rank}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True)
    stream = DocStream(args, rank, world_size, seed=args.seed + 12345, for_val=True)
    eos = tokenizer.eos_token_id
    parts: List[torch.Tensor] = []
    n = 0
    for text in stream:
        toks = tokenizer(text, add_special_tokens=False)["input_ids"]
        parts.append(torch.tensor(toks + [eos], dtype=torch.long))
        n += len(toks) + 1
        if n >= args.val_tokens:
            break
    if not parts:
        raise RuntimeError("could not stream any validation documents")
    val = torch.cat(parts)[:args.val_tokens]
    torch.save(val, cache)
    return val


# ----------------------------------------------------------------------------
# DDP helpers
# ----------------------------------------------------------------------------
def reduce_mean_scalar(t: torch.Tensor, world: int) -> float:
    if world <= 1 or not dist.is_available() or not dist.is_initialized():
        return float(t.item())
    if t.is_cuda and dist.get_backend() == "nccl":
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return float(t.item())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / world


def codes_sync_check(model: nn.Module, world: int) -> bool:
    """Verify DuoBIT codes are bitwise identical across ranks."""
    if world <= 1 or not dist.is_initialized():
        return True
    s = 0
    dev = torch.device("cpu")
    for m in model.modules():
        if isinstance(m, DuobitLinear):
            dev = m.codes.device
            s += int(m.codes.flatten()[::9973].to(torch.int64).sum().item())
    tmin = torch.tensor([s], dtype=torch.int64, device=dev)
    tmax = torch.tensor([s], dtype=torch.int64, device=dev)
    dist.all_reduce(tmin, op=dist.ReduceOp.MIN)
    dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
    return bool(int(tmin.item()) == int(tmax.item()) == s)


def find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to a 10% floor (repo Trainer schedule)."""
    if warmup > 0 and step <= warmup:
        return float(step) / float(warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, val_tokens: torch.Tensor, args, device, world: int) -> Tuple[float, float]:
    """Mean validation loss. Evaluation always runs in FP32 so the reported
    quality of an AMP run is not measured through a different numeric path."""
    net.eval()
    B, T = args.batch_size, args.seq_len
    n_b = max(1, val_tokens.numel() // (B * T))
    total = torch.zeros((), device=device, dtype=torch.float32)
    for i in range(n_b):
        ids = val_tokens[i * B * T:(i + 1) * B * T].view(B, T)
        _, loss = net(ids, targets=ids)
        total += loss.detach().float()
    total /= n_b
    if world > 1 and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        total /= world
    avg = float(total)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    net.train()
    return avg, ppl


@torch.no_grad()
def greedy_sample(model, tokenizer, device, args,
                  prompt: str = "Once upon a time") -> str:
    model.eval()
    ids = tokenizer(prompt)["input_ids"]
    for _ in range(args.sample_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        ids.append(int(logits[0, -1].argmax().item()))
    model.train()
    return tokenizer.decode(ids)


def collect_hw_info() -> Dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpus": [],
    }
    if torch.cuda.is_available():
        info["cudnn"] = torch.backends.cudnn.version()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({"name": p.name, "cc": f"{p.major}.{p.minor}",
                                 "mem_gb": round(p.total_memory / 2 ** 30, 1)})
    try:
        import datasets
        import transformers
        info["datasets"] = datasets.__version__
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


# ----------------------------------------------------------------------------
# One experiment run
# ----------------------------------------------------------------------------
def run_experiment(name: str, mode: str, rank: int, world: int, args,
                   device) -> Dict[str, Any]:
    """Train one regime end to end and return its logs, curve and summary.

    `args` is this run's own namespace (already merged with its run-spec
    overrides), so a sweep can vary any hyper-parameter per run while the data
    stream, seed, architecture and step count stay shared.
    """
    t_start = time.time()
    out_dir = Path(args.out_dir)
    is_main = (rank == 0)
    tag = name
    use_duobit = (mode == "duobit")
    amp = bool(int(getattr(args, "amp", 0))) or mode == "fp16"
    amp_ok = amp and torch.cuda.is_available()
    if amp and not amp_ok and is_main:
        print(f"[{tag}] autocast requested but no CUDA device: running FP32",
              flush=True)
    if is_main:
        print("\n" + "=" * 78, flush=True)
        print(f"  RUN {name.upper()} (mode={mode}{', amp=fp16' if amp_ok else ''})"
              f"  |  steps={args.steps}  |  world_size={world}  |  device={device}",
              flush=True)
        print("=" * 78, flush=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build the model directly on the target device: this also runs the
    # DuobitLinear init quantization through the CUDA kernels (avoids ~1 GB
    # CPU temporaries per worker from the fallback argmin path).
    try:
        with torch.device(device):
            model = Transformer(args, use_duobit=use_duobit)
    except Exception:
        model = Transformer(args, use_duobit=use_duobit).to(device)
    n_params = model.count_parameters()

    scaler = None
    if use_duobit:
        optimizer = DuobitAdam(model, args, device, seed=args.seed)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        if amp_ok:
            scaler = torch.amp.GradScaler("cuda")

    net = model
    if world > 1:
        net = nn.parallel.DistributedDataParallel(
            model, device_ids=([rank] if torch.cuda.is_available() else None),
            broadcast_buffers=False)

    tokenizer = get_tokenizer(args)
    if is_main:
        print(f"[{tag}] building validation set "
              f"({args.val_tokens:,} tokens/rank, cached across runs)...", flush=True)
    val_tokens = build_val_tokens(args, tokenizer, rank, world).to(device)
    loader = StreamingTokenLoader(args, tokenizer, rank, world, seed=args.seed)

    persist = persistent_state_report(model, args, mode, optimizer)
    if is_main:
        print(f"[{tag}] params={n_params:,} | linear weights="
              f"{int(persist['n_linear_weights']):,} | fp32 params="
              f"{int(persist['n_fp32_params']):,}", flush=True)
        print(f"[{tag}] persistent linear train bits/wt = "
              f"{persist['linear_train_bits_per_weight']:.2f} "
              f"(FP32 Adam = 96.0) | train state = "
              f"{persist['persistent_train_mb']:.1f} MB vs "
              f"{persist['persistent_train_mb_fp32_ref']:.1f} MB FP32", flush=True)
        print(f"[{tag}] inference = {persist['inference_mb']:.1f} MB vs "
              f"{persist['inference_mb_fp32_ref']:.1f} MB FP32 "
              f"({persist['linear_infer_bits_per_weight']:.2f} bits/linear wt)",
              flush=True)
        if use_duobit:
            print(f"[{tag}] learn_scales={int(getattr(args, 'learn_scales', 0))} "
                  f"scale_lr={getattr(args, 'scale_lr', 0):.1e} "
                  f"scale_relative_lr={int(getattr(args, 'scale_relative_lr', 0))} "
                  f"duobit_lr={args.duobit_lr:.1e} pefa_clip={args.pefa_clip} "
                  f"duobit_wd={getattr(args, 'duobit_wd', 0.0)}", flush=True)

    jsonl = open(out_dir / f"logs_{tag}.jsonl", "w") if is_main else None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    logs: List[Dict[str, Any]] = []
    val_curve: List[Dict[str, Any]] = []
    skipped = 0
    tps_ema = None
    t_win = time.time()
    tok_win = 0
    sum_data_ms = sum_fb_ms = sum_opt_ms = 0.0
    n_timed = 0
    drift_checks = 0
    drift_failures = 0
    eval_sec = 0.0
    params_with_grad = [p for p in model.parameters() if p.requires_grad]

    # Loss and gradient norms are accumulated on the device and read back only
    # on logging steps. v1 called .item() every step (and once per layer inside
    # the optimizer), which serialised the pipeline ~150 times per step.
    loss_acc = torch.zeros((), device=device, dtype=torch.float32)
    gnorm_acc = torch.zeros((), device=device, dtype=torch.float32)
    nonfinite_acc = torch.zeros((), device=device, dtype=torch.float32)
    acc_n = 0

    try:
        for step in range(1, args.steps + 1):
            f = lr_factor(step, args.warmup_steps, args.steps)
            cur_lr = args.lr * f
            if use_duobit:
                optimizer.collect_stats = (step % args.log_freq == 0 or step == 1
                                           or step == args.steps)
                optimizer.set_lrs(cur_lr, args.duobit_lr * f,
                                  scale_lr=args.scale_lr * f)
            else:
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr

            t0 = time.time()
            ids = loader.next_batch().to(device, non_blocking=True)
            t_data = (time.time() - t0) * 1000.0

            if use_duobit:
                optimizer.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            if amp_ok:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, loss = net(ids, targets=ids)
            else:
                logits, loss = net(ids, targets=ids)
            loss_acc += loss.detach().float()
            nonfinite_acc += (~torch.isfinite(loss.detach())).float()

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if use_duobit and world > 1 and dist.is_initialized():
                handles = []
                nccl = torch.cuda.is_available() and dist.get_backend() == "nccl"
                op = dist.ReduceOp.AVG if nccl else dist.ReduceOp.SUM
                for lyr in optimizer.duobit_layers:
                    if lyr.ephemeral_w is not None and lyr.ephemeral_w.grad is not None:
                        handles.append(dist.all_reduce(lyr.ephemeral_w.grad, op=op,
                                                       async_op=True))
                for h in handles:
                    h.wait()
                if not nccl:
                    for lyr in optimizer.duobit_layers:
                        if lyr.ephemeral_w is not None and lyr.ephemeral_w.grad is not None:
                            lyr.ephemeral_w.grad.div_(world)
            t2 = time.time()

            if use_duobit:
                gnorm_p = torch.nn.utils.clip_grad_norm_(
                    [p for p in params_with_grad if p.grad is not None], args.grad_clip)
                gnorm_e = optimizer.clip_ephemeral_grads(args.grad_clip)
                gnorm_acc += torch.sqrt(gnorm_p * gnorm_p + gnorm_e * gnorm_e)
                optimizer.step()
            else:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                gnorm_acc += torch.nn.utils.clip_grad_norm_(
                    [p for p in params_with_grad if p.grad is not None], args.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            acc_n += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t3 = time.time()

            sum_data_ms += t_data
            sum_fb_ms += (t2 - t1) * 1000.0
            sum_opt_ms += (t3 - t2) * 1000.0
            n_timed += 1
            tok_win += world * args.batch_size * args.seq_len

            if step % args.log_freq == 0 or step == 1 or step == args.steps:
                dt_win = max(time.time() - t_win, 1e-9)
                tps_win = tok_win / dt_win
                tps_ema = tps_win if tps_ema is None else 0.85 * tps_ema + 0.15 * tps_win
                # single host sync per logging step, for every accumulated stat
                packed = torch.stack([loss_acc / max(acc_n, 1),
                                      gnorm_acc / max(acc_n, 1),
                                      nonfinite_acc])
                if world > 1 and dist.is_initialized():
                    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                    packed = packed / world
                loss_avg, gnorm_total, n_bad = [float(x) for x in packed.tolist()]
                loss_acc.zero_(); gnorm_acc.zero_(); nonfinite_acc.zero_(); acc_n = 0
                if n_bad > 0:
                    skipped += int(round(n_bad))
                    if is_main:
                        print(f"[{tag}] WARNING: {n_bad:.0f} non-finite loss value(s) "
                              f"in the last {args.log_freq} steps", flush=True)
                mem_a = mem_r = mem_p = 0.0
                if torch.cuda.is_available():
                    mem_a = torch.cuda.memory_allocated(device) / 1048576.0
                    mem_r = torch.cuda.memory_reserved(device) / 1048576.0
                    mem_p = torch.cuda.max_memory_allocated(device) / 1048576.0
                row: Dict[str, Any] = {
                    "step": step,
                    "loss": loss_avg,
                    "ppl": math.exp(min(loss_avg, 20.0)),
                    "lr": cur_lr,
                    "gnorm": gnorm_total,
                    "tok_per_s": tps_win,
                    "tok_per_s_ema": tps_ema,
                    "tokens_seen": step * world * args.batch_size * args.seq_len,
                    "step_ms": (t3 - t1) * 1000.0,
                    "data_ms": t_data,
                    "mem_alloc_mb": mem_a,
                    "mem_reserved_mb": mem_r,
                    "mem_peak_mb": mem_p,
                    "rss_mb": _rss_mb(),
                    "skipped_steps": skipped,
                }
                if use_duobit:
                    trans, resid = optimizer.read_stats()
                    smean, sratio = optimizer.read_scale_stats()
                    row["duobit_lr"] = args.duobit_lr * f
                    row["transition_rate"] = trans
                    row["residual_abs_mean"] = resid
                    row["scale_mean"] = smean
                    row["scale_ratio"] = sratio
                if scaler is not None:
                    row["loss_scale"] = float(scaler.get_scale())
                logs.append(row)
                if is_main:
                    jsonl.write(json.dumps(row) + "\n")
                    jsonl.flush()
                    if step % (args.log_freq * 5) == 0 or step == 1 or step == args.steps:
                        extra = (f" | trans={row['transition_rate'] * 100:.3f}%"
                                 f" | s/s0={row['scale_ratio']:.3f}"
                                 if use_duobit else "")
                        print(f"[{tag}] step {step}/{args.steps} | loss {loss_avg:.4f} | "
                              f"ppl {row['ppl']:.2f} | tok/s {tps_win:,.0f} | "
                              f"gnorm {gnorm_total:.3f} | lr {cur_lr:.2e}{extra} | "
                              f"mem {mem_a:.0f}/{mem_p:.0f} MB | rss {row['rss_mb']:.0f} MB",
                              flush=True)
                t_win = time.time()
                tok_win = 0

            if use_duobit and world > 1 and step % 250 == 0:
                drift_checks += 1
                if not codes_sync_check(model, world):
                    drift_failures += 1
                    if is_main:
                        print(f"[{tag}] WARNING: code drift across ranks at step {step}",
                              flush=True)

            if (args.eval_freq > 0 and step % args.eval_freq == 0) or step == args.steps:
                te = time.time()
                vl, vp = evaluate(net, val_tokens, args, device, world)
                eval_sec += time.time() - te
                val_curve.append({"step": step, "val_loss": vl, "val_ppl": vp})
                if is_main:
                    print(f"[{tag}] >> step {step} VAL | loss {vl:.4f} | ppl {vp:.2f} "
                          f"| rss {_rss_mb():.0f} MB", flush=True)
                t_win = time.time()
                tok_win = 0
            if step % 500 == 0:
                gc.collect()
    finally:
        loader.stop()

    sample_text = ""
    if is_main:
        try:
            sample_text = greedy_sample(model, tokenizer, device, args)
            print(f"[{tag}] greedy sample: {sample_text[:160]!r}", flush=True)
        except Exception as e:
            sample_text = f"(sampling failed: {e})"

    persist_final = persistent_state_report(model, args, mode, optimizer)
    wall = time.time() - t_start
    total_tokens = args.steps * world * args.batch_size * args.seq_len
    losses = [r["loss"] for r in logs] if logs else [float("nan")]
    tps_list = [r["tok_per_s"] for r in logs] if logs else [0.0]
    mems = [r["mem_alloc_mb"] for r in logs] if logs else [0.0]
    peaks = [r["mem_peak_mb"] for r in logs] if logs else [0.0]
    gnorms = [r["gnorm"] for r in logs] if logs else [0.0]
    trans = [r["transition_rate"] for r in logs if "transition_rate" in r]
    resids = [r["residual_abs_mean"] for r in logs if "residual_abs_mean" in r]
    ratios = [r["scale_ratio"] for r in logs if "scale_ratio" in r]
    train_ms = sum_data_ms + sum_fb_ms + sum_opt_ms

    summary: Dict[str, Any] = {
        "name": name,
        "mode": mode,
        "amp": bool(amp_ok),
        "n_params": n_params,
        "steps": args.steps,
        "tokens": total_tokens,
        "wall_sec": wall,
        "train_wall_sec": wall - eval_sec,
        "eval_sec": eval_sec,
        "avg_tok_per_s": total_tokens / max(wall, 1e-9),
        "avg_step_tok_per_s": total_tokens / max(train_ms / 1000.0, 1e-9),
        "final_train_loss": losses[-1],
        "train_loss_smoothed_final": sum(losses[-min(5, len(losses)):]) / min(5, len(losses)),
        "best_val_loss": min((c["val_loss"] for c in val_curve), default=None),
        "final_val_loss": val_curve[-1]["val_loss"] if val_curve else None,
        "best_val_ppl": min((c["val_ppl"] for c in val_curve if c["val_ppl"] != float("inf")),
                            default=None),
        "final_val_ppl": val_curve[-1]["val_ppl"] if val_curve else None,
        "peak_gpu_mem_mb": max(peaks) if peaks else 0.0,
        "avg_gpu_mem_mb": (sum(mems) / len(mems)) if mems else 0.0,
        "avg_gnorm": sum(gnorms) / len(gnorms) if gnorms else 0.0,
        "max_gnorm": max(gnorms) if gnorms else 0.0,
        "mean_transition_rate": (sum(trans) / len(trans)) if trans else None,
        "mean_residual_abs": (sum(resids) / len(resids)) if resids else None,
        "final_scale_ratio": ratios[-1] if ratios else None,
        "skipped_steps": skipped,
        "code_drift_checks": drift_checks,
        "code_drift_failures": drift_failures,
        "docs_streamed": loader.docs_seen,
        "sample_text": sample_text,
        "step_time_breakdown_ms": {
            "data_wait": sum_data_ms / max(n_timed, 1),
            "fwd_bwd_sync": sum_fb_ms / max(n_timed, 1),
            "optimizer": sum_opt_ms / max(n_timed, 1),
        },
        "persistent": persist_final,
        "config": {k: v for k, v in vars(args).items()
                   if k not in ("out_dir", "runs_json")},
    }

    if is_main:
        jsonl.close()
        (out_dir / f"val_curve_{tag}.json").write_text(json.dumps(val_curve, indent=1))
        (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1))
        if logs:
            keys = sorted({k for r in logs for k in r})
            with open(out_dir / f"step_logs_{tag}.csv", "w", newline="") as fcsv:
                w = csv.DictWriter(fcsv, fieldnames=keys)
                w.writeheader()
                w.writerows(logs)
        print(f"[{tag}] DONE | wall {wall / 60:.1f} min | "
              f"final val loss {summary['final_val_loss']} | "
              f"tok/s {summary['avg_step_tok_per_s']:,.0f}", flush=True)

    del optimizer, net, model, val_tokens, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"logs": logs, "val_curve": val_curve, "summary": summary}


# ----------------------------------------------------------------------------
# Report generation: every visualization + markdown + csv + json
# ----------------------------------------------------------------------------
def smooth(vals: List[float], k: int = 15) -> List[float]:
    if len(vals) < 3 or k <= 1:
        return list(vals)
    out = []
    dq_ = deque()
    s = 0.0
    for v in vals:
        dq_.append(v)
        s += v
        if len(dq_) > k:
            s -= dq_.popleft()
        out.append(s / len(dq_))
    return out


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return "inf"
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def _md_table(header: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
            "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def run_label(name: str, summary: Dict[str, Any], args) -> str:
    """Human-readable legend entry for one run."""
    mode = summary.get("mode", name)
    cfg = summary.get("config", {})
    if mode == "duobit":
        levels = int(cfg.get("n_levels", args.n_levels))
        bits = {2: "binary (1-bit)", 3: "ternary (1.58-bit)",
                4: "2-bit"}.get(levels, f"{levels}-level")
        base = f"DUOBIT {bits} QPEFA-{int(cfg.get('qpefa_bits', args.qpefa_bits))}"
    elif mode == "fp16":
        base = "FP16 AMP AdamW (baseline)"
    else:
        base = "FP32 AdamW (baseline)"
    return base if name == mode else f"{name}: {base}"


def make_report(results: Dict[str, Any], args, out_dir: Path, hw_info: Dict[str, Any]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bits_name = {2: "binary (1-bit)", 3: "ternary (1.58-bit)",
                 4: "2-bit (4-level)"}[args.n_levels]
    runs = {k: v for k, v in results.items()
            if isinstance(v, dict) and v.get("logs")}
    order = list(runs.keys())
    labels = {k: run_label(k, runs[k]["summary"], args) for k in order}
    colors = {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(order)}
    duobit_runs = [k for k in order if runs[k]["summary"].get("mode") == "duobit"]

    def save(fig, name):
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)
        print(f"[report] figures/{name}", flush=True)

    def series(k, key):
        rows = [r for r in runs[k]["logs"] if r.get(key) is not None]
        return [r["step"] for r in rows], [r[key] for r in rows]

    if not runs:
        print("[report] no completed runs to plot", flush=True)
        return

    # 1) train loss
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for k in order:
        xs, ys = series(k, "loss")
        ax.plot(xs, ys, alpha=0.22, color=colors[k], linewidth=0.8)
        ax.plot(xs, smooth(ys, max(3, len(ys) // 10)), color=colors[k],
                linewidth=1.8, label=labels[k])
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.set_title("Training loss on FineWeb-EDU (identical data & compute)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "train_loss.png")

    # 2) validation loss + ppl
    if any(runs[k]["val_curve"] for k in order):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        for k in order:
            vc = runs[k]["val_curve"]
            if not vc:
                continue
            xs = [c["step"] for c in vc]
            ax1.plot(xs, [c["val_loss"] for c in vc], marker="o", ms=4,
                     color=colors[k], label=labels[k])
            ax2.plot(xs, [c["val_ppl"] for c in vc], marker="o", ms=4,
                     color=colors[k], label=labels[k])
        ax1.set_xlabel("step"); ax1.set_ylabel("val loss")
        ax1.set_title("Validation loss"); ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax2.set_xlabel("step"); ax2.set_ylabel("val perplexity")
        ax2.set_yscale("log")
        ax2.set_title("Validation perplexity (log)"); ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        save(fig, "val_loss_ppl.png")

    # 3) throughput
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for k in order:
        xs, ys = series(k, "tok_per_s_ema")
        ax.plot(xs, ys, color=colors[k], linewidth=1.8, label=labels[k])
        ax.axhline(runs[k]["summary"]["avg_step_tok_per_s"], color=colors[k],
                   linestyle="--", alpha=0.5, linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("tokens / second (global, all ranks)")
    ax.set_title("Training throughput (dashed = run average)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "throughput.png")

    # 4) gradient norm
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for k in order:
        xs, ys = series(k, "gnorm")
        ax.plot(xs, ys, alpha=0.6, color=colors[k], linewidth=0.9, label=labels[k])
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("global gradient norm (pre-clip)")
    ax.set_title("Gradient norm (params + ephemeral weights, log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "gnorm.png")

    # 5) runtime GPU memory
    if torch.cuda.is_available():
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        for k in order:
            xs, ys = series(k, "mem_alloc_mb")
            ax.plot(xs, ys, color=colors[k], linewidth=1.4, label=f"{labels[k]}")
            xs2, ys2 = series(k, "mem_reserved_mb")
            ax.plot(xs2, ys2, color=colors[k], linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_xlabel("step"); ax.set_ylabel("MB")
        ax.set_title("CUDA memory over training (solid=allocated, dashed=reserved)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, "memory_runtime.png")

    # 6) DuoBIT transition / residual / scale dynamics
    if duobit_runs:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        for k in duobit_runs:
            xs, tr = series(k, "transition_rate")
            axes[0].plot(xs, [t * 100 for t in tr], color=colors[k], linewidth=1.2,
                         label=labels[k])
            xs2, rs = series(k, "residual_abs_mean")
            axes[1].plot(xs2, rs, color=colors[k], linewidth=1.2, label=labels[k])
            xs3, sr = series(k, "scale_ratio")
            if sr:
                axes[2].plot(xs3, sr, color=colors[k], linewidth=1.2, label=labels[k])
        axes[0].set_ylabel("% of codes changed per step")
        axes[0].set_title("Code transition rate")
        axes[1].set_ylabel("mean |QPEFA residual|")
        axes[1].set_title("QPEFA residual (error-feedback health)")
        axes[2].set_ylabel("mean s / s_init")
        axes[2].set_title("Group-scale drift from initialization")
        axes[2].axhline(1.0, color="k", linestyle=":", linewidth=1.0)
        for ax_ in axes:
            ax_.set_xlabel("step"); ax_.grid(True, alpha=0.3); ax_.legend(fontsize=7)
        save(fig, "duobit_transitions.png")

    # 7) learning-rate schedule
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for k in order:
        xs, ys = series(k, "lr")
        ax.plot(xs, ys, color=colors[k], linewidth=1.5, label=f"{labels[k]} lr")
        xs2, ys2 = series(k, "duobit_lr")
        if ys2:
            ax.plot(xs2, ys2, color=colors[k], linewidth=1.1, linestyle="--",
                    label=f"{k} discrete lr")
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("learning rate (log)")
    ax.set_title("LR schedule: warmup + cosine to 10% floor")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7)
    save(fig, "lr_schedule.png")

    # 8) step-time breakdown
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(order)), 5),
                           constrained_layout=True)
    comps = ["data_wait", "fwd_bwd_sync", "optimizer"]
    comp_labels = ["data wait", "fwd+bwd+gradsync", "optimizer+clip"]
    bottom = [0.0] * len(order)
    for comp, cl in zip(comps, comp_labels):
        vals = [runs[k]["summary"]["step_time_breakdown_ms"][comp] for k in order]
        ax.bar(range(len(order)), vals, 0.6, bottom=bottom, label=cl)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("ms / step (avg)")
    ax.set_title("Average step-time breakdown")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    save(fig, "step_time_breakdown.png")

    # 9) comparison bars across every run
    def _safe(v, lo=0.0):
        if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
            return lo
        return v

    panels = [
        ("final val loss (lower=better)", lambda s: _safe(s["final_val_loss"]), 3),
        ("final val PPL (lower=better)", lambda s: _safe(s["final_val_ppl"]), 1),
        ("avg tok/s (higher=better)", lambda s: _safe(s["avg_step_tok_per_s"]), 0),
        ("peak GPU memory MB", lambda s: _safe(s["peak_gpu_mem_mb"]), 0),
        ("persistent train state MB",
         lambda s: _safe(s["persistent"]["persistent_train_mb"]), 1),
        ("inference MB",
         lambda s: _safe(s["persistent"]["inference_mb"]), 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, (title, fn, nd) in zip(axes.flat, panels):
        vals = [fn(runs[k]["summary"]) for k in order]
        bars = ax.bar(range(len(order)), vals, color=[colors[k] for k in order])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=7, rotation=20, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), _fmt(v, nd),
                    ha="center", va="bottom", fontsize=7)
    save(fig, "comparison_bars.png")

    # 10) dashboard
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for k in order:
        xs, ys = series(k, "loss")
        axes[0, 0].plot(xs, smooth(ys, max(3, len(ys) // 10)), color=colors[k],
                        label=labels[k])
        vc = runs[k]["val_curve"]
        if vc:
            axes[0, 1].plot([c["step"] for c in vc], [c["val_loss"] for c in vc],
                            marker="o", ms=3, color=colors[k], label=labels[k])
            axes[0, 2].plot([c["step"] for c in vc], [c["val_ppl"] for c in vc],
                            marker="o", ms=3, color=colors[k], label=labels[k])
        x1, t2_ = series(k, "tok_per_s_ema")
        axes[1, 0].plot(x1, t2_, color=colors[k], label=labels[k])
        x2, gn = series(k, "gnorm")
        axes[1, 1].plot(x2, gn, alpha=0.6, color=colors[k], label=labels[k])
        x3, mm = series(k, "mem_alloc_mb")
        axes[1, 2].plot(x3, mm, color=colors[k], label=labels[k])
    axes[0, 0].set_title("train loss"); axes[0, 1].set_title("val loss")
    axes[0, 2].set_title("val ppl"); axes[0, 2].set_yscale("log")
    axes[1, 0].set_title("tokens/sec (ema)"); axes[1, 1].set_title("grad norm")
    axes[1, 1].set_yscale("log")
    axes[1, 2].set_title("GPU mem allocated MB")
    for ax_ in axes.flat:
        ax_.set_xlabel("step"); ax_.grid(True, alpha=0.3); ax_.legend(fontsize=6)
    fig.suptitle("DUOBIT-EST vs FP32 / FP16 on FineWeb-EDU -- dashboard", fontsize=13)
    save(fig, "dashboard.png")
    try:
        import shutil
        shutil.copy(fig_dir / "dashboard.png", out_dir / "dashboard.png")
    except Exception:
        pass

    # ---- markdown report ----
    md = build_markdown(results, args, hw_info, runs, labels, order, bits_name)
    (out_dir / "summary_report.md").write_text(md)
    print("[report] summary_report.md written", flush=True)

    # ---- all metrics json ----
    allm = {"config": vars(args), "hardware": hw_info,
            "results": {k: ({"summary": v["summary"], "val_curve": v["val_curve"],
                             "logs": v["logs"]} if "summary" in v else v)
                        for k, v in results.items()}}
    (out_dir / "all_metrics.json").write_text(json.dumps(allm, indent=1))
    print("[report] all_metrics.json written", flush=True)

    # ---- comparison csv (one column per run) ----
    metrics = [
        ("params", lambda s: s["n_params"], 0),
        ("steps", lambda s: s["steps"], 0),
        ("tokens seen", lambda s: s["tokens"], 0),
        ("wall time (s)", lambda s: s["wall_sec"], 1),
        ("avg tok/s (steps)", lambda s: s["avg_step_tok_per_s"], 0),
        ("final train loss", lambda s: s["final_train_loss"], 4),
        ("best val loss", lambda s: s["best_val_loss"], 4),
        ("final val loss", lambda s: s["final_val_loss"], 4),
        ("final val ppl", lambda s: s["final_val_ppl"], 2),
        ("peak GPU mem MB", lambda s: s["peak_gpu_mem_mb"], 0),
        ("avg GPU mem MB", lambda s: s["avg_gpu_mem_mb"], 0),
        ("persistent train MB", lambda s: s["persistent"]["persistent_train_mb"], 1),
        ("inference MB", lambda s: s["persistent"]["inference_mb"], 1),
        ("linear train bits/weight",
         lambda s: s["persistent"]["linear_train_bits_per_weight"], 2),
        ("linear infer bits/weight",
         lambda s: s["persistent"]["linear_infer_bits_per_weight"], 2),
        ("train compression vs FP32",
         lambda s: s["persistent"]["persistent_train_compression_vs_fp32"], 2),
        ("inference compression vs FP32",
         lambda s: s["persistent"]["inference_compression_vs_fp32"], 2),
        ("mean grad norm", lambda s: s["avg_gnorm"], 4),
        ("mean transition rate", lambda s: s["mean_transition_rate"], 5),
        ("mean |QPEFA residual|", lambda s: s["mean_residual_abs"], 6),
        ("final scale/init ratio", lambda s: s["final_scale_ratio"], 3),
        ("skipped steps", lambda s: s["skipped_steps"], 0),
    ]
    with open(out_dir / "comparison.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["metric"] + order)
        for label, fn, nd in metrics:
            row = [label]
            for k in order:
                try:
                    row.append(_fmt(fn(runs[k]["summary"]), nd))
                except Exception:
                    row.append("n/a")
            w.writerow(row)
    print("[report] comparison.csv written", flush=True)


def build_markdown(results, args, hw_info, runs, labels, order, bits_name) -> str:
    a = args
    lines: List[str] = []
    A = lines.append
    A(f"# DUOBIT-EST vs FP32 / FP16 on FineWeb-EDU (script v{__version__})")
    A("")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.")
    A("")
    gpus = hw_info.get("gpus", [])
    gpu_txt = ", ".join(f"{g['name']} ({g['mem_gb']} GB, cc {g['cc']})" for g in gpus) \
        or "CPU only"
    A(f"**Hardware**: {gpu_txt} | torch {hw_info.get('torch')} | "
      f"python {hw_info.get('python')}")
    A("")
    A("## 1. Setup")
    A("")
    A(f"- Architecture: {a.n_layers}-layer decoder, d_model {a.d_model}, "
      f"{a.n_heads} heads, SwiGLU d_ff {a.d_ff}, RMSNorm, RoPE, causal SDPA")
    A(f"- Data: `{a.dataset}` config `{a.dataset_config}`, GPT-2 tokenizer, "
      f"streamed; held-out validation of {a.val_tokens:,} tokens/rank from files "
      f"disjoint from training")
    A(f"- Budget per run: {a.steps:,} steps, batch {a.batch_size}/GPU x "
      f"{a.seq_len} tokens, seed {a.seed}")
    A(f"- DuoBIT: {bits_name} codebook (rho={a.rho:.4f}), group size {a.group_size}, "
      f"QPEFA {a.qpefa_bits}-bit (clip {a.pefa_clip}), first moment "
      f"{a.moment_bits}-bit block-wise, factored second moment")
    A("- Every run consumes the identical token stream (same seed, same rank "
      "file assignment), so the comparison is same-data and same-compute.")
    A("")
    A("### Runs")
    A("")
    rows = []
    for k in order:
        s = runs[k]["summary"]
        cfg = s.get("config", {})
        if s.get("mode") == "duobit":
            note = (f"learn_scales={cfg.get('learn_scales')}, "
                    f"scale_lr={cfg.get('scale_lr')}, "
                    f"rel_lr={cfg.get('scale_relative_lr')}, "
                    f"duobit_lr={cfg.get('duobit_lr')}, "
                    f"clip={cfg.get('pefa_clip')}, wd={cfg.get('duobit_wd')}")
        else:
            note = f"lr={cfg.get('lr')}, wd={cfg.get('wd')}, amp={s.get('amp')}"
        rows.append([k, s.get("mode"), note])
    A(_md_table(["run", "mode", "configuration"], rows))
    A("")

    A("## 2. Results")
    A("")
    hdr = ["metric"] + order
    def rowfor(label, fn, nd=4):
        out = [label]
        for k in order:
            try:
                out.append(_fmt(fn(runs[k]["summary"]), nd))
            except Exception:
                out.append("n/a")
        return out
    res_rows = [
        rowfor("final val loss", lambda s: s["final_val_loss"], 4),
        rowfor("final val PPL", lambda s: s["final_val_ppl"], 2),
        rowfor("best val loss", lambda s: s["best_val_loss"], 4),
        rowfor("final train loss", lambda s: s["final_train_loss"], 4),
        rowfor("tokens seen", lambda s: s["tokens"], 0),
        rowfor("wall time (min)", lambda s: s["wall_sec"] / 60.0, 1),
        rowfor("avg tok/s", lambda s: s["avg_step_tok_per_s"], 0),
        rowfor("peak GPU mem (MB)", lambda s: s["peak_gpu_mem_mb"], 0),
    ]
    A(_md_table(hdr, res_rows))
    A("")
    A("### Persistent state and precision")
    A("")
    mem_rows = [
        rowfor("linear train bits/weight",
               lambda s: s["persistent"]["linear_train_bits_per_weight"], 2),
        rowfor("linear inference bits/weight",
               lambda s: s["persistent"]["linear_infer_bits_per_weight"], 2),
        rowfor("persistent train state (MB)",
               lambda s: s["persistent"]["persistent_train_mb"], 1),
        rowfor("inference footprint (MB)",
               lambda s: s["persistent"]["inference_mb"], 1),
        rowfor("train compression vs FP32",
               lambda s: s["persistent"]["persistent_train_compression_vs_fp32"], 2),
        rowfor("inference compression vs FP32",
               lambda s: s["persistent"]["inference_compression_vs_fp32"], 2),
    ]
    A(_md_table(hdr, mem_rows))
    A("")
    A("Persistent training state counts what must be held for the whole run: for "
      "DuoBIT the 2-bit codes, the FP32 group scales, the int8 QPEFA residual, the "
      "block-wise int8 first moment and the factored second moment (plus two FP32 "
      "Adam moments per group when the scales are trained); for FP32 and FP16 the "
      "FP32 weight plus both FP32 Adam moments (96 bits/weight -- mixed precision "
      "keeps FP32 master weights, so only its inference export halves). Embeddings "
      "and norm gains are FP32 in every regime and are included in both totals.")
    A("")

    dq = [k for k in order if runs[k]["summary"].get("mode") == "duobit"]
    if dq:
        A("### DuoBIT training dynamics")
        A("")
        dyn_rows = [
            rowfor("mean code transition rate", lambda s: s["mean_transition_rate"], 5),
            rowfor("mean |QPEFA residual|", lambda s: s["mean_residual_abs"], 6),
            rowfor("final scale / init scale", lambda s: s["final_scale_ratio"], 3),
            rowfor("mean grad norm", lambda s: s["avg_gnorm"], 4),
            rowfor("code drift failures across ranks",
                   lambda s: s["code_drift_failures"], 0),
        ]
        A(_md_table(hdr, dyn_rows))
        A("")

    base = None
    for k in order:
        if runs[k]["summary"].get("mode") == "fp32":
            base = k
            break
    if base is not None and len(order) > 1:
        A("### Quality gap to the FP32 baseline")
        A("")
        bs = runs[base]["summary"]
        gap_rows = []
        for k in order:
            if k == base:
                continue
            s = runs[k]["summary"]
            try:
                d_loss = s["final_val_loss"] - bs["final_val_loss"]
                ratio = s["final_val_ppl"] / bs["final_val_ppl"]
                gap_rows.append([k, _fmt(d_loss, 4), _fmt(ratio, 2) + "x",
                                 _fmt(s["avg_step_tok_per_s"] /
                                      max(bs["avg_step_tok_per_s"], 1e-9), 3)])
            except Exception:
                gap_rows.append([k, "n/a", "n/a", "n/a"])
        A(_md_table(["run", "val loss delta", "PPL ratio", "throughput ratio"],
                    gap_rows))
        A("")

    A("## 3. Samples")
    A("")
    for k in order:
        txt = runs[k]["summary"].get("sample_text", "")
        A(f"**{k}** (greedy, prompt `Once upon a time`):")
        A("")
        A("```text")
        A(txt or "(none)")
        A("```")
        A("")

    A("## 4. Figures")
    A("")
    for fn in ["dashboard.png", "train_loss.png", "val_loss_ppl.png",
               "comparison_bars.png", "throughput.png", "duobit_transitions.png",
               "gnorm.png", "memory_runtime.png", "step_time_breakdown.png",
               "lr_schedule.png"]:
        A(f"![{fn}](figures/{fn})")
        A("")

    failed = {k: v for k, v in results.items() if isinstance(v, dict) and "error" in v}
    if failed:
        A("## 5. Failed runs")
        A("")
        for k, v in failed.items():
            A(f"- `{k}`: {v['error']}")
        A("")
    return "\n".join(lines)


def resolve_runs(args) -> List[Dict[str, Any]]:
    """Expand --mode / --runs-json into an ordered list of run specs.

    Each spec is {"name": str, "mode": str, "overrides": {dest: value}}. Any key
    other than "name" in a --runs-json entry overrides that argparse dest for
    that run only, which is how the tuning sweep varies one knob at a time while
    everything else (data stream, seed, architecture, step budget) is shared.
    """
    if args.runs_json:
        spec_src = args.runs_json
        if os.path.isfile(spec_src):
            spec_src = Path(spec_src).read_text()
        specs = json.loads(spec_src)
        if not isinstance(specs, list) or not specs:
            raise ValueError("--runs-json must be a non-empty JSON list")
        valid = set(vars(args).keys())
        out = []
        for i, sp in enumerate(specs):
            if not isinstance(sp, dict):
                raise ValueError(f"run spec {i} is not an object")
            mode = sp.get("mode", "duobit")
            if mode not in ("duobit", "fp32", "fp16"):
                raise ValueError(f"run spec {i}: bad mode {mode!r}")
            name = str(sp.get("name", f"run{i}"))
            overrides = {k: v for k, v in sp.items() if k not in ("name", "mode")}
            bad = sorted(set(overrides) - valid)
            if bad:
                raise ValueError(f"run spec {name!r}: unknown option(s) {bad}")
            out.append({"name": name, "mode": mode, "overrides": overrides})
        return out
    modes = {"both": ["duobit", "fp32"],
             "all": ["duobit", "fp32", "fp16"]}.get(args.mode, [args.mode])
    return [{"name": m, "mode": m, "overrides": {}} for m in modes]


def args_for_run(args, spec: Dict[str, Any]) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    for k, v in spec["overrides"].items():
        setattr(run_args, k, v)
    return run_args


def worker(rank: int, world: int, args):
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank, world_size=world)
    if rank == 0:
        # rank 0 compiles/validates kernels first; others reuse the build cache
        _get_ext()
    if world > 1:
        dist.barrier()
    _get_ext()

    hw = collect_hw_info() if rank == 0 else None
    specs = resolve_runs(args)
    results: Dict[str, Any] = {}
    for spec in specs:
        name = spec["name"]
        try:
            results[name] = run_experiment(name, spec["mode"], rank, world,
                                           args_for_run(args, spec), device)
        except Exception as e:
            traceback.print_exc()
            results[name] = {"error": f"{type(e).__name__}: {e}",
                             "summary": {"name": name, "mode": spec["mode"]}}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if world > 1:
            dist.barrier()
        if rank == 0:
            # write the report after every run so a run that is cut short by the
            # session time limit still leaves a complete report for what finished
            try:
                make_report(results, args, Path(args.out_dir),
                            hw or collect_hw_info())
            except Exception as e:
                traceback.print_exc()
                print(f"[report] generation failed: {e}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    args = build_argparser().parse_args()
    if args.smoke:
        args = apply_smoke_overrides(args)
    if args.no_kernels:
        os.environ["DUOBIT_NO_KERNELS"] = "1"
    args.qpefa_bits = min(args.qpefa_bits, 8)
    args.moment_bits = min(args.moment_bits, 8)
    if args.out_dir is None:
        args.out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./duobit_out"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_deps()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if torch.cuda.is_available():
        world = torch.cuda.device_count()
        if args.max_gpus > 0:
            world = min(world, args.max_gpus)
    else:
        world = 1

    specs = resolve_runs(args)   # fail fast on a bad --runs-json
    print("=" * 78, flush=True)
    print(f"  DUOBIT-EST vs FP32/FP16 on FineWeb-EDU  |  script v{__version__}", flush=True)
    print(f"  torch {torch.__version__} | world_size {world} | out_dir {out_dir}", flush=True)
    try:
        import datasets
        import transformers
        print(f"  datasets {datasets.__version__} | transformers {transformers.__version__}",
              flush=True)
    except Exception:
        pass
    print(f"  host RAM check: current RSS {_rss_mb():.0f} MB", flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory / 2 ** 30:.1f} GB, "
                  f"sm_{p.major}{p.minor})", flush=True)
    print(f"  model: d={args.d_model} L={args.n_layers} H={args.n_heads} ffn={args.d_ff} "
          f"| steps={args.steps} | B={args.batch_size}/gpu | T={args.seq_len}", flush=True)
    print(f"  duobit: n_levels={args.n_levels} G={args.group_size} "
          f"qpefa={args.qpefa_bits}bit | data: {args.dataset}/{args.dataset_config} "
          f"(streaming) + {args.tokenizer} tokenizer", flush=True)
    print(f"  runs ({len(specs)}): " + ", ".join(
        f"{sp['name']}[{sp['mode']}]" + (f" {sp['overrides']}" if sp["overrides"] else "")
        for sp in specs), flush=True)
    print("=" * 78, flush=True)

    (out_dir / "config.json").write_text(json.dumps(
        {"args": vars(args), "runs": specs}, indent=1))

    if world > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        tmp_mp.spawn(worker, args=(world, args), nprocs=world, join=True)
    else:
        worker(0, 1, args)

    print("\n[main] experiment complete. Report: "
          f"{out_dir / 'summary_report.md'}", flush=True)
    # Clean exit: background streaming/HTTP threads (daemon) do not survive
    # interpreter finalization gracefully; all outputs are closed by now.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
