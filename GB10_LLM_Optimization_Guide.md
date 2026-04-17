# NVIDIA GB10 / DGX Spark - Complete LLM Inference Optimization Guide
## From 5.5 tok/s to Maximum Performance

**Machine**: NVIDIA GB10 (Grace Blackwell), 128GB unified LPDDR5X, CUDA 13.0  
**Setup**: llama.cpp b8719 (commit 2dcb7f74e), Open WebUI + LiteLLM, 2× GX10 (256GB total)  
**Date**: April 17, 2026  
**Status**: 20 benchmark rounds complete — production configs validated

---

## TABLE OF CONTENTS

1. [Why 5.5 tok/s Is Expected (The Physics)](#part-1)
2. [Known Bugs Affecting Performance](#part-2)
3. [System-Level Optimizations](#part-3)
4. [llama.cpp Optimized Build & Configuration](#part-4)
5. [Model Selection Strategy](#part-5)
6. [KV Cache Quantization](#part-6)
7. [Backend Comparison](#part-7)
8. [Power, Thermal & Firmware](#part-8)
9. [Dual GB10 Setup (200GbE)](#part-9)
10. [Boot Optimization Script](#part-10)
11. [Benchmark Results — All 20 Rounds](#part-11)
12. [Current Model Inventory](#part-12)
13. [New Models Available (April 2026)](#part-13)
14. [Production Configs](#part-14)
15. [Realistic Performance Expectations](#part-15)
16. [Action Plan](#part-16)
17. [All Sources](#part-17)

---

<a id="part-1"></a>
## PART 1: WHY 5.5 TOK/S IS EXPECTED (The Physics)

The GB10 has **273 GB/s** memory bandwidth (LPDDR5X). A 31B Q8 model = ~33GB. Each generated token must read ALL weights:

```
Theoretical max = 273 GB/s / 33 GB = ~8.3 tok/s
Practical = ~5.5-8 tok/s (overhead, KV cache, shared CPU/GPU memory bus)
```

**This is a hard physical limit.** No software can exceed it for this model size and quantization.

For comparison:
```
GB10 LPDDR5X:     273 GB/s    ← you are here
RTX 4090 GDDR6X: 1,008 GB/s  (3.7x more, but only 24 GB capacity)
A100 HBM2e:      2,039 GB/s  (7.5x more)
H100 HBM3:       3,350 GB/s  (12.3x more)
```

**The GB10's advantage is CAPACITY (128 GB), not BANDWIDTH.** It holds models that won't fit on a 4090 but runs them slower per-token. MoE models are designed for exactly this trade-off.

**The 40-60 tok/s claims online are ALL MoE models** that only activate 3-4B params per token.

**Key formula** (confirmed by 20 rounds of benchmarking):
```
aggregate tok/s ≈ bandwidth / (2 × active_params_bytes) × n_concurrent
```
Where active_params_bytes = active_params × bits_per_param / 8.

---

<a id="part-2"></a>
## PART 2: KNOWN BUGS AFFECTING PERFORMANCE

### Bug 1: CUDA 13.0 Performance Penalty (CRITICAL)

Your system runs CUDA 13.0. On Blackwell:
- **CUDA 13.x MMQ kernels SEGFAULT**
- **CUDA 13.x cuBLAS is 30-40% SLOWER** than CUDA 12.8

| Configuration      | Prompt (t/s) | Generation (t/s) |
|--------------------|-------------|-------------------|
| CUDA 12.8 + MMQ    | 5,611       | 211               |
| CUDA 12.8 + cuBLAS | 989         | 152               |
| CUDA 13.1 + cuBLAS | 772         | 121               |
| CUDA 13.1 + MMQ    | SEGFAULT    | SEGFAULT          |

**Fix**: CUDA 12.8 installed at `/usr/local/cuda-12.8`. llama.cpp built against it.

**Additional note**: Use `CMAKE_CUDA_ARCHITECTURES=121a-real` (not `120-real`) — fixes CUDA 13.0 nvcc -O3 correctness bug on SM_121. The `-real` suffix disables PTX fallback, `-a` suffix is required for SM_121.

Source: https://zenn.dev/toki_mwc/articles/rtx5090-blackwell-cuda-toolkit-trap-llama-cpp

### Bug 2: Ollama Tied Embeddings CPU Fallback (PR #14804)

Ollama forces the token embedding tensor (~2.6 GB) onto CPU on unified memory systems. The GPU idles while a single CPU thread does the final logit multiply. **KNOWN BUG, still unmerged.**

- Impact: 10-30% performance loss
- Source: https://github.com/ollama/ollama/pull/14804

### Bug 3: Flash Attention Hangs on Gemma 4 31B (Issue #15350)

FA hangs with prompts >3-4K tokens. Gemma 4 has hybrid attention (50 sliding-window + 10 global layers with different head dims 256 vs 512). Partially fixed in Ollama 0.20.4.

Source: https://github.com/ollama/ollama/issues/15350

### Bug 4: Ollama GPU Not Actually Used (Issue #15237)

42 comments reporting Gemma 4 shows 100% GPU but actually runs on CPU.

Source: https://github.com/ollama/ollama/issues/15237

### Bug 5: USB PD Power Negotiation Failure

GPU can get stuck at **513MHz / 14W / 30W**. Caused by USB Power Delivery negotiation failure between power brick and system.

**Fix**: Unplug the power brick from the WALL OUTLET (not the Spark), wait several minutes, reconnect. This resets the PD controller.

Source: https://forums.developer.nvidia.com/t/investigating-513mhz-cap-for-gpu/361296

### Bug 6: CPU Frequency Stuck at 1.5 GHz

Some units have CPU stuck at ~1.5 GHz despite performance governor. ASUS units achieve 4+ GHz. Sysbench shows **3x performance difference**.

**Fix**: Same PD reset as Bug 5. Also update BIOS from factory version.

Source: https://forums.developer.nvidia.com/t/gb10-cpu-frequency-capped-at-1-5-ghz-despite-performance-governor-and-policy-set-to-2-8-ghz/362513

### Bug 7: Headless Auto-Suspend

DGX Spark auto-suspends when running headless (no display). Widely reported.

**Fix applied**: `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`

Source: https://forums.developer.nvidia.com/t/nvidia-dgx-spark-automatically-suspends-when-running-headless-despite-power-config-changes/353777

### Bug 8: Draft Model Bug in llm-start Wrapper (FIXED)

`DRAFT="${2:-default}"` treated `""` as unset → silently loaded a draft model even when nospec was intended. Fixed to `DRAFT="$2"`. This caused misleading benchmark results in early rounds — "TCQ nospec" was actually spec+TCQ.

---

<a id="part-3"></a>
## PART 3: SYSTEM-LEVEL OPTIMIZATIONS

### Already Applied

| Optimization | Setting | Why |
|---|---|---|
| GPU clocks locked | 2418-3003 MHz | No ramp-up delay |
| Persistence mode | Enabled | GPU stays initialized |
| Swappiness | 1 | Model weights stay in RAM |
| THP | always + defer+madvise defrag | Better ARM64 mTHP, fewer page faults |
| NVMe scheduler | none | Lowest latency |
| NVMe read-ahead | 4096 KB | Faster model loading |
| Dirty ratio | 10/5 | Better I/O |
| Network buffers | 128MB | Ready for 200GbE |
| Suspend disabled | Masked all sleep targets | No headless suspend |
| Unnecessary services | Disabled | More CPU/memory headroom |
| memlock limits | unlimited | Models can lock memory |
| Unified memory env | GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 | Correct memory reporting |
| CUDA lazy loading | CUDA_MODULE_LOADING=LAZY | Faster startup |
| Boot service | inference-optimize.service | All settings survive reboot |
| cgroup limits | MemoryHigh=110G, MemoryMax=124G | Prevents OOM kill of TCQ buffers |

### Still Recommended

**Kernel boot parameters** (add to GRUB cmdline for maximum performance):
```
# /etc/default/grub → GRUB_CMDLINE_LINUX_DEFAULT:
cpufreq.default_governor=performance iommu.passthrough=1 transparent_hugepage=always
```

**Drop caches before loading model**:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

---

<a id="part-4"></a>
## PART 4: LLAMA.CPP OPTIMIZED BUILD & CONFIGURATION

### Build (Current — April 11, 2026)

```bash
cd ~/llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_NATIVE=ON \
  -DGGML_RPC=ON \
  -DBLACKWELL_NATIVE_FP4=1 \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc

cmake --build build --config Release -j$(nproc)
```

**Critical build notes:**
- **Use `121a-real`** (not `120-real`) — `121a` fixes nvcc -O3 correctness bug on SM_121
- **`-DBLACKWELL_NATIVE_FP4=1`** — enables MXFP4_MOE native quantization support
- **NEVER use `-DGGML_CUDA_FORCE_CUBLAS=ON`** — 5-6x slowdown
- **Use CUDA 12.8, NOT 13.x** — MMQ kernels segfault on 13.x

Binary at: `/home/bbanz90/llama.cpp/build/bin/llama-server`

### Optimal Runtime Flags (validated across 20 rounds)

```bash
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=64
export GGML_CUDA_GRAPH_OPT=1

~/llama.cpp/build/bin/llama-server \
  -m model.gguf \
  -ngl 999 \              # offload ALL layers to GPU
  -fa on \                # flash attention (REQUIRED)
  --no-mmap \             # MANDATORY — 10x faster loading
  --no-warmup \           # skip empty forward pass at startup
  -ub 4096 -b 4096 \      # micro-batch and batch size
  -ctk q4_0 -ctv q4_0 \  # q4_0 KV cache (optimal for long-ctx with -kvu)
  -c 65536 \              # context length
  -np 64 \                # parallel slots (key throughput lever — see Part 11)
  -t 8 \                  # threads (bandwidth-bound, t>8 has no effect)
  --cache-reuse 256 \     # KV prefix cache
  -kvu \                  # unified KV pool (REQUIRED for long-ctx)
  --host 0.0.0.0 \
  --port 8001
```

**Critical runtime notes:**
- **`--no-mmap` is MANDATORY** — without it, loading takes 110s vs 10s for GPT-OSS-120B
- **`-kvu`** — unified KV pool. Required for long-ctx (all np slots share the pool). Costs ~26% on short-ctx throughput (232 vs 316 tok/s @ n=32 for gpt-oss np=32). Without it, slot ctx = total_ctx/np, so at np=32 ctx=64k each slot only gets 2048 tokens.
- **`q4_0 KV`** — best for long-ctx. Was q8_0 for short-ctx-only configs.
- **`-t 8`** — bandwidth-bound workloads. t=16 tested and reverted, identical results.
- **`CUDA_DEVICE_MAX_CONNECTIONS=64`** — critical for multi-slot throughput
- **`GGML_CUDA_GRAPH_OPT=1`** — CUDA graph optimization, ~5% throughput improvement

### llama.cpp vs Ollama Performance Gap

| Model | llama.cpp (t/s) | Ollama (t/s) | Gap |
|---|---|---|---|
| GPT-OSS-120B MXFP4 | 58 gen | 41 gen | +41% |
| Gemma 3 27B | 14-15 gen | 11-12 gen | +25% |
| Gemma 4 31B Q8 (RTX 6000) | 42 gen | 29 gen | +45% |

---

<a id="part-5"></a>
## PART 5: MODEL SELECTION STRATEGY

### The Core Principle: Active Params Determine Speed

```
Throughput ∝ Memory Bandwidth / Active Parameters Per Token
```

| Model type | Active params | Relative speed |
|---|---|---|
| Gemma4-26B MoE | 4B | Fastest |
| gpt-oss-120b MoE | ~5B | Very fast |
| Qwen3.5-122B MoE | 10B | Fast |
| Llama 4 Maverick MoE | 17B | Fast |
| Dense 120B | 120B | Very slow |

**MoE models have ALL the knowledge of a large model but only activate a tiny fraction per token.** Quality approaches the full model size; speed approaches the active param count.

### Dense Models — Single GB10

| Model | Q8 tok/s | Q4 tok/s | Notes |
|---|---|---|---|
| 7-8B | 30-40 | 50-70 | Comfortable |
| 14B | 15-25 | 25-40 | Good |
| 27-32B | 5.5-8 | 10-12 | Usable but slow |
| 70B | 3-5 | 5-8 | Fits, sluggish |

### MoE Models — Single GB10 (THE SWEET SPOT)

| Model | Active Params | Measured tok/s | Config |
|---|---|---|---|
| Gemma4-26B Q4KM | 4B | **474.8** short / **233** longctx | np=96 ctx=64k |
| gpt-oss-120b IQ4NL | ~5B | **277** short / **241** longctx | np=64 ctx=64k |
| Qwen3.5-122B Q4KM | 10B | 83 short / 74 longctx | np=16 |

### Cross-architecture spec decoding: DON'T

Tested Gemma-E2B as draft for Qwen3.5-122B: **−24% at n=1, −18% longctx.** Cross-arch spec catastrophic — draft tokens get rejected constantly, net overhead. Only same-arch drafts work.

---

<a id="part-6"></a>
## PART 6: KV CACHE QUANTIZATION

### Original benchmarks (Nemotron-30B, DGX Spark community)

| Context | f16 KV | q8_0 KV | q4_0 KV |
|---------|--------|---------|---------|
| 8K | 14.7 | ~14.3 | 14.2 |
| 64K | 13.3 | ~12.8 | 8.6 |

### Validated findings from 20 rounds (gpt-oss-120b, Gemma4-26B)

**`q4_0` with `-kvu` is the optimal setting for long-context:**

- q4_0 KV + -kvu: best long-ctx results across all rounds
- q8_0 tested and reverted — no benefit with -kvu
- cache-reuse 512 tested: identical to 256 (unsupported in -kvu mode)

**Recommendations:**
- **Long-ctx workloads**: `-ctk q4_0 -ctv q4_0 -kvu` — required
- **Short-ctx only**: `-ctk q8_0 -ctv q8_0` without -kvu — slightly higher throughput
- `-kvu` costs ~26% on short-ctx (gpt-oss: 316→232 tok/s @ n=32) but enables full context for all slots

---

<a id="part-7"></a>
## PART 7: BACKEND COMPARISON

### Benchmarks on GB10

| Model | Backend | Quant | Prompt (t/s) | Generation (t/s) |
|---|---|---|---|---|
| Qwen3 14B | TRT-LLM | NVFP4 | 5,929 | 22.71 |
| Llama 3.1 8B | TRT-LLM | NVFP4 | 10,257 | 38.65 |
| GPT-OSS-20B (MoE) | llama.cpp | MXFP4 | 3,670 | **82.74** |
| GPT-OSS-120B (MoE) | llama.cpp | MXFP4 | 1,725 | **55.37** |
| Qwen3.5 35B (MoE) | vLLM | FP8 | 3,080 | 35.75 |

### MXFP4 vs Q4_K_M (validated Round 17)

**MXFP4_MOE ≈ Q4_K_M (within 2%)** — Blackwell-native format provides zero throughput advantage over standard Q4_K_M on llama.cpp. Use Q4_K_M for better compatibility.

### When to Use Each

| Use Case | Best Backend |
|---|---|
| Single-user interactive chat | llama.cpp |
| Maximum generation throughput | llama.cpp + np scaling |
| Long-context / prefill-heavy | TensorRT-LLM |
| Multi-user serving / concurrency | vLLM or llama.cpp with high np |
| Easiest setup | Ollama |

---

<a id="part-8"></a>
## PART 8: POWER, THERMAL & FIRMWARE

### Power Budget

| Component | Power |
|---|---|
| Total system peak | 240W (USB-C PSU) |
| GB10 SoC (GPU+CPU) | 140W TDP |
| GPU max observed | ~120W |
| ConnectX-7 NIC | ~18W (0W if cable unplugged, Jan 2026 firmware) |

**No user-controllable power limits.** `nvidia-smi -pl` shows N/A.

### Thermal Management

**Fan control is NOT user-accessible.** Firmware-controlled only.

**OEM cooling comparison** (StorageReview):

| OEM | GPU Peak | CPU Peak | Best? |
|---|---|---|---|
| Acer | 68C | 74.6C | YES |
| Founders | 80-82C | 87-88C | Worst |
| Dell | 80C | 87C | Middle |
| MSI EdgeXpert | ~75C | ~80C | 5-10% better perf |

**Temperature thresholds:**
- Normal: 68-82C GPU
- Throttle: GPU drops to 513MHz
- CPU shutdown: ~95C

### Firmware (Current: Driver 580.142, UEFI 1.107.26)

```bash
# Check firmware
sudo nvidia-smi --query-gpu=vbios_version --format=csv
# Update via DGX Dashboard at http://localhost:11000
```

---

<a id="part-9"></a>
## PART 9: DUAL GB10 SETUP (200GbE)

### Network Architecture

The ConnectX-7 has a **unique dual-MAC topology**: two separate PCIe Gen5 x4 links, each providing ~100Gbps. Both "twins" must be used for full 200Gbps.

### Setup Steps

**1. Identify interfaces:**
```bash
ibdev2netdev
# rocep1s0f1 (RoCE twin 1) + roceP2p1s0f1 (RoCE twin 2) = data plane
# enp1s0f1np1 = control plane (SSH, coordination)
```

**2. Static IPs:**
```bash
# Node 1:
sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 mtu 9000 up

# Node 2:
sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 mtu 9000 up
```

**3. Validate bandwidth:**
```bash
# Node 1: ib_send_bw -d rocep1s0f1 --report_gbits
# Node 2: ib_send_bw -d rocep1s0f1 192.168.100.10 --report_gbits
# Expect: ~93.3 Gbps per link
```

### llama.cpp RPC (Quick Setup)

```bash
# Node 2 (compute):
~/llama.cpp/build/bin/rpc-server -H 0.0.0.0 -p 50052 -c

# Node 1 (main):
~/llama.cpp/build/bin/llama-server \
  -m model.gguf -ngl 999 --rpc 192.168.100.11:50052 \
  -fa on --no-mmap
```

RPC splits the model across both nodes — doubles effective VRAM (256GB) and bandwidth (~546 GB/s).

### vLLM with Tensor Parallelism

```bash
# NCCL environment (both nodes):
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1  # BOTH twins for full bandwidth

# Ray cluster:
# Node 1: ray start --head --port=6379
# Node 2: ray start --address=192.168.100.10:6379

# vLLM:
vllm serve model --tensor-parallel-size 2 --distributed-executor-backend ray
```

### Dual-Node Benchmarks

| Model | Framework | Quant | Nodes | Generation (t/s) |
|---|---|---|---|---|
| Qwen3-235B-A22B | vLLM | NVFP4 | 2 | 11.73 |
| GPT-OSS-120B | vLLM | MXFP4 | 2 | 75.96 |
| Qwen3-Coder-Next | SGLang | FP8 | 2 | 60 |

**GPUDirect RDMA is NOT supported on GB10** — unified memory has no separate GPU memory to DMA into. All inter-node traffic goes through host memory staging.

### TCP Tuning for 200GbE

```bash
# /etc/sysctl.d/99-200gbe.conf
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 1048576 134217728
net.ipv4.tcp_wmem = 4096 1048576 134217728
net.core.netdev_max_backlog = 250000
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
net.ipv4.tcp_slow_start_after_idle = 0
```

---

<a id="part-10"></a>
## PART 10: BOOT OPTIMIZATION SCRIPT

Saved at `~/inference-stack/inference-optimize.sh`:

```bash
#!/bin/bash
# DGX Spark Inference Optimization - Run Once or At Boot

# GPU
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 2418,3003

# CPU
sudo cpupower frequency-set -g performance

# Memory
sudo sysctl -w vm.swappiness=1
sudo sysctl -w vm.dirty_ratio=10
sudo sysctl -w vm.dirty_background_ratio=5
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# Prevent suspend
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

# I/O
echo none | sudo tee /sys/block/nvme0n1/queue/scheduler
echo 4096 | sudo tee /sys/block/nvme0n1/queue/read_ahead_kb

# Environment
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=64
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export GGML_CUDA_GRAPH_OPT=1

# Drop caches
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

echo "Optimizations applied."
```

---

<a id="part-11"></a>
## PART 11: BENCHMARK RESULTS — ALL 20 ROUNDS

### Summary of All Rounds

| Rounds | Focus |
|---|---|
| R1–R10 | gpt-oss-120b short-ctx: np scaling, spec decoding, KV types, ctx sizes |
| R11–R15 | Long-context campaign: -kvu flag, np vs ctx tradeoffs, prompt length effects |
| R16 | New models baseline: Gemma4-26B MXFP4, Qwen3.5-122B, gpt-oss, Nemotron, Gemma4-31B |
| R17 | Gemma4-26B np ceiling (48/64), quant compare, gpt-oss np=32 |
| R18 | Gemma4-26B np=96/128 ceiling, gpt-oss np=48/64 |
| R19 | gpt-oss np=96, ctx=32k effects |
| R20 | gpt-oss np=128 ceiling, ctx=32k np=96 |

---

### ALL-TIME RECORDS (single GB10, llama.cpp)

| Metric | Value | Config |
|---|---|---|
| **Short throughput** | **474.8 tok/s** | Gemma4-26B Q4KM np=96 n=96 |
| **Long-ctx throughput** | **241.0 tok/s** | gpt-oss-120b np=64 n=64 |
| **Single-request longctx latency** | **55.1 tok/s** | Gemma4-26B ctx=32k np=64 n=1 |

---

### Gemma4-26B Q4KM — np scaling (ctx=64k, q4_0, -kvu)

| np | Short n=np | Longctx best | n=1 longctx |
|---|---|---|---|
| np=16 | 254.8 | 171.8 (n=16) | 49.2 |
| np=32 | 359.7 | 211.4 (n=32) | 31.1 |
| np=48 | 411.0 | 228.0 (n=48) | 26.1 |
| np=64 | 455.8 (n=64) | 233.3 (n=64) | 22.4 |
| np=96 | **474.8** (n=96) | 233.3 (n=64) | ~18 |
| np=128 | CRASH at n=128 | — | — |

**Short scales +20 tok/s per +16 slots (slowing above np=64). Longctx ceiling: ~233 tok/s at np=64+.**

---

### gpt-oss-120b IQ4NL — np scaling (ctx=64k, q4_0, -kvu)

| np | Short n=np | Longctx best n | n=1 longctx |
|---|---|---|---|
| np=16 | 154.2 (n=16) | 141.3 (n=16) | 46.3 |
| np=32 | 232.7 (n=32) | 188.9 (n=32) | ~40 |
| np=48 | 277.3 (n=48) | 218.4 (n=48) | ~34 |
| np=64 | 277.3 (n=64) | **241.0** (n=64) | ~32 |
| np=96 | 353.4 (n=96) | 238.7 (n=64) | 29.2 |
| np=128 | **358.8** (n=128) | 235.6 (n=64) | 26.2 |

**gpt-oss longctx PEAKS at np=64. Short bandwidth ceiling ~360 tok/s (np=128 adds only +5 vs np=96). Production: np=64.**

---

### ctx=32k effect

| Model | Config | n=1 longctx | n=32 longctx | Crash |
|---|---|---|---|---|
| Gemma4-26B | ctx=64k np=64 | 22.4 | 233.3 | n>64 |
| Gemma4-26B | ctx=32k np=64 | **55.1** (+2.46×) | 212.5 (−9%) | n=64 (exact fill) |
| Gemma4-26B | ctx=32k np=96 | 27.1 (−39% vs np=64!) | 213.0 | n=48 |
| gpt-oss | ctx=32k np=32 | ~40 | 178.1 (−5.7%) | — |

**ctx=32k benefit only valid for Gemma4-26B at np=64. At np=96 the benefit disappears (96 idle slots eat the freed bandwidth). gpt-oss (57GB model) gets no benefit — model weight dominates.**

---

### R1–R15 Key Results (gpt-oss short-ctx campaign)

| n | tok/s | Config |
|---|---|---|
| n=1 | 56.3 | np=4 |
| n=4 | 140.3 | np=4 |
| n=8 | 182.3 | np≥8 |
| n=16 | 216.5 | np=24 |
| **n=32** | **316.0** | **np=32** |

**np=32 zero-queue effect**: when np=n, no user ever waits for a slot → GPU fully saturated → non-linear throughput jump (212 at np=16 → 316 at np=32 for n=32).

**Long-ctx (R11–R15) optimal: np=16 ctx=64k** → 78.3 tok/s at n=32 (without -kvu on Gemma-class models).

---

### Key Discoveries (all rounds)

- **MoE active params determine throughput**: Gemma4-26B (4B active) → 474 tok/s; gpt-oss (5B active) → 353 tok/s short; Qwen (10B active) → 83 tok/s
- **np determines ceiling**: More slots = higher batch throughput but worse single-request latency (more idle KV slots consume bandwidth)
- **np=n zero-queue effect**: When all slots are full simultaneously, throughput jumps non-linearly (+40% for gpt-oss at n=32 np=32)
- **n=128 is a hard OOM wall**: Both Gemma4-26B and gpt-oss crash at 128 concurrent requests regardless of np or ctx
- **MXFP4_MOE ≡ Q4_K_M**: Blackwell-native format provides zero throughput advantage in llama.cpp
- **Cross-arch spec catastrophic**: Gemma draft on Qwen base = −24% throughput. Same-arch only.
- **t=4 ≡ t=8**: All models — bandwidth-bound, thread count above 4 irrelevant
- **Longctx ceiling gpt-oss**: 241 tok/s at np=64. Going higher (np=96/128) degrades it.
- **ctx=32k best use**: Gemma4-26B np=64, bounded-ctx workloads where n≤32 needed (55.1 tok/s n=1 vs 22.4)

---

<a id="part-12"></a>
## PART 12: CURRENT MODEL INVENTORY

Located at `/home/bbanz90/models/`

| Model | Size | Best For |
|---|---|---|
| **gpt-oss-120b IQ4NL** | 60GB | Best quality — main production model |
| **Qwen3.5-122B abliterated Q4KM** | 70GB | Uncensored / unrestricted |
| **Gemma4-26B Q4KM** | 16GB | Maximum throughput (474 tok/s) |

**Total: ~146GB. Drive: 460GB free.**

All draft models deleted (confirmed useless — cross-arch spec doesn't work, same-arch spec not tested for these models).

---

<a id="part-13"></a>
## PART 13: NEW MODELS AVAILABLE (April 2026)

### Ranked by quality — fits on 1× or 2× GB10

| Rank | Model | SWE-bench | MMLU | Active / Total | GGUF size | 1×GB10 | 2×GB10 | Est. speed |
|---|---|---|---|---|---|---|---|---|
| 1 | **Kimi K2.5** | ~79% | 92.0 | 32B / 1T | ~240GB (IQ1.8) | ❌ | ⚠️ barely | ~200 tok/s |
| 2 | **MiniMax M2.5** | **80.2%** | 85% | 10B / 230B | ~101GB (IQ3) | ✅ | ✅ | ~350 tok/s |
| 3 | **GLM-5.1** | 77.8% | 96% | 40B / 744B | ~220GB (IQ2) | ❌ | ⚠️ tight | ~120 tok/s |
| 4 | **Qwen3.5-397B** | — | 81 overall | 17B / 397B | 116GB (IQ2) | ⚠️ tight | ✅ | ~220 tok/s |
| 5 | **gpt-oss-120b** *(have it)* | ~75% | ~85% | 5B / 120B | 60GB | ✅ | ✅ | **241 tok/s** ✓ |
| 6 | **DeepSeek V3.2** | 72-74% | ~88% | 37B / 671B | ~245GB (IQ2) | ❌ | ⚠️ barely | ~80 tok/s |
| 7 | **Llama 4 Maverick** | ~65% | 80.5 | 17B / 400B | 34–122GB | ✅ | ✅ | ~300 tok/s |
| 8 | **DeepSeek R2** | — | ~88% reasoning | 32B dense | ~20GB | ✅ | ✅ | ~180 tok/s |
| 9 | **Mistral Large 3** | — | solid | dense 123B | ~73GB | ✅ | ✅ | ~80 tok/s |
| 10 | **Qwen3.5-122B** *(have it)* | — | ~85% | 10B / 122B | 70GB | ✅ | ✅ | **83 tok/s** ✓ |
| 11 | **Gemma4-26B** *(have it)* | — | ~75% | 4B / 26B | 16GB | ✅ | ✅ | **474 tok/s** ✓ |

**Top picks for next download:**
1. **MiniMax M2.5** — best SWE-bench (80.2%), 10B active, 101GB, fits one GB10 comfortably
2. **Llama 4 Maverick IQ1.78** — 34GB, MoE, potentially very fast
3. **DeepSeek R2** — 20GB, best reasoning model, pairs well as secondary model
4. **Qwen3.5-397B IQ2** — 116GB, serious upgrade over current 122B, 17B active

---

<a id="part-14"></a>
## PART 14: PRODUCTION CONFIGS

### Config 1: Best Quality (gpt-oss-120b)
```bash
llm-start /home/bbanz90/llama.cpp/build/bin/llama-server \
  -m /home/bbanz90/models/openai_gpt-oss-120b-IQ4_NL/openai_gpt-oss-120b-IQ4_NL-00001-of-00002.gguf \
  -ngl 999 -fa on --no-mmap --no-warmup \
  -ub 4096 -b 4096 -ctk q4_0 -ctv q4_0 \
  -c 65536 -np 64 -t 8 --cache-reuse 256 -kvu \
  --host 0.0.0.0 --port 8001
```
**Results**: 277 tok/s short / **241 tok/s longctx** at n=64

### Config 2: Maximum Throughput (Gemma4-26B)
```bash
llm-start /home/bbanz90/llama.cpp/build/bin/llama-server \
  -m /home/bbanz90/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -ngl 999 -fa on --no-mmap --no-warmup \
  -ub 4096 -b 4096 -ctk q4_0 -ctv q4_0 \
  -c 65536 -np 96 -t 8 --cache-reuse 256 -kvu \
  --host 0.0.0.0 --port 8001
```
**Results**: **474.8 tok/s short** / 233 tok/s longctx at n=96

### Config 3: Best n=1 Latency (Gemma4-26B ctx=32k)
```bash
llm-start /home/bbanz90/llama.cpp/build/bin/llama-server \
  -m /home/bbanz90/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -ngl 999 -fa on --no-mmap --no-warmup \
  -ub 4096 -b 4096 -ctk q4_0 -ctv q4_0 \
  -c 32768 -np 64 -t 8 --cache-reuse 256 -kvu \
  --host 0.0.0.0 --port 8001
```
**Results**: **55.1 tok/s n=1** longctx (2.46× better than ctx=64k). Safe up to n=32 concurrency.

### Config 4: Low-Latency Single-User (gpt-oss)
```bash
# np=16: 46.3 tok/s n=1, 141 tok/s longctx batch
-c 65536 -np 16 ...
```

### Environment Variables (always set)
```bash
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=64
export GGML_CUDA_GRAPH_OPT=1
```

---

<a id="part-15"></a>
## PART 15: REALISTIC PERFORMANCE EXPECTATIONS

### Single GB10 — Measured Results

| Model | n=1 | n=8 | n=32 | n=64 | n=96 |
|---|---|---|---|---|---|
| **Gemma4-26B Q4KM np=96** | 55.0 short | 197.5 | 357.5 | — | **474.8** |
| **gpt-oss-120b np=64** | 54.2 short | 116.4 | 200.2 | **277.3** | — |
| **gpt-oss-120b longctx np=64** | 32 | 102 | 188 | **241.0** | — |
| **Gemma4-26B longctx np=64** | 22.4 | 136 | 233 | **233.3** | — |
| **Qwen3.5-122B np=16** | 25.8 | 65.1 | — | — | — |

### Dual GB10 — Expected (RPC or vLLM TP=2)

| Scenario | Expected tok/s | Notes |
|---|---|---|
| gpt-oss-120b TP=2 | ~400 short / ~400 longctx | Doubles bandwidth |
| Gemma4-26B TP=2 | ~700+ short | Overkill — one node already fast |
| MiniMax M2.5 (single) | ~350 tok/s | Fits one node |
| Qwen3.5-397B IQ2 (RPC) | ~220 tok/s | 116GB across both |
| Kimi K2.5 IQ1.8 (RPC) | ~200 tok/s | 240GB, needs both |
| Separate models per node | Independent | litellm routes between them |

---

<a id="part-16"></a>
## PART 16: ACTION PLAN

### Completed
- [x] llama.cpp built with CUDA 12.8 + SM_121a-real fix
- [x] 20 benchmark rounds complete
- [x] Production configs validated
- [x] Model inventory cleaned (deleted: MXFP4_MOE dup, E2B Q8_0, E2B NVFP4, E4B NVFP4, Qwen3.5-9B draft, Nemotron-120b, Gemma4-31B)
- [x] Cross-arch spec decoding confirmed useless
- [x] ctx=32k tradeoffs fully characterized
- [x] np scaling ceiling confirmed for all models
- [x] Open WebUI + LiteLLM stack running
- [x] GX10 #2 online

### Next Steps
- [ ] Download MiniMax M2.5 (~101GB) and benchmark
- [ ] Download Llama 4 Maverick IQ1.78 (~34GB) and benchmark
- [ ] Download DeepSeek R2 (~20GB) — reasoning model
- [ ] Consider Qwen3.5-397B IQ2 (116GB) as Qwen3.5-122B replacement
- [ ] Configure 2× GB10 RPC for models >128GB (Kimi K2.5, DeepSeek V3.2)
- [ ] Benchmark new models with same R16+ framework

---

<a id="part-17"></a>
## PART 17: ALL SOURCES

### NVIDIA Official
- DGX Spark Performance Blog: https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/
- DGX Spark Optimizations (Jan 2026): https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/
- DGX Spark Scaling Agents (Mar 2026): https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/
- DGX Spark Release Notes: https://docs.nvidia.com/dgx/dgx-spark/release-notes.html
- DGX Spark User Guide: https://docs.nvidia.com/dgx/dgx-spark/index.html
- DGX Spark Porting Guide: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html
- Grace Performance Tuning: https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html
- DGX Spark Playbooks: https://github.com/NVIDIA/dgx-spark-playbooks
- Gemma 4 DGX Spark Benchmarks: https://forums.developer.nvidia.com/t/gemma-4-day-1-inference-on-nvidia-dgx-spark-preliminary-benchmarks/365503
- Gemma 4 31B FP8 Dual-Node: https://forums.developer.nvidia.com/t/gemma-4-31b-on-dgx-spark-runtime-fp8-benchmarks-single-dual-node-tp-2/365814
- KV Cache Benchmarks: https://forums.developer.nvidia.com/t/kv-cache-quantization-benchmarks-on-dgx-spark-q4-0-vs-q8-0-vs-f16-llama-cpp-nemotron-30b-128k-context/365138
- DGX Spark Performance Thread: https://forums.developer.nvidia.com/t/dgx-spark-performance/356716
- Thermal Throttling: https://forums.developer.nvidia.com/t/dgx-spark-thermal-throttling/349647
- Thermal Test OEM Comparison: https://www.storagereview.com/review/nvidia-dgx-spark-thermal-test-how-oem-cooling-designs-stack-up
- Headless Suspend: https://forums.developer.nvidia.com/t/nvidia-dgx-spark-automatically-suspends-when-running-headless-despite-power-config-changes/353777
- 513MHz GPU Cap: https://forums.developer.nvidia.com/t/investigating-513mhz-cap-for-gpu/361296
- CPU 1.5GHz Cap: https://forums.developer.nvidia.com/t/gb10-cpu-frequency-capped-at-1-5-ghz-despite-performance-governor-and-policy-set-to-2-8-ghz/362513
- ConnectX-7 Networking: https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/
- Connecting Two Sparks: https://medium.com/@dorangao/connecting-two-dgx-spark-systems-via-200gb-s-roce-network-for-multi-node-gpu-training-50d67d3630a5

### Ollama
- DGX Spark Performance: https://ollama.com/blog/nvidia-spark-performance
- Tied Embeddings Bug: https://github.com/ollama/ollama/pull/14804
- FA Hang Gemma 4: https://github.com/ollama/ollama/issues/15350
- GPU Not Used: https://github.com/ollama/ollama/issues/15237
- Ollama vs llama.cpp Gap: https://github.com/ollama/ollama/issues/14861

### llama.cpp
- DGX Spark Discussion: https://github.com/ggml-org/llama.cpp/discussions/16578
- ARM Build Guide: https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/2_gb10_llamacpp_gpu/
- CUDA Graphs: https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/
- CUDA Toolkit Blackwell Trap: https://zenn.dev/toki_mwc/articles/rtx5090-blackwell-cuda-toolkit-trap-llama-cpp

### vLLM / Multi-Node
- vLLM for Two Sparks: https://build.nvidia.com/spark/vllm/stacked-sparks
- NCCL for Two Sparks: https://build.nvidia.com/spark/nccl/stacked-sparks
- vLLM SM121 PR: https://github.com/vllm-project/vllm/pull/38484
- LMSYS DGX Spark Review: https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/

### New Models (April 2026)
- MiniMax M2.5 (80.2% SWE-bench): https://automatio.ai/models/minimax-m2-5
- Kimi K2.5 architecture: https://help.apiyi.com/en/kimi-k2-5-paper-parameters-requirements-guide-en.html
- DeepSeek R2 (92.7% AIME): https://decodethefuture.org/en/deepseek-r2-explained/
- Llama 4 Maverick GGUF: https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF
- Qwen3.5-397B GGUF: https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF
- Best open source LLMs 2026: https://benchlm.ai/blog/posts/best-open-source-llm

### Community
- Dual Spark Cooling Cage: https://forums.developer.nvidia.com/t/dual-spark-ducted-cooling-cage/365302
- DGX Spark Community Playbooks: https://github.com/csabakecskemeti/dgx-spark-community-playbooks
