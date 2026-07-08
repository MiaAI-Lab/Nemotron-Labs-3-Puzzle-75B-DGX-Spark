# Nemotron-Labs-3-Puzzle-75B on DGX Spark

Serve [NVIDIA Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4) on a single **NVIDIA DGX Spark (GB10)** node using [vLLM](https://github.com/vllm-project/vllm) **0.24** inside Docker. The launcher applies the aarch64 / SM12.1 workarounds needed for stable NVFP4 MoE + Mamba inference on Blackwell Spark hardware.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>

| | |
|---|---|
| **Model** | 75.3B total / 9.3B active hybrid MoE (Mamba + MoE + Attention) |
| **Quantization** | NVFP4 weights, FP8 KV cache (default) |
| **Context** | 256k tokens (`max_position_embeddings=262144`) |
| **Decoding** | MTP speculative decoding (`k=3` default) |
| **API** | OpenAI-compatible on port **8888** |

## Why this repo

This is a minimal, Spark-tuned Docker recipe. `start.sh` handles:

- Resolving weights from the local Hugging Face cache (offline serving)
- Materializing symlinked custom model code (`.py`, `.json`, `.jinja`) so `trust_remote_code` loads correctly
- aarch64 NCCL and FlashInfer patches inside the container
- CUDA 13 / `sm_121a` environment variables for GB10
- Tool calling (`qwen3_coder`) and reasoning output (`nemotron_v3`)

A related layout with native `start.sh` / `stop.sh`, vendored `./model` weights, and additional tuning options is maintained separately as **Nemotron-Labs-3-Puzzle-75B** (same Spark tuning lineage).

## Requirements

### Hardware

- **1× NVIDIA DGX Spark (GB10)** — ~121 GB unified memory
- **~75 GB free disk** — ~50 GB model weights + ~21 GB Docker image + cache headroom

### Software

| Requirement | Notes |
|---|---|
| Docker with GPU support | `docker run --gpus all` must work |
| NVIDIA driver + container toolkit | Standard DGX Spark stack |
| Hugging Face cache | Model must be downloaded before first launch (`HF_HUB_OFFLINE=1` is set) |

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Optional

- `~/gpu-clear.sh` — if present, `start.sh` calls it during cleanup to free GPU memory before relaunching. The script is not required.

## Quick start

### 1. Download the model

```bash
pip install -U huggingface_hub
huggingface-cli download nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4
```

Weights land under `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/`.

### 2. Pull the vLLM image

```bash
docker pull vllm/vllm-openai:v0.24.0
```

### 3. Clone and launch

```bash
git clone https://github.com/MiaAI-Lab/Nemotron-Labs-3-Puzzle-75B-DGX-Spark.git
cd Nemotron-Labs-3-Puzzle-75B-DGX-Spark
chmod +x start.sh
./start.sh
```

Startup runs cleanup (stops prior containers `nemo-75b-vllm`, `puzzle_a4q`, `puzzle75b`), applies in-container patches, and starts vLLM in the background.

### 4. Verify

```bash
curl http://localhost:8888/v1/models
docker logs -f nemo-75b-vllm
```

Warmup can take several minutes on first boot while FlashInfer kernels compile.

### 5. Stop

```bash
docker stop nemo-75b-vllm
docker rm nemo-75b-vllm
```

Re-running `./start.sh` performs the same cleanup automatically.

## Configuration

All knobs are environment variables. Override them inline when calling `start.sh`:

```bash
SEQS=4 MAXLEN=131072 MOE_BACKEND=FLASHINFER_CUTLASS ./start.sh
```

| Variable | Default | Description |
|---|---|---|
| `KVD` | `fp8` | KV cache dtype (`--kv-cache-dtype`) |
| `SPEC` | `mtp` | Speculative decoding method; set to empty to disable |
| `SPEC_TOKENS` | `3` | MTP speculative token count |
| `SEQS` | `7` | `--max-num-seqs` (concurrent sequences) |
| `MAXLEN` | `262144` | `--max-model-len` (256k context) |
| `MOE_BACKEND` | `FLASHINFER_CUTLASS` | MoE execution backend |
| `EAGER` | `1` | `1` = `--enforce-eager` (CUDA graphs off; more stable on Spark) |

Fixed in `start.sh` (edit the script to change):

| Setting | Value |
|---|---|
| Container name | `nemo-75b-vllm` |
| Docker image | `vllm/vllm-openai:v0.24.0` |
| Port | `8888` |
| Tensor parallel | `1` |
| GPU memory util | `0.85` |
| Served model name | `Nemotron-75b-Puzzle` |

### Context and concurrency tuning

| Goal | Example |
|---|---|
| Shorter context, more headroom | `MAXLEN=131072 SEQS=12 ./start.sh` |
| Lower concurrency if OOM | `SEQS=1 EAGER=1 ./start.sh` |
| Disable MTP | `SPEC= ./start.sh` |
| Try CUDA graphs | `EAGER=0 ./start.sh` (may be less stable) |

## API examples

### Chat completion

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Nemotron-75b-Puzzle",
    "messages": [{"role": "user", "content": "Explain hybrid MoE in one paragraph."}],
    "max_tokens": 512
  }'
```

### Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8888/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Nemotron-75b-Puzzle",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

For coding agents, NVIDIA recommends passing `extra_body={"chat_template_kwargs": {"force_nonempty_content": True}}` on chat requests.

## Troubleshooting

| Symptom | Things to try |
|---|---|
| `no model snapshot found under .../hub/models--nvidia--...` | Run `huggingface-cli download nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4` first |
| Container exits immediately | `docker logs nemo-75b-vllm`; check driver/GPU visibility and disk space |
| OOM during warmup or inference | `SEQS=1 EAGER=1 ./start.sh`; reduce `MAXLEN` |
| Port already in use | Stop the old container: `docker rm -f nemo-75b-vllm` |
| API unreachable after launch | Wait for warmup; tail `docker logs -f nemo-75b-vllm` |
| MoE / FlashInfer errors on GB10 | Keep `MOE_BACKEND=FLASHINFER_CUTLASS` and `EAGER=1` (defaults) |

## What `start.sh` does

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Resolve HF cache snapshot for Puzzle-75B NVFP4           │
│ 2. Materialize symlinked model code files in snapshot         │
│ 3. Cleanup prior GPU/containers (optional gpu-clear.sh)       │
│ 4. docker run vllm/vllm-openai:v0.24.0                      │
│    ├─ NCCL lib symlink (aarch64)                              │
│    ├─ FlashInfer all-reduce + cuda_ipc patches              │
│    └─ vllm serve with MoE, Mamba, MTP, tool/reasoning APIs    │
└─────────────────────────────────────────────────────────────┘
```

## Project layout

```
.
├── start.sh    # Docker launcher for DGX Spark
└── README.md
```

## References

- [Model card — NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4)
- [Tech report — Compressing Hybrid MoE LLMs](https://arxiv.org/abs/2607.04371)
- [Nemotron 3 Super technical report](https://arxiv.org/abs/2604.12374)
- [vLLM](https://github.com/vllm-project/vllm)

## License

Model weights and use are governed by the [OpenMDW License Agreement v1.1](https://openmdw.ai/license/1-1/). See the [model README](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4) for benchmarks, safety information, and full documentation.

Deployment scripts in this repository are provided as-is for reproducibility on DGX Spark hardware.
