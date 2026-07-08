#!/usr/bin/env bash
# Launch NVIDIA Nemotron Lab Puzzle 75B (NVFP4) via vLLM in Docker.

set -uo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment)
# -----------------------------------------------------------------------------

# Model / inference
readonly KVD="${KVD:-fp8}"
readonly SPEC="${SPEC:-mtp}"
readonly SPEC_TOKENS="${SPEC_TOKENS:-3}"
readonly SEQS="${SEQS:-7}"
readonly MAXLEN="${MAXLEN:-262144}"
readonly MOE_BACKEND="${MOE_BACKEND:-FLASHINFER_CUTLASS}"
readonly EAGER="${EAGER:-1}"

# Container
readonly CONTAINER_NAME="nemo-75b-vllm"
readonly IMAGE="vllm/vllm-openai:v0.24.0"
readonly PORT=8888
readonly STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"

# Paths
readonly MODEL_HUB_ID="models--nvidia--NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4"
readonly HF_CACHE="${HOME}/.cache/huggingface"
readonly FLASHINFER_CACHE="${HOME}/.cache/flashinfer"

# -----------------------------------------------------------------------------
# Derived settings
# -----------------------------------------------------------------------------

EAGERFLAG=""
[[ "$EAGER" == "1" ]] && EAGERFLAG="--enforce-eager"

SPECARG=""
[[ "$SPEC" == "mtp" ]] && SPECARG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}'"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

resolve_snapshot() {
  local snap
  snap=$(ls -d "${HF_CACHE}/hub/${MODEL_HUB_ID}/snapshots"/*/ | head -1)
  if [[ -z "$snap" ]]; then
    echo "error: no model snapshot found under ${HF_CACHE}/hub/${MODEL_HUB_ID}" >&2
    exit 1
  fi
  echo "$snap"
}

materialize_symlinks() {
  local snap="$1"
  local f

  pushd "$snap" >/dev/null || exit 1
  for f in *.py *.json *.jinja; do
    if [[ -L "$f" ]]; then
      cp -L "$f" "$f.real" && mv "$f.real" "$f"
    fi
  done
  popd >/dev/null
}

cleanup() {
  docker rm -f "$CONTAINER_NAME" puzzle_a4q puzzle75b >/dev/null 2>&1 || true
}

container_setup_script() {
  local model_path="$1"

  cat <<EOF
rm -f /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null
ln -sf /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null
FAR=/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/flashinfer_all_reduce.py
sed -i 's/^except ImportError:/except Exception:/' "\$FAR" 2>/dev/null
CIPC=/usr/local/lib/python3.12/dist-packages/flashinfer/comm/cuda_ipc.py
sed -i 's/if lib_name in line:/if lib_name in line and "stub" not in line:/' "\$CIPC" 2>/dev/null
exec vllm serve ${model_path} \\
  --served-model-name Nemotron-75b-Puzzle Nemotron-75b-Puzzle --host 0.0.0.0 --port ${PORT} \\
  --trust-remote-code --tensor-parallel-size 1 \\
  --enable-expert-parallel --mamba-backend flashinfer \\
  --moe-backend ${MOE_BACKEND} --kv-cache-dtype ${KVD} \\
  --max-model-len ${MAXLEN} --max-num-seqs ${SEQS} --max-num-batched-tokens 8192 \\
  --gpu-memory-utilization 0.85 ${EAGERFLAG} ${SPECARG} \\
  --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --enable-auto-tool-choice \\
  --enable-prefix-caching
EOF
}

wait_for_ready() {
  local logs_pid ready=0 elapsed=0

  echo ""
  echo "Waiting for API at http://127.0.0.1:${PORT}/v1/models (container logs below) ..."
  echo ""

  docker logs -f --tail 0 "$CONTAINER_NAME" 2>&1 &
  logs_pid=$!

  cleanup_logs() {
    kill "$logs_pid" 2>/dev/null || true
    wait "$logs_pid" 2>/dev/null || true
  }
  trap cleanup_logs INT TERM

  while (( elapsed < STARTUP_TIMEOUT )); do
    if curl -sS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      cleanup_logs
      trap - INT TERM
      echo "" >&2
      echo "error: container ${CONTAINER_NAME} exited during startup" >&2
      docker logs --tail 80 "$CONTAINER_NAME" >&2 || true
      return 1
    fi
    sleep 3
    elapsed=$((elapsed + 3))
  done

  cleanup_logs
  trap - INT TERM

  if [[ "$ready" -ne 1 ]]; then
    echo "" >&2
    echo "error: API did not respond within ${STARTUP_TIMEOUT}s" >&2
    echo "Logs: docker logs -f ${CONTAINER_NAME}" >&2
    return 1
  fi

  echo ""
  echo "Server is ready."
  echo "  API:  http://127.0.0.1:${PORT}/v1"
  echo "  Test: curl http://127.0.0.1:${PORT}/v1/models"
  echo "  Logs: docker logs -f ${CONTAINER_NAME}"
  return 0
}

launch_container() {
  local model_path="$1"
  local setup_script

  setup_script=$(container_setup_script "$model_path")

  docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
    --memory 112g --memory-swap 112g --ulimit memlock=-1 --ulimit nofile=1048576 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
    --name "$CONTAINER_NAME" \
    -e TORCH_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
    -e MAX_JOBS=6 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
    -e LD_PRELOAD=/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcudart.so.13 \
    -e LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib \
    -e PATH=/usr/local/cuda-13/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --entrypoint bash "$IMAGE" \
    -lc "$setup_script"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
  local snap container_model_path rc

  snap=$(resolve_snapshot)
  container_model_path="/root/${snap#"$HOME"/}"

  materialize_symlinks "$snap"
  cleanup
  launch_container "$container_model_path"
  rc=$?

  if [[ "$rc" -ne 0 ]]; then
    echo "error: docker run failed (rc=${rc})" >&2
    exit "$rc"
  fi

  echo "launched ${CONTAINER_NAME} single-node kv=${KVD} moe=${MOE_BACKEND} eager=${EAGER} spec=${SPEC}/${SPEC_TOKENS}"
  wait_for_ready
}

main "$@"
