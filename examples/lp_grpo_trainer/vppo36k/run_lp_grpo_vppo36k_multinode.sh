#!/usr/bin/env bash
# LP-GRPO final algorithm on the vppo36k dataset (pure VPPO_ViRL39K, 36,581 samples,
# schema-aligned to the 30k combined file). Pair with run_grpo_baseline_vppo36k_multinode.sh
# as the apples-to-apples baseline (same data / batch / model / LR / N).
#
# Formula:
#   w = (1 - p_0)^γ × (1 + λ·|Δp|) × (4·p_t·(1-p_t))^β
#       └─ difficulty ┘   └ progress ┘    └── signal/allocation ──┘
#
# Three mechanisms vs baseline GRPO:
#   1) |Δp| replaces KL: linear, bounded, numerically stable, robust to N=8 noise.
#   2) Two-sided ZPD: signal-aware resource allocation — mastered (p_t≈1) and
#      full-fail (p_t≈0) get auto-compressed; plateau-medium (p_t≈0.5) retains.
#   3) Per-step EMA on p_0 (α=0.15): replaces epoch-end整批覆盖, lets |Δp| accumulate.
#
# Rollout-side adaptive_n is INTENTIONALLY OFF here (algo-only ablation).
#
# Usage:
#   On head node:   NODE_RANK=0 bash run_lp_grpo_vppo36k_multinode.sh
#   On worker node: NODE_RANK=1 bash run_lp_grpo_vppo36k_multinode.sh

set -xeuo pipefail

source /root/verl_env/bin/activate
export PATH=/root/verl_env/bin:$PATH

_TRITON_BIN=/root/verl_env/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=${_TRITON_BIN}/ptxas
export TRITON_CUOBJDUMP_PATH=${_TRITON_BIN}/cuobjdump
export TRITON_NVDISASM_PATH=${_TRITON_BIN}/nvdisasm
unset _TRITON_BIN
export LD_LIBRARY_PATH=/root/verl_env/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export http_proxy="${http_proxy:-http://10.140.24.177:3128}"
export https_proxy="${https_proxy:-http://10.140.15.68:3128}"
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"mruvFouNUCTig4mwbR0qJ"}

########################### multi-node bootstrap ###########################
NODE_RANK=${NODE_RANK:-0}
HEAD_IP=${HEAD_IP:-10.144.200.102}
RAY_PORT=${RAY_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}

NNODES=${NNODES:-2}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

if [ "${NODE_RANK}" = "0" ]; then
    echo "[head] starting Ray head on ${HEAD_IP}:${RAY_PORT}"
    ray stop --force >/dev/null 2>&1 || true
    pkill -9 -f 'raylet|gcs_server|plasma_store|redis-server|ray::' >/dev/null 2>&1 || true
    sleep 2
    rm -rf /tmp/ray /tmp/ray_current_cluster 2>/dev/null || true
    ray start --head \
              --node-ip-address=${HEAD_IP} \
              --port=${RAY_PORT} \
              --dashboard-host=0.0.0.0 \
              --dashboard-port=${DASHBOARD_PORT} \
              --num-gpus=${NDEVICES_PER_NODE}

    echo "[head] waiting for ${NNODES} nodes to join the Ray cluster..."
    deadline=$(( $(date +%s) + 1800 ))
    while :; do
        n_alive=$(python3 -c 'import ray; ray.init(address="auto", logging_level="ERROR"); print(sum(1 for n in ray.nodes() if n["Alive"]))' 2>/dev/null || echo 0)
        if [ "${n_alive}" -ge "${NNODES}" ]; then
            echo "[head] all ${NNODES} nodes joined."
            break
        fi
        if [ "$(date +%s)" -ge "${deadline}" ]; then
            echo "[head] timeout waiting for workers (have ${n_alive}/${NNODES})." >&2
            exit 1
        fi
        echo "[head] only ${n_alive}/${NNODES} alive, sleep 5s..."
        sleep 5
    done
else
    echo "[worker rank=${NODE_RANK}] connecting to head ${HEAD_IP}:${RAY_PORT}"
    ray stop --force >/dev/null 2>&1 || true
    rm -rf /tmp/ray /tmp/ray_current_cluster 2>/dev/null || true
    ray start --address=${HEAD_IP}:${RAY_PORT} \
              --num-gpus=${NDEVICES_PER_NODE} \
              --block
    exit 0
fi

########################### user-adjustable ###########################
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
MODEL_PATH=${MODEL_PATH:-/mnt/tidal-alsh01/dataset/perceptionVLM/code_guomingxiao/model/Qwen3-VL-8B-Instruct}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-28672}

ACTOR_LR=${ACTOR_LR:-5e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_N=${ROLLOUT_N:-8}
SP_SIZE=${SP_SIZE:-1}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-40}
TEST_FREQ=${TEST_FREQ:-20}

PROJECT_NAME=${PROJECT_NAME:-verl_lp_grpo_vppo36k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-lp_grpo_zpd_ema_vppo36k_2node_$(date +%Y%m%d_%H%M)}

TRAIN_FILE=${TRAIN_FILE:-/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/data/vppo/VPPO_ViRL39K_train_lp_grpo_aligned.parquet}
TEST_FILE=${TEST_FILE:-/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/data/geo3k/test.parquet}

# ----- LP-GRPO algorithm-side knobs (final v: |Δp| + two-sided ZPD + EMA) -----
LP_GAMMA=${LP_GAMMA:-0.5}                 # difficulty awareness exponent
LP_LAMBDA=${LP_LAMBDA:-3.0}               # progress amplification (linear in |Δp|)
LP_W_MAX=${LP_W_MAX:-5.0}                 # unused now (kept for backward compat / metric)
LP_EPS_P=${LP_EPS_P:-0.05}                # numerical clip for KL metric
LP_ZPD_STRENGTH=${LP_ZPD_STRENGTH:-1.0}   # two-sided ZPD: signal-aware resource allocation
LP_P0_EMA_ALPHA=${LP_P0_EMA_ALPHA:-0.15}  # per-step EMA on p_0 (in advantage fn)
# Bucket factors all 1.0 (no extra steering) — three axes do the work.
LP_BREAKTHROUGH_BOOST=${LP_BREAKTHROUGH_BOOST:-1.0}
LP_PROGRESS_BOOST=${LP_PROGRESS_BOOST:-1.0}
LP_REGRESSING_PENALTY=${LP_REGRESSING_PENALTY:-1.0}

DEFAULT_SYSTEM_PROMPT='You FIRST think about the reasoning process step by step as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <thought> </thought> tags. The final answer MUST BE put in \boxed{}.'
SYSTEM_PROMPT=${SYSTEM_PROMPT:-${DEFAULT_SYSTEM_PROMPT}}

LOG_DIR=${LOG_DIR:-$(dirname "$0")/logs}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$(dirname "$0")/rollout_data/${EXPERIMENT_NAME}}
########################### end user-adjustable ###########################

########################### logging ###########################
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "=== Run started at $(date) ==="
echo "Log file: ${LOG_FILE}"

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}

case "${DEVICE}" in
    gpu)
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=lp_grpo
    algorithm.use_kl_in_reward=False
    algorithm.lp_gamma=${LP_GAMMA}
    algorithm.lp_lambda=${LP_LAMBDA}
    algorithm.lp_w_max=${LP_W_MAX}
    algorithm.lp_eps_p=${LP_EPS_P}
    algorithm.lp_zpd_strength=${LP_ZPD_STRENGTH}
    algorithm.lp_p0_ema_alpha=${LP_P0_EMA_ALPHA}
    algorithm.lp_breakthrough_boost=${LP_BREAKTHROUGH_BOOST}
    algorithm.lp_progress_boost=${LP_PROGRESS_BOOST}
    algorithm.lp_regressing_penalty=${LP_REGRESSING_PENALTY}
    algorithm.lp_normalize_w=True
    algorithm.lp_adaptive_n=False
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.image_key=images
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=64
    data.truncation='error'
    data.dataloader_num_workers=64
    "+data.system_prompt='${SYSTEM_PROMPT}'"
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","swanlab"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
)

if [ "${DEVICE}" = npu ]; then
    EXTRA=(
        actor_rollout_ref.actor.use_torch_compile=False
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
        actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    )
else
    EXTRA=(
        actor_rollout_ref.model.use_fused_kernels=False
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
        actor_rollout_ref.rollout.enforce_eager=False
        actor_rollout_ref.rollout.free_cache_engine=True
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
