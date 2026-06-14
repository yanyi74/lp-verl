#!/usr/bin/env bash
# LP-GRPO v2: reliable Learning-Progress via dense revisit + advantage reweight.
#
# Two coupled components:
#   (1) LPRevisitSampler (data side): maintains a small active pool so prompts are
#       densely revisited -> double-EMA progress signal `dema` becomes reliable.
#       Drops mastered (p_t~1) and stuck (repeated 0/8) prompts (cooldown+revive),
#       refilling from reserve. This recovers wasted sigma=0 rollouts (sample eff.).
#   (2) lp_grpo_v2 advantage (algo side): A = A_GRPO * (1-p0)^gamma * f_prog(dema)
#       f_prog = 1 + alpha*tanh(k*dema)  [improving]
#              = 1 + beta *tanh(k*|dema|) [regressing, beta<alpha, still >1]
#       difficulty anchor (1-p0) is orthogonal to GRPO sigma (no redundant ZPD).
#
# Requires: data.dataloader_num_workers=0 (so the curriculum sampler can reorder).
#
# Usage:
#   On head node:   NODE_RANK=0 bash run_lp_grpo_v2.sh
#   On worker node: NODE_RANK=1 bash run_lp_grpo_v2.sh

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
HEAD_IP=${HEAD_IP:-10.144.203.206}
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

ACTOR_LR=${ACTOR_LR:-1e-5}
KL_LOSS_COEF=${KL_LOSS_COEF:-0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_N=${ROLLOUT_N:-8}
SP_SIZE=${SP_SIZE:-1}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-40}
TEST_FREQ=${TEST_FREQ:-20}

PROJECT_NAME=${PROJECT_NAME:-verl_lp_grpo_v2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-lp_grpo_v2_$(date +%Y%m%d_%H%M)}

TRAIN_FILE=${TRAIN_FILE:-/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/data/combined/filtered/strategy1_30k/mmrl14208_vppo16237_lp_grpo.parquet}
TEST_FILE=${TEST_FILE:-/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/data/geo3k/test.parquet}

# ----- LP-GRPO v2 advantage knobs:  w = (1-p0)^gamma * f_prog(dema) -----
LP_V2_GAMMA=${LP_V2_GAMMA:-0.5}      # difficulty anchor exponent
LP_V2_ALPHA=${LP_V2_ALPHA:-1.0}      # improving progress strength
LP_V2_BETA=${LP_V2_BETA:-0.4}        # regressing strength (<alpha, >0)
LP_V2_K=${LP_V2_K:-8.0}              # tanh saturation on dema
LP_EPS_P=${LP_EPS_P:-0.05}

# ----- LPRevisitSampler (SLIM: value-scored dense revisit) knobs -----
# No pool_size / revisit_ratio / cooldown / stuck_K: revisit, eviction, coverage
# all EMERGE from a single value score over the WHOLE dataset:
#   score = (1 + dema+) * sqrt(4 p_t (1-p_t))
#   eligibility gated by min_interval (revisit density) and max_visit (anti-overfit)
LP_MIN_INTERVAL=${LP_MIN_INTERVAL:-30}    # min steps between two visits of a prompt
LP_MAX_VISIT=${LP_MAX_VISIT:-5}           # HARD cap: retire a prompt after this many visits
LP_DEMA_ALPHA=${LP_DEMA_ALPHA:-0.4}       # dema EMA coefficient (progress smoothing)
LP_BASE_BETA=${LP_BASE_BETA:-0.2}         # base EMA coefficient (slow baseline)
LP_THETA=${LP_THETA:-0.05}                # progress deadzone (metrics only)

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
    algorithm.adv_estimator=lp_grpo_v2
    algorithm.use_kl_in_reward=False
    algorithm.lp_v2_gamma=${LP_V2_GAMMA}
    algorithm.lp_v2_alpha=${LP_V2_ALPHA}
    algorithm.lp_v2_beta=${LP_V2_BETA}
    algorithm.lp_v2_k=${LP_V2_K}
    algorithm.lp_eps_p=${LP_EPS_P}
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.image_key=images
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=64
    data.truncation='error'
    data.dataloader_num_workers=0
    data.sampler.class_path=pkg://verl.experimental.dataset.lp_revisit_sampler
    data.sampler.class_name=LPRevisitSampler
    "+data.lp_sampler.min_interval=${LP_MIN_INTERVAL}"
    "+data.lp_sampler.max_visit=${LP_MAX_VISIT}"
    "+data.lp_sampler.alpha=${LP_DEMA_ALPHA}"
    "+data.lp_sampler.beta=${LP_BASE_BETA}"
    "+data.lp_sampler.theta=${LP_THETA}"
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
