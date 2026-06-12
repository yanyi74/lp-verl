#!/usr/bin/env bash
# LP-GRPO (gamma=1.0 ablation) | vision | vLLM rollout | FSDP training | GPU/NPU
# Change vs baseline: LP_GAMMA 0.5 -> 1.0 (stronger difficulty weighting)
# Qwen3-VL-8B with LP-GRPO. Train parquet must have:
#   extra_info["index"]: unique int per prompt; p_zero: baseline pass rate float.

set -xeuo pipefail
# The NVIDIA container sets TRITON_PTXAS_PATH to the system CUDA 13.0 ptxas, but Triton 3.4.0
# only knows CUDA 10-12. tmux inherits these vars; fresh SSH sessions don't (hence tmux-only failure).
# Override to use Triton's bundled CUDA 12.8 tools so ptx_get_version works correctly.
_TRITON_BIN=/root/verl_env/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=${_TRITON_BIN}/ptxas
export TRITON_CUOBJDUMP_PATH=${_TRITON_BIN}/cuobjdump
export TRITON_NVDISASM_PATH=${_TRITON_BIN}/nvdisasm
unset _TRITON_BIN
# Prepend verl_env torch lib so libtorch_cuda.so is loaded from this venv rather than the
# system NVIDIA container torch (2.9.0), preventing CUDAPluggableAllocator conflicts.
export LD_LIBRARY_PATH=/root/verl_env/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export http_proxy="${http_proxy:-http://10.140.24.177:3128}"
export https_proxy="${https_proxy:-http://10.140.15.68:3128}"
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"mruvFouNUCTig4mwbR0qJ"}   # set your key or export before running

########################### user-adjustable ###########################
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
MODEL_PATH=${MODEL_PATH:-/mnt/tidal-alsh01/dataset/perceptionVLM/code_guomingxiao/model/Qwen3-VL-8B-Instruct}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-18432}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-28672}

ACTOR_LR=${ACTOR_LR:-1e-5}
KL_LOSS_COEF=${KL_LOSS_COEF:-0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-8}
SP_SIZE=${SP_SIZE:-1}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-200}
TEST_FREQ=${TEST_FREQ:-200}

PROJECT_NAME=${PROJECT_NAME:-verl_lp_grpo_1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_8b_lp_grpo_gamma1_vllm_fsdp_$(date +%Y%m%d_%H%M)}

TRAIN_FILE=${TRAIN_FILE:-/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/data/combined/filtered/strategy1_30k/mmrl14208_vppo16237_lp_grpo.parquet}
TEST_FILE=${TEST_FILE:-/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/data/geo3k/test.parquet}

# LP-GRPO: improving=(1-p_0)^gamma*clip(1+lambda*KL,1,w_max), regressing=(1-p_t)^gamma
LP_GAMMA=${LP_GAMMA:-1.0}
LP_LAMBDA=${LP_LAMBDA:-3.0}
LP_W_MAX=${LP_W_MAX:-3.0}
LP_EPS_P=${LP_EPS_P:-0.05}

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

# Per-device extras (single trailing array, never empty).
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
