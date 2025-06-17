import modal
import subprocess

app = modal.App("grpo")
checkpoints_volume = modal.Volume.from_name("grpo-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("grpo-data", create_if_missing=True)

VERL_REPO_PATH="/root/verl"
CHECKPOINTS_PATH = "/checkpoints"
DATA_PATH = "/data"
TRAINING_FILES_PATH = f"{DATA_PATH}/train.parquet"
VALIDATION_FILES_PATH = f"{DATA_PATH}/test.parquet"
MAX_PROMPT_LENGTH= 1024
MAX_RESPONSE_LENGTH = 1024
BATCH_SIZE = 1024
MODEL = "Qwen/QwQ-32B"
LEARNING_RATE = "1e-6"
MINI_BATCH_SIZE = 128
MICROBATCH_SIZE_PER_GPU = 16
PATH_TO_REWARD_FUNCTION="/root/reward.py"
REWARD_FUNCTION_NAME = "compute_reward"


image = (
    modal.Image.from_registry("whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3")
    .apt_install("git")
    .run_commands([f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH} && cd {VERL_REPO_PATH} && pip install -e .[vllm]"])
    .add_local_file("reward.py", PATH_TO_REWARD_FUNCTION)
)


# Replace with your own data processing
@app.function(image = image, volumes = { DATA_PATH: data_volume })
def prep_dataset():
    subprocess.run(["python", f"{VERL_REPO_PATH}/examples/data_preprocess/gsm8k.py", "--local_dir", DATA_PATH], check=True)
    

@app.function(
    image = image,
    gpu="H200:8",
    volumes = {
        CHECKPOINTS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    secrets = [modal.Secret.from_name("wandb-secret", environment_name="main")],
    timeout=86400
)
def train():
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={TRAINING_FILES_PATH}",
        f"data.val_files={VALIDATION_FILES_PATH}",        
        f"data.train_batch_size={BATCH_SIZE}",
        f"data.max_prompt_length={MAX_PROMPT_LENGTH}",
        f"data.max_response_length={MAX_RESPONSE_LENGTH}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={MODEL}",
        f"actor_rollout_ref.actor.optim.lr={LEARNING_RATE}",
        "actor_rollout_ref.model.use_remove_padding=False",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={MINI_BATCH_SIZE}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={MICROBATCH_SIZE_PER_GPU}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={MICROBATCH_SIZE_PER_GPU}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.n=5",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={MICROBATCH_SIZE_PER_GPU}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console', 'wandb']",
        "trainer.project_name=verl_grpo_example_qwq32b",
        "trainer.experiment_name=qwq32b_example",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.save_freq=5",
        "trainer.test_freq=5",
        "trainer.total_epochs=15",
        f"trainer.default_local_dir={CHECKPOINTS_PATH}",
        "trainer.resume_mode=auto",

    ]
    subprocess.run(cmd, check=True)

@app.local_entrypoint()
def main():
    # prep_dataset.remote()
    train.remote()