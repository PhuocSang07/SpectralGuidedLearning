"""VCORE (Gong et al., ACL 2026) as a baseline on P-ALIGN's 966 traces, run with its own code on Modal.

    modal run scripts/modal/vcore-baseline.py --smoke          # 8 traces, one epoch (4 updates), no eval
    modal run scripts/modal/vcore-baseline.py                  # full run: train (3 epochs) + eval
    modal run scripts/modal/vcore-baseline.py --only eval      # eval an adapter already on the volume

VCORE ships as a fork of LLaMA-Factory and transformers 4.52.4 (github.com/coder-gx/VCORE, pinned
below) and only supports LoRA, so the comparison is with the 1.5B LoRA arms (`sft-nll-lora`,
`iwc-gain-l05-lora`). To keep everything but the objective fixed, VCORE trains on the token ids and
loss mask that `sgl` prepares for those arms, handed to LLaMA-Factory as a pre-tokenized dataset,
with their recipe: LoRA r16 / alpha 16 / dropout 0.05 on all linear layers, 3 epochs, effective
batch 32, lr 5e-5 cosine to 1e-5, warmup 0.1, 32k context, seed 42. VCORE's own hyperparameters
are those of its released scripts (temperature 5e3, epsilon 1e-4, a probe at every update on a
random batch of 32). Evaluation is the shared P-ALIGN protocol, run by `sgl` on the adapter.
"""

import subprocess
from pathlib import Path

import modal

REMOTE = "/repo"
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE)
VCORE_COMMIT = "2dc91e161b1b06bb3876892c1930bbb194c5088b"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ARM = "configs/sgl/r1-qwen-1.5b-palign/sft-nll-lora.yaml"  # the matched LoRA arm whose data VCORE uses
RUN_NAME = "vcore-lora-r1-qwen-1.5b-palign"
RUNS = "/runs"
DATA = f"{RUNS}/data/r1-qwen-1.5b-palign"

sgl_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "vllm==0.27.1", "transformers==5.5.3", "peft", "accelerate", "datasets", "pandas", "pyarrow",
        "pyyaml", "tqdm", "matplotlib", "math-verify==0.8.0", "sympy", "json-repair",
    )
    .env({"PYTHONPATH": f"{REMOTE}/src", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(
        REPO, REMOTE, copy=False,
        ignore=[".git", ".claude", "data", "checkpoints", "paper", "results*", "logs", "experiments",
                "**/__pycache__", "**/*.pyc", ".venv*"],
    )
)
# VCORE's environment as its README sets it up (Python 3.10, torch 2.7.0 cu128, its requirements,
# its LLaMA-Factory and transformers forks installed editable)
vcore_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .pip_install("torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        f"git clone https://github.com/coder-gx/VCORE /vcore && git -C /vcore checkout {VCORE_COMMIT}",
        "pip install -r /vcore/requirements.txt",
        "pip install -e /vcore/llama_factory",
        "pip install -e /vcore/transformers-4.52.4",
    )
    .env({"HF_HOME": "/cache/hf", "DISABLE_VERSION_CHECK": "1", "WANDB_DISABLED": "true",
          "TOKENIZERS_PARALLELISM": "false"})
)
cache = modal.Volume.from_name("sgl-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("sgl-runs", create_if_missing=True)
app = modal.App("sgl-vcore-baseline")
volumes = {"/cache": cache, RUNS: runs}


def sh(command: str, cwd: str = REMOTE) -> str:
    print(f"\n$ {command}", flush=True)
    result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
    print(result.stdout[-8000:], result.stderr[-6000:], sep="\n", flush=True)
    # `sgl run` reports a failed stage in its output, so check that too
    if result.returncode != 0 or "failed with exit code" in result.stdout + result.stderr:
        raise RuntimeError(f"failed ({result.returncode}): {command}")
    return result.stdout


@app.function(image=sgl_image, cpu=8, memory=16384, timeout=3600, volumes=volumes)
def prepare() -> None:
    """The LoRA arms' training file: prompt + response token ids and the loss mask of every trace."""
    sgl = "python -c 'from sgl.cli import sgl_main; sgl_main()'"
    sh(f"{sgl} run {ARM} --stages prepare,vanilla data_dir={DATA}")
    runs.commit()


@app.function(image=vcore_image, gpu="H100", timeout=24 * 3600, volumes=volumes)
def train(smoke: bool) -> None:
    import json

    import yaml
    from datasets import Dataset, DatasetDict

    rows = [json.loads(line) for line in open(f"{DATA}/train-vanilla.jsonl")]
    rows = [r for r in rows if len(r["input_ids"]) <= 32768]  # --max-seq-len of the sgl arms
    if smoke:
        rows = rows[:8]
    tokenized = f"/tmp/vcore-tokenized{'-smoke' if smoke else ''}"
    # labels as sgl's collator derives them: the token id where the loss mask is 1, -100 elsewhere
    DatasetDict({"train": Dataset.from_dict({
        "input_ids": [r["input_ids"] for r in rows],
        "attention_mask": [[1] * len(r["input_ids"]) for r in rows],
        "labels": [[t if m else -100 for t, m in zip(r["input_ids"], r["loss_mask"])] for r in rows],
    })}).save_to_disk(tokenized)
    supervised = sum(sum(r["loss_mask"]) for r in rows)
    print(f"{len(rows)} traces, {supervised:,} supervised tokens", flush=True)

    output = f"{RUNS}/checkpoints/{RUN_NAME}{'-smoke' if smoke else ''}"
    accumulation = 2 if smoke else 32
    config = {
        "model_name_or_path": MODEL, "trust_remote_code": True, "flash_attn": "sdpa",
        "stage": "sft", "do_train": True, "finetuning_type": "lora",
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "lora_rank": 16, "lora_alpha": 16, "lora_dropout": 0.05,
        # the data is read from tokenized_path; LLaMA-Factory still asks for a dataset name, which it
        # ignores once tokenized data exists, and the template only sets up the tokenizer
        "dataset": "identity", "template": "deepseekr1", "tokenized_path": tokenized, "cutoff_len": 32768,
        "output_dir": output, "overwrite_output_dir": True, "save_strategy": "no", "save_only_model": True,
        "logging_steps": 1, "report_to": "none", "plot_loss": False,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": accumulation,
        "num_train_epochs": 3.0, "learning_rate": 5.0e-5,
        "lr_scheduler_type": "cosine_with_min_lr", "lr_scheduler_kwargs": {"min_lr": 1.0e-5},
        "warmup_ratio": 0.1, "weight_decay": 0.0, "max_grad_norm": 1.0,
        "bf16": True, "gradient_checkpointing": True, "seed": 42,
        "use_vcore": True, "vcore_single_process": True, "vcore_temperature": 5.0e3,
        "vcore_epsilon": 1.0e-4, "vcore_anchor_steps": 1, "vcore_pre_ga": accumulation,
    }
    if smoke:  # VCORE's training loop ignores max_steps, so the smoke run is one short epoch
        config["num_train_epochs"] = 1.0
    Path("/tmp/vcore.yaml").write_text(yaml.safe_dump(config))
    sh("llamafactory-cli train /tmp/vcore.yaml", cwd="/vcore/llama_factory")
    sh(f"ls -la {output}", cwd="/")
    runs.commit()


@app.function(image=sgl_image, gpu="H100", timeout=6 * 3600, volumes=volumes)
def evaluate() -> None:
    """The shared P-ALIGN protocol on the VCORE adapter, exactly as the sgl LoRA arms are evaluated."""
    sgl = "python -c 'from sgl.cli import sgl_main; sgl_main()'"
    sh(f"{sgl} run {ARM} --stages eval data_dir={DATA} run_name={RUN_NAME} "
       f"checkpoint={RUNS}/checkpoints/{RUN_NAME} results_dir={RUNS}/results")
    sh(f"cat {RUNS}/results/{RUN_NAME}/summary.json")
    runs.commit()


@app.local_entrypoint()
def main(smoke: bool = False, only: str = "") -> None:
    if only in ("", "train"):
        prepare.remote()
        train.remote(smoke)
    if only in ("", "eval") and not smoke:
        evaluate.remote()
