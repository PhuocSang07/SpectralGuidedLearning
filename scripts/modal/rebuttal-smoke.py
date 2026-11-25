"""Light Modal checks of the rebuttal controls: the full CPU test suite, then a short GPU smoke run.

    modal run scripts/modal/rebuttal-smoke.py              # tests + GPU smoke
    modal run scripts/modal/rebuttal-smoke.py --only tests # CPU tests and lint only (--only gpu: smoke only)

The GPU part runs the real configs on a few P-ALIGN traces with DeepSeek-R1-Distill-Qwen-1.5B:
answer gain (default, masked, reference target, other probe), the position / residual / top-k
weights, the gain report, and a few ASFT training steps (full fine-tuning and LoRA). It also checks
that the default answer-gain path still gives the gains of the code on main.
"""

import subprocess
from pathlib import Path

import modal

REMOTE = "/repo"
# the repo root on the local machine; inside the container the script sits in /root
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE)
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LIMIT = 16

image = (
    # a CUDA devel base: vLLM JIT-compiles kernels at engine start and needs nvcc (torch is cu130)
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "vllm==0.27.1", "transformers==5.5.3", "peft", "accelerate", "datasets", "pandas", "pyarrow",
        "pyyaml", "tqdm", "matplotlib", "math-verify==0.8.0", "sympy", "json-repair", "pytest", "ruff",
    )
    .env({"PYTHONPATH": f"{REMOTE}/src", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(
        REPO, REMOTE, copy=False,
        # results/, logs/ and experiments/ stay: the regression tests read them
        ignore=[".git", ".claude", "data", "checkpoints", "paper", "**/__pycache__", "**/*.pyc", ".venv*"],
    )
)
cache = modal.Volume.from_name("sgl-hf-cache", create_if_missing=True)
app = modal.App("sgl-rebuttal-smoke", image=image)


def sh(command: str, check: bool = True) -> str:
    print(f"\n$ {command}", flush=True)
    result = subprocess.run(command, shell=True, cwd=REMOTE, text=True, capture_output=True)
    print(result.stdout[-6000:], result.stderr[-4000:], sep="\n", flush=True)
    # `sgl run` reports a failed stage in its output, so check that too
    if check and (result.returncode != 0 or "failed with exit code" in result.stdout + result.stderr):
        raise RuntimeError(f"failed ({result.returncode}): {command}")
    return result.stdout


@app.function(cpu=8, memory=16384, timeout=1800, volumes={"/cache": cache})
def tests() -> None:
    sh("ruff check src tests scripts")
    sh("python -m pytest -q -p no:cacheprovider")


@app.function(gpu="H100", timeout=3600, volumes={"/cache": cache})
def gpu_smoke(main_answer_gain: str) -> None:
    import json

    sgl = "python -c 'from sgl.cli import sgl_main; sgl_main()'"
    arm = "configs/sgl/r1-qwen-1.5b-palign"
    limit = f"stages.answer_gain.args.limit={LIMIT} stages.gain_signal.args.strengths=null"
    data = "data/r1-qwen-1.5b-palign"

    sh(f"{sgl} run {arm}/iwc-gain-ref-l05.yaml --stages prepare,reference_answers")
    # default gain, then the same samples with main's code: the default path must be unchanged
    sh(f"{sgl} run {arm}/gain-topk59.yaml --stages answer_gain,gain_signal,weights {limit}")
    # vLLM is not bit-reproducible across runs (the same 16 samples moved by up to 0.2 nats between
    # two runs of identical code), so the default path is checked on what it sends to vLLM instead:
    # a stub `vllm` records a hash of every probe prompt from main's code and from the new code,
    # over all 966 samples, and the two lists must be identical.
    Path("/tmp/answer_gain_main.py").write_text(main_answer_gain)
    Path("/tmp/fakevllm/vllm").mkdir(parents=True, exist_ok=True)
    Path("/tmp/fakevllm/vllm/__init__.py").write_text(
        "import hashlib, json, os, sys\n"
        "class SamplingParams:\n    def __init__(self, **kwargs): pass\n"
        "class LLM:\n    def __init__(self, **kwargs): pass\n"
        "    def generate(self, prompts, params):\n"
        "        hashes = [hashlib.sha256(str(p['prompt_token_ids']).encode()).hexdigest() for p in prompts]\n"
        "        open(os.environ['PROMPT_HASHES'], 'w').write(json.dumps(hashes))\n"
        "        sys.exit(0)\n"
    )
    Path("/tmp/fakevllm/vllm/inputs.py").write_text("def TokensPrompt(prompt_token_ids):\n"
                                                     "    return {'prompt_token_ids': prompt_token_ids}\n")
    stub = f"PYTHONPATH=/tmp/fakevllm:{REMOTE}/src"
    common = f"--data-path {data}/train-segmented.jsonl --model {MODEL} --output /tmp/unused.json"
    sh(f"{stub} PROMPT_HASHES=/tmp/prompts-main.json python /tmp/answer_gain_main.py {common}")
    sh(f"{stub} PROMPT_HASHES=/tmp/prompts-new.json python -m sgl.signals.answer_gain {common}")
    old = json.loads(Path("/tmp/prompts-main.json").read_text())
    new = json.loads(Path("/tmp/prompts-new.json").read_text())
    assert old == new, f"default path builds different prompts ({len(old)} vs {len(new)})"
    print(f"default answer-gain prompts identical to main's code: {len(new):,} probes over all samples")

    for name in ("iwc-gain-masked-l05", "iwc-gain-ref-l05", "iwc-gain-probe-so"):
        sh(f"{sgl} run {arm}/{name}.yaml --stages answer_gain,gain_signal,weights {limit}")
    for name in ("iwc-position-l05", "iwc-residual-l05"):
        sh(f"{sgl} run {arm}/{name}.yaml --stages position_signal,weights {limit}")
    sh(f"python -m sgl.signals.gain_report --data-path {data}/train-segmented.jsonl "
       f"--gains {data}/signals/step_answer_gain.json --tokenizer {MODEL} --compare "
       f"masked={data}/signals/step_answer_gain-masked.json "
       f"reference={data}/signals/step_answer_gain-reference.json "
       f"probe-so={data}/signals/step_answer_gain-probe-so.json")
    sh(f"ls -la {data} {data}/signals; cat {data}/*-selection-stats.json | head -120")

    # chunked segmentation: prepare only (its gains would rerun the whole scoring)
    sh(f"{sgl} run {arm}/iwc-gain-l05-chunk64.yaml --stages prepare")
    sh("python -c \"import json; r=[json.loads(l) for l in open('data/r1-qwen-1.5b-palign-chunk64/"
       "train-segmented.jsonl')]; s=[st['token_end']-st['token_start'] for x in r for st in x['steps']]; "
       "print(len(r), 'traces', len(s), 'chunks', 'max', max(s), 'share of 64-token chunks', "
       "sum(v==64 for v in s)/len(s))\"")

    # a few ASFT steps: full fine-tuning (frozen reference copy) and LoRA (disabled adapter)
    sh(f"{sgl} run {arm}/sft-asft.yaml --stages vanilla")
    train = (f"python -m sgl.training.train --smoke --model-name {MODEL} --data-path {data}/train-vanilla.jsonl "
             "--epochs 1 --learning-rate 1e-5 --min-learning-rate 0 --warmup-ratio 0 --per-device-batch-size 1 "
             "--gradient-accumulation-steps 2 --max-seq-len 2048 --logging-steps 1 --save-strategy no "
             "--objective dft --kl-anchor-weight 0.03 --model-dtype bfloat16")
    sh(f"{train} --output-dir /tmp/asft-full")
    sh(f"{train} --output-dir /tmp/asft-lora --use-lora --lora-r 16 --lora-alpha 16 --no-lora-merge")


@app.local_entrypoint()
def main(only: str = "") -> None:
    if only != "gpu":
        tests.remote()
    if only != "tests":
        # the answer-gain module as it is on main, to check the default path is unchanged
        main_answer_gain = subprocess.run(
            ["git", "show", "origin/main:src/sgl/signals/answer_gain.py"],
            cwd=REPO, check=True, capture_output=True, text=True,
        ).stdout
        gpu_smoke.remote(main_answer_gain)
