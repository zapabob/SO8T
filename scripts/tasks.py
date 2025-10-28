from __future__ import annotations

from pathlib import Path

from invoke import task

ROOT = Path(__file__).parent.resolve()


@task(help={"python": "Python executable to use for environment bootstrap."})
def bootstrap(ctx, python="python"):
    """Create a virtual environment and install dependencies."""
    venv_dir = ROOT / ".venv"
    ctx.run(f"{python} -m venv {venv_dir}", echo=True)
    pip = venv_dir / "Scripts" / "pip" if (venv_dir / "Scripts").exists() else venv_dir / "bin" / "pip"
    ctx.run(f"{pip} install --upgrade pip", echo=True)
    ctx.run(f"{pip} install -r requirements.txt", echo=True)


@task(help={"count": "Number of synthetic samples to create.", "seed": "Random seed."})
def generate_data(ctx, count=3000, seed=7, output_dir="data"):
    """Generate the synthetic ENV/CMD/SAFE dataset."""
    ctx.run(
        f"python dataset_synth.py --count {int(count)} --seed {int(seed)} --output-dir {output_dir}",
        echo=True,
    )


@task(help={"config": "Path to YAML training configuration."})
def train_so8t(ctx, config="configs/train_default.yaml"):
    """Train the SO8T model with PET regularization."""
    ctx.run(f"python train.py --config {config}", echo=True)


@task(help={"checkpoint": "Model checkpoint for evaluation.", "split": "Dataset split name."})
def eval_so8t(ctx, checkpoint="chk/so8t_default.pt", split="val", output="chk/eval_default.json"):
    """Run evaluation on a saved checkpoint."""
    ctx.run(
        f"python eval.py --checkpoint {checkpoint} --split {split} --output {output}",
        echo=True,
    )


@task
def report(ctx):
    """Assemble a lightweight experiment summary."""
    ctx.run("python scripts/summarize_results.py", echo=True)
