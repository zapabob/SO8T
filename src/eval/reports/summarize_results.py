from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def main() -> None:
    console = Console()
    chk_dir = Path("chk")
    eval_path = chk_dir / "eval_summary.json"
    log_path = chk_dir / "so8t_default_train_log.jsonl"
    summary_path = chk_dir / "experiment_summary.md"

    eval_data = load_json(eval_path) if eval_path.exists() else {}
    log_entries = load_log(log_path)

    table = Table(title="SO8T Experiment Summary")
    table.add_column("Metric")
    table.add_column("Value")
    if eval_data:
        table.add_row("Eval Split", eval_data.get("split", "n/a"))
        table.add_row("Eval Accuracy", f"{eval_data.get('accuracy', 0.0):.3f}")
        table.add_row("Eval Macro F1", f"{eval_data.get('macro_f1', 0.0):.3f}")
    if log_entries:
        train_losses = [entry["loss"] for entry in log_entries if "loss" in entry]
        train_acc = [entry["train_accuracy"] for entry in log_entries if "train_accuracy" in entry]
        if train_losses:
            table.add_row("Train Loss (avg)", f"{mean(train_losses):.3f}")
        if train_acc:
            table.add_row("Train Accuracy (avg)", f"{mean(train_acc):.3f}")
    console.print(table)

    summary_lines = ["# Experiment Summary"]
    if eval_data:
        summary_lines.append(f"- Eval split: {eval_data.get('split', 'n/a')}")
        summary_lines.append(f"- Accuracy: {eval_data.get('accuracy', 0.0):.3f}")
        summary_lines.append(f"- Macro F1: {eval_data.get('macro_f1', 0.0):.3f}")
    if log_entries:
        summary_lines.append(
            f"- Training loss (avg): {mean(train_losses):.3f}" if log_entries else "- Training loss: n/a"
        )
        summary_lines.append(
            f"- Training accuracy (avg): {mean(train_acc):.3f}" if log_entries else "- Training accuracy: n/a"
        )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    console.print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
