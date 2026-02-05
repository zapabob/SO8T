"""Generate model card from template and YAML config."""
from __future__ import annotations

import argparse
from pathlib import Path
from string import Template
from typing import Dict

import yaml


def load_yaml(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate(template_path: Path, config_path: Path, out_path: Path) -> Path:
    template = Template(template_path.read_text(encoding='utf-8'))
    cfg = load_yaml(config_path)
    content = template.safe_substitute(
        MODEL_NAME=cfg.get('model_name', ''),
        BASE_MODEL=cfg.get('base_model', ''),
        TASK=cfg.get('task', ''),
        DATA_SUMMARY=cfg.get('data_summary', ''),
        INTENDED_USE=cfg.get('intended_use', ''),
        EVAL_SUMMARY=cfg.get('eval_summary', ''),
        TRAINING_PROCEDURE=cfg.get('training_procedure', ''),
        LIMITATIONS=cfg.get('limitations', ''),
        ETHICS=cfg.get('ethics', ''),
        CITATION=cfg.get('citation', ''),
    )
    out_path.write_text(content, encoding='utf-8')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate model card from template')
    parser.add_argument('--template', default='docs/templates/model_card_template.md')
    parser.add_argument('--config', default='config/model_card.yaml')
    parser.add_argument('--out', default='hf_readme_output/README.md')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generate(Path(args.template), Path(args.config), out_path)


if __name__ == '__main__':
    main()
