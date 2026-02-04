# HF Publish Templates

Templates live in `docs/templates/`:
- `model_card_template.md`
- `dataset_card_template.md`
- `hf_publish_template.md`

Generate model card:
```bash
py -m src.infra.hf.model_card_generator --config config/model_card.yaml --out hf_readme_output/README.md
```
