# Phase4 Data Pipeline

Runs API fetch, PDF ingest, and Hugging Face CLI download.

## Usage
```bash
py -m src.data.phase4_pipeline --config config/phase4_pipeline.yaml --out data/phase4
```

## Notes
- API sources are defined in `config/phase4_pipeline.yaml`.
- PDF ingest uses `pypdf` (see requirements).
- HF downloads call `huggingface-cli download` and respect allow patterns.
