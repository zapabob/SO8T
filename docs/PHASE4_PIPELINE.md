# Phase4 Data Pipeline

Runs API fetch, PDF ingest, and Hugging Face CLI download.

## Usage
```bash
py -m src.data.phase4_pipeline --config config/phase4_pipeline.yaml --out data/phase4
```

## Config options (highlights)
- `defaults`: timeout / retries / backoff / min_interval_seconds
- `api_sources[].pagination`: page/offset/cursor
- `api_sources[].results_path`: JSON path to extract items
- `api_sources[].sha256`: optional checksum for API output
- `pdf_sources[].sha256`: optional checksum for PDF file
- `hf_sources[].checksums`: list of `{path, sha256}` for downloaded files

## Notes
- API sources are defined in `config/phase4_pipeline.yaml`.
- PDF ingest uses `pypdf` (see requirements).
- HF downloads call `huggingface-cli download` and respect allow patterns.
