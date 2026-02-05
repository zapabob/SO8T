# HF Publish GitHub Actions

Workflow: `.github/workflows/hf_publish.yml`

## Secrets
- `HF_TOKEN`: Hugging Face access token

## Usage
1. GitHub → Actions → **HF Publish**
2. Set inputs:
   - `repo`: org/model
   - `artifact_dir`: directory with artifacts (default `hf_readme_output`)
   - `repo_type`: model or dataset

## Notes
- The workflow attempts to generate a model card before upload.
- If you prefer to skip model card generation, remove that step.
