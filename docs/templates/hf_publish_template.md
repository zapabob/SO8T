# Hugging Face Publish Checklist (Template)

1. **Login**
   ```bash
   huggingface-cli login
   ```
2. **Create repo**
   ```bash
   huggingface-cli repo create $REPO_NAME --type model --private $PRIVATE
   ```
3. **Prepare artifacts**
   - $ARTIFACTS
4. **Upload**
   ```bash
   python -m src.infra.hf.publish_to_hf --repo $REPO_NAME --artifact-dir $ARTIFACT_DIR
   ```
5. **Validate**
   - Check model card renders correctly
   - Verify safetensors/gguf checksums
