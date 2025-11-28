#!/bin/bash
# Upload AEGIS model to HuggingFace Hub

set -e

# Configuration
REPO_NAME="$1"
HF_TOKEN="$2"

if [ -z "$REPO_NAME" ]; then
    echo "❌ Usage: $0 <repo-name> [hf-token]"
    echo "Example: $0 your-username/AEGIS-v2.0-Phi3.5-thinking"
    exit 1
fi

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli not found. Install with: pip install huggingface_hub[cli]"
    exit 1
fi

# Get token from environment if not provided
if [ -z "$HF_TOKEN" ]; then
    HF_TOKEN="$HF_TOKEN"
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ HuggingFace token not found. Set HF_TOKEN environment variable or pass as second argument"
    exit 1
fi

echo "🚀 Starting AEGIS model upload to HuggingFace Hub"
echo "📁 Repository: $REPO_NAME"
echo "📂 Upload directory: D:/webdataset/models/aegis-huggingface-upload"

# Export token for huggingface-cli
export HF_TOKEN="$HF_TOKEN"

# Create repository if it doesn't exist
echo "📝 Creating/checking repository..."
huggingface-cli repo create "$REPO_NAME" --type model --yes || echo "Repository already exists"

# Upload files in batches to handle large files better
echo "📤 Uploading configuration files..."
huggingface-cli upload "$REPO_NAME" \
    huggingface_upload/AEGIS-v2.0-Phi3.5-thinking/README.md \
    huggingface_upload/AEGIS-v2.0-Phi3.5-thinking/config.json \
    huggingface_upload/AEGIS-v2.0-Phi3.5-thinking/generation_config.json \
    huggingface_upload/AEGIS-v2.0-Phi3.5-thinking/tokenizer.json \
    huggingface_upload/AEGIS-v2.0-Phi3.5-thinking/tokenizer.model \
    huggingface_upload/AEGIS-v2.0-Phi3.5-thinking/tokenizer_config.json \
    huggingface_upload/AEGIS-v2.0-Phi3.5-thinking/special_tokens_map.json \
    huggingface_upload/AEGIS-v2.0-Phi3.5-thinking/added_tokens.json

echo "📊 Uploading benchmark visualizations..."
huggingface-cli upload "$REPO_NAME" \
    huggingface_upload/AEGIS-v2.0-Phi3.5-thinking/benchmark_results/

echo "🤖 Uploading model weights (this may take a while)..."
huggingface-cli upload "$REPO_NAME" \
    models/aegis_adjusted/model-00001-of-00002.safetensors \
    models/aegis_adjusted/model-00002-of-00002.safetensors

echo "✅ Upload completed successfully!"
echo "🌐 Model available at: https://huggingface.co/$REPO_NAME"

# Update repository metadata
echo "📝 Updating repository metadata..."
huggingface-cli repo update "$REPO_NAME" \
    --tags "transformers,phi-3,enhanced-reasoning,ethical-ai,japanese,reasoning,safety,transformer,mathematical-reasoning,quadruple-reasoning,thinking-model"

echo "✅ Repository metadata updated"
echo "🎉 AEGIS model upload completed!"
