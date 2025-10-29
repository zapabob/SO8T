# SO8T CoT Safety Model - HuggingFace Format

This directory contains the SO8T CoT Safety model in HuggingFace format, ready for GGUF conversion.

## Files

- `pytorch_model.bin`: Model weights
- `config.json`: Model configuration
- `tokenizer_config.json`: Tokenizer configuration
- `generation_config.json`: Generation configuration
- `README.md`: Model card

## Usage

This model can be used with HuggingFace Transformers or converted to GGUF format for deployment.

## Next Steps

1. Convert to GGUF format using llama.cpp
2. Deploy with Ollama
3. Test safety judgments
