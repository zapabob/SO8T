# GGUF Export Notes

- Convert the PEFT adapted model to a merged float16 checkpoint before invoking `llama.cpp` conversion tools.
- Use `python scripts/export_gguf.py --checkpoint path/to/merged.ckpt --output modelfiles/So8tCodex.gguf`.
- Ensure tensor names follow the `model.layers.*` convention; the exporter maps SO8T rotations to the residual branch.
- Quantize with `--quantization q8_0` for RTX3060 compatibility; verify perplexity drift using `tests/test_perf.py`.
- Store audit metadata inside `modelfiles/So8tCodex.Modelfile` for traceability.
