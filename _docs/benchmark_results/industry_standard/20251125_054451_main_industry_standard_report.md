# Industry-standard Benchmark Report

- **Timestamp**: 20251125_054451
- **Worktree**: main
- **Run Directory**: `D:\webdataset\benchmark_results\industry_standard\industry_1\255\20_054451`

## Step Results

### lm_eval [OK]
- Command: `py -3 scripts/cuda_accelerated_benchmark.py`
- Exit code: 0
- Log: `D:\webdataset\benchmark_results\industry_standard\industry_1\255\20_054451\01_lm_eval.log`

### deepeval [NG]
- Command: `py -3 scripts/evaluation/deepeval_ethics_test.py --model-runner ollama --model-name aegis-borea-phi35-instinct-jp:q8_0`
- Exit code: 1
- Log: `D:\webdataset\benchmark_results\industry_standard\industry_1\255\20_054451\02_deepeval.log`

### elyza_model-a_q8_0 [OK]
- Command: `py -3 scripts/evaluation/elyza_benchmark.py --model-name model-a:q8_0 --output-dir D:\webdataset\benchmark_results\industry_standard\industry_1\255\20_054451\elyza`
- Exit code: 0
- Log: `D:\webdataset\benchmark_results\industry_standard\industry_1\255\20_054451\03_elyza_model-a_q8_0.log`

### elyza_aegis-phi3.5-fixed-0.8_latest [OK]
- Command: `py -3 scripts/evaluation/elyza_benchmark.py --model-name aegis-phi3.5-fixed-0.8:latest --output-dir D:\webdataset\benchmark_results\industry_standard\industry_1\255\20_054451\elyza`
- Exit code: 0
- Log: `D:\webdataset\benchmark_results\industry_standard\industry_1\255\20_054451\03_elyza_aegis-phi3.5-fixed-0.8_latest.log`

### promptfoo [NG]
- Command: `py -3 scripts/evaluation/promptfoo_ab_test.py --config configs\promptfoo_config.yaml --output-root D:\webdataset\benchmark_results\industry_standard\industry_1\255\20_054451\promptfoo --html --json --use-npx`
- Exit code: 1
- Log: `D:\webdataset\benchmark_results\industry_standard\industry_1\255\20_054451\04_promptfoo.log`
