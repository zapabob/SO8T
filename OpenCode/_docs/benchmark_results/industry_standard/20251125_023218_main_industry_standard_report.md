# Industry-standard Benchmark Report

- **Timestamp**: 20251125_023218
- **Worktree**: main
- **Run Directory**: `D:\webdataset\benchmark_results\industry_standard\industry_initial_run`

## Step Results

### lm_eval [OK]
- Command: `py -3 scripts/cuda_accelerated_benchmark.py`
- Exit code: 0
- Log: `D:\webdataset\benchmark_results\industry_standard\industry_initial_run\01_lm_eval.log`

### deepeval [NG]
- Command: `py -3 scripts/evaluation/deepeval_ethics_test.py --model-runner ollama --model-name aegis-borea-phi35-instinct-jp:q8_0`
- Exit code: 1
- Log: `D:\webdataset\benchmark_results\industry_standard\industry_initial_run\02_deepeval.log`
