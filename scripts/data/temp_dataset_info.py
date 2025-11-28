from huggingface_hub import HfApi
api = HfApi()
datasets = [
    'FreedomIntelligence/evol-instruct-japanese',
    'FreedomIntelligence/alpaca-gpt4-japanese',
    'Beluuuuuuga/Japanese-Instruction-Linux-Command-169',
    'az1/anthropic_hh_rlhf_japanese',
    'leemeng/mt_bench_japanese',
    'haih2/japanese-conala',
    'lyakaap/laion2B-japanese-subset',
    'izumi-lab/llm-japanese-dataset',
    'fujiki/japanese_alpaca_data',
    'FreedomIntelligence/MMLU_Japanese'
]
for ds in datasets:
    info = api.dataset_info(ds)
    license_name = (info.card_data or {}).get('license')
    print(ds, license_name)
