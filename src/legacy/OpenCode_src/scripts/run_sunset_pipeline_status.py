#!/usr/bin/env python3
"""
サンセットパイプライン状態確認スクリプト
統合されたデータセットの状態を確認
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def check_sunset_pipeline_status():
    """サンセットパイプラインの状態を確認"""
    project_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("サンセットパイプライン状態確認")
    print("=" * 80)
    print(f"確認日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 設定ファイル確認
    config_path = project_root / "config" / "dataset.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        sources = config.get('sources', [])
        print(f"[CONFIG] データソース数: {len(sources)}")
        
        # ドメイン知識データセット確認
        domain_sources = [s for s in sources if 'domain_knowledge' in s]
        print(f"[CONFIG] ドメイン知識データソース: {len(domain_sources)}")
        for ds in domain_sources:
            print(f"  - {ds}")
        
        # データセットタイプ別集計
        hf_count = len([s for s in sources if 'huggingface:' in s])
        moonshot_count = len([s for s in sources if 'moonshot:' in s])
        synthetic_count = len([s for s in sources if 'synthetic:' in s])
        local_count = len([s for s in sources if 'local:' in s])
        
        print(f"\n[CONFIG] データソース内訳:")
        print(f"  - HuggingFace: {hf_count}")
        print(f"  - Moonshot: {moonshot_count}")
        print(f"  - Synthetic: {synthetic_count}")
        print(f"  - Local: {local_count}")
    else:
        print("[ERROR] 設定ファイルが見つかりません")
        return False
    
    # 2. ドメイン知識データセット確認
    print("\n" + "=" * 80)
    print("ドメイン知識データセット確認")
    print("=" * 80)
    
    domain_data_dir = project_root / "data" / "domain_knowledge"
    if domain_data_dir.exists():
        jsonl_files = list(domain_data_dir.glob("*.jsonl"))
        print(f"[DATA] ドメイン知識データセットファイル数: {len(jsonl_files)}")
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                file_size = jsonl_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  - {jsonl_file.name}: {lines:,} items, {file_size:.2f} MB")
            except Exception as e:
                print(f"  - {jsonl_file.name}: [ERROR] {e}")
    else:
        print("[WARN] ドメイン知識データディレクトリが見つかりません")
    
    # 3. Arxiv/BioRxivデータ確認
    arxiv_data_dir = project_root / "data" / "arxiv_biorxiv" / "cleaned"
    if arxiv_data_dir.exists():
        jsonl_files = list(arxiv_data_dir.glob("*.jsonl"))
        print(f"\n[ARXIV] Arxiv/BioRxivデータセットファイル数: {len(jsonl_files)}")
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                file_size = jsonl_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  - {jsonl_file.name}: {lines:,} items, {file_size:.2f} MB")
            except Exception as e:
                print(f"  - {jsonl_file.name}: [ERROR] {e}")
    else:
        print("\n[INFO] Arxiv/BioRxivデータはまだ処理されていません")
        print("      実行: py -3.12 scripts/data_processing/process_arxiv_biorxiv.py --max-papers 100000")
    
    # 4. 非構造データ確認
    unstructured_dir = project_root / "data" / "unstructured" / "cleaned"
    if unstructured_dir.exists():
        jsonl_files = list(unstructured_dir.glob("*.jsonl"))
        print(f"\n[UNSTRUCTURED] 非構造データファイル数: {len(jsonl_files)}")
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                file_size = jsonl_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  - {jsonl_file.name}: {lines:,} items, {file_size:.2f} MB")
            except Exception as e:
                print(f"  - {jsonl_file.name}: [ERROR] {e}")
    else:
        print("\n[INFO] 非構造データはまだ処理されていません")
        print("      実行: py -3.12 scripts/data_processing/process_unstructured_data.py")
    
    # 5. パイプラインスクリプト確認
    print("\n" + "=" * 80)
    print("パイプラインスクリプト確認")
    print("=" * 80)
    
    scripts_to_check = [
        ("データパイプライン", project_root / "scripts" / "data_processing" / "dataset_pipeline.py"),
        ("統合スクリプト", project_root / "scripts" / "data_processing" / "integrate_domain_knowledge.py"),
        ("Arxiv処理", project_root / "scripts" / "data_processing" / "process_arxiv_biorxiv.py"),
        ("非構造データ処理", project_root / "scripts" / "data_processing" / "process_unstructured_data.py"),
        ("サンセットパイプライン", project_root / "scripts" / "run_sunset_pipeline.py"),
    ]
    
    for name, script_path in scripts_to_check:
        if script_path.exists():
            print(f"  [OK] {name}: {script_path.name}")
        else:
            print(f"  [NG] {name}: 見つかりません")
    
    # 6. 実行推奨コマンド
    print("\n" + "=" * 80)
    print("実行推奨コマンド")
    print("=" * 80)
    
    print("\n1. データセット統合確認:")
    print("   py -3.12 scripts/data_processing/integrate_domain_knowledge.py --update-config")
    
    print("\n2. データパイプライン実行（フェーズ1のみ）:")
    print("   py -3.12 scripts/run_sunset_pipeline.py --phase data")
    
    print("\n3. 完全なサンセットパイプライン実行:")
    print("   py -3.12 scripts/run_sunset_pipeline.py --phase full")
    
    print("\n" + "=" * 80)
    print("状態確認完了")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    check_sunset_pipeline_status()
