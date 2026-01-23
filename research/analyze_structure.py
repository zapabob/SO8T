#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リポジトリ構造分析スクリプト
重複ファイルと整理が必要な箇所を特定します
"""

import os
from pathlib import Path
from collections import defaultdict, Counter
import json

def analyze_repo_structure(root_path='.'):
    """リポジトリ構造を詳細に分析"""
    root = Path(root_path)
    structure = defaultdict(lambda: defaultdict(list))
    file_types = Counter()
    duplicates = defaultdict(list)

    # ファイルの収集
    for path in root.rglob('*'):
        if path.is_file() and not any(part.startswith('.') for part in path.parts):
            # ファイルタイプのカウント
            ext = path.suffix.lower()
            file_types[ext] += 1

            # 構造分析
            rel_path = path.relative_to(root)
            parts = rel_path.parts

            if len(parts) > 1:
                category = parts[0]
                if len(parts) > 2:
                    subcategory = '/'.join(parts[1:-1])
                    structure[category][subcategory].append(rel_path.name)
                else:
                    structure[category]['root_files'].append(rel_path.name)

            # 重複ファイルの検出（名前ベース）
            filename = path.name.lower()
            duplicates[filename].append(str(rel_path))

    # 重複ファイルのフィルタリング（3つ以上同じ名前、または重要な設定ファイル）
    significant_duplicates = {}
    for filename, paths in duplicates.items():
        if len(paths) > 2 or any(keyword in filename for keyword in ['config', 'pipeline', 'train']):
            significant_duplicates[filename] = paths

    return {
        'structure': dict(structure),
        'file_types': dict(file_types),
        'duplicates': significant_duplicates,
        'total_files': sum(file_types.values())
    }

def main():
    """メイン関数"""
    print('=== リポジトリ構造分析 ===')

    # 分析実行
    result = analyze_repo_structure()
    print(f'総ファイル数: {result["total_files"]}')
    print(f'ファイルタイプ数: {len(result["file_types"])}')

    print('\n=== トップレベルディレクトリ ===')
    for category, subcats in result['structure'].items():
        total_files = sum(len(files) for files in subcats.values())
        print(f'{category}/: {total_files} files')

    print('\n=== 重複ファイル（3つ以上または設定ファイル） ===')
    for filename, paths in list(result['duplicates'].items())[:30]:  # 最初の30個のみ表示
        print(f'{filename}: {len(paths)} 個')
        for path in paths[:5]:  # 各ファイルの最初の5つのパスを表示
            print(f'  - {path}')

    # JSON保存
    output_file = '_docs/repo_structure_analysis.json'
    os.makedirs('_docs', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'\n分析結果を {output_file} に保存しました')

    # 整理計画の提案
    print('\n=== 整理計画の提案 ===')
    consolidation_plan = {
        'config_consolidation': 'configs/ と so8t/config/ を統合',
        'docs_consolidation': 'docs/ と _docs/ を統合',
        'scripts_cleanup': 'scripts/ 内の重複スクリプトを整理',
        'core_modules': 'so8t_core/ と src/so8t_core/ を統合',
        'logs_centralization': '散らばったログファイルを logs/ に統合',
        'models_organization': 'モデル関連ディレクトリを models/ に統合'
    }

    for plan, description in consolidation_plan.items():
        print(f'- {plan}: {description}')

if __name__ == '__main__':
    main()
