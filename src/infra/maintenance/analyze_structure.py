#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Repository Structure Analyzer
リポジトリ構造分析スクリプト
"""

import os
from pathlib import Path
from collections import defaultdict
import json


def analyze_directory_structure(root_path="scripts"):
    """
    ディレクトリ構造を分析
    Analyze directory structure
    """
    root = Path(root_path)
    structure = defaultdict(lambda: {"files": 0, "subdirs": [], "file_types": defaultdict(int)})

    def analyze_dir(path, current_depth=0, max_depth=3):
        if current_depth > max_depth:
            return

        for item in path.iterdir():
            if item.is_file():
                structure[str(path)]["files"] += 1
                ext = item.suffix.lower()
                structure[str(path)]["file_types"][ext] += 1
            elif item.is_dir() and not item.name.startswith('.'):
                structure[str(path)]["subdirs"].append(item.name)
                analyze_dir(item, current_depth + 1, max_depth)

    analyze_dir(root)

    return dict(structure)


def find_large_directories(structure, threshold=50):
    """大規模ディレクトリを特定"""
    large_dirs = {}
    for path, info in structure.items():
        if info["files"] >= threshold:
            large_dirs[path] = info["files"]
    return large_dirs


def find_scattered_configs():
    """散在する設定ファイルを特定"""
    config_files = []
    config_extensions = ['.yaml', '.yml', '.json', '.toml', '.ini', '.cfg']

    for root, dirs, files in os.walk('.'):
        # .git, __pycache__, node_modulesなどは除外
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

        for file in files:
            if any(file.endswith(ext) for ext in config_extensions):
                config_files.append(os.path.join(root, file))

    return config_files


def find_duplicate_utilities():
    """重複ユーティリティを特定"""
    utilities = defaultdict(list)

    # utilsディレクトリのファイルを分析
    utils_path = Path("scripts/utils")
    if utils_path.exists():
        for file_path in utils_path.rglob("*.py"):
            if file_path.is_file():
                # ファイル名から機能を推測
                name = file_path.stem.lower()
                utilities[name].append(str(file_path))

    # 重複しているものを特定（同じ機能名のファイルが複数）
    duplicates = {k: v for k, v in utilities.items() if len(v) > 1}
    return duplicates


def analyze_test_structure():
    """テスト構造を分析"""
    test_structure = {}

    tests_path = Path("tests")
    scripts_testing_path = Path("scripts/testing")

    if tests_path.exists():
        test_structure["tests_dir"] = len(list(tests_path.rglob("*.py")))

    if scripts_testing_path.exists():
        test_structure["scripts_testing_dir"] = len(list(scripts_testing_path.rglob("*.py")))

    return test_structure


def main():
    """メイン関数"""
    print("=" * 60)
    print("SO8T Repository Structure Analysis")
    print("=" * 60)

    # 1. ディレクトリ構造分析
    print("\n1. Directory Structure Analysis")
    print("-" * 40)
    structure = analyze_directory_structure()

    for path, info in sorted(structure.items()):
        print(f"{path}: {info['files']} files")
        if info['subdirs']:
            print(f"  Subdirs: {', '.join(info['subdirs'][:5])}{'...' if len(info['subdirs']) > 5 else ''}")

    # 2. 大規模ディレクトリ特定
    print("\n2. Large Directories (50+ files)")
    print("-" * 40)
    large_dirs = find_large_directories(structure)
    for path, count in sorted(large_dirs.items(), key=lambda x: x[1], reverse=True):
        print(f"{path}: {count} files")

    # 3. 散在設定ファイル分析
    print("\n3. Scattered Configuration Files")
    print("-" * 40)
    config_files = find_scattered_configs()
    config_locations = defaultdict(int)
    for config_file in config_files:
        location = config_file.split('/')[0] if '/' in config_file else 'root'
        config_locations[location] += 1

    for location, count in sorted(config_locations.items(), key=lambda x: x[1], reverse=True):
        print(f"{location}: {count} config files")

    # 4. 重複ユーティリティ分析
    print("\n4. Duplicate Utilities")
    print("-" * 40)
    duplicates = find_duplicate_utilities()
    if duplicates:
        for name, files in sorted(duplicates.items())[:10]:  # 最初の10個のみ表示
            print(f"{name}: {len(files)} files")
            for file in files[:3]:  # 最初の3ファイルのみ表示
                print(f"  - {file}")
    else:
        print("No duplicate utilities found")

    # 5. テスト構造分析
    print("\n5. Test Structure Analysis")
    print("-" * 40)
    test_structure = analyze_test_structure()
    for location, count in test_structure.items():
        print(f"{location}: {count} test files")

    # 6. 改善提案
    print("\n6. Improvement Recommendations")
    print("-" * 40)

    if large_dirs:
        print("Large directories to split:")
        for path, count in sorted(large_dirs.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  - {path} ({count} files)")

    if len(config_files) > 20:
        print(f"Too many scattered config files ({len(config_files)} total)")
        print("Consider consolidating into configs/ directory")

    if duplicates:
        print(f"Found {len(duplicates)} duplicate utility functions")
        print("Consider creating common libraries")

    total_tests = sum(test_structure.values())
    if total_tests > 50:
        print(f"Large test codebase ({total_tests} files)")
        print("Consider organizing by functionality")

    # 結果をJSONで保存
    result = {
        "structure": structure,
        "large_dirs": large_dirs,
        "config_files": config_files,
        "duplicates": dict(duplicates),
        "test_structure": test_structure
    }

    with open("_docs/structure_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis saved to _docs/structure_analysis.json")


if __name__ == '__main__':
    main()


