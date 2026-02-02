#!/usr/bin/env python3
"""
大規模データセット・モデルファイルをH:\from_DからSO8Tリポジトリにコピー
tqdmで進捗を表示しながら安全にコピー
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import json

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] tqdm not installed. Install with: pip install tqdm")
    sys.exit(1)

class LargeFileCopier:
    """大規模ファイルを安全にコピー"""
    
    def __init__(self, source_root: str, dest_root: str):
        self.source_root = Path(source_root)
        self.dest_root = Path(dest_root)
        self.copied_files = []
        self.failed_files = []
        self.total_size = 0
        self.copied_size = 0
        
    def get_total_size(self, paths: List[Path]) -> int:
        """ファイル/ディレクトリの合計サイズを計算"""
        total = 0
        for path in paths:
            if path.is_file():
                total += path.stat().st_size
            elif path.is_dir():
                for root, dirs, files in os.walk(path):
                    for file in files:
                        try:
                            total += Path(root) / file
                        except:
                            pass
        return total
    
    def format_size(self, size_bytes: int) -> str:
        """バイト数を読みやすい形式に変換"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def copy_file_with_progress(self, src: Path, dst: Path) -> bool:
        """ファイルを進捗表示付きでコピー"""
        try:
            # 親ディレクトリを作成
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # ファイルサイズを取得
            file_size = src.stat().st_size
            
            # 既に存在する場合はスキップ（オプションで上書き可能）
            if dst.exists() and dst.stat().st_size == file_size:
                print(f"[SKIP] {dst.name} (already exists, same size)")
                return True
            
            # コピー実行（tqdmで進捗表示）
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst, \
                 tqdm(total=file_size, unit='B', unit_scale=True, 
                      desc=f"Copying {src.name[:50]}", 
                      ncols=100) as pbar:
                while True:
                    chunk = fsrc.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    fdst.write(chunk)
                    pbar.update(len(chunk))
            
            self.copied_size += file_size
            self.copied_files.append({
                'source': str(src),
                'dest': str(dst),
                'size': file_size
            })
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to copy {src.name}: {e}")
            self.failed_files.append({
                'source': str(src),
                'dest': str(dst),
                'error': str(e)
            })
            return False
    
    def copy_directory(self, src_dir: Path, dst_dir: Path, 
                      patterns: Optional[List[str]] = None) -> Tuple[int, int]:
        """ディレクトリを再帰的にコピー"""
        copied_count = 0
        failed_count = 0
        
        # パターンマッチング用
        if patterns:
            import fnmatch
        
        # ファイルを収集
        files_to_copy = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                src_file = Path(root) / file
                rel_path = src_file.relative_to(src_dir)
                dst_file = dst_dir / rel_path
                
                # パターンフィルタリング
                if patterns:
                    if not any(fnmatch.fnmatch(str(rel_path), p) for p in patterns):
                        continue
                
                files_to_copy.append((src_file, dst_file))
        
        # 合計サイズを計算
        total_size = sum(f[0].stat().st_size for f in files_to_copy)
        self.total_size += total_size
        
        print(f"\n[INFO] Copying {len(files_to_copy)} files ({self.format_size(total_size)})")
        
        # ファイルをコピー
        for src_file, dst_file in tqdm(files_to_copy, desc="Overall progress", ncols=100):
            if self.copy_file_with_progress(src_file, dst_file):
                copied_count += 1
            else:
                failed_count += 1
        
        return copied_count, failed_count
    
    def copy_so8t_models(self) -> dict:
        """SO8Tモデルをコピー"""
        print("\n" + "="*80)
        print("[COPY] SO8T Models from H:\\from_D\\SO8T_models")
        print("="*80)
        
        source = self.source_root / "SO8T_models"
        dest = self.dest_root / "models" / "from_h_drive"
        
        if not source.exists():
            print(f"[WARN] Source not found: {source}")
            return {'copied': 0, 'failed': 0}
        
        return self._copy_directory_safe(source, dest)
    
    def copy_webdataset_models(self) -> dict:
        """webdatasetのモデルをコピー"""
        print("\n" + "="*80)
        print("[COPY] Models from H:\\from_D\\webdataset\\models")
        print("="*80)
        
        source = self.source_root / "webdataset" / "models"
        dest = self.dest_root / "models" / "from_webdataset"
        
        if not source.exists():
            print(f"[WARN] Source not found: {source}")
            return {'copied': 0, 'failed': 0}
        
        return self._copy_directory_safe(source, dest)
    
    def copy_webdataset_gguf(self) -> dict:
        """webdatasetのGGUFモデルをコピー"""
        print("\n" + "="*80)
        print("[COPY] GGUF Models from H:\\from_D\\webdataset\\gguf_models")
        print("="*80)
        
        source = self.source_root / "webdataset" / "gguf_models"
        dest = self.dest_root / "gguf_models" / "from_webdataset"
        
        if not source.exists():
            print(f"[WARN] Source not found: {source}")
            return {'copied': 0, 'failed': 0}
        
        return self._copy_directory_safe(source, dest)
    
    def copy_webdataset_checkpoints(self) -> dict:
        """webdatasetのチェックポイントをコピー"""
        print("\n" + "="*80)
        print("[COPY] Checkpoints from H:\\from_D\\webdataset\\checkpoints")
        print("="*80)
        
        source = self.source_root / "webdataset" / "checkpoints"
        dest = self.dest_root / "checkpoints" / "from_webdataset"
        
        if not source.exists():
            print(f"[WARN] Source not found: {source}")
            return {'copied': 0, 'failed': 0}
        
        return self._copy_directory_safe(source, dest)
    
    def copy_webdataset_datasets(self) -> dict:
        """webdatasetのデータセットをコピー"""
        print("\n" + "="*80)
        print("[COPY] Datasets from H:\\from_D\\webdataset\\datasets")
        print("="*80)
        
        source = self.source_root / "webdataset" / "datasets"
        dest = self.dest_root / "data" / "from_webdataset"
        
        if not source.exists():
            print(f"[WARN] Source not found: {source}")
            return {'copied': 0, 'failed': 0}
        
        return self._copy_directory_safe(source, dest)
    
    def copy_so8t_transformar(self) -> dict:
        """SO8T-transformarのモデルをコピー"""
        print("\n" + "="*80)
        print("[COPY] Models from H:\\from_D\\SO8T-transformar\\models")
        print("="*80)
        
        source = self.source_root / "SO8T-transformar" / "models"
        dest = self.dest_root / "models" / "from_so8t_transformar"
        
        if not source.exists():
            print(f"[WARN] Source not found: {source}")
            return {'copied': 0, 'failed': 0}
        
        return self._copy_directory_safe(source, dest)
    
    def _copy_directory_safe(self, source: Path, dest: Path) -> dict:
        """ディレクトリを安全にコピー"""
        try:
            copied, failed = self.copy_directory(source, dest)
            return {'copied': copied, 'failed': failed}
        except Exception as e:
            print(f"[ERROR] Failed to copy directory {source}: {e}")
            return {'copied': 0, 'failed': 1}
    
    def generate_report(self, report_path: Optional[Path] = None):
        """コピー結果レポートを生成"""
        if report_path is None:
            report_path = self.dest_root / "_docs" / f"{datetime.now().strftime('%Y-%m-%d')}_大規模ファイルコピー結果.md"
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 大規模ファイルコピー結果レポート\n\n")
            f.write(f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**ソース**: {self.source_root}\n")
            f.write(f"**コピー先**: {self.dest_root}\n\n")
            
            f.write(f"## 統計\n\n")
            f.write(f"- **コピー成功**: {len(self.copied_files)} ファイル\n")
            f.write(f"- **コピー失敗**: {len(self.failed_files)} ファイル\n")
            f.write(f"- **合計サイズ**: {self.format_size(self.copied_size)}\n\n")
            
            if self.copied_files:
                f.write("## コピー成功ファイル\n\n")
                for item in self.copied_files[:50]:  # 最初の50個
                    f.write(f"- `{item['source']}` → `{item['dest']}` ({self.format_size(item['size'])})\n")
                if len(self.copied_files) > 50:
                    f.write(f"\n... 他 {len(self.copied_files) - 50} ファイル\n")
            
            if self.failed_files:
                f.write("\n## コピー失敗ファイル\n\n")
                for item in self.failed_files:
                    f.write(f"- `{item['source']}`: {item['error']}\n")
        
        print(f"\n[REPORT] Report saved: {report_path}")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Copy large datasets and models from H:\\from_D to SO8T repository')
    parser.add_argument('--source', default='H:\\from_D',
                       help='Source directory (default: H:\\from_D)')
    parser.add_argument('--dest', default=None,
                       help='Destination directory (default: current SO8T repository)')
    parser.add_argument('--models', action='store_true',
                       help='Copy models')
    parser.add_argument('--gguf', action='store_true',
                       help='Copy GGUF models')
    parser.add_argument('--checkpoints', action='store_true',
                       help='Copy checkpoints')
    parser.add_argument('--datasets', action='store_true',
                       help='Copy datasets')
    parser.add_argument('--all', action='store_true',
                       help='Copy all (models, GGUF, checkpoints, datasets)')
    
    args = parser.parse_args()
    
    # デフォルトのコピー先を設定
    if args.dest is None:
        # スクリプトの場所からリポジトリルートを推定
        script_path = Path(__file__).resolve()
        repo_root = script_path.parent.parent.parent
        args.dest = str(repo_root)
    
    copier = LargeFileCopier(args.source, args.dest)
    
    results = {}
    
    if args.all or args.models:
        results['so8t_models'] = copier.copy_so8t_models()
        results['webdataset_models'] = copier.copy_webdataset_models()
        results['so8t_transformar'] = copier.copy_so8t_transformar()
    
    if args.all or args.gguf:
        results['gguf_models'] = copier.copy_webdataset_gguf()
    
    if args.all or args.checkpoints:
        results['checkpoints'] = copier.copy_webdataset_checkpoints()
    
    if args.all or args.datasets:
        results['datasets'] = copier.copy_webdataset_datasets()
    
    # レポート生成
    copier.generate_report()
    
    # 結果サマリー
    print("\n" + "="*80)
    print("[SUMMARY] Copy Results")
    print("="*80)
    for category, result in results.items():
        print(f"{category}: {result['copied']} copied, {result['failed']} failed")
    print(f"\nTotal: {len(copier.copied_files)} files copied, {len(copier.failed_files)} failed")
    print(f"Total size: {copier.format_size(copier.copied_size)}")


if __name__ == "__main__":
    main()
