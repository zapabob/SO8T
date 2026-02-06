#!/usr/bin/env python3
"""
Unsloth互換性のためのパッケージバージョン調整
datasetsとtransformersをUnsloth互換バージョンにダウングレード
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def check_current_versions():
    """現在のバージョンを確認"""
    print("[CHECK] Checking current package versions...")
    
    try:
        import datasets
        datasets_version = datasets.__version__
        print(f"  datasets: {datasets_version}")
    except ImportError:
        datasets_version = None
        print("  datasets: Not installed")
    
    try:
        import transformers
        transformers_version = transformers.__version__
        print(f"  transformers: {transformers_version}")
    except ImportError:
        transformers_version = None
        print("  transformers: Not installed")
    
    try:
        import unsloth
        unsloth_version = unsloth.__version__
        print(f"  unsloth: {unsloth_version}")
    except (ImportError, AttributeError):
        try:
            import unsloth_zoo
            unsloth_version = unsloth_zoo.__version__
            print(f"  unsloth-zoo: {unsloth_version}")
        except (ImportError, AttributeError):
            unsloth_version = None
            print("  unsloth: Not installed")
    
    return datasets_version, transformers_version, unsloth_version

def install_compatible_versions(python_cmd: str = "py -3.12"):
    """Unsloth互換バージョンをインストール"""
    print("\n" + "="*80)
    print("[INSTALL] Installing Unsloth-compatible versions")
    print("="*80)
    
    # Unsloth互換バージョン
    # unsloth-zoo 2025.11.6 requirements:
    # - datasets: !=4.0.*,!=4.1.0,<4.4.0,>=3.4.1
    # - transformers: !=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3,!=4.53.0,!=4.54.0,!=4.55.0,!=4.55.1,<=4.57.2,>=4.51.3
    # - huggingface-hub: transformersが要求する <1.0,>=0.34.0
    
    # 最新の互換バージョンを指定
    # datasets: 4.3.x系の最新
    # transformers: 4.57.2 (Unslothがサポートする最新)
    # huggingface-hub: <1.0 (transformersの要求)
    
    target_versions = {
        'huggingface-hub': '>=0.34.0,<1.0',
        'datasets': '>=3.4.1,<4.4.0',
        'transformers': '>=4.51.3,<=4.57.2'
    }
    
    print(f"\n[INFO] Target versions:")
    print(f"  huggingface-hub: {target_versions['huggingface-hub']}")
    print(f"  datasets: {target_versions['datasets']}")
    print(f"  transformers: {target_versions['transformers']}")
    
    # パッケージをインストール
    for package, version_spec in target_versions.items():
        print(f"\n[INSTALL] Installing {package} {version_spec}...")
        cmd = f"{python_cmd} -m pip install '{package}{version_spec}' --upgrade"
        print(f"[CMD] {cmd}")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[OK] {package} installed successfully")
            if result.stdout:
                # インストールされたバージョンを表示
                for line in result.stdout.split('\n'):
                    if 'Successfully installed' in line or 'Requirement already satisfied' in line:
                        print(f"  {line}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install {package}: {e}")
            print(f"[STDERR] {e.stderr}")
            return False
    
    return True

def verify_versions():
    """インストール後のバージョンを確認"""
    print("\n" + "="*80)
    print("[VERIFY] Verifying installed versions")
    print("="*80)
    
    datasets_version, transformers_version, unsloth_version = check_current_versions()
    
    # バージョンチェック
    issues = []
    
    if datasets_version:
        from packaging import version
        v = version.parse(datasets_version)
        if v >= version.parse("4.4.0") or v < version.parse("3.4.1"):
            issues.append(f"datasets {datasets_version} is not compatible (need >=3.4.1,<4.4.0)")
        elif v >= version.parse("4.0.0") and v < version.parse("4.1.0"):
            issues.append(f"datasets {datasets_version} is not compatible (excluded: 4.0.x)")
        elif v >= version.parse("4.1.0") and v < version.parse("4.2.0"):
            issues.append(f"datasets {datasets_version} is not compatible (excluded: 4.1.0)")
        else:
            print(f"[OK] datasets {datasets_version} is compatible")
    
    if transformers_version:
        from packaging import version
        v = version.parse(transformers_version)
        excluded_versions = [
            "4.52.0", "4.52.1", "4.52.2", "4.52.3",
            "4.53.0", "4.54.0", "4.55.0", "4.55.1"
        ]
        if v > version.parse("4.57.2") or v < version.parse("4.51.3"):
            issues.append(f"transformers {transformers_version} is not compatible (need >=4.51.3,<=4.57.2)")
        elif transformers_version in excluded_versions:
            issues.append(f"transformers {transformers_version} is excluded by Unsloth")
        else:
            print(f"[OK] transformers {transformers_version} is compatible")
    
    if issues:
        print("\n[WARN] Compatibility issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[SUCCESS] All versions are compatible with Unsloth!")
        return True

def test_unsloth_import():
    """Unslothのインポートをテスト"""
    print("\n" + "="*80)
    print("[TEST] Testing Unsloth import")
    print("="*80)
    
    try:
        from unsloth import FastLanguageModel
        print("[OK] unsloth.FastLanguageModel imported successfully")
        return True
    except ImportError as e:
        print(f"[ERROR] Failed to import unsloth: {e}")
        print("[INFO] Install Unsloth with: pip install unsloth[colab-new]")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error importing unsloth: {e}")
        return False

def generate_report(python_cmd: str, success: bool, report_path: Path = None):
    """レポートを生成"""
    if report_path is None:
        report_path = Path(__file__).parent.parent.parent / "_docs" / f"{datetime.now().strftime('%Y-%m-%d')}_Unsloth互換性調整結果.md"
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    datasets_version, transformers_version, unsloth_version = check_current_versions()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Unsloth互換性調整結果レポート\n\n")
        f.write(f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Python**: {python_cmd}\n\n")
        
        f.write(f"## インストール済みバージョン\n\n")
        f.write(f"- **datasets**: {datasets_version or 'Not installed'}\n")
        f.write(f"- **transformers**: {transformers_version or 'Not installed'}\n")
        f.write(f"- **unsloth**: {unsloth_version or 'Not installed'}\n\n")
        
        f.write(f"## 互換性要件\n\n")
        f.write(f"### Unsloth互換バージョン\n\n")
        f.write(f"- **huggingface-hub**: `>=0.34.0,<1.0` (transformersの要求)\n")
        f.write(f"- **datasets**: `>=3.4.1,<4.4.0,!=4.0.*,!=4.1.0`\n")
        f.write(f"- **transformers**: `>=4.51.3,<=4.57.2,!=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3,!=4.53.0,!=4.54.0,!=4.55.0,!=4.55.1`\n\n")
        
        f.write(f"## 調整結果\n\n")
        if success:
            f.write(f"[OK] **成功**: Unsloth互換バージョンに調整完了\n\n")
        else:
            f.write(f"[NG] **失敗**: 互換性の問題が残っています\n\n")
        
        f.write(f"## 実行コマンド\n\n")
        f.write(f"```bash\n")
        f.write(f"{python_cmd} -m pip install 'huggingface-hub>=0.34.0,<1.0' --upgrade\n")
        f.write(f"{python_cmd} -m pip install 'datasets>=3.4.1,<4.4.0' --upgrade\n")
        f.write(f"{python_cmd} -m pip install 'transformers>=4.51.3,<=4.57.2' --upgrade\n")
        f.write(f"```\n")
    
    print(f"\n[REPORT] Report saved: {report_path}")

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix Unsloth compatibility by adjusting package versions')
    parser.add_argument('--python', default='py -3.12',
                       help='Python command (default: py -3.12)')
    parser.add_argument('--test-import', action='store_true',
                       help='Test Unsloth import after installation')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip installation, only verify versions')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Unsloth互換性調整スクリプト")
    print("="*80)
    
    # 現在のバージョンを確認
    datasets_version, transformers_version, unsloth_version = check_current_versions()
    
    # インストール（スキップしない場合）
    if not args.skip_install:
        success = install_compatible_versions(args.python)
        if not success:
            print("\n[ERROR] Installation failed. Please check the errors above.")
            sys.exit(1)
    else:
        print("\n[SKIP] Skipping installation (--skip-install)")
    
    # バージョンを確認
    success = verify_versions()
    
    # Unslothインポートテスト
    if args.test_import:
        import_success = test_unsloth_import()
        success = success and import_success
    
    # レポート生成
    generate_report(args.python, success)
    
    if success:
        print("\n" + "="*80)
        print("[SUCCESS] Unsloth compatibility fix completed!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("[WARN] Some compatibility issues remain. Please check the report.")
        print("="*80)
        sys.exit(1)

if __name__ == "__main__":
    main()
