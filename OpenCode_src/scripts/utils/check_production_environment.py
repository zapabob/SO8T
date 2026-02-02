#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本番環境確認スクリプト

ディスク容量、ネットワーク接続、依存ライブラリなどを確認
"""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionEnvironmentChecker:
    """本番環境チェッカー"""
    
    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []
        self.output_dir = Path("D:/webdataset")
        self.required_libraries = [
            'requests', 'beautifulsoup4', 'yaml', 'tqdm', 'numpy'
        ]
    
    def check_disk_space(self) -> Tuple[bool, str]:
        """ディスク容量確認"""
        try:
            if os.name == 'nt':  # Windows
                drive = str(self.output_dir)[0] + ":"
                total, used, free = shutil.disk_usage(drive)
                free_gb = free / (1024**3)
                
                required_gb = 100.0  # 100GB推奨
                
                if free_gb >= required_gb:
                    return True, f"ディスク容量: {free_gb:.1f} GB (十分)"
                else:
                    return False, f"ディスク容量: {free_gb:.1f} GB (不足: {required_gb}GB以上推奨)"
            else:
                # Linux/Mac
                total, used, free = shutil.disk_usage(self.output_dir)
                free_gb = free / (1024**3)
                required_gb = 100.0
                
                if free_gb >= required_gb:
                    return True, f"ディスク容量: {free_gb:.1f} GB (十分)"
                else:
                    return False, f"ディスク容量: {free_gb:.1f} GB (不足: {required_gb}GB以上推奨)"
        except Exception as e:
            return False, f"ディスク容量確認エラー: {e}"
    
    def check_output_directory(self) -> Tuple[bool, str]:
        """出力ディレクトリ確認"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 書き込みテスト
            test_file = self.output_dir / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                return True, f"出力ディレクトリ: {self.output_dir} (書き込み可能)"
            except Exception as e:
                return False, f"出力ディレクトリ書き込みエラー: {e}"
        except Exception as e:
            return False, f"出力ディレクトリ確認エラー: {e}"
    
    def check_network_connection(self) -> Tuple[bool, str]:
        """ネットワーク接続確認"""
        test_urls = [
            "https://www.e-gov.go.jp/",
            "https://zenn.dev/",
            "https://qiita.com/",
            "https://ja.wikipedia.org/"
        ]
        
        try:
            import requests
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        return True, f"ネットワーク接続: OK ({url})"
                except Exception:
                    continue
            return False, "ネットワーク接続: 一部のサイトに接続できません"
        except ImportError:
            return False, "ネットワーク接続確認: requestsライブラリが見つかりません"
        except Exception as e:
            return False, f"ネットワーク接続確認エラー: {e}"
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Pythonバージョン確認"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            return True, f"Pythonバージョン: {version.major}.{version.minor}.{version.micro} (OK)"
        else:
            return False, f"Pythonバージョン: {version.major}.{version.minor}.{version.micro} (Python 3.8以上推奨)"
    
    def check_required_libraries(self) -> Tuple[bool, str]:
        """依存ライブラリ確認"""
        missing = []
        for lib in self.required_libraries:
            try:
                if lib == 'beautifulsoup4':
                    __import__('bs4')
                else:
                    __import__(lib)
            except ImportError:
                missing.append(lib)
        
        if not missing:
            return True, f"依存ライブラリ: すべてインストール済み"
        else:
            return False, f"依存ライブラリ不足: {', '.join(missing)}"
    
    def check_checkpoint_directory(self) -> Tuple[bool, str]:
        """チェックポイントディレクトリ確認"""
        checkpoint_dir = Path("D:/webdataset/checkpoints/complete_ab_pipeline")
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 書き込みテスト
            test_file = checkpoint_dir / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                return True, f"チェックポイントディレクトリ: {checkpoint_dir} (書き込み可能)"
            except Exception as e:
                return False, f"チェックポイントディレクトリ書き込みエラー: {e}"
        except Exception as e:
            return False, f"チェックポイントディレクトリ確認エラー: {e}"
    
    def check_log_directory(self) -> Tuple[bool, str]:
        """ログディレクトリ確認"""
        log_dir = Path("logs")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            return True, f"ログディレクトリ: {log_dir} (準備完了)"
        except Exception as e:
            return False, f"ログディレクトリ確認エラー: {e}"
    
    def run_all_checks(self) -> Dict[str, Tuple[bool, str]]:
        """すべてのチェックを実行"""
        results = {}
        
        logger.info("="*80)
        logger.info("本番環境確認開始")
        logger.info("="*80)
        
        # 各チェックを実行
        checks = [
            ("Pythonバージョン", self.check_python_version),
            ("依存ライブラリ", self.check_required_libraries),
            ("ディスク容量", self.check_disk_space),
            ("出力ディレクトリ", self.check_output_directory),
            ("チェックポイントディレクトリ", self.check_checkpoint_directory),
            ("ログディレクトリ", self.check_log_directory),
            ("ネットワーク接続", self.check_network_connection),
        ]
        
        for name, check_func in checks:
            logger.info(f"[CHECK] {name}...")
            success, message = check_func()
            results[name] = (success, message)
            
            if success:
                logger.info(f"[OK] {message}")
            else:
                logger.warning(f"[NG] {message}")
        
        return results
    
    def print_summary(self, results: Dict[str, Tuple[bool, str]]):
        """サマリー表示"""
        logger.info("="*80)
        logger.info("環境確認サマリー")
        logger.info("="*80)
        
        all_ok = True
        for name, (success, message) in results.items():
            status = "[OK]" if success else "[NG]"
            logger.info(f"{status} {name}: {message}")
            if not success:
                all_ok = False
        
        logger.info("="*80)
        if all_ok:
            logger.info("[SUCCESS] すべての環境確認が成功しました")
            return 0
        else:
            logger.warning("[WARNING] 一部の環境確認が失敗しました。上記を確認してください。")
            return 1


def main():
    """メイン関数"""
    checker = ProductionEnvironmentChecker()
    results = checker.run_all_checks()
    exit_code = checker.print_summary(results)
    
    # 音声通知
    audio_file = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
    if audio_file.exists():
        try:
            import subprocess
            ps_cmd = f"""
            if (Test-Path '{audio_file}') {{
                Add-Type -AssemblyName System.Windows.Forms
                $player = New-Object System.Media.SoundPlayer '{audio_file}'
                $player.PlaySync()
                Write-Host '[OK] 音声通知送信完了' -ForegroundColor Green
            }}
            """
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                cwd=str(PROJECT_ROOT),
                check=False
            )
        except Exception as e:
            logger.warning(f"音声通知失敗: {e}")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

