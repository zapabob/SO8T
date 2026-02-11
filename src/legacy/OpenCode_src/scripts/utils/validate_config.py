#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
設定ファイル検証スクリプト

YAML設定ファイルの構造と値を検証
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

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


class ConfigValidator:
    """設定ファイル検証クラス"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def load_config(self) -> bool:
        """設定ファイル読み込み"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"[OK] 設定ファイル読み込み成功: {self.config_path}")
            return True
        except FileNotFoundError:
            self.errors.append(f"設定ファイルが見つかりません: {self.config_path}")
            return False
        except yaml.YAMLError as e:
            self.errors.append(f"YAML解析エラー: {e}")
            return False
        except Exception as e:
            self.errors.append(f"設定ファイル読み込みエラー: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """設定構造の検証"""
        required_sections = [
            'model_a', 'model_b', 'benchmarks', 'checkpoint', 'progress', 'data_sources'
        ]
        
        for section in required_sections:
            if section not in self.config:
                self.errors.append(f"必須セクションが見つかりません: {section}")
                return False
        
        logger.info("[OK] 設定構造検証成功")
        return True
    
    def validate_data_sources(self) -> bool:
        """データソース設定の検証"""
        if 'data_sources' not in self.config:
            self.errors.append("data_sourcesセクションが見つかりません")
            return False
        
        ds_config = self.config['data_sources']
        
        # 必須設定の確認
        required_settings = [
            'enable_specialized_crawlers',
            'enable_parallel_crawler',
            'max_pages_per_source'
        ]
        
        for setting in required_settings:
            if setting not in ds_config:
                self.errors.append(f"data_sources.{setting}が見つかりません")
                return False
        
        # データソース有効化確認
        data_sources = [
            'enable_kanpou_4web',
            'enable_egov',
            'enable_zenn',
            'enable_qiita',
            'enable_wikipedia_ja'
        ]
        
        enabled_sources = []
        for source in data_sources:
            if source in ds_config and ds_config[source]:
                enabled_sources.append(source.replace('enable_', ''))
        
        if not enabled_sources:
            self.warnings.append("有効なデータソースがありません")
        else:
            logger.info(f"[OK] 有効なデータソース: {', '.join(enabled_sources)}")
        
        # max_pages_per_sourceの値確認
        max_pages = ds_config.get('max_pages_per_source', 0)
        if max_pages <= 0:
            self.errors.append("max_pages_per_sourceは1以上である必要があります")
            return False
        
        logger.info("[OK] データソース設定検証成功")
        return True
    
    def validate_paths(self) -> bool:
        """パス設定の検証"""
        # model_aの出力ディレクトリ確認
        if 'model_a' in self.config:
            output_dir = self.config['model_a'].get('output_dir', '')
            if output_dir:
                output_path = Path(output_dir)
                if not output_path.parent.exists():
                    self.warnings.append(f"model_a出力ディレクトリの親ディレクトリが存在しません: {output_path.parent}")
        
        # checkpoint保存ディレクトリ確認
        if 'checkpoint' in self.config:
            checkpoint_dir = self.config['checkpoint'].get('save_dir', '')
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir)
                if not checkpoint_path.parent.exists():
                    self.warnings.append(f"チェックポイント保存ディレクトリの親ディレクトリが存在しません: {checkpoint_path.parent}")
        
        logger.info("[OK] パス設定検証成功")
        return True
    
    def validate_values(self) -> bool:
        """値の検証"""
        # checkpoint間隔の確認
        if 'checkpoint' in self.config:
            interval = self.config['checkpoint'].get('interval_seconds', 0)
            if interval <= 0:
                self.errors.append("checkpoint.interval_secondsは1以上である必要があります")
                return False
        
        # max_pages_per_sourceの確認
        if 'data_sources' in self.config:
            max_pages = self.config['data_sources'].get('max_pages_per_source', 0)
            if max_pages <= 0:
                self.errors.append("data_sources.max_pages_per_sourceは1以上である必要があります")
                return False
        
        logger.info("[OK] 値検証成功")
        return True
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """すべての検証を実行"""
        if not self.load_config():
            return False, self.errors, self.warnings
        
        validations = [
            ("構造検証", self.validate_structure),
            ("データソース設定検証", self.validate_data_sources),
            ("パス設定検証", self.validate_paths),
            ("値検証", self.validate_values),
        ]
        
        all_ok = True
        for name, validation_func in validations:
            logger.info(f"[VALIDATE] {name}...")
            if not validation_func():
                all_ok = False
        
        return all_ok, self.errors, self.warnings
    
    def print_summary(self, success: bool, errors: List[str], warnings: List[str]):
        """サマリー表示"""
        logger.info("="*80)
        logger.info("設定ファイル検証サマリー")
        logger.info("="*80)
        
        if errors:
            logger.error("[ERRORS]")
            for error in errors:
                logger.error(f"  - {error}")
        
        if warnings:
            logger.warning("[WARNINGS]")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("="*80)
        if success and not errors:
            logger.info("[SUCCESS] 設定ファイル検証が成功しました")
            return 0
        else:
            logger.error("[FAILED] 設定ファイル検証が失敗しました")
            return 1


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="設定ファイル検証")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/complete_automated_ab_pipeline.yaml',
        help='設定ファイルパス'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    
    validator = ConfigValidator(config_path)
    success, errors, warnings = validator.validate_all()
    exit_code = validator.print_summary(success, errors, warnings)
    
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

