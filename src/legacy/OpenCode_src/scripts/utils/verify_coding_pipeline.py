#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コーディングパイプライン動作確認スクリプト

Phase 4-8の各フェーズの動作確認、データ収集状況の確認、
パイプラインの各ステップの実行状況確認を行います。

Usage:
    python scripts/utils/verify_coding_pipeline.py --config configs/unified_master_pipeline_config.yaml
"""

import sys
import json
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/verify_coding_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CodingPipelineVerifier:
    """コーディングパイプライン動作確認クラス"""
    
    def __init__(self, config_path: Path):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = Path(config_path)
        
        # 設定を読み込み
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.verification_results: Dict = {}
        
        logger.info("="*80)
        logger.info("Coding Pipeline Verifier Initialized")
        logger.info("="*80)
    
    def verify_phase4_github_scraping(self) -> Dict:
        """Phase 4: GitHubリポジトリ検索の確認"""
        logger.info("[VERIFY] Phase 4: GitHub Repository Scraping")
        
        phase_config = self.config.get('phase4_github_scraping', {})
        if not phase_config.get('enabled', True):
            logger.info("[VERIFY] Phase 4: Disabled in config")
            return {'status': 'disabled', 'enabled': False}
        
        output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/processed/github'))
        
        result = {
            'phase': 'phase4_github_scraping',
            'enabled': True,
            'output_dir': str(output_dir),
            'exists': output_dir.exists(),
            'files': [],
            'total_samples': 0,
            'file_count': 0,
            'status': 'unknown'
        }
        
        if output_dir.exists():
            jsonl_files = list(output_dir.glob("*.jsonl"))
            result['file_count'] = len(jsonl_files)
            result['files'] = [str(f) for f in jsonl_files[:10]]  # 最初の10ファイル
            
            # サンプル数をカウント
            total_samples = 0
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        count = sum(1 for line in f if line.strip())
                        total_samples += count
                except Exception as e:
                    logger.warning(f"[VERIFY] Failed to count samples in {jsonl_file}: {e}")
            
            result['total_samples'] = total_samples
            
            if total_samples > 0:
                result['status'] = 'completed'
                logger.info(f"[VERIFY] Phase 4: Found {total_samples} samples in {len(jsonl_files)} files")
            else:
                result['status'] = 'no_data'
                logger.warning("[VERIFY] Phase 4: Output directory exists but no samples found")
        else:
            result['status'] = 'not_started'
            logger.warning(f"[VERIFY] Phase 4: Output directory not found: {output_dir}")
        
        return result
    
    def verify_phase5_engineer_sites(self) -> Dict:
        """Phase 5: エンジニア向けサイトスクレイピングの確認"""
        logger.info("[VERIFY] Phase 5: Engineer Sites Scraping")
        
        phase_config = self.config.get('phase5_engineer_sites', {})
        if not phase_config.get('enabled', True):
            logger.info("[VERIFY] Phase 5: Disabled in config")
            return {'status': 'disabled', 'enabled': False}
        
        output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/processed/engineer_sites'))
        
        result = {
            'phase': 'phase5_engineer_sites',
            'enabled': True,
            'output_dir': str(output_dir),
            'exists': output_dir.exists(),
            'files': [],
            'total_samples': 0,
            'file_count': 0,
            'status': 'unknown'
        }
        
        if output_dir.exists():
            jsonl_files = list(output_dir.glob("*.jsonl"))
            result['file_count'] = len(jsonl_files)
            result['files'] = [str(f) for f in jsonl_files[:10]]
            
            total_samples = 0
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        count = sum(1 for line in f if line.strip())
                        total_samples += count
                except Exception as e:
                    logger.warning(f"[VERIFY] Failed to count samples in {jsonl_file}: {e}")
            
            result['total_samples'] = total_samples
            
            if total_samples > 0:
                result['status'] = 'completed'
                logger.info(f"[VERIFY] Phase 5: Found {total_samples} samples in {len(jsonl_files)} files")
            else:
                result['status'] = 'no_data'
                logger.warning("[VERIFY] Phase 5: Output directory exists but no samples found")
        else:
            result['status'] = 'not_started'
            logger.warning(f"[VERIFY] Phase 5: Output directory not found: {output_dir}")
        
        return result
    
    def verify_phase6_coding_extraction(self) -> Dict:
        """Phase 6: コーディング関連データ抽出の確認"""
        logger.info("[VERIFY] Phase 6: Coding Data Extraction")
        
        phase_config = self.config.get('phase6_coding_extraction', {})
        if not phase_config.get('enabled', True):
            logger.info("[VERIFY] Phase 6: Disabled in config")
            return {'status': 'disabled', 'enabled': False}
        
        output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/coding_dataset'))
        
        result = {
            'phase': 'phase6_coding_extraction',
            'enabled': True,
            'output_dir': str(output_dir),
            'exists': output_dir.exists(),
            'files': [],
            'total_samples': 0,
            'file_count': 0,
            'task_type_distribution': {},
            'status': 'unknown'
        }
        
        if output_dir.exists():
            jsonl_files = list(output_dir.glob("coding_*.jsonl"))
            result['file_count'] = len(jsonl_files)
            result['files'] = [str(f) for f in jsonl_files[:10]]
            
            total_samples = 0
            task_types = Counter()
            
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    sample = json.loads(line.strip())
                                    total_samples += 1
                                    task_type = sample.get('task_type', 'unknown')
                                    task_types[task_type] += 1
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.warning(f"[VERIFY] Failed to process {jsonl_file}: {e}")
            
            result['total_samples'] = total_samples
            result['task_type_distribution'] = dict(task_types)
            
            if total_samples > 0:
                result['status'] = 'completed'
                logger.info(f"[VERIFY] Phase 6: Found {total_samples} coding samples")
                logger.info(f"[VERIFY] Phase 6: Task type distribution: {dict(task_types)}")
            else:
                result['status'] = 'no_data'
                logger.warning("[VERIFY] Phase 6: Output directory exists but no samples found")
        else:
            result['status'] = 'not_started'
            logger.warning(f"[VERIFY] Phase 6: Output directory not found: {output_dir}")
        
        return result
    
    def verify_phase7_coding_training_data(self) -> Dict:
        """Phase 7: コーディングタスク用データセット作成の確認"""
        logger.info("[VERIFY] Phase 7: Coding Training Data Preparation")
        
        phase_config = self.config.get('phase7_coding_training_data', {})
        if not phase_config.get('enabled', True):
            logger.info("[VERIFY] Phase 7: Disabled in config")
            return {'status': 'disabled', 'enabled': False}
        
        output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/coding_training_data'))
        
        result = {
            'phase': 'phase7_coding_training_data',
            'enabled': True,
            'output_dir': str(output_dir),
            'exists': output_dir.exists(),
            'files': [],
            'total_samples': 0,
            'file_count': 0,
            'task_type_distribution': {},
            'status': 'unknown'
        }
        
        if output_dir.exists():
            jsonl_files = list(output_dir.glob("coding_training_*.jsonl"))
            result['file_count'] = len(jsonl_files)
            result['files'] = [str(f) for f in jsonl_files[:10]]
            
            total_samples = 0
            task_types = Counter()
            
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    sample = json.loads(line.strip())
                                    total_samples += 1
                                    task_type = sample.get('task_type', 'unknown')
                                    task_types[task_type] += 1
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.warning(f"[VERIFY] Failed to process {jsonl_file}: {e}")
            
            result['total_samples'] = total_samples
            result['task_type_distribution'] = dict(task_types)
            
            if total_samples > 0:
                result['status'] = 'completed'
                logger.info(f"[VERIFY] Phase 7: Found {total_samples} training samples")
                logger.info(f"[VERIFY] Phase 7: Task type distribution: {dict(task_types)}")
            else:
                result['status'] = 'no_data'
                logger.warning("[VERIFY] Phase 7: Output directory exists but no samples found")
        else:
            result['status'] = 'not_started'
            logger.warning(f"[VERIFY] Phase 7: Output directory not found: {output_dir}")
        
        return result
    
    def verify_phase8_coding_retraining(self) -> Dict:
        """Phase 8: コーディング特化再学習の確認"""
        logger.info("[VERIFY] Phase 8: Coding-Focused Retraining")
        
        phase_config = self.config.get('phase8_coding_retraining', {})
        if not phase_config.get('enabled', True):
            logger.info("[VERIFY] Phase 8: Disabled in config")
            return {'status': 'disabled', 'enabled': False}
        
        config_path = Path(phase_config.get('config_path', 'configs/coding_focused_retraining_config.yaml'))
        
        result = {
            'phase': 'phase8_coding_retraining',
            'enabled': True,
            'config_path': str(config_path),
            'config_exists': config_path.exists(),
            'output_dir': None,
            'checkpoint_exists': False,
            'status': 'unknown'
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    retraining_config = yaml.safe_load(f)
                
                output_dir = Path(retraining_config.get('output', {}).get('output_dir', 'D:/webdataset/checkpoints/coding_retraining'))
                result['output_dir'] = str(output_dir)
                
                if output_dir.exists():
                    # チェックポイントファイルを検索
                    checkpoint_files = list(output_dir.glob("*.pt")) + list(output_dir.glob("*.pth"))
                    result_files = list(output_dir.glob("*_results_*.json"))
                    
                    if checkpoint_files or result_files:
                        result['checkpoint_exists'] = True
                        result['status'] = 'completed'
                        logger.info(f"[VERIFY] Phase 8: Found checkpoints/results in {output_dir}")
                    else:
                        result['status'] = 'in_progress'
                        logger.warning("[VERIFY] Phase 8: Output directory exists but no checkpoints found")
                else:
                    result['status'] = 'not_started'
                    logger.warning(f"[VERIFY] Phase 8: Output directory not found: {output_dir}")
            except Exception as e:
                logger.warning(f"[VERIFY] Failed to read retraining config: {e}")
                result['status'] = 'error'
        else:
            result['status'] = 'config_not_found'
            logger.warning(f"[VERIFY] Phase 8: Config file not found: {config_path}")
        
        return result
    
    def check_log_errors(self) -> Dict:
        """ログファイルからエラーを確認"""
        logger.info("[VERIFY] Checking log files for errors")
        
        log_dir = PROJECT_ROOT / "logs"
        errors = []
        warnings = []
        
        log_files = [
            'github_repository_scraper.log',
            'engineer_site_scraper.log',
            'extract_coding_dataset.log',
            'prepare_coding_training_data.log',
            'coding_focused_retraining_pipeline.log',
            'unified_master_pipeline.log'
        ]
        
        for log_file in log_files:
            log_path = log_dir / log_file
            if log_path.exists():
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines[-1000:], 1):  # 最後の1000行をチェック
                            if 'ERROR' in line or 'CRITICAL' in line:
                                errors.append({
                                    'file': log_file,
                                    'line': len(lines) - 1000 + i,
                                    'message': line.strip()
                                })
                            elif 'WARNING' in line:
                                warnings.append({
                                    'file': log_file,
                                    'line': len(lines) - 1000 + i,
                                    'message': line.strip()
                                })
                except Exception as e:
                    logger.warning(f"[VERIFY] Failed to read log file {log_file}: {e}")
        
        return {
            'errors': errors[-50:],  # 最新の50エラー
            'warnings': warnings[-50:],  # 最新の50警告
            'error_count': len(errors),
            'warning_count': len(warnings)
        }
    
    def generate_report(self) -> Dict:
        """検証レポートを生成"""
        logger.info("[REPORT] Generating verification report")
        
        report = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'phases': {
                'phase4': self.verify_phase4_github_scraping(),
                'phase5': self.verify_phase5_engineer_sites(),
                'phase6': self.verify_phase6_coding_extraction(),
                'phase7': self.verify_phase7_coding_training_data(),
                'phase8': self.verify_phase8_coding_retraining()
            },
            'logs': self.check_log_errors()
        }
        
        # サマリーを計算
        completed_phases = sum(1 for phase in report['phases'].values() if phase.get('status') == 'completed')
        total_phases = sum(1 for phase in report['phases'].values() if phase.get('enabled', True) != False)
        
        report['summary'] = {
            'total_phases': total_phases,
            'completed_phases': completed_phases,
            'completion_rate': completed_phases / total_phases if total_phases > 0 else 0.0,
            'total_samples': {
                'github': report['phases']['phase4'].get('total_samples', 0),
                'engineer_sites': report['phases']['phase5'].get('total_samples', 0),
                'coding_extraction': report['phases']['phase6'].get('total_samples', 0),
                'coding_training': report['phases']['phase7'].get('total_samples', 0)
            },
            'error_count': report['logs']['error_count'],
            'warning_count': report['logs']['warning_count']
        }
        
        return report
    
    def save_report(self, report: Dict, output_path: Optional[Path] = None):
        """レポートを保存"""
        if output_path is None:
            output_path = PROJECT_ROOT / "_docs" / f"coding_pipeline_verification_{self.session_id}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[REPORT] Verification report saved to {output_path}")
        
        # サマリーを表示
        summary = report['summary']
        logger.info("="*80)
        logger.info("Verification Summary")
        logger.info("="*80)
        logger.info(f"Completed phases: {summary['completed_phases']}/{summary['total_phases']} ({summary['completion_rate']*100:.1f}%)")
        logger.info(f"Total samples:")
        logger.info(f"  - GitHub: {summary['total_samples']['github']}")
        logger.info(f"  - Engineer sites: {summary['total_samples']['engineer_sites']}")
        logger.info(f"  - Coding extraction: {summary['total_samples']['coding_extraction']}")
        logger.info(f"  - Coding training: {summary['total_samples']['coding_training']}")
        logger.info(f"Errors: {summary['error_count']}, Warnings: {summary['warning_count']}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Verify Coding Pipeline')
    parser.add_argument('--config', type=str, default='configs/unified_master_pipeline_config.yaml', help='Configuration file path')
    parser.add_argument('--output', type=str, help='Output report path (optional)')
    
    args = parser.parse_args()
    
    verifier = CodingPipelineVerifier(config_path=args.config)
    report = verifier.generate_report()
    
    output_path = Path(args.output) if args.output else None
    verifier.save_report(report, output_path)


if __name__ == '__main__':
    main()

