#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T統合自動パイプライン

Webスクレイピング → データクレンジング → SO8T統合 → QLoRA 8bitファインチューニング
を全自動で実行する統合パイプライン

Usage:
    python scripts/pipelines/automated_so8t_pipeline.py --config configs/automated_so8t_pipeline.yaml
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent


class AutomatedSO8TPipeline:
    """
    SO8T統合自動パイプライン
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config = self._load_config(config_path)
        self.pipeline_config = self.config.get('pipeline', {})
        self.output_base_dir = Path(self.pipeline_config.get('output_base_dir', 'D:/webdataset'))
        self.checkpoint_dir = Path(self.pipeline_config.get('checkpoint_dir', 'D:/webdataset/pipeline_checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = self.checkpoint_dir / f"pipeline_session_{self.session_id}.json"
        
        self.phase_status = {
            'phase1_web_scraping': {'completed': False, 'output': None},
            'phase2_data_cleansing': {'completed': False, 'output': None},
            'phase3_modeling_so8t': {'completed': False, 'output': None},
            'phase4_integration': {'completed': False, 'output': None},
            'phase5_qlora_training': {'completed': False, 'output': None},
            'phase7_1_testing': {'completed': False, 'output': None},
            'phase7_2_evaluation': {'completed': False, 'output': None},
            'phase7_3_gguf_conversion': {'completed': False, 'output': None},
            'phase7_4_ollama_import': {'completed': False, 'output': None},
            'phase7_5_japanese_test': {'completed': False, 'output': None},
        }
        
        logger.info(f"Automated SO8T Pipeline initialized")
        logger.info(f"  Session ID: {self.session_id}")
        logger.info(f"  Output base dir: {self.output_base_dir}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _save_checkpoint(self):
        """チェックポイントを保存"""
        checkpoint_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'phase_status': self.phase_status,
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[CHECKPOINT] Saved checkpoint to {self.checkpoint_file}")
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """チェックポイントを読み込み"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            logger.info(f"[CHECKPOINT] Loaded checkpoint from {self.checkpoint_file}")
            return checkpoint_data
        return None
    
    def _play_audio_notification(self, on_completion: bool = True, on_error: bool = False):
        """音声通知を再生（複数のフォールバック方法を使用）"""
        notifications = self.config.get('notifications', {})
        if not notifications.get('audio_notification', True):
            return
        
        audio_file = Path(notifications.get('audio_file', 'C:/Users/downl/Desktop/SO8T/.cursor/marisa_owattaze.wav'))
        
        if (on_completion and notifications.get('play_on_completion', True)) or \
           (on_error and notifications.get('play_on_error', True)):
            if audio_file.exists():
                # 方法1: winsound (最も確実な方法)
                try:
                    import winsound
                    winsound.PlaySound(str(audio_file), winsound.SND_FILENAME | winsound.SND_ASYNC)
                    logger.info("[AUDIO] Audio notification played successfully (winsound)")
                    return
                except Exception as e:
                    logger.warning(f"[AUDIO] Method 1 (winsound) failed: {e}")
                    
                    # 方法2: PowerShell SoundPlayer (PlaySyncで同期再生)
                    try:
                        import subprocess
                        import tempfile
                        # 一時的なPowerShellスクリプトファイルを作成
                        ps_script_content = f'''$audioPath = "{str(audio_file).replace('"', '`"')}"
if (Test-Path $audioPath) {{
    Add-Type -AssemblyName System.Windows.Forms
    $player = New-Object System.Media.SoundPlayer($audioPath)
    $player.PlaySync()
    Write-Host "[OK] Audio notification played successfully" -ForegroundColor Green
}} else {{
    Write-Host "[WARNING] Audio file not found" -ForegroundColor Yellow
}}'''
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False) as f:
                            f.write(ps_script_content)
                            ps_script_path = f.name
                        
                        try:
                            result = subprocess.run(
                                ['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_script_path],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            if result.returncode == 0:
                                logger.info("[AUDIO] Audio notification played successfully (PowerShell SoundPlayer)")
                                return
                            else:
                                logger.warning(f"[AUDIO] PowerShell error: {result.stderr}")
                                raise Exception("PowerShell execution failed")
                        finally:
                            import os
                            try:
                                os.unlink(ps_script_path)
                            except:
                                pass
                    except Exception as e2:
                        logger.warning(f"[AUDIO] Method 2 (PowerShell) failed: {e2}")
                        
                        # 方法3: システムビープ (緊急フォールバック)
                        try:
                            import winsound
                            winsound.Beep(1000, 500)
                            logger.info("[AUDIO] Fallback beep played successfully")
                        except Exception as e3:
                            logger.error(f"[AUDIO] All methods failed: {e}, {e2}, {e3}")
            else:
                logger.warning(f"[AUDIO] Audio file not found: {audio_file}")
                # ファイルが見つからない場合もビープを試行
                try:
                    import winsound
                    winsound.Beep(800, 1000)
                    logger.info("[AUDIO] Emergency beep played")
                except Exception as e:
                    logger.error(f"[AUDIO] Emergency beep also failed: {e}")
    
    def run_phase1_web_scraping(self) -> bool:
        """Phase 1: Webスクレイピングとデータ収集"""
        phase_config = self.config.get('phase1_web_scraping', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 1] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 1] Webスクレイピングとデータ収集")
        logger.info("="*80)
        
        try:
            # 既存のcomplete_data_pipeline.pyを実行
            pipeline_config_path = phase_config.get('pipeline_config', 'configs/data_pipeline_config.yaml')
            pipeline_script = PROJECT_ROOT / "scripts" / "pipelines" / "complete_data_pipeline.py"
            
            if not pipeline_script.exists():
                logger.error(f"[PHASE 1] Pipeline script not found: {pipeline_script}")
                return False
            
            logger.info(f"[PHASE 1] Running data collection pipeline...")
            logger.info(f"  Script: {pipeline_script}")
            logger.info(f"  Config: {pipeline_config_path}")
            
            # サブプロセスで実行（エラー出力もキャプチャ）
            result = subprocess.run(
                [sys.executable, str(pipeline_script), '--config', pipeline_config_path],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )
            
            # エラー出力をログに記録
            if result.stderr:
                logger.warning(f"[PHASE 1] stderr output: {result.stderr[:1000]}")  # 最初の1000文字
            if result.stdout:
                logger.info(f"[PHASE 1] stdout output (last 500 chars): {result.stdout[-500:]}")
            
            if result.returncode == 0:
                # complete_data_pipeline.pyの実際の出力パスを探す
                # 出力は D:/webdataset/processed/four_class/four_class_{session_id}.jsonl
                base_output_dir = Path(self.pipeline_config.get('output_base_dir', 'D:/webdataset'))
                processed_dir = base_output_dir / "processed" / "four_class"
                
                # 最新のfour_class JSONLファイルを探す
                if processed_dir.exists():
                    jsonl_files = list(processed_dir.glob("four_class_*.jsonl"))
                    if jsonl_files:
                        # 最新のファイルを取得
                        latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
                        output_path = latest_file
                        logger.info(f"[PHASE 1] Found output file: {output_path}")
                    else:
                        # フォールバック: ディレクトリ自体を出力として設定
                        output_path = processed_dir
                        logger.warning(f"[PHASE 1] No JSONL files found, using directory: {output_path}")
                else:
                    # フォールバック: 設定で指定されたディレクトリ
                    output_path = Path(phase_config.get('output_dir', 'D:/webdataset/processed/finetuning'))
                    logger.warning(f"[PHASE 1] four_class directory not found, using fallback: {output_path}")
                
                self.phase_status['phase1_web_scraping']['completed'] = True
                self.phase_status['phase1_web_scraping']['output'] = str(output_path)
                self._save_checkpoint()
                logger.info("[PHASE 1] [OK] Web scraping completed successfully")
                logger.info(f"[PHASE 1] Output path: {output_path}")
                return True
            else:
                logger.error(f"[PHASE 1] [FAILED] Web scraping failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"[PHASE 1] [FAILED] Web scraping failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_phase2_data_cleansing(self) -> bool:
        """Phase 2: データクレンジングと前処理"""
        phase_config = self.config.get('phase2_data_cleansing', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 2] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 2] データクレンジングと前処理")
        logger.info("="*80)
        
        try:
            # Phase 1の出力を使用（既存パイプラインで自動的にクレンジングされる）
            phase1_output = self.phase_status['phase1_web_scraping'].get('output')
            if phase1_output:
                output_path = Path(phase1_output)
                
                # ファイルパスの場合
                if output_path.is_file():
                    if output_path.exists() and output_path.suffix == '.jsonl':
                        self.phase_status['phase2_data_cleansing']['completed'] = True
                        self.phase_status['phase2_data_cleansing']['output'] = str(output_path)
                        self._save_checkpoint()
                        logger.info("[PHASE 2] [OK] Data cleansing completed (already done in Phase 1)")
                        logger.info(f"[PHASE 2] Output file: {output_path}")
                        return True
                
                # ディレクトリパスの場合
                elif output_path.is_dir():
                    if output_path.exists():
                        # クレンジング済みデータが存在するか確認
                        jsonl_files = list(output_path.glob("*.jsonl"))
                        if jsonl_files:
                            # 最新のファイルを使用
                            latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
                            self.phase_status['phase2_data_cleansing']['completed'] = True
                            self.phase_status['phase2_data_cleansing']['output'] = str(latest_file)
                            self._save_checkpoint()
                            logger.info("[PHASE 2] [OK] Data cleansing completed (already done in Phase 1)")
                            logger.info(f"[PHASE 2] Output file: {latest_file}")
                            return True
                
                # パスが存在しない場合、four_classディレクトリを直接探す
                base_output_dir = Path(self.pipeline_config.get('output_base_dir', 'D:/webdataset'))
                four_class_dir = base_output_dir / "processed" / "four_class"
                
                if four_class_dir.exists():
                    jsonl_files = list(four_class_dir.glob("four_class_*.jsonl"))
                    if jsonl_files:
                        latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
                        self.phase_status['phase2_data_cleansing']['completed'] = True
                        self.phase_status['phase2_data_cleansing']['output'] = str(latest_file)
                        self._save_checkpoint()
                        logger.info("[PHASE 2] [OK] Data cleansing completed (found in four_class directory)")
                        logger.info(f"[PHASE 2] Output file: {latest_file}")
                        return True
            
            logger.warning("[PHASE 2] [WARNING] No cleaned data found, Phase 1 may not have completed")
            return False
            
        except Exception as e:
            logger.error(f"[PHASE 2] [FAILED] Data cleansing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_phase3_modeling_so8t(self) -> bool:
        """Phase 3: modeling_phi3.pyのSO8T統合"""
        phase_config = self.config.get('phase3_modeling_so8t', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 3] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 3] modeling_phi3.pyのSO8T統合")
        logger.info("="*80)
        
        try:
            # modeling_phi3_so8t.pyが存在するか確認
            modeling_file = Path(phase_config.get('modeling_file', 
                'models/Borea-Phi-3.5-mini-Instruct-Jp/modeling_phi3_so8t.py'))
            modeling_file = PROJECT_ROOT / modeling_file
            
            if modeling_file.exists():
                self.phase_status['phase3_modeling_so8t']['completed'] = True
                self.phase_status['phase3_modeling_so8t']['output'] = str(modeling_file)
                self._save_checkpoint()
                logger.info("[PHASE 3] [OK] SO8T modeling file exists")
                return True
            else:
                logger.error(f"[PHASE 3] [FAILED] SO8T modeling file not found: {modeling_file}")
                return False
                
        except Exception as e:
            logger.error(f"[PHASE 3] [FAILED] SO8T modeling check failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_phase4_integration(self) -> bool:
        """Phase 4: SO8T統合スクリプト"""
        phase_config = self.config.get('phase4_integration', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 4] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 4] SO8T統合スクリプト")
        logger.info("="*80)
        
        try:
            integration_script = Path(phase_config.get('script', 
                'scripts/conversion/integrate_phi3_so8t.py'))
            integration_script = PROJECT_ROOT / integration_script
            
            if not integration_script.exists():
                logger.error(f"[PHASE 4] Integration script not found: {integration_script}")
                return False
            
            model_path = phase_config.get('model_path', 'models/Borea-Phi-3.5-mini-Instruct-Jp')
            output_path = phase_config.get('output_path', 'D:/webdataset/models/so8t_integrated/phi3_so8t')
            device = phase_config.get('device', 'cuda')
            verify = phase_config.get('verify', True)
            torch_dtype = phase_config.get('torch_dtype', 'bfloat16')
            
            logger.info(f"[PHASE 4] Running SO8T integration...")
            logger.info(f"  Script: {integration_script}")
            logger.info(f"  Model path: {model_path}")
            logger.info(f"  Output path: {output_path}")
            
            # サブプロセスで実行
            cmd = [
                sys.executable, str(integration_script),
                '--model_path', str(model_path),
                '--output_path', str(output_path),
                '--device', device,
                '--torch_dtype', torch_dtype,
            ]
            if not verify:
                cmd.append('--no-verify')
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                self.phase_status['phase4_integration']['completed'] = True
                self.phase_status['phase4_integration']['output'] = str(output_path)
                self._save_checkpoint()
                logger.info("[PHASE 4] [OK] SO8T integration completed successfully")
                return True
            else:
                logger.error(f"[PHASE 4] [FAILED] SO8T integration failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"[PHASE 4] [FAILED] SO8T integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_phase5_qlora_training(self) -> bool:
        """Phase 5: QLoRA 8bitファインチューニング"""
        phase_config = self.config.get('phase5_qlora_training', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 5] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 5] QLoRA 8bitファインチューニング")
        logger.info("="*80)
        
        try:
            training_script = Path(phase_config.get('script', 
                'scripts/training/train_so8t_phi3_qlora.py'))
            training_script = PROJECT_ROOT / training_script
            
            if not training_script.exists():
                logger.error(f"[PHASE 5] Training script not found: {training_script}")
                return False
            
            training_config = phase_config.get('config', 'configs/train_so8t_phi3_qlora.yaml')
            resume = phase_config.get('resume', False)
            
            logger.info(f"[PHASE 5] Running QLoRA training...")
            logger.info(f"  Script: {training_script}")
            logger.info(f"  Config: {training_config}")
            
            # サブプロセスで実行
            cmd = [
                sys.executable, str(training_script),
                '--config', str(training_config),
            ]
            if resume:
                # 最新のチェックポイントから再開
                checkpoint_dir = Path(self.config.get('training', {}).get('output_dir', 
                    'D:/webdataset/checkpoints/finetuning/so8t_phi3'))
                if checkpoint_dir.exists():
                    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
                    if checkpoints:
                        cmd.extend(['--resume', str(checkpoints[-1])])
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                self.phase_status['phase5_qlora_training']['completed'] = True
                self.phase_status['phase5_qlora_training']['output'] = str(checkpoint_dir)
                self._save_checkpoint()
                logger.info("[PHASE 5] [OK] QLoRA training completed successfully")
                return True
            else:
                logger.error(f"[PHASE 5] [FAILED] QLoRA training failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"[PHASE 5] [FAILED] QLoRA training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_pipeline(self) -> bool:
        """完全パイプラインを実行"""
        logger.info("="*80)
        logger.info("SO8T統合自動パイプライン開始")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        start_time = time.time()
        
        # チェックポイントから再開
        if self.pipeline_config.get('resume', True):
            checkpoint_data = self._load_checkpoint()
            if checkpoint_data:
                self.phase_status = checkpoint_data.get('phase_status', self.phase_status)
                logger.info("[RESUME] Resuming from checkpoint")
        
        # Phase 1: Webスクレイピング
        if not self.phase_status['phase1_web_scraping']['completed']:
            if not self.run_phase1_web_scraping():
                logger.error("[FAILED] Phase 1 failed, stopping pipeline")
                self._play_audio_notification(on_completion=False, on_error=True)
                return False
        else:
            logger.info("[PHASE 1] Already completed, skipping")
        
        # Phase 2: データクレンジング
        if not self.phase_status['phase2_data_cleansing']['completed']:
            if not self.run_phase2_data_cleansing():
                logger.error("[FAILED] Phase 2 failed, stopping pipeline")
                self._play_audio_notification(on_completion=False, on_error=True)
                return False
        else:
            logger.info("[PHASE 2] Already completed, skipping")
        
        # Phase 3: SO8T統合（既に実装済み）
        if not self.phase_status['phase3_modeling_so8t']['completed']:
            if not self.run_phase3_modeling_so8t():
                logger.error("[FAILED] Phase 3 failed, stopping pipeline")
                self._play_audio_notification(on_completion=False, on_error=True)
                return False
        else:
            logger.info("[PHASE 3] Already completed, skipping")
        
        # Phase 4: SO8T統合スクリプト
        if not self.phase_status['phase4_integration']['completed']:
            if not self.run_phase4_integration():
                logger.error("[FAILED] Phase 4 failed, stopping pipeline")
                self._play_audio_notification(on_completion=False, on_error=True)
                return False
        else:
            logger.info("[PHASE 4] Already completed, skipping")
        
        # Phase 5: QLoRA訓練
        if not self.phase_status['phase5_qlora_training']['completed']:
            if not self.run_phase5_qlora_training():
                logger.error("[FAILED] Phase 5 failed, stopping pipeline")
                self._play_audio_notification(on_completion=False, on_error=True)
                return False
        else:
            logger.info("[PHASE 5] Already completed, skipping")
        
        # Phase 7: 評価・変換・デプロイ
        phase7_config = self.config.get('phase7_evaluation_deployment', {})
        if phase7_config.get('enabled', True):
            # Phase 7.1: 実行テスト
            if phase7_config.get('phase7_1_testing', {}).get('enabled', True):
                if not self.phase_status['phase7_1_testing']['completed']:
                    if not self.run_phase7_1_testing():
                        logger.warning("[WARNING] Phase 7.1 failed, continuing...")
                else:
                    logger.info("[PHASE 7.1] Already completed, skipping")
            
            # Phase 7.2: モデル評価
            if phase7_config.get('phase7_2_evaluation', {}).get('enabled', True):
                if not self.phase_status['phase7_2_evaluation']['completed']:
                    if not self.run_phase7_2_evaluation():
                        logger.warning("[WARNING] Phase 7.2 failed, continuing...")
                else:
                    logger.info("[PHASE 7.2] Already completed, skipping")
            
            # Phase 7.3: GGUF変換
            if phase7_config.get('phase7_3_gguf_conversion', {}).get('enabled', True):
                if not self.phase_status['phase7_3_gguf_conversion']['completed']:
                    if not self.run_phase7_3_gguf_conversion():
                        logger.warning("[WARNING] Phase 7.3 failed, continuing...")
                else:
                    logger.info("[PHASE 7.3] Already completed, skipping")
            
            # Phase 7.4: Ollamaインポート
            if phase7_config.get('phase7_4_ollama_import', {}).get('enabled', True):
                if not self.phase_status['phase7_4_ollama_import']['completed']:
                    if not self.run_phase7_4_ollama_import():
                        logger.warning("[WARNING] Phase 7.4 failed, continuing...")
                else:
                    logger.info("[PHASE 7.4] Already completed, skipping")
            
            # Phase 7.5: 日本語パフォーマンステスト
            if phase7_config.get('phase7_5_japanese_test', {}).get('enabled', True):
                if not self.phase_status['phase7_5_japanese_test']['completed']:
                    if not self.run_phase7_5_japanese_test():
                        logger.warning("[WARNING] Phase 7.5 failed, continuing...")
                else:
                    logger.info("[PHASE 7.5] Already completed, skipping")
        
        # 完了
        elapsed_time = time.time() - start_time
        logger.info("="*80)
        logger.info("[SUCCESS] SO8T統合自動パイプライン完了")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed time: {elapsed_time/3600:.2f} hours ({elapsed_time:.0f} seconds)")
        logger.info("="*80)
        
        # 最終レポート生成
        self._generate_final_report(elapsed_time)
        
        # 音声通知
        self._play_audio_notification(on_completion=True, on_error=False)
        
        return True
    
    def _generate_final_report(self, elapsed_time: float):
        """最終レポートを生成"""
        report_file = self.checkpoint_dir / f"pipeline_report_{self.session_id}.md"
        
        report_content = f"""# SO8T統合自動パイプライン実行レポート

## 実行情報
- **Session ID**: {self.session_id}
- **開始時刻**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **実行時間**: {elapsed_time/3600:.2f}時間 ({elapsed_time:.0f}秒)

## フェーズ実行状況

### Phase 1: Webスクレイピングとデータ収集
- **ステータス**: {'[OK] 完了' if self.phase_status['phase1_web_scraping']['completed'] else '[NG] 未完了'}
- **出力**: {self.phase_status['phase1_web_scraping'].get('output', 'N/A')}

### Phase 2: データクレンジングと前処理
- **ステータス**: {'[OK] 完了' if self.phase_status['phase2_data_cleansing']['completed'] else '[NG] 未完了'}
- **出力**: {self.phase_status['phase2_data_cleansing'].get('output', 'N/A')}

### Phase 3: modeling_phi3.pyのSO8T統合
- **ステータス**: {'[OK] 完了' if self.phase_status['phase3_modeling_so8t']['completed'] else '[NG] 未完了'}
- **出力**: {self.phase_status['phase3_modeling_so8t'].get('output', 'N/A')}

### Phase 4: SO8T統合スクリプト
- **ステータス**: {'[OK] 完了' if self.phase_status['phase4_integration']['completed'] else '[NG] 未完了'}
- **出力**: {self.phase_status['phase4_integration'].get('output', 'N/A')}

### Phase 5: QLoRA 8bitファインチューニング
- **ステータス**: {'[OK] 完了' if self.phase_status['phase5_qlora_training']['completed'] else '[NG] 未完了'}
- **出力**: {self.phase_status['phase5_qlora_training'].get('output', 'N/A')}

## 結果サマリー
- **全フェーズ完了**: {'はい' if all(phase['completed'] for phase in self.phase_status.values()) else 'いいえ'}
- **チェックポイント**: {self.checkpoint_file}

## 次のステップ
1. 訓練済みモデルの評価
2. GGUF形式への変換
3. Ollamaへのインポート
4. パフォーマンステスト

---
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"[REPORT] Final report saved to {report_file}")
    
    def run_phase7_1_testing(self) -> bool:
        """Phase 7.1: 実行テスト"""
        phase_config = self.config.get('phase7_evaluation_deployment', {}).get('phase7_1_testing', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 7.1] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 7.1] 実行テスト")
        logger.info("="*80)
        
        try:
            test_script = Path(phase_config.get('test_script', 'scripts/testing/test_so8t_pipeline_components.py'))
            test_script = PROJECT_ROOT / test_script
            
            if not test_script.exists():
                logger.error(f"[PHASE 7.1] Test script not found: {test_script}")
                return False
            
            logger.info(f"[PHASE 7.1] Running tests...")
            result = subprocess.run(
                [sys.executable, str(test_script)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.phase_status['phase7_1_testing']['completed'] = True
                self.phase_status['phase7_1_testing']['output'] = "Tests passed"
                self._save_checkpoint()
                logger.info("[PHASE 7.1] [OK] Tests completed successfully")
                return True
            else:
                logger.error(f"[PHASE 7.1] [FAILED] Tests failed")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"[PHASE 7.1] [FAILED] Testing failed: {e}")
            return False
    
    def run_phase7_2_evaluation(self) -> bool:
        """Phase 7.2: モデル評価"""
        phase_config = self.config.get('phase7_evaluation_deployment', {}).get('phase7_2_evaluation', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 7.2] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 7.2] モデル評価")
        logger.info("="*80)
        
        try:
            eval_script = Path(phase_config.get('eval_script', 'scripts/evaluation/evaluate_so8t_phi3.py'))
            eval_script = PROJECT_ROOT / eval_script
            
            if not eval_script.exists():
                logger.error(f"[PHASE 7.2] Evaluation script not found: {eval_script}")
                return False
            
            # 訓練済みモデルパスを取得（Phase 5の出力から）
            trained_model_path = self.phase_status['phase5_qlora_training'].get('output')
            if not trained_model_path:
                # Phase 4の出力を使用
                trained_model_path = self.phase_status['phase4_integration'].get('output')
            
            if not trained_model_path or not Path(trained_model_path).exists():
                logger.warning("[PHASE 7.2] Trained model not found, skipping evaluation")
                return False
            
            eval_dataset = phase_config.get('eval_dataset', None)
            output_path = self.output_base_dir / "evaluation_results" / f"evaluation_{self.session_id}.json"
            
            logger.info(f"[PHASE 7.2] Running evaluation...")
            cmd = [
                sys.executable, str(eval_script),
                '--model-path', str(trained_model_path),
                '--output', str(output_path),
                '--device', 'cuda'
            ]
            if eval_dataset:
                cmd.extend(['--eval-dataset', str(eval_dataset)])
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                self.phase_status['phase7_2_evaluation']['completed'] = True
                self.phase_status['phase7_2_evaluation']['output'] = str(output_path)
                self._save_checkpoint()
                logger.info("[PHASE 7.2] [OK] Evaluation completed successfully")
                return True
            else:
                logger.error(f"[PHASE 7.2] [FAILED] Evaluation failed")
                return False
                
        except Exception as e:
            logger.error(f"[PHASE 7.2] [FAILED] Evaluation failed: {e}")
            return False
    
    def run_phase7_3_gguf_conversion(self) -> bool:
        """Phase 7.3: GGUF変換"""
        phase_config = self.config.get('phase7_evaluation_deployment', {}).get('phase7_3_gguf_conversion', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 7.3] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 7.3] GGUF変換")
        logger.info("="*80)
        
        try:
            # GGUFConverterクラスをインポート
            from scripts.pipelines.complete_ab_test_post_processing_pipeline import GGUFConverter
            
            # 訓練済みモデルパスを取得
            trained_model_path = self.phase_status['phase5_qlora_training'].get('output')
            if not trained_model_path:
                trained_model_path = self.phase_status['phase4_integration'].get('output')
            
            if not trained_model_path or not Path(trained_model_path).exists():
                logger.warning("[PHASE 7.3] Trained model not found, skipping GGUF conversion")
                return False
            
            convert_script_path = Path(phase_config.get('convert_script', 
                'external/llama.cpp-master/convert_hf_to_gguf.py'))
            convert_script_path = PROJECT_ROOT / convert_script_path
            
            if not convert_script_path.exists():
                logger.error(f"[PHASE 7.3] Convert script not found: {convert_script_path}")
                return False
            
            quantizations = phase_config.get('quantizations', ['f16', 'q8_0'])
            output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/gguf_models'))
            model_name = f"so8t_phi3_{self.session_id}"
            
            converter = GGUFConverter(convert_script_path)
            gguf_files = converter.convert_to_gguf(
                model_dir=Path(trained_model_path),
                output_dir=output_dir / model_name,
                model_name=model_name,
                quantizations=quantizations
            )
            
            self.phase_status['phase7_3_gguf_conversion']['completed'] = True
            self.phase_status['phase7_3_gguf_conversion']['output'] = str(output_dir / model_name)
            self._save_checkpoint()
            logger.info("[PHASE 7.3] [OK] GGUF conversion completed successfully")
            return True
                
        except Exception as e:
            logger.error(f"[PHASE 7.3] [FAILED] GGUF conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_phase7_4_ollama_import(self) -> bool:
        """Phase 7.4: Ollamaインポート"""
        phase_config = self.config.get('phase7_evaluation_deployment', {}).get('phase7_4_ollama_import', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 7.4] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 7.4] Ollamaインポート")
        logger.info("="*80)
        
        try:
            from scripts.pipelines.complete_ab_test_post_processing_pipeline import OllamaImporter
            
            # GGUFファイルを取得（Phase 7.3の出力から）
            gguf_output_dir = self.phase_status['phase7_3_gguf_conversion'].get('output')
            if not gguf_output_dir:
                logger.warning("[PHASE 7.4] GGUF files not found, skipping Ollama import")
                return False
            
            gguf_dir = Path(gguf_output_dir)
            # Q8_0またはf16のGGUFファイルを探す
            gguf_file = None
            for quant_type in ['q8_0', 'f16']:
                candidate = gguf_dir / f"so8t_phi3_{self.session_id}_{quant_type}.gguf"
                if candidate.exists():
                    gguf_file = candidate
                    break
            
            if not gguf_file:
                logger.warning("[PHASE 7.4] GGUF file not found, skipping Ollama import")
                return False
            
            model_name = phase_config.get('model_name', f"so8t-phi3-{self.session_id}")
            modelfile_dir = Path(phase_config.get('modelfile_dir', 'modelfiles'))
            modelfile_path = modelfile_dir / f"{model_name}.modelfile"
            
            # Modelfile作成
            OllamaImporter.create_modelfile(gguf_file, model_name, modelfile_path)
            
            # Ollamaインポート
            success = OllamaImporter.import_to_ollama(modelfile_path, model_name)
            
            if success:
                self.phase_status['phase7_4_ollama_import']['completed'] = True
                self.phase_status['phase7_4_ollama_import']['output'] = f"{model_name}:latest"
                self._save_checkpoint()
                logger.info("[PHASE 7.4] [OK] Ollama import completed successfully")
                return True
            else:
                logger.error("[PHASE 7.4] [FAILED] Ollama import failed")
                return False
                
        except Exception as e:
            logger.error(f"[PHASE 7.4] [FAILED] Ollama import failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_phase7_5_japanese_test(self) -> bool:
        """Phase 7.5: 日本語パフォーマンステスト"""
        phase_config = self.config.get('phase7_evaluation_deployment', {}).get('phase7_5_japanese_test', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 7.5] Skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 7.5] 日本語パフォーマンステスト")
        logger.info("="*80)
        
        try:
            from scripts.testing.japanese_llm_performance_test import JapaneseLLMPerformanceTester
            
            # Ollamaモデル名を取得（Phase 7.4の出力から）
            ollama_model_name = self.phase_status['phase7_4_ollama_import'].get('output')
            if not ollama_model_name:
                logger.warning("[PHASE 7.5] Ollama model not found, skipping Japanese test")
                return False
            
            output_dir = Path(phase_config.get('output_dir', '_docs'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            tester = JapaneseLLMPerformanceTester(
                model_name=ollama_model_name,
                output_dir=output_dir
            )
            
            results_path = tester.run_all_tests()
            
            self.phase_status['phase7_5_japanese_test']['completed'] = True
            self.phase_status['phase7_5_japanese_test']['output'] = str(results_path)
            self._save_checkpoint()
            logger.info("[PHASE 7.5] [OK] Japanese performance test completed successfully")
            return True
                
        except Exception as e:
            logger.error(f"[PHASE 7.5] [FAILED] Japanese test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Automated SO8T Integration Pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/automated_so8t_pipeline.yaml',
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = AutomatedSO8TPipeline(args.config)
        success = pipeline.run_complete_pipeline()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.warning("[WARNING] Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"[FAILED] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

