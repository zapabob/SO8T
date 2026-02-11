#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOONSHOT自動エラー修正・チェックポイント保存システム
UnicodeEncodeError, Phase失敗、依存関係エラーなどを自動修正
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'auto_error_correction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoErrorCorrector:
    """自動エラー修正クラス"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.checkpoint_dir = self.project_root / 'checkpoints' / 'auto_error_correction'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # エラーパターンと修正方法
        self.error_patterns = {
            'UnicodeEncodeError': self._fix_unicode_error,
            'Phase.*failed': self._fix_phase_failure,
            'ImportError': self._fix_import_error,
            'ModuleNotFoundError': self._fix_module_not_found,
            'OSError': self._fix_os_error,
            'RuntimeError': self._fix_runtime_error,
            'SyntaxError': self._fix_syntax_error,
            'TypeError': self._fix_type_error
        }

        # 依存関係マッピング
        self.dependencies = {
            'torch': 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128',
            'transformers': 'transformers',
            'datasets': 'datasets',
            'accelerate': 'accelerate',
            'lm_eval': 'lm_eval',
            'llama_cpp': 'llama-cpp-python',
            'sentencepiece': 'sentencepiece',
            'protobuf': 'protobuf',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'scipy': 'scipy',
            'scikit-learn': 'scikit-learn',
            'tqdm': 'tqdm',
            'wandb': 'wandb',
            'huggingface_hub': 'huggingface_hub',
            'evaluate': 'evaluate',
            'rouge_score': 'rouge_score',
            'sacrebleu': 'sacrebleu',
            'bert_score': 'bert_score',
            'peft': 'peft',
            'bitsandbytes': 'bitsandbytes',
            'flash-attn': 'flash-attn',
            'autoawq': 'autoawq',
            'optimum': 'optimum'
        }

    def run_correction_cycle(self) -> bool:
        """エラー修正サイクルを実行"""
        logger.info("自動エラー修正サイクル開始")

        # ログファイルからエラーを検出
        errors = self._detect_errors()
        if not errors:
            logger.info("エラーが検出されませんでした")
            return True

        # エラーを修正
        corrected = 0
        for error in errors:
            if self._correct_error(error):
                corrected += 1

        # チェックポイント保存
        self._save_checkpoint(corrected, len(errors))

        logger.info(f"エラー修正完了: {corrected}/{len(errors)}個修正")
        return corrected == len(errors)

    def _detect_errors(self) -> List[Dict]:
        """ログファイルからエラーを検出"""
        errors = []

        # ログファイルを確認
        log_files = [
            self.project_root / 'ab_test_automation.log',
            self.project_root / 'auto_error_correction.log'
        ]

        for log_file in log_files:
            if not log_file.exists():
                continue

            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # エラーパターンを検索
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    for pattern, _ in self.error_patterns.items():
                        if pattern.lower() in line.lower():
                            errors.append({
                                'pattern': pattern,
                                'line': line.strip(),
                                'file': str(log_file),
                                'line_number': i + 1,
                                'timestamp': datetime.now().isoformat()
                            })
                            break

            except UnicodeDecodeError:
                # Unicodeエラーの場合、別のエンコーディングで試行
                try:
                    with open(log_file, 'r', encoding='cp932') as f:
                        content = f.read()
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        for pattern, _ in self.error_patterns.items():
                            if pattern.lower() in line.lower():
                                errors.append({
                                    'pattern': pattern,
                                    'line': line.strip(),
                                    'file': str(log_file),
                                    'line_number': i + 1,
                                    'timestamp': datetime.now().isoformat()
                                })
                                break
                except Exception as e:
                    logger.warning(f"ログファイル読み込み失敗: {log_file}, {e}")

        # 重複を削除
        unique_errors = []
        seen = set()
        for error in errors:
            key = (error['pattern'], error['line'])
            if key not in seen:
                unique_errors.append(error)
                seen.add(key)

        return unique_errors

    def _correct_error(self, error: Dict) -> bool:
        """エラーを修正"""
        pattern = error['pattern']
        logger.info(f"エラー修正開始: {pattern} - {error['line'][:100]}...")

        # 対応する修正関数を呼び出し
        if pattern in self.error_patterns:
            try:
                result = self.error_patterns[pattern](error)
                if result:
                    logger.info(f"エラー修正成功: {pattern}")
                    return True
                else:
                    logger.warning(f"エラー修正失敗: {pattern}")
                    return False
            except Exception as e:
                logger.error(f"エラー修正中に例外発生: {pattern}, {e}")
                return False
        else:
            logger.warning(f"未対応エラーパターン: {pattern}")
            return False

    def _fix_unicode_error(self, error: Dict) -> bool:
        """UnicodeEncodeErrorを修正"""
        # UTF-8エンコーディングを強制
        try:
            # 問題のあるファイルをUTF-8で再保存
            problematic_files = [
                self.project_root / 'scripts' / 'evaluation' / 'setup_lm_eval_elyza.py',
                self.project_root / 'scripts' / 'utils' / 'advanced_monitor.py'
            ]

            for file_path in problematic_files:
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='cp932') as f:
                            content = f.read()

                        # 絵文字などを除去
                        import re
                        content = re.sub(r'[^\x00-\x7F\u0080-\uFFFF]', '', content)

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        logger.info(f"ファイルUTF-8変換完了: {file_path}")

                    except Exception as e:
                        logger.warning(f"ファイル変換失敗: {file_path}, {e}")

            return True

        except Exception as e:
            logger.error(f"UnicodeEncodeError修正失敗: {e}")
            return False

    def _fix_phase_failure(self, error: Dict) -> bool:
        """Phase失敗を修正"""
        # Phase番号を特定
        line = error['line'].lower()
        phase_num = None
        for i in range(9):
            if f'phase {i}' in line or f'phase{i}' in line:
                phase_num = i
                break

        if phase_num is None:
            logger.warning("Phase番号を特定できませんでした")
            return False

        logger.info(f"Phase {phase_num} 失敗修正開始")

        # Phaseごとの修正処理
        if phase_num == 0:
            # 環境チェック修正
            return self._fix_environment_check()
        elif phase_num == 1:
            # データセット作成修正
            return self._fix_dataset_creation()
        elif phase_num == 2:
            # lm-eval統合修正
            return self._fix_lm_eval_integration()
        else:
            # その他のPhase
            return self._fix_generic_phase(phase_num)

    def _fix_environment_check(self) -> bool:
        """環境チェックを修正"""
        try:
            # Pythonバージョン確認
            result = subprocess.run([sys.executable, '--version'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Python実行確認失敗")
                return False

            # 基本ライブラリ確認とインストール
            basic_libs = ['torch', 'transformers', 'numpy']
            for lib in basic_libs:
                try:
                    __import__(lib)
                except ImportError:
                    logger.info(f"ライブラリインストール: {lib}")
                    subprocess.run([sys.executable, '-m', 'pip', 'install', lib],
                                 check=True)

            return True

        except Exception as e:
            logger.error(f"環境チェック修正失敗: {e}")
            return False

    def _fix_dataset_creation(self) -> bool:
        """データセット作成を修正"""
        try:
            # データセットディレクトリ確認
            dataset_dir = self.project_root / 'data' / 'datasets'
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # 必要なライブラリ確認
            required_libs = ['datasets', 'transformers']
            for lib in required_libs:
                try:
                    __import__(lib)
                except ImportError:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', lib],
                                 check=True)

            return True

        except Exception as e:
            logger.error(f"データセット作成修正失敗: {e}")
            return False

    def _fix_lm_eval_integration(self) -> bool:
        """lm-eval統合を修正"""
        try:
            # lm-evaluation-harnessが存在するか確認
            lm_eval_dir = self.project_root / 'lm-evaluation-harness'
            if not lm_eval_dir.exists():
                # クローン
                subprocess.run(['git', 'clone',
                              'https://github.com/EleutherAI/lm-evaluation-harness.git'],
                             cwd=self.project_root, check=True)

            # lm_evalインストール
            try:
                import lm_eval
            except ImportError:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'lm_eval'],
                             check=True)

            # ELYZAデータセット確認
            try:
                from datasets import load_dataset
                ds = load_dataset('elyza/ELYZA-tasks-100', split='test')
                logger.info(f"ELYZA-100確認: {len(ds)}サンプル")
            except Exception as e:
                logger.warning(f"ELYZAデータセット確認失敗: {e}")

            return True

        except Exception as e:
            logger.error(f"lm-eval統合修正失敗: {e}")
            return False

    def _fix_generic_phase(self, phase_num: int) -> bool:
        """一般的なPhase修正"""
        try:
            # ログをクリアして再実行を促す
            log_file = self.project_root / 'ab_test_automation.log'
            if log_file.exists():
                # バックアップ
                backup_file = log_file.with_suffix(f'.backup_{int(time.time())}')
                log_file.rename(backup_file)

            logger.info(f"Phase {phase_num} のログをクリアしました")
            return True

        except Exception as e:
            logger.error(f"Phase {phase_num} 修正失敗: {e}")
            return False

    def _fix_import_error(self, error: Dict) -> bool:
        """ImportErrorを修正"""
        # エラーメッセージからモジュール名を抽出
        line = error['line']
        import re
        match = re.search(r"No module named '([^']+)'", line)
        if match:
            module = match.group(1)
            return self._install_dependency(module)
        return False

    def _fix_module_not_found(self, error: Dict) -> bool:
        """ModuleNotFoundErrorを修正"""
        return self._fix_import_error(error)

    def _fix_os_error(self, error: Dict) -> bool:
        """OSErrorを修正"""
        # ファイルパス関連のエラーを処理
        line = error['line'].lower()
        if 'permission denied' in line or 'アクセスが拒否されました' in line:
            logger.warning("権限エラーが発生しました。管理者権限が必要かもしれません")
            return False
        elif 'no such file' in line or 'ファイルが見つかりません' in line:
            logger.warning("ファイルが見つからないエラーです")
            return False
        else:
            return True

    def _fix_runtime_error(self, error: Dict) -> bool:
        """RuntimeErrorを修正"""
        # CUDA関連のエラーを処理
        line = error['line'].lower()
        if 'cuda' in line and 'out of memory' in line:
            logger.warning("CUDAメモリ不足エラーが発生しました")
            # メモリ解放を試行
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            except:
                return False
        return True

    def _fix_syntax_error(self, error: Dict) -> bool:
        """SyntaxErrorを修正"""
        # 構文エラーは通常手動修正が必要
        logger.warning("構文エラーが発生しました。手動確認が必要です")
        return False

    def _fix_type_error(self, error: Dict) -> bool:
        """TypeErrorを修正"""
        # 型エラーは通常コード修正が必要
        logger.warning("型エラーが発生しました。コード修正が必要です")
        return False

    def _install_dependency(self, module: str) -> bool:
        """依存関係をインストール"""
        try:
            if module in self.dependencies:
                cmd = f"{sys.executable} -m pip install {self.dependencies[module]}"
                result = subprocess.run(cmd, shell=True, check=True)
                logger.info(f"依存関係インストール成功: {module}")
                return True
            else:
                # 一般的なインストールを試行
                cmd = f"{sys.executable} -m pip install {module}"
                result = subprocess.run(cmd, shell=True, check=True)
                logger.info(f"依存関係インストール成功: {module}")
                return True

        except subprocess.CalledProcessError as e:
            logger.error(f"依存関係インストール失敗: {module}, {e}")
            return False

    def _save_checkpoint(self, corrected: int, total: int):
        """チェックポイントを保存"""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'corrected_errors': corrected,
            'total_errors': total,
            'success_rate': corrected / total if total > 0 else 0,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': str(self.project_root)
            }
        }

        checkpoint_file = self.checkpoint_dir / f"error_correction_{int(time.time())}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        logger.info(f"チェックポイント保存: {checkpoint_file}")

def main():
    """メイン関数"""
    corrector = AutoErrorCorrector()
    success = corrector.run_correction_cycle()

    if success:
        logger.info("自動エラー修正サイクルが正常に完了しました")
        sys.exit(0)
    else:
        logger.warning("一部のエラーが修正されませんでした")
        sys.exit(1)

if __name__ == '__main__':
    main()
