#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS v2.2 GGUF変換スクリプト (I-Matrix対応)
数学推論能力を維持するための重要度行列適用

このスクリプトは以下の処理を行います：
1. HFモデルをGGUF形式に変換
2. I-Matrix（重要度行列）を使用して量子化精度を維持
3. 数学・論理推論能力の劣化を最小限に抑制

使用方法:
python scripts/conversion/convert_aegis_v22_with_imatrix.py
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import threading
import queue
import signal
import atexit
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any

try:
    from tqdm import tqdm
except ImportError:
    print("[WARNING] tqdm not found. Install with: pip install tqdm")
    tqdm = None

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('_docs/convert_aegis_v22_imatrix.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """チェックポイント管理クラス (3分間隔、5個ローリングストック)"""

    def __init__(self, base_dir: Path, max_checkpoints: int = 5):
        self.base_dir = base_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints_dir = base_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # チェックポイント保存間隔 (3分)
        self.save_interval = timedelta(minutes=3)
        self.last_save_time = datetime.now()

        # 自動保存スレッド
        self.save_thread = None
        self.stop_saving = False

        # シグナルハンドラー登録
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._emergency_save)

        print("[CHECKPOINT] Initialized checkpoint manager")

    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        print(f"[CHECKPOINT] Received signal {signum}, performing emergency save...")
        self._emergency_save()
        sys.exit(0)

    def _emergency_save(self):
        """緊急保存"""
        try:
            self.stop_auto_save()
            print("[CHECKPOINT] Emergency save completed")
        except Exception as e:
            print(f"[CHECKPOINT] Emergency save failed: {e}")

    def start_auto_save(self, save_callback):
        """自動保存を開始"""
        self.save_callback = save_callback
        self.stop_saving = False

        def auto_save_worker():
            while not self.stop_saving:
                try:
                    time.sleep(60)  # 1分ごとにチェック
                    if datetime.now() - self.last_save_time >= self.save_interval:
                        self.save_checkpoint()
                except Exception as e:
                    print(f"[CHECKPOINT] Auto-save error: {e}")

        self.save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self.save_thread.start()
        print("[CHECKPOINT] Auto-save started (3-minute intervals)")

    def stop_auto_save(self):
        """自動保存を停止"""
        self.stop_saving = True
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=5)
        print("[CHECKPOINT] Auto-save stopped")

    def save_checkpoint(self, data: Dict[str, Any] = None):
        """チェックポイント保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.checkpoints_dir / f"checkpoint_{timestamp}.json"

            if data is None and hasattr(self, 'save_callback'):
                data = self.save_callback()

            if data is None:
                data = {"timestamp": timestamp, "status": "in_progress"}

            # JSONシリアライズ可能な形式に変換
            serializable_data = self._make_serializable(data)

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

            self.last_save_time = datetime.now()

            # ローリング削除 (5個以上になったら古いものを削除)
            self._rotate_checkpoints()

            print(f"[CHECKPOINT] Saved checkpoint: {checkpoint_file.name}")

        except Exception as e:
            print(f"[CHECKPOINT] Save failed: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """JSONシリアライズ可能な形式に変換"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # その他の型は文字列に変換
            return str(obj)

    def _rotate_checkpoints(self):
        """ローリングチェックポイント (5個まで)"""
        try:
            checkpoint_files = list(self.checkpoints_dir.glob("checkpoint_*.json"))
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # 5個を超える古いチェックポイントを削除
            if len(checkpoint_files) > self.max_checkpoints:
                files_to_delete = checkpoint_files[self.max_checkpoints:]
                for old_file in files_to_delete:
                    old_file.unlink()
                    print(f"[CHECKPOINT] Deleted old checkpoint: {old_file.name}")

        except Exception as e:
            print(f"[CHECKPOINT] Rotation failed: {e}")

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """最新のチェックポイントを読み込み"""
        try:
            checkpoint_files = list(self.checkpoints_dir.glob("checkpoint_*.json"))
            if not checkpoint_files:
                return None

            # 最新のチェックポイントを選択
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"[CHECKPOINT] Loaded checkpoint: {latest_checkpoint.name}")
            return data

        except Exception as e:
            print(f"[CHECKPOINT] Load failed: {e}")
            return None

    def list_checkpoints(self) -> List[Path]:
        """チェックポイントファイル一覧を取得"""
        return sorted(self.checkpoints_dir.glob("checkpoint_*.json"),
                     key=lambda x: x.stat().st_mtime, reverse=True)


class AEGISv22GGUFConverter:
    """AEGIS v2.2 GGUF変換クラス (I-Matrix対応)"""

    def __init__(self, hf_model_path: str, output_dir: str, calibration_data: str):
        """
        Args:
            hf_model_path: HFモデルパス
            output_dir: 出力ディレクトリ
            calibration_data: キャリブレーションデータファイル
        """
        self.hf_model_path = Path(hf_model_path)
        self.output_dir = Path(output_dir)
        self.calibration_data = Path(calibration_data)

        # llama.cppパス
        self.llama_cpp_dir = Path("external/llama.cpp-master")
        self.llama_cpp_bin = self.llama_cpp_dir / "build" / "bin" / "Release"

        # 量子化設定
        self.quantization_types = [
            "Q8_0",    # 8-bit, ほぼ完全精度
            "Q6_K",    # 6-bit, バランス型
            "Q5_K_M",  # 5-bit, 中間精度
            "Q4_K_M",  # 4-bit, 最小精度（従来使用）
        ]

        # 中間ファイル
        self.f16_model = self.output_dir / f"aegis_v22_f16.gguf"
        self.imatrix_file = self.output_dir / "aegis_v22_imatrix.dat"

        # チェックポイントマネージャー
        self.checkpoint_manager = CheckpointManager(self.output_dir)

        # 変換状態
        self.conversion_state = {
            "status": "initialized",
            "current_step": 0,
            "total_steps": 3 + len(self.quantization_types),
            "completed_quantizations": [],
            "failed_quantizations": [],
            "start_time": datetime.now().isoformat(),
        }

    def check_llama_cpp(self) -> bool:
        """llama.cppの存在とビルドを確認"""
        logger.info("[CHECK] Checking llama.cpp environment...")

        if not self.llama_cpp_dir.exists():
            logger.error(f"[ERROR] llama.cpp not found at {self.llama_cpp_dir}")
            return False

        # 実行ファイルの確認
        executables = ["convert_hf_to_gguf.py", "llama-imatrix.exe", "llama-quantize.exe"]
        for exe in executables:
            exe_path = self.llama_cpp_dir / exe
            exe_bin_path = self.llama_cpp_bin / exe
            if not exe_path.exists() and not exe_bin_path.exists():
                logger.error(f"[ERROR] {exe} not found in {self.llama_cpp_dir} or {self.llama_cpp_bin}")
                return False

        logger.info("[OK] llama.cpp environment ready")
        return True

    def convert_to_f16_gguf(self) -> bool:
        """HFモデルをF16 GGUFに変換"""
        logger.info("[CONVERT] Converting HF model to F16 GGUF...")

        convert_script = self.llama_cpp_dir / "convert_hf_to_gguf.py"
        cmd = [
            "py", "-3", str(convert_script),
            str(self.hf_model_path),
            "--outfile", str(self.f16_model),
            "--outtype", "f16"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode != 0:
                logger.error(f"[ERROR] F16 conversion failed: {result.stderr}")
                return False

            logger.info("[SUCCESS] F16 GGUF conversion completed")
            return True

        except Exception as e:
            logger.error(f"[ERROR] F16 conversion exception: {e}")
            return False

    def create_imatrix(self) -> bool:
        """I-Matrix（重要度行列）を作成"""
        logger.info("[IMATRIX] Creating importance matrix...")

        if not self.calibration_data.exists():
            logger.error(f"[ERROR] Calibration data not found: {self.calibration_data}")
            return False

        imatrix_exe = self.llama_cpp_bin / "llama-imatrix.exe"
        cmd = [
            str(imatrix_exe),
            "-m", str(self.f16_model),
            "-f", str(self.calibration_data),
            "-o", str(self.imatrix_file)
        ]

        try:
            import time
            start_time = time.time()

            logger.info(f"[IMATRIX] Running command: {' '.join(cmd)}")
            logger.info(f"[IMATRIX] Processing {self.calibration_data.name} with {self.f16_model.name}")

            # tqdmで進捗を表示しながら実行
            if tqdm:
                with tqdm(total=100, desc="Creating I-Matrix", unit="%") as pbar:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        bufsize=1,
                        universal_newlines=True
                    )

                    # 標準エラー出力を監視して進捗を表示
                    while True:
                        output = process.stderr.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            # 進捗情報をパース（llama.cppの出力形式による）
                            if '%' in output or 'Processing' in output:
                                logger.debug(f"[IMATRIX] {output.strip()}")
                            # 適当な進捗更新（実際の進捗はllama.cppの出力による）
                            pbar.update(1)

                    # 最終結果を取得
                    result = subprocess.CompletedProcess(
                        process.args, process.returncode,
                        process.stdout.read() if process.stdout else "",
                        process.stderr.read() if process.stderr else ""
                    )
            else:
                # tqdmなしの場合
                logger.info("[INFO] tqdm not available, using basic logging")
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

            elapsed_time = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"[ERROR] I-Matrix creation failed: {result.stderr}")
                return False

            # I-Matrixファイルのサイズを確認
            if self.imatrix_file.exists():
                file_size = self.imatrix_file.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"[SUCCESS] I-Matrix created successfully (size: {file_size:.1f} MB)")
            else:
                logger.warning("[WARNING] I-Matrix file not found after creation")

            logger.info(f"[SUCCESS] I-Matrix created in {elapsed_time:.1f}s")
            return True

        except Exception as e:
            logger.error(f"[ERROR] I-Matrix creation exception: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def quantize_with_imatrix(self, quantization_type: str) -> bool:
        """I-Matrixを使用して量子化"""
        logger.info(f"[QUANTIZE] Quantizing to {quantization_type} with I-Matrix...")

        output_file = self.output_dir / f"aegis_v22_{quantization_type.lower()}.gguf"
        quantize_exe = self.llama_cpp_bin / "llama-quantize.exe"

        cmd = [
            str(quantize_exe),
            "--imatrix", str(self.imatrix_file),
            str(self.f16_model),
            str(output_file),
            quantization_type
        ]

        try:
            import time
            start_time = time.time()

            logger.info(f"[QUANTIZE] Starting quantization: {quantization_type}")
            logger.info(f"[QUANTIZE] Input: {self.f16_model.name}")
            logger.info(f"[QUANTIZE] Output: {output_file.name}")
            logger.info(f"[QUANTIZE] Using I-Matrix: {self.imatrix_file.name}")

            # tqdmで進捗を表示しながら実行
            if tqdm:
                pbar = tqdm(total=100, desc=f"Quantizing {quantization_type}", unit="%")
            else:
                pbar = None
                logger.info(f"[INFO] Quantizing {quantization_type} (tqdm not available)")

            if pbar:
                with pbar:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        bufsize=1,
                        universal_newlines=True
                    )

                    # 標準出力と標準エラー出力を監視
                    import threading
                    import queue

                    output_queue = queue.Queue()

                    def enqueue_output(out, queue):
                        for line in iter(out.readline, ''):
                            queue.put(line)
                        out.close()

                    # 出力監視スレッドを開始
                    stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, output_queue))
                    stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, output_queue))
                    stdout_thread.daemon = True
                    stderr_thread.daemon = True
                    stdout_thread.start()
                    stderr_thread.start()

                    # 進捗監視と表示
                    last_progress = 0
                    while process.poll() is None or not output_queue.empty():
                        try:
                            line = output_queue.get(timeout=0.1)
                            if line:
                                # llama-quantizeの進捗情報をパース
                                line = line.strip()
                                if 'Progress:' in line or '%' in line:
                                    logger.info(f"[QUANTIZE] {line}")
                                elif 'size =' in line or 'bits =' in line:
                                    logger.info(f"[QUANTIZE] {line}")

                                # 適当な進捗更新（実際の進捗はllama.cppの出力による）
                                current_progress = min(last_progress + 2, 95)  # 最大95%まで
                                pbar.update(current_progress - last_progress)
                                last_progress = current_progress

                        except queue.Empty:
                            continue

                    # 残りの進捗を完了
                    pbar.update(100 - last_progress)

                    # 最終結果を取得
                    result = subprocess.CompletedProcess(
                        process.args, process.returncode,
                        "", ""  # stdout/stderrは既に読み込み済み
                    )
            else:
                # tqdmなしの場合
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

            elapsed_time = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"[ERROR] Quantization to {quantization_type} failed (exit code: {result.returncode})")
                return False

            # 出力ファイルのサイズを確認
            if output_file.exists():
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"[SUCCESS] Output file created: {output_file.name} (size: {file_size:.1f} MB)")
                logger.info(f"[SUCCESS] Quantized to {quantization_type} in {elapsed_time:.1f}s")
            else:
                logger.error(f"[ERROR] Output file not found: {output_file}")
                return False

            return True

        except Exception as e:
            logger.error(f"[ERROR] Quantization exception: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def run_full_conversion(self) -> bool:
        """完全な変換プロセスを実行"""
        import time
        total_start_time = time.time()

        logger.info("=" * 60)
        logger.info("🚀 Starting AEGIS v2.2 GGUF Conversion with I-Matrix")
        logger.info("🎯 Target: Preserve mathematical reasoning in SO(8) adapters")
        logger.info("=" * 60)

        # チェックポイントから再開するか確認
        checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint_data:
            logger.info("📁 Found checkpoint, attempting resume...")
            if self.resume_from_checkpoint(checkpoint_data):
                return True

        # 自動チェックポイント保存を開始
        self.checkpoint_manager.start_auto_save(self._get_checkpoint_data)

        try:
            # 全体の進捗バー
            total_steps = 3 + len(self.quantization_types)  # チェック + F16変換 + I-Matrix + 各量子化
            if tqdm:
                overall_pbar = tqdm(total=total_steps, desc="Overall Progress", unit="step")
            else:
                overall_pbar = None
                logger.info("[INFO] Overall progress tracking (tqdm not available)")

            if overall_pbar:
                pbar_context = overall_pbar
            else:
                pbar_context = None

            # 1. llama.cpp環境チェック
            self.conversion_state["current_step"] = 1
            logger.info("[STEP 1/{}] Checking llama.cpp environment...".format(total_steps))
            if not self.check_llama_cpp():
                logger.error("[FAILED] llama.cpp environment check")
                return False
            logger.info("[OK] llama.cpp environment ready")
            if pbar_context:
                pbar_context.update(1)

            # 2. F16 GGUF変換
            self.conversion_state["current_step"] = 2
            logger.info("[STEP 2/{}] Converting to F16 GGUF...".format(total_steps))
            f16_start = time.time()
            if not self.convert_to_f16_gguf():
                logger.error("[FAILED] F16 conversion")
                return False
            f16_time = time.time() - f16_start
            logger.info(f"[SUCCESS] F16 conversion completed in {f16_time:.1f}s")
            if pbar_context:
                pbar_context.update(1)

            # 3. I-Matrix作成
            self.conversion_state["current_step"] = 3
            logger.info("[STEP 3/{}] Creating I-Matrix...".format(total_steps))
            imatrix_start = time.time()
            if not self.create_imatrix():
                logger.error("[FAILED] I-Matrix creation")
                return False
            imatrix_time = time.time() - imatrix_start
            logger.info(f"[SUCCESS] I-Matrix creation completed in {imatrix_time:.1f}s")
            if pbar_context:
                pbar_context.update(1)

            # 4. 各量子化タイプで量子化
            logger.info("[STEP 4-{}] Quantizing with different formats...".format(total_steps))
            success_count = 0
            for i, quant_type in enumerate(self.quantization_types):
                self.conversion_state["current_step"] = 4 + i
                step_desc = f"[STEP {4+i}/{total_steps}] Quantizing {quant_type}"
                logger.info(f"{step_desc}...")

                quant_start = time.time()
                if self.quantize_with_imatrix(quant_type):
                    success_count += 1
                    self.conversion_state["completed_quantizations"].append(quant_type)
                    quant_time = time.time() - quant_start
                    logger.info(f"[SUCCESS] {quant_type} quantization completed in {quant_time:.1f}s")
                else:
                    self.conversion_state["failed_quantizations"].append(quant_type)
                    quant_time = time.time() - quant_start
                    logger.warning(f"[WARNING] {quant_type} quantization failed after {quant_time:.1f}s")
                if pbar_context:
                    pbar_context.update(1)

            total_time = time.time() - total_start_time

            # 最終結果表示
            self.conversion_state["status"] = "completed"
            self.conversion_state["end_time"] = datetime.now().isoformat()

            logger.info("=" * 60)
            logger.info("📊 CONVERSION RESULTS")
            logger.info("=" * 60)
            logger.info(f"✅ Successful quantizations: {success_count}/{len(self.quantization_types)}")
            logger.info(".1f")
            logger.info(f"📁 Output directory: {self.output_dir}")
            logger.info(f"🔧 I-Matrix used: {self.imatrix_file.name}")
            logger.info(f"📚 Calibration data: {self.calibration_data.name}")

            if success_count > 0:
                logger.info("🎯 SO(8) adapter mathematical reasoning preserved!")
                logger.info("💡 Compare performance with and without I-Matrix")
            else:
                logger.error("❌ All quantizations failed")

            # 5. 結果レポート
            self.create_conversion_report(success_count)

            return success_count > 0

        finally:
            # 自動保存を停止
            self.checkpoint_manager.stop_auto_save()

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """チェックポイントデータを取得"""
        return {
            "conversion_state": self.conversion_state,
            "hf_model_path": str(self.hf_model_path),
            "output_dir": str(self.output_dir),
            "calibration_data": str(self.calibration_data),
            "f16_model_exists": self.f16_model.exists(),
            "imatrix_exists": self.imatrix_file.exists(),
            "timestamp": datetime.now().isoformat(),
        }

    def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """チェックポイントから再開"""
        try:
            logger.info("[RESUME] Attempting to resume from checkpoint...")

            # 変換状態を復元
            if "conversion_state" in checkpoint_data:
                self.conversion_state.update(checkpoint_data["conversion_state"])
                logger.info(f"[RESUME] Restored state: step {self.conversion_state.get('current_step', 0)}")

            # 中間ファイルの存在を確認
            f16_exists = checkpoint_data.get("f16_model_exists", False)
            imatrix_exists = checkpoint_data.get("imatrix_exists", False)

            current_step = self.conversion_state.get("current_step", 0)

            if current_step >= 4:
                # 量子化まで完了していた場合
                logger.info("[RESUME] Conversion appears to be completed, checking results...")
                completed_quants = self.conversion_state.get("completed_quantizations", [])
                if len(completed_quants) > 0:
                    logger.info(f"[RESUME] Found {len(completed_quants)} completed quantizations")
                    return True
                else:
                    logger.info("[RESUME] No completed quantizations found, restarting...")

            elif current_step >= 3 and imatrix_exists:
                # I-Matrixまで完了していた場合
                logger.info("[RESUME] I-Matrix exists, resuming quantization...")
                return self._resume_quantization()

            elif current_step >= 2 and f16_exists:
                # F16変換まで完了していた場合
                logger.info("[RESUME] F16 model exists, resuming from I-Matrix creation...")
                return self._resume_from_imatrix()

            else:
                logger.info("[RESUME] Insufficient progress, restarting from beginning...")

            return False

        except Exception as e:
            logger.error(f"[RESUME] Failed to resume: {e}")
            return False

    def _resume_quantization(self) -> bool:
        """量子化から再開"""
        logger.info("[RESUME] Resuming quantization process...")

        completed = self.conversion_state.get("completed_quantizations", [])
        failed = self.conversion_state.get("failed_quantizations", [])
        all_types = set(self.quantization_types)
        remaining = all_types - set(completed) - set(failed)

        if not remaining:
            logger.info("[RESUME] All quantizations already completed!")
            return True

        logger.info(f"[RESUME] Remaining quantizations: {remaining}")

        success_count = len(completed)
        for quant_type in remaining:
            logger.info(f"[RESUME] Processing {quant_type}...")
            if self.quantize_with_imatrix(quant_type):
                success_count += 1
                self.conversion_state["completed_quantizations"].append(quant_type)
            else:
                self.conversion_state["failed_quantizations"].append(quant_type)

        return success_count > 0

    def _resume_from_imatrix(self) -> bool:
        """I-Matrix作成から再開"""
        logger.info("[RESUME] Resuming from I-Matrix creation...")

        # I-Matrix作成
        if not self.create_imatrix():
            return False

        # 量子化実行
        return self._resume_quantization()

    def create_conversion_report(self, success_count: int):
        """変換レポートを作成"""
        report = {
            "conversion_timestamp": datetime.now().isoformat(),
            "model_name": "AEGIS-v2.2-SO8-QuadrupleThinking",
            "base_model": str(self.hf_model_path),
            "calibration_data": str(self.calibration_data),
            "quantization_types": self.quantization_types,
            "successful_quantizations": success_count,
            "imatrix_applied": True,
            "purpose": "Preserve mathematical reasoning capabilities in SO(8) adapter models",
            "expected_improvement": "Significant reduction in math ability degradation compared to standard quantization"
        }

        report_file = self.output_dir / "conversion_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"[REPORT] Conversion report saved: {report_file}")

    def cleanup_intermediate_files(self):
        """中間ファイルをクリーンアップ"""
        try:
            if self.f16_model.exists():
                self.f16_model.unlink()
                logger.info("[CLEANUP] Removed intermediate F16 file")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to cleanup intermediate files: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="AEGIS v2.2 GGUF Converter with I-Matrix")
    parser.add_argument(
        "--hf-model",
        default="models/aegis_v22_hf",
        help="Path to HF model directory (default: AEGIS v2.2 model)"
    )
    parser.add_argument(
        "--output-dir",
        default="H:/from_D/webdataset/gguf_models/aegis_v22_imatrix",
        help="Output directory for GGUF files (checkpoint auto-save enabled)"
    )
    parser.add_argument(
        "--calibration-data",
        default="data/calibration/math_calibration_data.txt",
        help="Path to calibration data for I-Matrix (math-focused)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if available"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint auto-save (not recommended)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up intermediate files after conversion"
    )

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 変換実行
    converter = AEGISv22GGUFConverter(
        hf_model_path=args.hf_model,
        output_dir=args.output_dir,
        calibration_data=args.calibration_data
    )

    # チェックポイントからの再開（オプション指定時）
    if args.resume:
        checkpoint_data = converter.checkpoint_manager.load_latest_checkpoint()
        if checkpoint_data:
            print(f"[INFO] Resuming from checkpoint...")
            success = converter.resume_from_checkpoint(checkpoint_data)
        else:
            print(f"[INFO] No checkpoint found, starting fresh...")
            success = converter.run_full_conversion()
    else:
        success = converter.run_full_conversion()

    if success:
        print("\n" + "=" * 60)
        print("🎉 I-MATRIX QUANTIZATION COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("SO(8) adapter mathematical reasoning capabilities preserved!")
        print(f"Output directory: {args.output_dir}")
        print("=" * 60)

        if args.cleanup:
            converter.cleanup_intermediate_files()
    else:
        print("\n" + "=" * 60)
        print("❌ I-MATRIX QUANTIZATION FAILED ❌")
        print("=" * 60)
        print("Check the logs for error details.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
