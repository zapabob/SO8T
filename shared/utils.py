from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def play_completion_sound() -> None:
    """
    CursorびAgentやPlanが終わった後に音声ファイルを再生する関数
    marisa_owattaze.wavを再生して完了を通知する
    """
    try:
        # 音声ファイルのパスを取得
        current_dir = Path(__file__).resolve().parent.parent
        audio_file = current_dir / ".cursor" / "marisa_owattaze.wav"
        
        if not audio_file.exists():
            print(f"⚠️ 音声ファイルが見つかりません: {audio_file}")
            return
        
        # Windows環境での音声再生
        if sys.platform == "win32":
            try:
                # winsoundモジュールを使用（Windows標準）
                import winsound
                winsound.PlaySound(str(audio_file), winsound.SND_FILENAME | winsound.SND_ASYNC)
                print("🔊 完了音声を再生しました！")
            except ImportError:
                # winsoundが使えない場合はPowerShellで再生
                try:
                    subprocess.run([
                        "powershell", "-Command", 
                        f"Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('{audio_file}').PlaySync()"
                    ], check=True, capture_output=True)
                    print("🔊 完了音声を再生しました！")
                except subprocess.CalledProcessError:
                    print("⚠️ 音声再生に失敗しました")
        else:
            # Linux/Mac環境での音声再生
            try:
                subprocess.run(["aplay", str(audio_file)], check=True, capture_output=True)
                print("🔊 完了音声を再生しました！")
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(["afplay", str(audio_file)], check=True, capture_output=True)
                    print("🔊 完了音声を再生しました！")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("⚠️ 音声再生に失敗しました")
                    
    except Exception as e:
        print(f"⚠️ 音声再生中にエラーが発生しました: {e}")


def notify_task_completion(task_name: str = "タスク") -> None:
    """
    タスク完了を通知する関数
    音声再生とコンソール出力を組み合わせる
    """
    print(f"\n🎉 {task_name}が完了しました！")
    print("=" * 50)
    play_completion_sound()
    print("=" * 50)
