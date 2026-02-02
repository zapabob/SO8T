"""
音声通知再生ヘルパースクリプト

Agent/Plan/Task完了時に音声通知を再生する。
"""

import os
import sys


def play_audio_notification():
    """
    音声通知を再生
    
    複数の方法を試行:
    1. winsound (Windows)
    2. システムビープ (フォールバック)
    """
    audio_file = r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
    
    if os.path.exists(audio_file):
        try:
            # Windowsの場合: winsoundを使用
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(audio_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                print("[OK] Audio notification played successfully (winsound)")
                return True
            else:
                # Linux/Macの場合
                import subprocess
                if sys.platform == "darwin":
                    subprocess.run(["afplay", audio_file], check=False)
                else:
                    subprocess.run(["aplay", audio_file], check=False)
                print("[OK] Audio notification played successfully")
                return True
        except Exception as e:
            print(f"[WARNING] Failed to play audio: {e}")
            # フォールバック: システムビープ
            try:
                if sys.platform == "win32":
                    import winsound
                    winsound.Beep(1000, 500)
                    print("[OK] Fallback beep played")
                    return True
            except Exception as beep_error:
                print(f"[ERROR] Beep also failed: {beep_error}")
                return False
    else:
        print(f"[WARNING] Audio file not found: {audio_file}")
        # 緊急ビープ
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(800, 1000)
                print("[OK] Emergency beep played")
                return True
        except Exception:
            print("[ERROR] All audio methods failed")
            return False
    
    return False


if __name__ == "__main__":
    play_audio_notification()

