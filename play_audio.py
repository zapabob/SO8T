"""
音声通知再生スクリプト
"""
import os
import sys

audio_file = r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"

if os.path.exists(audio_file):
    try:
        # Windowsの場合
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(audio_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
            print("[OK] Audio notification played successfully (winsound)")
        else:
            # Linux/Macの場合
            import subprocess
            if sys.platform == "darwin":
                subprocess.run(["afplay", audio_file])
            else:
                subprocess.run(["aplay", audio_file])
            print("[OK] Audio notification played successfully")
    except Exception as e:
        print(f"[ERROR] Failed to play audio: {e}")
        # フォールバック: システムビープ
        try:
            import winsound
            winsound.Beep(1000, 500)
            print("[OK] Fallback beep played")
        except:
            print("[ERROR] All audio methods failed")
else:
    print(f"[WARNING] Audio file not found: {audio_file}")
    # 緊急ビープ
    try:
        import winsound
        winsound.Beep(800, 1000)
        print("[OK] Emergency beep played")
    except:
        print("[ERROR] Beep also failed")


