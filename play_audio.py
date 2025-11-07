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
            # 方法1: PowerShell経由でSystem.Media.SoundPlayerを使用（最も確実）
            try:
                import subprocess
                # パスをエスケープ
                escaped_path = audio_file.replace("'", "''").replace('"', '""')
                ps_script = f'Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer("{escaped_path}"); $player.PlaySync()'
                result = subprocess.run(
                    ["powershell", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print("[OK] Audio notification played successfully (PowerShell SoundPlayer)")
                else:
                    print(f"[WARNING] PowerShell error: {result.stderr}")
                    raise Exception("PowerShell execution failed")
            except Exception as e:
                # 方法2: winsound（フォールバック）
                try:
                    import winsound
                    winsound.PlaySound(audio_file, winsound.SND_FILENAME)
                    print("[OK] Audio notification played successfully (winsound)")
                except Exception as e2:
                    raise Exception(f"Both methods failed: {e}, {e2}")
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
