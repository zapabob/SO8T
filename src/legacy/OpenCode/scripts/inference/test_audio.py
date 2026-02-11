"""音声通知テストスクリプト"""
import sys
from pathlib import Path

audio_path = Path(__file__).parent.parent / ".cursor" / "marisa_owattaze.wav"

if not audio_path.exists():
    print(f"Audio file not found: {audio_path}")
    sys.exit(1)

print(f"Playing audio: {audio_path}")

try:
    import winsound
    winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
    print("Audio played successfully using winsound")
except ImportError:
    print("winsound not available, trying subprocess...")
    import subprocess
    audio_str = str(audio_path).replace('\\', '/')
    cmd = f'powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer \'{audio_str}\'; $player.PlaySync()"'
    result = subprocess.run(cmd, shell=True)
    print(f"Audio played using subprocess (return code: {result.returncode})")

