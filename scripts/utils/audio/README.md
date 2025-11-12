# Audio Utilities

音声通知機能のユーティリティスクリプト集。

## ファイル一覧

### play_audio_notification.py
Agent/Plan/Task完了時に音声通知を再生するPythonスクリプト。

**使用方法:**
```python
from scripts.utils.audio.play_audio_notification import play_audio_notification
play_audio_notification()
```

### play_completion_sound.ps1
PowerShell版の音声通知スクリプト。

**使用方法:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/utils/audio/play_completion_sound.ps1
```

### テストスクリプト
- `audio_test_fixed.bat`: 音声再生テスト（修正版）
- `enhanced_audio_test.bat`: 強化版音声テスト
- `simple_audio_test.bat`: シンプル音声テスト
- `test_audio.bat`: 音声テスト
- `test_audio_now.bat`: 即座に音声テスト

### その他
- `volume_up.bat`: 音量を上げるユーティリティ

## 音声ファイル

音声ファイルは `.cursor/marisa_owattaze.wav` に配置されています。

## 統合方法

他のスクリプトから音声通知を使用する場合:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.utils.audio.play_audio_notification import play_audio_notification

# タスク完了時
play_audio_notification()
```

## 注意事項

- Windows環境では`winsound`を使用
- Linux/Mac環境では`aplay`または`afplay`を使用
- フォールバックとしてシステムビープを使用





















