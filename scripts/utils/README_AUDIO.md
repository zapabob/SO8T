# Audio Notification Scripts

## Standard Script
- **`play_audio_notification.ps1`**: 標準の音声通知スクリプト
  - 魔理沙のおわったぜ（marisa_owattaze.wav）を再生
  - フォールバックとしてビープ音を使用

## Deprecated Scripts (後方互換性のため保持)
- **`play_audio.ps1`**: `play_audio_notification.ps1`へのリダイレクト
- **`audio/play_audio.ps1`**: `play_audio_notification.ps1`へのリダイレクト

## Usage
```powershell
# 標準的な使用方法
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

# または、後方互換性のため
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio.ps1"
```

## Commit with Audio
```powershell
# コミット前に音声を鳴らしてからコミット
powershell -ExecutionPolicy Bypass -File "scripts\utils\commit_with_audio.ps1" -Message "コミットメッセージ" -All
```




























































