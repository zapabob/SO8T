# SO8T Audio Notification Script
# PlaySync()を使用して確実に音声を再生

param(
    [string]$AudioPath = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
)

if (Test-Path $AudioPath) {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer($AudioPath)
        $player.PlaySync()
        Write-Host "[OK] Audio notification played successfully" -ForegroundColor Green
        exit 0
    } catch {
        Write-Host "[WARNING] Failed to play audio: $($_.Exception.Message)" -ForegroundColor Yellow
        # フォールバック: システムビープ
        [System.Console]::Beep(1000, 500)
        exit 1
    }
} else {
    Write-Host "[WARNING] Audio file not found: $AudioPath" -ForegroundColor Yellow
    # フォールバック: システムビープ
    [System.Console]::Beep(800, 1000)
    exit 1
}








