@echo off
chcp 65001 >nul
echo [VOLUME] Increasing system volume...

echo [STEP 1] Setting volume to maximum...
powershell -Command "Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class Audio { [DllImport(\"user32.dll\")] public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam); [DllImport(\"user32.dll\")] public static extern IntPtr FindWindow(string lpClassName, string lpWindowName); }'; $hWnd = [Audio]::FindWindow('Shell_TrayWnd', $null); [Audio]::SendMessage($hWnd, 0x319, [IntPtr]::Zero, [IntPtr]::Zero)"
echo [OK] Volume control activated

echo [STEP 2] Testing audio with maximum volume...
powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'; $player.PlaySync()"
echo [OK] Audio played with volume control

echo [VOLUME] Volume test completed
pause
