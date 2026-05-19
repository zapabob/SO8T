# SuperGemma4 llama-server 完全ガイド (RTX 3060 / SO8T)

Windows 11 + RTX 3060 12GB で `supergemma4-Q8_0.gguf` を OpenAI 互換 API として動かす手順まとめ。

---

## 日常の使い方（これだけ覚えればOK）

| やりたいこと | 操作 |
|-------------|------|
| **手動起動** | デスクトップ `SuperGemma4 llama-server (RTX3060).lnk` をダブルクリック |
| **API** | `http://127.0.0.1:8080/v1` |
| **APIキー** | `llama-cpp-local` |
| **自動起動** | ログオン時に Startup VBS が同スクリプトを実行（既定: turbo4 + ctx 100000） |
| **安定モード** | `.\scripts\start-supergemma-server.ps1 -Profile Stable`（KV q8_0 + ctx 8192） |
| **VRAM節約** | 既定の `-Profile Turbo`（KV turbo4） |
| **安全なコンテキスト** | `.\scripts\start-supergemma-server.ps1 -ContextSize 8192` |
| **OOM時** | `.\scripts\start-supergemma-server.ps1 -AutoFallbackContext` |

---

## ファイル一覧

| パス | 用途 |
|------|------|
| `scripts\start-supergemma-server.ps1` | メイン起動（127.0.0.1） |
| `scripts\start-supergemma-server-lan.ps1` | LAN (`0.0.0.0`) |
| `scripts\install-autostart-supergemma.ps1` | Startup VBS 登録/削除 |
| `logs\llama-server-supergemma4.log` | サーバーログ |
| `%APPDATA%\...\Startup\SuperGemma4-LlamaServer.vbs` | ログオン自動起動 |
| デスクトップ `SuperGemma4 llama-server (RTX3060).lnk` | 手動起動 |
| デスクトップ `SuperGemma4 API Docs (localhost 8080).url` | `/docs` を開く |

**バイナリ:** `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe` (v9458)

**モデル:** `gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.gguf` (7.48 GB)

---

## 既定コマンド（ユーザー要望: ctx 100000）

```text
llama-server.exe
  -m ...\supergemma4-Q8_0.gguf
  --host 127.0.0.1 --port 8080 --api-key llama-cpp-local
  -ngl 99 -c 100000 -n 4096 -t 6 -b 512 -ub 512 -np 1 -cb -fa on
  --cache-type-k turbo4 --cache-type-v turbo4
  --reasoning off --no-cache-prompt --checkpoint-every-n-tokens -1
  --jinja --metrics
```

PowerShell から:

```powershell
cd C:\Users\downl\Desktop\SO8T
.\scripts\start-supergemma-server.ps1
```

---

## KV プロファイル: turbo4 vs q8_0

| プロファイル | KV cache | 既定 ctx | 用途 |
|-------------|----------|---------|------|
| **Turbo** (既定) | turbo4 | **100000** | VRAM節約・ユーザー要望の大コンテキスト |
| **Stable** | q8_0 | **8192** | 品質優先・VRAM安全 |

```powershell
.\scripts\start-supergemma-server.ps1 -Profile Turbo          # turbo4 + c=100000
.\scripts\start-supergemma-server.ps1 -Profile Stable         # q8_0 + c=8192
.\scripts\start-supergemma-server.ps1 -Profile Stable -ContextSize 16384
```

---

## コンテキスト 100000 と VRAM（重要）

### 理論上の見積もり（RTX 3060 12GB）

| 構成 | 目安 VRAM |
|------|----------|
| モデル Q8_0 (-ngl 99) | ~7.5 GB |
| KV turbo4 @ ctx 8192 | ~0.6 GB |
| KV turbo4 @ ctx 100000 | **~7+ GB**（線形スケール近似） |
| 合計 @ 100k | **14GB超の可能性** → 理論上は厳しい |

### 実測（2026-05-19, llama-turboquant v9458）

| -c | -ngl | KV | 起動 /health | 備考 |
|----|------|-----|-------------|------|
| **100000** | 99 | turbo4 | **成功** | VRAM ~6.8GB 時点（KVは利用時に伸びる可能性） |
| 32768 | 99 | turbo4 | 成功 | 余裕あり |
| 8192 | 99 | turbo4 | 成功 | 日常推奨の安全値 |

**結論:**

- **`-c 100000` は起動・/health まで成功**（ユーザー要望どおりスクリプト既定に設定済み）
- 長文を実際に 100k まで埋めると VRAM が増え **OOM の可能性あり**
- 安定運用: `-ContextSize 8192` または `-Profile Stable`
- OOM 時: `-AutoFallbackContext`（32768→16384→8192 を自動試行）

### 100k を本当に使い切りたい場合

1. `-Profile Turbo`（turbo4）を維持
2. OOM したら `-GpuLayers 28` などで一部 CPU オフロード
3. または `-AutoFallbackContext` で実測可能な最大 ctx を自動選択
4. 24GB+ GPU なら `-c 100000` が現実的

---

## 自動起動

**場所:** `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SuperGemma4-LlamaServer.vbs`

**無効化:**

```powershell
.\scripts\install-autostart-supergemma.ps1 -Action uninstall
# または shell:startup から VBS を削除
```

**再登録:**

```powershell
.\scripts\install-autostart-supergemma.ps1 -Action install
```

---

## Hermes Agent 連携

`~/.hermes/config.yaml`:

```yaml
model:
  base_url: "http://127.0.0.1:8080/v1"
  api_key: "llama-cpp-local"
  name: "supergemma4-Q8_0"
```

環境変数:

```text
OPENAI_API_BASE=http://127.0.0.1:8080/v1
OPENAI_API_KEY=llama-cpp-local
```

---

## トラブルシューティング

### ポート 8080 が使用中

```powershell
Get-Process llama-server | Stop-Process -Force
.\scripts\start-supergemma-server.ps1
```

### VRAM 不足 / クラッシュ

```powershell
.\scripts\start-supergemma-server.ps1 -AutoFallbackContext
.\scripts\start-supergemma-server.ps1 -Profile Stable
.\scripts\start-supergemma-server.ps1 -ContextSize 8192 -GpuLayers 28
```

### 動作確認

```powershell
Invoke-WebRequest http://127.0.0.1:8080/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8080/v1/models -Headers @{Authorization="Bearer llama-cpp-local"} -UseBasicParsing
Get-Content .\logs\llama-server-supergemma4.log -Tail 30
nvidia-smi
```

### LAN 公開（注意）

```powershell
.\scripts\start-supergemma-server-lan.ps1
# または -Lan スイッチ
```

ファイアウォールで 8080 を制限すること。api-key は必須。

---

## パラメータ早見表

| スイッチ | 既定 | 説明 |
|---------|------|------|
| `-Profile` | Turbo | Turbo=turbo4 / Stable=q8_0 |
| `-ContextSize` | **100000** | `-c` / `--ctx-size` |
| `-GpuLayers` | 99 | `-ngl` |
| `-Lan` | off | `0.0.0.0` で待受 |
| `-AutoFallbackContext` | off | OOM時 ctx 段階的に下げる |
| `-DryRun` | off | コマンド表示のみ |

---

## 参考

- 詳細ログ: `_docs/2026-05-19_llama-turboquant-rtx3060-autostart_Cursor.md`
- TurboQuant: https://github.com/TheTom/turboquant_plus
