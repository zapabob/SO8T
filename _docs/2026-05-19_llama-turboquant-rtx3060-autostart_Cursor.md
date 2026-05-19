# 2026-05-19 SuperGemma4 llama-server RTX3060 TurboQuant 自動起動セットアップ

<!-- なんJ風: ガチ勢向け実装ログや -->

## 概要

SuperGemma4-E4B (abliterated) GGUF モデルを RTX 3060 (12GB VRAM) 上で
llama-turboquant の `llama-server.exe` で OpenAI 互換 API として提供する。
Flash Attention + KV cache q8_0 でメモリ効率を最適化し、ログオン時に自動起動する。

---

## 調査結果サマリー (Deep Research)

### TurboQuant (TQ) とは

| 項目 | 内容 |
|------|------|
| 開発者 | TheTom / turbo-tan (コミュニティフォーク) |
| 変換方式 | Walsh-Hadamard Transform (WHT) ローテーション + 最適スカラーコードブック量子化 |
| フォーマット | TQ3_0, TQ3_1S (~3.5 bit/val), TQ4_0, TQ4_1S (~4.0-4.5 bit/val) |
| KV cache タイプ | `-ctv turbo3` / `-ctv turbo4` として llama-server に指定可能 |
| upstream 状態 | PR #21089 でマージ提案中 (2026/05 時点、まだ未マージ) |
| 推奨設定 | `-ctk q8_0 -ctv turbo4 -fa on` (品質重視) |
| 最大圧縮 | `-ctk turbo3 -ctv turbo3 -fa on` |

**参考 URL:**
- https://github.com/TheTom/turboquant_plus/blob/main/docs/getting-started.md
- https://github.com/turbo-tan/llama.cpp-tq3
- https://turbo-quant.com/turboquant-llama-cpp (Issue #20977, PR #21089)
- https://github.com/TheTom/llama-cpp-turboquant/pull/45 (TQ4_1S CUDA port)

### RTX 3060 (Ampere) + llama.cpp 最適化ポイント

| パラメータ | 推奨値 | 理由 |
|-----------|--------|------|
| `-ngl 99` | 全レイヤーGPU | 12GB あれば全層オフロード可 |
| `-fa on` | Flash Attention ON | Ampere (sm86) 対応、VRAM使用量削減 |
| `-ctk q8_0` | K キャッシュ q8_0 | 標準的な量子化。turbo4 より互換性高 |
| `-ctv q8_0` | V キャッシュ q8_0 | 同上 |
| `-c 8192` | コンテキスト長 | 12GB での安定動作上限 (Q8_0 モデル使用時) |
| `-b 512` | バッチサイズ | Ampere 最適値 |
| `-ub 512` | マイクロバッチ | 同上 |
| `-np 1` | スロット数 | 12GB シングルスロット推奨 |
| `-cb` | 連続バッチ | スループット向上 |

**Flash Attention 参考:**
- https://xhinker.medium.com/the-5-llama-cpp-parameters-that-actually-matter-9f2c38b53755
- https://inferencerig.com/performance/best-settings-for-llama-cpp-speed-vs-quality-optimization-guide/

---

## 環境情報

| 項目 | 値 |
|------|-----|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 3060 |
| VRAM | 12288 MiB |
| CUDA | 13.2 (Driver 596.49) |
| llama-server バージョン | 9458 (31b900be6) — ビルド 2026-05-18 |
| バイナリパス | `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe` |

---

## モデル情報

### 使用モデル (メイン)
| 項目 | 値 |
|------|-----|
| モデル | supergemma4-e4b-abliterated (Abiray) |
| 量子化 | Q8_0 |
| ファイルサイズ | **7.48 GB** |
| フルパス | `C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.gguf` |

### フォールバックモデル (TurboQuant版 — VRAM不足時)
| 項目 | 値 |
|------|-----|
| 量子化 | TQ4_1S (TurboQuant 4-bit) |
| ファイルサイズ | **6.43 GB** |
| フルパス | `...supergemma4-Q8_0.tq4_1s.gguf` |

### VRAM 見積もり (Q8_0 モデル使用時)

```
モデルウェイト:   ~7,680 MiB  (7.48GB × 1,024)
KV cache q8_0:   ~1,024 MiB  (ctx=8192, Gemma4-E4B 32層)
バッファ・オーバーヘッド: ~300 MiB
─────────────────────────────────
合計推定:       ~9,000 MiB  ← RTX 3060 12,288 MiB に収まる ✓
余裕:           ~3,288 MiB
```

> **注意:** コンテキストを 16384 に増やすと KV キャッシュが ~2GB 増えて厳しくなる。
> 12GB では `-c 8192` を上限とすることを推奨。

---

## 参考コマンド vs SuperGemma4 適応 (2026-05-19 更新)

**参考元:** Qwen3.6-35B-A3B MoE on larger GPU (ユーザー貼付)  
**バイナリ:** llama-server v9458 (31b900be6) — 全マージ対象フラグを `--help` で確認済

| フラグ | 参考 (35B MoE) | SuperGemma4 (RTX 3060) | 根拠 |
|--------|----------------|----------------------|------|
| `-m` | Qwen3.6-35B Q4_K_M | supergemma4-Q8_0.gguf (7.48GB) | 指定モデル |
| `-ngl` | 999 | **99** | 全層GPU; 99で十分 (dense 4B) |
| `-ncmoe` | 30 | **未使用** | MoE専用; SuperGemmaはdense |
| `-fa on` | on | **on** | Flash Attn; Ampere対応 |
| `--cache-type-k/v` | q8_0 | **turbo4** (既定) / q8_0 (`-Profile Stable`) | turbo4でKV VRAM約40%削減 |
| `-c` | 32768 | **8192** | 32768はKV~4GB+で12GB OOM |
| `-n` | 8192 | **4096** | 最大生成トークン; APIで短く可 |
| `-np` | 1 | **1** | シングルスロット |
| `-t` | 6 | **6** | CPUスレッド (prefill/HTTP) |
| `--reasoning` | off | **off** | 思考モード不要 |
| `--no-cache-prompt` | (ref) | **あり** | プロンプトキャッシュ無効 |
| `--checkpoint-every-n-tokens` | -1 | **-1** | prefill中チェックポイント無効 |
| `--jinja` | あり | **あり** | チャットテンプレ (既定ONだが明示) |
| `--metrics` | あり | **あり** | Prometheus `/metrics` |
| `--host` | 0.0.0.0 | **127.0.0.1** (既定) | セキュリティ; LANは別スクリプト |
| `-cb` | (なし) | **あり** | 連続バッチ (スループット) |
| `-b / -ub` | (なし) | **512** | Ampere向けバッチ |

**未サポート / 該当なし:** なし (v9458 で上記すべて存在)

**LAN 用:** `scripts\start-supergemma-server-lan.ps1` → `--host 0.0.0.0`（ファイアウォール必須）

---

## 最終起動コマンドライン

```powershell
& "C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe" `
    -m "C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.gguf" `
    --host 127.0.0.1 --port 8080 --api-key llama-cpp-local `
    -ngl 99 -c 8192 -n 4096 -t 6 -b 512 -ub 512 -np 1 -cb -fa on `
    --cache-type-k turbo4 --cache-type-v turbo4 `
    --reasoning off --no-cache-prompt --checkpoint-every-n-tokens -1 `
    --jinja --metrics
```

**DryRun:** `.\scripts\start-supergemma-server.ps1 -DryRun`  
**KVプロファイル:** `-Profile Turbo` (既定, turbo4) / `-Profile Stable` (q8_0)  
**LAN:** `-Lan` または `start-supergemma-server-lan.ps1`

**API エンドポイント:** `http://127.0.0.1:8080/v1`  
**Metrics:** `http://127.0.0.1:8080/metrics`  
**API キー:** `llama-cpp-local`

---

## 作成ファイル一覧

| ファイル | 用途 |
|---------|------|
| `C:\Users\downl\Desktop\SO8T\scripts\start-supergemma-server.ps1` | 起動スクリプト (127.0.0.1) |
| `C:\Users\downl\Desktop\SO8T\scripts\start-supergemma-server-lan.ps1` | LAN用 (`0.0.0.0`) |
| `C:\Users\downl\Desktop\SO8T\scripts\install-autostart-supergemma.ps1` | 自動起動管理スクリプト |
| `C:\Users\downl\Desktop\SO8T\scripts\create-desktop-shortcut.ps1` | ショートカット作成スクリプト |
| `C:\Users\downl\Desktop\SO8T\logs\llama-server-supergemma4.log` | サーバーログ |
| `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SuperGemma4-LlamaServer.vbs` | ログオン自動起動 VBS |
| `C:\Users\downl\Desktop\SuperGemma4 llama-server (RTX3060).lnk` | デスクトップショートカット |
| `C:\Users\downl\Desktop\SuperGemma4 API Docs (localhost 8080).url` | API ドキュメントショートカット |

---

## 自動起動の仕組み

### 採用方式: Startup フォルダ (管理者権限不要)

```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SuperGemma4-LlamaServer.vbs
```

**VBS 内容:**
```vbscript
Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -WindowStyle Hidden -NonInteractive -ExecutionPolicy Bypass -File ""[スクリプトパス]""", 0, False
```

- ログオン時に自動実行される
- ウィンドウは非表示 (バックグラウンド実行)
- `False` = 非同期起動 (ログオンをブロックしない)

### 自動起動の無効化方法

```powershell
# 方法1: スクリプトで削除
& "C:\Users\downl\Desktop\SO8T\scripts\install-autostart-supergemma.ps1" -Action uninstall

# 方法2: 直接削除
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\SuperGemma4-LlamaServer.vbs"

# 方法3: エクスプローラーから削除
# Win + R → shell:startup → SuperGemma4-LlamaServer.vbs を削除
```

---

## サーバーの手動停止

```powershell
# llama-server プロセスを停止
Get-Process -Name "llama-server" | Stop-Process -Force

# または ポート 8080 で特定
$pid = (netstat -ano | Select-String ":8080" | Where-Object {$_ -match "LISTENING"} | ForEach-Object { ($_ -split "\s+")[-1] }) | Select-Object -First 1
Stop-Process -Id $pid -Force
```

---

## Hermes Agent 連携

`~/.hermes/config.yaml` に以下を追記:

```yaml
model:
  base_url: "http://127.0.0.1:8080/v1"
  api_key: "llama-cpp-local"
  name: "supergemma4-e4b-abliterated"
```

または環境変数:
```bash
OPENAI_API_BASE=http://127.0.0.1:8080/v1
OPENAI_API_KEY=llama-cpp-local
```

---

## トラブルシューティング

### ポート 8080 が既に使用中
```powershell
# 確認
netstat -ano | Select-String ":8080" | Where-Object {$_ -match "LISTENING"}

# llama-server なら停止して再起動
Get-Process -Name "llama-server" | Stop-Process -Force
# その後 start-supergemma-server.ps1 を実行
```

### VRAM 不足でクラッシュ
- コンテキスト長を下げる: `-c 4096`
- TQ4_1S モデルに切り替え (6.43GB): スクリプト内 `$ModelPath = $MODEL_TQ` に変更
- KV cache を turbo4 に変更: `-ctk turbo4 -ctv turbo4`
- GPU層を減らす: `-ngl 28` (一部 CPU オフロード)

### Flash Attention が無効になる
```powershell
# -fa on を明示指定 (auto では無効になる場合あり)
# RTX 3060 はAmpere (sm86) なので対応済
```

### サーバーが起動しない
```powershell
# ログ確認
Get-Content "C:\Users\downl\Desktop\SO8T\logs\llama-server-supergemma4.log" -Tail 50

# 手動テスト起動
& "C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe" --version
nvidia-smi
```

### CUDA が認識されない
```powershell
nvidia-smi
# → CUDA Version: 13.2 が表示されれば OK
# ggml-cuda.dll の存在確認:
Test-Path "C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\ggml-cuda.dll"
```

---

## 現行稼働サーバー (既存)

起動スクリプト実行時点での既存サーバー情報:

```
PID     : 31296
コマンド: llama-server.exe -m H:\elt_data\releases\...-q8_0.gguf --host 127.0.0.1 --port 8080 --api-key llama-cpp-local -ngl 99 -c 16384
```

新しい start-supergemma-server.ps1 はポート競合を検出し、既存の llama-server を自動停止してから起動する。

---

## 残留リスク

| リスク | 重大度 | 対策 |
|--------|--------|------|
| VRAM 不足 (9GBが既に使用中) | 中 | 他GPUプロセス終了、TQ4_1S利用 |
| ポート 8080 占有 | 低 | スクリプトが自動検出・停止 |
| Startup VBS はログオン後30秒程度かかる | 低 | 許容範囲 |
| Task Scheduler (Admin) は未登録 | 低 | Startup フォルダで代替 |
| モデルの量子化品質 | 低 | Q8_0は最高品質の整数量子化 |

---

## 次の推奨アクション

1. **動作確認**: `start-supergemma-server.ps1` を手動実行してサーバーが立ち上がることを確認
2. **API テスト**: `curl http://127.0.0.1:8080/v1/models -H "Authorization: Bearer llama-cpp-local"`
3. **TQ KV cache 試験**: `-ctk turbo4 -ctv turbo4` でVRAM節約&速度向上を計測
4. **Hermes 連携**: `config.yaml` に上記 API 設定を追加
5. **ベンチマーク**: `llama-bench.exe -m [model] -ngl 99 -fa 1` でトークン/秒計測
6. **Task Scheduler (管理者で)**: 管理者 PowerShell から `install-autostart-supergemma.ps1 -Action install` を再試行

---

## 実装者

- AI: Cursor Sonnet 4.6 (subagent)
- 日付: 2026-05-19
- ワークスペース: `C:\Users\downl\Desktop\hermes-agent-main\hermes-agent-main`
