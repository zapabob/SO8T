# uv 依存関係インストールガイド

## 概要

`uv`を使用した依存関係のインストール方法を説明します。`uv`は高速なPythonパッケージマネージャーです。

## インストール方法

### Windows

```powershell
# 1. Pythonインタープリターの検出とインストール
.\scripts\utils\setup\fix_uv_python_path.ps1

# 2. 依存関係のインストール
.\scripts\utils\setup\install_dependencies_uv.ps1

# 3. Flash Attentionのみインストール（オプショナル）
.\scripts\utils\setup\install_flash_attn_uv.ps1
```

### Linux/WSL2

```bash
# 1. スクリプトに実行権限を付与
chmod +x scripts/utils/setup/*.sh

# 2. Pythonインタープリターの検出
bash scripts/utils/setup/fix_uv_python_path.sh

# 3. 依存関係のインストール
bash scripts/utils/setup/install_dependencies_uv.sh

# 4. Flash Attentionのみインストール（オプショナル）
bash scripts/utils/setup/install_flash_attn_uv.sh
```

## uv のインストール

### Windows

```powershell
pip install uv
```

### Linux/WSL2

```bash
# 方法1: 公式インストーラー（推奨）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 方法2: pip経由
pip install uv

# 方法3: snap経由
sudo snap install astral-uv
```

インストール後、PATHに追加：
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## 手動インストール

### 明示的にPythonパスを指定

#### Windows (PowerShell)

```powershell
$pythonExe = "C:\Users\downl\AppData\Local\Programs\Python\Python312\python.exe"
uv pip install --python $pythonExe -e .
```

#### Linux/WSL2 (Bash)

```bash
pythonExe=$(python3 -c "import sys; print(sys.executable)")
uv pip install --python "$pythonExe" -e .
```

### 環境変数で設定

#### Windows (PowerShell)

```powershell
$env:UV_PYTHON = "C:\Users\downl\AppData\Local\Programs\Python\Python312\python.exe"
uv pip install -e .
```

#### Linux/WSL2 (Bash)

```bash
export UV_PYTHON=$(python3 -c "import sys; print(sys.executable)")
uv pip install -e .
```

## Flash Attention について

### Windows

- Flash Attentionのビルドは非常に困難です
- 推奨: WSL2またはLinux環境でインストール
- オプショナル: インストールされなくても標準attentionで動作します

### Linux/WSL2

- Flash Attentionのインストールが可能です
- CUDA環境が必要です
- インストールコマンド:
  ```bash
  uv pip install --python "$(python3 -c 'import sys; print(sys.executable)')" "flash-attn>=2.5.8" --no-build-isolation
  ```

## トラブルシューティング

### Python 3.13 エラー

**エラー**: `Failed to inspect Python interpreter from first executable in the search path at python3.13.exe`

**解決策**:
1. 明示的にPythonパスを指定
2. `fix_uv_python_path.ps1` (Windows) または `fix_uv_python_path.sh` (Linux) を実行

### uv コマンドが見つからない

**エラー**: `Command 'uv' not found`

**解決策**:
- Windows: `pip install uv`
- Linux/WSL2: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Chocolatey Python 3.13 の問題

**エラー**: `Cannot find file at 'c:\python313\python.exe'`

**解決策**:
- 実際にインストールされているPython（3.12など）を使用
- `fix_uv_python_path.ps1`で正しいPythonパスを検出

## 依存関係の構成

- **基本依存関係**: `pyproject.toml`の`dependencies`セクション
- **オプショナル依存関係**: `pyproject.toml`の`[project.optional-dependencies]`セクション
  - `flash-attention`: Flash Attention（Linux/WSL2のみ）
  - `gguf`: GGUFサポート
  - `models`: 追加モデルサポート
  - `database`: データベースサポート
  - `queues`: メッセージキューサポート
  - `monitoring`: 監視・可観測性
  - `security`: セキュリティ機能
  - `cloud`: クラウドストレージサポート

## 参考

- [uv公式ドキュメント](https://github.com/astral-sh/uv)
- [PyTorch CUDA 12.1インストール](https://pytorch.org/get-started/locally/)
































































































