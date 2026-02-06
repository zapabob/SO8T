#!/usr/bin/env python3
"""
HF Hubアップロードスクリプト
AEGIS-Phi-3.5-Instinct-JP-v2.0モデルのアップロード
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, HfFolder, login
import subprocess

def upload_to_hf_hub():
    """HF Hubにモデルをアップロード"""

    # HF Hub設定
    repo_name = "AEGIS-Phi-3.5-Instinct-JP-v2.0"
    repo_id = f"zapabobouj/{repo_name}"  # 実際のユーザー名に置き換え

    # パッケージパス
    package_path = Path("H:/from_D/webdataset/hf_upload_AEGIS-Phi-3.5-Instinct-JP-v2.0")

    if not package_path.exists():
        print(f"[ERROR] Package not found: {package_path}")
        return False

    try:
        # HF Hubログイン確認
        print("[INFO] Checking HF Hub login...")
        api = HfApi()

        # トークン取得
        token = HfFolder.get_token()
        if not token:
            print("[ERROR] HF Hub token not found. Please login first:")
            print("Run: huggingface-cli login")
            return False

        print("[OK] HF Hub login confirmed")

        # リポジトリ作成（存在しない場合）
        try:
            api.repo_info(repo_id=repo_id, token=token)
            print(f"[INFO] Repository already exists: {repo_id}")
        except Exception:
            print(f"[INFO] Creating new repository: {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                token=token,
                private=False,
                repo_type="model"
            )
            print(f"[OK] Repository created: {repo_id}")

        # ファイルアップロード
        print("[INFO] Starting file upload...")

        # アップロード対象ファイルの収集
        files_to_upload = []
        for root, dirs, files in os.walk(package_path):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(package_path)
                files_to_upload.append((file_path, str(relative_path)))

        print(f"[INFO] Found {len(files_to_upload)} files to upload")

        # ファイルアップロード（バッチ処理）
        batch_size = 10
        for i in range(0, len(files_to_upload), batch_size):
            batch = files_to_upload[i:i+batch_size]
            print(f"[INFO] Uploading batch {i//batch_size + 1}/{(len(files_to_upload)-1)//batch_size + 1}")

            for local_path, repo_path in batch:
                try:
                    print(f"[UPLOAD] {repo_path}")
                    api.upload_file(
                        path_or_fileobj=str(local_path),
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        token=token
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to upload {repo_path}: {e}")
                    return False

        print(f"[SUCCESS] All files uploaded to: https://huggingface.co/{repo_id}")

        # リポジトリ情報の更新
        print("[INFO] Updating repository metadata...")

        # READMEの内容を確認してリポジトリ情報を更新
        readme_path = package_path / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()

            # モデルカード更新
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=token
            )

        print("[SUCCESS] Model uploaded successfully!")
        print(f"[LINK] https://huggingface.co/{repo_id}")

        return True

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return False

def create_repo_manually():
    """手動でのリポジトリ作成手順を表示"""
    print("="*60)
    print("MANUAL REPOSITORY CREATION INSTRUCTIONS")
    print("="*60)
    print("1. Go to https://huggingface.co/new")
    print("2. Create a new repository with these settings:")
    print(f"   - Repository name: AEGIS-Phi-3.5-Instinct-JP-v2.0")
    print("   - Repository type: Model")
    print("   - Make it public")
    print("3. After creation, get your username from the URL")
    print("4. Update the repo_id in this script with your username")
    print("5. Run the upload again")
    print("="*60)

def main():
    """メイン関数"""
    print("[UPLOAD] Starting HF Hub upload for AEGIS-Phi-3.5-Instinct-JP-v2.0")
    print("="*60)

    # HF Hubログイン確認
    try:
        login_status = subprocess.run(
            ["hf", "auth", "whoami"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[OK] Logged in as: {login_status.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("[ERROR] Not logged in to HF Hub")
        print("Please run: huggingface-cli login")
        create_repo_manually()
        return

    # アップロード実行
    success = upload_to_hf_hub()

    if success:
        print("\n" + "="*60)
        print("[DONE] UPLOAD COMPLETED SUCCESSFULLY! [DONE]")
        print("="*60)
        print("Your model is now available on Hugging Face Hub!")
        print("Share the link with the research community.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[NG] UPLOAD FAILED [NG]")
        print("="*60)
        print("Check the error messages above and try again.")
        create_repo_manually()

if __name__ == "__main__":
    main()
