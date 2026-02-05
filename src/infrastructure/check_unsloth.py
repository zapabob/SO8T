try:
    import unsloth
    print(f"Unsloth available: {unsloth.__version__}")
except ImportError as e:
    print(f"Unsloth not available: {e}")
    print("Installing Unsloth...")

    import subprocess
    import sys

    try:
        # Unslothインストール
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth[colab-new]"])

        # xformersもインストール（Unsloth推奨）
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "xformers", "trl", "peft", "accelerate", "bitsandbytes"])

        import unsloth
        print(f"Unsloth installed successfully: {unsloth.__version__}")
    except Exception as install_error:
        print(f"Failed to install Unsloth: {install_error}")