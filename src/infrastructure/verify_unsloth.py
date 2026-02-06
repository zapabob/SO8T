try:
    import unsloth
    print(f"[OK] Unsloth {unsloth.__version__} successfully installed!")
    print("[START] Ready for Lightning-Fast SO8T Training!")

    # Unslothの機能をテスト
    from unsloth import FastLanguageModel
    print("[OK] FastLanguageModel imported successfully")

except ImportError as e:
    print(f"[NG] Unsloth not yet installed: {e}")
    print("⏳ Installation still in progress...")