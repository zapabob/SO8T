try:
    import unsloth
    print(f"✅ Unsloth {unsloth.__version__} successfully installed!")
    print("🚀 Ready for Lightning-Fast SO8T Training!")

    # Unslothの機能をテスト
    from unsloth import FastLanguageModel
    print("✅ FastLanguageModel imported successfully")

except ImportError as e:
    print(f"❌ Unsloth not yet installed: {e}")
    print("⏳ Installation still in progress...")