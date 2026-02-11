try:
    import unsloth
    print('Unsloth version:', unsloth.__version__)
except ImportError:
    print('Unsloth not installed')
