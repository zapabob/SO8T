
try:
    with open('logs/aegis_v3_pipeline.log', 'r', encoding='utf-16') as f:
        content = f.read()
    with open('temp_log_utf8.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully converted {len(content)} characters.")
except Exception as e:
    print(f"Error: {e}")
