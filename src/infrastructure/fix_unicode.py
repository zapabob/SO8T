#!/usr/bin/env python3
import re

# Unicode絵文字と特殊文字を削除してASCII互換にする
with open('prepare_arxiv_biorxiv_data.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 絵文字とチェックマークを削除
content = content.replace('[RESEARCH]', '').replace('📚', '').replace('[FIX]', '').replace('💾', '').replace('[OK]', '')
content = content.replace('[STATS]', '').replace('🏷️', '').replace('📚', '').replace('[DONE]', '').replace('[DIR]', '').replace('[START]', '')
content = content.replace('[OK]', '').replace('[OK]', '')

with open('prepare_arxiv_biorxiv_data.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Unicode emojis removed from prepare_arxiv_biorxiv_data.py")