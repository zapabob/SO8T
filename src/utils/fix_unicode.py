#!/usr/bin/env python3

with open('scripts/training/aegis_finetuning_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode characters with ASCII
content = content.replace('[OK]', '[OK]')
content = content.replace('[NG]', '[NG]')
content = content.replace('[NG]', '[ERROR]')

with open('scripts/training/aegis_finetuning_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Unicode characters replaced successfully')










































