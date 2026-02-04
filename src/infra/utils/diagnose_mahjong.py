import os
import pathlib
from pathlib import Path

target = r'C:\Users\downl\AppData\Local\Programs\Python\Python312\Lib\site-packages\mahjong-1.3.0.dist-info'
entry_points = os.path.join(target, 'entry_points.txt')

print(f"Checking: {target}")
if os.path.exists(target):
    print(f"Exists: {target}")
    print(f"Is Dir: {os.path.isdir(target)}")
    print(f"Is File: {os.path.isfile(target)}")
    
    if os.path.exists(entry_points):
        print(f"Exists: {entry_points}")
        try:
            with open(entry_points, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"Content length: {len(content)}")
        except Exception as e:
            print(f"Error reading entry_points.txt: {e}")
    else:
        print(f"Missing: {entry_points}")
else:
    print(f"Missing: {target}")
