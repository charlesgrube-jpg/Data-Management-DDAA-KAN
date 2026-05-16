import csv
import os
from pathlib import Path

# Simulate the exact path logic of the training scripts
base_dir = os.path.expanduser("~/project/Data-Management-DDAA-KAN")
os.chdir(base_dir)
print(f"Checked CWD: {os.getcwd()}")

csv_path = "unified_dataset/train.csv"
if not os.path.exists(csv_path):
    print(f"ERROR: CSV not found at {csv_path}")
    exit(1)

print(f"Reading {csv_path}...")
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 3: break # Just check first 3
        
        path_str = row.get('full_path', row.get('filename', ''))
        p = Path(path_str)
        exists = p.exists()
        
        print(f"\n[Row {i}]")
        print(f"  Path in CSV: {path_str}")
        print(f"  Absolute:    {p.absolute()}")
        print(f"  Exists?      {exists}")
        
        if not exists:
            # Try prepending unified_dataset just in case
            alt = Path("unified_dataset") / path_str
            print(f"  Alt Check ({alt}): {alt.exists()}")
