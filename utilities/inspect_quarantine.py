import sys
import zipfile
import pickle
from pathlib import Path

QUARANTINE_DIR = Path("models/rvc/quarantine")

def inspect_file(path: Path):
    print(f"Inspecting {path.name}...")
    
    # Check size
    size = path.stat().st_size
    print(f"  Size: {size / 1024 / 1024:.2f} MB")
    
    # Check Magic Bytes
    with open(path, 'rb') as f:
        magic = f.read(4)
    print(f"  Magic: {magic!r}")
    
    # Check if Zip
    if zipfile.is_zipfile(path):
        print("  Type: ZIP Archive")
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                print(f"  Contents: {zf.namelist()[:5]}")
        except Exception as e:
            print(f"  Zip Error: {e}")
    else:
        print("  Type: Not a Zip")
        try:
            # Try pickle load (unsafe but we are debugging)
            # Actually, let's just check standard pickle opcodes
            with open(path, 'rb') as f:
                header = f.read(2)
            print(f"  Pickle Header: {header!r}")
        except:
            pass

def main():
    files = list(QUARANTINE_DIR.glob("*_QUARANTINED*"))
    if not files:
        print("No files found.")
        return
        
    for f in files[:3]: # check first 3
        inspect_file(f)

if __name__ == "__main__":
    main()
