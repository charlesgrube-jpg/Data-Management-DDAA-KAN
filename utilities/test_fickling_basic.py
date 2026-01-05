import pickle
import sys
import subprocess
from pathlib import Path

SAFE_PKL = Path("safe_test.pkl")

def create_and_scan():
    # 1. Create trivial safe pickle
    print("Creating safe_test.pkl...")
    data = {"hello": "world", "numbers": [1, 2, 3]}
    with open(SAFE_PKL, "wb") as f:
        pickle.dump(data, f)
        
    # 2. Scan
    print("Scanning with Fickling...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "fickling", "--check-safety", str(SAFE_PKL)],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"Return Code: {result.returncode}")
        print(f"Stdout:\n{result.stdout}")
        print(f"Stderr:\n{result.stderr}")
        
        if result.returncode == 0:
            print("✅ SUCCESS: Trivial pickle scanned as SAFE.")
        else:
            print("❌ FAIL: Trivial pickle scan failed.")
            
    finally:
        if SAFE_PKL.exists():
            SAFE_PKL.unlink()

if __name__ == "__main__":
    create_and_scan()
