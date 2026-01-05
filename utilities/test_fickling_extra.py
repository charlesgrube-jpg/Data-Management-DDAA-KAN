import sys
import zipfile
import subprocess
from pathlib import Path

QUARANTINE_DIR = Path("models/rvc/quarantine")
TEST_FILE = QUARANTINE_DIR / "biden_QUARANTINED.pth"
TEMP_PKL = Path("temp_data.pkl")

def test_extract_and_scan():
    if not TEST_FILE.exists():
        print(f"Test file {TEST_FILE} not found.")
        return

    print(f"Extracting data.pkl from {TEST_FILE.name}...")
    try:
        with zipfile.ZipFile(TEST_FILE, 'r') as zf:
            # Find the pickle file (usually ends in data.pkl)
            pkl_name = next((n for n in zf.namelist() if n.endswith('data.pkl')), None)
            if not pkl_name:
                print("No data.pkl found in zip.")
                return
                
            with zf.open(pkl_name) as source, open(TEMP_PKL, "wb") as target:
                target.write(source.read())
        print(f"Extracted to {TEMP_PKL}")
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return

    print("Running Fickling on extracted pickle...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "fickling", "--check-safety", str(TEMP_PKL)],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"Return Code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        if result.returncode == 0:
            print("✅ SUCCESS: Extracted pickle scanned as SAFE.")
        else:
            print("❌ FAIL: Extracted pickle scanned as UNSAFE or failed.")
            
    finally:
        if TEMP_PKL.exists():
            TEMP_PKL.unlink()

if __name__ == "__main__":
    test_extract_and_scan()
