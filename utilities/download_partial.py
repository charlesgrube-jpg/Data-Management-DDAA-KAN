"""
Mozilla Common Voice Partial Download

Downloads only the first N bytes using Range header.
Note: Partial tar.gz won't be extractable but tests API access.
"""

import urllib.request
from pathlib import Path

# Config
API_KEY = "76224e89f20202746b3df283183916b676c25936608c9fd57546c5d4aae5d616"
CLIENT_ID = "mdc_e271ec33f12df105e4210e9d0dde9458"
DATASET_ID = "cmj8u3p1w0075nxxbe8bedl00"  # Note: ends with l00 not 100
TOKEN = "dlt_92636c20-bbf1-45be-9ca0-c9239ebfd255"  # Expires 2025-12-24

# Download size (500 MB)
MAX_BYTES = 500 * 1024 * 1024

def download_partial():
    print("=" * 50)
    print("MOZILLA CV PARTIAL DOWNLOAD")
    print("=" * 50)
    print(f"Target: First {MAX_BYTES / 1024 / 1024:.0f} MB")
    
    output_dir = Path("./mozilla_cv_data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "cv_partial_500mb.tar.gz"
    
    url = f"https://datacollective.mozillafoundation.org/api/datasets/{DATASET_ID}/download/{TOKEN}"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Range": f"bytes=0-{MAX_BYTES-1}"
    }
    
    print(f"\nURL: {url[:60]}...")
    print(f"Range: bytes=0-{MAX_BYTES-1}")
    
    request = urllib.request.Request(url, headers=headers)
    
    try:
        print("\nDownloading...")
        with urllib.request.urlopen(request, timeout=300) as response:
            status = response.status
            content_range = response.headers.get('Content-Range')
            content_length = response.headers.get('Content-Length')
            
            print(f"  Status: {status}")
            print(f"  Content-Range: {content_range}")
            print(f"  Content-Length: {content_length}")
            
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"\r  Downloaded: {downloaded/1024/1024:.1f} MB", end="", flush=True)
            
            print(f"\n\n✓ Saved to {output_path}")
            print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
            return True
            
    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP {e.code}: {e.reason}")
        try:
            print(f"  Body: {e.read().decode()[:500]}")
        except:
            pass
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = download_partial()
    exit(0 if success else 1)
