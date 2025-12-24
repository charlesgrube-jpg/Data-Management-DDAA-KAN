"""
Download Mozilla Common Voice via REST API (v2)

Uses the exact API endpoints from Mozilla Data Collective documentation.

Usage:
    python download_mozilla_api.py
"""

import os
import json
import urllib.request
from pathlib import Path


# API Configuration
API_KEY = "76224e89f20202746b3df283183916b676c25936608c9fd57546c5d4aae5d616"
CLIENT_ID = "mdc_e271ec33f12df105e4210e9d0dde9458"

# Common Voice Scripted Speech 24.0 - English (87.74 GB)
DATASET_ID = "cmj8u3p1w0075nxxbe8bedl00"  # Note: ends with l00 not 100
API_BASE = "https://datacollective.mozillafoundation.org/api/datasets"

# Current download token (expires 2025-12-24)
DOWNLOAD_TOKEN = "dlt_92636c20-bbf1-45be-9ca0-c9239ebfd255"


def test_api_connection():
    """Test basic API connectivity and list available datasets."""
    print("[0/3] Testing API connection...")
    
    # Try to list datasets or get info
    endpoints_to_try = [
        f"{API_BASE}/{DATASET_ID}",  # Get dataset info
        f"{API_BASE}",  # List datasets
        "https://datacollective.mozillafoundation.org/api",  # API root
    ]
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    for url in endpoints_to_try:
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=10) as response:
                data = response.read().decode()
                print(f"      ✓ {url[:50]}...")
                print(f"      Response: {data[:300]}")
                return True
        except urllib.error.HTTPError as e:
            print(f"      ✗ {url[:50]}... -> {e.code}")
        except Exception as e:
            print(f"      ✗ {url[:50]}... -> {e}")
    
    return False


def create_download_session():
    """
    Step 1: Create a download session to get a download token.
    Using exact URL from the screenshot.
    """
    print("\n[1/3] Creating download session...")
    
    # Exact URL from screenshot
    url = f"https://datacollective.mozillafoundation.org/api/datasets/{DATASET_ID}/download"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Empty JSON body for POST
    data = b'{}'
    
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_data = response.read().decode()
            print(f"      ✓ Response received")
            print(f"      Raw: {response_data[:300]}")
            
            try:
                return json.loads(response_data)
            except:
                return response_data
                
    except urllib.error.HTTPError as e:
        print(f"      ✗ HTTP {e.code}: {e.reason}")
        try:
            error_body = e.read().decode()
            print(f"      Body: {error_body[:500]}")
            
            # Try to parse error for hints
            try:
                error_json = json.loads(error_body)
                if "message" in error_json:
                    print(f"      Message: {error_json['message']}")
            except:
                pass
        except:
            pass
        return None
    except Exception as e:
        print(f"      ✗ Error: {e}")
        return None


def download_with_token(download_token: str, output_path: str, max_bytes: int = None):
    """
    Step 2: Download the dataset using the token.
    """
    print(f"\n[2/3] Downloading with token...")
    
    url = f"https://datacollective.mozillafoundation.org/api/datasets/{DATASET_ID}/download/{download_token}"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    if max_bytes:
        headers["Range"] = f"bytes=0-{max_bytes-1}"
        print(f"      Requesting {max_bytes / 1024 / 1024:.1f} MB")
    
    request = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get('Content-Length')
            content_type = response.headers.get('Content-Type')
            print(f"      Content-Type: {content_type}")
            print(f"      Content-Length: {content_length}")
            
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"\r      Downloaded: {downloaded/1024/1024:.1f} MB", end="", flush=True)
                    
                    if max_bytes and downloaded >= max_bytes:
                        break
            
            print(f"\n      ✓ Saved to {output_path}")
            return True
            
    except urllib.error.HTTPError as e:
        print(f"\n      ✗ HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"\n      ✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("MOZILLA DATA COLLECTIVE API DOWNLOAD")
    print("=" * 60)
    print(f"Dataset ID: {DATASET_ID}")
    print(f"API Key: {API_KEY[:20]}...")
    
    output_dir = Path("./mozilla_cv_data")
    output_dir.mkdir(exist_ok=True)
    
    # Test connectivity first
    test_api_connection()
    
    # Step 1: Create session
    session = create_download_session()
    
    if session is None:
        print("\n❌ Failed to create download session.")
        print("\nPossible issues:")
        print("  1. Dataset ID may be incorrect")
        print("  2. API key may need different permissions")
        print("  3. API endpoint structure may differ")
        print("\nTry visiting the Mozilla Data Collective website directly")
        print("to verify the correct download process.")
        return False
    
    # Extract token from response
    if isinstance(session, str):
        download_token = session
    elif isinstance(session, dict):
        # Try common key names
        for key in ["download_token", "token", "downloadToken", "id", "url"]:
            if key in session:
                download_token = session[key]
                break
        else:
            print(f"Session data: {json.dumps(session, indent=2)}")
            download_token = input("Enter download token from response: ")
    else:
        download_token = str(session)
    
    print(f"      Token: {str(download_token)[:50]}...")
    
    # Step 2: Download (1GB sample)
    output_path = output_dir / "common_voice_sample.tar.gz"
    max_bytes = 1 * 1024 * 1024 * 1024  # 1 GB
    
    success = download_with_token(download_token, str(output_path), max_bytes)
    
    if success:
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
