"""Test corrected expansion model URLs."""
import requests

CORRECTED_EXPANSION = {
    "Biden": "https://huggingface.co/0x3e9/0x3e9_RVC_models/resolve/main/biden.zip",
    "Trump": "https://huggingface.co/0x3e9/0x3e9_RVC_models/resolve/main/trump.zip",
    "Obama": "https://huggingface.co/0x3e9/0x3e9_RVC_models/resolve/main/obama.zip",
}

print("Testing corrected expansion URLs...")
for name, url in CORRECTED_EXPANSION.items():
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        ct = r.headers.get("Content-Type", "?")
        cl = int(r.headers.get("Content-Length", 0)) / 1024 / 1024
        status = "✅" if r.status_code == 200 and "zip" in ct else "❌"
        print(f"{status} {name}: Status={r.status_code}, Type={ct[:30]}, Size={cl:.1f}MB")
    except Exception as e:
        print(f"❌ {name}: ERROR - {e}")
