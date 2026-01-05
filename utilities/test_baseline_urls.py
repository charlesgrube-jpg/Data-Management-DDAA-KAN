"""Quick baseline model URL test."""
import requests

BASELINE = {
    "TaylorSwift": "https://huggingface.co/bennetJL/TaylorSwift/resolve/main/TaylorSwift2024.zip?download=true",
    "EdSheeran": "https://huggingface.co/SUP3RMASS1VE/Ed-Sheeran/resolve/main/Ed%20Sheeran.zip?download=true",
    "KanyeWest": "https://huggingface.co/TheRealheavy/KanyeWestGraduationEra/resolve/main/KanyeWestGraduation.zip?download=true"
}

for name, url in BASELINE.items():
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        ct = r.headers.get("Content-Type", "?")
        cl = int(r.headers.get("Content-Length", 0)) / 1024 / 1024
        print(f"{name}: Status={r.status_code}, Type={ct[:30]}, Size={cl:.1f}MB")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
