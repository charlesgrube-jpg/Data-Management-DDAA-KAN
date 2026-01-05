"""
RVC URL Resolver

Uses HuggingFace API to find correct file paths for RVC models in a repo.
Specifically targets the 'QuickWick/Music-AI-Voices' repo which has complex naming.
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

try:
    from pipeline.utils.huggingface_hub import HFClient
except ImportError:
    print("pipeline.utils not found.")
    sys.exit(1)

TARGET_REPO = "QuickWick/Music-AI-Voices"

def find_best_match(files: list[str], query: str) -> str | None:
    """Find zip file matching query."""
    query = query.lower()
    
    # 1. Exact match in filename
    matches = [f for f in files if f.endswith('.zip') and query in f.lower()]
    
    if not matches:
        return None
        
    # Prefer shortest match (least noise)
    matches.sort(key=len)
    return matches[0]

def main():
    client = HFClient()
    print(f"resolving URLs for repo: {TARGET_REPO}...")
    
    try:
        files = client.list_files(TARGET_REPO)
        print(f"Found {len(files)} files in repo.")
    except Exception as e:
        print(f"Failed to list repo: {e}")
        return

    # Models to fix/find (Gender Balance Expansion)
    targets = [
        "Ariana", "Billie", "Adele", "Rihanna", "Katy", "Shakira", "Beyonce", "LadyGaga"
    ]
    
    print("\nResolved URLs:")
    print("-" * 50)
    
    for target in targets:
        match = find_best_match(files, target)
        if match:
            # Construct HF URL
            # https://huggingface.co/QuickWick/Music-AI-Voices/resolve/main/{path}
            # Need to handle special chars? requests will quote them, but we need the raw string for the registry usually
            url = f"https://huggingface.co/{TARGET_REPO}/resolve/main/{match}"
            print(f"{target}: {url}")
        else:
            print(f"{target}: [Not Found]")

if __name__ == "__main__":
    main()
