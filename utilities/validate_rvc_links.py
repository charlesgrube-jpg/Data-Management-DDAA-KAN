"""
Validate all RVC model links in the registry.
HEAD requests only - no downloads.
"""
import requests
import yaml
from pathlib import Path

REGISTRY_PATH = Path("utilities/rvc_model_registry.yaml")

def validate_url(name: str, url: str) -> dict:
    """Check if URL is valid without downloading."""
    result = {
        "id": name,
        "url": url,
        "status": None,
        "content_type": None,
        "size_mb": None,
        "valid": False,
        "error": None
    }
    
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        result["status"] = r.status_code
        result["content_type"] = r.headers.get("Content-Type", "unknown")
        
        cl = r.headers.get("Content-Length")
        if cl:
            result["size_mb"] = round(int(cl) / (1024 * 1024), 1)
        
        # Validity checks
        if r.status_code == 404:
            result["error"] = "404 Not Found"
        elif r.status_code == 401 or r.status_code == 403:
            result["error"] = f"Gated/Private ({r.status_code})"
        elif r.status_code != 200:
            result["error"] = f"HTTP {r.status_code}"
        elif "text/html" in result["content_type"]:
            result["error"] = "HTML response (likely error page)"
        elif result["size_mb"] and result["size_mb"] < 1:
            result["error"] = f"Too small ({result['size_mb']} MB)"
        else:
            result["valid"] = True
            
    except requests.Timeout:
        result["error"] = "Timeout"
    except requests.RequestException as e:
        result["error"] = str(e)[:50]
    
    return result


def main():
    print("=" * 70)
    print("RVC Model Link Validation (HEAD Requests Only)")
    print("=" * 70)
    
    with open(REGISTRY_PATH, "r") as f:
        registry = yaml.safe_load(f)
    
    all_models = []
    
    # Baseline
    for model in registry.get("baseline", []):
        all_models.append(model)
    
    # Expansion
    for model in registry.get("expansion", []):
        all_models.append(model)
    
    print(f"\nChecking {len(all_models)} models...\n")
    
    valid_count = 0
    invalid_count = 0
    
    for model in all_models:
        result = validate_url(model["id"], model["url"])
        
        if result["valid"]:
            print(f"✅ {model['id']}: {result['status']} | {result['content_type'][:20]} | {result['size_mb']} MB")
            valid_count += 1
        else:
            print(f"❌ {model['id']}: {result['error']}")
            invalid_count += 1
    
    print("\n" + "=" * 70)
    print(f"Summary: {valid_count} valid, {invalid_count} invalid")
    print("=" * 70)


if __name__ == "__main__":
    main()
