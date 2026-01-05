"""
.pth File Security Scanning

Scans PyTorch .pth files for potential malicious pickle payloads using Fickling.
Must be called BEFORE loading any untrusted model weights.

Usage:
    from pipeline.utils.pth_security import scan_pth_file, is_safe_to_load
    
    result = scan_pth_file("models/rvc/Biden/biden.pth")
    if result.is_safe:
        # Proceed with loading
    else:
        print(f"BLOCKED: {result.issues}")

Security Model:
    - If Fickling is installed: Use it for static analysis
    - If Fickling is missing: FAIL SAFE (reject) unless ALLOW_UNSAFE_PTH=1 is set
    - Always perform basic sanity checks (size, extension)
"""

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# Configuration
# =============================================================================

# Environment variable to allow loading without Fickling (DANGEROUS)
ALLOW_UNSAFE_ENV = "ALLOW_UNSAFE_PTH"

# Expected size range for RVC models (MB)
MIN_MODEL_SIZE_MB = 1
MAX_MODEL_SIZE_MB = 1000  # Increased to allow base checkpoints

# Valid extensions
VALID_EXTENSIONS = {'.pth', '.pt', '.pkl', '.pickle', '.bin'}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ScanResult:
    """Result of a .pth security scan."""
    file_path: Path
    is_safe: bool
    issues: List[str] = field(default_factory=list)
    scan_method: str = "none"  # "fickling", "basic", "skipped"
    error: Optional[str] = None
    size_mb: float = 0.0


# =============================================================================
# Scanning Functions
# =============================================================================

import zipfile
import tempfile
import sys
import os

def _check_fickling_available() -> bool:
    """Check if Fickling is installed and compatible."""
    
    # Fickling 0.1.6 is incompatible with Python 3.14+ (AST changes)
    if sys.version_info >= (3, 14):
        print(f"[SECURITY] WARNING: Fickling is incompatible with Python {sys.version}. Skipping deep scan.")
        return False

    try:
        # Using sys.executable -m to ensure we use the same python env and handle path issues
        result = subprocess.run(
            [sys.executable, "-m", "fickling", "--help"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _scan_single_file(path: Path) -> tuple[bool, List[str]]:
    """Helper to run Fickling on a single unzipped file."""
    issues = []
    try:
        result = subprocess.run(
            [sys.executable, "-m", "fickling", "--check-safety", str(path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            output = result.stdout + result.stderr
            if "UNSAFE" in output.upper() or "MALICIOUS" in output.upper():
                issues.append(f"Fickling UNSAFE: {output[:200].strip()}")
            else:
                # Some non-zero exits are just parsing errors on non-pickles, which is fine
                # But for safety we log it.
                if "No pickle files detected" not in output:
                     # Capture full output if stderr is empty/unhelpful
                     msg = output[:500].strip() if output.strip() else f"Silent Failure (RC={result.returncode})"
                     issues.append(f"Fickling error: {msg}")
        return len(issues) == 0, issues
    except Exception as e:
        return False, [f"Scan execution error: {e}"]


def _run_fickling_scan(file_path: Path) -> tuple[bool, List[str]]:
    """
    Run Fickling static analysis on a file.
    Handles both direct .pth (pickle) and .pth (zip archive) formats.
    """
    issues = []
    
    # Check if it's a ZIP (most RVC models are zips containing data.pkl)
    if zipfile.is_zipfile(file_path):
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Look for data.pkl or valid extensions
                targets = [n for n in zf.namelist() if n.endswith(('.pkl', '.pth', '.bin'))]
                
                if not targets:
                    issues.append("Zip archive contains no recognizable model files (.pkl/.pth)")
                    return False, issues
                
                # Scan each target
                with tempfile.TemporaryDirectory() as temp_dir:
                    for target in targets:
                        # Extract to temp
                        extracted_path = Path(temp_dir) / Path(target).name
                        with zf.open(target) as source, open(extracted_path, "wb") as dest:
                            dest.write(source.read())
                        
                        # Scan extracted file
                        is_safe, sub_issues = _scan_single_file(extracted_path)
                        if not is_safe:
                            issues.extend([f"[{target}] {i}" for i in sub_issues])

        except Exception as e:
            issues.append(f"Zip extraction error: {e}")
            return False, issues
            
    else:
        # Scan directly
        is_safe, direct_issues = _scan_single_file(file_path)
        issues.extend(direct_issues)
                
    return len(issues) == 0, issues
        
    return len(issues) == 0, issues


def scan_pth_file(file_path: str | Path) -> ScanResult:
    """
    Scan a .pth file for potential malicious payloads.
    
    Uses layered approach:
    1. Basic checks (existence, extension, size)
    2. Fickling static analysis (if available)
    3. Fail-safe if Fickling missing (unless overridden)
    
    Args:
        file_path: Path to .pth file
        
    Returns:
        ScanResult with is_safe flag and any issues
    """
    path = Path(file_path)
    result = ScanResult(file_path=path, is_safe=False, scan_method="none")
    
    # =========================================================================
    # Layer 1: Basic Checks
    # =========================================================================
    
    # Existence
    if not path.exists():
        result.issues.append("File not found")
        return result
    
    # Extension
    if path.suffix.lower() not in VALID_EXTENSIONS:
        result.issues.append(f"Unexpected extension: {path.suffix}")
    
    # Size sanity
    size_bytes = path.stat().st_size
    result.size_mb = size_bytes / (1024 * 1024)
    
    if result.size_mb < MIN_MODEL_SIZE_MB:
        result.issues.append(f"Suspiciously small ({result.size_mb:.1f} MB < {MIN_MODEL_SIZE_MB} MB)")
    if result.size_mb > MAX_MODEL_SIZE_MB:
        result.issues.append(f"Unusually large ({result.size_mb:.1f} MB > {MAX_MODEL_SIZE_MB} MB)")
    
    # If basic checks failed, don't proceed
    if result.issues:
        result.scan_method = "basic"
        return result
    
    # =========================================================================
    # Layer 2: Fickling Scan
    # =========================================================================
    
    if _check_fickling_available():
        is_safe, fickling_issues = _run_fickling_scan(path)
        result.issues.extend(fickling_issues)
        result.scan_method = "fickling"
        result.is_safe = is_safe
        return result
    
    # =========================================================================
    # Layer 3: Fail-Safe Fallback
    # =========================================================================
    
    # Fickling not available - check for override
    allow_unsafe = os.environ.get(ALLOW_UNSAFE_ENV, "0") == "1"
    
    if allow_unsafe:
        result.issues.append(f"WARNING: Fickling not installed, proceeding due to {ALLOW_UNSAFE_ENV}=1")
        result.scan_method = "skipped"
        result.is_safe = True  # User explicitly allowed
    else:
        result.issues.append(
            f"Fickling not installed. Cannot verify safety of {path.name}. "
            f"Install with 'pip install fickling' or set {ALLOW_UNSAFE_ENV}=1 to skip."
        )
        result.scan_method = "blocked"
        result.is_safe = False
    
    return result


def is_safe_to_load(file_path: str | Path) -> bool:
    """
    Quick check if a file is safe to load.
    
    Returns True only if scan passes.
    """
    return scan_pth_file(file_path).is_safe


def quarantine_file(file_path: Path, quarantine_dir: Path) -> Path:
    """
    Move suspicious file to quarantine directory.
    
    Args:
        file_path: File to quarantine
        quarantine_dir: Target quarantine directory
        
    Returns:
        New path of quarantined file
    """
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest = quarantine_dir / f"{file_path.stem}_QUARANTINED{file_path.suffix}"
    
    # Avoid overwriting
    counter = 1
    while dest.exists():
        dest = quarantine_dir / f"{file_path.stem}_QUARANTINED_{counter}{file_path.suffix}"
        counter += 1
    
    file_path.rename(dest)
    print(f"[SECURITY] Quarantined: {file_path} -> {dest}")
    return dest
