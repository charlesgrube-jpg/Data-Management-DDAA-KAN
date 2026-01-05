"""
Centralized Hugging Face Hub utilities.

Provides single-point-of-truth for all HF interactions:
- Authentication handling (token optional/env-based)
- Repo existence checks
- File listing
- Robust download with revision pinning
- Integrity checks (size, magic bytes)
- Clear error classes

Usage:
    from pipeline.utils.huggingface_hub import HFClient
    
    client = HFClient()
    if client.repo_exists("0x3e9/0x3e9_RVC_models"):
        result = client.download_file("0x3e9/0x3e9_RVC_models", "biden.zip")
"""

import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

# =============================================================================
# Error Classes
# =============================================================================

class HFError(Exception):
    """Base exception for HuggingFace operations."""
    pass

class HFRepoNotFoundError(HFError):
    """Repository does not exist or is private."""
    pass

class HFFileNotFoundError(HFError):
    """File not found in repository."""
    pass

class HFGatedRepoError(HFError):
    """Repository requires authentication/agreement."""
    pass

class HFIntegrityError(HFError):
    """Downloaded file failed integrity checks."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HFDownloadResult:
    """Result of a download operation."""
    success: bool
    local_path: Optional[Path] = None
    error: Optional[str] = None
    size_bytes: Optional[int] = None


# =============================================================================
# Client
# =============================================================================

class HFClient:
    """Unified HuggingFace Hub client with integrity checks."""
    
    # Minimum file size for valid model archives (10MB)
    MIN_MODEL_SIZE_BYTES = 10 * 1024 * 1024
    
    # ZIP magic bytes
    ZIP_MAGIC = b'PK\x03\x04'
    
    def __init__(self, token: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Args:
            token: HF API token (optional, reads from env if not provided)
            cache_dir: Local cache directory (default: ~/.cache/huggingface)
        """
        # Lazy import to avoid dependency issues
        try:
            from huggingface_hub import HfApi, HfFolder
            self._hf_available = True
            self.token = token or os.environ.get("HF_TOKEN") or HfFolder.get_token()
            self.api = HfApi(token=self.token)
        except ImportError:
            self._hf_available = False
            self.token = None
            self.api = None
            
        self.cache_dir = Path(cache_dir) if cache_dir else None
    
    def is_available(self) -> bool:
        """Check if huggingface_hub is installed."""
        return self._hf_available
    
    def repo_exists(self, repo_id: str) -> bool:
        """Check if repository exists and is accessible."""
        if not self._hf_available:
            return False
        try:
            self.api.repo_info(repo_id)
            return True
        except Exception:
            return False
    
    def list_files(self, repo_id: str, revision: str = "main") -> List[str]:
        """List all files in a repository."""
        if not self._hf_available:
            raise HFError("huggingface_hub not installed")
            
        try:
            from huggingface_hub import list_repo_files
            return list_repo_files(repo_id, revision=revision, token=self.token)
        except Exception as e:
            error_str = str(e).lower()
            if "404" in error_str:
                raise HFRepoNotFoundError(f"Repository not found: {repo_id}")
            elif "401" in error_str or "403" in error_str:
                raise HFGatedRepoError(f"Repository gated/private: {repo_id}")
            raise HFError(f"Cannot list files in {repo_id}: {e}")
    
    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: Optional[Path] = None,
        revision: str = "main",
        force: bool = False,
        check_integrity: bool = True
    ) -> HFDownloadResult:
        """
        Download a single file from a HF repository.
        
        Args:
            repo_id: e.g., "0x3e9/0x3e9_RVC_models"
            filename: e.g., "biden.zip"
            local_dir: Where to save (default: cache)
            revision: Git revision (default: "main")
            force: Re-download even if cached
            check_integrity: Verify file size and magic bytes
        """
        if not self._hf_available:
            return HFDownloadResult(success=False, error="huggingface_hub not installed")
            
        try:
            from huggingface_hub import hf_hub_download
            
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=self.token,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                local_dir=str(local_dir) if local_dir else None,
                force_download=force
            )
            
            path = Path(local_path)
            size = path.stat().st_size if path.exists() else 0
            
            # Integrity checks
            if check_integrity:
                # Size check
                if size < self.MIN_MODEL_SIZE_BYTES:
                    return HFDownloadResult(
                        success=False,
                        local_path=path,
                        size_bytes=size,
                        error=f"File too small ({size/1024/1024:.1f}MB < 10MB min)"
                    )
                
                # Magic bytes check for ZIP
                if filename.endswith('.zip'):
                    with open(path, 'rb') as f:
                        magic = f.read(4)
                    if magic != self.ZIP_MAGIC:
                        return HFDownloadResult(
                            success=False,
                            local_path=path,
                            size_bytes=size,
                            error=f"Invalid ZIP file (bad magic bytes: {magic!r})"
                        )
            
            return HFDownloadResult(
                success=True,
                local_path=path,
                size_bytes=size
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                return HFDownloadResult(success=False, error=f"File not found: {filename}")
            elif "401" in error_str or "403" in error_str or "gated" in error_str:
                return HFDownloadResult(success=False, error="Gated/private repository")
            else:
                return HFDownloadResult(success=False, error=str(e)[:100])
    
    def resolve_file_url(
        self,
        repo_id: str,
        pattern: str,
        revision: str = "main"
    ) -> Optional[str]:
        """
        Find a file in repo matching pattern and return download URL.
        
        Args:
            repo_id: HF repository ID
            pattern: Substring to match in filename (case-insensitive)
            revision: Git revision
            
        Returns:
            Full URL for matching file, or None if not found
        """
        if not self._hf_available:
            return None
            
        try:
            from huggingface_hub import hf_hub_url
            
            files = self.list_files(repo_id, revision)
            matches = [f for f in files if pattern.lower() in f.lower()]
            
            if matches:
                # Prefer .zip files
                zip_matches = [f for f in matches if f.endswith('.zip')]
                selected = zip_matches[0] if zip_matches else matches[0]
                return hf_hub_url(repo_id, selected, revision=revision)
            return None
        except Exception:
            return None
