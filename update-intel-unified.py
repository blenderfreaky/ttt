#!/usr/bin/env python3
"""
Update script for Intel unified runtime and memory framework packages

This script checks for new releases and updates the version/hash information
in the corresponding nix files.
"""

import json
import re
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional, Tuple

# Configuration for unified packages
UNIFIED_CONFIG = {
    "unified-runtime": {
        "owner": "oneapi-src",
        "repo": "unified-runtime",
        "nix_file": "nix/intel/unified-runtime.nix",
        "use_tags": True,  # Use latest tag, not commit
        "version_pattern": r'version = "([^"]+)";',
        "rev_pattern": r'rev = "([^"]+)";',
        "hash_pattern": r'sha256 = "([^"]+)";',
    },
    "unified-memory-framework": {
        "owner": "oneapi-src", 
        "repo": "unified-memory-framework",
        "nix_file": "nix/intel/unified-memory-framework.nix",
        "use_tags": True,
        "version_pattern": r'version = "([^"]+)";',
        "tag_pattern": r'tag = "([^"]+)";',
        "hash_pattern": r'sha256 = "([^"]+)";',
    },
}

def fetch_latest_release(owner: str, repo: str) -> Optional[Dict]:
    """Fetch the latest release information from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"Error fetching latest release for {owner}/{repo}: {e}")
    return None

def fetch_latest_tags(owner: str, repo: str, limit: int = 10) -> Optional[list]:
    """Fetch the latest tags from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/tags?per_page={limit}"
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"Error fetching tags for {owner}/{repo}: {e}")
    return None

def calculate_nix_hash_for_tag(owner: str, repo: str, tag: str) -> Optional[str]:
    """Calculate nix hash for a specific tag."""
    try:
        url = f"https://github.com/{owner}/{repo}"
        result = subprocess.run([
            "nix-prefetch-git", "--url", url, "--rev", tag, "--quiet"
        ], capture_output=True, text=True, check=True)
        
        data = json.loads(result.stdout)
        return data.get("sha256")
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error calculating hash for {owner}/{repo}@{tag}: {e}")
        return None

def extract_current_version(content: str, pattern: str) -> Optional[str]:
    """Extract current version from nix file."""
    match = re.search(pattern, content)
    return match.group(1) if match else None

def version_compare(v1: str, v2: str) -> int:
    """Simple version comparison. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal."""
    def version_tuple(v):
        # Remove 'v' prefix if present and split on dots
        clean_v = v.lstrip('v')
        try:
            return tuple(map(int, clean_v.split('.')))
        except ValueError:
            # Fallback for non-standard version formats
            return (0,)
    
    v1_tuple = version_tuple(v1)
    v2_tuple = version_tuple(v2)
    
    if v1_tuple > v2_tuple:
        return 1
    elif v1_tuple < v2_tuple:
        return -1
    else:
        return 0

def update_unified_runtime(config: Dict) -> Optional[Tuple[str, str, str]]:
    """Update unified-runtime specific logic."""
    # For unified-runtime, we might use commits instead of tags
    # Check the current nix file to see what pattern it uses
    nix_file = Path(config["nix_file"])
    if not nix_file.exists():
        print(f"Error: {nix_file} not found")
        return None
        
    content = nix_file.read_text()
    
    # Check if using tag or rev
    if "tag =" in content:
        # Using tags
        latest_release = fetch_latest_release(config["owner"], config["repo"])
        if not latest_release:
            return None
            
        tag_name = latest_release["tag_name"]
        version = tag_name.lstrip('v')  # Remove 'v' prefix
        
        # Calculate hash
        new_hash = calculate_nix_hash_for_tag(config["owner"], config["repo"], tag_name)
        if not new_hash:
            return None
            
        return version, tag_name, new_hash
    else:
        # Using rev - check for latest commit
        # This is more complex and might need specific logic
        print(f"  {config['repo']} uses rev, skipping automatic update")
        return None

def update_nix_file(nix_file: Path, config: Dict, new_version: str, new_ref: str, new_hash: str) -> None:
    """Update the nix file with new version, reference, and hash."""
    content = nix_file.read_text()
    
    # Update version
    if "version_pattern" in config:
        content = re.sub(config["version_pattern"], 
                        f'version = "{new_version}";', content)
    
    # Update tag or rev
    if "tag_pattern" in config:
        content = re.sub(config["tag_pattern"], 
                        f'tag = "{new_ref}";', content)
    elif "rev_pattern" in config:
        content = re.sub(config["rev_pattern"], 
                        f'rev = "{new_ref}";', content)
    
    # Update hash
    content = re.sub(config["hash_pattern"], 
                    f'sha256 = "{new_hash}";', content)
    
    nix_file.write_text(content)

def main():
    """Main update routine."""
    for pkg_name, config in UNIFIED_CONFIG.items():
        print(f"Checking {pkg_name}...")
        
        nix_file = Path(config["nix_file"])
        if not nix_file.exists():
            print(f"  Error: {nix_file} not found")
            continue
            
        content = nix_file.read_text()
        current_version = extract_current_version(content, config["version_pattern"])
        
        if not current_version:
            print(f"  Could not extract current version from {nix_file}")
            continue
            
        print(f"  Current version: {current_version}")
        
        if config["use_tags"]:
            # Check latest release
            latest_release = fetch_latest_release(config["owner"], config["repo"])
            if not latest_release:
                continue
                
            latest_version = latest_release["tag_name"].lstrip('v')
            
            if version_compare(latest_version, current_version) > 0:
                print(f"  New version available: {latest_version}")
                
                # Calculate new hash
                new_hash = calculate_nix_hash_for_tag(
                    config["owner"], config["repo"], latest_release["tag_name"]
                )
                
                if new_hash:
                    print(f"  New hash: {new_hash}")
                    update_nix_file(nix_file, config, latest_version, 
                                   latest_release["tag_name"], new_hash)
                    print(f"  Updated {nix_file}")
                else:
                    print(f"  Could not calculate hash")
            else:
                print(f"  Already up to date")
        else:
            # Handle commit-based updates
            result = update_unified_runtime(config)
            if result:
                version, ref, new_hash = result
                update_nix_file(nix_file, config, version, ref, new_hash)
                print(f"  Updated {nix_file}")

if __name__ == "__main__":
    main()