#!/usr/bin/env python3
"""
Update script for Intel LLVM dependencies in nix/intel/llvm/deps.nix

This script fetches version information from CMake files in Intel repositories
using the GitHub API, avoiding the need to clone entire repositories.
"""

import json
import re
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional, Tuple

# Configuration for each dependency
DEPS_CONFIG = {
    "vc-intrinsics": {
        "owner": "intel",
        "repo": "vc-intrinsics", 
        "cmake_path": "llvm/lib/SYCLLowerIR/CMakeLists.txt",
        "cmake_repo": "intel/intel-llvm",
        "pattern": r"set\(LLVMGenXIntrinsics_GIT_TAG\s+([a-f0-9]+)\)",
    },
    "spirv-headers": {
        "owner": "KhronosGroup",
        "repo": "SPIRV-Headers",
        "cmake_path": "llvm-spirv/spirv-headers-tag.conf", 
        "cmake_repo": "intel/intel-llvm",
        "pattern": r"^([a-f0-9]+)$",
    },
    "oneapi-ck": {
        "owner": "uxlfoundation",
        "repo": "oneapi-construction-kit",
        "cmake_path": "llvm/lib/SYCLNativeCPUUtils/CMakeLists.txt",
        "cmake_repo": "intel/intel-llvm", 
        "pattern": r"set\(NATIVECPU_OCK_GIT_TAG\s+([a-f0-9]+)\)",
    },
    "opencl-headers": {
        "owner": "KhronosGroup",
        "repo": "OpenCL-Headers",
        "cmake_path": "opencl/CMakeLists.txt",
        "cmake_repo": "intel/intel-llvm",
        "pattern": r"set\(OPENCL_HEADERS_COMMIT\s+([a-f0-9]+)\)",
    },
    "opencl-icd-loader": {
        "owner": "KhronosGroup", 
        "repo": "OpenCL-ICD-Loader",
        "cmake_path": "opencl/CMakeLists.txt",
        "cmake_repo": "intel/intel-llvm",
        "pattern": r"set\(OPENCL_ICD_LOADER_COMMIT\s+([a-f0-9]+)\)",
    },
    "emhash": {
        "owner": "ktprime",
        "repo": "emhash",
        "cmake_path": "sycl/cmake/modules/FetchEmhash.cmake",
        "cmake_repo": "intel/intel-llvm",
        "pattern": r"GIT_TAG\s+([a-f0-9]+)",
    },
    "parallel-hashmap": {
        "owner": "greg7mdp",
        "repo": "parallel-hashmap", 
        "cmake_path": "xptifw/src/CMakeLists.txt",
        "cmake_repo": "intel/intel-llvm",
        "pattern": r"GIT_TAG\s+([a-f0-9]+)",
    },
}

def fetch_github_content(repo: str, path: str, ref: str = "main") -> Optional[str]:
    """Fetch file content from GitHub using the API."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            if data.get("encoding") == "base64":
                import base64
                return base64.b64decode(data["content"]).decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"Error fetching {url}: {e}")
    return None

def extract_version_from_cmake(content: str, pattern: str) -> Optional[str]:
    """Extract version/commit hash from CMake file content."""
    match = re.search(pattern, content, re.MULTILINE)
    return match.group(1) if match else None

def get_latest_commit_sha(owner: str, repo: str) -> Optional[str]:
    """Get the latest commit SHA from the default branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/HEAD"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            return data["sha"]
    except urllib.error.HTTPError as e:
        print(f"Error fetching latest commit for {owner}/{repo}: {e}")
    return None

def calculate_nix_hash(url: str) -> Optional[str]:
    """Calculate nix hash for a fetchFromGitHub source."""
    try:
        # Use nix-prefetch-git to get the hash
        result = subprocess.run([
            "nix-prefetch-git", "--url", url, "--quiet"
        ], capture_output=True, text=True, check=True)
        
        data = json.loads(result.stdout)
        return data.get("sha256")
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error calculating hash for {url}: {e}")
        return None

def update_deps_nix(deps_file: Path, updates: Dict[str, Tuple[str, str]]) -> None:
    """Update the deps.nix file with new revisions and hashes."""
    content = deps_file.read_text()
    
    for dep_name, (new_rev, new_hash) in updates.items():
        # Update revision
        rev_pattern = rf'({dep_name}.*?rev = ")([a-f0-9]+)(";)'
        content = re.sub(rev_pattern, rf'\g<1>{new_rev}\g<3>', content, flags=re.DOTALL)
        
        # Update hash
        hash_pattern = rf'({dep_name}.*?sha256 = ")([^"]+)(";)'
        content = re.sub(hash_pattern, rf'\g<1>{new_hash}\g<3>', content, flags=re.DOTALL)
    
    deps_file.write_text(content)

def main():
    """Main update routine."""
    deps_file = Path("nix/intel/llvm/deps.nix")
    if not deps_file.exists():
        print(f"Error: {deps_file} not found")
        sys.exit(1)
    
    updates = {}
    
    for dep_name, config in DEPS_CONFIG.items():
        print(f"Checking {dep_name}...")
        
        # Fetch CMake file from intel-llvm repo
        cmake_content = fetch_github_content(
            config["cmake_repo"], 
            config["cmake_path"]
        )
        
        if not cmake_content:
            print(f"  Could not fetch CMake file for {dep_name}")
            continue
            
        # Extract version from CMake
        version = extract_version_from_cmake(cmake_content, config["pattern"])
        if not version:
            print(f"  Could not extract version from CMake for {dep_name}")
            continue
            
        print(f"  Found version: {version}")
        
        # Calculate new hash
        repo_url = f"https://github.com/{config['owner']}/{config['repo']}"
        new_hash = calculate_nix_hash(repo_url)
        
        if not new_hash:
            print(f"  Could not calculate hash for {dep_name}")
            continue
            
        updates[dep_name] = (version, new_hash)
        print(f"  New hash: {new_hash}")
    
    if updates:
        print(f"\nUpdating {deps_file}...")
        update_deps_nix(deps_file, updates)
        print("Done!")
    else:
        print("No updates found.")

if __name__ == "__main__":
    main()