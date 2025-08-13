#!/bin/bash
"""
Master update script for all Intel dependencies and packages

This script runs all update scripts and provides a summary of changes.
"""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔄 Updating Intel LLVM Dependencies..."
echo "======================================"

# Check for required tools
echo "Checking for required tools..."
for tool in python3 nix-prefetch-git git; do
    if ! command -v "$tool" &> /dev/null; then
        echo "❌ Error: $tool is not installed or not in PATH"
        exit 1
    fi
done
echo "✅ All required tools found"

echo
echo "📦 Updating Intel LLVM dependencies (deps.nix)..."
if python3 "$SCRIPT_DIR/update-intel-deps.py"; then
    echo "✅ Intel LLVM dependencies updated"
else
    echo "❌ Failed to update Intel LLVM dependencies"
fi

echo
echo "🔧 Updating unified runtime and memory framework..."
if python3 "$SCRIPT_DIR/update-intel-unified.py"; then
    echo "✅ Unified packages updated"
else
    echo "❌ Failed to update unified packages"
fi

echo
echo "📊 Summary of changes:"
echo "======================"

# Show git diff if we're in a git repo
if git rev-parse --git-dir &> /dev/null; then
    if git diff --quiet; then
        echo "No changes detected."
    else
        echo "Changes detected:"
        git diff --name-only | while read -r file; do
            echo "  📝 $file"
        done
        
        echo
        echo "💡 You may want to review the changes with:"
        echo "   git diff"
        echo
        echo "💡 To test the changes, you can build the packages:"
        echo "   nix-build -A intel-llvm"
        echo "   nix-build -A unified-runtime"  
        echo "   nix-build -A unified-memory-framework"
    fi
else
    echo "Not in a git repository, cannot show diff."
fi

echo
echo "✨ Update complete!"