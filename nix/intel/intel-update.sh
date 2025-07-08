#!/usr/bin/env nix-shell
#!nix-shell -i bash -p ripgrep
set -euo pipefail

echo "Updating Intel OneAPI Toolkit..."

# The URLs will look like this:
#   https://registrationcenter-download.intel.com/akdlm/IRC_NAS/bd1d0273-a931-4f7e-ab76-6a2a67d646c7/intel-oneapi-base-toolkit-2025.2.0.592_offline.sh
#                                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                           ^^^^^^^^^^^^ ^^^^^^
#                                                               UUID                                                           Version      See below
#
# Sometimes, the _offline.sh isn't linked, but it has the same UUID and version as the online version.
#
# The code changes with new releases.

BASEKIT_PATTERN='https://registrationcenter-download.intel.com/akdlm/IRC_NAS/([0-9a-z-]+)/intel-oneapi-base-toolkit-(\d+\.\d+\.\d+\.\d+)(_offline)?\.sh'
HPCKIT_PATTERN='https://registrationcenter-download.intel.com/akdlm/IRC_NAS/([0-9a-z-]+)/intel-oneapi-hpc-toolkit-(\d+\.\d+\.\d+\.\d+)(_offline)?\.sh'

BASEKIT_URL=$(curl -s 'https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline' \
  | rg -o $BASEKIT_PATTERN)
HPCKIT_URL=$(curl -s 'https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html?packages=hpc-toolkit&hpc-toolkit-os=linux&hpc-toolkit-lin=offline' \
  | rg -o $HPCKIT_PATTERN)

echo "Successfully got current URLs"

BASEKIT_UUID=$(echo "$BASEKIT_URL" | rg -N $BASEKIT_PATTERN --replace '$1')
BASEKIT_VERSION=$(echo "$BASEKIT_URL" | rg -N $BASEKIT_PATTERN --replace '$2')
BASEKIT_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/$BASEKIT_UUID/intel-oneapi-base-toolkit-${BASEKIT_VERSION}_offline.sh"

HPCKIT_UUID=$(echo "$HPCKIT_URL" | rg -N $HPCKIT_PATTERN --replace '$1')
HPCKIT_VERSION=$(echo "$HPCKIT_URL" | rg -N $HPCKIT_PATTERN --replace '$2')
HPCKIT_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/$HPCKIT_UUID/intel-oneapi-hpc-toolkit-${HPCKIT_VERSION}_offline.sh"

echo "BASEKIT_URL=$BASEKIT_URL"
echo "BASEKIT_UUID=$BASEKIT_UUID"
echo "BASEKIT_VERSION=$BASEKIT_VERSION"

echo "HPCKIT_URL=$HPCKIT_URL"
echo "HPCKIT_UUID=$HPCKIT_UUID"
echo "HPCKIT_VERSION=$HPCKIT_VERSION"

BASEKIT_SHA=$(nix-prefetch-url --type sha256 "$BASEKIT_URL")
BASEKIT_SRI_SHA=$(nix hash-to-sri "$BASEKIT_SHA")

HPCKIT_SHA=$(nix-prefetch-url --type sha256 "$HPCKIT_URL")
HPCKIT_SRI_SHA=$(nix hash-to-sri "$HPCKIT_SHA")

echo "BASEKIT_SHA=$BASEKIT_SRI_SHA"
echo "HPCKIT_SHA=$HPCKIT_SRI_SHA"

TARGET_FILE="./base.nix"

sd '(?<=^\s*version\s*=\s*)".*?"' "\"$BASEKIT_VERSION\"" $TARGET_FILE
sd '(?<=^\s*uuid\s*=\s*)".*?"' "\"$BASEKIT_UUID\"" $TARGET_FILE
sd '(?<=^\s*hash\s*=\s*)".*?"' "\"$BASEKIT_SRI_SHA\"" $TARGET_FILE

TARGET_FILE="./hpc.nix"

sd '(?<=^\s*version\s*=\s*)".*?"' "\"$HPCKIT_VERSION\"" $TARGET_FILE
sd '(?<=^\s*uuid\s*=\s*)".*?"' "\"$HPCKIT_UUID\"" $TARGET_FILE
sd '(?<=^\s*hash\s*=\s*)".*?"' "\"$HPCKIT_SRI_SHA\"" $TARGET_FILE
