#!/usr/bin/env bash
# Frontend code quality script
# Run from the frontend/ directory or the project root.

set -e

FRONTEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$FRONTEND_DIR"

# Install dependencies if node_modules is missing
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

case "${1:-check}" in
  format)
    echo "Formatting frontend files with Prettier..."
    npx prettier --write .
    echo "Done."
    ;;
  check)
    echo "Checking frontend formatting with Prettier..."
    npx prettier --check .
    echo "All files are properly formatted."
    ;;
  *)
    echo "Usage: $0 [format|check]"
    echo "  format  - auto-format all frontend files"
    echo "  check   - verify formatting without making changes (default)"
    exit 1
    ;;
esac
