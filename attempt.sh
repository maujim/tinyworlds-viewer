#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/new-attempt-dir" >&2
  exit 2
fi

DEST="$1"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"

if [[ "$DEST" == "$SRC" ]]; then
  echo "Destination must be different from the scaffold directory." >&2
  exit 1
fi

# Refuse to write into a non-empty directory, counting hidden files too.
if find "$DEST" -mindepth 1 -maxdepth 1 | read -r _; then
  echo "Destination exists and is not empty: $DEST" >&2
  exit 1
fi

# Copy scaffold contents, including dotfiles, while respecting scaffold/.gitignore.
# Do not copy a git repository by default; each attempt should usually be its own clean workspace.
rsync -a \
  --exclude='.git/' \
  --exclude-from="$SRC/.gitignore" \
  "$SRC"/ "$DEST"/

cd "$DEST"
uv sync --dev

echo "Attempt created at: $DEST"
echo "Next steps:"
echo "  cd '$DEST'"
echo "  read CONTESTANT_TASK.md"
echo "  uv run streamlit run app.py"
echo "  ./test.sh"
