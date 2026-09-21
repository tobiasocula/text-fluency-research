#!/usr/bin/env bash
set -uo pipefail

# Run from the project root, since the Python script uses Path.cwd()
cd "$(dirname "$0")/.."
source venv/bin/activate

SCRIPT="new_pipeline/full_run_both_new.py"

# name=path pairs
FONTS=(
  "times=times-new-roman/times.ttf"
  "cormorant=Cormorant/Cormorant-VariableFont_wght.ttf"
  "chunkfive=chunkfive/ChunkFive-Regular.otf"
  "arial=arial/ARIAL.TTF"
)

# Optional: pass font names as arguments to run only those, e.g.
#   ./run_all_fonts.sh cormorant arial
if [ "$#" -gt 0 ]; then
  SELECTED=("$@")
else
  SELECTED=()
  for entry in "${FONTS[@]}"; do SELECTED+=("${entry%%=*}"); done
fi

failed=()
for want in "${SELECTED[@]}"; do
  found=0
  for entry in "${FONTS[@]}"; do
    name="${entry%%=*}"
    path="${entry#*=}"
    if [ "$name" = "$want" ]; then
      found=1
      echo "=== Running font: $name ($path) ==="
      python3 "$SCRIPT" --font "$path" --name "$name" || failed+=("$name")
    fi
  done
  [ "$found" -eq 0 ] && { echo "Unknown font name: $want"; failed+=("$want"); }
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "Failed: ${failed[*]}"
  exit 1
fi

#python3 new_vertical/run.py

echo "All done."

# ./new_pipeline/run.sh