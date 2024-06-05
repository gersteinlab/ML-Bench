#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: bash combine_files.sh OUTPUT_FILE PATH_TO_FILES [-e EXCLUDE_PATTERNS]"
  exit 1
fi

OUTPUT_FILE="$1"
BASE_DIR="$2"
> $OUTPUT_FILE
PATH=$(brew --prefix)/opt/findutils/libexec/gnubin:$PATH

# Check if the -e flag is given
if [ "$3" == "-e" ]; then
  EXCLUDE_FILE=$4
else
  # Use .gitignore as default exclude file
  EXCLUDE_FILE="$BASE_DIR/.gitignore"
fi

# Check if exclude file exists
if [ -f "$EXCLUDE_FILE" ]; then
  # Read patterns from exclude file and escape them properly
  EXCLUDE_PATTERNS=$(grep -v '^$\|^\s*\#' $EXCLUDE_FILE | sed 's/\./\\./g' | sed 's/\*/\.\*/g' | paste -sd '|' -)

  find "$BASE_DIR" -type f ! -path "$BASE_DIR/.git/*" -regextype posix-egrep ! -regex "$BASE_DIR/($EXCLUDE_PATTERNS).*" -print0 | 
  while read -d '' -r file; do
    if grep -q "\\x00" "$file"; then
      echo "$file is a binary file."
      continue
    fi
    grep -q "\\x00" "$file"
    echo "=== $file ===" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo >> "$OUTPUT_FILE"
  done

else
  find "$BASE_DIR" -type f ! -path "$BASE_DIR/.git/*" -print0 | 
  while read -d '' -r file; do
    if [[ $(grep -q "\\x00" "$file") ]]; then
      continue
    fi
    echo "=== $file ===" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo >> "$OUTPUT_FILE"
  done
fi
