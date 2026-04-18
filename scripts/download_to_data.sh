#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dataset_name> <url> [output_root=/data/$USER/datasets]"
  exit 1
fi

DATASET_NAME="$1"
URL="$2"
OUTPUT_ROOT="${3:-/data/$USER/datasets}"

TARGET_DIR="$OUTPUT_ROOT/$DATASET_NAME"
ARCHIVE_DIR="$TARGET_DIR/_archive"
mkdir -p "$TARGET_DIR" "$ARCHIVE_DIR"

FILENAME="$(basename "${URL%%\?*}")"
if [[ -z "$FILENAME" || "$FILENAME" == "/" ]]; then
  FILENAME="$DATASET_NAME.download"
fi
ARCHIVE_PATH="$ARCHIVE_DIR/$FILENAME"

echo "Downloading to: $ARCHIVE_PATH"
curl -L --fail --retry 3 --retry-delay 5 "$URL" -o "$ARCHIVE_PATH"

echo "Extracting if archive is recognized..."
case "$ARCHIVE_PATH" in
  *.zip)
    unzip -o "$ARCHIVE_PATH" -d "$TARGET_DIR"
    ;;
  *.tar)
    tar -xf "$ARCHIVE_PATH" -C "$TARGET_DIR"
    ;;
  *.tar.gz|*.tgz)
    tar -xzf "$ARCHIVE_PATH" -C "$TARGET_DIR"
    ;;
  *.tar.bz2)
    tar -xjf "$ARCHIVE_PATH" -C "$TARGET_DIR"
    ;;
  *.tar.xz)
    tar -xJf "$ARCHIVE_PATH" -C "$TARGET_DIR"
    ;;
  *)
    echo "No extractor rule for $ARCHIVE_PATH; file kept in _archive/."
    ;;
esac

echo "Done. Dataset path: $TARGET_DIR"
