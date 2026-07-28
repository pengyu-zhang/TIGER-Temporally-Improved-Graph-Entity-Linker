#!/usr/bin/env bash
# Download Graph-TempEL and build the processed layout under
# data/processed/graph_tempel.
#
# Primary source : Zenodo record 10977757 (official, CC-BY-4.0)
# Fallback mirror: GitHub Release assets of this repository
#
#   bash scripts/prepare_data.sh                # years 2019-2022 (paper main table)
#   YEARS=2013,...,2022 bash scripts/prepare_data.sh
#   MIRROR_ONLY=1 bash scripts/prepare_data.sh  # skip Zenodo, use the mirror
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
YEARS="${YEARS:-2019,2020,2021,2022}"
RAW="${RAW_DIR:-data/raw/graph_tempel}"
OUT="${OUT_DIR:-data/processed/graph_tempel}"

ZENODO_BASE="https://zenodo.org/records/10977757/files"
MIRROR_BASE="${MIRROR_BASE:-https://github.com/pengyu-zhang/TIGER-Temporally-Improved-Graph-Entity-Linker/releases/download/v1.0}"

# component file -> md5 (from data/md5sums.txt, kept in sync)
COMPONENTS=(
  "01_documents_no_duplicates.7z"
  "02_new_entities_1764.7z"
  "02_continual_entities_1764.7z"
  "03_relationship.7z"
  "04_feature.7z"
  "05_knn_relation.zip"
)

mkdir -p "$RAW"

fetch() { # fetch <url> <dest>
  echo "[data] downloading $1"
  curl -fSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

verify_md5() { # verify_md5 <file>
  local name expected actual
  name=$(basename "$1")
  expected=$(grep " ${name}\$" data/md5sums.txt | cut -d' ' -f1 || true)
  if [ -z "$expected" ]; then
    echo "[data] WARNING: no md5 recorded for $name, skipping check"
    return 0
  fi
  actual=$("$PYTHON" -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1")
  if [ "$expected" != "$actual" ]; then
    echo "[data] md5 mismatch for $name: expected $expected got $actual" >&2
    return 1
  fi
  echo "[data] md5 OK $name"
}

for comp in "${COMPONENTS[@]}"; do
  dest="$RAW/$comp"
  if [ -f "$dest" ] && verify_md5 "$dest"; then
    echo "[data] cached $comp"
    continue
  fi
  ok=0
  if [ "${MIRROR_ONLY:-0}" != "1" ]; then
    if fetch "$ZENODO_BASE/$comp?download=1" "$dest" && verify_md5 "$dest"; then ok=1; fi
  fi
  if [ "$ok" != "1" ]; then
    echo "[data] falling back to mirror for $comp"
    fetch "$MIRROR_BASE/$comp" "$dest"
    verify_md5 "$dest"
  fi
done

echo "[data] extracting ..."
"$PYTHON" - "$RAW" <<'EOF'
import os, sys, zipfile
import py7zr
raw = sys.argv[1]
for name in os.listdir(raw):
    path = os.path.join(raw, name)
    stem = name.rsplit(".", 1)[0]
    marker = os.path.join(raw, "extracted", stem, ".done")
    if os.path.exists(marker):
        continue
    dest = os.path.join(raw, "extracted", stem)
    os.makedirs(dest, exist_ok=True)
    if name.endswith(".7z"):
        with py7zr.SevenZipFile(path) as arc:
            arc.extractall(dest)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(path) as arc:
            arc.extractall(dest)
    else:
        continue
    open(marker, "w").close()
    print(f"[data] extracted {name}")
EOF

PYTHONPATH=src "$PYTHON" -m tiger.prepare --source extracted \
  --input "$RAW/extracted" --output "$OUT" --years "$YEARS"

echo "[data] processed data ready under $OUT"
