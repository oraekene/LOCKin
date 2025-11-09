#!/usr/bin/env bash
set -e
DATA_DIR=${DATA_DIR:-"./data"}
BACKUP_DIR="${DATA_DIR}/user_data_backups"
mkdir -p "$BACKUP_DIR"
timestamp=$(date -u +"%Y%m%dT%H%M%SZ")
manifest="${BACKUP_DIR}/backup_manifest_${timestamp}.json"
# ensure manifest exists as JSON array
if [ ! -f "$manifest" ]; then
  echo "[]" > "$manifest"
fi

for f in "${DATA_DIR}"/user_*.dat; do
  if [ -f "$f" ]; then
    cp "$f" "${f}.bak"
    sha=$(sha256sum "${f}.bak" | awk '{print $1}')
    jq --arg o "$f" --arg b "${f}.bak" --arg s "$sha" --arg t "$timestamp" \
       '. += [{"original":$o,"backup":$b,"sha256":$s,"timestamp":$t}]' "$manifest" > "${manifest}.tmp" && mv "${manifest}.tmp" "$manifest"
  fi
done

echo "Backups created. Manifest: $manifest"
