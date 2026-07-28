#!/usr/bin/env bash
# Backup the OneGov SQLite DB (sessions, shared answers, cache).
#
# The DB lives inside the container and is lost on every redeploy unless
# DATA_DIR points at a mounted Azure Files volume. Until then, run this
# periodically (or before each deploy) to keep shared answer links alive.
#
# Local backup:   ./scripts/backup_db.sh
# To Azure blob:  AZ_STORAGE_ACCOUNT=... AZ_CONTAINER=backups ./scripts/backup_db.sh
# Cron (daily 2am):  0 2 * * * /path/to/law-agent/scripts/backup_db.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DB="${DATA_DIR:-data}/onegov.db"
OUT_DIR="backups"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$OUT_DIR/onegov-$STAMP.db"

[ -f "$DB" ] || { echo "No DB at $DB — nothing to back up"; exit 0; }
mkdir -p "$OUT_DIR"

# .backup is safe against concurrent writes (unlike cp)
sqlite3 "$DB" ".backup '$OUT'"
gzip -f "$OUT"
echo "Backed up to $OUT.gz"

# Keep the newest 14 local backups
ls -t "$OUT_DIR"/onegov-*.db.gz 2>/dev/null | tail -n +15 | xargs -r rm --

# Optional: push to Azure blob storage
if [ -n "${AZ_STORAGE_ACCOUNT:-}" ]; then
  az storage blob upload --account-name "$AZ_STORAGE_ACCOUNT" \
    --container-name "${AZ_CONTAINER:-backups}" \
    --file "$OUT.gz" --name "onegov-$STAMP.db.gz" --auth-mode login
  echo "Uploaded to Azure blob ${AZ_CONTAINER:-backups}/onegov-$STAMP.db.gz"
fi
