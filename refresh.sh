#!/usr/bin/env bash
# Refresh the song DB after the MPD library changed
# (new songs, fixed tags, removed songs, ...).
set -euo pipefail

# run from the project dir so uv finds the project and relative paths work
cd "$(dirname "$(readlink -f "$0")")"

echo "==> Updating MPD database"
mpc --wait update >/dev/null

echo "==> Dumping MPD songs"
uv run mpd_dump.py

echo "==> Syncing embedding cache"
uv run emb_extract.py

echo "==> Building DB"
uv run cache_to_db.py

if pgrep -f 'python.*smart_playlist\.py' >/dev/null; then
    echo "==> NOTE: smart_playlist.py is running - restart it to load the new DB"
fi
