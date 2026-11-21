"""Collect cached embeddings for songs in mpd_dump.csv into db.npz + db.csv.

Usage: python cache_to_db.py <prefix>    (e.g. effnet, clap, effnet-onnx)

Row i of db.csv corresponds to db.npz["embeddings"][i]. Songs without a cached
embedding are skipped.
"""

import csv
import pickle
import sys
from hashlib import md5
from pathlib import Path

import numpy as np

from smart_playlist import MUSIC_DIR

CACHE_DIR = Path("cache")


def load_embedding(prefix: str, rel_path: str) -> np.ndarray | None:
    full_path = str(MUSIC_DIR / rel_path)
    cached = CACHE_DIR / f"{prefix}-{md5(full_path.encode()).hexdigest()}"
    if not cached.exists():
        return None
    with open(cached, "rb") as f:
        return pickle.load(f)


def main(prefix: str, dump_path: str = "mpd_dump.csv") -> None:
    with open(dump_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        songs = list(reader)

    rows, embeddings = [], []
    for song in songs:
        emb = load_embedding(prefix, song["path"])
        if emb is not None:
            rows.append(song)
            embeddings.append(emb)

    if not embeddings:
        sys.exit(f"No cached '{prefix}' embeddings found for songs in {dump_path}")

    np.savez("db.npz", embeddings=np.stack(embeddings).astype(np.float32))
    with open("db.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)}/{len(songs)} songs have '{prefix}' embeddings "
          f"({len(songs) - len(rows)} missing) -> db.npz, db.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
