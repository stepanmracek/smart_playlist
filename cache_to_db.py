"""Collect cached embeddings for songs in mpd_dump.csv into db.npz + db.csv.

Usage: python cache_to_db.py [prefix]

prefix defaults to "emb" (emb_extract.py); notebook caches (e.g. effnet, clap,
effnet-onnx) can be used to build a DB for experiments.

Row i of db.csv corresponds to db.npz["embeddings"][i]. Songs without a cached
embedding are skipped.

db.npz also holds the ranking space for the daemon (search, density), computed with
the CENTER and CSLS_K environment settings (stored as center, csls_k): it's the slow
part of loading the DB, so the daemon doesn't compute it at startup when they match.
"""

import csv
import pickle
import sys
from hashlib import md5
from pathlib import Path

import numpy as np

from smart_playlist import CENTER, CSLS_K, ranking_space

CACHE_DIR = Path("cache")


def load_embedding(prefix: str, rel_path: str) -> np.ndarray | None:
    cached = CACHE_DIR / f"{prefix}-{md5(rel_path.encode()).hexdigest()}"
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

    embeddings = np.stack(embeddings).astype(np.float32)
    search, density = ranking_space(embeddings, CENTER, CSLS_K)
    np.savez(
        "db.npz",
        embeddings=embeddings,
        search=search,
        density=density,
        center=CENTER,
        csls_k=CSLS_K,
    )
    with open("db.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{len(rows)}/{len(songs)} songs have '{prefix}' embeddings "
        f"({len(songs) - len(rows)} missing) -> db.npz, db.csv "
        f"(ranking space: CENTER={int(CENTER)} CSLS_K={CSLS_K})"
    )


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    main(sys.argv[1] if len(sys.argv) == 2 else "emb")
