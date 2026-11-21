"""Sync the effnet embedding cache with the songs in mpd_dump.csv.

- computes embeddings for songs that don't have one cached yet
- deletes cached embeddings of songs that are no longer in the dump

Usage: python emb_extract.py [--dry-run]
"""

import csv
import os
import pickle
import re
import sys
from hashlib import md5
from pathlib import Path

import numpy as np
from tqdm import tqdm

from smart_playlist import MUSIC_DIR

PREFIX = "effnet"
CACHE_DIR = Path("cache")
MODEL_FILE = "discogs-effnet-bs64-1.pb"
SR = 16_000
MAX_SEC = 60

# exact match, so e.g. "effnet-onnx-<hash>" files are never touched
CACHE_NAME_RE = re.compile(rf"^{re.escape(PREFIX)}-[0-9a-f]{{32}}$")


def cache_path(full_path: str) -> Path:
    return CACHE_DIR / f"{PREFIX}-{md5(full_path.encode()).hexdigest()}"


def load_model():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TensorFlow startup logs
    import essentia
    from essentia.standard import TensorflowPredictEffnetDiscogs

    essentia.log.infoActive = False
    essentia.log.warningActive = False

    return TensorflowPredictEffnetDiscogs(graphFilename=MODEL_FILE, output="PartitionedCall:1")


def extract_emb(model, full_path: str) -> np.ndarray:
    from essentia.standard import MonoLoader

    audio = MonoLoader(filename=full_path, sampleRate=SR, resampleQuality=4)()
    n = MAX_SEC * SR
    if len(audio) > n:
        start = (len(audio) - n) // 2
        audio = audio[start:start + n]

    emb = model(audio).mean(axis=0)
    return emb / np.linalg.norm(emb)


def save_atomic(path: Path, emb: np.ndarray) -> None:
    # write to a temp file first so an interrupted run never leaves a broken pickle
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(emb, f)
    os.replace(tmp, path)


def main(dry_run: bool, dump_path: str = "mpd_dump.csv") -> None:
    with open(dump_path, newline="", encoding="utf-8") as f:
        full_paths = [str(MUSIC_DIR / row["path"]) for row in csv.DictReader(f)]
    if not full_paths:
        sys.exit(f"{dump_path} has no songs, refusing to touch the cache")

    expected = {cache_path(p).name: p for p in full_paths}
    cached = {p.name for p in CACHE_DIR.iterdir() if CACHE_NAME_RE.match(p.name)}

    to_delete = sorted(cached - expected.keys())
    to_add = [expected[name] for name in expected.keys() - cached]
    print(f"{len(cached)} cached, {len(expected)} songs: "
          f"{len(to_add)} to compute, {len(to_delete)} to delete")

    if dry_run:
        return

    for name in to_delete:
        (CACHE_DIR / name).unlink()

    if not to_add:
        return
    model = load_model()
    failed = []
    for full_path in tqdm(sorted(to_add)):
        try:
            save_atomic(cache_path(full_path), extract_emb(model, full_path))
        except Exception as ex:
            failed.append(full_path)
            tqdm.write(f"{full_path} -> {ex}")
    print(f"computed {len(to_add) - len(failed)}, failed {len(failed)}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv[1:])
