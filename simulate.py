"""Simulate smart playlists offline (no MPD) to compare settings.

Usage: python simulate.py <seed path> [--weights 0.6,0.3,0.1 1,1,1 ...] [--runs 10]
                          [--length 20] [--show 2]

--weights: history weight settings to compare (current song first; default
HISTORY_WEIGHTS). CSLS_K and CENTER are read from the environment like in the daemon.
"""

import argparse
import contextlib
import io
import itertools
import random
from collections import Counter
from pathlib import Path

import numpy as np

from smart_playlist import HISTORY_WEIGHTS, RECENT_N, SongDB, on_song_start


class FakeClient:
    def __init__(self, queue: list[str]):
        self.queue = queue

    def add(self, path: str) -> None:
        self.queue.append(path)


def simulate(db: SongDB, seed: str, weights: tuple[float, ...], length: int) -> list[str]:
    queue = [seed]
    client = FakeClient(queue)
    pos = 0
    # same logic as watch_song_changes: on each song start, top up when < 3 left
    while len(queue) < length and pos < len(queue):
        if len(queue) - pos - 1 < 3:
            recent = queue[-RECENT_N:]
            artists = set().union(*(db.artists[db.index[p]] for p in recent))
            history = queue[: pos + 1][-len(weights) :]
            with contextlib.redirect_stdout(io.StringIO()):
                on_song_start(client, db, history, artists, set(recent), weights)
        pos += 1
    return queue


def label(db: SongDB, path: str) -> str:
    return f"{'; '.join(sorted(db.artists[db.index[path]]))} – {Path(path).stem}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed")
    ap.add_argument("--weights", nargs="+", default=[",".join(map(str, HISTORY_WEIGHTS))])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--show", type=int, default=2, help="example playlists printed per K")
    args = ap.parse_args()

    db = SongDB.load()
    # metrics use plain cosine similarity of the raw embeddings: comparable across
    # CENTER / CSLS_K settings, unlike the ranking space
    embeddings = np.load("db.npz")["embeddings"]
    seed_emb = embeddings[db.index[args.seed]]
    checkpoints = [p for p in (5, 10, 15, 20) if p <= args.length]

    for weights_arg in args.weights:
        weights = tuple(float(w) for w in weights_arg.split(","))
        playlists = []
        for run in range(args.runs):
            random.seed(run)
            playlists.append(simulate(db, args.seed, weights, args.length))

        sims = np.array([[embeddings[db.index[p]] @ seed_emb for p in pl] for pl in playlists])
        steps = np.array(
            [
                [
                    embeddings[db.index[a]] @ embeddings[db.index[b]]
                    for a, b in itertools.pairwise(pl)
                ]
                for pl in playlists
            ]
        )
        songs = Counter(p for pl in playlists for p in pl[1:])
        artists = [len(set().union(*(db.artists[db.index[p]] for p in pl))) for pl in playlists]

        print(f"=== weights {weights_arg} ===")
        print(
            "similarity to seed at position "
            + ", ".join(f"{p}: {sims[:, p - 1].mean():.3f}" for p in checkpoints)
        )
        print(f"mean similarity between consecutive songs: {steps.mean():.3f}")
        print(f"distinct artists per playlist: {np.mean(artists):.1f}")
        print(f"distinct songs over all runs: {len(songs)} / {sum(songs.values())}")
        print(
            "most frequent songs: "
            + ", ".join(f"{label(db, p)} ({n}x)" for p, n in songs.most_common(3))
        )
        for i, pl in enumerate(playlists[: args.show], 1):
            print(f"\n  example {i}:")
            for n, p in enumerate(pl, 1):
                print(f"  {n:2}. {label(db, p)}")
        print()


if __name__ == "__main__":
    main()
