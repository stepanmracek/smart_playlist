import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpd import MPDClient

MUSIC_DIR = Path(os.environ.get("MUSIC_DIR", "/home/stepo/Hudba"))
TOP_N = 5  # pick the next song randomly from this many best candidates
HISTORY_K = int(os.environ.get("HISTORY_K", 3))  # average embeddings of this many songs up to the current one
RECENT_N = 10  # artists and songs from this many last queue songs are not picked again
# CSLS ranking against hubs: candidates are ranked by 2*cos - r, where r is the mean
# similarity of a song to its CSLS_K nearest neighbours (0 = plain cosine similarity)
CSLS_K = int(os.environ.get("CSLS_K", 10))


def connect_mpd() -> MPDClient:
    host = os.environ.get("MPD_HOST", "localhost")
    port = int(os.environ.get("MPD_PORT", 6600))
    password = None
    # MPD convention: MPD_HOST may be "password@host"
    if "@" in host:
        password, host = host.split("@", 1)

    client = MPDClient()
    client.timeout = 10
    client.idletimeout = None
    client.connect(host, port)
    if password:
        client.password(password)
    return client


@dataclass
class SongDB:
    paths: list[str]  # MPD paths, relative to MUSIC_DIR
    artists: list[set[str]]
    embeddings: np.ndarray  # (n, dim), L2-normalized
    index: dict[str, int]  # path -> row
    density: np.ndarray  # (n,) CSLS penalty r: mean similarity to own CSLS_K nearest neighbours

    @classmethod
    def load(cls, csv_path: str = "db.csv", npz_path: str = "db.npz") -> "SongDB":
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        embeddings = np.load(npz_path)["embeddings"]
        assert len(rows) == len(embeddings), "db.csv and db.npz are out of sync"
        paths = [row["path"] for row in rows]
        # mpd_dump joins multiple artist tags with "; "
        artists = [set(filter(None, row["artist"].split("; "))) for row in rows]
        index = {p: i for i, p in enumerate(paths)}
        return cls(paths, artists, embeddings, index, neighbourhood_density(embeddings, CSLS_K))


def neighbourhood_density(embeddings: np.ndarray, k: int, chunk: int = 1000) -> np.ndarray:
    """Mean cosine similarity of each song to its k nearest neighbours (zeros if k == 0).
    Hubs sit in dense regions (high value) and get penalized by CSLS; isolated songs
    get a boost. Computed in chunks to keep memory low."""
    density = np.zeros(len(embeddings), np.float32)
    if k == 0:
        return density
    for start in range(0, len(embeddings), chunk):
        sims = embeddings[start:start + chunk] @ embeddings.T
        sims[np.arange(len(sims)), np.arange(start, start + len(sims))] = -np.inf  # not itself
        density[start:start + chunk] = np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)
    return density


def on_song_start(
    client: MPDClient, db: SongDB, history: list[str], recent_artists: set[str], recent_paths: set[str]
) -> None:
    """history: paths of the last HISTORY_K queue songs, ending with the current one."""
    path = history[-1]
    if path not in db.index:
        return

    # mean of history embeddings (songs not in DB are skipped), re-normalized so the
    # dot product with the L2-normalized DB embeddings is cosine similarity
    query = db.embeddings[[db.index[p] for p in history if p in db.index]].mean(axis=0)
    query /= np.linalg.norm(query)
    similarity = db.embeddings @ query
    # CSLS: 2*cos(query, song) - r(song) - r(query); r(query) is the same for all songs
    score = 2 * similarity - db.density
    candidates = []
    for candidate in np.argsort(-score):
        # path check also covers the current song and songs without artist tags
        if db.paths[candidate] not in recent_paths and not db.artists[candidate] & recent_artists:
            candidates.append(candidate)
            if len(candidates) == TOP_N:
                break
    if not candidates:
        return

    chosen = random.choice(candidates)
    next_path = db.paths[chosen]
    client.add(next_path)
    print(f"{path} -> {next_path} (similarity {similarity[chosen]:.3f})", flush=True)


def song_artists(song: dict) -> set[str]:
    artist = song.get("artist", [])
    return {artist} if isinstance(artist, str) else set(artist)


def watch_song_changes(client: MPDClient, db: SongDB) -> None:
    last_songid = client.status().get("songid")
    while True:
        client.idle("player")  # blocks until play/pause/stop/seek/song change
        status = client.status()
        songid = status.get("songid")
        if status.get("state") == "play" and songid != last_songid:
            length = int(status["playlistlength"])
            songs_left = length - int(status["song"]) - 1  # after the current one
            if songs_left >= 3:
                last_songid = songid
                continue
            current = int(status["song"])
            # Fetch one slice of the queue that covers both windows we need:
            # - recent: the last RECENT_N queue songs (their artists and paths
            #   must not be picked again)
            # - history: the last HISTORY_K songs up to and including the current
            #   one (their embeddings are averaged to find the next song)
            # The current song is among the last 3, so with a large HISTORY_K the
            # history window can start before the recent one; start at whichever
            # begins first (never below queue position 0).
            start = max(min(length - RECENT_N, current - HISTORY_K + 1), 0)
            songs = client.playlistinfo((start, length))
            recent = songs[-RECENT_N:]
            artists = set().union(*(song_artists(song) for song in recent))
            history = [song["file"] for song in songs[: current - start + 1][-HISTORY_K:]]
            on_song_start(client, db, history, artists, {song["file"] for song in recent})
        last_songid = songid


if __name__ == "__main__":
    db = SongDB.load()
    print(f"Loaded {len(db.paths)} songs from DB", flush=True)
    client = connect_mpd()
    try:
        watch_song_changes(client, db)
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
