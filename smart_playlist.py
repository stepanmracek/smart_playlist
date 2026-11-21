import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpd import MPDClient

MUSIC_DIR = Path(os.environ.get("MUSIC_DIR", "/home/stepo/Hudba"))
TOP_N = 5  # pick the next song randomly from this many best candidates


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

    @classmethod
    def load(cls, csv_path: str = "db.csv", npz_path: str = "db.npz") -> "SongDB":
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        embeddings = np.load(npz_path)["embeddings"]
        assert len(rows) == len(embeddings), "db.csv and db.npz are out of sync"
        paths = [row["path"] for row in rows]
        # mpd_dump joins multiple artist tags with "; "
        artists = [set(filter(None, row["artist"].split("; "))) for row in rows]
        return cls(paths, artists, embeddings, {p: i for i, p in enumerate(paths)})


def on_song_start(
    client: MPDClient, db: SongDB, path: str, recent_artists: set[str], recent_paths: set[str]
) -> None:
    idx = db.index.get(path)
    if idx is None:
        return

    # embeddings are L2-normalized, so the dot product is cosine similarity
    similarity = db.embeddings @ db.embeddings[idx]
    candidates = []
    for candidate in np.argsort(-similarity):
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
            start = max(length - 10, 0)
            tail = client.playlistinfo((start, length))
            artists = set().union(*(song_artists(song) for song in tail))
            # current song is among the last 3, so it's always within tail
            current = tail[int(status["song"]) - start]
            on_song_start(client, db, current["file"], artists, {song["file"] for song in tail})
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
