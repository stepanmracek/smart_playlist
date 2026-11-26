import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpd import MPDClient

MUSIC_DIR = Path(os.environ.get("MUSIC_DIR", "/home/stepo/Hudba"))
TOP_N = 5  # pick the next song randomly from this many best candidates
# the query is a weighted average of the embeddings of the current song and the ones
# before it (current song first): the current song drives the transition, older
# songs gently anchor it and slow down drift
HISTORY_WEIGHTS = tuple(float(w) for w in os.environ.get("HISTORY_WEIGHTS", "0.6,0.3,0.1").split(","))
HISTORY_K = len(HISTORY_WEIGHTS)
RECENT_N = 10  # artists and songs from this many last queue songs are not picked again
# CSLS ranking against hubs: candidates are ranked by 2*cos - r, where r is the mean
# similarity of a song to its CSLS_K nearest neighbours (0 = plain cosine similarity)
CSLS_K = int(os.environ.get("CSLS_K", 10))
# rank in a centered space (global mean embedding subtracted, re-normalized): averaging
# history embeddings pulls the query towards the global mean, where hubs live;
# centering removes that shared direction
CENTER = os.environ.get("CENTER", "1") != "0"


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
    index: dict[str, int]  # path -> row
    search: np.ndarray  # (n, dim) embeddings used for ranking: centered + re-normalized if CENTER
    density: np.ndarray  # (n,) CSLS penalty r: mean similarity (in search space) to own CSLS_K nearest neighbours

    @classmethod
    def load(cls, csv_path: str = "db.csv", npz_path: str = "db.npz") -> "SongDB":
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # npz arrays are read lazily: the raw embeddings are only read from disk when
        # the ranking space has to be computed
        with np.load(npz_path) as data:
            # cache_to_db.py precomputes the ranking space (slow on weak hardware:
            # ~2 min on a Raspberry Pi 2); use it if built with the current settings
            if "search" in data and bool(data["center"]) == CENTER and int(data["csls_k"]) == CSLS_K:
                search, density = data["search"], data["density"]
            else:
                print(f"{npz_path} has no ranking space precomputed for CENTER={int(CENTER)} "
                      f"CSLS_K={CSLS_K}, computing it (rerun cache_to_db.py to precompute)",
                      file=sys.stderr, flush=True)
                search, density = ranking_space(data["embeddings"], CENTER, CSLS_K)
        assert len(rows) == len(search), "db.csv and db.npz are out of sync"
        paths = [row["path"] for row in rows]
        # mpd_dump joins multiple artist tags with "; "
        artists = [set(filter(None, row["artist"].split("; "))) for row in rows]
        index = {p: i for i, p in enumerate(paths)}
        return cls(paths, artists, index, search, density)


def ranking_space(embeddings: np.ndarray, center: bool, csls_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings used for ranking (centered + re-normalized if center) and their CSLS
    density. The density costs O(n² · dim): precomputed by cache_to_db.py."""
    search = embeddings
    if center:
        search = embeddings - embeddings.mean(axis=0)
        search /= np.linalg.norm(search, axis=1, keepdims=True)
    return search, neighbourhood_density(search, csls_k)


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
    client: MPDClient,
    db: SongDB,
    history: list[str],
    recent_artists: set[str],
    recent_paths: set[str],
    weights: tuple[float, ...] = HISTORY_WEIGHTS,
) -> None:
    """history: paths of the last len(weights) queue songs, ending with the current one."""
    path = history[-1]
    if path not in db.index:
        return

    # weighted average of history embeddings (weights[0] for the current song; songs
    # not in DB are skipped), re-normalized so the dot product with the L2-normalized
    # DB embeddings is cosine similarity
    rows = [(db.index[p], w) for p, w in zip(reversed(history), weights) if p in db.index]
    query = sum(w * db.search[i] for i, w in rows)
    query /= np.linalg.norm(query)
    # CSLS: 2*cos(query, song) - r(song) - r(query); r(query) is the same for all songs
    score = 2 * (db.search @ query) - db.density
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
    print(f"{path} -> {next_path} (CSLS score {score[chosen]:.3f})", flush=True)


def song_artists(song: dict) -> set[str]:
    artist = song.get("artist", [])
    return {artist} if isinstance(artist, str) else set(artist)


def watch_song_changes(client: MPDClient, db: SongDB) -> None:
    # songid of the song the queue was last extended from: each song extends the
    # queue at most once, even if it stays last (e.g. no candidate was found)
    extended_songid = None
    # Check first, then wait: the first pass handles the state at startup (the daemon
    # misses events while down, and a song already playing emits no new event).
    while True:
        status = client.status()
        songid = status.get("songid")
        length = int(status["playlistlength"])
        # Off in these modes: random makes queue order meaningless as "last" and
        # "history", consume deletes the history, and with repeat the queue loops
        # (appending would break e.g. a repeated album). consume can also be "oneshot".
        modes_off = any(status.get(mode, "0") != "0" for mode in ("random", "consume", "repeat"))
        if (
            status.get("state") == "play"
            and songid != extended_songid
            and not modes_off
            and int(status["song"]) == length - 1  # only extend from the last queue song
        ):
            extended_songid = songid
            # The current song is the last one, so both windows end with it:
            # - recent: the last RECENT_N queue songs (their artists and paths
            #   must not be picked again)
            # - history: the last HISTORY_K songs (their embeddings are averaged
            #   to find the next song)
            # Fetch one slice covering the longer of the two.
            start = max(length - max(RECENT_N, HISTORY_K), 0)
            songs = client.playlistinfo((start, length))
            recent = songs[-RECENT_N:]
            artists = set().union(*(song_artists(song) for song in recent))
            history = [song["file"] for song in songs[-HISTORY_K:]]
            on_song_start(client, db, history, artists, {song["file"] for song in recent})
        # player: play/pause/stop/seek/song change; options: random/consume/repeat
        # toggled (switching a mode off while the last song plays must still extend
        # the queue, otherwise playback stops after it)
        client.idle("player", "options")


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
