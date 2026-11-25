# Smart MPD playlist

Smart playlist for [MPD](https://www.musicpd.org/) based on audio embeddings.

`smart_playlist.py` watches MPD and, when the queue is about to run out, appends
a song that sounds similar to the one currently playing — picked from the whole
library by cosine similarity of [Discogs-EffNet](https://essentia.upf.edu/models.html)
embeddings.

## Requirements

- MPD server and `mpc`
- [uv](https://docs.astral.sh/uv/) — manages Python and the dependencies
  (`pyproject.toml`, `uv.lock`):
  - `uv sync` — runtime (`python-mpd2`, `numpy`, `tqdm`, `av` (PyAV),
    `onnxruntime`) plus the notebook packages (`dev` group: marimo,
    essentia-tensorflow, CPU-only torch, ...)
  - `uv sync --no-dev` — runtime only, enough for the daemon and `refresh.sh`
- The ONNX effnet model (`discogs-effnet-bsdynamic-1.onnx`, from
  [Heyian/discogs-effnet-onnx](https://huggingface.co/Heyian/discogs-effnet-onnx),
  pinned revision, checksum verified) is downloaded into the project directory
  on the first run of `emb_extract.py`. The effnet notebook additionally needs
  `discogs-effnet-bs64-1.pb`.
- Music files accessible locally under the same layout as MPD's music directory

## Configuration

Environment variables:

| Variable    | Default             | Meaning                                          |
|-------------|---------------------|--------------------------------------------------|
| `MPD_HOST`  | `localhost`         | MPD host, may be `password@host`                 |
| `MPD_PORT`  | `6600`              | MPD port                                         |
| `MUSIC_DIR` | `/home/stepo/Hudba` | local path of MPD's music directory              |
| `HISTORY_WEIGHTS` | `0.6,0.3,0.1` | weights of the current song and the ones before it in the query (weighted average of embeddings) |
| `CSLS_K`    | `10`                | CSLS neighbourhood size against hubs; `0` = plain cosine similarity |
| `CENTER`    | `1`                 | rank in a centered embedding space (global mean subtracted), against the pull of averaged queries towards hubs; `0` = off |

`CSLS_K` and `CENTER` are also read by `cache_to_db.py`, which precomputes the
ranking space for them (see [Pipeline](#pipeline)); set them the same for both.

## Usage

Build (or refresh) the song database — run this whenever the MPD library
changes (new songs, fixed tags, removed songs):

```sh
./refresh.sh
```

Then start the playlist daemon (restart it after each refresh, the DB is loaded
only at startup):

```sh
uv run smart_playlist.py
```

When a new song starts and fewer than 3 songs are left after it, a song is
randomly picked from the `TOP_N` (5) most similar candidates whose artist isn't
among the last 10 queued songs, and appended to the queue. Similar to a
weighted average of the current song and the two before it (mostly the current
one, the older ones slow down drift), ranked by
[CSLS](https://arxiv.org/abs/1710.04087) in a centered embedding space so that
"hub" songs don't keep coming back.

## Pipeline

`refresh.sh` runs these steps:

1. `mpc --wait update` — MPD rescans its library.
2. `mpd_dump.py` — writes all MPD songs to `mpd_dump.csv`
   (`artist, album, track, path`; path is relative to the music directory).
3. `emb_extract.py` — syncs the embedding cache with the dump: computes embeddings
   for new songs and deletes cached embeddings of removed songs
   (`--dry-run` only reports what would change). Hidden-track files (song, long
   silence, hidden song), minutes of dead air and silent tracks are skipped:
   their cache entry is `None`, so they're left out of the DB.
4. `cache_to_db.py` — collects cached embeddings into `db.npz`
   (`embeddings`, L2-normalized, float32) and `db.csv` (row *i* ↔ embedding *i*).
   Songs without an embedding are skipped. It also precomputes the ranking space
   (`search`, `density`) for the current `CENTER` and `CSLS_K` settings: the CSLS
   density compares every song with every other one, which takes ~2 s on a desktop
   but ~2 min on a Raspberry Pi 2. The daemon uses the precomputed arrays when its
   settings match and computes them at startup otherwise. To run the daemon on
   a weak machine (e.g. the MPD server), build the DB on a fast one and copy
   `db.csv` + `db.npz` over.

Embeddings are cached in `cache/emb-<md5 of path relative to the music
directory>` as pickled numpy arrays (the relative path is MPD's song path), so
moving the whole library keeps the cache; renamed files are recomputed. The
file names don't say which model produced them: after switching the model,
delete `cache/emb-*` and run `./refresh.sh`.

## Other files

- `notebooks/` — [marimo](https://marimo.io/) notebooks for exploring
  embeddings (t-SNE plots, similarity search): `emb-extract-effnet.py`,
  `emb-extract-effnet-onnx.py`, `emb-extract-clap.py`. They use the project's
  `cache/` directory with their own prefixes (`effnet-`, `effnet-onnx-`,
  `clap-`), separate from `emb-` used by the playlist; `uv run cache_to_db.py
  <prefix>` builds a DB from them for experiments. Open one with
  `uv run marimo edit notebooks/emb-extract-effnet.py`.

## Known limitations

- Tracks of albums stored as a single file + `.cue` sheet show up in MPD as
  virtual paths (`album.cue/track0001`) and can't be embedded; they're reported
  as failures by `emb_extract.py` and left out of the DB.
