# Smart MPD playlist

Smart playlist for [MPD](https://www.musicpd.org/) based on audio embeddings.

`smart_playlist.py` watches MPD and, when the queue is about to run out, appends
a song that sounds similar to the one currently playing — picked from the whole
library by cosine similarity of [Discogs-EffNet](https://essentia.upf.edu/models.html)
embeddings.

## Requirements

- MPD server and `mpc`
- Python with `python-mpd2`, `numpy`, `tqdm`, `av` (PyAV), `onnxruntime`,
  `huggingface_hub`
- The ONNX effnet model is downloaded automatically from Hugging Face
  ([Heyian/discogs-effnet-onnx](https://huggingface.co/Heyian/discogs-effnet-onnx))
  on the first run of `emb_extract.py`. The exploration notebooks additionally
  need `essentia-tensorflow` and `discogs-effnet-bs64-1.pb`.
- Music files accessible locally under the same layout as MPD's music directory

## Configuration

Environment variables:

| Variable    | Default             | Meaning                                          |
|-------------|---------------------|--------------------------------------------------|
| `MPD_HOST`  | `localhost`         | MPD host, may be `password@host`                 |
| `MPD_PORT`  | `6600`              | MPD port                                         |
| `MUSIC_DIR` | `/home/stepo/Hudba` | local path of MPD's music directory              |

## Usage

Build (or refresh) the song database — run this whenever the MPD library
changes (new songs, fixed tags, removed songs):

```sh
./refresh.sh
```

Then start the playlist daemon (restart it after each refresh, the DB is loaded
only at startup):

```sh
python smart_playlist.py
```

When a new song starts and fewer than 3 songs are left after it, a song is
randomly picked from the `TOP_N` (5) most similar candidates whose artist isn't
among the last 10 queued songs, and appended to the queue.

## Pipeline

`refresh.sh` runs these steps:

1. `mpc --wait update` — MPD rescans its library.
2. `mpd_dump.py` — writes all MPD songs to `mpd_dump.csv`
   (`artist, album, track, path`; path is relative to the music directory).
3. `emb_extract.py` — syncs the embedding cache with the dump: computes effnet
   embeddings for new songs and deletes cached embeddings of removed songs
   (`--dry-run` only reports what would change).
4. `cache_to_db.py effnet` — collects cached embeddings into `db.npz`
   (`embeddings`, L2-normalized, float32) and `db.csv` (row *i* ↔ embedding *i*).
   Songs without an embedding are skipped.

Embeddings are cached in `cache/<prefix>-<md5 of path relative to the music
directory>` as pickled numpy arrays (the relative path is MPD's song path), so
moving the whole library keeps the cache; renamed files are recomputed.

## Other files

- `emb-extract-effnet.py`, `emb-extract-effnet-onnx.py`, `emb-extract-clap.py` —
  [marimo](https://marimo.io/) notebooks for exploring embeddings (t-SNE plots,
  similarity search). They share the `cache/` directory.

## Known limitations

- Tracks of albums stored as a single file + `.cue` sheet show up in MPD as
  virtual paths (`album.cue/track0001`) and can't be embedded; they're reported
  as failures by `emb_extract.py` and left out of the DB.
