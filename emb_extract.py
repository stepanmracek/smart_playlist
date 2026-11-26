"""Sync the embedding cache (cache/emb-<md5>) with the songs in mpd_dump.csv.

- computes embeddings for songs that don't have one cached yet
- deletes cached embeddings of songs that are no longer in the dump
- skips songs with a long silence ("hidden track" files: song, minutes of
  silence, hidden song) and (almost) silent tracks: their cache entry is None,
  so they're not retried and cache_to_db.py leaves them out of the DB

Embeddings come from the Discogs-EffNet model (ONNX), averaged over the middle
60 s of a song. Decoding (PyAV), resampling and the mel spectrogram reproduce
essentia's MonoLoader(sampleRate=16000, resampleQuality=4) +
TensorflowPredictEffnetDiscogs closely (cosine similarity ~0.9998), so
embeddings computed earlier with essentia/TensorFlow stay valid.

Usage: python emb_extract.py [--dry-run]
"""

import csv
import os
import pickle
import re
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import md5, sha256
from pathlib import Path

import av
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from smart_playlist import MUSIC_DIR

PREFIX = "emb"
CACHE_DIR = Path("cache")
MODEL_PATH = Path("discogs-effnet-bsdynamic-1.onnx")
# pinned revision + checksum, so the model can't change under existing embeddings
MODEL_URL = (
    "https://huggingface.co/Heyian/discogs-effnet-onnx/resolve/"
    "99d4b0ef1fd5d798f2afe049285c7d551e768e07/discogs-effnet-bsdynamic-1.onnx"
)
MODEL_SHA256 = "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c"
SR = 16_000
MAX_SEC = 60
FRAME, HOP = 512, 256  # mel spectrogram framing (essentia's TensorflowInputMusiCNN)
N_MELS = 96
PATCH_SIZE, PATCH_HOP = 128, 62  # model input patches (TensorflowPredictEffnetDiscogs)
SILENCE_DB = -60  # 100 ms frames quieter than this count as silent
# a song is skipped as a hidden-track / silent file if it has
HIDDEN_GAP_SEC = 20  # - a silence this long, not at the very start, followed by
HIDDEN_AFTER_SEC = 10  #   at least this much more sound (song, gap, hidden song)
MAX_SILENCE_SEC = 60  # - or a silence this long anywhere (minutes of dead air)
MAX_SILENT_FRACTION = 0.5  # - or is mostly silent

# exact match, so notebook cache files are never touched
CACHE_NAME_RE = re.compile(rf"^{re.escape(PREFIX)}-[0-9a-f]{{32}}$")


def cache_path(rel_path: str) -> Path:
    return CACHE_DIR / f"{PREFIX}-{md5(rel_path.encode()).hexdigest()}"


# --- audio ---------------------------------------------------------------------


def decode_mono_chunks(path: str) -> tuple[list[np.ndarray], int]:
    """Decode to mono float32 chunks at the file's native sample rate.
    Channels are averaged (like essentia's MonoMixer; ffmpeg's own downmix is
    (L+R)/sqrt(2), too loud) in batches as decoding goes, so only one channel
    of the song is ever kept in memory."""
    chunks, batch, batch_len = [], [], 0
    resampler_used = False

    def flush():
        nonlocal batch, batch_len
        if batch:
            chunks.append(np.concatenate(batch, axis=1).mean(axis=0, dtype=np.float32))
            batch, batch_len = [], 0

    with av.open(path) as container:
        stream = container.streams.audio[0]
        sr = stream.codec_context.sample_rate
        to_float = av.AudioResampler(format="fltp", rate=sr)  # planar float, same rate
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue  # skip corrupt packets like ffmpeg / essentia do
            for frame in frames:
                # most decoders (mp3, aac, vorbis) already output planar float
                if frame.format.name == "fltp":
                    outs = [frame]
                else:
                    outs, resampler_used = to_float.resample(frame), True
                for out in outs:
                    arr = out.to_ndarray()
                    batch.append(arr)
                    batch_len += arr.shape[1]
                    if batch_len >= 1 << 20:
                        flush()
        if resampler_used:  # flush samples buffered in the resampler
            for out in to_float.resample(None):
                batch.append(out.to_ndarray())
        flush()
    return chunks, sr


def slice_chunks(chunks: list[np.ndarray], lo: int, hi: int) -> np.ndarray:
    """Samples [lo, hi) of the concatenation of chunks, zero outside the signal."""
    out = np.zeros(hi - lo, np.float32)
    offset = 0
    for chunk in chunks:
        a, b = max(lo, offset), min(hi, offset + len(chunk))
        if a < b:
            out[a - lo : b - lo] = chunk[a - offset : b - offset]
        offset += len(chunk)
    return out


def silent_frames(chunks: list[np.ndarray], sr: int) -> np.ndarray:
    """Per 100 ms frame: is it silent? (partial frames at chunk ends are ignored)"""
    frame = sr // 10
    silent = []
    for chunk in chunks:
        frames = chunk[: len(chunk) // frame * frame].reshape(-1, frame)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        silent.append(20 * np.log10(rms + 1e-10) < SILENCE_DB)
    return np.concatenate(silent) if silent else np.zeros(0, bool)


def silence_stats(chunks: list[np.ndarray], sr: int) -> tuple[float, float]:
    """Longest silent stretch in seconds and the silent fraction of the song."""
    silent = silent_frames(chunks, sr)
    if not silent.any():
        return 0.0, 0.0
    edges = np.diff(np.concatenate([[0], silent.astype(np.int8), [0]]))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return runs.max() / 10, silent.mean()


def is_hidden_track(chunks: list[np.ndarray], sr: int) -> bool:
    """Hidden-track file (song, long silence, hidden song), minutes of dead
    air, or a (mostly) silent track. Quiet intros and silent endings shorter
    than MAX_SILENCE_SEC are fine."""
    silent = silent_frames(chunks, sr)
    if not silent.any():
        return False
    if silent.mean() > MAX_SILENT_FRACTION:
        return True
    edges = np.diff(np.concatenate([[0], silent.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1) / 10, np.flatnonzero(edges == -1) / 10
    total = len(silent) / 10
    for start, end in zip(starts, ends):
        gap = end - start
        if gap > MAX_SILENCE_SEC:
            return True
        if gap > HIDDEN_GAP_SEC and start > 5 and total - end >= HIDDEN_AFTER_SEC:
            return True
    return False


def load_middle(chunks: list[np.ndarray], sr: int) -> np.ndarray:
    """Middle MAX_SEC seconds at 16 kHz, same as MonoLoader(sampleRate=16000,
    resampleQuality=4) followed by taking the middle MAX_SEC seconds.
    Resampling is plain linear interpolation like libsamplerate's SRC_LINEAR
    (incl. its one-sample start delay, no anti-aliasing filter), computed only
    for the output samples that are kept."""
    n_native = sum(len(c) for c in chunks)
    if n_native == 0:
        return np.zeros(0, np.float32)
    n_out = round(n_native * SR / sr)  # length of the full 16 kHz signal
    n = min(MAX_SEC * SR, n_out)
    start = (n_out - n) // 2
    if sr == SR:  # no resampling needed
        return slice_chunks(chunks, start, start + n)
    pos = np.arange(start, start + n) * (sr / SR) - 1
    lo = int(np.floor(pos[0]))
    i = np.floor(pos).astype(np.int64)
    window = slice_chunks(chunks, lo, i[-1] + 2)  # covers x[i] and x[i + 1]
    frac = (pos - i).astype(np.float32)
    left, right = window[i - lo], window[i - lo + 1]
    return left + (right - left) * frac


# --- mel spectrogram -----------------------------------------------------------


def hz_to_mel(f):  # Slaney mel scale: linear below 1 kHz, logarithmic above
    f = np.asarray(f, dtype=np.float64)
    log_part = 15 + 27 * np.log(np.maximum(f, 1e-10) / 1000) / np.log(6.4)
    return np.where(f < 1000, 3 * f / 200, log_part)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    return np.where(m < 15, 200 * m / 3, 1000 * np.exp(np.log(6.4) / 27 * (m - 15)))


def mel_filterbank() -> np.ndarray:
    """Triangular Slaney mel filters with area normalization ("unit_tri"),
    equivalent to librosa.filters.mel(sr, n_fft, n_mels, htk=False, norm="slaney")."""
    fft_freqs = np.linspace(0, SR / 2, FRAME // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(SR / 2), N_MELS + 2))
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (fft_freqs - lower) / (center - lower)
    falling = (upper - fft_freqs) / (upper - center)
    weights = np.maximum(0, np.minimum(rising, falling))
    return (weights * (2 / (upper - lower))).astype(np.float32)  # [n_mels, n_bins]


WINDOW = np.hanning(FRAME).astype(np.float32)  # symmetric Hann, like essentia's Windowing
MEL_FB = mel_filterbank()


def melspec(audio: np.ndarray) -> np.ndarray:
    """Same as essentia FrameGenerator(512, 256, startFromZero) + TensorflowInputMusiCNN."""
    # essentia emits frames starting at s while s + HOP < len (last ones zero-padded)
    n_frames = max(0, (len(audio) - HOP - 1) // HOP + 1)
    padded = np.pad(audio, (0, FRAME))
    frames = np.lib.stride_tricks.sliding_window_view(padded, FRAME)[::HOP][:n_frames]
    power = np.abs(np.fft.rfft(frames * WINDOW, axis=1)) ** 2
    return np.log10(1 + 10000 * (power @ MEL_FB.T)).astype(np.float32)  # [n_frames, 96]


def patches(mel: np.ndarray) -> np.ndarray:
    if len(mel) < PATCH_SIZE:
        mel = np.pad(mel, ((0, PATCH_SIZE - len(mel)), (0, 0)))
    starts = range(0, len(mel) - PATCH_SIZE + 1, PATCH_HOP)
    return np.stack([mel[s : s + PATCH_SIZE] for s in starts])  # [n_patches, 128, 96]


# --- model -----------------------------------------------------------------------


def model_file() -> Path:
    """Path to the ONNX model, downloaded on first use."""
    if not MODEL_PATH.exists():
        print(f"downloading {MODEL_URL}")
        with urllib.request.urlopen(MODEL_URL) as resp:
            data = resp.read()
        if sha256(data).hexdigest() != MODEL_SHA256:
            sys.exit(f"{MODEL_URL}: checksum mismatch, not using it")
        tmp = MODEL_PATH.with_name(MODEL_PATH.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, MODEL_PATH)
    return MODEL_PATH


def load_model():
    """Single-threaded ONNX session for the model (parallelism is across songs)."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model_file()), opts, providers=["CPUExecutionProvider"])
    emb_out = next(o.name for o in sess.get_outputs() if o.shape[-1] == 1280)
    return sess, sess.get_inputs()[0].name, emb_out


def extract_emb(model, full_path: str) -> np.ndarray | None:
    """Embedding of a song, or None for hidden-track / silent files."""
    chunks, sr = decode_mono_chunks(full_path)
    if is_hidden_track(chunks, sr):
        return None
    sess, inp, out = model
    audio = load_middle(chunks, sr)
    emb = sess.run([out], {inp: patches(melspec(audio))})[0].mean(axis=0)
    return emb / np.linalg.norm(emb)


def save_atomic(path: Path, emb: np.ndarray | None) -> None:
    # write to a temp file first so an interrupted run never leaves a broken pickle
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(emb, f)
    os.replace(tmp, path)


_model = None  # per worker process


def _init_worker() -> None:
    global _model
    _model = load_model()


def _process(rel_path: str) -> tuple[str, str]:
    """Compute and cache one song's embedding -> (status, error message)."""
    try:
        emb = extract_emb(_model, str(MUSIC_DIR / rel_path))
        save_atomic(cache_path(rel_path), emb)
        return ("skipped" if emb is None else "ok"), ""
    except Exception as ex:  # noqa: BLE001 -- report and continue with other songs
        return "failed", str(ex)


def main(dry_run: bool, dump_path: str = "mpd_dump.csv") -> None:
    with open(dump_path, newline="", encoding="utf-8") as f:
        rel_paths = [row["path"] for row in csv.DictReader(f)]
    if not rel_paths:
        sys.exit(f"{dump_path} has no songs, refusing to touch the cache")

    expected = {cache_path(p).name: p for p in rel_paths}
    cached = {p.name for p in CACHE_DIR.iterdir() if CACHE_NAME_RE.match(p.name)}

    to_delete = sorted(cached - expected.keys())
    to_add = [expected[name] for name in expected.keys() - cached]
    print(
        f"{len(cached)} cached, {len(expected)} songs: "
        f"{len(to_add)} to compute, {len(to_delete)} to delete"
    )

    if dry_run:
        return

    for name in to_delete:
        (CACHE_DIR / name).unlink()

    if not to_add:
        return
    model_file()  # download once here, not in every worker
    failed, skipped = [], []
    with ProcessPoolExecutor(os.cpu_count(), initializer=_init_worker) as pool:
        futures = {pool.submit(_process, p): p for p in sorted(to_add)}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rel_path = futures[fut]
            status, err = fut.result()
            if status == "skipped":
                skipped.append(rel_path)
                tqdm.write(f"{rel_path} -> skipped (hidden track / silence)")
            elif status == "failed":
                failed.append(rel_path)
                tqdm.write(f"{rel_path} -> {err}")
    print(
        f"computed {len(to_add) - len(failed) - len(skipped)}, "
        f"skipped {len(skipped)} (hidden track / silence), failed {len(failed)}"
    )


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv[1:])
