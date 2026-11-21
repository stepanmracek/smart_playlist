import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import numpy as np
    import random
    import os
    import marimo as mo
    import pickle
    import pandas as pd
    import altair as alt
    from pathlib import Path
    from hashlib import md5
    from sklearn.manifold import TSNE
    from glob import glob
    import onnxruntime as ort
    import essentia.standard as es
    from huggingface_hub import hf_hub_download


@app.cell
def _():
    REPO = "Heyian/discogs-effnet-onnx"
    SR = 16_000
    MAX_SEC = 60
    PATCH_SIZE = 128   # frames per patch (~2 s)
    PATCH_HOP = 62     # Essentia's default for this model
    return MAX_SEC, PATCH_HOP, PATCH_SIZE, REPO, SR


@app.cell
def _(REPO):
    model_path = hf_hub_download(REPO, "discogs-effnet-bsdynamic-1.onnx")
    return (model_path,)


@app.cell
def _(model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return (sess,)


@app.cell
def _(sess):
    input_name = sess.get_inputs()[0].name
    emb_idx = next(k for k, o in enumerate(sess.get_outputs()) if o.shape[-1] == 1280)
    return emb_idx, input_name


@app.cell
def _(MAX_SEC, SR):
    def melspec(path):
        audio = es.MonoLoader(filename=path, sampleRate=SR, resampleQuality=4)()
        n = MAX_SEC * SR
        if len(audio) > n:
            start = (len(audio) - n) // 2
            audio = audio[start:start + n]
        extractor = es.TensorflowInputMusiCNN()  # 512/256 frames, 96 mels, log10(1+10000x)
        frames = [
            extractor(f) for f in es.FrameGenerator(
                audio, frameSize=512, hopSize=256,startFromZero=True
            )
        ]
        return np.array(frames, dtype=np.float32)  # [n_frames, 96]

    return (melspec,)


@app.cell
def _(PATCH_HOP, PATCH_SIZE):
    def make_patches(mel):
        if len(mel) < PATCH_SIZE:  # pad very short clips
            mel = np.pad(mel, ((0, PATCH_SIZE - len(mel)), (0, 0)))
        starts = range(0, len(mel) - PATCH_SIZE + 1, PATCH_HOP)
        return np.stack([mel[s:s + PATCH_SIZE] for s in starts])  # [n_patches, 128, 96]

    return (make_patches,)


@app.cell
def _(emb_idx, input_name, make_patches, melspec, sess):
    def embed(path):
        cached_path = Path("cache") / ("effnet-onnx-" + md5(path.encode()).hexdigest())
        if cached_path.exists():
            with open(cached_path, "rb") as f:
                return pickle.load(f)
            
        mel = melspec(path)
        patches = make_patches(mel)
        patch_emb = sess.run(None, {input_name: patches})[emb_idx]
        emb = patch_emb.mean(axis=0)
        emb = emb / np.linalg.norm(emb)

        with open(cached_path, "wb") as f:
            pickle.dump(emb, f)

        return emb

    return (embed,)


@app.cell
def _():
    files = sorted(glob("/home/stepo/Hudba/*/*/*.mp3"))
    random.seed(42)
    files = random.sample(files, 1000)
    files[:10]
    return (files,)


@app.cell
def _(embed, files):
    embeddings = np.array([embed(file) for file in mo.status.progress_bar(files)])
    return (embeddings,)


@app.cell
def _(embeddings):
    projected = TSNE().fit_transform(embeddings)
    return (projected,)


@app.cell
def _(files, projected):
    # Extract metadata from file paths (assuming /home/stepo/Hudba/Artist/Album/Track.mp3)
    data = []
    for f, coords in zip(files, projected):
        path_obj = Path(f)
        track = path_obj.stem
        album = path_obj.parent.name
        artist = path_obj.parent.parent.name
        data.append({
            "x": coords[0],
            "y": coords[1],
            "track": track,
            "album": album,
            "artist": artist,
            "path": f
        })

    df_tsne = pd.DataFrame(data)
    return (df_tsne,)


@app.cell
def _(df_tsne):
    chart = (
        alt.Chart(df_tsne)
        .mark_circle(size=100, opacity=0.8)
        .encode(
            x=alt.X("x:Q", title="t-SNE 1", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", title="t-SNE 2", scale=alt.Scale(zero=False)),
            color=alt.Color("artist:N", title="Artist"),
            tooltip=[
                alt.Tooltip("artist:N", title="Artist"),
                alt.Tooltip("album:N", title="Album"),
                alt.Tooltip("track:N", title="Track"),
            ],
        )
        .properties(
            title="t-SNE Projection of Audio Embeddings",
            width="container",
            height=500,
        )
        #.interactive()
    )

    tsne_plot = mo.ui.altair_chart(chart)
    tsne_plot
    return (tsne_plot,)


@app.cell
def _(tsne_plot):
    # Play audio for the selected points in the plot
    selected_tracks = tsne_plot.value

    if not selected_tracks.empty:
        audio_players = [
            mo.vstack([
                mo.md(f"**{row['artist']} - {row['track']}** (*{row['album']}*)"),
                mo.audio(row['path'])
            ])
            for _, row in selected_tracks.head(5).iterrows() # Limit to 5 to avoid clutter
        ]
        output = mo.vstack([
            mo.md(f"### Selected Tracks ({len(selected_tracks)})"),
            *audio_players
        ])
    else:
        output = mo.md("*Click or drag on the plot to select tracks and listen to them.*")

    output
    return


@app.cell
def _(embeddings):
    similarity_scores = embeddings[0:5] @ embeddings[5:].T
    return (similarity_scores,)


@app.cell(expand_output=True)
def _(df_tsne, similarity_scores):
    # Sort all candidate indices by similarity in descending order for each of the first 5 songs
    # Since similarity_scores is computed against embeddings_np[5:], we add 5 to the indices
    sorted_indices = np.argsort(similarity_scores, axis=1)[:, ::-1] + 5

    tabs = {}
    for i in range(5):
        query_song = df_tsne.iloc[i]

        # Header with the query song and its audio player
        query_info = mo.vstack([
            mo.md(f"### Query Song: **{query_song['artist']} - {query_song['track']}** (*{query_song['album']}*)"),
            mo.audio(query_song['path']),
            mo.md("#### Top 10 Most Similar Songs (Unique Artists):")
        ])

        # List of similar songs with scores and audio players, skipping already seen artists
        similar_items = []
        seen_artists = {query_song['artist']}
        rank = 1
        for idx in sorted_indices[i]:
            sim_song = df_tsne.iloc[idx]
            if sim_song['artist'] in seen_artists:
                continue
            seen_artists.add(sim_song['artist'])

            score = similarity_scores[i, idx - 5]
            similar_items.append(
                mo.vstack([
                    mo.md(f"**{rank}. {sim_song['artist']} - {sim_song['track']}** (*{sim_song['album']}*) — Similarity: `{score:.3f}`"),
                    mo.audio(sim_song['path'])
                ])
            )
            rank += 1
            if rank > 10:
                break

        tabs[f"{i+1}. {query_song['artist']} - {query_song['track']}"] = mo.vstack([query_info, *similar_items])

    mo.ui.tabs(tabs)
    return


if __name__ == "__main__":
    app.run()
