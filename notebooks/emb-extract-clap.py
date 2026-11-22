import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import random
    import librosa
    import numpy as np
    import torch
    import marimo as mo
    import pandas as pd
    import altair as alt
    import pickle
    from pathlib import Path
    from hashlib import md5
    from sklearn.manifold import TSNE
    from transformers import ClapModel, ClapProcessor
    from glob import glob


@app.cell
def _(mo):
    MODEL_NAME = "laion/larger_clap_music"
    SR = 48_000
    CHUNK_SEC = 10
    MAX_SEC = 60
    MUSIC_DIR = Path("/home/stepo/Hudba")
    PROJECT_DIR = mo.notebook_dir().parent  # cache/ and model files live there
    CACHE_DIR = PROJECT_DIR / "cache"
    return CACHE_DIR, CHUNK_SEC, MAX_SEC, MODEL_NAME, MUSIC_DIR, SR


@app.cell
def _(MODEL_NAME):
    model = ClapModel.from_pretrained(MODEL_NAME).eval()
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    return model, processor


@app.cell
def _(CACHE_DIR, CHUNK_SEC, MAX_SEC, MUSIC_DIR, SR, model, processor):
    @torch.no_grad()
    def embed(path):
        cached_path = CACHE_DIR / ("clap-" + md5(str(Path(path).relative_to(MUSIC_DIR)).encode()).hexdigest())
        if cached_path.exists():
            with open(cached_path, "rb") as f:
                return pickle.load(f)

        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as ex:
            print(path, "->", ex)
            return None

        n = MAX_SEC * SR
        if len(audio) > n:
            start = (len(audio) - n) // 2
            audio = audio[start:start + n]
        step = CHUNK_SEC * SR
        chunks = [audio[i:i + step] for i in range(0, len(audio), step)]
        chunks = [c for c in chunks if len(c) > SR * 2] or [audio]

        chunk_embs = []
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            inputs = processor(audio=chunks[i:i+batch_size], sampling_rate=SR, return_tensors="pt")
            chunk_embs.append(model.get_audio_features(**inputs).pooler_output)

        emb = torch.cat(chunk_embs).mean(dim=0)
        emb = emb / emb.norm()
        emb = emb.numpy()

        with open(cached_path, "wb") as f:
            pickle.dump(emb, f)

        return emb

    return (embed,)


@app.cell
def _(MUSIC_DIR):
    files = sorted(glob(str(MUSIC_DIR / "*/*/*.mp3")))
    random.seed(42)
    random.shuffle(files)
    #files = random.sample(files, 100)
    return (files,)


@app.cell
def _(embed, files):
    maybe_embeds = [embed(file) for file in mo.status.progress_bar(files)]
    return (maybe_embeds,)


@app.cell
def _(files, maybe_embeds):
    _valid_files, _valid_embeds = zip(
        *[
            (_f, _e)
            for _f, _e in zip(files, maybe_embeds)
            if _e is not None
        ]
    )
    valid_files = list(_valid_files)
    embeds = np.array(_valid_embeds)
    return embeds, valid_files


@app.cell
def _(embeds):
    projected = TSNE(n_components=2).fit_transform(embeds)
    return (projected,)


@app.cell
def _(projected, valid_files):
    # Extract metadata from file paths (assuming /home/stepo/Hudba/Artist/Album/Track.mp3)
    data = []
    for f, coords in zip(valid_files, projected):
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


@app.cell(expand_output=True)
def _(df_tsne):
    #artist_param = alt.param(
    #    name="artist_search",
    #    value="",
    #    bind=alt.binding(input="text", name="Filter Artist: "),
    #)

    chart = (
        alt.Chart(df_tsne)
        .mark_circle(size=25, opacity=0.5)
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
        #.add_params(artist_param)
        #.transform_filter("test(regexp(artist_search, 'i'), datum.artist)")
        .properties(
            title="t-SNE Projection of Audio Embeddings",
            width="container",
            height=750,
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
            for _, row in selected_tracks.head(10).iterrows() # Limit to 5 to avoid clutter
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
def _(embeds):
    similarity_scores = embeds[0:5] @ embeds.T
    return (similarity_scores,)


@app.cell
def _(df_tsne, similarity_scores):
    # Sort all candidate indices by similarity in descending order for each of the first 5 songs
    sorted_indices = np.argsort(similarity_scores, axis=1)[:, ::-1]

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

            score = similarity_scores[i, idx]
            similar_items.append(
                mo.vstack([
                    mo.md(f"**{rank}. {sim_song['artist']} - {sim_song['track']}** (*{sim_song['album']}*) — Similarity: `{score:.3f}`"),
                    mo.audio(sim_song['path'])
                ])
            )
            rank += 1
            if rank > 10:
                break

        tabs[f"{query_song['artist'][:15]} - {query_song['track'][:15]}"] = mo.vstack([query_info, *similar_items])

    mo.ui.tabs(tabs)
    return


if __name__ == "__main__":
    app.run()
