import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import random
    import os
    import marimo as mo
    import pickle
    import pandas as pd
    import altair as alt
    import umap
    from pathlib import Path
    from pathlib import Path
    from hashlib import md5
    from sklearn.manifold import TSNE, MDS
    from sklearn.decomposition import PCA
    from glob import glob
    from essentia.standard import TensorflowPredictEffnetDiscogs, MonoLoader


@app.cell
def _():
    MODEL_FILE = "discogs-effnet-bs64-1.pb"
    SR = 16_000
    MAX_SEC = 60 
    return MAX_SEC, MODEL_FILE, SR


@app.cell
def _(MODEL_FILE):
    model = TensorflowPredictEffnetDiscogs(graphFilename=MODEL_FILE, output="PartitionedCall:1")
    return (model,)


@app.cell
def _():
    files = sorted(glob("/home/stepo/Hudba/*/*/*.mp3"))
    random.seed(42)
    random.shuffle(files)
    return (files,)


@app.cell
def _(MAX_SEC, SR, model):
    def extract_emb(file):
        cached_path = Path("cache") / ("effnet-" + md5(file.encode()).hexdigest())
        if cached_path.exists():
            with open(cached_path, "rb") as f:
                return pickle.load(f)

        audio = MonoLoader(filename=file, sampleRate=SR, resampleQuality=4)()
        n = MAX_SEC * SR
        if len(audio) > n:
            start = (len(audio) - n) // 2
            audio = audio[start:start + n]

        emb = model(audio).mean(axis=0)
        emb = emb / np.linalg.norm(emb)

        with open(cached_path, "wb") as f:
            pickle.dump(emb, f)

        return emb

    return (extract_emb,)


@app.cell
def _(extract_emb, files):
    embeddings = np.array(
        [extract_emb(file) for file in mo.status.progress_bar(files)]
    )
    return (embeddings,)


@app.cell
def _(embeddings, files):
    _album_groups: dict[tuple[str, str], list[int]] = {}
    for _idx, _file in enumerate(files):
        _path_obj = Path(_file)
        _album = _path_obj.parent.name
        _artist = _path_obj.parent.parent.name
        _album_groups.setdefault((_artist, _album), []).append(_idx)

    artists: list[str] = []
    albums: list[str] = []
    _embeds_list = []

    for (_artist, _album), _idxs in _album_groups.items():
        artists.append(_artist)
        albums.append(_album)
    
        # Calculate mean embedding of all songs in the album and L2-normalize it
        _mean_emb = embeddings[_idxs].mean(axis=0)
        _mean_emb = _mean_emb / np.linalg.norm(_mean_emb)
        _embeds_list.append(_mean_emb)

    album_embeds = np.array(_embeds_list)
    return album_embeds, albums, artists


@app.cell
def _(album_embeds):
    projected_albums = TSNE(n_components=2).fit_transform(album_embeds)
    return (projected_albums,)


@app.cell(expand_output=True)
def _(albums: list[str], artists: list[str], projected_albums):
    df_albums_tsne = pd.DataFrame({
        "x": projected_albums[:, 0],
        "y": projected_albums[:, 1],
        "artist": artists,
        "album": albums,
    })

    album_chart = (
        alt.Chart(df_albums_tsne)
        .mark_circle(size=50, opacity=0.7)
        .encode(
            x=alt.X("x:Q", title="t-SNE 1", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", title="t-SNE 2", scale=alt.Scale(zero=False)),
            color=alt.Color("artist:N", title="Artist"),
            tooltip=[
                alt.Tooltip("artist:N", title="Artist"),
                alt.Tooltip("album:N", title="Album"),
            ],
        )
        .properties(
            title="t-SNE Projection of Album Embeddings",
            width="container",
            height=750,
        )
    )

    album_tsne_plot = mo.ui.altair_chart(album_chart)
    album_tsne_plot
    return


@app.cell
def _(embeddings):
    projected = TSNE(n_components=2).fit_transform(embeddings)
    #projected = MDS(n_components=2).fit_transform(embeddings)
    #projected = PCA(n_components=2).fit_transform(embeddings)
    #projected = umap.UMAP(n_components=2).fit_transform(embeddings)
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


@app.cell(expand_output=True)
def _(df_tsne):
    #artist_param = alt.param(
    #    name="artist_search",
    #    value="",
    #    bind=alt.binding(input="text", name="Filter Artist: "),
    #)

    chart = (
        alt.Chart(df_tsne)
        .mark_circle(size=25, opacity=0.8)
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
        .interactive()
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
def _(embeddings):
    similarity_scores = embeddings[0:5] @ embeddings.T
    return (similarity_scores,)


@app.cell(expand_output=True)
def _(df_tsne, similarity_scores):
    # Sort all candidate indices by similarity in descending order for each of the first 5 songs
    sorted_indices = np.argsort(similarity_scores, axis=1)#[:, ::-1]

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
