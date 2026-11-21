import csv
import sys

from smart_playlist import connect_mpd


def tag(song: dict, name: str) -> str:
    value = song.get(name, "")
    return "; ".join(value) if isinstance(value, list) else value


def main(out_path: str = "mpd_dump.csv") -> None:
    client = connect_mpd()
    try:
        songs = [entry for entry in client.listallinfo() if "file" in entry]
    finally:
        client.disconnect()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["artist", "album", "track", "path"])
        for song in songs:
            writer.writerow([
                tag(song, "artist"),
                tag(song, "album"),
                tag(song, "track"),
                song["file"],
            ])
    print(f"Wrote {len(songs)} songs to {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
