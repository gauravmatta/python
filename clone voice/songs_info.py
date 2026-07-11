"""
Fetch song metadata from the Genius API for a list of songs and write results to a file.

Setup:
1. Get a free API access token: https://genius.com/api-clients
2. Set it as an environment variable before running:
       export GENIUS_ACCESS_TOKEN="your_token_here"
3. Put your song names in a text file, one per line (e.g. songs.txt)

For each song, this fetches:
  - Basic metadata (title, artist, release date, Genius URL, thumbnail, popularity stats)
  - A description written by Genius contributors (background on the song, its meaning, etc.)
  - The album -- which for film songs is usually the movie's soundtrack name
  - Writer, producer, and featured artist credits

Note: This does NOT return full lyrics text -- Genius doesn't expose lyrics via
the API, and reproducing copyrighted lyrics isn't something this script does.
"""

import os
import time
import json
import difflib
import requests
from pathlib import Path

GENIUS_API_BASE = "https://api.genius.com"


def _similarity(a: str, b: str) -> float:
    """Return a 0-1 similarity score between two strings (case-insensitive)."""
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _extract_description(song_detail: dict) -> str:
    """
    Genius stores the song description as a nested rich-text document
    (a list of DOM-like nodes) rather than a plain string. This walks that
    structure and concatenates the plain text parts into a single string.
    """
    desc = song_detail.get("description", {})
    dom = desc.get("dom")
    if not dom:
        return ""

    parts = []

    def walk(node):
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, dict):
            for child in node.get("children", []):
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(dom)
    text = " ".join(p.strip() for p in parts if p and p.strip())
    return text


def get_song_details(song_id: int, access_token: str) -> dict:
    """
    Fetch the full song detail page for richer info: description, album
    (often the movie name for film songs), writer/producer credits, and
    featured artists.

    Args:
        song_id: The Genius song ID (from a prior search result's "id" field).
        access_token: Your Genius API access token.

    Returns:
        A dict with description, album/movie, writers, producers, and
        featured artists. Empty/default values if the detail call fails.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        resp = requests.get(f"{GENIUS_API_BASE}/songs/{song_id}", headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return {}

    song = resp.json().get("response", {}).get("song", {})

    album = song.get("album")
    album_name = album.get("name") if album else None

    return {
        "description": _extract_description(song),
        "album_or_movie": album_name,
        "writer_artists": [a.get("name") for a in song.get("writer_artists", [])],
        "producer_artists": [a.get("name") for a in song.get("producer_artists", [])],
        "featured_artists": [a.get("name") for a in song.get("featured_artists", [])],
        "release_date_full": song.get("release_date"),
    }



def get_song_info(
    song_name: str,
    access_token: str,
    artist: str = "",
    title_threshold: float = 0.5,
    debug: bool = False,
) -> dict:
    """
    Search Genius for a song and return its metadata.

    Args:
        song_name: The song title to search for.
        access_token: Your Genius API access token.
        artist: The song's actual performing artist. Strongly recommended --
                Genius's Hindi/Bollywood catalog is patchy and full of
                same-titled songs from different years/films, so artist is
                the main disambiguator.
        title_threshold: Minimum title similarity (0-1) required to accept a
                candidate. Prevents matching wildly unrelated pages (movie
                scripts, playlists) just because they contain the query word.
        debug: If True, prints all candidate hits so you can see what
               Genius actually returned.

    Returns:
        A dict of metadata, or a dict with an "error" key if no confident
        match was found (better than silently returning a wrong song).
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    query = f"{song_name} {artist}".strip()
    params = {"q": query}

    try:
        resp = requests.get(f"{GENIUS_API_BASE}/search", headers=headers, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"query": song_name, "error": str(e)}

    hits = resp.json().get("response", {}).get("hits", [])
    if not hits:
        return {"query": song_name, "error": "No results found"}

    candidates = [h["result"] for h in hits]

    scored = []
    for c in candidates:
        title = c.get("title", "")
        primary = (c.get("primary_artist", {}).get("name") or "")
        title_score = _similarity(song_name, title)
        artist_score = _similarity(artist, primary) if artist else 0.0
        # Weight artist match heavily when we have one to check against
        combined = title_score + (artist_score * 1.5 if artist else 0)
        scored.append((combined, title_score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    if debug:
        print(f"\n--- Candidates for '{query}' (sorted by match score) ---")
        for combined, title_score, c in scored:
            print(f"  [{combined:.2f}] {c.get('title')} — "
                  f"{c.get('primary_artist', {}).get('name')} "
                  f"({c.get('release_date_for_display')})")

    best_combined, best_title_score, best = scored[0]

    # Refuse to guess if even the best candidate doesn't look like the right song
    if best_title_score < title_threshold:
        return {
            "query": song_name,
            "error": f"No confident match (best title similarity {best_title_score:.2f}, "
                     f"closest was '{best.get('title')}' by "
                     f"{best.get('primary_artist', {}).get('name')})",
        }

    return {
        "query": song_name,
        "id": best.get("id"),
        "title": best.get("title"),
        "primary_artist": best.get("primary_artist", {}).get("name"),
        "release_date": best.get("release_date_for_display"),
        "genius_url": best.get("url"),
        "thumbnail_url": best.get("song_art_image_thumbnail_url"),
        "stats": best.get("stats", {}),
    }


def _parse_song_line(line: str) -> tuple[str, str]:
    """Return (name, artist) from a 'Name, Artist' or plain 'Name' line."""
    if "," in line:
        name, artist = line.split(",", 1)
        return name.strip(), artist.strip()
    return line, ""


def fetch_all_songs(
    input_file: str | Path,
    output_file: str | Path,
    access_token: str,
    delay: float = 0.5,
) -> None:
    """
    Read song names from input_file (one per line), fetch metadata for each,
    and write the results as JSON to output_file.

    Each line in input_file can optionally include the artist, comma-separated:
        Saiyaara, Faheem Abdullah
        Barbaad
    Including the artist is strongly recommended -- Genius's Hindi/Bollywood
    search coverage is inconsistent, and without an artist to disambiguate,
    generic titles often match unrelated compilation pages instead of
    returning "no match".

    Args:
        input_file: Path to a text file with one song name (and optional artist) per line.
        output_file: Path to write the JSON results.
        access_token: Your Genius API access token.
        delay: Seconds to wait between requests (be polite to the API).
    """
    with open(input_file, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    results = []
    for i, line in enumerate(raw_lines, 1):
        song, artist = _parse_song_line(line)
        print(f"[{i}/{len(raw_lines)}] Fetching: {song} ({artist or 'no artist'})")
        info = get_song_info(song, access_token, artist=artist, debug=True)

        if "id" in info and info["id"]:
            time.sleep(delay)
            details = get_song_details(info["id"], access_token)
            info.update(details)

        results.append(info)
        time.sleep(delay)  # avoid hammering the API

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Wrote {len(results)} entries to {output_file}")


if __name__ == "__main__":
    TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not TOKEN:
        raise SystemExit(
            "Missing GENIUS_ACCESS_TOKEN environment variable.\n"
            "Get a token at https://genius.com/api-clients and run:\n"
            '  export GENIUS_ACCESS_TOKEN="your_token_here"'
        )

    fetch_all_songs(
        input_file="sample/text/songs.txt",
        output_file="sample/text/songs_info.json",
        access_token=TOKEN,
    )