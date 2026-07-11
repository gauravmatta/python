"""
Fetch song metadata from the Genius API for a list of songs and write results to a file.

Setup:
1. Get a free API access token: https://genius.com/api-clients
2. Set it as an environment variable before running:
       export GENIUS_ACCESS_TOKEN="your_token_here"
3. Put your song names in a text file, one per line (e.g. songs.txt)

Note: The Genius API returns metadata only (title, artist, album, release date,
song URL, thumbnail, etc.) -- it does not return full lyrics text via the API.
"""

import os
import time
import json
import difflib
import requests

GENIUS_API_BASE = "https://api.genius.com"


def _similarity(a: str, b: str) -> float:
    """Return a 0-1 similarity score between two strings (case-insensitive)."""
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


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
        "title": best.get("title"),
        "primary_artist": best.get("primary_artist", {}).get("name"),
        "release_date": best.get("release_date_for_display"),
        "genius_url": best.get("url"),
        "thumbnail_url": best.get("song_art_image_thumbnail_url"),
        "stats": best.get("stats", {}),
    }


def fetch_all_songs(
    input_file: str,
    output_file: str,
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

    songs = []
    for line in raw_lines:
        if "," in line:
            name, artist = [part.strip() for part in line.split(",", 1)]
        else:
            name, artist = line, ""
        songs.append((name, artist))

    results = []
    for i, (song, artist) in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}] Fetching: {song} ({artist or 'no artist'})")
        info = get_song_info(song, access_token, artist=artist, debug=True)
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