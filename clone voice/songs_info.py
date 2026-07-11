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
import requests

GENIUS_API_BASE = "https://api.genius.com"


def get_song_info(song_name: str, access_token: str, artist_hint: str = "Bollywood", year: str = "") -> dict:
    """
    Search Genius for a song and return its metadata.

    Args:
        song_name: The song title to search for.
        access_token: Your Genius API access token.
        artist_hint: Extra keyword appended to the search query to bias results
                     (e.g. "Bollywood") toward the right match.
        year: Optional release year to further narrow the search
              (helps a lot for generic/one-word titles).

    Returns:
        A dict of metadata, or a dict with an "error" key if not found.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    query = f"{song_name} {artist_hint} {year}".strip()
    params = {"q": query}

    try:
        resp = requests.get(f"{GENIUS_API_BASE}/search", headers=headers, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"query": song_name, "error": str(e)}

    hits = resp.json().get("response", {}).get("hits", [])
    if not hits:
        return {"query": song_name, "error": "No results found"}

    # Take the top hit
    result = hits[0]["result"]

    return {
        "query": song_name,
        "title": result.get("title"),
        "primary_artist": result.get("primary_artist", {}).get("name"),
        "release_date": result.get("release_date_for_display"),
        "genius_url": result.get("url"),
        "thumbnail_url": result.get("song_art_image_thumbnail_url"),
        "stats": result.get("stats", {}),
    }


def fetch_all_songs(
    input_file: str,
    output_file: str,
    access_token: str,
    delay: float = 0.5,
    default_year: str = "",
) -> None:
    """
    Read song names from input_file (one per line), fetch metadata for each,
    and write the results as JSON to output_file.

    Each line in input_file can optionally include a year, comma-separated:
        Saiyaara, 2025
        Barbaad, 2025
        Shaky
    If a line has no year, default_year is used instead.

    Args:
        input_file: Path to a text file with one song name (and optional year) per line.
        output_file: Path to write the JSON results.
        access_token: Your Genius API access token.
        delay: Seconds to wait between requests (be polite to the API).
        default_year: Year to use for lines that don't specify one.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    songs = []
    for line in raw_lines:
        if "," in line:
            name, year = [part.strip() for part in line.split(",", 1)]
        else:
            name, year = line, default_year
        songs.append((name, year))

    results = []
    for i, (song, year) in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}] Fetching: {song} ({year or 'no year'})")
        info = get_song_info(song, access_token, year=year)
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
        default_year="2025",
    )