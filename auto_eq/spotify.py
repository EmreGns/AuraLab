"""Optional Spotify metadata integration without embedded credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SpotifyTrack:
    track_id: str
    title: str
    artist: str
    album: str
    genre: str | None = None


class SpotifyIntegration:
    def __init__(self) -> None:
        self._client = None
        self._last_track_id: str | None = None

    @property
    def available(self) -> bool:
        return bool(os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET"))

    def connect(self) -> bool:
        if not self.available:
            return False
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth

            self._client = spotipy.Spotify(
                auth_manager=SpotifyOAuth(
                    client_id=os.environ["SPOTIFY_CLIENT_ID"],
                    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
                    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback/"),
                    scope="user-read-playback-state user-read-currently-playing",
                )
            )
        except Exception:
            self._client = None
        return self._client is not None

    def current_track(self) -> SpotifyTrack | None:
        if self._client is None and not self.connect():
            return None
        try:
            item = self._client.current_user_playing_track().get("item")
            if not item:
                return None
            artists = item.get("artists", [])
            artist_id = artists[0].get("id") if artists else None
            genres: list[str] = []
            if artist_id:
                artist = self._client.artist(artist_id)
                genres = artist.get("genres", [])
            return SpotifyTrack(
                track_id=item["id"],
                genre=genres[0] if genres else None,
                title=item["name"],
                artist=artists[0]["name"] if artists else "",
                album=item.get("album", {}).get("name", ""),
            )
        except Exception:
            return None
