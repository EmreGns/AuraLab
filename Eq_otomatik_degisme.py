import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- 1. Spotify API Kimlik Bilgileri ---
# Buraya kendi Spotify Developer hesabından aldığın bilgileri koy
CLIENT_ID = "1e42ea17c1a44e86984bdfae1e581aac"
CLIENT_SECRET = "181786a66e27430eb216f53b642b25f8"
REDIRECT_URI = "http://127.0.0.1:8888/callback/"

scope = "user-read-playback-state user-read-currently-playing"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=scope
))

# --- 2. Çalan Şarkıyı Al ---
current = sp.current_playback()
if current and current['is_playing']:
    track_id = current['item']['id']
    track_name = current['item']['name']
    artist = current['item']['artists'][0]['name']
    print(f"Şu an çalan: {artist} - {track_name} (ID: {track_id})")
else:
    print("Şu anda Spotify'da şarkı çalmıyor.")
    exit()

# --- 3. Audio Features & Analysis ---
features = sp.audio_features(track_id)[0]
analysis = sp.audio_analysis(track_id)

print("\n--- Audio Features ---")
for key, value in features.items():
    print(f"{key}: {value}")

print("\n--- Audio Analysis (ilk segment) ---")
print(analysis['segments'][0])

# --- 4. Basit EQ Parametre Önerisi ---
eq_settings = {
    "bass": 0,
    "mid": 0,
    "treble": 0
}

# Bas enerjisi -> daha fazla bas
if features['energy'] > 0.7:
    eq_settings["bass"] += 3
elif features['energy'] < 0.4:
    eq_settings["bass"] -= 2

# Acousticness -> midrange öne çıkar
if features['acousticness'] > 0.5:
    eq_settings["mid"] += 2

# Valence (mutluluk) -> treble aç
if features['valence'] > 0.6:
    eq_settings["treble"] += 2

print("\n--- EQ Önerisi ---")
print(eq_settings)
