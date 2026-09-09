# Auto EQ Desktop MVP

Bu prototip Windows'taki seçilebilir bir sistem ses kaynağından (speaker loopback) sesi alır, 5 bantlı EQ'dan geçirerek seçilen Windows çıkış cihazına gönderir ve 2048 sample Hann window ile 20 Hz-20 kHz canlı spektrum gösterir. Masaüstü UI'si Auto/Manual EQ state'i ve presetler sağlar.

Auto EQ yalnızca VB-CABLE routing ile çalışır. Önce VB-Audio VB-CABLE kurun. Uygulama Windows default output'u geçici olarak `CABLE Input (VB-Audio Virtual Cable)` yapar ve `CABLE Output` üzerinden capture eder. Fiziksel cihazlar yalnızca Stereo Routing içindeki sol/sağ output olarak kullanılır; capture kaynağı yapılmaz.

PyQt6 tabanlı masaüstü arayüzü `auto_eq` paketindedir. Arayüz capture, EQ ve playback işini worker thread'de yürütür; Qt ana thread'i yalnızca 33 ms timer ile snapshot okur.

## Kurulum

```powershell
python -m pip install -r requirements.txt
```

## Çalıştırma

Önce cihazları görmek için:

```powershell
python -m spectrum_analyzer --list-devices
```

Ardından cihaz ID'si ile başlatın:

```powershell
python -m spectrum_analyzer --device 3
```

Masaüstü arayüzünü başlatmak için:

```powershell
python -m auto_eq
```

Uygulamayı kolayca başlatmak için proje kökündeki `start.bat` dosyasına çift tıklayabilirsiniz. Bu dosya `.venv` içindeki Python'u doğrudan kullanır; manuel `activate` gerekmez. Pencere kapatıldığında uygulama tray'de kalır. Tray menüsündeki `Exit and restore audio` Windows'un başlangıçtaki output, volume ve mute durumunu geri yükler. Crash sonrası `.auto_eq_audio_recovery.json` dosyası bir sonraki açılışta otomatik işlenir.

Uygulama açılışta mevcut Windows default output, volume ve mute değerlerini kaydeder; default output'u CABLE Input'a alır ve yalnızca fiziksel render cihazlarını Auto EQ output listesine koyar. Master Volume ve Mute, Windows taskbar ile aynı endpoint state'i üzerinden UI timer'ı ile senkronlanır. Gerçek Exit için tray menüsünü kullanın; pencereyi kapatmak uygulamayı tray'e küçültür.

Yeni arayüz sabit EQ kontrolü kullanır; eski spektruma göre otomatik gain hesaplaması kaldırılmıştır. Genre EQ değerleri `auto_eq/engine.py` içindeki merkezi `GENRE_PROFILES` tablosundan düzenlenir. Stereo Routing içinde sol/sağ speaker, speaker başına EQ, Swap L/R, master volume ve balance yönetilir. Ses backend'i Windows endpointleri üzerinden 48 kHz `sounddevice` callback akışıdır.

Spotify entegrasyonu isteğe bağlıdır. Kimlik bilgilerini dosyaya yazmadan ortam değişkenleriyle verin:

```powershell
$env:SPOTIFY_CLIENT_ID = "..."
$env:SPOTIFY_CLIENT_SECRET = "..."
$env:SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback/"
python -m auto_eq
```

Stereo Routing alanında sol ve sağ fiziksel speaker'ı seçin. `Swap L/R`, seçili cihaz adlarını ve ses kanallarını birlikte değiştirir.