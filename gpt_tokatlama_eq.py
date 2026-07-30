import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pyaudio
import numpy as np
import threading
import time
import sys

# Spotify API Bilgileri
CLIENT_ID = "1e42ea17c1a44e86984bdfae1e581aac"
CLIENT_SECRET = "181786a66e27430eb216f53b642b25f8"
REDIRECT_URI = "http://127.0.0.1:8888/callback/"

class SpotifyAutoEQ:
    def __init__(self, client_id, client_secret, redirect_uri):
        # Spotify API setup
        scope = "user-read-playback-state user-read-currently-playing"
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope
        ))
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32  # Daha iyi ses kalitesi için float32
        self.CHANNELS = 2  # Varsayılan kanal sayısı
        self.RATE = 44100
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # EQ parameters
        self.bass_gain = 0.0
        self.mid_gain = 0.0
        self.treble_gain = 0.0
        
        # Filter memory for biquad filters (stereo)
        self.bass_x1 = np.zeros(2)
        self.bass_x2 = np.zeros(2)
        self.bass_y1 = np.zeros(2)
        self.bass_y2 = np.zeros(2)
        
        self.mid_x1 = np.zeros(2)
        self.mid_x2 = np.zeros(2)
        self.mid_y1 = np.zeros(2)
        self.mid_y2 = np.zeros(2)
        
        self.treble_x1 = np.zeros(2)
        self.treble_x2 = np.zeros(2)
        self.treble_y1 = np.zeros(2)
        self.treble_y2 = np.zeros(2)
        
        # Control variables
        self.previous_track_id = None
        self.eq_active = False
        
        print("Kullanılabilir ses cihazları:")
        self.list_audio_devices()

    def list_audio_devices(self):
        """Kullanılabilir ses cihazlarını listele"""
        print("-" * 80)
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                print(f"ID {i:2d}: {device_info['name']}")
                print(f"       Max Input: {device_info['maxInputChannels']}, Max Output: {device_info['maxOutputChannels']}")
                if 'cable' in device_info['name'].lower():
                    print("       *** VB-CABLE CİHAZI ***")
                print()
            except:
                pass
        print("-" * 80)

    def find_vb_cable_devices(self):
        """VB-Cable cihazlarını otomatik bul"""
        input_device = None
        output_device = None
        default_output = None
        
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                device_name = device_info['name'].lower()
                
                # VB-Cable Output (sistem sesini dinlemek için)
                # Cable Output yerine System Virtual Line'ı tercih et
                if device_info['maxInputChannels'] > 0:
                    if 'system virtual line' in device_name:
                        input_device = i
                        print(f"Giriş cihazı bulundu (System Virtual): {device_info['name']} (ID: {i}, Channels: {device_info['maxInputChannels']})")
                    elif 'cable output' in device_name and input_device is None:
                        input_device = i
                        print(f"VB-Cable giriş bulundu: {device_info['name']} (ID: {i}, Channels: {device_info['maxInputChannels']})")
                
                # Çıkış cihazı öncelikleri
                if device_info['maxOutputChannels'] > 0:
                    device_desc = (f"{device_info['name']} (ID: {i}, "
                                f"Rate: {int(device_info['defaultSampleRate'])}, "
                                f"Channels: {device_info['maxOutputChannels']})")
                    
                    # 1. Öncelik: Line Out (Intelligo)
                    if 'line out' in device_name and default_output is None:
                        default_output = i
                        print(f"Çıkış cihazı bulundu (Line Out): {device_desc}")
                    # 2. Öncelik: Emre Buds FE
                    elif 'emre buds fe' in device_name and default_output is None:
                        default_output = i
                        print(f"Çıkış cihazı bulundu (Emre Buds): {device_desc}")
                    # 3. Öncelik: Diğer cihazlar
                    elif default_output is None and ('speakers' in device_name or 'headphones' in device_name or 'realtek' in device_name):
                        default_output = i
                        print(f"Alternatif çıkış cihazı bulundu: {device_desc}")
                            
            except Exception as e:
                print(f"Cihaz bilgisi alınırken hata: {str(e)}")
                continue
                
        return input_device, default_output

    def calculate_eq_from_features(self, features):
        """Audio features'dan EQ parametreleri hesapla"""
        bass = 0.0
        mid = 0.0
        treble = 0.0
        
        # Energy -> Bas kontrolü
        energy = features.get('energy', 0.5)
        if energy > 0.8:
            bass += 6.0
        elif energy > 0.6:
            bass += 3.0
        elif energy < 0.3:
            bass -= 3.0
            
        # Danceability -> Bas güçlendirme
        danceability = features.get('danceability', 0.5)
        if danceability > 0.8:
            bass += 4.0
        elif danceability > 0.6:
            bass += 2.0
            
        # Acousticness -> Mid-range ayarı
        acousticness = features.get('acousticness', 0.5)
        if acousticness > 0.7:
            mid += 4.0
        elif acousticness > 0.4:
            mid += 2.0
        elif acousticness < 0.1:
            mid -= 2.0
            
        # Valence -> Treble ayarı (mutlu şarkılar daha parlak)
        valence = features.get('valence', 0.5)
        if valence > 0.8:
            treble += 3.0
        elif valence > 0.6:
            treble += 1.5
        elif valence < 0.3:
            treble -= 2.0
            
        # Loudness kompensasyonu
        loudness = features.get('loudness', -10)
        if loudness < -15:
            bass += 3.0
            mid += 2.0
            treble += 2.0
        elif loudness > -5:
            bass -= 2.0
            mid -= 1.0
            
        # Instrumentalness -> Mid azaltma
        instrumentalness = features.get('instrumentalness', 0)
        if instrumentalness > 0.5:
            mid -= 1.0
            
        # Sınırlandır
        bass = max(-12, min(12, bass))
        mid = max(-8, min(8, mid))
        treble = max(-8, min(8, treble))
        
        return bass, mid, treble

    def biquad_coefficients(self, freq, gain, q=0.7):
        """Biquad peaking EQ katsayılarını hesapla"""
        if abs(gain) < 0.1:
            return 1, 0, 0, 0, 0  # Bypass
            
        w = 2.0 * np.pi * freq / self.RATE
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        A = 10.0 ** (gain / 40.0)
        alpha = sin_w / (2.0 * q)
        
        # Peaking EQ formülü
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha / A
        
        # Normalizasyon
        return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0

    def apply_biquad(self, sample, b0, b1, b2, a1, a2, x1, x2, y1, y2, channel):
        """Tek bir biquad filtresi uygula"""
        # Direct Form II
        output = b0 * sample + b1 * x1[channel] + b2 * x2[channel] - a1 * y1[channel] - a2 * y2[channel]
        
        # Delay line güncelle
        x2[channel] = x1[channel]
        x1[channel] = sample
        y2[channel] = y1[channel]
        y1[channel] = output
        
        return output

    def process_audio_chunk(self, audio_data):
        """Ses parçacığına EQ uygula"""
        # Byte veriden int16 array'e çevir
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Float'a çevir ve normalize et
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Stereo kanallara ayır
        left = audio_float[0::2].copy()
        right = audio_float[1::2].copy()
        
        # EQ filtrelerini uygula
        if abs(self.bass_gain) > 0.1:
            b0, b1, b2, a1, a2 = self.biquad_coefficients(80, self.bass_gain, 0.7)
            for i in range(len(left)):
                left[i] = self.apply_biquad(left[i], b0, b1, b2, a1, a2, 
                                          self.bass_x1, self.bass_x2, self.bass_y1, self.bass_y2, 0)
                right[i] = self.apply_biquad(right[i], b0, b1, b2, a1, a2,
                                           self.bass_x1, self.bass_x2, self.bass_y1, self.bass_y2, 1)
                
        if abs(self.mid_gain) > 0.1:
            b0, b1, b2, a1, a2 = self.biquad_coefficients(1000, self.mid_gain, 0.7)
            for i in range(len(left)):
                left[i] = self.apply_biquad(left[i], b0, b1, b2, a1, a2,
                                          self.mid_x1, self.mid_x2, self.mid_y1, self.mid_y2, 0)
                right[i] = self.apply_biquad(right[i], b0, b1, b2, a1, a2,
                                           self.mid_x1, self.mid_x2, self.mid_y1, self.mid_y2, 1)
                
        if abs(self.treble_gain) > 0.1:
            b0, b1, b2, a1, a2 = self.biquad_coefficients(8000, self.treble_gain, 0.7)
            for i in range(len(left)):
                left[i] = self.apply_biquad(left[i], b0, b1, b2, a1, a2,
                                          self.treble_x1, self.treble_x2, self.treble_y1, self.treble_y2, 0)
                right[i] = self.apply_biquad(right[i], b0, b1, b2, a1, a2,
                                           self.treble_x1, self.treble_x2, self.treble_y1, self.treble_y2, 1)
        
        # Stereo tekrar birleştir
        output_float = np.empty(len(audio_float), dtype=np.float32)
        output_float[0::2] = left
        output_float[1::2] = right
        
        # Clipping kontrolü
        output_float = np.clip(output_float, -1.0, 1.0)
        
        # Int16'ya geri çevir
        output_int16 = (output_float * 32767).astype(np.int16)
        
        return output_int16.tobytes()

    def spotify_monitor(self):
        """Spotify şarkı değişikliklerini izle"""
        print("Spotify izleme başlatıldı...")
        print("🔄 Spotify bağlantısı kontrol ediliyor...")
        
        try:
            # İlk başta bağlantıyı test et
            current = self.sp.current_playback()
            if current is None:
                print("❌ Spotify'da aktif bir oynatma bulunamadı!")
                print("   1. Spotify'ı açın ve bir şarkı çalın")
                print("   2. Spotify'da 'Şarkıyı Paylaş > Cihaza Bağlan' seçeneğini kullanın")
                print("   3. Bu bilgisayarı seçin")
            else:
                print("✅ Spotify bağlantısı başarılı!")
        except Exception as e:
            print(f"❌ Spotify bağlantı hatası: {e}")
            print("   1. Spotify'ın açık olduğundan emin olun")
            print("   2. Spotify Premium hesabınızın aktif olduğunu kontrol edin")
            print("   3. İnternet bağlantınızı kontrol edin")
        
        while self.eq_active:
            try:
                current = self.sp.current_playback()
                
                if current and current['is_playing']:
                    current_track_id = current['item']['id']
                    track_name = current['item']['name']
                    artist_name = current['item']['artists'][0]['name']
                    
                    # Her zaman mevcut şarkıyı göster
                    if current_track_id != self.previous_track_id:
                        print(f"\n🎵 Yeni şarkı: {artist_name} - {track_name}")
                        
                        # Audio features al
                        try:
                            features = self.sp.audio_features(current_track_id)[0]
                            
                            if features:
                                # EQ parametreleri hesapla
                                bass, mid, treble = self.calculate_eq_from_features(features)
                                
                                # EQ güncelle
                                self.bass_gain = bass
                                self.mid_gain = mid
                                self.treble_gain = treble
                                
                                # Filtre hafızasını sıfırla
                                self.clear_filter_memory()
                                
                                print(f"🎛️  EQ Güncellendi:")
                                print(f"   Bas: {bass:+.1f}dB | Mid: {mid:+.1f}dB | Treble: {treble:+.1f}dB")
                                print(f"   Energy: {features.get('energy', 0):.2f} | Valence: {features.get('valence', 0):.2f} | Danceability: {features.get('danceability', 0):.2f}")
                            
                            self.previous_track_id = current_track_id
                            
                        except Exception as e:
                            print(f"⚠️ Audio features alınamadı: {e}")
                    else:
                        # Her 30 saniyede bir çalan şarkıyı göster
                        if int(time.time()) % 30 == 0:
                            print(f"🎵 Çalıyor: {artist_name} - {track_name}")
                            
                else:
                    if self.previous_track_id is not None:
                        print("⏸️ Müzik durdu.")
                        self.previous_track_id = None
                    elif int(time.time()) % 10 == 0:  # Her 10 saniyede bir hatırlat
                        print("💭 Spotify'da müzik çalmıyor...")
                        
            except spotipy.exceptions.SpotifyException as e:
                print(f"❌ Spotify API hatası: {e}")
                if "The access token expired" in str(e):
                    print("🔄 Token yenileniyor...")
                    self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                        client_id=self.client_id,
                        client_secret=self.client_secret,
                        redirect_uri=self.redirect_uri,
                        scope=self.scope
                    ))
            except Exception as e:
                print(f"⚠️ Beklenmeyen hata: {e}")
                
            time.sleep(1)  # Daha sık kontrol et

    def clear_filter_memory(self):
        """Filtre hafızasını temizle (pop sesleri önlemek için)"""
        self.bass_x1.fill(0)
        self.bass_x2.fill(0)
        self.bass_y1.fill(0)
        self.bass_y2.fill(0)
        
        self.mid_x1.fill(0)
        self.mid_x2.fill(0)
        self.mid_y1.fill(0)
        self.mid_y2.fill(0)
        
        self.treble_x1.fill(0)
        self.treble_x2.fill(0)
        self.treble_y1.fill(0)
        self.treble_y2.fill(0)

    def start_eq(self, input_device=None, output_device=None):
        """EQ sistemini başlat"""
        # Cihazları otomatik bul
        if input_device is None or output_device is None:
            auto_input, auto_output = self.find_vb_cable_devices()
            if input_device is None:
                input_device = auto_input
            if output_device is None:
                output_device = auto_output
        
        if input_device is None:
            print("\n❌ HATA: VB-Cable Output bulunamadı!")
            print("Kurulum kontrol listesi:")
            print("1. VB-Cable kuruldu mu?")
            print("2. Windows Sound Settings'de VB-Cable Input varsayılan çıkış olarak seçildi mi?")
            print("3. Bilgisayar yeniden başlatıldı mı?")
            return False
            
        if output_device is None:
            print("\n❌ HATA: Çıkış cihazı bulunamadı!")
            print("Manuel olarak çıkış cihazı ID'sini belirtin.")
            return False
            
        # Cihaz özelliklerini kontrol et
        try:
            input_info = self.audio.get_device_info_by_index(input_device)
            output_info = self.audio.get_device_info_by_index(output_device)
            
            print("\n🔍 Seçilen cihaz özellikleri:")
            print(f"Giriş: {input_info['name']}")
            print(f"- Sample Rate: {int(input_info['defaultSampleRate'])}Hz")
            print(f"- Channels: {input_info['maxInputChannels']}")
            
            print(f"\nÇıkış: {output_info['name']}")
            print(f"- Sample Rate: {int(output_info['defaultSampleRate'])}Hz")
            print(f"- Channels: {output_info['maxOutputChannels']}")
            
            # Sample rate uyumsuzluğu kontrolü
            if int(input_info['defaultSampleRate']) != int(output_info['defaultSampleRate']):
                print("\n⚠️ UYARI: Sample rate uyumsuzluğu tespit edildi!")
                print(f"Giriş: {int(input_info['defaultSampleRate'])}Hz")
                print(f"Çıkış: {int(output_info['defaultSampleRate'])}Hz")
                print("Bu durum ses sorunlarına neden olabilir.")
                
        except Exception as e:
            print(f"\n⚠️ UYARI: Cihaz bilgileri alınamadı: {str(e)}")
        
        print(f"\n✅ Kullanılacak cihazlar:")
        print(f"   Giriş (VB-Cable): ID {input_device}")
        print(f"   Çıkış (Hoparlör): ID {output_device}")
        
        # Ses akışlarını aç
        try:
            # Cihaz özelliklerini al
            input_info = self.audio.get_device_info_by_index(input_device)
            output_info = self.audio.get_device_info_by_index(output_device)
            
            # En uygun yapılandırmayı seç
            input_channels = min(2, input_info['maxInputChannels'])
            output_channels = min(2, output_info['maxOutputChannels'])
            
            # Sample rate uyumluluğu için en uygun değeri seç
            input_rate = int(input_info['defaultSampleRate'])
            output_rate = int(output_info['defaultSampleRate'])
            common_rate = min(input_rate, output_rate, self.RATE)
            
            print(f"\n🎛️ Ses yapılandırması:")
            print(f"Giriş: {input_channels} kanal, {input_rate}Hz")
            print(f"Çıkış: {output_channels} kanal, {output_rate}Hz")
            print(f"Kullanılacak: {input_channels} kanal, {common_rate}Hz")
            
            # Giriş akışı
            try:
                input_stream = self.audio.open(
                    format=self.FORMAT,
                    channels=input_channels,
                    rate=common_rate,
                    input=True,
                    input_device_index=input_device,
                    frames_per_buffer=self.CHUNK,
                    stream_callback=None
                )
            except Exception as e:
                print(f"\n❌ Giriş akışı açılamadı: {e}")
                print("🔍 Farklı bir giriş cihazı deneyin (manuel mod)")
                return False
            
            # Çıkış akışı
            try:
                output_stream = self.audio.open(
                    format=self.FORMAT,
                    channels=output_channels,
                    rate=common_rate,
                    output=True,
                    output_device_index=output_device,
                    frames_per_buffer=self.CHUNK,
                    stream_callback=None
                )
            except Exception as e:
                print(f"\n❌ Çıkış akışı açılamadı: {e}")
                print("🔍 Farklı bir çıkış cihazı deneyin (manuel mod)")
                input_stream.close()
                return False
            
            self.eq_active = True
            
            # Spotify izleme thread'ini başlat
            spotify_thread = threading.Thread(target=self.spotify_monitor, daemon=True)
            spotify_thread.start()
            
            print("\n🎧 Spotify Auto EQ aktif!")
            print("📱 Spotify'dan müzik çalmaya başlayın.")
            print("⏹️  Durdurmak için Ctrl+C basın.\n")
            
            # Ana ses işleme döngüsü
            while self.eq_active:
                try:
                    # VB-Cable'dan ses oku
                    audio_data = input_stream.read(self.CHUNK, exception_on_overflow=False)
                    
                    # EQ uygula
                    processed_audio = self.process_audio_chunk(audio_data)
                    
                    # Hoparlöre yaz
                    output_stream.write(processed_audio)
                    
                except Exception as e:
                    if "Input overflowed" not in str(e):
                        print(f"Ses işleme hatası: {e}")
                        
        except KeyboardInterrupt:
            print("\n🛑 EQ durduruldu.")
        except Exception as e:
            print(f"\n❌ Ses akışı hatası: {e}")
            print("Cihaz ID'lerini kontrol edin veya manuel olarak belirtin.")
        finally:
            self.eq_active = False
            try:
                input_stream.stop_stream()
                input_stream.close()
                output_stream.stop_stream() 
                output_stream.close()
            except:
                pass
            self.audio.terminate()
            
        return True

def main():
    print("🎵 Spotify Otomatik EQ v1.0")
    print("=" * 50)
    
    try:
        # EQ sistemini başlat
        eq_system = SpotifyAutoEQ(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
        
        # Manuel cihaz seçimi seçeneği
        choice = input("\nOtomatik cihaz seçimi kullanılsın mı? (Y/n): ").strip().lower()
        
        if choice == 'n':
            input_id = input("VB-Cable Output cihaz ID'si: ")
            output_id = input("Çıkış cihazı ID'si: ")
            eq_system.start_eq(int(input_id), int(output_id))
        else:
            eq_system.start_eq()
            
    except Exception as e:
        print(f"❌ Başlatma hatası: {e}")
        print("\nOlası çözümler:")
        print("1. Spotify API bilgilerinizi kontrol edin")
        print("2. VB-Cable kurulumunu kontrol edin")
        print("3. Python paketlerinin yüklü olduğundan emin olun")

if __name__ == "__main__":
    main()