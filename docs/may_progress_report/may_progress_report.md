# Bitirme Projesi Mayıs İlerleme Raporu

## 1. Mart'tan Bu Yana Yapılan Genel İyileştirmeler

Mart raporunun ardından proje üç ana doğrultuda derinleştirilmiştir:

1. Web uygulamasının ses kalitesi ve detection güvenilirliği,
2. Veri seti genişletme ve eğitim pipeline iyileştirmeleri,
3. Kod tabanının yeniden organize edilmesi ve isimlendirme standartlarının
   oturtulması.

---

## 2. Web Uygulaması (Gradio) İyileştirmeleri

### 2.1. MP4 / Video Dosyası Crash Düzeltmesi

WhatsApp gibi uygulamalardan gelen ses-only `.mp4` dosyaları, ffmpeg muxing
aşamasında `Stream map '0:v:0' matches no streams` hatasıyla çöküyordu.

**Çözüm:** Muxing öncesi `ffprobe` ile video stream varlığı kontrol ediliyor;
video yoksa yalnızca temizlenmiş ses döndürülüyor.

### 2.2. Hatalı Sınıf Tespiti (False Positive Detection)

**Sorun:** Eski detection sadece *bağıl* eşik (`0.20 × kazanan`) kullanıyordu.
Geniş bantlı (broadband) gürültü içeren girişlerde tüm sınıflar benzer enerji
üretiyor, ekrana 10–15 alakasız sınıf çıkıyordu.

**Çözüm:** İki bileşenli yeni puan:

```
score = energy_ratio × (1 + std(mask) / mean(mask))
```

Maskenin Coefficient of Variation (CoV) bileşeni, yoğunlaştırılmış maskeleri
ödüllendiriyor; broadband gürültünün ürettiği düz maskeleri cezalandırıyor.
Buna ek olarak:

- **Mutlak alt sınır:** `0.30` — zayıf bir kazanan varken çöp sınıfların
  sızmasını engelliyor.
- **Bağıl tavan:** `0.40 × winner` — aralığı dar tutuyor.

### 2.3. Aşırı Kaldırma (Over-Removal) ve Su Altı Sesi

**Sorun:** Doğrudan lineer çıkarma (`mix - α·est`) spektrumda delikler
oluşturuyor, sonuç boğuk / su altında gibi duyuluyordu.

**Çözüm:** Power-domain ratio maskeleme:

```
mask = clip(1 − strength · (est² / mix²), 0, 1)
```

Birden çok sınıf seçildiğinde maskeler çarpımsal birleştiriliyor
(`combined *= mask_k`). Bu, deliği engelleyen sınırlı bir bastırma sağlar.

### 2.4. Click ve Sızıntı (Window Boundary Artifacts)

**Sorun:** Her 1 sn'lik pencere için ayrı STFT/ISTFT alınıyor, faz pencere
başı sıfırlanıyor, sınırlarda click duyuluyordu.

**Çözüm: Spectrogram-level Overlap-Add (OLA):**

- Tüm dosya için tek bir `librosa.stft` → faz globalde tutarlı.
- `TIME_FRAMES // 2 = 64` frame'lik adımla pencereleme.
- Hanning ağırlıklı akümülasyon.
- Tek bir `librosa.istft` ile orijinal faz geri kullanılıyor.
- Ek olarak, maskelere zaman ekseninde 5-frame (~40 ms) konvolüsyonel
  düzleştirme uygulanıyor — bu da musical noise'u bastırıyor.

---

## 3. Eğitim Pipeline'ı — v2.0 İyileştirmeleri

### 3.1. Negatif Örnek Oranı Artırıldı (negative_prob)

- **Önce:** `negative_prob = 0.25` — model eğitimin yalnızca %25'inde yok
  olan sınıf sorgusu görüyordu, yokluk tespiti zayıftı.
- **Sonra:** `negative_prob = 0.45` — kafe gürültüsü gibi sahnelerde
  alakasız sınıflar için maskenin sıfıra inmesini güçlendirir.

### 3.2. Background Noise Augmentation

Eğitim seti yalnızca temiz stüdyo kayıtlarından üretildiğinden, gerçek dünya
(kafe, trafik, HVAC) girişlerinde domain gap belirginleşmişti.

**Eklenen:** `_maybe_add_background_noise()`:

- %50 olasılıkla beyaz veya pembe gürültü (`1 / √f` FFT şekillendirmesi).
- SNR aralığı: `5 – 20 dB`, rastgele.
- Gürültü yalnızca mixture'a eklenir; target stem'e dokunulmaz → model
  gerçek hedef ile ambiance'ı ayırt etmeyi öğrenir.

SNR hesabı (`mix_rms / 10^(snr_db / 20)`) izole bir numpy testi ile
10.00 dB hedefte 10.00 dB ölçülerek doğrulandı.

### 3.3. Model İsimlendirme Standardizasyonu (RULES.md)

`<task>_<architecture>_<dataset-or-keyfeature>_v<major>.<minor>` formatına
geçildi:

- **Eski:** `best_conditioned_separator.h5` + `conditioned_class_names.json`
- **Yeni:** `separator_unet_film_multi_v2.0.h5` +
  `separator_unet_film_multi_v2.0_classes.json`

Tüm eğitim, değerlendirme, web uygulaması ve Colab notebook'ları bu yeni
isimleri kullanacak şekilde güncellendi.

---

## 4. Veri Seti Genişletme

`src/data_preparation/dataset_sources.py` artık `data/raw/` altındaki tüm
dataset'leri otomatik birleştiriyor:

| Dataset       | Konum                       | Sınıf Sayısı            |
|---------------|-----------------------------|-------------------------|
| ESC-50        | `data/raw/archive/`         | 50                      |
| UrbanSound8K  | `data/raw/urbansound8k/`    | 10 (4'ü ESC-50'ye alias)|

**Alias kuralı:** `CLASS_ALIASES = {"dog_bark": "dog", "engine_idling":
"engine", ...}` ile çakışan sınıflar tek havuzda toplanıyor. İki dataset
birlikte → **~56 sınıf**. Yeni dataset eklemek için yalnızca `load_<isim>()`
yazıp `load_all_datasets`'e bağlamak yeterli; mixer ve model query boyutu
otomatik adapte oluyor.

---

## 5. Kod Tabanı Reorganizasyonu

Mart raporundan sonra biriken legacy kod temizlendi ve
`feature/separator-quality-overhaul` branch'inde her şey toparlandı.

### 5.1. Kaldırılan Dosyalar

Edge AI / ANC iş kollarından kalan, artık pipeline'da yer almayan modüller
silindi:

- `src/model_training/`: `separator_unet.py`, `train_separator.py`,
  `evaluate_separator.py`, `train.py` (MobileNetV2)
- `src/application/`: `selective_separation.py`,
  `conditioned_selective_separation.py`, `inference.py`, `canceller.py`,
  `simulate_anc.py`, `verify_cancellation.py`
- `src/data_preparation/`: `separation_dataset.py`,
  `synthetic_data_generator.py`
- `src/model_optimization/` tümü (TFLite niceleme, C header export)
- `tests/test_model.py`
- Eski dokümanlar: `model_and_data.md`, `pipeline_guide.md`,
  `separation_guide.md`
- Eski notebook'lar: `colab_train_separator.ipynb`,
  `colab_source_separation.ipynb`, `snc_dataset_preperation.ipynb`

### 5.2. Yeniden Yazılan Dokümanlar

- **`CLAUDE.md`:** Eski ANC / Edge AI içeriği tamamen kaldırıldı; pipeline
  yalnızca query-conditioned separation üzerine.
- **`README.md`:** Source separation + webapp odaklı yeniden yazıldı.
- **`docs/conditioned_separation_guide.md`:** Branch ve dosya referansları
  güncellendi; webapp.py'a yönlendirildi.

---

## 6. Pipeline Özeti (Mayıs Sonu Durumu)

```
ESC-50 + UrbanSound8K (raw WAVs)
    → dataset_sources.load_all_datasets    in-memory clip cache, ~56 sınıf
    → SeparationMixer                      on-the-fly mixtures
                                           negative_prob = 0.45
                                           bg noise %50, SNR 5–20 dB
    → ConditionedSeparatorTrainer          FiLM-conditioned 2D U-Net
                                           L1 loss on magnitude
    → separator_unet_film_multi_v2.0.h5
    → webapp.py                            spectrogram-OLA,
                                           power-ratio mask,
                                           mask-CoV detection
```

**Spectrogram Contract:** 16 kHz mono, `n_fft = 512`, `hop_length = 128`,
`FREQ_BINS = 256`, `TIME_FRAMES = 128` (~1 sn pencere).

---

## 7. Sıradaki Adımlar

1. v2.0 modelini T4 üzerinde tam eğitime sokmak (~45–90 dk).
2. Mart raporundaki SI-SDRi tablosunu yeni model ile yeniden üretmek; iki
   dataset varlığında per-class artışı raporlamak.
3. Web uygulamasında *Cafe / Restaurant / Street* gibi gerçek dünya
   kayıtlarıyla kalitatif dinleme örnekleri toplamak.
4. Tez yazımı için query-conditioning'in class-imbalance avantajını ölçen
   küçük bir karşılaştırma (fixed multi-output baseline'a karşı).
