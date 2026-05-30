---
marp: true
theme: default
paginate: true
size: 16:9
---

# Selective Noise Cancelling

### Query-Conditioned Source Separation ile Seçici Ses Temizleme

**Bitirme Projesi — v2.x İterasyon Raporu**

*Mayıs 2026 sonu durumu*

---

## 1. Tek Cümlede Hedef

> Kullanıcı bir ses ya da video yükler → uygulama içinde **hangi tür seslerin** geçtiğini tespit eder → kullanıcı silmek istediklerini işaretler → dosya temizlenmiş halde geri döner.

Mart raporundan farklı olarak artık **ürüne** dönüşmüş bir Gradio web uygulaması mevcut. Bu sunum, web uygulamasına geçişten itibaren (v2.0) bugünkü v2.4 planına kadar yapılan tüm adımları, **neyi neden** yaptığımızı ve metriklerin nasıl evrildiğini anlatıyor.

---

## 2. Genel Mimari — Kuş Bakışı

```
   Ses / video dosyası
           │
           ▼
   ┌─────────────────────┐
   │  webapp.py (Gradio) │
   └────────┬────────────┘
            │
   ┌────────▼────────┐        Hangi sınıflar
   │  DETECT         │ ──►    geçiyor?
   │  (her sınıf     │
   │   için maskeyi  │        (Kullanıcı seçer)
   │   skorla)       │              │
   └────────┬────────┘              │
            │            ◄──────────┘
   ┌────────▼────────┐
   │  REMOVE         │ ──►    Temizlenmiş ses
   │  (seçilen       │        + (varsa) video
   │   sınıfları     │
   │   maskele)      │
   └─────────────────┘
```

Tek bir FiLM-koşullu U-Net, **"bana X sınıfını ayır"** sorgusunu cevaplayan tek elemanlı bir araç olarak hem detection hem removal'a hizmet ediyor.

---

## 3. Neden "Query-Conditioned" Yaklaşım?

**Klasik alternatif:** Her sınıf için ayrı bir model, veya N çıkışlı sabit-kafa bir model.

**Sorun:** ESC-50 + UrbanSound8K + FSD50K = onlarca / yüzlerce sınıf. Sınıf başına model imkânsız. Sabit N-çıkışlı model ise her yeni sınıfta yeniden eğitim gerektirir.

**Bizim çözüm:** **Tek** bir U-Net, ek bir girişten (one-hot **class query**) hangi sınıfı çıkaracağını öğrenir. FiLM (Feature-wise Linear Modulation) katmanları, sorgu vektörünü her seviyede özellik haritalarına çevirip *modülasyon parametreleri* (γ, β) üretir.

```
   log_mag  ───►  Encoder ───►  Bottleneck ───►  Decoder ───►  Mask (256×128)
                     ▲             ▲                 ▲
                     │             │                 │
        ┌────────────┴─────────────┴─────────────────┘
        │
   class_query (one-hot)  ──►  Embed (128-d)  ──►  γ, β  her seviyeye
```

Yeni sınıf eklemenin maliyeti = **bir satır query vektörünü genişletmek**.

---

## 4. Spectrogram Sözleşmesi (Sabit Kalan Şey)

Tüm modelin gördüğü/ürettiği her şey aynı boyutlu spektrograma indirgenir. Bu, web uygulaması, eğitim ve değerlendirme kodlarının tek bir kontratla konuşmasını sağlıyor.

| Parametre | Değer | Anlam |
|---|---|---|
| `SAMPLE_RATE` | 16 kHz mono | Tüm sesler resample |
| `N_FFT` | 512 | STFT pencere boyu |
| `HOP_LENGTH` | 128 | Pencere kayması (75% overlap) |
| `FREQ_BINS` | 256 | Nyquist bin'i atılır, U-Net girişi |
| `TIME_FRAMES` | 128 | ~1 saniyelik chunk |
| Çıkış | (256, 128, 1) | `[0,1]` aralığında soft mask |

Bir U-Net çağrısı **≈ 1 saniyelik ses**'i işliyor. Uzun dosyalar, web uygulamasında overlap-add ile birleştiriliyor (Bölüm 7).

---

## 5. Eğitim Pipeline'ı — "Sentetik Karışım Üretici"

Eğitim için **dataset dosyası yok**. `SeparationMixer`, her adımda canlı bir örnek üretir:

```
1. Rastgele 1–4 sınıf seç     →  ESC-50 cat + US8K siren + ...
2. Her birinden 1 sn pencere   →  rastgele amplitüd ile karıştır
3. Bazen background noise ekle (white / pink, 15–30 dB SNR)
4. Bir query sınıfı seç:
      • %85: mixture'da geçen sınıflardan biri  (pozitif örnek)
      • %15: mixture'da geçmeyen bir sınıf       (negatif örnek → target = sıfır)
5. ((log_mag, query, lin_mag), target_stem_mag)  döndür
```

**Pozitif örnek:** "X sınıfı mixture'da var, onu çıkar" → maske güçlü.
**Negatif örnek:** "Y sınıfı yok, çıkarmaya çalışma" → maske sıfır.

Bu basit kurgu, modelin hem **çıkarma** hem **yok**-tespiti yeteneklerini aynı kayıp fonksiyonu ile öğrenmesini sağlıyor.

---

## 6. Loss ve Eğitim Hedefi

Çıktı bir maske olmasına rağmen, eğitim sırasında **maske doğrudan mixture'a uygulanır** (graph içinde Multiply) ve **tahmin edilen stem magnitude** üzerinden L1 alınır:

```
   mask = U-Net(log_mag, query)
   estimated_stem = mask × linear_mag        ← grafiğin içinde
   loss = MultiResL1(estimated_stem, true_stem_mag)
```

**Çok-çözünürlüklü L1 (v2.2'de eklendi):**

```
   L = L1(full) + 0.5·L1(½-res) + 0.25·L1(¼-res)
```

Kaba ölçeklerde önce **spektral şekli** öğrenir, sonra ince detayı düzeltir → daha temiz maske, daha az "musical noise".

---

## 7. Web Uygulaması — `webapp.py`

**Detection (`detect_sounds`):**

Her sınıf için maskeyi tahmin et, iki bileşenli puan ver:

```
score = energy_ratio × (1 + CoV(mask)²)
        ▲                     ▲
        │                     └── Maske ne kadar yoğunlaşmış?
        └── Maske mixture enerjisinin ne kadarını yakalıyor?
```

Mutlak eşik (`0.05`) **ve** bağıl eşik (`0.65 × kazanan`) → "5–15 alakasız sınıf" sorunu kapanır.

**Removal (`remove_sounds`):**

Tüm dosya için **tek** bir STFT al → faz globalde tutarlı.
Her `TIME_FRAMES` (128-frame) chunk için (75% overlap):

```
   amplitude_ratio = clip(est_mag / mix_mag, 0, 1)
   mask           = clip(1 − strength · amplitude_ratio, 0, 1)
   processed      = mix_mag × mask
```

Çoklu sınıf: maskeler **çarpımsal** birleşir (`combined *= mask_k`).
Hanning-pencereli OLA → ISTFT (orijinal faz) → temizlenmiş dalga.

---

## 8. Değerlendirme — Üç Otomatik Test

| Script | Ne Ölçer | Sağlıklı Aralık |
|---|---|---|
| `diagnose_model.py` | Model çökmüş mü? FiLM sorguyu kullanıyor mu? | 3/3 PASS |
| `evaluate_conditioned_separator.py` | **SI-SDRi** — separation kalitesi (dB) | ≥ +5 dB |
| `evaluate_detection.py` | Per-class **Precision / Recall / F1** | mean F1 ≥ 0.5 |

**Önemli:** Her checkpoint, yan dosyası `*_classes.json` ile birlikte saklanıyor — query vektörünün boyutu **modelin gördüğü vocab** ile sabitleniyor (FSD50K eklenince fark eden bu).

Bu üç skor, her eğitim sonrası tek bir Colab notebook (`colab_evaluate.ipynb`) ile dakikalar içinde alınıyor.

---

## 9. İterasyon Yolculuğu — Üst Düzey Tablo

| Versiyon | Ana Müdahale | Sınıf Sayısı | SI-SDRi | F1 | Durum |
|---|---|---|---|---|---|
| v2.0 | İlk web app + agresif augmentation | 56 | — | — | **Çökmüş** |
| v2.1 | Augmentation geri çekildi, normalize fix | 56 | −22.18 dB | 0.21 | Çalışıyor, sınır artifaktları var |
| v2.2 | Full-encoder FiLM + multi-res loss + 75% OLA | 56 (FSD50K=0) | −22.79 dB | 0.13 | Sınır artifaktları gitti, kaldırma yumuşak |
| v2.3 | `negative_prob: 0.30→0.15` + FSD50K doğru yüklendi | **235** | −21.49 dB | **0.02** | FSD50K FP felaketi |
| v2.4 | `min_clips_per_class=40` ile FSD50K kuyruk budanması | TBD | pending | pending | **Eğitim aşamasında** |

Aşağıdaki slaytlar her satırın hikayesini ayrı ayrı anlatıyor.

---

## 10. v2.0 — İlk Web Uygulaması, Ama Çökmüş Eğitim

**Bağlam:** Web app ilk kez ayağa kalktı. Eğitim hiperparametreleri ise agresif tutulmuştu:

- `negative_prob = 0.45` (örneklerin %45'i "yok-sınıf")
- `bg_noise_prob = 0.50` (eğitim örneklerinin yarısı gürültü içerir)
- `bg_snr_db_range = (5, 20) dB` (gürültü amplitüdü target'ın **%56**'sı)

**Sonuç — üç soruna bir arada:**

1. **%45 negatif** → L1, "her zaman sıfır çıkar" stratejisini ödüllendirir. Model "güvenli sessiz maske" dengesine çöker.
2. **5 dB SNR** → süpervizyon sinyali gürültünün altında kalır, ayırma öğrenilemez.
3. **Inference normalizasyon uyumsuzluğu:** eğitim peak=1 normalize ediyor, webapp ham ses besliyordu → STFT magnitüdleri eğitim dağılımının 3–10× altında, model **eğitilmemiş bölgede**.

**Belirtiler:** Detection 0–2 alakasız sınıf veriyor. Removal hiçbir sınıfta duyulabilir etki yapmıyor.

---

## 11. v2.1 — Sağlıklı Hiperparametreler + Normalize Düzeltmesi

| Parametre | v2.0 | v2.1 |
|---|---|---|
| `negative_prob` | 0.45 | **0.30** |
| `bg_noise_prob` | 0.50 | **0.10** |
| `bg_snr_db_range` | (5, 20) dB | **(15, 30) dB** |
| Inference normalize | yok | **peak-norm önce STFT** |

**Sonuç — model artık öğreniyor:**

- Diagnose: 3/3 PASS, HEALTHY.
- SI-SDRi −22.18 dB (kötü ama anlamlı bir başlangıç).
- F1 0.21 (precision 0.10–0.25, recall 0.3–0.6 → FP : TP ≈ 5.5 : 1).
- Gerçek dünya: removal duyulabilir, ama **iki problem** var:

> ① Sınır artifaktı: her 0.25 sn'de "tık-tık" gibi pulsing.
> ② Detection çok geniş ağ atıyor — 8–10 alakasız sınıf yüzeye çıkıyor.

Bunların ikisi de **inference tarafında** problemler — model değil. v2.2'nin hedefi bu.

---

## 12. v2.2 — Mimari + Inference Cilası

**Mimari değişiklik (model):**

- v2.1'de FiLM yalnızca bottleneck'te idi. v2.2'de **her encoder seviyesinde** (e1–e4 + bottleneck). Skip-connection'lar artık sınıf-özel aktivasyon taşıyor → daha keskin maske.
- **Multi-resolution L1 loss** (Slayt 6): full + ½ + ¼ çözünürlük.

**Inference değişiklikleri (webapp):**

| Şey | v2.1 | v2.2 |
|---|---|---|
| OLA adımı | `TIME_FRAMES/2` (%50 overlap) | **`TIME_FRAMES/4`** (%75 overlap) |
| Detection skoru | `energy × (1 + CoV)` | **`energy × (1 + CoV²)`** |
| Detection cutoff | 0.40 × winner | **0.65 × winner** |

**Sonuç — bir kazanım, bir tuzak:**

✓ **Pulsing artifaktı: tamamen gitti.** (75% overlap çözdü.)
✗ Kaldırma **çok yumuşak** — strength=1.0'da bile hedef ses fonda duyuluyor.
✗ FSD50K loader'da **single-label filtre bug'ı**: AudioSet hiyerarşik etiketleri (`"Bark,Dog,Animal"`) reddediliyordu → FSD50K **0 klip** yükledi. F1 düşüşü (0.21→0.13) bu yüzden — vocab değişmedi ama detection cutoff sertleşti.

---

## 13. v2.3 — FSD50K Gerçekten Yüklendi + `negative_prob` Düşürüldü

**İki değişiklik:**

1. **FSD50K loader fix:** `raw_labels[0]` (hiyerarşinin yaprağı) canonical sınıf olarak kullanılıyor. → FSD50K 179 yeni sınıf ekledi → **vocab 56 → 235**.
2. **`negative_prob: 0.30 → 0.15`**: v2.2'deki "kaldırma çok yumuşak" semptomunun kök sebebi olarak: %30 negatifte L1 büyük ölçüde "sıfıra git" yönünde itiyordu. Yarıya indirince model pozitif sınıf cevabını daha güçlü vermeli.

**Sonuç — beklenen kazanım, beklenmedik **felaket**:**

| Metrik | v2.2 | v2.3 | Δ |
|---|---|---|---|
| SI-SDRi | −22.79 dB | **−21.49 dB** | +1.30 dB (marjinal iyileşme) |
| **Detection F1** | 0.13 | **0.02** | ↓↓↓ |
| Total FP / TP | 564 / 92 | **1092 / 13** | felaket |

Detection skorunda ne oldu? → Bir sonraki slayt.

---

## 14. v2.3'ün Tanısı — FSD50K Aşırı-Hevesli Sınıflar

`evaluate_detection.py`'a eklediğimiz "**en çok FP üreten sınıflar**" tablosu sebebi anında ortaya çıkardı:

| Sınıf | FP | Yerel ses var mı? |
|---|---|---|
| purr | 43 | hayır (FSD50K-only) |
| bass_guitar | 42 | hayır |
| ringtone | 40 | hayır |
| telephone | 32 | hayır |
| thunderstorm | 29 | evet |
| boom | 27 | hayır |
| animal | 26 | hayır |
| bass_drum | 24 | hayır |

**Manuel test ile uyumlu:** fan sesinde, kafe gürültüsünde, su sesinde "bass_guitar var" diye sürekli işaretliyordu.

**Kök sebep — neden ESC-50 sınıfları aynı problemi yapmıyor da bunlar yapıyor?**

- ESC-50: sınıf başına **40 temiz, izole** klip.
- FSD50K: AudioSet leaf etiketlerinin **uzun kuyruğu** var — bazı sınıflar yalnızca 2–10 (gürültülü, çok-etiketli) klip.
- Az veri + gürültülü etiket → model o sorgu için **dağınık, ayrımcı olmayan maske** öğreniyor → her ses içeren chunk'ta yüksek `energy × (1+CoV²)` skoru veriyor.

Pozitif:negatif oranı (5.6:1) ESC-50 ile **birebir aynı**. Sorun oran değil, **sınıf başına veri miktarı/kalitesi**.

---

## 15. v2.4 — Tek Değişkenli Düzeltme: Clip-Floor

**Tek satırlık değişiklik (kavramsal):** `load_all_datasets`'e `min_clips_per_class=40` parametresi.

```python
# load_all_datasets içinde — birleştirme sonrası
dropped = sorted(c for c, v in merged.items()
                 if len(v) < min_clips_per_class)
for cls in dropped:
    del merged[cls]
```

**Neden 40?** ESC-50'nin sınıf başına klip sayısının aynısı → ESC-50 / UrbanSound8K hiç etkilenmiyor; eşik **yalnızca FSD50K kuyruğunu** kesiyor.

**Neden post-merge?** Cross-dataset alias'lar (FSD50K `bark` → ESC-50 `dog`) önce havuzlanıyor → FSD50K'nın "iyi" sınıfları korunuyor.

**Atıf netliği:** Tek değişken değişti → eğer FP düşerse veya SI-SDRi değişirse, sebebi bilebiliriz. v2.2'de iki şeyi (mimari + FSD50K bug) aynı anda değiştirdiğimiz için neyin ne kadar etki ettiğini ayıramadığımız dersini aldık.

**Şu anki durum:** Kod feature/separator-quality-overhaul branch'inde push'lu (commit `b6ff06a`). Colab eğitimi sıraya alındı.

---

## 16. Aldığımız Metodolojik Dersler

**Ders 1 — Çok-değişkenli değişiklikler, atıf kaybı:**
v2.2'de mimari (full-FiLM) + loss (multi-res) + FSD50K loader aynı anda değişti. FSD50K bug yüzünden net etki ölçülemedi. → v2.3'ten itibaren tek-değişkenli adımlara döndük.

**Ders 2 — "İyi" hiperparametre bile yanlış dozda zarar verir:**
`negative_prob` 0.45 (v2.0) çökerttiği için 0.30'a indi. 0.30 ise "kaldırma çok yumuşak" yaratıyordu. 0.15 düzeltti. Negatif örnekler ileri-geri salınımına en hassas knob.

**Ders 3 — Yeni dataset = yeni risk yüzeyi:**
FSD50K eklendiğinde "daha çok veri = daha iyi model" varsayımı uzun kuyruğu nedeniyle ters döndü. Vocab şişerken **her sınıfın yeterince beslendiğini** kontrol etmediğimiz için detection patladı.

**Ders 4 — Otomatik testler, manuel gözlemi doğrulamalı:**
"Bass_guitar her seste çıkıyor" manuel gözlemini `evaluate_detection.py`'a eklediğimiz **top-FP tablosu** ile sayısallaştırdık. Şimdi her checkpoint o tabloyu üretiyor.

---

## 17. Bugünkü Açık Sorunlar

**1. SI-SDRi hâlâ negatif (−21 dB civarı):**
Tek başına U-Net + mixture fazını yeniden kullanma (spectrogram-only yaklaşım) bu metrikte yapısal bir tavan koyuyor. Olası iyileştirmeler: complex spectrogram (faz tahmini), conv-tasnet tarzı time-domain model, dataset-içi hard-negative mining.

**2. Bazı sınıflar diagnose'da hâlâ zayıf (`cat`, `chirping_birds`):**
v2.1'den beri −1.0× / −0.6× advantage gösteriyorlar. ESC-50 cat klipleri çok kısa ve transient — modeli zorluyor.

**3. v2.4'ün gerçek dünya kalitesi henüz ölçülmedi:**
Sentetik metrikler iyileşse bile, gerçek kafe / sokak kayıtlarında removal'ın kalitatif kalitesi ayrı bir test.

---

## 18. Sıradaki Adımlar — Kısa Vadeli Yol Haritası

**v2.4 değerlendirmesi (bu hafta):**

- Colab T4'te eğitim (~45–90 dk).
- `colab_evaluate.ipynb` ile diagnose + SI-SDRi + F1 + top-FP tablosu.
- En az 3 farklı gerçek dünya örneği (kafe, trafik, hayvan sesli) ile manuel dinleme.

**Eğer FP hâlâ sürerse → v2.5:**

Hard cross-dataset negative mining: eğitim örneklerinin bir bölümü için, query bir FSD50K sınıfı **ve** mixture sadece ESC-50/UrbanSound8K içeriyor (target = sıfır) — model FSD50K sınıflarının "diğer datasetlere ait" seslerde tetiklenmemesi gerektiğini doğrudan öğreniyor.

**Eğer SI-SDRi düzelmiyorsa → v3.x yönü:**

- Complex spectrogram maskeleme (magnitude + phase ayrı maske).
- Loss eklemeleri: SI-SDR diferansiyel loss waveform üzerinde.
- Time-domain mimari (Conv-TasNet) ile karşılaştırma — tez bölümüne baseline olarak girer.

---

## 19. Nihai Hedef ve Tez Konumlandırması

**Ürün hedefi:** Tek tıklama ile çalışan, gerçek dünya ses/video dosyalarında **seçici** ses temizleme yapabilen bir uygulama. Klasik gürültü bastırıcılar tüm gürültüyü bastırır; bu sistem **kullanıcının silmek istediğini sorar**.

**Tez konumlandırması:**

- **Katkı 1:** Query-conditioning ile scalable bir mimari — sınıf-başına model çoğaltma probleminin elimine edilmesi (Bölüm 3).
- **Katkı 2:** Detection + removal pipeline'ının **tek modelle** yürütülmesi — energy-ratio × CoV² puanlaması.
- **Katkı 3 (metodolojik):** Çok-sınıflı, çok-dataset'li audio separation eğitiminin **veri-dengeleme** hassasiyetinin sistematik analizi (v2.3 → v2.4 deneyi).

**Kapanış mesajı:** Sistem bir ürün olarak çalışıyor; metrik tarafında SI-SDRi tavanına kadar henüz yol var, ama her sürüm hem ölçülebilir bir adım atıyor hem de bir sonraki sürüm için kontrollü bir hipotez bırakıyor.

---

## EK: Versiyon-Versiyon Hiperparametre Özet Tablosu

| Param | v2.0 | v2.1 | v2.2 | v2.3 | v2.4 |
|---|---|---|---|---|---|
| `negative_prob` | 0.45 | 0.30 | 0.30 | **0.15** | 0.15 |
| `bg_noise_prob` | 0.50 | 0.10 | 0.10 | 0.10 | 0.10 |
| `bg_snr_db_range` | (5, 20) | (15, 30) | (15, 30) | (15, 30) | (15, 30) |
| FiLM seviyesi | bottleneck | bottleneck | **tüm encoder** | tüm encoder | tüm encoder |
| Loss | L1 | L1 | **Multi-res L1** | Multi-res L1 | Multi-res L1 |
| OLA step | — | `T/2` | **`T/4`** | `T/4` | `T/4` |
| Det. cutoff | — | 0.40·w | **0.65·w** | 0.65·w | 0.65·w |
| Det. skor | — | `e·(1+CoV)` | **`e·(1+CoV²)`** | `e·(1+CoV²)` | `e·(1+CoV²)` |
| FSD50K | yok | yok | bug, 0 klip | **235 sınıf** | **40 klip eşiği** |
| Inference norm | bug | düzeltildi | — | — | — |

---

## EK: Repo ve Branch Durumu

- **Branch:** `feature/separator-quality-overhaul`
- **Son commit:** `56ac331` (FSD50K Drive symlink fix)
- **Eğitim entry point:** `python src/model_training/train_conditioned_separator.py`
- **Değerlendirme entry point:** `python src/model_training/evaluate_conditioned_separator.py`, `evaluate_detection.py`, `diagnose_model.py`
- **Webapp:** `python src/application/webapp.py`
- **Eğitim log'u:** `docs/model_training_log.md` (her sürüm için tam hiperparametre + sonuç tablosu)
- **Colab notebook'ları:** `notebooks/colab_train_conditioned_separator.ipynb`, `notebooks/colab_evaluate.ipynb`

---

# Teşekkürler

**Sorular?**
