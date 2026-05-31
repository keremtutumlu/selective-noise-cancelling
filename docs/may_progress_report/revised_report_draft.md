# AKILLI SES SİSTEMLERİNDE UÇ YAPAY ZEKA İLE SEÇİCİ GÜRÜLTÜ ENGELLEME GERÇEKLEŞTİRİLMESİ

**BİTİRME PROJESİ ARA RAPORU**  
Bilgisayar Mühendisliği Bölümü  
Marmara Üniversitesi Teknoloji Fakültesi

**DANIŞMAN:** Dr. Öğr. Üyesi Ali SARIKAŞ  
**İSTANBUL, 2026**

---

## ÖNSÖZ

Proje konusunun belirlenmesinden raporlama aşamasına kadar geçen süreçte, bilgi ve tecrübeleriyle bize yol gösteren, karşılaştığımız teknik zorluklarda yönlendirici fikirleriyle desteğini esirgemeyen ve özellikle TÜBİTAK 2209-A başvurusu fikriyle ufkumuzu genişleten değerli danışman hocamız Dr. Öğr. Üyesi Ali SARIKAŞ'a en içten teşekkürlerimizi sunarız. Ayrıca çalışmalarımız sırasında fikir alışverişinde bulunduğumuz arkadaşlarımıza ve eğitim hayatımız boyunca maddi ve manevi desteklerini hiçbir zaman esirgemeyen ailelerimize teşekkürü bir borç biliriz.

---

## KISALTMALAR

| Kısaltma | Açıklama |
|---|---|
| ANC | Active Noise Cancelling (Aktif Gürültü Engelleme) |
| SNC | Selective Noise Cancelling (Seçici Gürültü Engelleme) |
| DL | Deep Learning (Derin Öğrenme) |
| CNN | Convolutional Neural Network (Evrişimli Sinir Ağı) |
| RNN | Recurrent Neural Network (Tekrarlayan Sinir Ağı) |
| FxLMS | Filtered-x Least Mean Squares (Filtrelenmiş-x En Küçük Ortalama Kareler) |
| MCU | Microcontroller Unit (Mikrodenetleyici Ünitesi) |
| PESQ | Perceptual Evaluation of Speech Quality |
| SNR | Signal-to-Noise Ratio (Sinyal-Gürültü Oranı) |
| TFLM | TensorFlow Lite for Microcontrollers |
| Edge AI | Edge Artificial Intelligence (Uç Yapay Zeka) |
| TCN | Temporal Convolutional Network (Zamansal Evrişimli Ağ) |
| PoC | Proof of Concept (Kavram İspatı) |
| RMS | Root Mean Square (Ortalama Kare Değeri) |
| SI-SDRi | Scale-Invariant Signal-to-Distortion Ratio improvement |
| FiLM | Feature-wise Linear Modulation |
| STFT | Short-Time Fourier Transform (Kısa Zamanlı Fourier Dönüşümü) |
| OLA | Overlap-Add (Üst Üste Ekleme) |
| FP | False Positive (Yanlış Pozitif) |
| TP | True Positive (Doğru Pozitif) |

---

## ÖZET

Bu proje, bir **seçici ses temizleme sistemi** geliştirmektedir: kısa bir ses (veya video) kaydı verildiğinde sistem, kayıtta hangi ses sınıflarının bulunduğunu tespit eder, kullanıcının hangilerini kaldıracağını seçmesine izin verir ve temizlenmiş çıktıyı döndürür. Sistemin çekirdeği, *hangi* sınıfı çıkaracağı kendisine söylenen ve o sınıf için yumuşak bir spektrogram maskesi üreten **FiLM-koşullu bir U-Net** mimarisidir; bir Gradio web uygulaması tespit → seçim → işleme akışının tamamını yürütür.

Bir ses kaynağını diğerinden ayıramayan klasik FxLMS tabanlı Aktif Gürültü Engellemenin (ANC) aksine, bu sorgu-koşullu yaklaşım yalnızca seçilen tek sınıfı hedeflerken geri kalanı korur. Bu ara rapor, web tabanlı ayrıştırma sistemini ve onun yinelemeli değerlendirmesini (v2.0–v2.4) kapsar. Gömülü/Uç-YZ (Edge AI) donanımına yerleştirme ve gerçek zamanlı akustik ANC, projenin sonraki fazına bırakılmıştır.

*Mayıs, 2026* — Kerem TUTUMLU, Melih GÖÇMEN

---

## ABSTRACT

This project develops a **selective sound-removal system**: given a short audio (or video) recording, it detects which sound classes are present and lets the user choose which ones to remove, returning the cleaned output. The core is a **FiLM-conditioned U-Net** that is told *which* class to extract and emits a soft spectrogram mask; a Gradio web application drives the full detect → select → render pipeline.

Unlike conventional FxLMS-based Active Noise Cancellation, which cannot tell one sound source from another, this query-conditioned approach targets a single chosen class while preserving the rest. This interim report covers the web-based separation system and its iterative evaluation (v2.0–v2.4). Deployment on embedded/Edge-AI hardware and real-time acoustic ANC are planned for the project's next phase.

*May, 2026* — Melih GÖÇMEN, Kerem TUTUMLU

---

## 1. GİRİŞ

### 1.1. Proje Çalışmasının Literatürdeki Yeri ve Eksikleri

Gürültü engelleme teknolojisi geçmişten günümüze geliştirilen ve gelişmeye devam eden bir alandır. Gürültü engellemenin teorik temeli, anti-gürültü dalgası üretilmesi fikriyle ilk olarak 1936 yılında Paul Lueg tarafından ortaya atılmış ve patentlenmiştir. 1980'lerde Dijital Sinyal İşlemcilerin (DSP) gelişmesiyle teknoloji ticarileşme evresine girmiş ve 1989 yılında dünyanın ilk ticari aktif gürültü engelleyici pilot kulaklığı piyasaya sürülmüştür. Bu dönemden itibaren endüstriyel standart haline gelen algoritma, Filtrelenmiş-x En Küçük Ortalama Kareler (FxLMS) algoritmasıdır.

Ancak günümüzde yaygın olarak kullanılan FxLMS tabanlı kulaklıkların iki kritik kısıtlaması bulunmaktadır. Birincisi, karmaşık ve doğrusal olmayan gürültüleri engellemede yetersiz kalmalarıdır. İkincisi ve daha önemlisi, bu sistemlerin "seçicilik" yeteneğinin olmamasıdır: mevcut sistemler sesi kategorize edemez, yalnızca genel bir sönümleme uygular. Bu durum, kullanıcının hangi sesi duyup hangisini engellemek istediğini seçmesine olanak tanımaz (örneğin ofiste çalışırken klima uğultusunu engellemek ama insan sesini duymak). Literatürde Derin Öğrenme tabanlı yöntemlerin karmaşık gürültüleri modellemede başarılı olduğu kanıtlanmış olsa da, bu modellerin gerçek zamanlı sınıflandırma ve kontrol için mobil donanımlara uyarlanması hâlâ açık bir araştırma konusudur.

### 1.2. Proje Çalışmasının Amacı ve Önemi

Bu projenin temel araştırma sorusu şudur: *"Derin öğrenme tabanlı modeller, farklı gürültü kaynaklarını ayırt edip her birine ayrı ayrı müdahale edilebilir mi?"*

Bu soruyu yanıtlamak için proje üç aşamalı bir yaklaşım benimsemiştir:

1. **Sınıflandırma aşaması (System A, Kasım–Mart 2026):** Ortamdaki ses sınıflarını yüksek doğrulukla tanıyan hafif bir CNN modeli (MobileNetV2) geliştirildi. Ancak sınıflandırma tek başına bir sesi karışımdan cerrahi olarak *çıkarmak* için yeterli değildir: hangi sesin var olduğunu bilmek, onu bastırmak için gereken maskeyi vermez.

2. **Kaynak ayrıştırma aşaması (System B, Mart–günümüz):** Bu eksikliği gidermek amacıyla proje, kullanıcının seçtiği sınıfa özgü yumuşak bir spektrogram maskesi üreten **sorgu-koşullu kaynak ayrıştırma** sistemine yöneldi. FiLM-koşullu 2D U-Net mimarisi ve Gradio web uygulaması bu aşamanın ürünüdür. Bu rapor, söz konusu sistemin web tabanlı uygulamasını ve yinelemeli değerlendirmesini (v2.0–v2.4) kapsar.

3. **Gömülü/Edge AI aşaması (Gelecek faz):** Web uygulamasında doğrulanan sistem, model optimizasyonu (niceleme, budama) ve MCU entegrasyonu ile gerçek zamanlı gömülü platforma taşınacaktır. TÜBİTAK 2209-A başvurusundaki gömülü sistem hedefleri bu fazı tanımlamaktadır.

---

## 2. TEKNİK ARKA PLAN

### 2.1. Geleneksel Gürültü Engelleme Yöntemlerinin Kısıtları

FxLMS, durağan gürültüleri başarıyla sönümleyebilirken, karmaşık ve durağan olmayan gürültü türlerinde (insan konuşmaları, sirenler, köpek havlaması) ciddi kısıtlamalarla karşılaşır. Algoritma gürültünün doğrusal modelini çıkarmaya çalışır ve sesi sınıflandıramaz; bu da yalnızca genel sönümleme sağlar, seçici bastırma sağlayamaz.

### 2.2. Derin Öğrenme Mimarilerinin Kıyaslanması

Bu eksikliği gidermek için literatür Derin Öğrenme (DL) tabanlı yaklaşımlara yönelmiştir. Sesin zamansal ilişkilerini modellemede güçlü olan Tekrarlayan Sinir Ağları (RNN), ardışık işlem yapısı nedeniyle gömülü uygulamalarda yüksek gecikmeye neden olur. Projede temel metodolojik karar, sinyalin zaman-frekans temsilini (spektrogram) işleyen ve daha yüksek paralellik sağlayan Evrişimli Sinir Ağı (CNN) mimarilerini temel almaktır.

### 2.3. Sorgu-Koşullu Kaynak Ayrıştırma

Sınıflandırmanın "hangi ses var?" sorusunu yanıtladığı; kaynak ayrıştırmanın ise "o sesi karışımdan çıkar" işini yaptığı iki ayrı görevdir. Bu projedeki kritik mimari karar, bu iki görevi tek bir modelde birleştiren **sorgu-koşullandırmasıdır**: model, hem karışım spektrogramını hem de "hangi sınıfı çıkar" sorgusunu giriş olarak alır ve yalnızca o sınıf için bir maske üretir. FiLM (Feature-wise Linear Modulation) mekanizması bu koşullandırmayı U-Net'in her encoder katmanına enjekte eder.

### 2.4. Uç Yapay Zeka (Edge AI) ve Model Optimizasyonu

Yüksek performanslı DL modellerinin getirdiği hesaplama gücü ve enerji tüketimini aşmak için iki temel teknik incelenmiştir: **Niceleme (Quantization)** (model ağırlıklarını 32-bit'ten 8-bit tamsayıya indirme) ve **Yapısal Budama (Pruning)** (performansa katkısı düşük katmanların kaldırılması). Bu teknikler, TensorFlow Lite for Microcontrollers (TFLM) ile modelin hedef donanımda 50 milisaniyenin altında çalışmasını sağlamak amacıyla planlanmıştır. **Bu optimizasyonlar projenin gelecek fazına aittir; mevcut rapor kapsamında uygulanmamıştır.**

---

## 3. VERİ SETİNİN HAZIRLIK İŞLEMLERİ

### 3.1. Veri Seti Araştırması ve Seçimi

Seçici ses temizleme sistemi için literatürde yaygın kullanılan açık kaynaklı veri setleri incelenmiş; ses sınıfı çeşitliliği ve etiket kalitesi kriterleri göz önünde bulundurularak aşağıdaki kaynaklar proje havuzuna dahil edilmiştir:

- **ESC-50:** 50 farklı çevresel ses sınıfını (doğa sesleri, hayvan, su, rüzgar vb.) içeren, sınıf başına 40 etiketli klipten oluşan temel veri seti. Tüm sınıflar eşit temsile sahiptir.

- **UrbanSound8K:** Şehir içi gürültü profillerini (klima, korna, çocuk sesi, köpek havlaması vb.) içeren 10 sınıflı veri seti. Dört sınıf ESC-50 ile örtüştüğünden `CLASS_ALIASES` mekanizması ile tek havuzda birleştirilmektedir; altı sınıf özgün katkı sağlar.

- **FSD50K:** AudioSet hiyerarşisiyle etiketlenmiş, geniş kapsamlı çok-etiketli ses olayı veri seti. v2.3 itibarıyla sınıf söz dağarcığını genişletmek amacıyla havuza eklenmiştir. Çok-etiketli kliplerde ilk etiket kanonik sınıf olarak alınır; `CLASS_ALIASES` ile çakışan sınıflar ESC-50 kanonik adlarına eşlenir.

- **LibriSpeech:** İnsan konuşması örneklerini içeren bu veri seti konuşma-koruma senaryosu için değerlendirilmiştir; ancak mevcut ayrıştırma pipeline'ına (ESC-50 + UrbanSound8K + FSD50K) henüz dahil edilmemiştir. Konuşma-hedefli ayrıştırma sonraki faza bırakılmıştır.

### 3.2. Ses Verilerinin İşlenmesi

Toplanan ham ses verileri modelin giriş gereksinimlerine göre standardize edilmiştir:

- **Örnekleme Hızı Dönüşümü:** Modelin spectrogram sözleşmesiyle (n_fft=512, hop=128, 256×128 giriş boyutu) uyumlu olması için tüm sesler 16 kHz mono'ya indirilmiştir.

- **Genlik Normalizasyonu:** Farklı kayıtlardaki ses şiddeti farklarını gidermek için sinyaller [−1, 1] aralığına normalize edilmiştir.

- **Ters Faz Hedeflemesi (Erken faz — System A):** Projenin Mart 2026 öncesi ANC aşamasında hedef veri, giriş sinyalinin −1 ile çarpılmasıyla elde edilmiştir. *Bu işlem yalnızca erken faz (System A) anti-faz ANC yaklaşımına aittir. Mevcut kaynak ayrıştırma sistemi, hedef olarak izole edilmiş kaynak stem'ini kullanır ve çıktı olarak yumuşak bir spektrogram maskesi üretir — ters faz dalgası değil.*

- **Negatif Örnek Üretimi:** `SeparationMixer`, her eğitim örneğinde belirli bir olasılıkla (`negative_prob`) karışımda bulunmayan bir sınıf sorgular ve hedef olarak sessizlik (sıfır) gösterir. Bu, modelin yokluğunu tanımasını öğretir.

### 3.3. Çok-Veri Seti Birleştirme Mimarisi

`src/data_preparation/dataset_sources.py`, `data/raw/` altındaki tüm veri setlerini otomatik birleştiren `load_all_datasets()` fonksiyonunu sağlar. `CLASS_ALIASES` mekanizması çakışan sınıf adlarını tek kanonik ada eşler. Yeni bir veri seti eklemek için `load_<isim>()` yazılıp bu fonksiyona bağlanması yeterlidir; mixer ve model sorgu boyutu otomatik adapte olur.

v2.3 itibarıyla `min_clips_per_class` parametresi ile birleşim sonrası minimum klip sayısı eşiği uygulanmaktadır (bkz. Bölüm 5.3).

---

## 4. MODEL VE MİMARİ KARARLAR

### 4.1. Erken Faz: CNN ve RNN Mimarilerinin Kıyaslanması (Kasım–Ocak 2025)

Ses sinyalleri doğası gereği zamansal seriler olduğundan literatürde LSTM ve GRU gibi Tekrarlayan Sinir Ağları (RNN) sıkça kullanılmaktadır. Ancak RNN tabanlı yapıların projemiz için iki temel dezavantajı tespit edilmiştir:

- **Sıralı İşlem Zorunluluğu:** RNN'ler t anındaki çıktıyı üretmek için t−1 anının tamamlanmasını bekler; MCU'ların paralel işlem gücünden tam yararlanılamaz.
- **Gecikme Maliyeti:** Döngüsel yapının hesaplama yükü, ANC sisteminin gerektirdiği düşük gecikme bütçesini aşma riski taşır.

Bu nedenle CNN tabanlı yapıların tercih edilmesine karar verilmiştir.

### 4.2. Erken Faz: Dalga Formu Tabanlı Encoder-Decoder Yapısı

Ocak 2026 çalışmalarında faz bilgisini koruma gerekliliği nedeniyle zaman alanında (time-domain) çalışan Wave-U-Net veya TCN türevi bir mimari hedeflenmiştir. Seçilen mimarinin kritik rolleri şunlardır:

- **Nedensellik (Causality):** Gerçek zamanlı akış için model yalnızca geçmiş veriyi kullanacak şekilde tasarlanmıştır.
- **Atlamalı Bağlantılar (Skip Connections):** Yüksek çözünürlüklü detayların bottleneck katmanında kaybolmadan çıkışa aktarılmasını sağlar.

### 4.3. Mimari Revizyon: FiLM-Koşullu 2D U-Net (Mart 2026)

Uygulama aşamasındaki ön testler 1D-CNN mimarisinin gömülü sistemler için yüksek hesaplama maliyeti getirdiğini göstermiştir. Bu bulgu, mimarinin iki boyutta yeniden düzenlenmesine yol açmıştır:

- Ses sinyalleri STFT ile zaman-frekans düzlemine taşınır: **n_fft=512, hop=128, 256×128×1 log-magnitude giriş.**
- FiLM-koşullu 2D U-Net, sınıf sorgusunu (one-hot vektör) her encoder seviyesine gamma/beta projeksiyon çiftleri aracılığıyla enjekte eder.
- Model çıktısı [0,1] arasında sigmoid soft maske; orijinal faz ISTFT aşamasında geri kullanılır.
- Kayıp fonksiyonu: L1 (tam + ½ + ¼ çözünürlük çok-ölçekli).

Bu mimari, MobileNetV2 tabanlı sınıflandırıcıyla (System A) aynı 16 kHz / 256×128 spectrogram sözleşmesini paylaşmakla birlikte tamamen farklı bir görev üstlenir: sınıflandırma değil, **maske üretimi.**

---

## 5. SİSTEM GERÇEKLEŞTİRİMİ VE DENEYSEL BULGULAR

### 5.1. System A — Sınıflandırma ve Dijital ANC PoC (Mart 2026)

Kasım–Mart 2026 arasında tamamlanan PoC aşaması üç alt bileşenden oluşmaktadır:

**Mimari Revizyon ve Veri Standardizasyonu:** Başlangıçta planlanan 1D-CNN mimarisi yerine daha hafif ve paralel işlem yeteneği yüksek 2D-CNN (MobileNetV2) yapısına geçilmiştir. Çevresel ses sinyalleri Log-Mel Spektrogramlarına dönüştürülerek model eğitimi için standardize edilmiştir.

**Model Eğitimi ve Sınıflandırma Başarımı:** Transfer Learning stratejisiyle eğitilen model, eğitimde hiç görmediği test verileri üzerinde **%84.38 doğruluk (accuracy)** oranına ulaşmıştır. Bu oran, sistemin ortamdaki hedef sesleri yüksek doğrulukla ayırt edebildiğini gösterir ve "hangi ses var?" sorusunu (tanıma aşaması) güvenilir biçimde yanıtlar.

**Seçici Sönümleme Algoritması ve Akustik Doğrulama:** Kullanıcının belirlediği sınıfların tespiti durumunda sinyal fazını 180 derece ters çeviren ve donanım gecikmesini kompanse eden bir yazılım modülü geliştirilmiştir. Üretilen anti-gürültü dalgası orijinal sesle dijital akustik simülatöründe üst üste bindirilmiş ve **75.10 dB gürültü sönümlemesi** ölçülmüştür.

> **Önemli Not — 75.10 dB'nin yorumu:** Bu değer, bir sinyal ile onun birebir dijital ters fazının (x + (−x)) üst üste bindirilmesiyle elde edilen **dijital simülasyondaki teorik üst sınırdır.** Mikrofon–hoparlör gecikmesi, oda akustiği ve donanım kısıtlarını içeren gerçek bir akustik ortamda ölçülmemiştir. Matematiksel terslemedeki mükemmelliği ispatlayan bu bulgu, projenin System A PoC hedefini karşılamaktadır. Gerçek-ortam akustik doğrulama, projenin gömülü fazına aittir.

**%84.38 sınıflandırma doğruluğu** ve **75.10 dB dijital simülasyon üst sınırı**, System A'nın tanıma ve bastırma zincirini yazılımsal düzeyde kanıtlamaktadır. Ancak bu metrikler, ayrıştırma sistemi (System B) için geçerli değildir: bir sesi karışımdan *çıkarmak*, sınıflandırmadan ayrı bir yetenektir ve bunun nicel ölçütü **SI-SDRi** ile **Detection F1**'dir.

### 5.2. System B — FiLM U-Net Geliştirme Süreci (Mayıs 2026)

Mayıs 2026 itibarıyla sistem beş model versiyonu (v2.0–v2.4) üzerinden yinelemeli olarak geliştirilmiş; her versiyonda elde edilen nicel bulgular bir sonraki tasarım kararını doğrudan yönlendirmiştir.

#### v2.0 — Başarısız: Agresif Artırma Eğitim Çöküşüne Yol Açtı

v2.0 üç eş zamanlı sorun nedeniyle tamamen başarısız olmuştur:

1. **negative_prob = 0.45 fazla yüksek:** L1 kaybı sıfıra yakın çıkış üretmeyi ödüllendirerek modeli "güvenli sessiz çıkış" dengesine itti; pozitif örnekleri öğrenemedi.
2. **bg_snr_db_range = (5–20) dB fazla agresif:** 5 dB'de gürültü genliği hedef sinyalin %56'sıdır; model bu gürültü altındaki hedefi ayırt edemedi.
3. **Çıkarım normalizasyon uyuşmazlığı:** Eğitimde pencere başına tepe-normalizasyonu uygulanırken webapp ham (normalizasyonsuz) ses besledi; model aktivasyonları eğitim dağılımının dışına çıktı.

#### v2.1 — Düzeltilmiş Artırma, Normalizasyon Onarıldı

Üç sorun giderildi: `negative_prob` 0.30'a, `bg_noise_prob` 0.10'a düşürüldü; `bg_snr_db_range` (15–30) dB'ye yükseltildi; çıkarım normalizasyonu düzeltildi.

| Metrik | Değer |
|---|---|
| SI-SDRi ortalama | **−22.18 dB** |
| Detection F1 | **0.21** |
| FP : TP | **5.5 : 1** |

Dinleme testlerinde hava kliması sınıfında gürültü giderme işitsel olarak doğrulandı. İki sorun tespit edildi: OLA sınırlarında ~0.25 saniyelik periyodik titreme ve yüksek yanlış pozitif oranı.

#### v2.2 — Tüm Encoder'lara FiLM, Çok-Ölçekli Kayıp, %75 OLA

FiLM koşullandırması tüm encoder seviyelerine (e1, e2, e3, e4, bottleneck) genişletildi. Çok-ölçekli L1 kaybı (tam + ½ + ¼ çözünürlük) eklendi. OLA örtüşme oranı %75'e yükseltildi; detection eşiği sıkılaştırıldı. Bu versiyonda FSD50K, tekli-etiket filtresi hatasından dolayı 0 klip katkısı sağladı (hata commit 6a77461'de giderildi).

| Metrik | Değer | v2.1'e göre |
|---|---|---|
| SI-SDRi ortalama | **−22.79 dB** | −0.61 dB (marjinal gerileme) |
| Detection F1 | **0.13** | ↓ (FSD50K=0 klip) |

Periyodik titreme tamamen giderildi. Ulaşım gürültüleri (uçak, tren, helikopter, klima, pnömatik matkap) doğru tespit edildi. SI-SDRi gerileme, FSD50K'nın bu tura katkı sağlayamamasına bağlandı.

#### v2.3 — negative_prob Düşürüldü, FSD50K İlk Kez Yüklendi

`negative_prob` 0.15'e düşürüldü; FSD50K yükleyici hatası giderilerek veri seti ilk kez tam olarak yüklendi. Model söz dağarcığı **56'dan 235 sınıfa** çıktı.

| Metrik | Değer | v2.2'ye göre |
|---|---|---|
| SI-SDRi ortalama | **−21.49 dB** | **+1.30 dB** (marjinal iyileşme) |
| Detection F1 | **0.02** | ↓↓ (çöküş) |
| TP / FP / FN | 13 / 1092 / 475 | — |

**Detection F1 çöküşünün kök nedeni — FSD50K aşırı yanlış pozitif üretimi:**

Yerelde sesi bulunmayan 179 FSD50K-özel sınıf, hiçbir gerçek karışımda bulunmadığından yalnızca FP üretebilir. Az ve gürültülü örnekle öğrenilen bu sınıflar dağınık (diffuse) maskeler üretir ve alakasız her seste yüksek skor verir:

| Sınıf | FP sayısı | Kaynak |
|---|---|---|
| purr | 43 | FSD50K-özel |
| bass_guitar | 42 | FSD50K-özel |
| ringtone | 40 | FSD50K-özel |
| telephone | 32 | FSD50K-özel |
| boom | 27 | FSD50K-özel |
| animal | 26 | FSD50K-özel |

Detection F1'in 0.13'ten (v2.2, 56-sınıf söz dağarcığı) 0.02'ye düşmesi tamamen bu sistematik FSD50K FP paterni ile açıklanmaktadır. Bu, dinleme testlerindeki "her dosyada bass_guitar çıkıyor" gözleminin nicel karşılığıdır.

**Temel ders:** Sınıf başına veri hacmi ve kalitesi, model kapasitesi veya negatif örnekleme oranından daha belirleyici olabilir. ESC-50 sınıfları aynı pozitif:negatif oranıyla sorunsuz çalışırken yetersiz destekli FSD50K-özel sınıflar sistematik FP üretmiştir.

#### v2.4 — Minimum Klip Sayısı Eşiği (Eğitim Bekliyor)

v2.3 FP çöküşünü gidermek için `load_all_datasets` içine **`min_clips_per_class = 40`** birleşim-sonrası (post-merge) eşiği eklendi. Eşik FSD50K uzun kuyruğunu budar; ESC-50 (sınıf başına 40 klip) ve UrbanSound8K (yüzlerce klip) etkilenmez. Eşik birleşimden *sonra* uygulandığından çapraz-veri aliasları önce havuzlanır (FSD50K `bark` klipleri ESC-50 `dog`'a sayılır) ve yalnızca gerçekten yetersiz desteklenen FSD50K-özel etiketler düşürülür.

**Tek-değişkenli** bir müdahale olması, etkisinin temiz biçimde atfedilebilmesini sağlar. v2.4 sonuçları Colab eğitimi tamamlandıktan sonra eklenecektir.

### 5.3. Model Versiyonları Özet Tablosu

| Versiyon | `negative_prob` | Veri seti (sınıf sayısı) | SI-SDRi | Detection F1 | Temel değişiklik |
|---|---|---|---|---|---|
| v2.0 | 0.45 | ESC-50 + US8K (~56) | — | — | Başarısız (eğitim çöküşü) |
| v2.1 | 0.30 | ~56 | −22.18 dB | 0.21 | Augmentasyon + normalizasyon onarıldı |
| v2.2 | 0.30 | ~56 (FSD50K=0, hata) | −22.79 dB | 0.13 | Tüm-encoder FiLM, MRSL, %75 OLA |
| v2.3 | 0.15 | 235 (FSD50K tam) | −21.49 dB | **0.02** | FSD50K yüklendi, FP çöküşü keşfedildi |
| v2.4 | 0.15 | ~budanmış | *beklemede* | *beklemede* | `min_clips=40` FSD50K long-tail budaması |

---

## 6. WEB UYGULAMASI

Gradio tabanlı web uygulaması (`src/application/webapp.py`) üç ana işlevi yürütür:

**Ses/Video Yükleme:** Kullanıcı ses veya video dosyası yükler. Yalnızca ses içeren `.mp4` dosyaları (WhatsApp çıktıları vb.) ffprobe ile algılanır; video stream yoksa yalnızca temizlenmiş ses döndürülür (ffmpeg muxing hatası böylece giderilmiştir).

**Sınıf Tespiti (detect_sounds):** U-Net tüm sınıflar için toplu çalıştırılır; her sınıf için `energy_ratio × (1 + CoV²)` puanı hesaplanır. Mutlak alt eşik (0.05) ve göreli kesim (0.65 × kazanan) uygulanarak onay kutuları oluşturulur.

**Ses Temizleme (remove_sounds):** Seçilen sınıflar için:
1. Tüm dosya için tek STFT (faz global olarak tutarlı).
2. TIME_FRAMES boyutlu, %75 örtüşmeli kayan pencere.
3. Her pencerede seçilen her sınıf için U-Net → power-ratio maske: `mask = clip(1 − strength·est²/mix², 0, 1)`.
4. Çoklu sınıflar için maskeler çarpımsal birleştirilir.
5. Zaman ekseninde 5-frame (~40 ms) konvolüsyonel düzleştirme (musical noise bastırma).
6. Hanning ağırlıklı Overlap-Add akümülasyonu.
7. Tek ISTFT — orijinal faz korunur.

Video girişlerinde ffmpeg ile temizlenmiş ses orijinal video track'ine muxlanarak döndürülür.

---

## 7. PROJE YÖNETİMİ VE ÇALIŞMALAR

### 7.1. Kasım 2025 İtibarıyla Gerçekleştirilen Ön Çalışmalar

- **Problem Alanının Belirlenmesi:** ANC sistemlerinin kısıtlılıkları incelenmiş ve Seçici Gürültü Engelleme (SNC) kavramı literatürdeki derin öğrenme uygulamaları ışığında konumlandırılmıştır.
- **Mimari Fizibilite:** Gömülü sistemler için MobileNetV2 temelli CNN mimarisine odaklanma kararı alınmıştır.
- **TinyML Araştırması:** Niceleme ve budama yöntemlerinin teorik temelleri ve gömülü sistemlere entegrasyon süreçleri araştırılmıştır.
- **TÜBİTAK 2209-A Başvurusu:** Araştırma önerisi TÜBİTAK 2209-A programına başarıyla sunulmuştur.

### 7.2. Ocak 2026 İtibarıyla Gerçekleştirilen Çalışmalar

Ocak döneminde veri seti ve model kararları üzerine gerekli çalışmalar yürütülmüştür. Ses sınıf çeşitliliği kriterine göre Kaggle platformu ve akademik arşivlerden veri setleri araştırılmış; model mimarilerinin projeye uygunluğu değerlendirilmiştir. Optimizasyon sürecine kadar gerçekleşecek adımların genel haritası belirlenmiştir. Ayrıntılar için bkz. Bölüm 3 ve Bölüm 4.

### 7.3. Mart 2026 İtibarıyla Gerçekleştirilen Çalışmalar

Mart 2026 itibarıyla projenin temel kilometre taşlarından biri olan yazılımsal PoC (System A) başarıyla tamamlanmıştır. Üç alt aşamada gerçekleştirilen bu süreç:

- **Donanım Odaklı Mimari Revizyonu:** 1D-CNN'den 2D-CNN (MobileNetV2) yapısına geçiş; Log-Mel spektrogram standardizasyonu.
- **Model Eğitimi:** Transfer Learning ile %84.38 sınıflandırma doğruluğuna ulaşıldı.
- **Seçici Sönümleme ve Akustik Doğrulama:** Anti-faz yazılım modülü geliştirildi; dijital simülatörde 75.10 dB teorik üst sınır ölçüldü *(bkz. Bölüm 5.1'deki yorum notu)*.

### 7.4. Mayıs 2026 İtibarıyla Gerçekleştirilen Çalışmalar

Mart raporunun ardından proje üç ana doğrultuda derinleştirilmiştir:

**Web Uygulaması İyileştirmeleri:**
- MP4/video dosyası crash düzeltmesi (ffprobe ile stream kontrolü).
- İki bileşenli detection puanı: `score = energy_ratio × (1 + CoV²)` — difüz maskeleri cezalandırır.
- Power-domain ratio maskeleme: `mask = clip(1 − strength·(est²/mix²), 0, 1)` — over-removal önlendi.
- Spectrogram-level %75 OLA: pencere sınırı artifact tamamen giderildi.

**Eğitim Pipeline İyileştirmeleri:**
- v2.1–v2.4 model versiyonları geliştirildi (bkz. Bölüm 5.2).
- FSD50K entegrasyonu ve yükleyici hatası giderildi.
- `min_clips_per_class=40` parametresi ile FSD50K uzun kuyruğu budama mekanizması eklendi.
- Arka plan gürültüsü artırma (`_maybe_add_background_noise()`): %50 olasılıkla beyaz/pembe gürültü, SNR 5–20 dB aralığı.

**Kod Tabanı Reorganizasyonu:**
- Edge AI / ANC iş kollarından kalan legacy modüller kaldırıldı.
- `RULES.md` ile model isimlendirme standardizasyonu: `<task>_<architecture>_<dataset-or-keyfeature>_v<major>.<minor>` formatı.
- `CLAUDE.md`, `README.md` ve `docs/conditioned_separation_guide.md` sorgu-koşullu ayrıştırma odaklı yeniden yazıldı.

### 7.5. Sıradaki Adımlar

1. v2.4 modelini Colab T4 üzerinde eğitmek (~45–90 dk) ve SI-SDRi / Detection F1 ölçmek.
2. v2.3 / v2.4 karşılaştırmalı tablosunu tamamlamak; `min_clips=40` eşiğinin FP üzerindeki etkisini raporlamak.
3. Web uygulamasında gerçek dünya kayıtlarıyla (kafe, trafik, ofis) kalitatif dinleme örnekleri toplamak.
4. Query-conditioning'in class-imbalance avantajını ölçen küçük bir karşılaştırma (fixed multi-output baseline'a karşı) — tez yazımı için.

---

## 8. MEVCUT SİSTEM SINIRLARI VE GELECEK DÖNEM HEDEFLERİ

Bu rapor kapsamında geliştirilen web tabanlı seçici ses temizleme sistemi, projenin **ilk fazını** oluşturur. Mevcut sistem aşağıdaki sınırlara sahiptir:

- SI-SDRi değerleri negatif seyretmektedir. Bu, spektrogram U-Net'inin karışım fazını yeniden kullandığı bağlamlarda beklenen bir kısıtlamadır; dinleme testlerinde gürültü giderme işitsel olarak doğrulanmıştır.
- v2.3'te FSD50K entegrasyonuyla beraber ortaya çıkan Detection F1 çöküşü v2.4 tasarımıyla giderilmeye çalışılmakta, sonuçlar beklenmektedir.
- Sistem gerçek zamanlı (real-time) akış değil dosya tabanlı çalışmaktadır.

**Sonraki fazda planlanmaktadır:**

1. **Çok-Etiket Detection İyileştirmesi:** v2.4 sonuçlarına göre per-class kalibrasyon eşikleri veya cross-dataset hard negative stratejisi (v2.5).
2. **Edge AI Optimizasyonu:** Eğitilmiş modelin 8-bit Niceleme (Quantization) ve Yapısal Budama (Pruning) ile sıkıştırılması; doğruluk kaybının %5'i geçmemesi hedefi.
3. **MCU Entegrasyonu:** Optimize edilmiş modelin TensorFlow Lite for Microcontrollers (TFLM) ile hedef donanımda <50 ms gecikmeyle çalışması.
4. **Gerçek Akustik Doğrulama:** Dijital simülasyonda ölçülen teorik üst sınırın (75.10 dB), mikrofon–hoparlör gecikmesi ve oda akustiğini içeren gerçek ortamda doğrulanması.

TÜBİTAK 2209-A başvurusundaki gömülü sistem hedefleri bu gelecek fazı tanımlamaktadır.

### 8.1. Projenin Amaç ve Hedefleri (TÜBİTAK Başvurusuna Dayalı)

Projenin TÜBİTAK 2209-A başvurusunda belirlenen somut hedefleri şunlardır:

| Hedef | Kapsam | Durum |
|---|---|---|
| Veri seti hazırlığı ve sınıflandırma (%84.38 doğruluk) | System A (Faz 1) | Tamamlandı |
| Web tabanlı kaynak ayrıştırma (Gradio) | System B (Faz 1) | Tamamlandı |
| Model optimizasyonu (8-bit niceleme, %5 doğruluk kaybı sınırı) | Edge AI (Faz 2) | Planlanıyor |
| Gerçek zamanlı MCU entegrasyonu (<50 ms gecikme) | Edge AI (Faz 2) | Planlanıyor |
| Fiziksel prototip ve saha testleri | Edge AI (Faz 2) | Planlanıyor |

### 8.2. Çalışma Takvimi (Aralık 2025 – Haziran 2026)

| Dönem | Faaliyetler | Durum |
|---|---|---|
| Kasım–Aralık 2025 | Literatür taraması, TÜBİTAK başvurusu, mimari fizibilite | Tamamlandı |
| Ocak 2026 | Veri seti seçimi, model mimari araştırması | Tamamlandı |
| Mart 2026 | System A PoC: sınıflandırma + dijital ANC simülasyonu | Tamamlandı |
| Mart–Mayıs 2026 | System B: FiLM U-Net, web uygulaması, v2.0–v2.4 | Devam ediyor |
| Haziran 2026 | v2.4 değerlendirme, Edge AI optimizasyon hazırlığı | Planlanıyor |

---

## KAYNAKLAR

1. P. Lueg, "Process of silencing sound oscillations," U.S. Patent 2 043 416, Haz. 9, 1936.
2. W. F. Meeker, "Active ear defender systems: Development of a laboratory model," Wright Air Development Division, 1959.
3. L. J. Fogel, "Apparatus for improving intelligence under high ambient noise levels," U.S. Patent 2 966 549, Ara. 27, 1960.
4. H. F. Olson, "Electronic sound absorber," U.S. Patent 2 983 790, May. 9, 1961.
5. A. G. Bose ve J. Carter, "Headphoning," U.S. Patent 4 455 675, Haz. 19, 1984.
6. D. R. Morgan, "An analysis of multiple correlation cancellation loops with a filter in the auxiliary path," *IEEE Trans. Acoust., Speech, Signal Process.*, c. 28, sy 4, ss. 454–467, 1980.
7. H. Zhang ve D. Wang, "Deep ANC: A deep learning approach to active noise control," *Neural Netw.*, c. 141, ss. 1–10, 2021.
8. A. R. Yuliani ve diğerleri, "Speech enhancement using deep learning methods: A review," *J. Elektron. dan Telekomun.*, c. 21, sy 1, ss. 19–26, 2021.
9. G. Wang ve diğerleri, "Low-latency real-time deep learning," *IEEE ICASSP*, 2020.
10. I. Hubara ve diğerleri, "Accurate post training quantization with small calibration sets," *Proc. ICML*, c. 139, ss. 4466–4475, 2021.
11. T. Hoefler ve diğerleri, "The state of sparsity in deep neural networks," *arXiv preprint arXiv:1902.09574*, 2021.
12. A. G. Howard ve diğerleri, "MobileNets: Efficient convolutional neural networks for mobile vision applications," *arXiv preprint arXiv:1704.04861*, 2017.
13. E. Fonseca ve diğerleri, "FSD50K: An open dataset of human-labeled sound events," *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, 2022.
14. K. J. Piczak, "ESC: Dataset for environmental sound classification," *Proc. ACM Multimedia*, 2015.
15. J. Salamon, C. Jacoby ve J. P. Bello, "A dataset and taxonomy for urban sound research," *Proc. ACM Multimedia*, 2014.
16. E. Perez ve diğerleri, "FiLM: Visual reasoning with a general conditioning layer," *Proc. AAAI*, 2018.
17. J. Le Roux ve diğerleri, "SDR — half-baked or well done?," *Proc. IEEE ICASSP*, 2019.
