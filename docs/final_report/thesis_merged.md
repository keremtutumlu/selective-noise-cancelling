# MARMARA ÜNİVERSİTESİ
# TEKNOLOJİ FAKÜLTESİ
# BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ

  


## DERİN ÖĞRENME İLE SEÇİCİ GÜRÜLTÜ ENGELLEME

  


### BİTİRME PROJESİ

  


⟨ÖĞRENCİ ADI SOYADI⟩
⟨ÖĞRENCİ NUMARASI⟩

  


**DANIŞMAN**
⟨Unvan Ad Soyad⟩

  


İSTANBUL, 2026

---

## JÜRİ ONAY SAYFASI

Marmara Üniversitesi Teknoloji Fakültesi Bilgisayar Mühendisliği öğrencisi ⟨Öğrenci Adı Soyadı⟩ tarafından hazırlanan "Derin Öğrenme ile Seçici Gürültü Engelleme" başlıklı proje çalışması, ⟨gg.aa.2026⟩ tarihinde savunulmuş ve jüri üyeleri tarafından başarılı bulunmuştur.

| Jüri Üyeleri | Kurum | İmza |
|---|---|---|
| ⟨Unvan Ad Soyad⟩ (Danışman) | Marmara Üniversitesi | …………… |
| ⟨Unvan Ad Soyad⟩ (Üye) | Marmara Üniversitesi | …………… |
| ⟨Unvan Ad Soyad⟩ (Üye) | Marmara Üniversitesi | …………… |

---

## ÖNSÖZ

Proje çalışması süresince karşılaşılan problemlerde yardım ve bilgisini esirgemeyen değerli danışman hocaya, çalışmanın her aşamasındaki yönlendirmeleri için teşekkür edilir. Ayrıca, eğitim süreci boyunca destek olan tüm bölüm öğretim üyelerine ve aileye teşekkür edilir.



⟨Öğrenci Adı Soyadı⟩
Haziran, 2026



---

## İÇİNDEKİLER

- ÖZET
- ABSTRACT
- ŞEKİL LİSTESİ
- TABLO LİSTESİ
- ALGORİTMA LİSTESİ
- SİMGELER VE KISALTMALAR
- **1. GİRİŞ**
  - 1.1 Problemin Tanımı ve Motivasyonu
  - 1.2 Seçici Gürültü Engelleme ve Sorgu-Koşullu Ayrıştırma
  - 1.3 Projenin Amacı ve Kapsamı
  - 1.4 Bilimsel Katkı ve Özgün Değer
  - 1.5 Tezin Organizasyonu
- **2. LİTERATÜR TARAMASI**
  - 2.1 Ses Kaynağı Ayrıştırma Probleminin Tanımı
  - 2.2 Klasik Sinyal İşleme Yaklaşımları
  - 2.3 Zaman-Frekans Maskeleme
  - 2.4 Derin Öğrenme Tabanlı Ayrıştırma
  - 2.5 Sorgu-Koşullu ve Hedef-Yönlü Ayrıştırma
  - 2.6 Koşullandırma Mekanizmaları: FiLM
  - 2.7 Ses Olayı Tespiti
- **3. MATERYAL VE YÖNTEM**
  - 3.1 Genel Sistem Mimarisi ve Veri Akışı
  - 3.2 Ses Ön İşleme ve Spektrogram Temsili
  - 3.3 Veri Kümeleri
  - 3.4 Anlık Veri Üretimi: SeparationMixer
  - 3.5 FiLM-Koşullu U-Net Mimarisi
  - 3.6 Kayıp Fonksiyonları
  - 3.7 Optimizasyon ve Eğitim Süreci
  - 3.8 Çıkarım Hattı
- **4. BULGULAR VE TARTIŞMA**
  - 4.1 Değerlendirme Metodolojisi ve Metrikler
  - 4.2 Tasarım Kararlarının Deneysel Gerekçeleri
  - 4.3 Ayrıştırma Başarımı
  - 4.4 Tespit Başarımı
  - 4.5 FiLM Koşullandırmasının Katkısı
  - 4.6 Eşik Taraması ve Çalışma Noktası Seçimi
  - 4.7 Niteliksel Sonuçlar
  - 4.8 Sınırlılıklar ve Tartışma
- **5. SONUÇ VE ÖNERİLER**
  - 5.1 Genel Değerlendirme
  - 5.2 Gelecek Çalışmalar İçin Öneriler
- KAYNAKLAR

---

## ÖZET

### DERİN ÖĞRENME İLE SEÇİCİ GÜRÜLTÜ ENGELLEME

Bu çalışmada, bir ses ya da video kaydından seçilen ses sınıflarının, kaydın geri kalan içeriğine dokunulmadan çıkarılmasını sağlayan, derin öğrenme tabanlı bir seçici gürültü engelleme sistemi sunulmaktadır. Problem, çıkarılacak sınıfın modele bir tek-sıcak sorgu vektörüyle bildirildiği, sorgu-koşullu ve denetimli bir kaynak ayrıştırma görevi olarak biçimlendirilmiştir. Önerilen model, sınıf sorgusunun her kodlayıcı seviyesinde ve darboğazda ölçek ve öteleme parametrelerine dönüştürüldüğü, FiLM ile koşullandırılmış iki boyutlu bir U-Net mimarisidir. Model, $16$ kHz örnekleme hızında, bir saniyelik pencerelerin logaritmik genlik spektrogramları üzerinde çalışmakta ve sorgulanan sınıf için bir yumuşak maske üretmektedir. Eğitim verisi, bellek içi bir klip önbelleğinden anlık olarak sentezlenmekte; negatif örnekler, ağırlıklı zor-negatif örnekleme ve arka plan gürültüsü artırımı ile modelin seçici bastırma ve gürültü dayanıklılığı kazanması sağlanmaktadır. Sınıf varlığının kestirimi için öğrenilmiş bir tespit başı kullanılmış; eğitim süreci Adam optimize edicisi, karma hassasiyetli hesaplama ve XLA derlemesi ile hızlandırılmıştır. Sistem, iteratif bir deneysel metodolojiyle geliştirilmiş ve ayrıştırma başarımı en yüksek on beş sınıftan oluşan düzenlenmiş bir sözcük dağarcığı üzerinde tespit makro $F_1$ değeri $0{,}692$'ye ulaşmıştır. Geliştirilen uçtan uca web uygulaması, kullanıcının yüklediği dosyada bulunan sınıfları tespit etmekte ve seçilen sınıfları örtüşmeli toplama yöntemiyle dosyadan çıkarmaktadır.

**Anahtar Kelimeler:** Seçici gürültü engelleme, kaynak ayrıştırma, sorgu-koşullu öğrenme, FiLM, U-Net, derin öğrenme, ses olayı tespiti.



⟨Öğrenci Adı Soyadı⟩
Haziran, 2026



---

## ABSTRACT

### SELECTIVE NOISE CANCELLATION WITH DEEP LEARNING

This study presents a deep-learning-based selective noise cancellation system that removes chosen sound classes from an audio or video recording while leaving the rest of the content untouched. The problem is formulated as a query-conditioned, supervised source-separation task in which the class to be extracted is signalled to the model through a one-hot query vector. The proposed model is a two-dimensional U-Net conditioned with FiLM, where the class query is transformed into scale and shift parameters at every encoder level and at the bottleneck. The model operates on log-magnitude spectrograms of one-second windows at a $16$ kHz sampling rate and produces a soft mask for the queried class. Training data is synthesised on the fly from an in-memory clip cache; negative examples, weighted hard-negative sampling and background-noise augmentation enable the model to learn selective suppression and noise robustness. A learned detection head is used to estimate class presence, and the training process is accelerated with the Adam optimiser, mixed-precision computation and XLA compilation. The system was developed through an iterative experimental methodology and reached a detection macro $F_1$ score of $0.692$ on a curated vocabulary of the fifteen best-separated classes. The end-to-end web application detects the classes present in an uploaded file and removes the selected classes via an overlap-add procedure.

**Keywords:** Selective noise cancellation, source separation, query-conditioned learning, FiLM, U-Net, deep learning, sound event detection.



⟨Student Name⟩
June, 2026



---

## ŞEKİL LİSTESİ

| Şekil | Açıklama |
|---|---|
| Şekil 3.1 | Önerilen sistemin uçtan uca veri akışı şeması |
| Şekil 3.2 | On beş sınıflı sözcük dağarcığında sınıf başına klip sayısı |
| Şekil 3.3 | FiLM-koşullu U-Net mimarisi |
| Şekil 4.1 | Sınıf bazlı SI-SDRi değerleri |
| Şekil 4.2 | İşlenmemiş karışım ile model kestiriminin SI-SDR karşılaştırması |
| Şekil 4.3 | Sınıf bazlı kesinlik, duyarlılık ve $F_1$ değerleri |
| Şekil 4.4 | Toplam doğru pozitif, yanlış pozitif ve yanlış negatif sayıları |
| Şekil 4.5 | Tespit başının ROC ve kesinlik-duyarlılık eğrileri |
| Şekil 4.6 | FiLM koşullandırmasının sınıf bazlı ayrımcılık üstünlüğü |
| Şekil 4.7 | Sınıf bazlı çıkış-giriş enerji oranı |
| Şekil 4.8 | Göreli kesme ve $k$ parametrelerinin eşik taraması |
| Şekil 4.9 | Tespit puanlarının dağılımı |
| Şekil 4.10 | En sık yanlış pozitif üreten sınıflar |
| Şekil 4.11 | Sınıfların birlikte tespit edilme (co-occurrence) matrisi |
| Şekil 4.12 | Çalar saat sınıfı için spektrogramlar |
| Şekil 4.13 | Helikopter sınıfı için spektrogramlar |
| Şekil 4.14 | Elektrikli süpürge sınıfı için spektrogramlar |
| Şekil 4.15 | Çıkarma öncesi ve sonrası dalga biçimi karşılaştırması |

---

## TABLO LİSTESİ

| Tablo | Açıklama |
|---|---|
| Tablo 3.1 | Kullanılan veri kümelerinin sayısal özeti |
| Tablo 3.2 | Düzenlenmiş 15 sınıflı sözcük dağarcığının sınıf bazlı klip sayıları |
| Tablo 3.3 | Önerilen modelin eğitim hiperparametreleri |
| Tablo 4.1 | Denenen tasarım değişiklikleri ve gözlemlenen sonuçlar |
| Tablo 4.2 | Sınıf bazlı tespit ve ayrıştırma başarımı |

---

## ALGORİTMA LİSTESİ

| Algoritma | Açıklama |
|---|---|
| Algoritma 3.1 | Anlık karışım örneği üretimi (SeparationMixer) |
| Algoritma 3.2 | Koşullu ayrıştırıcının eğitim döngüsü |
| Algoritma 3.3 | Ses olayı tespiti |
| Algoritma 3.4 | Hanning örtüşmeli toplama ile kaynak çıkarma |

---

## SİMGELER VE KISALTMALAR

**Kısaltmalar**

| Kısaltma | Açıklama |
|---|---|
| ANC | Aktif Gürültü Engelleme (Active Noise Cancellation) |
| BCE | İkili Çapraz Entropi (Binary Cross-Entropy) |
| CASA | Hesaplamalı İşitsel Sahne Analizi |
| CRNN | Evrişimsel-Yinelemeli Sinir Ağı |
| FFT | Hızlı Fourier Dönüşümü |
| FiLM | Özellik-Bazlı Doğrusal Modülasyon (Feature-wise Linear Modulation) |
| ISTFT | Ters Kısa Zamanlı Fourier Dönüşümü |
| JIT | Tam-Zamanında Derleme (Just-In-Time) |
| MMSE | En Küçük Ortalama Karesel Hata |
| NMF | Negatif Olmayan Matris Ayrıştırması |
| OLA | Örtüşmeli Toplama (Overlap-Add) |
| ReLU | Doğrultulmuş Doğrusal Birim |
| RMS | Karekök-Ortalama-Kare |
| ROC | Alıcı İşletim Karakteristiği |
| SED | Ses Olayı Tespiti (Sound Event Detection) |
| SI-SDR | Ölçek-Değişmez İşaret-Bozulma Oranı |
| SI-SDRi | SI-SDR İyileştirmesi (improvement) |
| SNR | İşaret-Gürültü Oranı |
| STFT | Kısa Zamanlı Fourier Dönüşümü |
| XLA | Hızlandırılmış Doğrusal Cebir (Accelerated Linear Algebra) |

**Simgeler**

| Simge | Açıklama |
|---|---|
| $\gamma, \beta$ | FiLM ölçek ve öteleme parametreleri |
| $\lambda$ | Çıkarma kuvveti katsayısı |
| $P_{\text{negatif}}$ | Negatif örnek olasılığı |
| $P_{\text{gürültü}}$ | Arka plan gürültüsü olasılığı |
| $N_{\min}$ | Sınıf başına minimum klip sayısı |
| $\eta$ | Öğrenme oranı |
| $\beta_1, \beta_2$ | Adam moment katsayıları |
| $M_c$ | $c$ sınıfı için yumuşak maske |

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 1. GİRİŞ

Akustik ortamlar, eş zamanlı olarak etkin olan çok sayıda ses kaynağının doğrusal biçimde üst üste binmesiyle oluşan bileşik sinyaller barındırır. Bir video kaydının arka planında çalan siren, bir telekonferans oturumuna karışan klavye tıkırtısı ya da bir saha kaydının üzerine binen helikopter uğultusu; hedef içeriğin anlaşılırlığını düşüren, yapısal ve geniş bantlı bozucu bileşenler olarak gözlemlenmektedir. Söz konusu bileşenlerin, kaydın geri kalan içeriğine dokunulmaksızın ayıklanması, klasik sinyal işleme çerçevelerinin tek başına çözmekte yetersiz kaldığı koşullu bir kaynak ayrıştırma problemi olarak ele alınmıştır. Bu bölümde problemin biçimsel tanımı ve mühendislik motivasyonu ortaya konmuş, seçici gürültü engellemenin sorgu-koşullu ayrıştırma olarak modellenişi açıklanmış, projenin amacı, kapsamı ve özgün katkıları belirtilmiş ve tezin bölüm organizasyonu sunulmuştur.

## 1.1 Problemin Tanımı ve Motivasyonu

Tek kanallı bir kayıtta gözlemlenen $x(t)$ karışım sinyali, izole kaynak imzaları $s_i(t)$ cinsinden toplamsal bir model ile ifade edilmektedir:

$$x(t) = \sum_{i=1}^{K} s_i(t),$$

burada $K$, o an etkin olan ses kaynağı sayısını göstermektedir. Bu modelde bozucu kaynaklar ile korunması istenen kaynaklar, aynı genlik mertebesinde ve örtüşen zaman-frekans bölgelerinde bulunabildiğinden, ayrıştırma probleminin kötü konumlanmış (ill-posed) doğası belirginleşmektedir.

Geleneksel gürültü engelleme yöntemleri, bozucu bileşeni durağan (stationary) ya da yavaş değişen istatistiksel bir artık olarak modelleme varsayımına dayanmaktadır. Aktif gürültü engelleme (ANC) donanımları, bozucu dalga biçiminin ters fazlı bir kopyasını üreterek akustik alanda yıkıcı girişim oluşturur; spektral çıkarma ve Wiener süzgeci gibi tekil kanallı yöntemler ise gürültü güç spektrumunu kestirip karışım spektrumundan çıkarır. Bu yaklaşımların ortak kısıtı, "gürültü" kavramının içeriğinden bağımsız bir taban düzeyi olarak ele alınmasıdır. Dolayısıyla, konuşma kadar yapısal ve geniş bantlı bir bozucu sesin (örneğin köpek havlaması veya siren) hedef sinyalden ayrıştırılması, bu çerçevelerin varsayımları dışında kalmaktadır. Durağanlık varsayımı geçersizleştiğinde, spektral çıkarma yöntemleri "müzikal gürültü" olarak adlandırılan yapaylıklar üretmekte ve hedef sinyalin spektral içeriğine zarar vermektedir.

Pratik gereksinim ise farklı bir formülasyon gerektirmektedir: bozucu bileşenin yalnızca bir taban düzeyi olarak değil, belirli bir *ses sınıfı* olarak tanımlanması ve bu sınıfın karışımdan seçici biçimde bastırılması beklenmektedir. Video sonrası prodüksiyonunda belirli bir çevresel sesin temizlenmesi, işitme destek sistemlerinde rahatsız edici sınıfların zayıflatılması ve telekonferans uygulamalarında ortam seslerinin ayıklanması, bu seçici bastırma yeteneğine duyulan ihtiyacı somutlaştıran kullanım durumlarıdır. Bu çalışma kapsamında problem, "karışımdan hangi sınıfın çıkarılacağı" bilgisinin modele dışarıdan verildiği, koşullu ve denetimli bir kaynak ayrıştırma görevi olarak biçimlendirilmiştir.

## 1.2 Seçici Gürültü Engelleme ve Sorgu-Koşullu Ayrıştırma

Toplamsal karışım modelinin zaman-frekans düzlemindeki karşılığı, Kısa Zamanlı Fourier Dönüşümü (STFT) altında yine toplamsaldır:

$$X(f, \tau) = \sum_{i=1}^{K} S_i(f, \tau),$$

burada $f$ frekans bini, $\tau$ ise zaman çerçevesi indisini göstermektedir. Maskeleme tabanlı ayrıştırma, hedef kaynağın genlik spektrogramını, $[0, 1]$ aralığında değer alan bir yumuşak maske $M_c(f, \tau)$ ile karışım genliğinin çarpımı olarak kestirir:

$$\hat{S}_c(f, \tau) = M_c(f, \tau) \odot |X(f, \tau)|.$$

Bu formülasyon üzerine iki ayrı mimari kurgu inşa edilebilmektedir. Birinci kurgu, sabit çok çıkışlı (multi-output) bir modeldir: ağ, her sınıf için ayrı bir maske kanalı üreterek tek geçişte tüm kaynakları kestirir. Bu kurgunun üç temel kısıtı bulunmaktadır. Öncelikle, çıkış katmanının genişliği sınıf sayısı ile doğru orantılı olarak büyümekte, dolayısıyla yeni bir veri kümesi eklenmesi mimarinin yeniden tasarlanmasını gerektirmektedir. İkinci olarak, ağır bir sınıf dengesizliği ortaya çıkmaktadır: $K$ kaynaklı bir karışımda toplam sınıf sayısı $N$ ise, $N - K$ kanalın hedefi sessizliktir; elli sınıflı bir sözcük dağarcığında çoğu eğitim örneği için kanalların büyük bölümü yalnızca sessizliği öngörmeye zorlanmakta, bu da öğrenme sinyalini seyreltmektedir. Üçüncü olarak, önceden üretilmiş hedef stem dosyalarının diskte saklanması, elli sınıf ve saniyede $16\,000$ örnek için yaklaşık $19$ GB mertebesinde bir depolama yükü doğurmaktadır.

İkinci kurgu, bu çalışmada benimsenen sorgu-koşullu (query-conditioned) yaklaşımdır. Bu yaklaşımda ağa, $(256, 128, 1)$ boyutlu logaritmik genlik spektrogramının yanı sıra, çıkarılması istenen sınıfı seçen bir tek-sıcak (one-hot) sorgu vektörü $q \in \{0, 1\}^{N}$ verilmektedir. Ağ, yalnızca sorgulanan sınıf için tek bir maske üretir. Bu tasarımda sınıf sayısı yalnızca sorgu girişinin genişliğini etkilemekte; evrişimli gövde ise sınıf sayısından bağımsız kalmaktadır. Böylece aynı mimari, sekiz sınıflı bir kümeden yüzlerce sınıflı bir sözcük dağarcığına kadar yeniden eğitime gerek kalmadan ölçeklenebilmektedir. Birden çok sesin çıkarılması gerektiğinde, model her hedef sınıf için bir kez sorgulanmaktadır. Bu kurgu, hem sınıf dengesizliği sorununu hem de depolama yükünü yapısal olarak ortadan kaldırmaktadır; çünkü eğitim örnekleri bir veri dosyasından okunmak yerine bellek içi bir klip önbelleğinden anlık olarak sentezlenmektedir.

## 1.3 Projenin Amacı ve Kapsamı

Bu projenin amacı, sorgulanan bir ses sınıfı için yumuşak spektrogram maskesi üreten, FiLM (Feature-wise Linear Modulation) ile koşullandırılmış iki boyutlu bir U-Net mimarisi tasarlamak, eğitmek ve bu modeli uçtan uca bir gürültü temizleme uygulamasında işlevsel kılmaktır. Tüm ses verileri ortak bir spektrogram sözleşmesine indirgenmiştir: sinyaller $16$ kHz örnekleme hızında ve tek kanala (mono) dönüştürülmüş, STFT parametreleri $n_{\text{fft}} = 512$ ve sıçrama uzunluğu $hop = 128$ olarak belirlenmiştir. Nyquist bini düşürülerek modele $256$ frekans bini sunulmuş, zaman ekseni bir saniyelik pencereye karşılık gelen $128$ çerçeveye sabitlenmiştir. Buna göre her model çağrısı yaklaşık bir saniyelik bir akustik bağlamı işlemektedir.

Sistemin işlevsel kapsamı, dört aşamalı bir çıkarım zinciri ile tanımlanmıştır: kullanıcının ses veya video dosyası yüklemesi, dosyada bulunan ses sınıflarının tespit edilmesi, kullanıcının çıkarılmasını istediği sınıfları işaretlemesi ve temizlenmiş çıktının üretilip indirilmesi. Video girişlerinde, temizlenen ses izi `ffmpeg` aracılığıyla özgün görüntü izinin üzerine yeniden bindirilmektedir. Önerilen model, ayrıştırma ve tespit başarımının en yüksek olduğu on beş sınıftan oluşan, düzenlenmiş (curated) bir sözcük dağarcığı üzerinde çalışmaktadır.

Çalışmanın kapsamı, genlik spektrogramı üzerinde maskeleme ve karışım fazının yeniden kullanımı ilkesiyle sınırlandırılmıştır. Faz bilgisi modelce kestirilmediğinden, yeniden sentez aşamasında karışımın özgün fazı korunmaktadır; bu seçim, sayısal kararlılık ve hesaplama maliyeti açısından bir mühendislik ödünleşimi olarak benimsenmiştir. Dalga biçimi düzleminde uçtan uca ayrıştırma yapan modeller ile faz-duyarlı kestirim teknikleri kapsam dışında bırakılmış; bu yöntemler ileride yapılması önerilen çalışmalar arasında değerlendirilmiştir.

## 1.4 Bilimsel Katkı ve Özgün Değer

Bu tez çalışması kapsamında üretilen özgün katkılar aşağıda sıralanmıştır.

**Sınıf sayısından bağımsız mimari.** Sorgu-koşullu tasarım sayesinde evrişimli gövde sabit tutulmuş, sınıf sayısı yalnızca sorgu vektörünün boyutunu belirleyen bir parametreye indirgenmiştir. Böylece veri kümesi genişletildiğinde mimari değişmeden kalmaktadır.

**Anlık karışım üretimi.** Eğitim örnekleri, önceden üretilmiş bir veri dosyasından okunmak yerine, bellek içi klip önbelleğinden her adımda rastgele sentezlenmektedir. Bu yaklaşım hem depolama yükünü ortadan kaldırmakta hem de görece sınırsız bir karışım çeşitliliği sağlamaktadır.

**Çok seviyeli FiLM koşullandırması.** Sınıf sorgusu, paylaşılan bir gömme katmanından geçirilip her kodlayıcı seviyesinde ve darboğazda ayrı $\gamma$ (ölçek) ve $\beta$ (öteleme) parametrelerine dönüştürülmektedir. Koşullandırmanın yalnızca darboğazda değil tüm kodlayıcı seviyelerinde uygulanması, atlama bağlantılarının da sınıfa özgü etkinlikler taşımasını sağlayarak maske kesinliğini artırmaktadır.

**Öğrenilmiş tespit başı.** Sınıf varlığının maske enerjisine dayalı bir sezgisel ile kestirilmesi yerine, FiLM ile koşullandırılmış darboğaz üzerinden $P(\text{sorgulanan sınıf mevcut} \mid \text{karışım})$ olasılığını üreten hafif bir sınıflandırma başı eğitilmiştir. Bu baş, geniş bantlı sınıfların yarattığı yapısal yanlış pozitif sorununu hafifletmektedir.

**İteratif deneysel metodoloji.** Model, bir dizi denetimli deney aracılığıyla iteratif bir biçimde geliştirilmiş; her aşamada gözlemlenen başarısızlık örüntüleri (aşırı veri artırımının yol açtığı eğitim çöküşü, dış veri kümesi kaynaklı fantom yanlış pozitifler ve odak kaybının yol açtığı gradyan çöküşü) çözümlenip bir sonraki tasarım kararına yansıtılmıştır. Bu metodoloji, tespit makro $F_1$ ölçütünün $0,692$ değerine, doğru pozitif, yanlış pozitif ve yanlış negatif sayılarının ise sırasıyla $433$, $34$ ve $299$ değerlerine ulaşmasını sağlamıştır.

## 1.5 Tezin Organizasyonu

Tezin geri kalanı dört ana bölümden oluşmaktadır. İkinci bölümde, ses kaynağı ayrıştırma problemine ilişkin klasik sinyal işleme yaklaşımları, zaman-frekans maskeleme teknikleri, derin öğrenme tabanlı ayrıştırma modelleri, sorgu-koşullu ayrıştırma ve FiLM koşullandırma mekanizması ile ses olayı tespiti konularını kapsayan literatür taraması sunulmuştur. Üçüncü bölümde, ses ön işleme ve spektrogram temsili, kullanılan veri kümeleri, anlık karışım üreteci, FiLM-koşullu U-Net mimarisi, kayıp fonksiyonları, optimizasyon süreci (Adam optimize edicisi, karma hassasiyetli eğitim ve XLA derlemesi) ve çıkarım hattı ayrıntılı olarak açıklanmıştır. Dördüncü bölümde, değerlendirme metrikleri tanımlanmış, tasarım kararlarının deneysel gerekçeleri çözümlenmiş ve ayrıştırma ile tespit başarımına ilişkin niceliksel ve niteliksel bulgular tartışılmıştır. Beşinci bölümde ise elde edilen sonuçlar genel olarak değerlendirilmiş ve gelecekte yapılması önerilen çalışmalar belirtilmiştir.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 2. LİTERATÜR TARAMASI

Bu bölümde, seçici gürültü engellemenin dayandığı ses kaynağı ayrıştırma probleminin kuramsal temelleri ve ilgili yöntem aileleri incelenmiştir. Önce problemin biçimsel tanımı ve sınıflandırması verilmiş; ardından klasik sinyal işleme yaklaşımları, zaman-frekans maskeleme teknikleri ve derin öğrenme tabanlı ayrıştırma modelleri sırasıyla ele alınmıştır. Devamında, bu çalışmanın mimari temelini oluşturan sorgu-koşullu ayrıştırma paradigması ile FiLM koşullandırma mekanizması açıklanmış ve son olarak ses olayı tespiti alanyazını özetlenmiştir.

## 2.1 Ses Kaynağı Ayrıştırma Probleminin Tanımı

Ses kaynağı ayrıştırma (audio source separation), gözlemlenen bir karışım sinyalinden, onu oluşturan bağımsız kaynak sinyallerinin geri kestirilmesi problemi olarak tanımlanmaktadır. Problemin kavramsal kökeni, insan işitme sisteminin çok konuşmacılı ve gürültülü ortamlarda tek bir kaynağa odaklanabilme yeteneğini betimleyen "kokteyl parti problemi" olgusuna dayandırılmaktadır [1]. İşitsel sistemin görece kolaylıkla başardığı bu seçici dinleme yeteneğinin algoritmik olarak modellenmesi, sinyal işleme ve makine öğrenmesi alanlarının ortak çalışma konularından biri olmuştur.

Problem, gözlem sayısı ile kaynak sayısı arasındaki ilişkiye göre sınıflandırılmaktadır. $M$ adet gözlem kanalı ve $N$ adet kaynak için, $M \geq N$ durumu *belirli* (determined) ya da *aşırı belirli* (overdetermined) probleme; $M < N$ durumu ise *eksik belirli* (underdetermined) probleme karşılık gelmektedir. Bu çalışmanın da konusunu oluşturan tek kanallı (monaural, $M = 1$) ayrıştırma, eksik belirli problemin en uç hâlidir: tek bir denklemle birden çok bilinmeyenin çözülmesi gerektiğinden, problem ancak kaynakların yapısına ilişkin güçlü ön bilgilerin (priors) ya da veriden öğrenilmiş istatistiksel modellerin devreye sokulmasıyla anlamlı biçimde çözülebilmektedir.

Yöntemler, kaynaklara ilişkin ön bilginin türüne göre de ayrılmaktadır. *Kör kaynak ayrıştırma* (blind source separation), kaynaklar hakkında en az varsayımla (örneğin istatistiksel bağımsızlık) çalışan ve bağımsız bileşen analizi gibi tekniklerle temsil edilen yaklaşımdır [2], [3]. Buna karşılık *bilgilendirilmiş* (informed) ya da *denetimli* (supervised) ayrıştırma, hedef kaynağın sınıfı, kimliği veya örnek kayıtları gibi yardımcı bilgilerden yararlanmaktadır. Bu çalışmada benimsenen sorgu-koşullu kurgu, çıkarılması istenen sınıfın tek-sıcak bir vektörle modele bildirildiği denetimli ayrıştırma ailesine girmektedir; dolayısıyla problem, kör ayrıştırmanın belirsizliklerinden (özellikle çıkışların hangi kaynağa karşılık geldiğinin bilinmemesi anlamına gelen permütasyon belirsizliğinden) yapısal olarak arındırılmıştır.

Tek kanallı ayrıştırmanın kuramsal çerçevelerinden biri, insan işitsel sahne çözümlemesini hesaplamalı olarak modelleyen Hesaplamalı İşitsel Sahne Analizi (Computational Auditory Scene Analysis) yaklaşımıdır [2]. Bu çerçevede ayrıştırma, zaman-frekans düzlemindeki enerji birikimlerinin ortak başlangıç, harmonik ilişki ve zamansal süreklilik gibi gruplama ilkelerine göre kaynaklara atanması olarak ele alınmaktadır. Veriye dayalı derin öğrenme yöntemlerinin yükselişiyle birlikte problem, elle tasarlanmış gruplama kuralları yerine, çok sayıda karışım–kaynak örneği üzerinden bir eşleme fonksiyonunun (genellikle bir zaman-frekans maskesinin) doğrudan öğrenilmesi biçiminde yeniden çerçevelenmiştir.

Ayrıştırma başarımının niceliksel olarak ölçülmesi de ayrı bir araştırma konusu olmuştur. Kestirilen kaynağın referans kaynağa olan benzerliği, sinyalin hedef bileşeni ile bozulma bileşeni arasındaki güç oranına dayanan ölçütlerle değerlendirilmektedir [4]. Bu ölçütlerin ölçek değişimlerine karşı duyarsız hâle getirilmiş türevleri, bu tezin dördüncü bölümünde ayrıntılandırılan ölçek-değişmez işaret-bozulma oranının (SI-SDR) temelini oluşturmaktadır. İzleyen alt başlıklarda, problemin tarihsel ve yöntemsel gelişimi; klasik sinyal işleme yaklaşımlarından (Alt Başlık 2.2) başlanarak zaman-frekans maskeleme (Alt Başlık 2.3) ve derin öğrenme tabanlı modeller (Alt Başlık 2.4) üzerinden, bu çalışmanın temelini oluşturan sorgu-koşullu ayrıştırmaya (Alt Başlık 2.5) doğru ele alınmaktadır.

## 2.2 Klasik Sinyal İşleme Yaklaşımları

Derin öğrenme yöntemlerinin yaygınlaşmasından önce, tek kanallı gürültü engelleme ve kaynak ayrıştırma problemleri ağırlıklı olarak istatistiksel sinyal işleme yöntemleriyle ele alınmıştır. Bu yöntemler, kestirilecek niceliğe ilişkin açık olasılıksal varsayımlara ve elle tasarlanmış spektral işlemlere dayanmaktadır. Ortak özellikleri, gürültünün ya da bozucu kaynağın istatistiklerinin görece durağan kabul edilmesi ve kestirimin bir kapalı-biçim (closed-form) ifadeyle ya da yinelemeli bir eniyilemeyle yapılmasıdır.

**Spektral çıkarma.** Spektral çıkarma yöntemi, gürültünün güç spektral yoğunluğunun konuşma içermeyen (suskun) bölümlerden kestirildiği ve bu kestirimin karışım spektrumundan çıkarıldığı bir işlemdir [5]. Güç düzleminde kestirim, bir spektral taban katsayısı $\beta$ ile birlikte

$$|\hat{S}(f,\tau)|^2 = \max\!\big(|X(f,\tau)|^2 - \mathbb{E}\big[|N(f)|^2\big],\; \beta\,|X(f,\tau)|^2\big)$$

biçiminde yazılmaktadır. Negatif değerlerin kırpılması, ardışık çerçeveler arasında rastgele konumlanan ve "müzikal gürültü" olarak adlandırılan dar bantlı yapay tonlar üretmektedir. Yöntem, ancak gürültü istatistiklerinin yavaş değiştiği durumlarda kabul edilebilir sonuç vermekte; konuşma kadar yapısal ve zamanla hızla değişen bir bozucu kaynak söz konusu olduğunda, gürültü güç spektrumunun kestirimi geçersizleşmektedir.

**Wiener süzgeci ve MMSE kestiriciler.** Wiener süzgeci, hedef ve gürültü güç spektral yoğunlukları bilindiğinde, ortalama karesel hata ölçütünde eniyi olan doğrusal kazanç fonksiyonunu vermektedir:

$$G(f,\tau) = \frac{\xi(f,\tau)}{1 + \xi(f,\tau)}, \qquad \xi(f,\tau) = \frac{P_S(f,\tau)}{P_N(f,\tau)},$$

burada $\xi$ önsel işaret-gürültü oranını (a priori SNR), $P_S$ ve $P_N$ ise sırasıyla hedef ve gürültü güç spektral yoğunluklarını göstermektedir. Kısa zamanlı genlik spektrumunun en küçük ortalama karesel hatalı kestirimini logaritmik ve algısal ölçütlerle iyileştiren türevler de önerilmiştir [6]. Bu kestiricilerin ortak kısıtı, $P_S$ ve $P_N$ niceliklerinin önceden bilinmesini ya da güvenilir biçimde kestirilmesini gerektirmeleridir; her iki nicelik de durağanlık varsayımı altında kestirildiğinden, geniş bantlı ve durağan olmayan bozucu kaynaklarda kazanç fonksiyonu hatalı hesaplanmaktadır.

**Negatif olmayan matris ayrıştırması.** Negatif olmayan matris ayrıştırması (Non-negative Matrix Factorization), negatif olmayan bir genlik spektrogramı matrisi $V \in \mathbb{R}_+^{F \times T}$ değerini, bir spektral sözlük $W \in \mathbb{R}_+^{F \times R}$ ile etkinlik matrisi $H \in \mathbb{R}_+^{R \times T}$ çarpımına yaklaştıran bir düşük-ranklı modeldir [7]:

$$V \approx W H.$$

Çarpanlar, bir uzaklık ölçütünün (örneğin Kullback–Leibler ıraksaması) çarpımsal güncelleme kurallarıyla azaltılmasıyla kestirilmektedir. Denetimli kaynak ayrıştırmada her kaynak için ayrı bir spektral sözlük önceden öğrenilmekte, karışım bu sözlüklerin birleşimiyle çözümlenmekte ve etkinlikler kaynaklara göre gruplanarak ayrıştırma gerçekleştirilmektedir [8]. Bu çerçeve, harmonik yapısı belirgin müzik sinyallerinde başarılı sonuçlar vermekle birlikte, doğrusal ve sabit bir baz varsayımına dayandığından, benzer tınıya sahip ya da geniş bantlı çevresel seslerin örtüştüğü karışımlarda ayırt edici bir çözüm üretememektedir. Ayrıca sözlüklerin temiz, izole kayıtlardan öğrenilmesi gerekliliği, yöntemin gerçek dünya verisine genellenmesini sınırlandırmaktadır.

**Klasik yaklaşımların ortak kısıtı.** Bu yöntemlerin tümü, sığ (shallow) ve elle tasarlanmış temsillere, durağanlık varsayımına ve doğrusal modellere dayanmaktadır. Bir ses sınıfını tanımlayan yüksek düzeyli, doğrusal olmayan ve bağlama duyarlı spektro-zamansal örüntüler, bu çerçevelerin temsil gücünün dışında kalmaktadır. Söz konusu sınır, kaynak ayrıştırma probleminin, çok sayıda karışım–kaynak örneğinden doğrusal olmayan bir eşleme fonksiyonunun öğrenildiği veriye dayalı derin öğrenme yöntemleriyle yeniden ele alınmasını gerektirmiştir. Bu yöntemlerin temelini oluşturan zaman-frekans maskeleme kavramı Alt Başlık 2.3'te, maskeyi kestiren derin ağ mimarileri ise Alt Başlık 2.4'te incelenmiştir.

## 2.3 Zaman-Frekans Maskeleme

Tek kanallı denetimli ayrıştırmada baskın yaklaşım, problemi doğrudan kaynak spektrumunu kestirmek yerine, karışımın zaman-frekans temsiline uygulanacak bir *maske* $M(f,\tau)$ kestirmeye indirgemektir. Maske, her zaman-frekans hücresinde hedef kaynağa ait enerji oranını belirtmekte; kestirilen kaynak ise maskenin karışımla noktasal çarpımıyla elde edilmektedir. Bu çerçevenin avantajı, öğrenme hedefinin sınırlı ve iyi koşullanmış bir aralıkta ($[0,1]$) tanımlanması ve karışımın halihazırda taşıdığı spektral yapının yeniden üretilmesine gerek kalmamasıdır. Alanyazında birbirinden türeyen birkaç maske tanımı önerilmiştir.

**İdeal ikili maske.** İdeal ikili maske (Ideal Binary Mask), her zaman-frekans hücresini, yerel işaret-gürültü oranının bir eşik $\theta$ değerini aşıp aşmamasına göre $\{0,1\}$ değerlerinden birine atayan sert (hard) bir maskedir [9]:

$$M_{\text{IBM}}(f,\tau) = \begin{cases} 1, & \text{SNR}(f,\tau) > \theta \\ 0, & \text{aksi hâlde.} \end{cases}$$

Bu tanım, işitsel maskeleme olgusundan esinlenmiş ve ayrıştırılmış sinyalin anlaşılırlığını yükselttiği gösterilmiştir. Ancak ikili karar yapısı, hücre sınırlarında ani geçişlere ve spektral çıkarmadakine benzer yapay tonlara yol açmaktadır.

**İdeal oran maskesi.** İdeal oran maskesi (Ideal Ratio Mask), sert kararı yumuşatarak hedef ve gürültü güçlerinin oranına dayalı sürekli bir değer atamaktadır:

$$M_{\text{IRM}}(f,\tau) = \left(\frac{P_S(f,\tau)}{P_S(f,\tau) + P_N(f,\tau)}\right)^{\beta},$$

burada $\beta$ bir biçimlendirme üssüdür. Bu ifade, $\beta = 1$ için Wiener kazanç fonksiyonuyla yapısal benzerlik taşımakta; ancak burada oran, öğrenilmiş bir ağ tarafından kestirilmektedir [10]. Oran maskesi, sürekli ve sınırlı doğası nedeniyle bu çalışmada benimsenen yumuşak maske kurgusunun da temelini oluşturmaktadır.

**Genlik maskesi ve faz duyarlılığı.** Genlik maskesi (Spectral Magnitude Mask), hedef ile karışım genliklerinin doğrudan oranı $M_{\text{MM}}(f,\tau) = |S(f,\tau)| / |X(f,\tau)|$ olarak tanımlanır ve örtüşen kaynaklarda $1$ değerini aşabilir. Bu tanımların tümü, yeniden sentez sırasında karışımın fazını kullanmakta; oysa hedef ile karışım arasındaki faz farkı, genlik düzeyinde dahi bir hataya yol açmaktadır. Faz duyarlı maske (Phase-Sensitive Mask), bu farkı kosinüs terimiyle hesaba katmaktadır [11]:

$$M_{\text{PSM}}(f,\tau) = \frac{|S(f,\tau)|}{|X(f,\tau)|}\,\cos\!\big(\theta_S(f,\tau) - \theta_X(f,\tau)\big).$$

Karmaşık oran maskesi (Complex Ideal Ratio Mask) ise maskenin gerçek ve sanal bileşenlerini ayrı ayrı kestirerek faz bilgisini de geri kazanmayı amaçlamaktadır [12]. Bu iki yaklaşım, faz yeniden yapılandırmasının getirdiği başarım artışını sağlamakla birlikte, kestirim hedefini karmaşık düzleme taşıdığından eğitim kararlılığı ve hesaplama maliyeti açısından ek yük getirmektedir.

Kuramsal olarak, ideal oran maskesi, hedef ile gürültünün istatistiksel bağımsızlığı varsayımı altında kestirilen genlik ile gerçek genlik arasındaki ortalama karesel hatayı en aza indiren maskeye karşılık gelmekte ve $\beta = 1$ için Wiener süzgeciyle özdeşleşmektedir. Yer-gerçek istatistiklerinden hesaplanan oracle maskeler, öğrenilebilir herhangi bir modelin ulaşabileceği başarımın bir üst sınırını (performans tavanını) tanımlamakta; uygulamada ağ, bu oracle maskeyi yalnızca karışımdan kestirmeye çalışmaktadır. Maske tabanlı yaklaşımın bir başka kuramsal kısıtı faz yeniden kullanımıdır: yalnızca genlik üzerinde tanımlı bir maske, en iyi durumda dahi hedef ile karışımın fazı farklı olduğunda bir artık hata bırakmaktadır; çünkü karmaşık düzlemde iki vektörün genlikleri eşitlense bile faz farkı bir uzaklık terimi doğurmaktadır.

Bu tez çalışmasında, hedef sınıfın genlik spektrogramı için $[0,1]$ aralığında değer alan yumuşak bir maske kestirilmekte ve yeniden sentez aşamasında karışımın özgün fazı korunmaktadır. Dolayısıyla benimsenen kurgu, oran maskesi ve genlik maskesi ailesine girmekte; faz duyarlı ve karmaşık maske türevleri, ilgili başarım–maliyet ödünleşimi gerekçesiyle kapsam dışında bırakılmaktadır. Maskenin oracle bir işaret-gürültü oranından hesaplanması yerine, bir derin sinir ağı tarafından doğrudan öğrenilmesi söz konusudur; bu ağların mimari gelişimi Alt Başlık 2.4'te ele alınmıştır.

## 2.4 Derin Öğrenme Tabanlı Ayrıştırma

Zaman-frekans maskesinin oracle istatistiklerden hesaplanması yerine veriden öğrenilmesi fikri, derin sinir ağlarının ayrıştırma problemine uygulanmasıyla olgunlaşmıştır. İlk çalışmalar, karışım spektrogramından hedef maskeyi ya da hedef genliği kestiren ileri beslemeli ve yinelemeli ağları kullanmıştır [13]. Bu modeller, klasik yöntemlerin durağanlık varsayımını ortadan kaldırarak sınıfa özgü spektro-zamansal örüntülerin örtük biçimde öğrenilmesini sağlamış; ancak çok kaynaklı çıkışlarda, kestirilen kaynakların hangi referansa karşılık geldiğinin belirsiz olması anlamına gelen permütasyon problemiyle karşılaşmıştır.

Permütasyon problemine iki temel çözüm önerilmiştir. Derin kümeleme (deep clustering), her zaman-frekans hücresini bir gömme uzayına eşleyip aynı kaynağa ait hücreleri kümeleyerek ayrıştırmayı bir gömme öğrenme problemine dönüştürmektedir [14]. Permütasyondan bağımsız eğitim (Permutation Invariant Training) ise, kayıp hesaplanırken çıkış–referans eşlemelerinin tüm olası permütasyonları arasından en düşük kayıplı olanın seçilmesi ilkesine dayanmaktadır [15]. Bu çalışmada benimsenen sorgu-koşullu kurgu, tek bir hedef çıkışı kestirdiğinden permütasyon problemini yapısal olarak ortadan kaldırmakta ve bu tekniklere gerek bırakmamaktadır.

Mimari açıdan, biyomedikal görüntü bölütlemesi için önerilen U-Net mimarisi [16], kodlayıcı–kod çözücü yapısı ve atlama bağlantıları sayesinde hem yüksek düzeyli bağlamı hem de ince çözünürlüklü ayrıntıyı koruyabildiğinden spektrogram tabanlı ayrıştırmaya uyarlanmıştır. Şarkıcı sesi ayrıştırmasında U-Net'in genlik spektrogramı üzerinde uygulanması, maske tabanlı ayrıştırmada referans bir başarım düzeyi belirlemiştir [17]. Bunun yanı sıra, doğrudan dalga biçimi düzleminde çalışan Conv-TasNet [18] ve Demucs [19] gibi modeller, öğrenilmiş bir analiz–sentez süzgeç takımı kullanarak zaman-frekans maskelemenin ideal başarımını dahi aşmış; ancak bu kazanç, hesaplama maliyeti ve eğitim kararlılığı açısından ek yük getirmiştir. Konuşma ötesinde rastgele çevresel seslerin ayrıştırılmasını hedefleyen evrensel ses ayrıştırma (universal sound separation) çalışmaları ise, bu çalışmanın da kapsamına giren açık sözcük dağarcıklı ayrıştırma problemini tanımlamıştır [20]. Bu tez kapsamında, başarım–maliyet ödünleşimi ve sayısal kararlılık gerekçeleriyle, atlama bağlantılı bir spektrogram U-Net'i üzerinde genlik maskeleme yaklaşımı benimsenmiş ve bu mimari, izleyen başlıklarda açıklanan sorgu koşullandırmasıyla genişletilmiştir.

## 2.5 Sorgu-Koşullu ve Hedef-Yönlü Ayrıştırma

Çok kaynaklı ayrıştırmada çıkış sayısının sabit olması ve permütasyon belirsizliği, sınıf sayısının önceden bilinmediği açık sözcük dağarcıklı senaryolarda yapısal bir kısıt oluşturmaktadır. Hedef-yönlü (target-oriented) ayrıştırma, bu kısıtı, ağı çıkarılması istenen kaynağı tanımlayan bir ipucuyla (cue) koşullandırarak aşmaktadır. Bu yaklaşımda model, karışımdaki tüm kaynakları aynı anda ayrıştırmak yerine, yalnızca ipucunun işaret ettiği hedefi kestirmektedir.

Hedef ipucunun türüne göre iki ana kol gelişmiştir. Hedef konuşmacı çıkarımında (target speaker extraction), ipucu olarak hedef konuşmacıya ait bir kayıttan elde edilen gömme vektörü kullanılmaktadır; SpeakerBeam [21] ve VoiceFilter [22], bu gömmeyi ağa katarak yalnızca ilgili konuşmacının sesini çıkarmaktadır. İkinci kolda, ipucu hedef ses sınıfını gösteren bir etiket ya da tek-sıcak vektördür; sınıf-koşullu evrensel ses seçici çalışmaları, bu kurguyla rastgele çevresel ses türlerinin karışımdan çekilebileceğini göstermiştir [23]. Bu çalışmada benimsenen kurgu, ikinci kola, yani sınıf-koşullu hedef-yönlü ayrıştırmaya karşılık gelmektedir: hedef sınıf, model sözcük dağarcığı üzerindeki tek-sıcak bir sorgu vektörüyle bildirilmekte ve model bu sınıf için tek bir maske üretmektedir. Bu seçim, hem çıkış sayısını sınıf sayısından bağımsız kılmakta hem de permütasyon problemini ortadan kaldırmaktadır.

## 2.6 Koşullandırma Mekanizmaları: FiLM

Sorgu vektörünün ağın iç hesaplamasına dahil edilme biçimi, sorgu-koşullu ayrıştırmanın başarımını belirleyen tasarım kararlarından biridir. En yalın yöntem, sorgu vektörünün ara katman etkinliklerine eklenmesidir (concatenation); ancak bu yöntem, koşullandırma sinyalinin etkisini katmanlar boyunca seyreltmekte ve evrişimsel etkinlikler üzerinde doğrudan, kanal-bazlı bir denetim sağlayamamaktadır. Özellik-bazlı doğrusal modülasyon (Feature-wise Linear Modulation, FiLM), bu sınırı, koşullandırma girdisinden türetilen kanal-bazlı bir afin dönüşümle aşmaktadır [24]:

$$\mathrm{FiLM}(x_c) = \gamma_c \cdot x_c + \beta_c,$$

burada $x_c$, $c$ kanalındaki etkinlik haritasını; $\gamma_c$ ve $\beta_c$ ise koşullandırma girdisinden (bu çalışmada sınıf sorgusu gömmesinden) öğrenilen ölçek ve öteleme katsayılarını göstermektedir. FiLM, koşullu örnek normalizasyonu ve koşullu toplu normalizasyon gibi öncül tekniklerin [25], [26] genelleştirilmiş bir biçimidir; hesaplama yükü düşüktür, koşullandırma yolunu ana evrişimsel yoldan ayrıştırır ve ağın birden çok derinliğinde uygulanabilir. Müzik kaynağı ayrıştırmasında U-Net'in FiLM ile koşullandırıldığı Conditioned-U-Net mimarisi, tek bir ağın denetim girdisiyle farklı kaynakları çıkarabildiğini göstererek bu çalışmanın doğrudan öncülünü oluşturmuştur [27]. Bu tez kapsamında FiLM, yalnızca darboğazda değil, her kodlayıcı seviyesinde ayrı projeksiyonlarla uygulanmakta; böylece atlama bağlantılarının taşıdığı etkinlikler de sınıfa özgü hâle getirilmektedir. İlgili mimari ayrıntılar Alt Başlık 3.5'te verilmiştir.

## 2.7 Ses Olayı Tespiti

Seçici gürültü engelleme hattı, ayrıştırmadan önce karışımda hangi ses sınıflarının bulunduğunun belirlenmesini gerektirmektedir; bu görev, ses olayı tespiti (Sound Event Detection) alanyazınıyla doğrudan ilişkilidir. Ses olayı tespiti, bir kayıtta etkin olan ses sınıflarının ve isteğe bağlı olarak zamansal sınırlarının kestirildiği çok-etiketli bir sınıflandırma problemidir [28]. Evrişimsel ve evrişimsel-yinelemeli ağlar (CRNN), spektrogram girdisi üzerinde hem yerel spektral örüntüleri hem de zamansal bağlamı modelleyerek bu alanda baskın mimariler hâline gelmiştir [29]. Büyük ölçekli ontoloji ve veri kümeleri, özellikle yüzlerce sınıfı kapsayan AudioSet, zayıf etiketli (weak-label) öğrenmeyi ve geniş sözcük dağarcıklı tespiti olanaklı kılmıştır [30].

Bu çalışmada tespit görevi, karışımda her aday sınıfın varlık olasılığının kestirilmesi biçiminde ele alınmıştır. İki strateji değerlendirilmiştir. Birinci strateji, ayrıştırma maskesinin enerjisini sınıf varlığı için dolaylı bir gösterge olarak kullanan bir sezgiseldir; ancak geniş bantlı ve dağınık maske üreten sınıflar (örneğin siren ve gök gürültüsü), karışımda bulunmasalar dahi yüksek maske enerjisi ürettiğinden bu sezgisel yapısal olarak güvenilmezdir. İkinci strateji, FiLM ile koşullandırılmış darboğaz temsili üzerine eklenen ve ikili çapraz entropi ile eğitilen öğrenilmiş bir tespit başıdır. Ses olayı tespiti alanyazınının, doğrudan sınıflandırmanın enerji tabanlı göstergelerden üstün olduğuna ilişkin bulguları doğrultusunda, bu tezde öğrenilmiş tespit başı benimsenmiş ve ilgili tasarım Alt Başlık 3.5.6'da ayrıntılandırılmıştır. Bu bölümde özetlenen yöntemsel arka plan üzerine, üçüncü bölümde önerilen sistemin tüm bileşenleri ayrıntılı biçimde sunulmaktadır.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 3. MATERYAL VE YÖNTEM

Bu bölümde, önerilen seçici gürültü engelleme sisteminin tüm bileşenleri ayrıntılı biçimde açıklanmıştır. Önce genel sistem mimarisi ve veri akışı tanıtılmış; ardından ses ön işleme ve spektrogram temsili, kullanılan veri kümeleri, anlık karışım üreteci, FiLM-koşullu U-Net mimarisi, kayıp fonksiyonları, optimizasyon süreci ve çıkarım hattı sırasıyla ele alınmıştır. Açıklamalar, önerilen modelde kullanılan yapılandırma temel alınarak verilmiştir.

## 3.1 Genel Sistem Mimarisi ve Veri Akışı

Sistem, eğitim ve çıkarım olmak üzere iki ayrı veri akışından oluşmaktadır. Eğitim akışında, ham ses kümeleri bellek içi bir klip önbelleğine çözülmekte, bu önbellekten anlık olarak karışımlar sentezlenmekte ve FiLM-koşullu U-Net bu karışımlar üzerinde eğitilmektedir. Çıkarım akışında ise eğitilmiş model, kullanıcının yüklediği dosyada bulunan sınıfları tespit etmekte ve seçilen sınıfları örtüşmeli toplama yöntemiyle dosyadan çıkarmaktadır. Sistemin uçtan uca veri akışı Şekil 3.1'de şematik olarak gösterilmiştir.

```
ESC-50 + UrbanSound8K (ham WAV kayıtları)
    → load_all_datasets        : {sınıf: [dalga biçimi]} bellek içi önbellek
    → SeparationMixer          : anlık karışım, sorgu ve hedef stem üretimi
    → ConditionedSeparatorTrainer : FiLM U-Net, çok çözünürlüklü L1 + BCE
    → eğitilmiş model (.h5)
    → webapp                   : tespit → maske → ISTFT → yeniden sentez
```
**Şekil 3.1:** Önerilen sistemin uçtan uca veri akışı şeması.

Yazılım mimarisi, sorumlulukların ayrıştırıldığı modüler bir yapıda tasarlanmıştır. Veri hazırlama katmanı (`dataset_sources.py` ve `separation_mixer.py`), veri kümelerinin yüklenmesinden ve karışımların üretilmesinden; model eğitimi katmanı (`conditioned_separator.py` ve `train_conditioned_separator.py`), ağ mimarisinin tanımlanmasından ve eğitilmesinden; uygulama katmanı (`webapp.py`) ise çıkarımın kullanıcı arayüzüyle bütünleştirilmesinden sorumludur. Yapılandırma ve dosya yolları, ortam değişkeniyle denetlenebilen merkezi bir yapılandırma modülünde (`model_config.py`) tutulmaktadır. Bu modüler kurgu, dördüncü bölümde sunulan deneysel metodolojinin izlenebilirliğini de sağlamaktadır.

## 3.2 Ses Ön İşleme ve Spektrogram Temsili

Modelin tüm girişleri, ortak bir spektrogram sözleşmesine indirgenmiştir. Bu sözleşme, hem eğitim hem de çıkarım hatlarında birebir aynı parametrelerle uygulanarak eğitim ve çıkarım dağılımlarının tutarlılığı güvence altına alınmıştır.

### 3.2.1 Örnekleme ve Tek Kanala İndirgeme

Tüm ses verileri $16$ kHz örnekleme hızında yeniden örneklenmiş ve tek kanala (mono) indirgenmiştir. $16$ kHz örnekleme hızı, Nyquist–Shannon örnekleme kuramı gereği $8$ kHz'e kadar olan frekans bileşenlerinin temsil edilmesine olanak tanımaktadır; bu bant, çevresel seslerin algısal olarak baskın enerjisini içermektedir. Daha yüksek örnekleme hızları ($44{,}1$ kHz gibi) ek bir üst-bant ayrıntısı sağlasa da, spektrogram boyutunu ve dolayısıyla hesaplama ile bellek yükünü orantısız biçimde artırmaktadır. Bu nedenle $16$ kHz, temsil yeterliliği ile hesaplama maliyeti arasında bir uzlaşı noktası olarak belirlenmiştir. Çok kanallı kayıtlar, kanal ortalaması alınarak tek kanala indirgenmiş; böylece modelin girişi, kanal sayısından bağımsız hâle getirilmiştir.

### 3.2.2 Kısa Zamanlı Fourier Dönüşümü Sözleşmesi

Zaman düzlemindeki dalga biçimi, Kısa Zamanlı Fourier Dönüşümü (STFT) ile zaman-frekans düzlemine taşınmıştır. STFT, bir analiz penceresi $w[n]$ kaydırılarak hesaplanan ardışık Fourier dönüşümleri olarak tanımlanmaktadır:

$$X[k, m] = \sum_{n=0}^{L-1} w[n]\, x[n + mH]\, e^{-j 2\pi k n / L},$$

burada $L$ pencere uzunluğu (FFT boyutu), $H$ sıçrama uzunluğu (hop length), $k$ frekans bini ve $m$ zaman çerçevesi indisidir. Bu çalışmada $L = n_{\text{fft}} = 512$ ve $H = hop = 128$ değerleri seçilmiştir. $16$ kHz örnekleme hızında bu değerler, $32$ ms'lik bir analiz penceresine ve $8$ ms'lik bir çerçeve adımına karşılık gelmekte; ardışık pencereler arasında $\%75$ örtüşme sağlamaktadır. Pencereleme işlevi olarak Hann penceresi kullanılmıştır; bu pencere, spektral sızıntıyı sınırlandırması ve örtüşmeli toplama altında birim-zarf koşulunu yaklaşık olarak sağlaması bakımından tercih edilmiştir.

$512$ noktalı FFT, $257$ adet tek-yanlı frekans bini üretmektedir. Nyquist bini ($k = 256$) düşürülerek model girişi $256$ frekans binine indirgenmiş; böylece frekans ekseni, evrişimsel alt örnekleme için elverişli olan $2$'nin kuvveti bir boyuta ($256$) sabitlenmiştir. Bir saniyelik bir pencere ($16\,000$ örnek), merkez hizalı STFT altında yaklaşık $126$ zaman çerçevesi üretmekte; bu eksen, sabit girdi boyutu için $128$ çerçeveye sıfır-doldurma ile tamamlanmaktadır. Sonuç olarak modelin girişi, $(256, 128, 1)$ boyutlu bir genlik spektrogramı tensörüdür ve her tensör yaklaşık bir saniyelik akustik bağlamı temsil etmektedir.

STFT parametrelerinin seçimi, zaman ve frekans çözünürlüğü arasında bir ödünleşim içermektedir. Frekans çözünürlüğü $\Delta f = f_s / n_{\text{fft}} = 16000/512 = 31{,}25$ Hz, zaman çözünürlüğü ise $\Delta t = hop / f_s = 8$ ms olarak hesaplanmaktadır. Gabor–Heisenberg belirsizlik ilkesi gereği, $\Delta t \cdot \Delta f$ çarpımı bir alt sınırla sınırlandığından, pencerenin kısaltılması zaman çözünürlüğünü iyileştirirken frekans çözünürlüğünü, uzatılması ise tersini kötüleştirmektedir. $32$ ms'lik pencere, çevresel seslerin harmonik yapısını ayırt edebilecek frekans çözünürlüğü ile geçici (transient) olayları yakalayabilecek zaman çözünürlüğü arasında bir uzlaşı noktası olarak belirlenmiştir.

Analiz–sentez çiftinin yapaylık üretmeden tersinir olabilmesi için, pencerenin sabit örtüşmeli toplama (Constant Overlap-Add, COLA) koşulunu sağlaması gerekmektedir. Bu koşul, kaydırılmış pencere karelerinin toplamının tüm örnekler için sabit kalmasını gerektirmektedir:

$$\sum_{m} w^{2}[n - mH] = C, \qquad \forall n.$$

$\%75$ örtüşmeli ($H = n_{\text{fft}}/4$) Hann penceresi bu koşulu sağladığından, maske uygulanmadığında karışım birebir geri elde edilebilmekte; maske uygulandığında ise çıkarım hattındaki yeniden sentez yapaylıkları en aza inmektedir (Alt Başlık 3.8.3).

### 3.2.3 Logaritmik Genlik Sıkıştırması

Ses sinyallerinin genlik spektrumu, birkaç on yıllık (decade) bir dinamik aralığa yayılmaktadır. Bu geniş aralığın doğrudan ağa verilmesi, yüksek enerjili bileşenlerin gradyanları baskılamasına ve düşük enerjili ancak algısal olarak önemli bileşenlerin göz ardı edilmesine yol açmaktadır. Bu nedenle model girişinde, genlik spektrogramına logaritmik sıkıştırma uygulanmıştır:

$$X_{\log}[k, m] = \log\!\big(1 + |X[k, m]|\big).$$

$\log(1 + \cdot)$ biçimindeki sıkıştırma, sıfır genlikte tanımlı kalması (logaritmanın tekilliğinden kaçınması) ve küçük genlikler için yaklaşık doğrusal davranması bakımından tercih edilmiştir. Bu dönüşüm, dinamik aralığı sıkıştırarak eğitim kararlılığını artırmakta ve insan işitmesinin logaritmik yükseklik algısıyla örtüşmektedir. Modelin yalnızca logaritmik genlik girişiyle koşullandırıldığı; maskenin uygulanacağı doğrusal genliğin ise ayrı bir giriş olarak ağ grafiğine taşındığı vurgulanmalıdır. Bu ayrım sayesinde maske, doğrusal genlik üzerinde uygulanmakta ve kestirilen stem, ölçek bilgisini koruyarak yeniden sentezlenebilmektedir (Alt Başlık 3.5.5).

## 3.3 Veri Kümeleri

Modelin sözcük dağarcığı, halka açık çevresel ses veri kümelerinin birleştirilmesiyle oluşturulmuştur. Birleştirme işlemi, her veri kümesini ortak bir $\{$sınıf$:$ dalga biçimi listesi$\}$ sözlüğüne çözen ve bu sözlükleri tek bir önbellekte toplayan veri yükleme katmanı tarafından yürütülmektedir. Kullanılan veri kümelerinin sayısal özeti Tablo 3.1'de verilmiştir.

**Tablo 3.1:** Kullanılan veri kümelerinin sayısal özeti.

| Veri kümesi | Sınıf sayısı | Klip sayısı | Klip süresi | Özgün örnekleme hızı |
|---|---|---|---|---|
| ESC-50 | 50 | 2.000 | 5 s (sabit) | 44,1 kHz |
| UrbanSound8K | 10 | 8.732 | $\leq 4$ s | 16–48 kHz (değişken) |
| FSD50K | ~200 | ~51.200 | 0,3–30 s | 44,1 kHz |

Tüm kayıtlar, üçüncü bölümün ön işleme sözleşmesi (Alt Başlık 3.2.1) uyarınca $16$ kHz örnekleme hızında ve tek kanala yeniden örneklenmiştir; FSD50K klipleri, bellek kullanımını sınırlandırmak için yüklenirken $4$ saniyeye kırpılmaktadır. Önerilen modelde yalnızca ESC-50 ve UrbanSound8K kullanılmakta; bu iki kümenin birleşimi, takma ad eşlemesi sonrası yaklaşık elli altı sınıf ve toplam $10.732$ klip içermektedir.

### 3.3.1 ESC-50

ESC-50, elli çevresel ses sınıfından oluşan ve sınıf başına tam olarak kırk klip içeren, toplam iki bin kayıtlık dengeli bir veri kümesidir [31]. Her klip beş saniye uzunluğunda ve $44{,}1$ kHz örnekleme hızındadır; toplam ses süresi yaklaşık $2{,}8$ saattir. Kayıtlar Freesound arşivinden derlenmiş olup, hayvan sesleri, doğal ses olayları, insan kaynaklı sesler, iç mekân/ev sesleri ve kentsel gürültü olmak üzere beş üst kategoriye ayrılmış; veri kümesi beş çapraz-doğrulama katmanına bölünmüştür. Sınıf başına klip sayısının kırk gibi sınırlı bir değerde olması, bu çalışmadaki minimum klip tabanı eşiğinin (Alt Başlık 3.3.4) $N_{\min} = 40$ olarak belirlenmesinde de belirleyici olmuştur.

### 3.3.2 UrbanSound8K ve Sınıf Birleştirme

UrbanSound8K, on kentsel ses sınıfından oluşan ve toplam $8.732$ klip içeren bir veri kümesidir [32]. Klipler en çok dört saniye uzunluğunda olup veri kümesi on çapraz-doğrulama katmanına (fold) bölünmüştür. ESC-50'nin aksine sınıflar dengesiz dağılmıştır: sınıf başına klip sayısı, en az desteklenen `gun_shot` sınıfındaki $374$ klipten, `air_conditioner` ve `street_music` gibi sınıflardaki $1.000$ klibe kadar değişmektedir. Bu kümenin bazı sınıfları ESC-50 ile anlamsal olarak örtüşmektedir. Örtüşen sınıfların ayrı etiketler hâline gelmesini önlemek için bir takma ad eşlemesi (`CLASS_ALIASES`) tanımlanmış; örneğin UrbanSound8K'deki köpek havlaması sınıfı (`dog_bark`) ESC-50'deki köpek sınıfına (`dog`), motor rölantisi sınıfı (`engine_idling`) ise motor sınıfına (`engine`) eşlenmiştir. Bu eşleme sayesinde örtüşen sınıfların klipleri tek bir kanonik etiket altında havuzlanmakta; on sınıfın dördü ESC-50 ile birleşmekte, altısı ise yeni sınıf olarak eklenmektedir. İki veri kümesinin birlikte yüklenmesiyle modelin sözcük dağarcığı yaklaşık elli altı sınıfa ulaşmaktadır.

### 3.3.3 FSD50K ve Uzun Kuyruk Problemi

FSD50K, AudioSet ontolojisine göre etiketlenmiş, $200$ sınıf ve yaklaşık $51.200$ klip ($40.966$ geliştirme + $10.231$ değerlendirme) içeren büyük ölçekli bir ses olayı veri kümesidir [33]. Klip süreleri $0{,}3$ ile $30$ saniye arasında değişmekte ve toplam süre yüz saati aşmaktadır. Bu kümede etiketler hiyerarşiktir ve çoğu klip virgülle ayrılmış birden çok etikete sahiptir; bu çalışmada her klibin ilk (en özgül/yaprak) etiketi kanonik sınıf olarak alınmıştır. FSD50K'nin sözcük dağarcığını genişletme potansiyeli bulunmakla birlikte, $200$ yaprak etiketinin önemli bir bölümünün yalnızca birkaç klip tarafından desteklenmesi, bir "uzun kuyruk" problemi doğurmaktadır. Az sayıda ve çok-etiketli örnekten öğrenilen bir sınıf, ayırt edici olmayan ve dağınık bir maske üretmekte; bu maske, ilgili sınıf karışımda bulunmasa dahi yüksek enerji üreterek yanlış pozitiflere yol açmaktadır. Bu olgu, dördüncü bölümde ayrıntılandırılan deneysel gözlemlerde belirleyici bir başarısızlık örüntüsü olarak ortaya çıkmış ve önerilen modelde FSD50K bütünüyle dışarıda bırakılmıştır.

### 3.3.4 Minimum Klip Tabanı ve Düzenlenmiş Sözcük Dağarcığı

Az desteklenen sınıfların yarattığı yanlış pozitif eğilimini sınırlandırmak için, birleştirme sonrası uygulanan bir minimum klip tabanı eşiği tanımlanmıştır. Bu eşik, sınıf başına en az kırk klip ($N_{\min} = 40$) koşulunu sağlamayan sınıfları sözcük dağarcığından çıkarmaktadır. Eşik birleştirmeden *sonra* uygulandığından, veri kümeleri arası takma adlar önce havuzlanmakta; yalnızca gerçekten yetersiz desteklenen sınıflar elenmektedir. ESC-50 (sınıf başına kırk klip) ve UrbanSound8K (sınıf başına yüzlerce klip) sınıfları bu eşikten etkilenmemektedir.

Önerilen model, ayrıştırma ve tespit başarımının en yüksek olduğu on beş sınıftan oluşan, düzenlenmiş (curated) bir alt küme üzerinde eğitilmiştir. Bu alt küme, `keep_classes` parametresiyle yalnızca döndürülen sözlüğe uygulanmakta; diskteki çözülmüş önbellek tam sözcük dağarcığını koruduğundan, düzenlenmiş alt küme çalışması ile tam sözcük dağarcığı çalışması aynı önbellek dosyasını paylaşabilmektedir. Seçilen on beş sınıfın klip dağılımı Şekil 3.2'de gösterilmiştir; `gun_shot` sınıfı (UrbanSound8K katkısıyla) $374$ klip içerirken, kalan on dört sınıf $N_{\min} = 40$ tabanında dengelenmiştir.

![Şekil 3.2](../thesis_figures/01_dataset_clip_counts.png)

**Şekil 3.2:** On beş sınıflı düzenlenmiş sözcük dağarcığında sınıf başına klip sayısı.

Seçilen on beş sınıfın klip sayıları ve kaynak veri kümeleri Tablo 3.2'de listelenmiştir. Düzenlenmiş sözcük dağarcığı toplam $934$ klip içermekte; bu kliplerin $560$'ı ESC-50 kaynaklı on dört sınıftan ($14 \times 40$), $374$'ü ise UrbanSound8K kaynaklı `gun_shot` sınıfından gelmektedir. `gun_shot` dışındaki sınıfların tümü, ESC-50'nin sınıf başına kırk klip dengesini koruduğundan, sözcük dağarcığı `gun_shot` haricinde tümüyle dengelidir.

**Tablo 3.2:** Düzenlenmiş on beş sınıflı sözcük dağarcığının sınıf bazlı klip sayıları.

| Sınıf | Klip sayısı | Kaynak |
|---|---|---|
| gun_shot | 374 | UrbanSound8K |
| brushing_teeth | 40 | ESC-50 |
| church_bells | 40 | ESC-50 |
| clapping | 40 | ESC-50 |
| clock_alarm | 40 | ESC-50 |
| coughing | 40 | ESC-50 |
| crow | 40 | ESC-50 |
| crying_baby | 40 | ESC-50 |
| hand_saw | 40 | ESC-50 |
| helicopter | 40 | ESC-50 |
| rain | 40 | ESC-50 |
| sea_waves | 40 | ESC-50 |
| sneezing | 40 | ESC-50 |
| toilet_flush | 40 | ESC-50 |
| vacuum_cleaner | 40 | ESC-50 |
| **Toplam** | **934** | — |

## 3.4 Anlık Veri Üretimi: SeparationMixer

Eğitim örnekleri, önceden üretilmiş bir veri dosyasından okunmak yerine, bellek içi klip önbelleğinden her eğitim adımında anlık olarak sentezlenmektedir. Bu görev, sonsuz bir örnek akışı üreten `SeparationMixer` sınıfı tarafından yürütülmektedir. Her örnek, bir karışım spektrogramı, bir sınıf sorgusu ve o sınıfın hedef stem genliğinden oluşan bir üçlüdür. Anlık üretim yaklaşımı, hem depolama yükünü ortadan kaldırmakta hem de aynı kliplerin farklı genlik, pencere ve birleşimlerle yeniden kullanılması sayesinde görece sınırsız bir karışım çeşitliliği sağlamaktadır.

### 3.4.1 Karışım Sentezi ve Genlik Örnekleme

Bir karışım örneği oluşturulurken, önce karışıma katılacak kaynak sayısı $k$, $\{1, 2, \dots, K_{\max}\}$ kümesinden düzgün dağılımla çekilmektedir; bu çalışmada $K_{\max} = 4$ alınmıştır. Ardından sözcük dağarcığından $k$ adet sınıf yerine koymadan örneklenmekte ve her sınıf için önbellekten rastgele bir klip seçilmektedir. Seçilen her klipten, bir saniyelik bir pencere rastgele konumdan kırpılmakta (klip bir saniyeden kısaysa sıfır-doldurma uygulanmakta) ve bu pencere, $[0{,}4;\, 1{,}0]$ aralığından düzgün dağılımla çekilen bir genlik katsayısı $a_i$ ile ölçeklenmektedir. Karışım, ölçeklenmiş pencerelerin toplamı olarak elde edilmektedir:

$$x[n] = \sum_{i=1}^{k} a_i\, s_i[n], \qquad a_i \sim \mathcal{U}(0{,}4;\, 1{,}0).$$

Genlik katsayılarının rastgeleleştirilmesi, modelin farklı bağıl ses düzeylerine karşı dayanıklılık kazanmasını sağlamakta; her kaynağın hedef stem'i, ilgili ölçeklenmiş pencere olarak ayrı ayrı saklanmaktadır.

### 3.4.2 Negatif Örnekler ve Sessizlik Hedefi

Modelin, karışımda bulunmayan bir sınıf sorgulandığında yakın-sıfır bir maske üretmeyi öğrenmesi, seçici bastırmanın temel koşuludur. Bu amaçla, $P_{\text{negatif}}$ olasılığıyla bir *negatif örnek* üretilmektedir: sorgu, karışımda bulunmayan bir sınıfı işaret etmekte ve hedef stem sessizlik (tümüyle sıfır) olarak atanmaktadır. Geri kalan örneklerde sorgu, karışımda bulunan bir sınıfı işaret etmekte ve hedef, o sınıfın stem'i olmaktadır.

$P_{\text{negatif}}$ parametresinin seçimi, kritik bir ödünleşim içermektedir. Çok yüksek bir değer, L1 kaybının her şey için yakın-sıfır çıktıyı ödüllendirmesine ve modelin "güvenli sessizlik" dengesine çökmesine yol açmaktadır; bu olgu, dördüncü bölümde ayrıntılandırıldığı üzere $P_{\text{negatif}} = 0{,}45$ gibi yüksek bir değerde gözlemlenmiştir. Çok düşük bir değer ise yetersiz bastırmaya neden olmaktadır. Önerilen modelde, tespit başının yeterli negatif maruziyetle eğitilebilmesi için $P_{\text{negatif}} = 0{,}50$ değeri benimsenmiştir; bu değer, ayrıştırma kaybının pozitif örneklerce, tespit kaybının ise dengeli bir pozitif–negatif karışımıyla beslenmesini sağlamaktadır.

### 3.4.3 Ağırlıklı Zor-Negatif Örnekleme

Negatif örneklerde sorgulanacak yok sınıfının düzgün dağılımla seçilmesi, geniş bantlı ve dağınık maske üreten sınıfların yeterince bastırma örneği görmemesine yol açabilmektedir. Bu sorunu hafifletmek için, ağırlıklı zor-negatif örnekleme mekanizması tanımlanmıştır. Aşırı-tetikleyen (over-firing) olarak işaretlenen sınıflara bir ağırlık katsayısı $w_{\text{of}} = 3{,}0$, diğer tüm sınıflara ise $1{,}0$ atanmakta; oluşan ağırlık vektörü bir olasılık dağılımına normalize edilmektedir. Negatif örnekte yok sınıfı, mevcut sınıflar dışlandıktan sonra bu dağılımdan çekilmektedir:

$$P(\text{yok sınıfı} = c) = \frac{w_c}{\displaystyle\sum_{c' \notin \text{mevcut}} w_{c'}}, \qquad c \notin \text{mevcut}.$$

Bu mekanizma, problemli sınıflar için zor-negatif örneklerin sıklığını, karışım üretim mantığını ya da pozitif örnekleri değiştirmeden artırmaktadır. Önerilen modelde, bilinen aşırı-tetikleyen sınıflar sözcük dağarcığından çıkarıldığından, bu mekanizma etkin değildir ($w_{\text{of}}$ listesi boştur); ancak geniş sözcük dağarcıklı denemelerde geniş bantlı sınıfların bastırılmasında belirleyici bir rol oynamıştır (Bölüm 4).

### 3.4.4 Arka Plan Gürültüsü Artırımı

Modelin hedef kaynağı, yapısız ortam gürültüsünden ayırt etmeyi öğrenmesi için, $P_{\text{gürültü}} = 0{,}10$ olasılığıyla karışıma geniş bantlı gürültü eklenmektedir. Eklenen gürültünün düzeyi, $[15;\, 30]$ dB aralığından düzgün dağılımla çekilen bir işaret-gürültü oranı (SNR) ile belirlenmektedir. Hedef SNR değeri için gürültünün karekök-ortalama-kare (RMS) genliği,

$$\text{RMS}_{\text{gürültü}}^{\text{hedef}} = \frac{\text{RMS}_{\text{karışım}}}{10^{\,\text{SNR}_{\text{dB}}/20}}$$

bağıntısıyla hesaplanmakta ve gürültü, bu hedef RMS'e ölçeklenerek karışıma eklenmektedir. $[15;\, 30]$ dB aralığı, gürültü genliğinin işaret genliğinin yaklaşık $\%6$–$\%18$'i mertebesinde kalmasını sağlayarak gerçekçi ortam düzeylerini taklit etmektedir; aşırı düşük SNR değerlerinin (örneğin $5$ dB gibi) denetim sinyalini gürültü altında bıraktığı gözlemlenmiştir. Üretilen gürültünün yarısı beyaz gürültü, diğer yarısı ise frekansla $1/\sqrt{f}$ oranında zayıflayan pembe gürültüdür; pembe gürültü, FFT düzleminde genlik tayfının $1/\sqrt{k}$ ile ölçeklenmesiyle elde edilmektedir. Bu iki bileşen, düz ve $1/f$-eğimli gerçek dünya ortam seslerini (havalandırma, kalabalık uğultusu, trafik gürültüsü) birlikte temsil etmektedir. Önemli bir tasarım kararı olarak, gürültü yalnızca karışıma eklenmekte; hedef stem'e *eklenmemektedir*. Böylece model, gürültüyü maske dışında bırakmayı, yani yok saymayı öğrenmektedir.

### 3.4.5 Tepe Normalizasyonu ve Eğitim-Çıkarım Tutarlılığı

Karışım oluşturulduktan sonra, tepe genliği $1{,}0$ olacak biçimde normalize edilmekte ve aynı ölçek katsayısı hedef stem pencerelerine de uygulanmaktadır:

$$x \leftarrow \frac{x}{\max_n |x[n]|}, \qquad s_i \leftarrow \frac{s_i}{\max_n |x[n]|}.$$

Tepe normalizasyonu, STFT genliklerinin eğitim boyunca tutarlı bir dağılımda kalmasını güvence altına almaktadır. Bu adımın çıkarım hattında birebir tekrarlanması zorunludur; aksi hâlde modelin etkinlikleri eğitilmemiş bir çalışma bölgesine kaymaktadır. Nitekim bir denemede, çıkarım hattının ham (normalize edilmemiş) sesi modele verilmesi, STFT genliklerinin eğitim dağılımına göre üç ila on kat küçük kalmasına ve modelin tüm sınıflarda işlevsiz hâle gelmesine yol açmıştır. Bu hata, çıkarım hattında tam dosya genliğinin tepe normalizasyonuyla giderilmiş ve eğitim ile çıkarım dağılımlarının tutarlılığı sağlanmıştır.

Yukarıda açıklanan bir örneğin üretim süreci, Algoritma 3.1'de bütünleşik olarak verilmiştir.

```
Algoritma 3.1: Anlık karışım örneği üretimi (SeparationMixer)
─────────────────────────────────────────────────────────────────────
Girdi : C = {sınıf → dalga biçimi listesi}      (bellek içi klip önbelleği)
        K_max (en çok kaynak sayısı), [a_alt, a_üst] (genlik aralığı)
        P_neg (negatif olasılığı), P_gür (gürültü olasılığı)
        [SNR_alt, SNR_üst] (gürültü düzeyi), w (sınıf ağırlık vektörü)
Çıktı : (X_log, q, X_lin, T)  — log-genlik, sorgu, doğrusal genlik, hedef genlik
─────────────────────────────────────────────────────────────────────
 1:  k ← DüzgünTamsayı(1, K_max)
 2:  mevcut ← C'nin sınıflarından k tanesini yerine koymadan örnekle
 3:  x ← 0;   pencereler ← {}
 4:  for each c ∈ mevcut do
 5:      klip ← RastgeleSeç(C[c])
 6:      p ← BirSaniyelikPencereKırp(klip)            ▷ kısa ise sıfır-doldur
 7:      a ← Düzgün(a_alt, a_üst);   p ← a · p
 8:      pencereler[c] ← p;   x ← x + p
 9:  if Düzgün(0,1) < P_gür then
10:      x ← x + GenişBantGürültü(x, [SNR_alt, SNR_üst])   ▷ ½ beyaz, ½ pembe
11:  tepe ← max_n |x[n]|
12:  x ← x / tepe;   for each c do pencereler[c] ← pencereler[c] / tepe
13:  yok ← C'nin sınıfları ∖ mevcut
14:  if yok ≠ ∅ and Düzgün(0,1) < P_neg then          ▷ negatif örnek
15:      c* ← AğırlıklıÖrnekle(yok, w);   T ← 0
16:  else                                             ▷ pozitif örnek
17:      c* ← RastgeleSeç(mevcut);        T ← pencereler[c*]
18:  q ← OneHot(c*)
19:  X_lin ← |STFT(x)|;   X_log ← log(1 + X_lin);   T ← |STFT(T)|
20:  return (X_log, q, X_lin, T)
```

## 3.5 FiLM-Koşullu U-Net Mimarisi

Önerilen modelin çekirdeği, iki girişli ve FiLM ile koşullandırılmış iki boyutlu bir U-Net mimarisidir. Birinci giriş, $(256, 128, 1)$ boyutlu logaritmik genlik spektrogramı; ikinci giriş ise sözcük dağarcığı üzerindeki $(N,)$ boyutlu tek-sıcak sınıf sorgusudur. Ağın çıkışı, sorgulanan sınıf için $[0, 1]$ aralığında değer alan tek bir yumuşak maskedir. İlk kodlayıcı bloğunun kanal sayısı (`base_filters`) $32$ alındığında model yaklaşık $8{,}3$ milyon parametre içermektedir. Mimarinin genel yapısı Şekil 3.3'te gösterilmiştir.

![Şekil 3.3](../thesis_figures/architecture.png)

**Şekil 3.3:** FiLM-koşullu U-Net mimarisi; sınıf sorgusu, her kodlayıcı seviyesinde ve darboğazda kanal-bazlı ölçek ve öteleme parametrelerine dönüştürülmektedir.

### 3.5.1 U-Net Kodlayıcı–Kod Çözücü Yapısı ve Atlama Bağlantıları

Kodlayıcı, art arda gelen dört çözünürlük seviyesi ve bir darboğaçtan oluşmaktadır. Her seviyede, çift evrişim bloğunun ardından $2\times 2$ en büyük havuzlama (max pooling) uygulanarak uzamsal çözünürlük yarıya indirilmekte, kanal sayısı ise iki katına çıkarılmaktadır. Kanal ilerlemesi $32 \to 64 \to 128 \to 256 \to 512$, uzamsal ilerleme ise $(256, 128) \to (128, 64) \to (64, 32) \to (32, 16) \to (16, 8)$ biçimindedir. Darboğaz, en düşük çözünürlükte ($16 \times 8$) en yüksek kanal sayısına ($512$) ulaşarak girdinin yüksek düzeyli, soyut bir temsilini taşımaktadır.

Kod çözücü, simetrik bir biçimde çözünürlüğü kademeli olarak geri yükselten transpoze evrişim (Conv2DTranspose) katmanlarından oluşmaktadır. Her kod çözücü seviyesinde, ilgili kodlayıcı seviyesinden gelen etkinlik haritası bir *atlama bağlantısıyla* (skip connection) birleştirilmekte (concatenate) ve ardından bir çift evrişim bloğundan geçirilmektedir. Atlama bağlantıları, havuzlama sırasında yitirilen ince çözünürlüklü spektral ayrıntıyı kod çözücüye taşıyarak maskenin keskin sınırlar üretebilmesini sağlamaktadır. Bu yapı, U-Net mimarisinin kaynak ayrıştırmada tercih edilmesinin temel gerekçesidir.

### 3.5.2 Evrişim Bloğu, Toplu Normalizasyon ve ReLU

Mimarinin temel yapı taşı, art arda iki adet $3 \times 3$ evrişim katmanından oluşan çift evrişim bloğudur. Her evrişim katmanını bir toplu normalizasyon (Batch Normalization) ve bir ReLU etkinleştirme izlemektedir. Toplu normalizasyon, her mini-yığın için ara etkinlikleri ortalama ve varyans bakımından standartlaştırarak iç ortak değişken kaymasını (internal covariate shift) azaltmakta ve eğitimi hızlandırmaktadır [34]. Evrişim katmanlarında yanlılık (bias) terimi kullanılmamıştır; bunun nedeni, ardından gelen toplu normalizasyonun zaten öğrenilebilir bir öteleme parametresi içermesi ve böylece yanlılık teriminin gereksiz hâle gelmesidir. ReLU etkinleştirmesi, doğrusal olmamayı sağlarken gradyan akışını koruyan, hesaplama açısından yalın bir seçimdir.

### 3.5.3 FiLM Koşullandırması

Sınıf koşullandırması, özellik-bazlı doğrusal modülasyon (FiLM) ile gerçekleştirilmektedir. Bir etkinlik haritası $x$ üzerinde FiLM dönüşümü, kanal-bazlı bir ölçek $\gamma$ ve bir öteleme $\beta$ ile

$$\mathrm{FiLM}(x) = \gamma \odot x + \beta$$

biçiminde tanımlanmaktadır; burada $\odot$ kanal ekseni boyunca yayınımlı (broadcast) çarpımı göstermektedir. Bu dönüşüm, hesaplama grafiğinde bir çarpma ($x \odot \gamma$) ve bir toplama ($+\,\beta$) işlemiyle uygulanmakta; $\gamma$ ve $\beta$ parametreleri, ilgili seviyenin kanal sayısına eşit uzunlukta üretilip $(1, 1, C)$ biçimine yeniden şekillendirilerek tüm uzamsal konumlara aynı biçimde uygulanmaktadır. FiLM, koşullandırma sinyalini ana evrişimsel yoldan ayrıştırması ve çarpımsal modülasyonla kanal seçiciliği sağlaması bakımından, sorgu vektörünün doğrudan eklenmesine göre daha güçlü bir denetim mekanizması sunmaktadır.

FiLM dönüşümünün geri yayılımdaki davranışı, koşullandırmanın neden etkili olduğunu açıklamaktadır. Etkinliğe göre türev $\partial\,\mathrm{FiLM}(x)/\partial x = \gamma$ olduğundan, koşullandırma parametresi $\gamma$, ilgili kanaldan geçen gradyanı kanal-bazlı olarak ölçeklemekte, yani bir çarpımsal geçit (multiplicative gating) işlevi görmektedir. Koşullandırma parametrelerine göre türevler ise

$$\frac{\partial \mathcal{L}}{\partial \gamma_c} = \sum_{u} \frac{\partial \mathcal{L}}{\partial\,\mathrm{FiLM}(x)_{u,c}}\, x_{u,c}, \qquad \frac{\partial \mathcal{L}}{\partial \beta_c} = \sum_{u} \frac{\partial \mathcal{L}}{\partial\,\mathrm{FiLM}(x)_{u,c}}$$

biçimindedir; burada $u$ indisi uzamsal konumlar üzerinden toplamı göstermektedir. Afin dönüşüm, yalnızca ölçek ($\gamma$) ya da yalnızca öteleme ($\beta$) içeren kısıtlı biçimlere göre daha yüksek ifade gücüne sahiptir: $\gamma_c \to 0$ ataması bir kanalı ilgili sınıf için bütünüyle bastırabilmekte, $\beta_c$ ise kendisini izleyen ReLU etkinleştirmesinin eşiğini kaydırarak kanalın etkinlik durumunu sınıfa göre değiştirebilmektedir. Sınıf-seçici maskelemenin temel mekanizması, bu kanal-bazlı bastırma ve eşik kaydırma işlemleridir.

### 3.5.4 Sorgu Gömme ve Çok Seviyeli Projeksiyon

Tek-sıcak sınıf sorgusu, önce paylaşılan bir gömme katmanından geçirilmektedir; bu katman, $128$ boyutlu bir yoğun (Dense) gömme üretmekte ve ReLU ile etkinleştirilmektedir. Paylaşılan gömme, her FiLM seviyesi için ayrı ölçek ve öteleme projeksiyonlarına beslenmektedir: her kodlayıcı seviyesi ve darboğaz, kendi $\gamma$ ve $\beta$ parametrelerini üreten bağımsız yoğun katmanlara sahiptir. FiLM koşullandırması yalnızca darboğazda değil, beş ayrı noktada — birinci ila dördüncü kodlayıcı seviyeleri (e1–e4) ve darboğaz — uygulanmaktadır. Koşullandırmanın tüm kodlayıcı seviyelerinde uygulanması, kodlayıcının kendisinin sınıfa özgü özellikler kurmasını zorlamakta; böylece kod çözücüye giren atlama bağlantıları da sınıfa özgü etkinlikler taşımaktadır. Bu çok seviyeli koşullandırmanın maske kesinliğine katkısı, dördüncü bölümde niceliksel olarak gösterilmiştir.

### 3.5.5 Maske Üretimi ve Float32 Çıkış Sabitleme

Kod çözücünün son etkinlik haritası, $1 \times 1$ evrişim ve sigmoid etkinleştirme ile tek kanallı, $[0, 1]$ aralığında bir yumuşak maskeye dönüştürülmektedir. Eğitim sırasında bu maske, ağ grafiğinin içinde doğrudan doğrusal genlik girişiyle çarpılarak kestirilen stem genliğini üretmektedir. Bu amaçla eğitim modeli, üç giriş ($[\,$logaritmik genlik, sınıf sorgusu, doğrusal genlik$\,]$) ve bir çıkış (kestirilen stem genliği) ile sarmalanmıştır. Maskenin uygulanması, bir çarpma katmanıyla gerçekleştirildiğinden ve özel (Lambda) katman içermediğinden, kaydedilen model dosyası özel nesne tanımına gerek kalmadan yeniden yüklenebilmektedir.

Sayısal kararlılık açısından kritik bir tasarım kararı, hem maske çıkış katmanının hem de maske uygulama çarpımının tam hassasiyette (float32) sabitlenmesidir. Eğitim, hesaplama hızı için karma hassasiyet (mixed_float16) politikası altında yürütülmesine karşın (Alt Başlık 3.7.2), sigmoid maske ve onu izleyen L1 kaybı yarı hassasiyette (float16) sayısal doygunluğa ve aşırı yuvarlamaya açıktır. Çıkış katmanının float32'ye sabitlenmesi, ağın gövdesi yarı hassasiyette çalışırken dahi maskenin ve kaybın tam hassasiyette hesaplanmasını güvence altına almaktadır.

### 3.5.6 Tespit Başı

Sınıf varlığının kestirimi için, FiLM ile koşullandırılmış darboğaz temsili üzerine hafif bir tespit başı eklenmiştir. Bu baş, darboğaz etkinliklerini küresel ortalama havuzlama (Global Average Pooling) ile $512$ boyutlu sınıf-duyarlı bir vektöre indirgemekte; ardından $128$ nöronlu bir ReLU yoğun katmanı ve tek nöronlu bir sigmoid katmanıyla, sorgulanan sınıfın varlık olasılığı $P(\text{sorgulanan sınıf mevcut} \mid \text{karışım})$ değerini üretmektedir. Darboğaz halihazırda sınıf sorgusuyla koşullandırıldığından, tespit başı hangi sınıfı değerlendirdiğini örtük olarak bilmektedir.

Tespit başı etkinleştirildiğinde, model iki çıkışlı hâle gelmektedir: kestirilen stem genliği ve sınıf varlık olasılığı. Varlık etiketleri, hedef stem genliğinden otomatik olarak türetilmektedir: hedefin mutlak değerinin en büyüğü bir eşiği ($10^{-6}$) aşıyorsa varlık $1{,}0$, aksi hâlde (negatif örneklerde) $0{,}0$ olarak atanmaktadır. Geriye dönük uyumluluk için, değerlendirme ve uygulama kodu modelin çıkış sayısını denetlemekte; tek çıkışlı modellerde maske-enerjisi sezgiseline, iki çıkışlı modellerde ise tespit başının olasılığına dayanmaktadır. Bu kurgu sayesinde farklı çıkış yapısına sahip modeller, kod değişikliği gerektirmeden aynı çıkarım hattıyla sunulabilmektedir.

## 3.6 Kayıp Fonksiyonları

Modelin eğitiminde, ayrıştırma ve tespit görevleri için iki ayrı kayıp fonksiyonu tanımlanmış ve ağırlıklı bir toplamla birleştirilmiştir.

### 3.6.1 Çok Çözünürlüklü L1 Kaybı

Ayrıştırma görevi, kestirilen stem genliği ile gerçek stem genliği arasındaki L1 (mutlak fark) kaybıyla denetlenmektedir. L1 kaybı, L2 (karesel) kaybına kıyasla aykırı değerlere karşı daha az duyarlıdır ve genlik spektrogramlarında daha keskin kestirimler üretmektedir. Bu çalışmada L1 kaybı, tek çözünürlükte değil, çok çözünürlüklü bir biçimde uygulanmıştır:

$$\mathcal{L}_{\text{ayr}} = \sum_{i=0}^{2} \left(\frac{1}{2}\right)^{i} \big\lVert y^{(i)} - \hat{y}^{(i)} \big\rVert_{1},$$

burada $y^{(0)}$ ve $\hat{y}^{(0)}$ tam çözünürlüklü gerçek ve kestirilen genlikleri; $y^{(i)}$ ve $\hat{y}^{(i)}$ ise $i$ kez $2\times 2$ ortalama havuzlama uygulanmış (yarı ve çeyrek çözünürlüklü) karşılıklarını göstermektedir. Çözünürlük seviyeleri sırasıyla $1{,}0$, $0{,}5$ ve $0{,}25$ katsayılarıyla ağırlıklandırılmaktadır. Daha kaba çözünürlükler, ince bin değerleri eniyilenmeden önce genel spektral biçimin oturmasını sağlamakta; bu da maske kalitesini ve eğitim kararlılığını artırmaktadır.

### 3.6.2 İkili Çapraz Entropi ve Odak Kaybı Karşılaştırması

Tespit görevi, sınıf varlık olasılığı üzerinde ikili çapraz entropi (Binary Cross-Entropy, BCE) kaybıyla denetlenmektedir. Eğitim sürecinde, sınıf dengesizliğini ele almak amacıyla odak kaybının (focal loss) BCE yerine kullanılması da denenmiştir. Odak kaybı, her örneğin BCE değerini bir $\alpha_t (1 - p_t)^{\gamma}$ çarpanıyla ölçekleyerek iyi sınıflandırılmış örneklerin katkısını azaltmaktadır. Ancak rastgele başlangıçta ($p \approx 0{,}5$), $\alpha = 0{,}25$ ve $\gamma = 2$ değerleri için bu çarpan

$$\alpha_t (1 - p_t)^{\gamma} = 0{,}25 \times 0{,}5^{2} = 0{,}0625$$

değerini almakta; aynı başlangıçta BCE değeri ise $-\log(0{,}5) \approx 0{,}693$ olmaktadır. Dolayısıyla odak kaybının gradyanı, başlangıçta BCE'ye göre yaklaşık on kat küçüktür. Dördüncü bölümde ayrıntılandırıldığı üzere, bu durum, odak kaybı denendiğinde tespit başının gradyan yetersizliğinden ötürü hiç öğrenememesine ve her giriş için yaklaşık $0{,}5$ üretecek biçimde çökmesine yol açmıştır. Bu gözlem doğrultusunda, tespit kaybı olarak BCE benimsenmiştir.

### 3.6.3 Çok Görevli Kayıp Ağırlıklandırması

Tespit başı etkin olduğunda, toplam kayıp, ayrıştırma ve tespit kayıplarının ağırlıklı toplamı olarak tanımlanmaktadır:

$$\mathcal{L} = \mathcal{L}_{\text{ayr}} \cdot w_{\text{ayr}} + \mathcal{L}_{\text{tespit}} \cdot w_{\text{tespit}}.$$

Önerilen modelde ayrıştırma ağırlığı $w_{\text{ayr}} = 1{,}0$ ve tespit ağırlığı $w_{\text{tespit}} = 0{,}5$ alınmıştır. Tespit ağırlığının ayrıştırma ağırlığından düşük tutulması, ana görev olan ayrıştırmanın baskın gradyan sinyalini korurken tespit başının da yeterli denetim almasını sağlamaktadır. Tespit için gereken varlık etiketleri, eğitim hattının içinde hedef stem genliğinden otomatik olarak türetilmektedir (Alt Başlık 3.5.6).

## 3.7 Optimizasyon ve Eğitim Süreci

Bu alt başlıkta, modelin eğitiminde kullanılan optimize edici, hesaplama hızlandırma teknikleri, öğrenme oranı denetimi, veri besleme hattı ve donanım ortamı açıklanmıştır.

### 3.7.1 Adam Optimize Edici

Model, Adam (Adaptive Moment Estimation) optimize edicisiyle eğitilmiştir [35]. Adam, momentum ve RMSProp yöntemlerini birleştirerek her parametre için uyarlanır bir öğrenme oranı sağlamaktadır. Gradyan $g_t$ için birinci ve ikinci moment kestirimleri,

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^{2}$$

bağıntılarıyla güncellenmekte; başlangıç yanlılığı düzeltilmiş momentler

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^{t}}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^{t}}$$

biçiminde hesaplanmakta ve parametre güncellemesi

$$\theta_t = \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

ile yapılmaktadır. Bu çalışmada başlangıç öğrenme oranı $\eta = 10^{-3}$, moment katsayıları ise olağan değerlerinde ($\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$) alınmıştır. Adam'ın parametre-bazlı uyarlanır adımları, farklı ölçeklerdeki gradyanlara sahip evrişimsel ve FiLM projeksiyon katmanlarının birlikte kararlı biçimde eğitilmesini sağlamaktadır.

Adam'ın iki belirleyici bileşeni, başlangıç yanlılığı düzeltmesi ve uyarlanır adım ölçeklemesidir. Birinci moment $m_t = (1 - \beta_1)\sum_{i=1}^{t}\beta_1^{\,t-i} g_i$ açılımıyla yazıldığında beklenen değeri $\mathbb{E}[m_t] = \mathbb{E}[g_t]\,(1 - \beta_1^{t})$ olur; sıfırdan başlatılan momentler bu nedenle ilk adımlarda gerçek gradyana göre sıfıra doğru yanlıdır. $1 - \beta_1^{t}$ ile bölme, bu yanlılığı tam olarak düzelterek özellikle eğitimin başındaki güncellemelerin güvenilir olmasını sağlamaktadır. İkinci moment $\hat{v}_t$ ile yapılan ölçekleme ise her parametreye, gradyan büyüklüğünün karekök-ortalamasıyla ters orantılı, uyarlanır bir adım atfetmektedir; böylece güncelleme adımının büyüklüğü yaklaşık olarak

$$|\Delta\theta_t| \approx \eta\,\frac{|\hat{m}_t|}{\sqrt{\hat{v}_t} + \epsilon} \lesssim \eta$$

ile sınırlanmaktadır. Bu özellik, momentum (birinci moment) ve RMSProp (ikinci moment) yöntemlerinin güçlü yönlerini birleştirerek, gradyan ölçekleri katmandan katmana değişen koşullu ağın kararlı yakınsamasını desteklemektedir.

### 3.7.2 Karma Hassasiyet Eğitimi ve Kayıp Ölçekleme

Eğitim, hesaplama hızı ve bellek kullanımı açısından karma hassasiyet (mixed precision) politikası altında yürütülmüştür. Bu politikada ileri ve geri yayılım hesaplamaları yarı hassasiyette (float16) yapılırken, ana ağırlıklar tam hassasiyette (float32) tutulmaktadır [36]. Yarı hassasiyet, Tensor Çekirdeği (Tensor Core) donanımına sahip grafik işlemcilerde (T4, A100, L4) işlem hızını yaklaşık $1{,}5$–$2$ kat artırmakta ve etkinlik belleğini yarıya indirmektedir.

Yarı hassasiyetin temel riski, küçük gradyanların float16 alt-taşma sınırının altında kalarak sıfıra yuvarlanmasıdır. Bu sorun, kayıp ölçekleme (loss scaling) ile giderilmiştir: kayıp, geri yayılımdan önce büyük bir katsayıyla ölçeklenmekte, gradyanlar hesaplandıktan sonra aynı katsayıyla geri ölçeklenmektedir. Böylece gradyanlar, ara hesaplama boyunca float16'nın temsil edilebilir aralığında tutulmaktadır. Bu işlev, Adam optimize edicisinin bir kayıp-ölçekleme optimize edicisiyle (LossScaleOptimizer) sarmalanmasıyla sağlanmış; sarmalama, herhangi bir başarısızlık ya da tam hassasiyet politikası durumunda yalın Adam'a güvenli biçimde geri dönecek şekilde korumaya alınmıştır. Maske çıkış katmanı ve maske uygulama çarpımının tam hassasiyette sabitlenmesiyle (Alt Başlık 3.5.5) birlikte, karma hassasiyet doğruluk kaybı olmaksızın uygulanmaktadır.

Yarı hassasiyetin sayısal kısıtı, sayı biçiminden kaynaklanmaktadır: float16, bir işaret, beş üs ve on mantis bitiyle temsil edilmekte; en küçük pozitif normal değeri yaklaşık $6{,}1 \times 10^{-5}$, en büyük değeri ise $65\,504$ olmaktadır. Bu dinamik aralık, float32'ye göre belirgin biçimde dardır. Derin ağların geri yayılımında gradyanların önemli bir bölümü $10^{-5}$ mertebesinin altına inebildiğinden, doğrudan float16 ile temsil edildiklerinde sıfıra yuvarlanmaktadır. Kayıp ölçekleme, gradyanın doğrusallığından ($\nabla(S\,\mathcal{L}) = S\,\nabla\mathcal{L}$) yararlanarak kaybı bir $S$ katsayısıyla çarpmakta ve tüm gradyanları $S$ kat yukarı, temsil edilebilir aralığa kaydırmakta; parametre güncellemesinden önce gradyanlar $S$'e bölünerek özgün ölçeklerine döndürülmektedir. Böylece sayısal doğruluk korunurken yarı hassasiyetin hız ve bellek kazanımı elde edilmektedir.

### 3.7.3 XLA (JIT) Derlemesi

Eğitim adımı, TensorFlow çerçevesinin [37] XLA (Accelerated Linear Algebra) tam-zamanında (Just-In-Time, JIT) derleyicisiyle derlenmiştir. XLA, hesaplama grafiğindeki ardışık işlemleri tek bir eniyilenmiş çekirdekte birleştirerek (operator fusion) çekirdek başlatma yükünü ve ara sonuçların bellek üzerinden taşınmasını azaltmaktadır. Bu birleştirme, özellikle yüksek verimli grafik işlemcilerde (A100, L4) belirgin bir hızlanma sağlamakta; T4 üzerinde de katkı sunmaktadır. JIT derleme, eğitim betiğinde bir ortam değişkeniyle denetlenebilmekte ve herhangi bir işlemin derlenememesi durumunda devre dışı bırakılabilmektedir.

### 3.7.4 Öğrenme Oranı Çizelgeleme ve Erken Durdurma

Eğitim, üç geri çağırma (callback) işleviyle denetlenmiştir. Birincisi, doğrulama kaybının (Doğrulama Kaybı) dört dönem boyunca iyileşmemesi durumunda öğrenme oranını yarıya indiren ve alt sınırı $10^{-6}$ olan uyarlanır öğrenme oranı çizelgeleyicisidir (ReduceLROnPlateau). İkincisi, Doğrulama Kaybının on dönem boyunca iyileşmemesi durumunda eğitimi durduran ve en iyi ağırlıkları geri yükleyen erken durdurma (EarlyStopping) işlevidir. Üçüncüsü, Doğrulama Kaybı en düşük olan modeli diske kaydeden model kontrol noktası (ModelCheckpoint) işlevidir. Bu üçlü, hem aşırı uyumu (overfitting) sınırlandırmakta hem de eğitim sonunda en başarılı kontrol noktasının korunmasını sağlamaktadır.

### 3.7.5 Paralel Veri Hattı

Tek bir Python üreteci üzerinde çalışan ve örnek başına iki librosa STFT hesaplayan veri hattının, grafik işlemcisini aç bıraktığı gözlemlenmiştir; bir A100 üzerinde yaklaşık $20$ ms'de çalışması beklenen bir eğitim adımı, üretecin darboğazı nedeniyle yaklaşık $100$ ms sürmüştür. Bu darboğaz, birden çok bağımsız karışım üretecinin işçi iş parçacıkları arasında dönüşümlü olarak işletilmesiyle (tf.data interleave) giderilmiştir. Varsayılan olarak dört paralel üretici kullanılmakta; her üretici ayrı bir tohum (seed) değeriyle başlatılmaktadır. Çözülmüş veri kümesi, tüm üreticiler ve doğrulama üreteci tarafından paylaşılan tek bir bellek içi sözlükte tutulmaktadır; böylece dört veri işçisi, çok gigabaytlık veri kümesinin dört kopyası yerine yalnızca dört iş parçacığı maliyeti getirmektedir. Hat, ön getirme (prefetch) ile tamamlanarak veri hazırlamanın hesaplama ile örtüşmesi sağlanmıştır.

Bu alt başlıkta açıklanan optimizasyon ve denetim mekanizmalarının bütünleştiği eğitim süreci, Algoritma 3.2'de özetlenmiştir.

```
Algoritma 3.2: Koşullu ayrıştırıcının eğitim döngüsü
─────────────────────────────────────────────────────────────────────
Girdi : üreteç akışı G (Algoritma 3.1), model f_θ, Adam(η),
        dönem sayısı E, dönem başına adım S, kayıp ölçeği s_kayıp,
        tespit kaybı ağırlığı w_t
Çıktı : en düşük doğrulama kaybına sahip θ* parametreleri
─────────────────────────────────────────────────────────────────────
 1:  Karma hassasiyet politikasını ata (mixed_float16); XLA derlemesini etkinleştir
 2:  for e ← 1 to E do
 3:      for adım ← 1 to S do
 4:          (X_log, q, X_lin, T) ← G'den bir yığın al
 5:          y_var ← [ max|T| > 10⁻⁶ ]                ▷ varlık etiketi (1/0)
 6:          (M, p̂) ← f_θ(X_log, q)                   ▷ maske ve varlık olasılığı
 7:          Ŝ ← M ⊙ X_lin                            ▷ kestirilen stem genliği
 8:          L ← L_ÇÇL1(Ŝ, T) + w_t · BCE(p̂, y_var)
 9:          L̃ ← s_kayıp · L                          ▷ kayıp ölçekleme (float16 alt-taşması)
10:          g ← ∇_θ L̃;   g ← g / s_kayıp             ▷ gradyanı geri ölçekle
11:          θ ← AdamGüncelle(θ, g, η)
12:      L_doğ ← SabitDoğrulamaKümesindeKayıp(f_θ)
13:      θ*, η ← KontrolNoktası_ÖğrenmeOranı_ErkenDurdurma(L_doğ, θ, η)
14:  return θ*
```

### 3.7.6 Hiperparametreler ve Donanım Ortamı

Önerilen modelde kullanılan başlıca hiperparametreler Tablo 3.3'te özetlenmiştir.

**Tablo 3.3:** Önerilen modelin eğitim hiperparametreleri.

| Parametre | Değer |
|---|---|
| Temel filtre sayısı (`base_filters`) | 32 |
| Yığın boyutu | 32 |
| Dönem sayısı | 60 |
| Dönem başına adım | 500 |
| Doğrulama örneği sayısı | 800 |
| Başlangıç öğrenme oranı ($\eta$) | $10^{-3}$ |
| Erken durdurma sabrı | 10 dönem |
| Negatif örnek olasılığı ($P_{\text{negatif}}$) | 0,50 |
| Arka plan gürültüsü olasılığı ($P_{\text{gürültü}}$) | 0,10 |
| Gürültü SNR aralığı | 15–30 dB |
| Minimum klip tabanı ($N_{\min}$) | 40 |
| Tespit kaybı (BCE) ağırlığı | 0,5 |
| Tespit başı boyutu | 128 |

Eğitim, Tensor Çekirdeği destekli grafik işlemcileri (Google Colab ortamında A100 80 GB ve T4) üzerinde, karma hassasiyet ve XLA etkin biçimde yürütülmüştür. Eğitim süreci boyunca Doğrulama Kaybı izlenmiş; önerilen modelde altmış dönem boyunca kararlı bir yakınsama gözlemlenmiştir. Eğitim, çıkarım ve değerlendirme betikleri, GPU bulunmayan ortamlar için Colab not defterleriyle de yinelenebilir kılınmıştır.

## 3.8 Çıkarım Hattı

Eğitilmiş model, bir web uygulaması üzerinden uçtan uca bir gürültü temizleme hattıyla işlevsel kılınmıştır. Çıkarım hattı, kullanıcının yüklediği dosyada bulunan sınıfların tespit edilmesi ve seçilen sınıfların dosyadan çıkarılması olmak üzere iki ana aşamadan oluşmaktadır.

### 3.8.1 Ses Olayı Tespiti

Tespit aşamasında, yüklenen dosyadan hız için en çok sekiz saniyelik bir bölüm bir saniyelik pencerelere ayrılmaktadır. Her pencere, eğitimle tutarlılığı korumak için tepe genliği $1{,}0$ olacak biçimde normalize edilmekte ve modele beslenmektedir. Tespit başına sahip modellerde, her aday sınıf için pencereler üzerinden ortalama varlık olasılığı bir tespit puanı olarak kullanılmaktadır. Tespit başının yakınsamadığı ve tüm sınıflar için yaklaşık $0{,}5$ değerinde neredeyse düzgün bir olasılık ürettiği durumlara karşı bir koruma tanımlanmıştır: puan aralığı $0{,}15$ eşiğinin altında kalırsa, hat maske-enerjisi sezgiseline geri dönmektedir.

Aday sınıflar, varsa modele ait tespit izin listesiyle (allow-list) sınırlandırılmakta; böylece yerel verinin doğrulayamadığı sınıflar aday havuzundan çıkarılmaktadır. Yüzeye çıkarılacak sınıflar, iki ölçütle belirlenmektedir: mutlak bir taban ($0{,}05$) ve kazanan sınıf puanının göreli bir oranı ($0{,}80 \times$ kazanan). Bu iki eşiğin büyüğü kesme değeri olarak alınmakta ve eşiği aşan en çok beş sınıf kullanıcıya sunulmaktadır. Bu çalışma noktası ($\text{taban} = 0{,}05$, $\text{göreli kesme} = 0{,}80$, $k = 5$), dördüncü bölümde sunulan eşik taramasında en az sahte yüzeyleme ile en yüksek kesinliği veren ayar olarak seçilmiştir. Tespit süreci Algoritma 3.3'te verilmiştir.

```
Algoritma 3.3: Ses olayı tespiti
─────────────────────────────────────────────────────────────────────
Girdi : ses, model f, sınıf listesi 𝒞, taban τ=0,05, göreli kesme ρ=0,80,
        üst sınır k=5, izin listesi 𝒜
Çıktı : tespit edilen sınıflar kümesi D
─────────────────────────────────────────────────────────────────────
 1:  W ← ses'i 1 sn pencerelere böl (en çok 8 pencere)
 2:  for each w ∈ W do
 3:      w ← w / max|w|;   (X_log^w, X_lin^w) ← Spektrogram(w)
 4:  for each c ∈ 𝒞 do
 5:      p_c ← ort_{w ∈ W} f(X_log^w, OneHot(c)).varlık     ▷ pencereler üzeri ortalama
 6:  if ( max_c p_c − min_c p_c ) < 0,15 then               ▷ tespit başı çöküş koruması
 7:      p ← MaskeEnerjisiPuanı(·)                           ▷ sezgisele geri dön
 8:  𝒞 ← 𝒞 ∩ 𝒜                                              ▷ izin listesiyle sınırla
 9:  kazanan ← argmax_{c ∈ 𝒞} p_c
10:  kesme ← max(τ, ρ · p_kazanan)
11:  D ← { c ∈ 𝒞 : p_c ≥ kesme } kümesinin en yüksek puanlı k tanesi
12:  return D
```

### 3.8.2 Oran Maskesi ile Kaynak Çıkarma

Çıkarma aşamasında, her seçili sınıf için model bir genlik kestirimi üretmekte ve bu kestirimden bir bastırma maskesi türetilmektedir. Bir zaman-frekans hücresinde, kestirilen stem genliği $\hat{e}_c$ ile karışım genliği $|X|$ arasındaki genlik oranı

$$r_c = \mathrm{clip}\!\left(\frac{\hat{e}_c}{|X| + \varepsilon},\, 0,\, 1\right)$$

biçiminde hesaplanmaktadır. Genlik oranının (güç oranı $\hat{e}_c^2/|X|^2$ yerine) tercih edilmesi, modelin tutucu (küçük) genlik kestirimleri ürettiği durumlarda daha yüksek bir tepkisellik sağlamasındandır. Seçilen tüm sınıflar için bastırma maskeleri çarpımsal olarak birleştirilmekte; bir kullanıcı denetimli *çıkarma kuvveti* katsayısı $\lambda \in [0, 1]$ ile bastırma şiddeti ayarlanmaktadır:

$$M = \prod_{c \in \text{seçili}} \mathrm{clip}\!\big(1 - \lambda\, r_c,\, 0,\, 1\big).$$

$\lambda = 1$ tam bastırmaya, $\lambda = 0$ ise hiç bastırma yapılmamasına karşılık gelmektedir. Çarpımsal birleştirme, birden çok sınıfın aynı hücredeki katkılarının kümülatif olarak bastırılmasını sağlamaktadır.

### 3.8.3 Hanning Pencereli Örtüşmeli Toplama

Çıkarma işlemi, tüm dosya boyunca, model girişinin sabit zaman boyutuna ($128$ çerçeve) uyacak biçimde parçalar hâlinde uygulanmaktadır. Parçalar arası adım, $\text{TIME\_FRAMES}/4 = 32$ çerçeve (yaklaşık $0{,}25$ s) olarak belirlenmiş; bu, ardışık parçalar arasında $\%75$ örtüşme sağlamaktadır. Her parça, bir Hann penceresiyle ağırlıklandırılıp örtüşmeli toplama (overlap-add) ile birleştirilmekte ve birikmiş ağırlıklara bölünerek normalize edilmektedir. Örtüşme oranının $\%50$'den $\%75$'e çıkarılması, parça sınırlarında düzenli aralıklarla duyulan ve dördüncü bölümde belgelenen darbeli yapaylığı (boundary pulsing) gidermiştir. Ayrıca, müzikal gürültüyü bastırmak için maske, zaman ekseninde beş çerçevelik (yaklaşık $40$ ms) bir ortalama çekirdeğiyle düzleştirilmektedir. Çıkarma aşamasının tümü, oran maskesi (Alt Başlık 3.8.2), örtüşmeli toplama ve faz yeniden kullanımı (Alt Başlık 3.8.4) adımlarını birleştiren Algoritma 3.4'te verilmiştir.

```
Algoritma 3.4: Hanning örtüşmeli toplama ile kaynak çıkarma
─────────────────────────────────────────────────────────────────────
Girdi : ses, seçili sınıflar 𝒮, çıkarma kuvveti λ, pencere TF=128,
        adım H=TF/4, model f
Çıktı : temizlenmiş ses
─────────────────────────────────────────────────────────────────────
 1:  tepe ← max|ses|;   ses ← ses / tepe
 2:  𝐗 ← STFT(ses);   Φ ← ∠𝐗;   |𝐗| ← genlik;   T ← çerçeve sayısı
 3:  A ← 0_{F×T};   Wʷ ← 0_T;   ℎ ← Hann(TF)
 4:  for başla ← 0, H, 2H, … , T−1 do
 5:      bitir ← min(başla+TF, T);   C ← |𝐗|[:, başla:bitir]   ▷ gerekirse sıfır-doldur
 6:      M ← 1
 7:      for each c ∈ 𝒮 do
 8:          ê_c ← f(log(1+C), OneHot(c)).maske ⊙ C
 9:          r_c ← clip( ê_c / (C + ε), 0, 1 )                  ▷ genlik oranı
10:          M ← M · clip( 1 − λ · r_c, 0, 1 )                  ▷ çarpımsal birleştirme
11:      M ← ZamandaDüzleştir(M, 5)                             ▷ ~40 ms ortalama
12:      A[:, başla:bitir] ← A[:, başla:bitir] + ℎ · (C · M)
13:      Wʷ[başla:bitir] ← Wʷ[başla:bitir] + ℎ
14:  |Ŷ| ← A / max(Wʷ, ε)
15:  return ISTFT( |Ŷ| · e^{jΦ} ) · tepe
```

### 3.8.4 Faz Yeniden Kullanımı ve Ters STFT

Model yalnızca genlik düzleminde çalıştığından, faz bilgisi kestirilmemekte; bunun yerine karışımın özgün fazı yeniden kullanılmaktadır. Tüm dosya için tek bir STFT hesaplanmakta ve böylece faz, parça sınırları boyunca küresel olarak tutarlı kalmaktadır. Maskelenmiş genlik, düşürülen Nyquist bini sıfırla yeniden eklendikten sonra karışımın karmaşık fazıyla çarpılarak ters STFT'ye verilmekte ve zaman düzlemine geri döndürülmektedir. Yeniden sentezlenen dalga biçimi, başlangıçtaki tepe normalizasyonu tersine çevrilerek özgün ölçeğine döndürülmekte; gerekirse kırpılmayı önlemek için yeniden ölçeklenmektedir. Genlikten faz kestiren yinelemeli Griffin–Lim türü yöntemler [38] daha yüksek bir yeniden sentez kalitesi sunabilmekle birlikte, ek hesaplama maliyeti ve yakınsama belirsizliği nedeniyle bu çalışmada faz yeniden kullanımı tercih edilmiştir.

### 3.8.5 Web Uygulaması ve Video Entegrasyonu

Çıkarım hattı, bir Gradio web uygulaması üzerinden sunulmaktadır. Kullanıcı bir ses ya da video dosyası yüklemekte, tespit edilen sınıfları işaretlemekte, çıkarma kuvvetini bir kaydırma çubuğuyla ayarlamakta ve temizlenmiş çıktıyı indirmektedir. Yalnızca ses içeren girişlerde, işlem öncesi ve sonrası karşılaştırma için bir "önce/sonra" ses çifti sunulmaktadır. Ek olarak, her seçili sınıfın model tarafından çıkarılan stem'i ayrı ayrı oynatılabilmekte; böylece her sınıfın hangi içerik olarak algılandığı işitsel olarak doğrulanabilmektedir. Video girişlerinde, temizlenen ses izi `ffmpeg` aracılığıyla özgün görüntü izinin üzerine yeniden bindirilerek video çıktısı üretilmektedir. Bu uçtan uca hat, üçüncü bölümde açıklanan tüm bileşenleri (ön işleme, koşullu model, tespit ve çıkarma) tek bir kullanıcı arayüzünde birleştirmektedir.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 4. BULGULAR VE TARTIŞMA

Bu bölümde, önerilen sistemin başarımı niceliksel ve niteliksel olarak değerlendirilmiştir. Önce değerlendirme metrikleri tanımlanmış; ardından nihai yapılandırmaya yol açan tasarım kararları, kendilerine yol açan deneysel gözlemler ve başarısızlık örüntüleriyle birlikte çözümlenmiştir. Sonraki başlıklarda ayrıştırma başarımı, tespit başarımı, FiLM koşullandırmasının katkısı, eşik taraması ve niteliksel sonuçlar sunulmuş; bölüm, sistemin sınırlılıklarının tartışılmasıyla tamamlanmıştır. Niceliksel sonuçlar, önerilen model için iki yüz sentetik karışımdan oluşan bir test kümesi üzerinde elde edilmiştir.

## 4.1 Değerlendirme Metodolojisi ve Metrikler

Sistem, ayrıştırma ve tespit olmak üzere iki ayrı görev için ayrı metriklerle değerlendirilmiştir. Değerlendirme, eğitimden bağımsız bir tohum değeriyle üretilen ve bileşen sınıfları bilinen sentetik karışımlar üzerinde yapılmıştır.

### 4.1.1 Ölçek-Değişmez İşaret-Bozulma Oranı

Ayrıştırma kalitesi, ölçek-değişmez işaret-bozulma oranı (Scale-Invariant Signal-to-Distortion Ratio, SI-SDR) ile ölçülmüştür [39]. SI-SDR, kestirilen kaynak $\hat{s}$ ile referans kaynak $s$ arasındaki benzerliği, ölçek farklılıklarına karşı duyarsız biçimde değerlendirmektedir. Ortalamaları çıkarılmış sinyaller için, referans yönündeki izdüşüm katsayısı

$$\alpha = \frac{\hat{s}^{\top} s}{\lVert s \rVert^{2}}$$

ile hesaplanmakta; hedef bileşeni $s_{\text{hedef}} = \alpha s$ ve bozulma bileşeni $e = \hat{s} - \alpha s$ olmak üzere ölçüt

$$\text{SI-SDR}(\hat{s}, s) = 10 \log_{10} \frac{\lVert \alpha s \rVert^{2}}{\lVert \hat{s} - \alpha s \rVert^{2}}$$

biçiminde tanımlanmaktadır. Başlıca sonuç değeri olan SI-SDRi (improvement), modelin kestirdiği stem'in SI-SDR değeri ile işlenmemiş karışımın ("hiçbir şey yapmama" temel çizgisinin) SI-SDR değeri arasındaki farktır:

$$\text{SI-SDRi} = \text{SI-SDR}(\hat{s}, s) - \text{SI-SDR}(x, s).$$

Pozitif bir SI-SDRi değeri, modelin işlenmemiş karışıma kıyasla bir iyileştirme sağladığını göstermektedir. SI-SDR ölçütü sessizliğe karşı tanımsız olduğundan, ayrıştırma değerlendirmesi yalnızca pozitif örnekler (sorgu sınıfının karışımda bulunduğu durumlar) üzerinde yapılmaktadır.

SI-SDR'nin ölçek-değişmezliği, hedef bileşeninin kestirim üzerine dik izdüşümünden kaynaklanmaktadır. $\alpha = \hat{s}^{\top}s / \lVert s \rVert^{2}$ katsayısı, $\hat{s}$ vektörünü $s$ doğrultusuna izdüşüren en küçük kareler çözümüdür; bu seçim, hedef bileşeni $s_{\text{hedef}} = \alpha s$ ile bozulma bileşeni $e = \hat{s} - \alpha s$ vektörlerini birbirine dik kılmaktadır ($s_{\text{hedef}}^{\top} e = 0$). Kestirimin sabit bir $\kappa$ katsayısıyla ölçeklenmesi ($\hat{s} \to \kappa\hat{s}$), hem $s_{\text{hedef}}$ hem de $e$ bileşenlerini aynı $\kappa$ katıyla büyüttüğünden, ikisinin güç oranı ve dolayısıyla SI-SDR değeri değişmeden kalmaktadır. Bu özellik, ölçütün modelin genel bir kazanç (ölçek) hatasından etkilenmemesini; bunun yerine yalnızca hedefin spektro-zamansal yapısının ne ölçüde geri kazanıldığını ölçmesini sağlamaktadır.

### 4.1.2 Tespit Metrikleri

Tespit başarımı, kesinlik (precision), duyarlılık (recall) ve bunların harmonik ortalaması olan $F_1$ ölçütüyle değerlendirilmiştir. Bir sınıf için doğru pozitif (DP), yanlış pozitif (YP) ve yanlış negatif (YN) sayıları üzerinden

$$\text{Kesinlik} = \frac{\text{DP}}{\text{DP} + \text{YP}}, \qquad \text{Duyarlılık} = \frac{\text{DP}}{\text{DP} + \text{YN}},$$

$$F_1 = \frac{2 \cdot \text{Kesinlik} \cdot \text{Duyarlılık}}{\text{Kesinlik} + \text{Duyarlılık}}$$

tanımları kullanılmaktadır. Genel başarım, sınıf bazlı $F_1$ değerlerinin ortalaması olan makro $F_1$ ile özetlenmektedir. Makro ortalama, her sınıfa eşit ağırlık verdiğinden, sınıf dengesizliğinden bağımsız bir başarım göstergesi sağlamaktadır. Her sentetik karışım için bileşen sınıfları bilindiğinden, tespit edilen sınıflar bu yer-gerçek (ground-truth) kümesiyle karşılaştırılarak DP, YP ve YN sayıları biriktirilmektedir.

## 4.2 Tasarım Kararlarının Deneysel Gerekçeleri

Önerilen modelin nihai yapılandırması, tek bir tasarım turunda değil, bir dizi denetimli deneyin gözlemlerine dayanarak belirlenmiştir. Bu alt başlıkta, başlıca tasarım kararları, kendilerine yol açan deneysel gözlemlerle birlikte sunulmuştur. Söz konusu gözlemler, derin öğrenme tabanlı bir ayrıştırma sisteminin geliştirilmesinde karşılaşılan tipik tuzakları da ortaya koymaktadır. Denenen başlıca değişiklikler ve gözlemlenen sonuçlar Tablo 4.1'de özetlenmiştir.

**Tablo 4.1:** Denenen tasarım değişiklikleri ve gözlemlenen sonuçlar.

| Denenen değişiklik | Gözlemlenen sonuç |
|---|---|
| Agresif veri artırımı ($P_{\text{negatif}}=0{,}45$; SNR 5–20 dB) | Sessizliğe çöküş |
| Artırma parametrelerinin ölçülü değerlere çekilmesi + çıkarım normalizasyonu | $F_1=0{,}21$; SI-SDRi $-22{,}18$ dB |
| Tam-kodlayıcı FiLM + çok çözünürlüklü L1 + %75 örtüşme | Sınır darbesi yapaylığı giderildi |
| Geniş dış veri kümesinin eklenmesi (235 sınıf) | $F_1=0{,}02$ (fantom yanlış pozitifler) |
| Minimum klip tabanı + tespit izin listesi | $F_1=0{,}09$; çalışma noktası cap$=0{,}80$, $k=5$ |
| Maske-enerjisi yerine öğrenilmiş tespit başı | $F_1=0{,}17$ (baş yetersiz uyum) |
| Tespit kaybında odak kaybı (focal loss) | Gradyan çöküşü |
| BCE tespit kaybı + dış veri kümesinin kaldırılması (56 sınıf) | $F_1=0{,}32$ |
| Düzenlenmiş 15 sınıflı sözcük dağarcığı | $F_1=0{,}692$; SI-SDRi $-13{,}07$ dB |

**Veri artırımının dengelenmesi.** Veri artırımının agresif biçimde uygulanması — negatif örnek olasılığının $0{,}45$'e ve gürültü düzeyinin $5$ dB SNR'a çıkarılması — modeli her sorgu için yakın-sıfır maske üreten "güvenli sessizlik" dengesine iterek bütünüyle işlevsiz kılmıştır. Bu gözlem, negatif örnek oranının (Alt Başlık 3.4.2) ölçülü bir değerde tutulması ve gürültü SNR aralığının gerçekçi düzeylere ($15$–$30$ dB) çekilmesi kararının doğrudan gerekçesidir. Buna ek olarak, çıkarım hattının ham (normalize edilmemiş) ses beslemesi, eğitim-çıkarım ölçek uyumsuzluğunu açığa çıkarmış ve tepe normalizasyonunun (Alt Başlık 3.4.5) çıkarım hattında birebir uygulanması gerekliliğini ortaya koymuştur.

**Mimari ve kayıp iyileştirmeleri.** FiLM koşullandırmasının tüm kodlayıcı seviyelerine yayılması, çok çözünürlüklü L1 kaybının eklenmesi ve örtüşme oranının $\%75$'e çıkarılması; sırasıyla maske kesinliğini artırmış, spektral biçim yakınsamasını kararlı kılmış ve parça sınırlarındaki darbeli yapaylığı gidermiştir.

**Sözcük dağarcığı ve yanlış pozitifler.** Geniş bir dış veri kümesinin eklenmesi, sözcük dağarcığını yüzlerce sınıfa genişletmiş; ancak yerel sesle desteklenmeyen sınıfların fantom yanlış pozitifler üretmesiyle makro $F_1$ değeri $0{,}02$'ye gerilemiştir. Bu çöküş, önce sınıf başına minimum klip tabanının uygulanmasının, ardından bu dış veri kümesinin bütünüyle kaldırılmasının gerekçesini oluşturmuştur.

**Öğrenilmiş tespit ve kayıp seçimi.** Tespit probleminin maske-enerjisi sezgiseliyle yapısal olarak çözülemeyeceğinin anlaşılması üzerine, öğrenilmiş bir tespit başı eklenmiştir. Bu başın odak kaybıyla (focal loss) eğitilmesi denendiğinde, Alt Başlık 3.6.2'de çözümlenen gradyan çöküşü gözlemlenmiş; başın çıkışı her sınıf için yaklaşık $0{,}5$ değerine çökmüştür. İkili çapraz entropiye dönülmesi ve dış veri kümesinin kaldırılmasıyla makro $F_1$ değeri $0{,}32$'ye yükselmiştir.

**Düzenlenmiş sözcük dağarcığı.** Bu aşamadaki sınıf bazlı sonuçların belirgin biçimde iki kutuplu olması — bir grup sınıfın yüksek, geniş bantlı bir grup sınıfın ise sıfıra yakın $F_1$ üretmesi — nihai tasarım ilkesini belirlemiştir: ayrıştırma ve tespit başarımının en yüksek olduğu on beş sınıfın korunması. Bu düzenleme, hem makro ortalamayı yalnızca başarılı sınıflar üzerinden hesaplatmış hem de aşırı-tetikleyen geniş bantlı sınıfları aday havuzundan çıkararak göreli kesme eşiğinin gerçek sınıfları bastırmasını önlemiştir. Sonuçta makro $F_1$ değeri $0{,}692$'ye ulaşmıştır.

## 4.3 Ayrıştırma Başarımı

Önerilen modelde ortalama SI-SDRi değeri $-13{,}07$ dB olarak ölçülmüştür. Bu değerin negatif olması, ilk bakışta modelin işlenmemiş karışıma kıyasla bir iyileştirme sağlamadığını düşündürse de, sınıf bazlı ve karışım-bazlı çözümleme daha incelikli bir tabloyu ortaya koymaktadır. Sınıf bazlı SI-SDRi değerleri Şekil 4.1'de, işlenmemiş karışım ile modelin kestirimine ait SI-SDR değerlerinin karşılaştırması ise Şekil 4.2'de gösterilmiştir.

![Şekil 4.1](../thesis_figures/09_sisdr_per_class.png)

**Şekil 4.1:** Sınıf bazlı SI-SDRi değerleri.

![Şekil 4.2](../thesis_figures/10_sisdr_mix_vs_model.png)

**Şekil 4.2:** İşlenmemiş karışım (SI-SDR mix) ile model kestiriminin (SI-SDR model) sınıf bazlı karşılaştırması.

Sınıf bazlı çözümleme, SI-SDRi değerinin sınıflar arasında geniş bir aralığa yayıldığını göstermektedir. Bir grup sınıf pozitif SI-SDRi üretmektedir: çalar saat ($+10{,}39$ dB), öksürük ($+5{,}12$ dB), hapşırık ($+4{,}43$ dB), karga ($+2{,}99$ dB) ve el testeresi ($+0{,}85$ dB). Buna karşılık bazı sınıflar belirgin biçimde negatif değerler vermektedir: alkış ($-28{,}79$ dB), elektrikli süpürge ($-27{,}12$ dB) ve sifon ($-22{,}08$ dB).

Bu dağılımın temel nedeni, SI-SDR ölçütünün karışımdaki hedef baskınlığına olan duyarlılığıdır. İşlenmemiş karışımın SI-SDR değeri zaten yüksek olan, yani hedef kaynağın karışıma hâkim olduğu örneklerde (örneğin elektrikli süpürge için işlenmemiş karışımın SI-SDR değeri $47{,}59$ dB'dir), herhangi bir spektrogram maskeleme işlemi dalga biçimi düzeyindeki SI-SDR değerini düşürmektedir; çünkü hedef hâlihazırda neredeyse izole hâldedir ve maske ancak hata ekleyebilir. Tersine, hedefin karışım içinde gömülü olduğu örneklerde (örneğin çalar saat için işlenmemiş karışımın SI-SDR değeri $0{,}42$ dB'dir, model bunu $10{,}80$ dB'ye çıkarmaktadır) model belirgin bir iyileştirme sağlamaktadır. Dolayısıyla negatif ortalama SI-SDRi, modelin başarısızlığından çok, test kümesindeki hedef-baskın karışımların ağırlığını ve seçilen ölçütün dalga biçimi-düzeyli doğasını yansıtmaktadır. Ek olarak SI-SDR, izole stem'in dalga biçimi yeniden yapılandırma kalitesini ölçtüğünden, faz yeniden kullanımıyla (Alt Başlık 3.8.4) yapısal olarak sınırlanmaktadır; web uygulamasında algılanan ve çıkarma sonrası artığa dayanan algısal kalite, bu ölçütle birebir yakalanmamaktadır.

## 4.4 Tespit Başarımı

Önerilen modelde tespit makro $F_1$ değeri $0{,}692$ olarak ölçülmüş; toplam doğru pozitif, yanlış pozitif ve yanlış negatif sayıları sırasıyla $433$, $34$ ve $299$ olarak elde edilmiştir. Sınıf bazlı kesinlik, duyarlılık ve $F_1$ değerleri Şekil 4.3'te, toplam DP/YP/YN dağılımı ise Şekil 4.4'te gösterilmiştir. Sınıf bazlı kesinlik, duyarlılık, $F_1$ ve SI-SDRi değerleri Tablo 4.2'de bir arada sunulmuştur.

![Şekil 4.3](../thesis_figures/02_detection_per_class_prf.png)

**Şekil 4.3:** Sınıf bazlı kesinlik, duyarlılık ve $F_1$ değerleri.

![Şekil 4.4](../thesis_figures/03_detection_totals.png)

**Şekil 4.4:** Toplam doğru pozitif, yanlış pozitif ve yanlış negatif sayıları.

**Tablo 4.2:** Sınıf bazlı tespit ve ayrıştırma başarımı.

| Sınıf | Kesinlik | Duyarlılık | $F_1$ | SI-SDRi (dB) |
|---|---|---|---|---|
| clock_alarm | 1,00 | 0,89 | 0,939 | +10,39 |
| helicopter | 0,97 | 0,90 | 0,933 | −5,54 |
| vacuum_cleaner | 0,98 | 0,87 | 0,918 | −27,12 |
| church_bells | 0,91 | 0,89 | 0,903 | −16,35 |
| hand_saw | 0,97 | 0,73 | 0,831 | +0,85 |
| clapping | 0,96 | 0,67 | 0,788 | −28,79 |
| crying_baby | 0,92 | 0,68 | 0,783 | −16,00 |
| sea_waves | 0,90 | 0,63 | 0,740 | −17,23 |
| crow | 0,97 | 0,53 | 0,686 | +2,99 |
| rain | 0,90 | 0,54 | 0,675 | −18,03 |
| brushing_teeth | 1,00 | 0,44 | 0,611 | −8,65 |
| gun_shot | 0,74 | 0,46 | 0,567 | −0,37 |
| toilet_flush | 0,89 | 0,38 | 0,531 | −22,08 |
| coughing | 0,80 | 0,17 | 0,281 | +5,12 |
| sneezing | 0,55 | 0,12 | 0,197 | +4,43 |

Sonuçlar, tespit başının yüksek kesinlikli ancak tutucu bir çalışma noktasında olduğunu göstermektedir. Kesinlik değerleri çoğu sınıfta $0{,}90$ ve üzerindedir; toplam yanlış pozitif sayısının yalnızca $34$ olması, modelin nadiren sahte tespit ürettiğini ortaya koymaktadır. Buna karşılık duyarlılık, başarımı sınırlayan baskın etmendir; toplam yanlış negatif sayısının $299$ olması, modelin bazı gerçek sınıfları kaçırdığını göstermektedir. En düşük duyarlılık, kısa süreli ve geçici (transient) sesler olan hapşırık ($0{,}12$) ve öksürük ($0{,}17$) sınıflarında gözlemlenmektedir; bu sınıflar hem düşük enerjili hem de akustik olarak benzer olduklarından, tespit başının bu sınıflar için güvenli bir varlık olasılığı üretmesi güçleşmektedir. Çalar saat, helikopter ve elektrikli süpürge gibi sürekli ve ayırt edici tınıya sahip sınıflar ise $0{,}90$'ı aşan $F_1$ değerleriyle en başarılı sınıflardır. Tespit puanlarının alıcı işletim karakteristiği (ROC) ve kesinlik-duyarlılık (PR) eğrileri Şekil 4.5'te sunulmuştur.

![Şekil 4.5](../thesis_figures/05_detection_roc_pr.png)

**Şekil 4.5:** Tespit başının ROC ve kesinlik-duyarlılık eğrileri.

Şekil 4.5'teki eğriler, tespit başının ham sıralama (ranking) kalitesinin, tek bir eşikle elde edilen $F_1$ değerinden daha yüksek olduğunu ortaya koymaktadır. Mikro-ortalamalı ROC eğrisinin altındaki alan $0{,}907$, kesinlik-duyarlılık eğrisinin ortalama kesinliği (AP) ise $0{,}811$ değerindedir. ROC eğrisinin düşük yanlış-pozitif oranlarında dahi yüksek doğru-pozitif oranına ulaşması, varlık olasılığının ayırt edici bir sinyal taşıdığını göstermektedir. Makro $F_1$ değerinin ($0{,}692$) bu sıralama ölçütlerinin altında kalması, başarımın bir eşikleme (çalışma noktası) tercihinden kaynaklandığına; modelin öğrendiği olasılık sinyalinin kendisinin daha yüksek bir ayrım gücüne sahip olduğuna işaret etmektedir.

## 4.5 FiLM Koşullandırmasının Katkısı

FiLM koşullandırmasının ayrıştırmaya katkısı, doğru sınıf sorgulandığında üretilen çıkış enerjisi ile yanlış sınıf sorgulandığında üretilen çıkış enerjisinin oranıyla (ayrımcılık üstünlüğü, advantage) değerlendirilmiştir. Yüksek bir oran, modelin sorgulanan sınıfa göre çıktısını güçlü biçimde farklılaştırdığını, yani koşullandırmanın etkin çalıştığını göstermektedir. Sınıf bazlı ayrımcılık üstünlüğü Şekil 4.6'da, çıkış-giriş enerji oranı ise Şekil 4.7'de gösterilmiştir.

![Şekil 4.6](../thesis_figures/11_film_advantage.png)

**Şekil 4.6:** FiLM koşullandırmasının sınıf bazlı ayrımcılık üstünlüğü (doğru sorgu / yanlış sorgu enerji oranı).

![Şekil 4.7](../thesis_figures/12_out_in_ratio.png)

**Şekil 4.7:** Sınıf bazlı çıkış-giriş enerji oranı.

Çoğu sınıfta, doğru sorgu yüksek bir çıkış enerjisi üretirken yanlış sorgu yakın-sıfır enerji üretmektedir; örneğin diş fırçalama sınıfında doğru sorgunun enerjisi $8{,}43$, yanlış sorgunun enerjisi ise $1{,}58 \times 10^{-5}$ mertebesindedir; bu da yüz binleri aşan bir ayrımcılık üstünlüğüne karşılık gelmektedir. Deniz dalgaları ve elektrikli süpürge gibi sınıflarda bu oran milyonlar mertebesine ulaşmaktadır. Bu sonuç, çok seviyeli FiLM koşullandırmasının (Alt Başlık 3.5.4) modelin sorgulanan sınıfa göre seçici davranmasını sağladığını doğrulamaktadır. Görece zayıf bir durum, doğru ve yanlış sorgu enerjilerinin birbirine yakın olduğu (üstünlük $\approx 6{,}66$) helikopter sınıfında gözlemlenmektedir; bu sınıfta yanlış sorgu da kayda değer bir enerji üretmekte, ancak sınıfın yüksek tespit $F_1$ değeri ($0{,}933$) genel başarımı korumaktadır.

## 4.6 Eşik Taraması ve Çalışma Noktası Seçimi

Tespit aşamasında hangi sınıfların yüzeye çıkarılacağı, üç parametreyle denetlenmektedir: mutlak taban, kazanan sınıf puanına göre belirlenen göreli kesme ve yüzeye çıkarılacak en çok sınıf sayısı ($k$). Bu parametrelerin uygun çalışma noktası, bir ızgara taramasıyla belirlenmiştir. Tarama sonuçları Şekil 4.8'de, tespit puanlarının dağılımı Şekil 4.9'da ve en sık yanlış pozitif üreten sınıflar Şekil 4.10'da gösterilmiştir.

![Şekil 4.8](../thesis_figures/08_detection_threshold_sweep.png)

**Şekil 4.8:** Göreli kesme ve $k$ parametrelerinin makro $F_1$ üzerindeki etkisini gösteren eşik taraması.

![Şekil 4.9](../thesis_figures/06_detection_score_hist.png)

**Şekil 4.9:** Mevcut ve mevcut olmayan sınıflar için tespit puanlarının dağılımı.

![Şekil 4.10](../thesis_figures/07_detection_top_fp.png)

**Şekil 4.10:** En sık yanlış pozitif üreten sınıflar.

Şekil 4.8, makro $F_1$ değerinin göreli kesme $0{,}5$ ile $0{,}7$ aralığında yaklaşık $0{,}71$ düzeyinde bir plato oluşturduğunu; eşik sıkılaştıkça toplam yanlış pozitif sayısının (yaklaşık $165$'ten $20$'ye) toplam doğru pozitif sayısından (yaklaşık $510$'dan $390$'a) daha hızlı düştüğünü göstermektedir. Bu nedenle göreli kesmenin $0{,}80$'e çekilmesi, $F_1$ değerinden yalnızca küçük bir ödün vererek yanlış pozitifleri belirgin biçimde azaltmaktadır. Şekil 4.9 ise mevcut ($n = 732$) ve mevcut olmayan ($n = 3768$) sınıf-karışım çiftleri için tespit puanlarının yoğunluk dağılımını karşılaştırmaktadır; mevcut olmayan sınıflar sıfıra yakın bir tepe etrafında yoğunlaşırken, mevcut sınıflar $1{,}0$ değerinde belirgin bir tepe oluşturmaktadır. İki dağılımın $0{,}1$–$0{,}4$ aralığındaki örtüşmesi, kaçırılan gerçek sınıfların ve seyrek yanlış pozitiflerin ortak kaynağıdır. Şekil 4.10'da görülen en sık yanlış pozitif üreten sınıfların katkısı ise düşüktür; toplam yanlış pozitif sayısı $34$ ile sınırlı kaldığından, hiçbir sınıf sistematik bir sahte tespit kaynağı oluşturmamaktadır.

Tarama, göreli kesme eşiğinin yükseltilmesinin yanlış pozitifleri azaltmakla birlikte doğru pozitifleri de düşürdüğünü, dolayısıyla kesinlik ile duyarlılık arasında doğrudan bir ödünleşim bulunduğunu göstermektedir. Yürütülen eşik taramalarında, gevşek bir kesme (göreli kesme $0{,}65$, $k = 10$) daha çok doğru pozitif ancak yüksek yanlış pozitif; sıkı bir kesme (göreli kesme $0{,}90$, $k = 5$) ise az yanlış pozitif ancak çok sayıda kaçırılan sınıf üretmiştir. Bu uçların arasında, mutlak taban $0{,}05$, göreli kesme $0{,}80$ ve $k = 5$ değerleriyle tanımlanan çalışma noktası, en az sahte yüzeyleme ile kabul edilebilir bir duyarlılığı sağlayan ayar olarak seçilmiş ve hem web uygulamasında hem de değerlendirmede varsayılan olarak benimsenmiştir. Sınıfların birlikte tespit edilme örüntüleri Şekil 4.11'deki eş-zamanlı görünüm (co-occurrence) matrisinde sunulmuştur.

![Şekil 4.11](../thesis_figures/04_detection_cooccurrence.png)

**Şekil 4.11:** Sınıfların birlikte tespit edilme (co-occurrence) matrisi.

Şekil 4.11'deki satır-normalize edilmiş eş-zamanlı görünüm matrisinde baskın bir köşegen gözlemlenmektedir; bu, tespit edilen sınıfın çoğunlukla gerçekte mevcut olan sınıfla örtüştüğünü göstermektedir. Çalar saat, kilise çanları, helikopter ve elektrikli süpürge satırlarında köşegen değeri $0{,}8$ ve üzerindedir. Buna karşılık öksürük ve hapşırık satırları soluk bir köşegen sergilemekte; bu da bu sınıfların düşük duyarlılığını (sırasıyla $0{,}17$ ve $0{,}12$) doğrulamaktadır. Köşegen dışı değerlerin genel olarak düşük kalması, sınıflar arasında sistematik bir karışmanın bulunmadığına işaret etmektedir.

## 4.7 Niteliksel Sonuçlar

Niceliksel metriklerin yanı sıra, modelin çıktısı spektrogram görselleştirmeleri ve dinleme testleriyle niteliksel olarak da değerlendirilmiştir. Çalar saat, helikopter ve elektrikli süpürge sınıfları için karışım, hedef ve model kestirimi spektrogramları sırasıyla Şekil 4.12, Şekil 4.13 ve Şekil 4.14'te gösterilmiştir. Bu görselleştirmeler, modelin sorgulanan sınıfın spektro-zamansal yapısını koruyarak karışımın geri kalanını bastırdığını ortaya koymaktadır.

![Şekil 4.12](../thesis_figures/13_spectrogram_clock_alarm.png)

**Şekil 4.12:** Çalar saat sınıfı için karışım, hedef ve model kestirimi spektrogramları.

![Şekil 4.13](../thesis_figures/13_spectrogram_helicopter.png)

**Şekil 4.13:** Helikopter sınıfı için karışım, hedef ve model kestirimi spektrogramları.

![Şekil 4.14](../thesis_figures/13_spectrogram_vacuum_cleaner.png)

**Şekil 4.14:** Elektrikli süpürge sınıfı için karışım, hedef ve model kestirimi spektrogramları.

Şekil 4.12–4.14'teki dört panelli gösterimlerde (karışım, kestirilen maske, kestirilen stem ve gerçek stem) modelin kestirdiği stem'in gerçek stem ile görsel olarak büyük ölçüde örtüştüğü gözlemlenmektedir. Örneğin çalar saatin yatay harmonik bantları, karışımdaki diğer kaynakların (diş fırçalama ve kilise çanları) enerjisi bastırılırken kestirilen stem'de korunmaktadır. Bu görselleştirmeler, dalga biçimi düzlemindeki SI-SDR sınırlamasına karşın modelin, sorgulanan sınıfın spektro-zamansal yapısını algısal olarak doğru biçimde geri kazandığını desteklemektedir.

Çıkarma işleminin zaman düzlemindeki etkisi Şekil 4.15'teki gösterimde sunulmuştur; bu gösterim, seçilen sınıfın karışımdan çıkarılmasından önceki ve sonraki dalga biçimlerini karşılaştırmaktadır. Dinleme testlerinde, üçüncü bölümde açıklanan örtüşme oranı düzeltmesinin (Alt Başlık 3.8.3) ardından parça sınırlarındaki düzenli darbeli yapaylığın ortadan kalktığı doğrulanmıştır. Sürekli ve ayırt edici tınıya sahip sınıflarda (örneğin elektrikli süpürge, helikopter, çalar saat) çıkarmanın işitsel olarak belirgin biçimde etkili olduğu; kısa süreli geçici seslerde ise çıkarmanın daha sınırlı kaldığı gözlemlenmiştir.

![Şekil 4.15](../thesis_figures/14_removal_demo.png)

**Şekil 4.15:** Seçilen sınıfın çıkarılmasından önce ve sonra dalga biçimi karşılaştırması.

Şekil 4.15, çalar saatin bir karışımdan çıkarılması örneğini zaman ve frekans düzlemlerinde göstermektedir. Çıkarma öncesi dalga biçiminin genliği tüm süre boyunca yüksekken, çıkarma sonrası dalga biçiminin genliği belirgin biçimde düşmekte; yalnızca hedef dışında kalan bileşenlerin düşük düzeyli artığı korunmaktadır. Alt sıradaki spektrogramlarda, özgün kayıttaki çalar saatin düzenli yatay harmonik bantlarının temizlenmiş çıktıda büyük ölçüde kaybolması, çıkarmanın frekans düzlemindeki etkisini doğrudan görünür kılmaktadır.

## 4.8 Sınırlılıklar ve Tartışma

Elde edilen sonuçlar, sistemin çeşitli sınırlılıklarını da ortaya koymaktadır. Birinci sınırlılık, faz yeniden kullanımından kaynaklanmaktadır. Model yalnızca genlik düzleminde çalıştığından ve yeniden sentez karışımın fazını kullandığından, dalga biçimi düzeyindeki yeniden yapılandırma kalitesi yapısal olarak sınırlanmaktadır; bu durum, özellikle hedef-baskın karışımlarda negatif SI-SDRi değerlerinin başlıca nedenidir. Faz duyarlı ya da karmaşık maske türevleri (Alt Başlık 2.3) ile dalga biçimi düzlemli modeller (Alt Başlık 2.4) bu sınırı aşma potansiyeli taşımaktadır.

İkinci sınırlılık, kısa süreli ve düşük enerjili geçici seslerin (hapşırık, öksürük) düşük duyarlılıkla tespit edilmesidir. Bu sınıflar hem akustik olarak birbirine benzemekte hem de bir saniyelik pencere içinde seyrek biçimde konumlandığından, tespit başının güvenli bir varlık olasılığı üretmesi güçleşmektedir. Daha uzun bağlam pencereleri ya da geçici seslere özgü veri artırımı, bu sınıflarda duyarlılığı artırabilir.

Üçüncü sınırlılık, SI-SDR ölçütünün hedef-baskın karışımlardaki davranışıdır. Bu ölçüt, hedefin hâlihazırda izole olduğu örneklerde maskelemeyi cezalandırdığından, algısal kaliteyi tam olarak yansıtmamaktadır. Gelecekteki değerlendirmelerde, hedef baskınlığı yüksek örneklerin süzülmesi ya da algısal ölçütlerin (örneğin algısal değerlendirme tabanlı metrikler) eklenmesi, başarımın daha doğru ölçülmesini sağlayabilir.

Dördüncü sınırlılık, önerilen modelin on beş sınıflık düzenlenmiş bir sözcük dağarcığıyla sınırlı olmasıdır. Bu sınırlama, başarımı yüksek tutmak amacıyla bilinçli olarak yapılmış bir ödünleşimdir; ancak web uygulamasının yalnızca bu on beş sesi tespit edip çıkarabilmesi anlamına gelmektedir. Sözcük dağarcığının, her yeni sınıf için yeterli ve temiz veriyle dengeli biçimde genişletilmesi, ileride yapılması önerilen çalışmalar arasındadır. Son olarak, değerlendirmenin sentetik karışımlar üzerinde yapılması, gerçek dünyanın yankılı (reverberant) ve değişken kayıt koşullarıyla bir alan boşluğu (domain gap) oluşturmaktadır; gerçek kayıtlar üzerinde yapılan dinleme testleri olumlu olsa da, bu boşluğun nicel olarak ölçülmesi ek bir değerlendirme gerektirmektedir.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 5. SONUÇ VE ÖNERİLER

## 5.1 Genel Değerlendirme

Bu tez çalışmasında, bir ses ya da video kaydından seçilen ses sınıflarının, kaydın geri kalanına dokunulmadan çıkarılmasını sağlayan, derin öğrenme tabanlı bir seçici gürültü engelleme sistemi tasarlanmış, eğitilmiş ve uçtan uca bir web uygulamasıyla işlevsel kılınmıştır. Problem, çıkarılacak sınıfın modele dışarıdan bir tek-sıcak sorgu vektörüyle bildirildiği, sorgu-koşullu ve denetimli bir kaynak ayrıştırma görevi olarak biçimlendirilmiştir. Bu kurgu, sabit çok çıkışlı modellerin sınıf dengesizliği ve depolama yükü kısıtlarını yapısal olarak ortadan kaldırmaktadır.

Önerilen modelin çekirdeği, sınıf sorgusunun her kodlayıcı seviyesinde ve darboğazda ölçek ve öteleme parametrelerine dönüştürüldüğü, FiLM ile koşullandırılmış iki boyutlu bir U-Net mimarisidir. Eğitim verisi, bellek içi bir klip önbelleğinden anlık olarak sentezlenmiş; negatif örnekler, ağırlıklı zor-negatif örnekleme ve arka plan gürültüsü artırımı gibi tekniklerle modelin seçici bastırma ve gürültü dayanıklılığı kazanması sağlanmıştır. Sınıf varlığının kestirimi için, maske-enerjisi sezgiseli yerine öğrenilmiş bir tespit başı benimsenmiştir. Eğitim süreci; Adam optimize edicisi, karma hassasiyetli hesaplama, XLA derlemesi ve paralel veri hattı ile hızlandırılmıştır.

Sistem, bir dizi denetimli deney aracılığıyla iteratif bir süreçle geliştirilmiş; her aşamada gözlemlenen başarısızlık örüntüleri (agresif veri artırımının yol açtığı sessizlik çöküşü, eğitim-çıkarım ölçek uyumsuzluğu, az desteklenen sınıfların fantom yanlış pozitifleri ve odak kaybının gradyan çöküşü) çözümlenip bir sonraki tasarım kararına yansıtılmıştır. Bu metodoloji, önerilen modelde tespit makro $F_1$ değerinin $0{,}692$'ye ulaşmasını ve yanlış pozitif sayısının $34$ gibi düşük bir düzeyde tutulmasını sağlamıştır. Ayrıştırma başarımı, faz yeniden kullanımının dalga biçimi düzlemindeki yapısal kısıtına karşın, hedefin karışım içinde gömülü olduğu örneklerde pozitif SI-SDRi değerleriyle bir iyileştirme sağladığını göstermiştir. Çalışmanın başında belirlenen amaçlar — sınıf sayısından bağımsız bir koşullu ayrıştırma modeli tasarlanması, bu modelin eğitilip değerlendirilmesi ve uçtan uca bir uygulamada sunulması — karşılanmıştır.

## 5.2 Gelecek Çalışmalar İçin Öneriler

Elde edilen sonuçlar ve belirlenen sınırlılıklar doğrultusunda, gelecekte yapılması önerilen çalışmalar aşağıda sıralanmıştır.

**Faz-duyarlı ve dalga biçimi düzlemli ayrıştırma.** Mevcut sistemin faz yeniden kullanımından kaynaklanan dalga biçimi-düzeyli kısıtı, faz duyarlı maske ya da karmaşık oran maskesi türevleriyle ele alınabilir. Alternatif olarak, doğrudan dalga biçimi düzleminde çalışan Conv-TasNet ve Demucs benzeri mimariler, faz yeniden yapılandırma sınırını tümüyle ortadan kaldırabilir.

**Sözcük dağarcığının dengeli genişletilmesi.** Önerilen model, başarımı yüksek tutmak amacıyla on beş sınıfla sınırlandırılmıştır. Her yeni sınıf için yeterli ve temiz veriyle, sınıf dengesini koruyan bir genişletme; ayrıca aşırı-tetikleyen sınıflar için ağırlıklı zor-negatif eğitimin sistematik biçimde uygulanması, sözcük dağarcığını başarımı düşürmeden büyütebilir.

**Geçici sesler için bağlam ve artırım.** Kısa süreli geçici seslerin (hapşırık, öksürük) düşük tespit duyarlılığı, daha uzun bağlam pencereleri, geçici seslere özgü veri artırımı ya da çok ölçekli zamansal modelleme ile iyileştirilebilir.

**Algısal değerlendirme ölçütleri.** SI-SDR ölçütünün hedef-baskın karışımlardaki kısıtı göz önüne alınarak, algısal kaliteyi daha doğru yansıtan ölçütlerin ve öznel dinleme testlerinin değerlendirme protokolüne eklenmesi önerilmektedir.

**Gerçek dünya alan uyarlaması ve gerçek zamanlı çıkarım.** Sentetik karışımlar ile gerçek yankılı kayıtlar arasındaki alan boşluğunun, yankı ve oda darbe yanıtı artırımıyla daraltılması; ayrıca çıkarım hattının düşük gecikmeli, akış tabanlı (streaming) bir kurguya uyarlanması, sistemin canlı uygulamalarda kullanılabilirliğini artıracaktır.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# KAYNAKLAR

> Not: Kaynaklar, IEEE atıf biçiminde ve metinde ilk geçtikleri sıraya göre numaralandırılmıştır. Liste, bölümler yazıldıkça artımlı olarak güncellenmektedir; birleştirme aşamasında nihai numaralandırma ve sıralama denetlenecektir.

[1] E. C. Cherry, "Some experiments on the recognition of speech, with one and two ears," *The Journal of the Acoustical Society of America*, vol. 25, no. 5, pp. 975–979, 1953.

[2] D. Wang and G. J. Brown, *Computational Auditory Scene Analysis: Principles, Algorithms, and Applications*. Hoboken, NJ, USA: Wiley-IEEE Press, 2006.

[3] P. Comon, "Independent component analysis, a new concept?," *Signal Processing*, vol. 36, no. 3, pp. 287–314, 1994.

[4] E. Vincent, R. Gribonval, and C. Févotte, "Performance measurement in blind audio source separation," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 14, no. 4, pp. 1462–1469, 2006.

[5] S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 27, no. 2, pp. 113–120, 1979.

[6] Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 6, pp. 1109–1121, 1984.

[7] D. D. Lee and H. S. Seung, "Learning the parts of objects by non-negative matrix factorization," *Nature*, vol. 401, no. 6755, pp. 788–791, 1999.

[8] P. Smaragdis and J. C. Brown, "Non-negative matrix factorization for polyphonic music transcription," in *Proc. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*, New Paltz, NY, USA, 2003, pp. 177–180.

[9] D. Wang, "On ideal binary mask as the computational goal of auditory scene analysis," in *Speech Separation by Humans and Machines*, P. Divenyi, Ed. Boston, MA, USA: Springer, 2005, pp. 181–197.

[10] A. Narayanan and D. Wang, "Ideal ratio mask estimation using deep neural networks for robust speech recognition," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Vancouver, Canada, 2013, pp. 7092–7096.

[11] H. Erdogan, J. R. Hershey, S. Watanabe, and J. Le Roux, "Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Brisbane, Australia, 2015, pp. 708–712.

[12] D. S. Williamson, Y. Wang, and D. Wang, "Complex ratio masking for monaural speech separation," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 24, no. 3, pp. 483–492, 2016.

[13] P.-S. Huang, M. Kim, M. Hasegawa-Johnson, and P. Smaragdis, "Deep learning for monaural speech separation," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Florence, Italy, 2014, pp. 1562–1566.

[14] J. R. Hershey, Z. Chen, J. Le Roux, and S. Watanabe, "Deep clustering: Discriminative embeddings for segmentation and separation," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Shanghai, China, 2016, pp. 31–35.

[15] D. Yu, M. Kolbæk, Z.-H. Tan, and J. Jensen, "Permutation invariant training of deep models for speaker-independent multi-talker speech separation," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, New Orleans, LA, USA, 2017, pp. 241–245.

[16] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Munich, Germany, 2015, pp. 234–241.

[17] A. Jansson, E. J. Humphrey, N. Montecchio, R. M. Bittner, A. Kumar, and T. Weyde, "Singing voice separation with deep U-Net convolutional networks," in *Proc. International Society for Music Information Retrieval Conference (ISMIR)*, Suzhou, China, 2017, pp. 745–751.

[18] Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing ideal time–frequency magnitude masking for speech separation," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 27, no. 8, pp. 1256–1266, 2019.

[19] A. Défossez, N. Usunier, L. Bottou, and F. Bach, "Music source separation in the waveform domain," arXiv preprint arXiv:1911.13254, 2019.

[20] I. Kavalerov, S. Wisdom, H. Erdogan, B. Patton, K. Wilson, J. Le Roux, and J. R. Hershey, "Universal sound separation," in *Proc. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*, New Paltz, NY, USA, 2019, pp. 175–179.

[21] K. Žmolíková, M. Delcroix, K. Kinoshita, T. Ochiai, T. Nakatani, L. Burget, and J. Černocký, "SpeakerBeam: Speaker aware neural network for target speaker extraction in speech mixtures," *IEEE Journal of Selected Topics in Signal Processing*, vol. 13, no. 4, pp. 800–814, 2019.

[22] Q. Wang, H. Muckenhirn, K. Wilson, P. Sridhar, Z. Wu, J. R. Hershey, R. A. Saurous, R. J. Weiss, Y. Jia, and I. L. Moreno, "VoiceFilter: Targeted voice separation by speaker-conditioned spectrogram masking," in *Proc. Interspeech*, Graz, Austria, 2019, pp. 2728–2732.

[23] T. Ochiai, M. Delcroix, Y. Koizumi, H. Ito, K. Kinoshita, and S. Araki, "Listen to what you want: Neural network-based universal sound selector," in *Proc. Interspeech*, Shanghai, China, 2020, pp. 1441–1445.

[24] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a general conditioning layer," in *Proc. AAAI Conference on Artificial Intelligence*, New Orleans, LA, USA, 2018, pp. 3942–3951.

[25] V. Dumoulin, J. Shlens, and M. Kudlur, "A learned representation for artistic style," in *Proc. International Conference on Learning Representations (ICLR)*, Toulon, France, 2017.

[26] H. de Vries, F. Strub, J. Mary, H. Larochelle, O. Pietquin, and A. Courville, "Modulating early visual processing by language," in *Advances in Neural Information Processing Systems (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 6594–6604.

[27] G. Meseguer-Brocal and G. Peeters, "Conditioned-U-Net: Introducing a control mechanism in the U-Net for multiple source separations," in *Proc. International Society for Music Information Retrieval Conference (ISMIR)*, Delft, The Netherlands, 2019, pp. 159–165.

[28] A. Mesaros, T. Heittola, T. Virtanen, and M. D. Plumbley, "Sound event detection: A tutorial," *IEEE Signal Processing Magazine*, vol. 38, no. 5, pp. 67–83, 2021.

[29] E. Cakır, G. Parascandolo, T. Heittola, H. Huttunen, and T. Virtanen, "Convolutional recurrent neural networks for polyphonic sound event detection," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 25, no. 6, pp. 1291–1303, 2017.

[30] J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, "Audio Set: An ontology and human-labeled dataset for audio events," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, New Orleans, LA, USA, 2017, pp. 776–780.

[31] K. J. Piczak, "ESC: Dataset for environmental sound classification," in *Proc. ACM International Conference on Multimedia (ACM MM)*, Brisbane, Australia, 2015, pp. 1015–1018.

[32] J. Salamon, C. Jacoby, and J. P. Bello, "A dataset and taxonomy for urban sound research," in *Proc. ACM International Conference on Multimedia (ACM MM)*, Orlando, FL, USA, 2014, pp. 1041–1044.

[33] E. Fonseca, X. Favory, J. Pons, F. Font, and X. Serra, "FSD50K: An open dataset of human-labeled sound events," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 30, pp. 829–852, 2022.

[34] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. International Conference on Machine Learning (ICML)*, Lille, France, 2015, pp. 448–456.

[35] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. International Conference on Learning Representations (ICLR)*, San Diego, CA, USA, 2015.

[36] P. Micikevicius, S. Narang, J. Alben, G. Diamos, E. Elsen, D. García, B. Ginsburg, M. Houston, O. Kuchaiev, G. Venkatesh, and H. Wu, "Mixed precision training," in *Proc. International Conference on Learning Representations (ICLR)*, Vancouver, Canada, 2018.

[37] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard, M. Kudlur, J. Levenberg, R. Monga, S. Moore, D. G. Murray, B. Steiner, P. Tucker, V. Vasudevan, P. Warden, M. Wicke, Y. Yu, and X. Zheng, "TensorFlow: A system for large-scale machine learning," in *Proc. USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, Savannah, GA, USA, 2016, pp. 265–283.

[38] D. Griffin and J. Lim, "Signal estimation from modified short-time Fourier transform," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 2, pp. 236–243, 1984.

[39] J. Le Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, "SDR – half-baked or well done?," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Brighton, U.K., 2019, pp. 626–630.