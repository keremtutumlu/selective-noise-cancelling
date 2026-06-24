<!--
ÖN BÖLÜMLER — Birleştirme aşamasında tezin en başına yerleştirilecektir.
⟨...⟩ ile gösterilen alanlar kişiye/teze özeldir; doldurulması gerekmektedir.
Şekil ve tablo görselleri ../thesis_figures/ dizinindedir.
-->

# MARMARA ÜNİVERSİTESİ
# TEKNOLOJİ FAKÜLTESİ
# BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ

<br>

## DERİN ÖĞRENME İLE SEÇİCİ GÜRÜLTÜ ENGELLEME

<br>

### BİTİRME PROJESİ

<br>

⟨ÖĞRENCİ ADI SOYADI⟩
⟨ÖĞRENCİ NUMARASI⟩

<br>

**DANIŞMAN**
⟨Unvan Ad Soyad⟩

<br>

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

<div align="right">

⟨Öğrenci Adı Soyadı⟩
Haziran, 2026

</div>

---

## İÇİNDEKİLER

- ÖZET
- ABSTRACT
- ŞEKİL LİSTESİ
- TABLO LİSTESİ
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

<div align="right">

⟨Öğrenci Adı Soyadı⟩
Haziran, 2026

</div>

---

## ABSTRACT

### SELECTIVE NOISE CANCELLATION WITH DEEP LEARNING

This study presents a deep-learning-based selective noise cancellation system that removes chosen sound classes from an audio or video recording while leaving the rest of the content untouched. The problem is formulated as a query-conditioned, supervised source-separation task in which the class to be extracted is signalled to the model through a one-hot query vector. The proposed model is a two-dimensional U-Net conditioned with FiLM, where the class query is transformed into scale and shift parameters at every encoder level and at the bottleneck. The model operates on log-magnitude spectrograms of one-second windows at a $16$ kHz sampling rate and produces a soft mask for the queried class. Training data is synthesised on the fly from an in-memory clip cache; negative examples, weighted hard-negative sampling and background-noise augmentation enable the model to learn selective suppression and noise robustness. A learned detection head is used to estimate class presence, and the training process is accelerated with the Adam optimiser, mixed-precision computation and XLA compilation. The system was developed through an iterative experimental methodology and reached a detection macro $F_1$ score of $0.692$ on a curated vocabulary of the fifteen best-separated classes. The end-to-end web application detects the classes present in an uploaded file and removes the selected classes via an overlap-add procedure.

**Keywords:** Selective noise cancellation, source separation, query-conditioned learning, FiLM, U-Net, deep learning, sound event detection.

<div align="right">

⟨Student Name⟩
June, 2026

</div>

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
| Tablo 3.1 | Önerilen modelin eğitim hiperparametreleri |
| Tablo 4.1 | Denenen tasarım değişiklikleri ve gözlemlenen sonuçlar |
| Tablo 4.2 | Sınıf bazlı tespit ve ayrıştırma başarımı |

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
