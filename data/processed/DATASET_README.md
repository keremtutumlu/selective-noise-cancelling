
## Genel Bakış (Overview)
Bu veri seti, Seçici Aktif Gürültü Engelleme (SNC) sisteminin gerçek dünya koşullarında (üst üste binen çoklu sesler) çalışabilmesi için Derin Öğrenme modellerini eğitmek amacıyla sentetik olarak üretilmiş, çoklu etiketli (multi-label) ses özelliklerini içerir.

## Dosya Yapısı (Files)
- **`X_multi_features.npy`**: Giriş özellik matrisi (Z-score normalize edilmiş Log-Mel Spektrogramları).
- **`y_multi_labels.npy`**: Hedef "Multi-Hot" kodlanmış etiket matrisi.

## Veri Boyutları (Data Shapes)
- **X Matrisi Boyutu**: `(Örnek_Sayısı, 64, 101, 3)`
  - `64`: Mel frekans filtre sayısı.
  - `101`: Zaman adımları (16 kHz frekansında tam olarak 1.0 saniyelik sesi temsil eder).
  - `3`: Renk kanalı (MobileNetV2 giriş katmanına uyum sağlamak için tek kanallı spektrogram RGB olarak çoklanmıştır).
- **y Matrisi Boyutu**: `(Örnek_Sayısı, 8)`
  - Çoklu etiket (Multi-hot) vektörü (Örn: `[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]`).

## Hedef Sınıflar (Sıralı İndeksler)
Modelin tahmin tutarlılığını sağlamak için sınıflar alfabetik sıraya göre indekslenmiştir:
0. `car_horn` (Korna)
1. `crying_baby` (Bebek Ağlaması)
2. `dog` (Köpek)
3. `engine` (Motor)
4. `keyboard_typing` (Klavye Sesi)
5. `rain` (Yağmur)
6. `siren` (Siren)
7. `wind` (Rüzgar)

## Ön İşleme ve Veri Zenginleştirme (Preprocessing & Augmentation)
- **Örnekleme Hızı (Sample Rate)**: Tüm sesler gömülü sistem performansını artırmak için 16,000 Hz'e indirgenmiştir.
- **Pencere Boyutu (Window Length)**: Gerçek zamanlı Edge AI gecikme sınırlarını aşmamak için tüm sesler rastgele 1 saniyelik (16,000 sample) kesitler halinde işlenmiştir.
- **Süperpozisyon (Mixing)**: Doğadaki kaosu simüle etmek için her örnek, rastgele seçilmiş 1 ile 3 farklı sınıfın üst üste bindirilmesiyle (mix) oluşturulur.
- **Sinyal-Gürültü Oranı Değişkenliği (SNR Variance)**: Modelin ezberlemesini önlemek için, birleştirilen seslere (0.4 ile 1.0 arası) rastgele genlik (ses seviyesi) çarpanları uygulanmıştır.
- **Normalizasyon**: Her spektrograma Z-Score (Standartlaştırma) uygulanmıştır.

## Kullanım Amacı (Intended Use)
Bu veri seti, çıkış katmanında `Sigmoid` aktivasyon fonksiyonu ve kayıp fonksiyonu olarak `Binary Cross-Entropy` kullanan çoklu etiketli (Multi-Label) bir modelin eğitimi için özel olarak üretilmiştir.