# 5. SONUÇ VE ÖNERİLER

## 5.1 Genel Değerlendirme

Bu tez çalışmasında, bir ses ya da video kaydından seçilen ses sınıflarının, kaydın geri kalanına dokunulmadan çıkarılmasını sağlayan, derin öğrenme tabanlı bir seçici gürültü engelleme sistemi tasarlanmış, eğitilmiş ve uçtan uca bir web uygulamasıyla işlevsel kılınmıştır. Problem, çıkarılacak sınıfın modele dışarıdan bir tek-sıcak sorgu vektörüyle bildirildiği, sorgu-koşullu ve denetimli bir kaynak ayrıştırma görevi olarak biçimlendirilmiştir. Bu kurgu, sabit çok çıkışlı modellerin sınıf dengesizliği ve depolama yükü kısıtlarını yapısal olarak ortadan kaldırmaktadır.

Önerilen modelin çekirdeği, sınıf sorgusunun her kodlayıcı seviyesinde ve darboğazda ölçek ve öteleme parametrelerine dönüştürüldüğü, FiLM ile koşullandırılmış iki boyutlu bir U-Net mimarisidir. Eğitim verisi, bellek içi bir klip önbelleğinden anlık olarak sentezlenmiş; negatif örnekler, ağırlıklı zor-negatif örnekleme ve arka plan gürültüsü artırımı gibi tekniklerle modelin seçici bastırma ve gürültü dayanıklılığı kazanması sağlanmıştır. Sınıf varlığının kestirimi için, maske-enerjisi sezgiseli yerine öğrenilmiş bir tespit başı benimsenmiştir. Eğitim süreci; Adam optimize edicisi, karma hassasiyetli hesaplama, XLA derlemesi ve paralel veri hattı ile hızlandırılmıştır.

Sistem, v1.0'dan v3.0'a uzanan iteratif bir süreçle geliştirilmiş; her sürümde gözlemlenen başarısızlık örüntüleri (agresif veri artırımının yol açtığı sessizlik çöküşü, eğitim-çıkarım ölçek uyumsuzluğu, az desteklenen sınıfların fantom yanlış pozitifleri ve odak kaybının gradyan çöküşü) çözümlenip bir sonraki sürümün tasarımına yansıtılmıştır. Bu metodoloji, modelin son sürümünde tespit makro $F_1$ değerinin $0{,}692$'ye ulaşmasını ve yanlış pozitif sayısının $34$ gibi düşük bir düzeyde tutulmasını sağlamıştır. Ayrıştırma başarımı, faz yeniden kullanımının dalga biçimi düzlemindeki yapısal kısıtına karşın, hedefin karışım içinde gömülü olduğu örneklerde pozitif SI-SDRi değerleriyle bir iyileştirme sağladığını göstermiştir. Çalışmanın başında belirlenen amaçlar — sınıf sayısından bağımsız bir koşullu ayrıştırma modeli tasarlanması, bu modelin eğitilip değerlendirilmesi ve uçtan uca bir uygulamada sunulması — karşılanmıştır.

## 5.2 Gelecek Çalışmalar İçin Öneriler

Elde edilen sonuçlar ve belirlenen sınırlılıklar doğrultusunda, gelecekte yapılması önerilen çalışmalar aşağıda sıralanmıştır.

**Faz-duyarlı ve dalga biçimi düzlemli ayrıştırma.** Mevcut sistemin faz yeniden kullanımından kaynaklanan dalga biçimi-düzeyli kısıtı, faz duyarlı maske ya da karmaşık oran maskesi türevleriyle ele alınabilir. Alternatif olarak, doğrudan dalga biçimi düzleminde çalışan Conv-TasNet ve Demucs benzeri mimariler, faz yeniden yapılandırma sınırını tümüyle ortadan kaldırabilir.

**Sözcük dağarcığının dengeli genişletilmesi.** Modelin son sürümü, başarımı yüksek tutmak amacıyla on beş sınıfla sınırlandırılmıştır. Her yeni sınıf için yeterli ve temiz veriyle, sınıf dengesini koruyan bir genişletme; ayrıca aşırı-tetikleyen sınıflar için ağırlıklı zor-negatif eğitimin sistematik biçimde uygulanması, sözcük dağarcığını başarımı düşürmeden büyütebilir.

**Geçici sesler için bağlam ve artırım.** Kısa süreli geçici seslerin (hapşırık, öksürük) düşük tespit duyarlılığı, daha uzun bağlam pencereleri, geçici seslere özgü veri artırımı ya da çok ölçekli zamansal modelleme ile iyileştirilebilir.

**Algısal değerlendirme ölçütleri.** SI-SDR ölçütünün hedef-baskın karışımlardaki kısıtı göz önüne alınarak, algısal kaliteyi daha doğru yansıtan ölçütlerin ve öznel dinleme testlerinin değerlendirme protokolüne eklenmesi önerilmektedir.

**Gerçek dünya alan uyarlaması ve gerçek zamanlı çıkarım.** Sentetik karışımlar ile gerçek yankılı kayıtlar arasındaki alan boşluğunun, yankı ve oda darbe yanıtı artırımıyla daraltılması; ayrıca çıkarım hattının düşük gecikmeli, akış tabanlı (streaming) bir kurguya uyarlanması, sistemin canlı uygulamalarda kullanılabilirliğini artıracaktır.
