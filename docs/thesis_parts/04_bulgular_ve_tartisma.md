# 4. BULGULAR VE TARTIŞMA

Bu bölümde, önerilen sistemin başarımı niceliksel ve niteliksel olarak değerlendirilmiştir. Önce değerlendirme metrikleri tanımlanmış; ardından modelin v1.0'dan v3.0'a uzanan sürüm evrimi, her sürümde gözlemlenen başarısızlık örüntüleri ve bunlara getirilen çözümlerle birlikte çözümlenmiştir. Sonraki başlıklarda ayrıştırma başarımı, tespit başarımı, FiLM koşullandırmasının katkısı, eşik taraması ve niteliksel sonuçlar sunulmuş; bölüm, sistemin sınırlılıklarının tartışılmasıyla tamamlanmıştır. Niceliksel sonuçlar, modelin son sürümü (v3.0) için iki yüz sentetik karışımdan oluşan bir test kümesi üzerinde elde edilmiştir.

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

### 4.1.2 Tespit Metrikleri

Tespit başarımı, kesinlik (precision), duyarlılık (recall) ve bunların harmonik ortalaması olan $F_1$ ölçütüyle değerlendirilmiştir. Bir sınıf için doğru pozitif (DP), yanlış pozitif (YP) ve yanlış negatif (YN) sayıları üzerinden

$$\text{Kesinlik} = \frac{\text{DP}}{\text{DP} + \text{YP}}, \qquad \text{Duyarlılık} = \frac{\text{DP}}{\text{DP} + \text{YN}},$$

$$F_1 = \frac{2 \cdot \text{Kesinlik} \cdot \text{Duyarlılık}}{\text{Kesinlik} + \text{Duyarlılık}}$$

tanımları kullanılmaktadır. Genel başarım, sınıf bazlı $F_1$ değerlerinin ortalaması olan makro $F_1$ ile özetlenmektedir. Makro ortalama, her sınıfa eşit ağırlık verdiğinden, sınıf dengesizliğinden bağımsız bir başarım göstergesi sağlamaktadır. Her sentetik karışım için bileşen sınıfları bilindiğinden, tespit edilen sınıflar bu yer-gerçek (ground-truth) kümesiyle karşılaştırılarak DP, YP ve YN sayıları biriktirilmektedir.

## 4.2 Model Sürüm Evrimi

Önerilen model, tek bir eğitim turunda değil, her sürümde gözlemlenen başarısızlık örüntülerinin çözümlenip bir sonraki sürümün tasarımına yansıtıldığı iteratif bir süreçle geliştirilmiştir. Bu sürüm evrimi, hem nihai tasarım kararlarının gerekçelerini belgelemekte hem de derin öğrenme tabanlı bir ayrıştırma sisteminin geliştirilmesinde karşılaşılan tipik tuzakları ortaya koymaktadır. Sürümlerin başlıca değişiklikleri ve sonuçları Tablo 4.1'de özetlenmiştir.

**Tablo 4.1:** Model sürümlerinin evrimi, başlıca değişiklikler ve sonuçlar.

| Sürüm | Başlıca değişiklik | Sonuç |
|---|---|---|
| v1.0 | FiLM-koşullu U-Net + anlık karışım temel modeli (ESC-50) | Çalışan temel model |
| v2.0 | Agresif veri artırımı ($P_{\text{negatif}}=0{,}45$; SNR 5–20 dB) | Sessizliğe çöküş (bozuk) |
| v2.1 | Artırma düzeltmesi + çıkarım normalizasyonu | $F_1=0{,}21$; SI-SDRi $-22{,}18$ dB |
| v2.2 | Tam-kodlayıcı FiLM + çok çözünürlüklü L1 + %75 OLA | $F_1=0{,}13$ (FSD50K sıfır klip hatası) |
| v2.3 | $P_{\text{negatif}}=0{,}15$; FSD50K eklendi (235 sınıf) | $F_1=0{,}02$ (fantom yanlış pozitifler) |
| v2.4 | Minimum klip tabanı 40 + tespit izin listesi | $F_1=0{,}09$; çalışma noktası cap$=0{,}80$, $k=5$ |
| v2.6 | Öğrenilmiş tespit başı + 10 aşırı-tetikleyen sınıf | $F_1=0{,}17$ (baş yetersiz uyum) |
| v2.7 | $P_{\text{negatif}}=0{,}50$; büyük baş; odak kaybı | Gradyan çöküşü (başarısız) |
| v2.8 | BCE tespit + FSD50K kaldırıldı (56 sınıf) | $F_1=0{,}32$ |
| v3.0 | Düzenlenmiş 15 sınıflı sözcük dağarcığı | $F_1=0{,}692$; SI-SDRi $-13{,}07$ dB |

Temel model (v1.0), FiLM-koşullu U-Net mimarisini ve anlık karışım hattını kurmuş; tespit ve çıkarma işlevleri bu aşamada çalışır durumda olmuştur. İkinci sürüm (v2.0), veri artırımının agresif biçimde artırılmasıyla bütünüyle işlevsiz hâle gelmiştir: negatif örnek olasılığının $0{,}45$'e ve gürültü düzeyinin $5$ dB SNR'a çıkarılması, modeli her sorgu için yakın-sıfır maske üreten "güvenli sessizlik" dengesine itmiştir. Ayrıca çıkarım hattının ham ses beslemesi, eğitim-çıkarım ölçek uyumsuzluğunu açığa çıkarmıştır. Bu başarısızlık, üçüncü bölümde açıklanan negatif örnek oranı (Alt Başlık 3.4.2) ve tepe normalizasyonu (Alt Başlık 3.4.5) tasarım kararlarının doğrudan gerekçesini oluşturmaktadır.

İzleyen sürümler (v2.1–v2.2), artırma parametrelerini ölçülü değerlere çekmiş, çıkarım normalizasyonunu düzeltmiş, FiLM koşullandırmasını tüm kodlayıcı seviyelerine yaymış, çok çözünürlüklü L1 kaybını eklemiş ve örtüşme oranını $\%75$'e çıkararak sınır darbesi yapaylığını gidermiştir. FSD50K veri kümesinin eklenmesi (v2.3), sözcük dağarcığını $235$ sınıfa genişletmiş; ancak yerel sesle desteklenmeyen sınıfların fantom yanlış pozitifler üretmesiyle makro $F_1$ değeri $0{,}02$'ye gerilemiştir. Bu çöküş, minimum klip tabanının (v2.4) ve sonunda FSD50K'nin bütünüyle kaldırılmasının (v2.8) gerekçesini oluşturmuştur.

Tespit probleminin maske-enerjisi sezgiseliyle yapısal olarak çözülemeyeceğinin anlaşılması üzerine, öğrenilmiş bir tespit başı eklenmiştir (v2.6). Bu başın odak kaybıyla eğitilmesi (v2.7), Alt Başlık 3.6.2'de çözümlenen gradyan çöküşüne yol açmış; ikili çapraz entropiye dönülmesi ve FSD50K'nin kaldırılmasıyla (v2.8) makro $F_1$ değeri $0{,}32$'ye yükselmiştir. Bu sürümün sınıf bazlı sonuçlarının belirgin biçimde iki kutuplu olması — bir grup sınıfın yüksek, geniş bantlı bir grup sınıfın ise sıfıra yakın $F_1$ üretmesi — son sürümün (v3.0) tasarım ilkesini belirlemiştir: ayrıştırma ve tespit başarımının en yüksek olduğu on beş sınıfın korunması. Bu düzenleme, hem makro ortalamayı yalnızca başarılı sınıflar üzerinden hesaplatmış hem de aşırı-tetikleyen geniş bantlı sınıfları aday havuzundan çıkararak göreli kesme eşiğinin gerçek sınıfları bastırmasını önlemiştir. Sonuçta makro $F_1$ değeri $0{,}692$'ye ulaşmıştır.