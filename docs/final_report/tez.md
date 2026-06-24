# DERİN ÖĞRENME İLE SEÇİCİ GÜRÜLTÜ ENGELLEME

**Marmara Üniversitesi — Teknoloji Fakültesi — Bilgisayar Mühendisliği Bölümü**
**Bitirme Projesi**

> Not: Ön bölümler (Dış/İç Kapak, Jüri Onay Sayfası, Önsöz, İçindekiler, Özet/Abstract, Şekil ve Tablo Listeleri, Simgeler ve Kısaltmalar) tezin gövdesi tamamlandıktan sonra son geçişte üretilecektir. Bu dosya, onaylanan taslağa göre 1. Bölümden itibaren ilerlemektedir.

---

## 1. GİRİŞ

Akustik ortamlar, eş zamanlı olarak etkin olan çok sayıda ses kaynağının doğrusal biçimde üst üste binmesiyle oluşan bileşik sinyaller barındırır. Bir video kaydının arka planında çalan siren, bir telekonferans oturumuna karışan klavye tıkırtısı ya da bir saha kaydının üzerine binen helikopter uğultusu; hedef içeriğin anlaşılırlığını düşüren, yapısal ve geniş bantlı bozucu bileşenler olarak gözlemlenmektedir. Söz konusu bileşenlerin, kaydın geri kalan içeriğine dokunulmaksızın ayıklanması, klasik sinyal işleme çerçevelerinin tek başına çözmekte yetersiz kaldığı koşullu bir kaynak ayrıştırma problemi olarak ele alınmıştır. Bu bölümde problemin biçimsel tanımı ve mühendislik motivasyonu ortaya konmuş, seçici gürültü engellemenin sorgu-koşullu ayrıştırma olarak modellenişi açıklanmış, projenin amacı, kapsamı ve özgün katkıları belirtilmiş ve tezin bölüm organizasyonu sunulmuştur.

### 1.1 Problemin Tanımı ve Motivasyonu

Tek kanallı bir kayıtta gözlemlenen $x(t)$ karışım sinyali, izole kaynak imzaları $s_i(t)$ cinsinden toplamsal bir model ile ifade edilmektedir:

$$x(t) = \sum_{i=1}^{K} s_i(t),$$

burada $K$, o an etkin olan ses kaynağı sayısını göstermektedir. Bu modelde bozucu kaynaklar ile korunması istenen kaynaklar, aynı genlik mertebesinde ve örtüşen zaman-frekans bölgelerinde bulunabildiğinden, ayrıştırma probleminin kötü konumlanmış (ill-posed) doğası belirginleşmektedir.

Geleneksel gürültü engelleme yöntemleri, bozucu bileşeni durağan (stationary) ya da yavaş değişen istatistiksel bir artık olarak modelleme varsayımına dayanmaktadır. Aktif gürültü engelleme (ANC) donanımları, bozucu dalga biçiminin ters fazlı bir kopyasını üreterek akustik alanda yıkıcı girişim oluşturur; spektral çıkarma ve Wiener süzgeci gibi tekil kanallı yöntemler ise gürültü güç spektrumunu kestirip karışım spektrumundan çıkarır. Bu yaklaşımların ortak kısıtı, "gürültü" kavramının içeriğinden bağımsız bir taban düzeyi olarak ele alınmasıdır. Dolayısıyla, konuşma kadar yapısal ve geniş bantlı bir bozucu sesin (örneğin köpek havlaması veya siren) hedef sinyalden ayrıştırılması, bu çerçevelerin varsayımları dışında kalmaktadır. Durağanlık varsayımı geçersizleştiğinde, spektral çıkarma yöntemleri "müzikal gürültü" olarak adlandırılan yapaylıklar üretmekte ve hedef sinyalin spektral içeriğine zarar vermektedir.

Pratik gereksinim ise farklı bir formülasyon gerektirmektedir: bozucu bileşenin yalnızca bir taban düzeyi olarak değil, belirli bir *ses sınıfı* olarak tanımlanması ve bu sınıfın karışımdan seçici biçimde bastırılması beklenmektedir. Video sonrası prodüksiyonunda belirli bir çevresel sesin temizlenmesi, işitme destek sistemlerinde rahatsız edici sınıfların zayıflatılması ve telekonferans uygulamalarında ortam seslerinin ayıklanması, bu seçici bastırma yeteneğine duyulan ihtiyacı somutlaştıran kullanım durumlarıdır. Bu çalışma kapsamında problem, "karışımdan hangi sınıfın çıkarılacağı" bilgisinin modele dışarıdan verildiği, koşullu ve denetimli bir kaynak ayrıştırma görevi olarak biçimlendirilmiştir.

### 1.2 Seçici Gürültü Engelleme ve Sorgu-Koşullu Ayrıştırma

Toplamsal karışım modelinin zaman-frekans düzlemindeki karşılığı, Kısa Zamanlı Fourier Dönüşümü (STFT) altında yine toplamsaldır:

$$X(f, \tau) = \sum_{i=1}^{K} S_i(f, \tau),$$

burada $f$ frekans bini, $\tau$ ise zaman çerçevesi indisini göstermektedir. Maskeleme tabanlı ayrıştırma, hedef kaynağın genlik spektrogramını, $[0, 1]$ aralığında değer alan bir yumuşak maske $M_c(f, \tau)$ ile karışım genliğinin çarpımı olarak kestirir:

$$\hat{S}_c(f, \tau) = M_c(f, \tau) \odot |X(f, \tau)|.$$

Bu formülasyon üzerine iki ayrı mimari kurgu inşa edilebilmektedir. Birinci kurgu, sabit çok çıkışlı (multi-output) bir modeldir: ağ, her sınıf için ayrı bir maske kanalı üreterek tek geçişte tüm kaynakları kestirir. Bu kurgunun üç temel kısıtı bulunmaktadır. Öncelikle, çıkış katmanının genişliği sınıf sayısı ile doğru orantılı olarak büyümekte, dolayısıyla yeni bir veri kümesi eklenmesi mimarinin yeniden tasarlanmasını gerektirmektedir. İkinci olarak, ağır bir sınıf dengesizliği ortaya çıkmaktadır: $K$ kaynaklı bir karışımda toplam sınıf sayısı $N$ ise, $N - K$ kanalın hedefi sessizliktir; elli sınıflı bir sözcük dağarcığında çoğu eğitim örneği için kanalların büyük bölümü yalnızca sessizliği öngörmeye zorlanmakta, bu da öğrenme sinyalini seyreltmektedir. Üçüncü olarak, önceden üretilmiş hedef stem dosyalarının diskte saklanması, elli sınıf ve saniyede $16\,000$ örnek için yaklaşık $19$ GB mertebesinde bir depolama yükü doğurmaktadır.

İkinci kurgu, bu çalışmada benimsenen sorgu-koşullu (query-conditioned) yaklaşımdır. Bu yaklaşımda ağa, $(256, 128, 1)$ boyutlu logaritmik genlik spektrogramının yanı sıra, çıkarılması istenen sınıfı seçen bir tek-sıcak (one-hot) sorgu vektörü $q \in \{0, 1\}^{N}$ verilmektedir. Ağ, yalnızca sorgulanan sınıf için tek bir maske üretir. Bu tasarımda sınıf sayısı yalnızca sorgu girişinin genişliğini etkilemekte; evrişimli gövde ise sınıf sayısından bağımsız kalmaktadır. Böylece aynı mimari, sekiz sınıflı bir kümeden yüzlerce sınıflı bir sözcük dağarcığına kadar yeniden eğitime gerek kalmadan ölçeklenebilmektedir. Birden çok sesin çıkarılması gerektiğinde, model her hedef sınıf için bir kez sorgulanmaktadır. Bu kurgu, hem sınıf dengesizliği sorununu hem de depolama yükünü yapısal olarak ortadan kaldırmaktadır; çünkü eğitim örnekleri bir veri dosyasından okunmak yerine bellek içi bir klip önbelleğinden anlık olarak sentezlenmektedir.

### 1.3 Projenin Amacı ve Kapsamı

Bu projenin amacı, sorgulanan bir ses sınıfı için yumuşak spektrogram maskesi üreten, FiLM (Feature-wise Linear Modulation) ile koşullandırılmış iki boyutlu bir U-Net mimarisi tasarlamak, eğitmek ve bu modeli uçtan uca bir gürültü temizleme uygulamasında işlevsel kılmaktır. Tüm ses verileri ortak bir spektrogram sözleşmesine indirgenmiştir: sinyaller $16$ kHz örnekleme hızında ve tek kanala (mono) dönüştürülmüş, STFT parametreleri $n_{\text{fft}} = 512$ ve sıçrama uzunluğu $hop = 128$ olarak belirlenmiştir. Nyquist bini düşürülerek modele $256$ frekans bini sunulmuş, zaman ekseni bir saniyelik pencereye karşılık gelen $128$ çerçeveye sabitlenmiştir. Buna göre her model çağrısı yaklaşık bir saniyelik bir akustik bağlamı işlemektedir.

Sistemin işlevsel kapsamı, dört aşamalı bir çıkarım zinciri ile tanımlanmıştır: kullanıcının ses veya video dosyası yüklemesi, dosyada bulunan ses sınıflarının tespit edilmesi, kullanıcının çıkarılmasını istediği sınıfları işaretlemesi ve temizlenmiş çıktının üretilip indirilmesi. Video girişlerinde, temizlenen ses izi `ffmpeg` aracılığıyla özgün görüntü izinin üzerine yeniden bindirilmektedir. Modelin son sürümü (v3.0), ayrıştırma ve tespit başarımının en yüksek olduğu on beş sınıftan oluşan, düzenlenmiş (curated) bir sözcük dağarcığı üzerinde çalışmaktadır.

Çalışmanın kapsamı, genlik spektrogramı üzerinde maskeleme ve karışım fazının yeniden kullanımı ilkesiyle sınırlandırılmıştır. Faz bilgisi modelce kestirilmediğinden, yeniden sentez aşamasında karışımın özgün fazı korunmaktadır; bu seçim, sayısal kararlılık ve hesaplama maliyeti açısından bir mühendislik ödünleşimi olarak benimsenmiştir. Dalga biçimi düzleminde uçtan uca ayrıştırma yapan modeller ile faz-duyarlı kestirim teknikleri kapsam dışında bırakılmış; bu yöntemler ileride yapılması önerilen çalışmalar arasında değerlendirilmiştir.

### 1.4 Bilimsel Katkı ve Özgün Değer

Bu tez çalışması kapsamında üretilen özgün katkılar aşağıda sıralanmıştır.

**Sınıf sayısından bağımsız mimari.** Sorgu-koşullu tasarım sayesinde evrişimli gövde sabit tutulmuş, sınıf sayısı yalnızca sorgu vektörünün boyutunu belirleyen bir parametreye indirgenmiştir. Böylece veri kümesi genişletildiğinde mimari değişmeden kalmaktadır.

**Anlık karışım üretimi.** Eğitim örnekleri, önceden üretilmiş bir veri dosyasından okunmak yerine, bellek içi klip önbelleğinden her adımda rastgele sentezlenmektedir. Bu yaklaşım hem depolama yükünü ortadan kaldırmakta hem de görece sınırsız bir karışım çeşitliliği sağlamaktadır.

**Çok seviyeli FiLM koşullandırması.** Sınıf sorgusu, paylaşılan bir gömme katmanından geçirilip her kodlayıcı seviyesinde ve darboğazda ayrı $\gamma$ (ölçek) ve $\beta$ (öteleme) parametrelerine dönüştürülmektedir. Koşullandırmanın yalnızca darboğazda değil tüm kodlayıcı seviyelerinde uygulanması, atlama bağlantılarının da sınıfa özgü etkinlikler taşımasını sağlayarak maske kesinliğini artırmaktadır.

**Öğrenilmiş tespit başı.** Sınıf varlığının maske enerjisine dayalı bir sezgisel ile kestirilmesi yerine, FiLM ile koşullandırılmış darboğaz üzerinden $P(\text{sorgulanan sınıf mevcut} \mid \text{karışım})$ olasılığını üreten hafif bir sınıflandırma başı eğitilmiştir. Bu baş, geniş bantlı sınıfların yarattığı yapısal yanlış pozitif sorununu hafifletmektedir.

**Sürüm tabanlı deneysel metodoloji.** Model, v1.0 sürümünden v3.0 sürümüne kadar iteratif bir biçimde geliştirilmiş; her sürümde gözlemlenen başarısızlık örüntüleri (aşırı veri artırımının yol açtığı eğitim çöküşü, FSD50K kaynaklı fantom yanlış pozitifler ve odak kaybının yol açtığı gradyan çöküşü) çözümlenip bir sonraki sürümün tasarımına yansıtılmıştır. Bu metodoloji, son sürümde tespit makro $F_1$ ölçütünün $0,692$ değerine, doğru pozitif, yanlış pozitif ve yanlış negatif sayılarının ise sırasıyla $433$, $34$ ve $299$ değerlerine ulaşmasını sağlamıştır.

### 1.5 Tezin Organizasyonu

Tezin geri kalanı dört ana bölümden oluşmaktadır. İkinci bölümde, ses kaynağı ayrıştırma problemine ilişkin klasik sinyal işleme yaklaşımları, zaman-frekans maskeleme teknikleri, derin öğrenme tabanlı ayrıştırma modelleri, sorgu-koşullu ayrıştırma ve FiLM koşullandırma mekanizması ile ses olayı tespiti konularını kapsayan literatür taraması sunulmuştur. Üçüncü bölümde, ses ön işleme ve spektrogram temsili, kullanılan veri kümeleri, anlık karışım üreteci, FiLM-koşullu U-Net mimarisi, kayıp fonksiyonları, optimizasyon süreci (Adam optimize edicisi, karma hassasiyetli eğitim ve XLA derlemesi) ve çıkarım hattı ayrıntılı olarak açıklanmıştır. Dördüncü bölümde, değerlendirme metrikleri tanımlanmış, model sürümlerinin evrimi çözümlenmiş ve ayrıştırma ile tespit başarımına ilişkin niceliksel ve niteliksel bulgular tartışılmıştır. Beşinci bölümde ise elde edilen sonuçlar genel olarak değerlendirilmiş ve gelecekte yapılması önerilen çalışmalar belirtilmiştir.
