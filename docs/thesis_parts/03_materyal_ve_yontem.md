# 3. MATERYAL VE YÖNTEM

Bu bölümde, önerilen seçici gürültü engelleme sisteminin tüm bileşenleri ayrıntılı biçimde açıklanmıştır. Önce genel sistem mimarisi ve veri akışı tanıtılmış; ardından ses ön işleme ve spektrogram temsili, kullanılan veri kümeleri, anlık karışım üreteci, FiLM-koşullu U-Net mimarisi, kayıp fonksiyonları, optimizasyon süreci ve çıkarım hattı sırasıyla ele alınmıştır. Açıklamalar, modelin v3.0 sürümünde kullanılan yapılandırma temel alınarak verilmiştir.

## 3.1 Genel Sistem Mimarisi ve Veri Akışı

Sistem, eğitim ve çıkarım olmak üzere iki ayrı veri akışından oluşmaktadır. Eğitim akışında, ham ses kümeleri bellek içi bir klip önbelleğine çözülmekte, bu önbellekten anlık olarak karışımlar sentezlenmekte ve FiLM-koşullu U-Net bu karışımlar üzerinde eğitilmektedir. Çıkarım akışında ise eğitilmiş model, kullanıcının yüklediği dosyada bulunan sınıfları tespit etmekte ve seçilen sınıfları örtüşmeli toplama yöntemiyle dosyadan çıkarmaktadır. Sistemin uçtan uca veri akışı Şekil 3.1'de şematik olarak gösterilmiştir.

```
ESC-50 + UrbanSound8K (ham WAV kayıtları)
    → load_all_datasets        : {sınıf: [dalga biçimi]} bellek içi önbellek
    → SeparationMixer          : anlık karışım, sorgu ve hedef stem üretimi
    → ConditionedSeparatorTrainer : FiLM U-Net, çok çözünürlüklü L1 + BCE
    → separator_unet_film_multi_v3.0.h5
    → webapp                   : tespit → maske → ISTFT → yeniden sentez
```
**Şekil 3.1:** Önerilen sistemin uçtan uca veri akışı şeması.

Yazılım mimarisi, sorumlulukların ayrıştırıldığı modüler bir yapıda tasarlanmıştır. Veri hazırlama katmanı (`dataset_sources.py` ve `separation_mixer.py`), veri kümelerinin yüklenmesinden ve karışımların üretilmesinden; model eğitimi katmanı (`conditioned_separator.py` ve `train_conditioned_separator.py`), ağ mimarisinin tanımlanmasından ve eğitilmesinden; uygulama katmanı (`webapp.py`) ise çıkarımın kullanıcı arayüzüyle bütünleştirilmesinden sorumludur. Model sürümü ve dosya yolları, ortam değişkeniyle denetlenebilen merkezi bir yapılandırma modülünde (`model_config.py`) tutulmakta; böylece farklı sürümler arasında geçiş, kaynak kodu değiştirmeden yapılabilmektedir. Bu modüler kurgu, sürüm tabanlı deneysel metodolojinin (Bölüm 4) izlenebilirliğini de sağlamaktadır.

## 3.2 Ses Ön İşleme ve Spektrogram Temsili

Modelin tüm girişleri, ortak bir spektrogram sözleşmesine indirgenmiştir. Bu sözleşme, hem eğitim hem de çıkarım hatlarında birebir aynı parametrelerle uygulanarak eğitim ve çıkarım dağılımlarının tutarlılığı güvence altına alınmıştır.

### 3.2.1 Örnekleme ve Tek Kanala İndirgeme

Tüm ses verileri $16$ kHz örnekleme hızında yeniden örneklenmiş ve tek kanala (mono) indirgenmiştir. $16$ kHz örnekleme hızı, Nyquist–Shannon örnekleme kuramı gereği $8$ kHz'e kadar olan frekans bileşenlerinin temsil edilmesine olanak tanımaktadır; bu bant, çevresel seslerin algısal olarak baskın enerjisini içermektedir. Daha yüksek örnekleme hızları ($44{,}1$ kHz gibi) ek bir üst-bant ayrıntısı sağlasa da, spektrogram boyutunu ve dolayısıyla hesaplama ile bellek yükünü orantısız biçimde artırmaktadır. Bu nedenle $16$ kHz, temsil yeterliliği ile hesaplama maliyeti arasında bir uzlaşı noktası olarak belirlenmiştir. Çok kanallı kayıtlar, kanal ortalaması alınarak tek kanala indirgenmiş; böylece modelin girişi, kanal sayısından bağımsız hâle getirilmiştir.

### 3.2.2 Kısa Zamanlı Fourier Dönüşümü Sözleşmesi

Zaman düzlemindeki dalga biçimi, Kısa Zamanlı Fourier Dönüşümü (STFT) ile zaman-frekans düzlemine taşınmıştır. STFT, bir analiz penceresi $w[n]$ kaydırılarak hesaplanan ardışık Fourier dönüşümleri olarak tanımlanmaktadır:

$$X[k, m] = \sum_{n=0}^{L-1} w[n]\, x[n + mH]\, e^{-j 2\pi k n / L},$$

burada $L$ pencere uzunluğu (FFT boyutu), $H$ sıçrama uzunluğu (hop length), $k$ frekans bini ve $m$ zaman çerçevesi indisidir. Bu çalışmada $L = n_{\text{fft}} = 512$ ve $H = hop = 128$ değerleri seçilmiştir. $16$ kHz örnekleme hızında bu değerler, $32$ ms'lik bir analiz penceresine ve $8$ ms'lik bir çerçeve adımına karşılık gelmekte; ardışık pencereler arasında $\%75$ örtüşme sağlamaktadır. Pencereleme işlevi olarak Hann penceresi kullanılmıştır; bu pencere, spektral sızıntıyı sınırlandırması ve örtüşmeli toplama altında birim-zarf koşulunu yaklaşık olarak sağlaması bakımından tercih edilmiştir.

$512$ noktalı FFT, $257$ adet tek-yanlı frekans bini üretmektedir. Nyquist bini ($k = 256$) düşürülerek model girişi $256$ frekans binine indirgenmiş; böylece frekans ekseni, evrişimsel alt örnekleme için elverişli olan $2$'nin kuvveti bir boyuta ($256$) sabitlenmiştir. Bir saniyelik bir pencere ($16\,000$ örnek), merkez hizalı STFT altında yaklaşık $126$ zaman çerçevesi üretmekte; bu eksen, sabit girdi boyutu için $128$ çerçeveye sıfır-doldurma ile tamamlanmaktadır. Sonuç olarak modelin girişi, $(256, 128, 1)$ boyutlu bir genlik spektrogramı tensörüdür ve her tensör yaklaşık bir saniyelik akustik bağlamı temsil etmektedir.

### 3.2.3 Logaritmik Genlik Sıkıştırması

Ses sinyallerinin genlik spektrumu, birkaç on yıllık (decade) bir dinamik aralığa yayılmaktadır. Bu geniş aralığın doğrudan ağa verilmesi, yüksek enerjili bileşenlerin gradyanları baskılamasına ve düşük enerjili ancak algısal olarak önemli bileşenlerin göz ardı edilmesine yol açmaktadır. Bu nedenle model girişinde, genlik spektrogramına logaritmik sıkıştırma uygulanmıştır:

$$X_{\log}[k, m] = \log\!\big(1 + |X[k, m]|\big).$$

$\log(1 + \cdot)$ biçimindeki sıkıştırma, sıfır genlikte tanımlı kalması (logaritmanın tekilliğinden kaçınması) ve küçük genlikler için yaklaşık doğrusal davranması bakımından tercih edilmiştir. Bu dönüşüm, dinamik aralığı sıkıştırarak eğitim kararlılığını artırmakta ve insan işitmesinin logaritmik yükseklik algısıyla örtüşmektedir. Modelin yalnızca logaritmik genlik girişiyle koşullandırıldığı; maskenin uygulanacağı doğrusal genliğin ise ayrı bir giriş olarak ağ grafiğine taşındığı vurgulanmalıdır. Bu ayrım sayesinde maske, doğrusal genlik üzerinde uygulanmakta ve kestirilen stem, ölçek bilgisini koruyarak yeniden sentezlenebilmektedir (Alt Başlık 3.5.5).

## 3.3 Veri Kümeleri

Modelin sözcük dağarcığı, halka açık çevresel ses veri kümelerinin birleştirilmesiyle oluşturulmuştur. Birleştirme işlemi, her veri kümesini ortak bir $\{$sınıf$:$ dalga biçimi listesi$\}$ sözlüğüne çözen ve bu sözlükleri tek bir önbellekte toplayan veri yükleme katmanı tarafından yürütülmektedir.

### 3.3.1 ESC-50

ESC-50, elli çevresel ses sınıfından oluşan ve sınıf başına kırk klip içeren, toplam iki bin kayıtlık bir veri kümesidir [31]. Her klip beş saniye uzunluğundadır ve hayvan sesleri, doğal ses olayları, insan kaynaklı sesler, iç mekân sesleri ve kentsel gürültü gibi geniş bir akustik yelpazeyi kapsamaktadır. Sınıf başına klip sayısının sınırlı olması, bu çalışmadaki minimum klip tabanı eşiğinin (Alt Başlık 3.3.4) belirlenmesinde de belirleyici olmuştur.

### 3.3.2 UrbanSound8K ve Sınıf Birleştirme

UrbanSound8K, on kentsel ses sınıfından oluşan ve sekiz binin üzerinde klip içeren bir veri kümesidir [32]. Bu kümenin bazı sınıfları, ESC-50 ile anlamsal olarak örtüşmektedir. Örtüşen sınıfların ayrı etiketler hâline gelmesini önlemek için bir takma ad eşlemesi (`CLASS_ALIASES`) tanımlanmış; örneğin UrbanSound8K'deki köpek havlaması sınıfı (`dog_bark`), ESC-50'deki köpek sınıfına (`dog`), motor rölantisi sınıfı (`engine_idling`) ise motor sınıfına (`engine`) eşlenmiştir. Bu eşleme sayesinde örtüşen sınıfların klipleri tek bir kanonik etiket altında havuzlanmakta; UrbanSound8K'nin on sınıfından dördü ESC-50 ile birleşmekte, altısı ise yeni sınıf olarak eklenmektedir. İki veri kümesinin birlikte yüklenmesiyle modelin sözcük dağarcığı yaklaşık elli altı sınıfa ulaşmaktadır.

### 3.3.3 FSD50K ve Uzun Kuyruk Problemi

FSD50K, AudioSet ontolojisine göre etiketlenmiş, yaklaşık iki yüz sınıf içeren büyük ölçekli bir ses olayı veri kümesidir [33]. Bu kümede etiketler hiyerarşiktir ve çoğu klip virgülle ayrılmış birden çok etikete sahiptir; bu çalışmada her klibin ilk (en özgül/yaprak) etiketi kanonik sınıf olarak alınmıştır. FSD50K'nin sözcük dağarcığını genişletme potansiyeli bulunmakla birlikte, yaprak etiketlerinin önemli bir bölümünün yalnızca birkaç klip tarafından desteklenmesi, bir "uzun kuyruk" problemi doğurmaktadır. Az sayıda ve çok-etiketli örnekten öğrenilen bir sınıf, ayırt edici olmayan ve dağınık bir maske üretmekte; bu maske, ilgili sınıf karışımda bulunmasa dahi yüksek enerji üreterek yanlış pozitiflere yol açmaktadır. Bu olgu, dördüncü bölümde ayrıntılandırılan sürüm evriminde (özellikle v2.3–v2.4) belirleyici bir başarısızlık örüntüsü olarak gözlemlenmiş ve modelin son sürümünde FSD50K bütünüyle dışarıda bırakılmıştır.

### 3.3.4 Minimum Klip Tabanı ve Düzenlenmiş Sözcük Dağarcığı

Az desteklenen sınıfların yarattığı yanlış pozitif eğilimini sınırlandırmak için, birleştirme sonrası uygulanan bir minimum klip tabanı eşiği tanımlanmıştır. Bu eşik, sınıf başına en az kırk klip ($N_{\min} = 40$) koşulunu sağlamayan sınıfları sözcük dağarcığından çıkarmaktadır. Eşik birleştirmeden *sonra* uygulandığından, veri kümeleri arası takma adlar önce havuzlanmakta; yalnızca gerçekten yetersiz desteklenen sınıflar elenmektedir. ESC-50 (sınıf başına kırk klip) ve UrbanSound8K (sınıf başına yüzlerce klip) sınıfları bu eşikten etkilenmemektedir.

Modelin son sürümü (v3.0), ayrıştırma ve tespit başarımının en yüksek olduğu on beş sınıftan oluşan, düzenlenmiş (curated) bir alt küme üzerinde eğitilmiştir. Bu alt küme, `keep_classes` parametresiyle yalnızca döndürülen sözlüğe uygulanmakta; diskteki çözülmüş önbellek tam sözcük dağarcığını koruduğundan, düzenlenmiş alt küme çalışması ile tam sözcük dağarcığı çalışması aynı önbellek dosyasını paylaşabilmektedir. Seçilen on beş sınıfın klip dağılımı Şekil 3.2'de gösterilmiştir; `gun_shot` sınıfı (UrbanSound8K katkısıyla) $374$ klip içerirken, kalan on dört sınıf $N_{\min} = 40$ tabanında dengelenmiştir.

![Şekil 3.2](../thesis_figures_v3.0/01_dataset_clip_counts.png)

**Şekil 3.2:** v3.0 sürümünün on beş sınıflı düzenlenmiş sözcük dağarcığında sınıf başına klip sayısı.
