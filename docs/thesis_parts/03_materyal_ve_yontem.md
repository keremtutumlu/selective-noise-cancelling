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

## 3.4 Anlık Veri Üretimi: SeparationMixer

Eğitim örnekleri, önceden üretilmiş bir veri dosyasından okunmak yerine, bellek içi klip önbelleğinden her eğitim adımında anlık olarak sentezlenmektedir. Bu görev, sonsuz bir örnek akışı üreten `SeparationMixer` sınıfı tarafından yürütülmektedir. Her örnek, bir karışım spektrogramı, bir sınıf sorgusu ve o sınıfın hedef stem genliğinden oluşan bir üçlüdür. Anlık üretim yaklaşımı, hem depolama yükünü ortadan kaldırmakta hem de aynı kliplerin farklı genlik, pencere ve birleşimlerle yeniden kullanılması sayesinde görece sınırsız bir karışım çeşitliliği sağlamaktadır.

### 3.4.1 Karışım Sentezi ve Genlik Örnekleme

Bir karışım örneği oluşturulurken, önce karışıma katılacak kaynak sayısı $k$, $\{1, 2, \dots, K_{\max}\}$ kümesinden düzgün dağılımla çekilmektedir; bu çalışmada $K_{\max} = 4$ alınmıştır. Ardından sözcük dağarcığından $k$ adet sınıf yerine koymadan örneklenmekte ve her sınıf için önbellekten rastgele bir klip seçilmektedir. Seçilen her klipten, bir saniyelik bir pencere rastgele konumdan kırpılmakta (klip bir saniyeden kısaysa sıfır-doldurma uygulanmakta) ve bu pencere, $[0{,}4;\, 1{,}0]$ aralığından düzgün dağılımla çekilen bir genlik katsayısı $a_i$ ile ölçeklenmektedir. Karışım, ölçeklenmiş pencerelerin toplamı olarak elde edilmektedir:

$$x[n] = \sum_{i=1}^{k} a_i\, s_i[n], \qquad a_i \sim \mathcal{U}(0{,}4;\, 1{,}0).$$

Genlik katsayılarının rastgeleleştirilmesi, modelin farklı bağıl ses düzeylerine karşı dayanıklılık kazanmasını sağlamakta; her kaynağın hedef stem'i, ilgili ölçeklenmiş pencere olarak ayrı ayrı saklanmaktadır.

### 3.4.2 Negatif Örnekler ve Sessizlik Hedefi

Modelin, karışımda bulunmayan bir sınıf sorgulandığında yakın-sıfır bir maske üretmeyi öğrenmesi, seçici bastırmanın temel koşuludur. Bu amaçla, $P_{\text{negatif}}$ olasılığıyla bir *negatif örnek* üretilmektedir: sorgu, karışımda bulunmayan bir sınıfı işaret etmekte ve hedef stem sessizlik (tümüyle sıfır) olarak atanmaktadır. Geri kalan örneklerde sorgu, karışımda bulunan bir sınıfı işaret etmekte ve hedef, o sınıfın stem'i olmaktadır.

$P_{\text{negatif}}$ parametresinin seçimi, kritik bir ödünleşim içermektedir. Çok yüksek bir değer, L1 kaybının her şey için yakın-sıfır çıktıyı ödüllendirmesine ve modelin "güvenli sessizlik" dengesine çökmesine yol açmaktadır; bu olgu, dördüncü bölümde ayrıntılandırılan v2.0 sürümünde $P_{\text{negatif}} = 0{,}45$ değerinde gözlemlenmiştir. Çok düşük bir değer ise yetersiz bastırmaya neden olmaktadır. Modelin son sürümünde, tespit başının yeterli negatif maruziyetle eğitilebilmesi için $P_{\text{negatif}} = 0{,}50$ değeri benimsenmiştir; bu değer, ayrıştırma kaybının pozitif örneklerce, tespit kaybının ise dengeli bir pozitif–negatif karışımıyla beslenmesini sağlamaktadır.

### 3.4.3 Ağırlıklı Zor-Negatif Örnekleme

Negatif örneklerde sorgulanacak yok sınıfının düzgün dağılımla seçilmesi, geniş bantlı ve dağınık maske üreten sınıfların yeterince bastırma örneği görmemesine yol açabilmektedir. Bu sorunu hafifletmek için, ağırlıklı zor-negatif örnekleme mekanizması tanımlanmıştır. Aşırı-tetikleyen (over-firing) olarak işaretlenen sınıflara bir ağırlık katsayısı $w_{\text{of}} = 3{,}0$, diğer tüm sınıflara ise $1{,}0$ atanmakta; oluşan ağırlık vektörü bir olasılık dağılımına normalize edilmektedir. Negatif örnekte yok sınıfı, mevcut sınıflar dışlandıktan sonra bu dağılımdan çekilmektedir:

$$P(\text{yok sınıfı} = c) = \frac{w_c}{\displaystyle\sum_{c' \notin \text{mevcut}} w_{c'}}, \qquad c \notin \text{mevcut}.$$

Bu mekanizma, problemli sınıflar için zor-negatif örneklerin sıklığını, karışım üretim mantığını ya da pozitif örnekleri değiştirmeden artırmaktadır. Modelin son sürümünde, bilinen aşırı-tetikleyen sınıflar sözcük dağarcığından çıkarıldığından, bu mekanizma etkin değildir ($w_{\text{of}}$ listesi boştur); ancak v2.5–v2.8 sürümlerinde geniş bantlı sınıfların bastırılmasında belirleyici bir rol oynamıştır (Bölüm 4).

### 3.4.4 Arka Plan Gürültüsü Artırımı

Modelin hedef kaynağı, yapısız ortam gürültüsünden ayırt etmeyi öğrenmesi için, $P_{\text{gürültü}} = 0{,}10$ olasılığıyla karışıma geniş bantlı gürültü eklenmektedir. Eklenen gürültünün düzeyi, $[15;\, 30]$ dB aralığından düzgün dağılımla çekilen bir işaret-gürültü oranı (SNR) ile belirlenmektedir. Hedef SNR değeri için gürültünün karekök-ortalama-kare (RMS) genliği,

$$\text{RMS}_{\text{gürültü}}^{\text{hedef}} = \frac{\text{RMS}_{\text{karışım}}}{10^{\,\text{SNR}_{\text{dB}}/20}}$$

bağıntısıyla hesaplanmakta ve gürültü, bu hedef RMS'e ölçeklenerek karışıma eklenmektedir. $[15;\, 30]$ dB aralığı, gürültü genliğinin işaret genliğinin yaklaşık $\%6$–$\%18$'i mertebesinde kalmasını sağlayarak gerçekçi ortam düzeylerini taklit etmektedir; aşırı düşük SNR değerlerinin (örneğin v2.0'daki $5$ dB) denetim sinyalini gürültü altında bıraktığı gözlemlenmiştir. Üretilen gürültünün yarısı beyaz gürültü, diğer yarısı ise frekansla $1/\sqrt{f}$ oranında zayıflayan pembe gürültüdür; pembe gürültü, FFT düzleminde genlik tayfının $1/\sqrt{k}$ ile ölçeklenmesiyle elde edilmektedir. Bu iki bileşen, düz ve $1/f$-eğimli gerçek dünya ortam seslerini (havalandırma, kalabalık uğultusu, trafik gürültüsü) birlikte temsil etmektedir. Önemli bir tasarım kararı olarak, gürültü yalnızca karışıma eklenmekte; hedef stem'e *eklenmemektedir*. Böylece model, gürültüyü maske dışında bırakmayı, yani yok saymayı öğrenmektedir.

### 3.4.5 Tepe Normalizasyonu ve Eğitim-Çıkarım Tutarlılığı

Karışım oluşturulduktan sonra, tepe genliği $1{,}0$ olacak biçimde normalize edilmekte ve aynı ölçek katsayısı hedef stem pencerelerine de uygulanmaktadır:

$$x \leftarrow \frac{x}{\max_n |x[n]|}, \qquad s_i \leftarrow \frac{s_i}{\max_n |x[n]|}.$$

Tepe normalizasyonu, STFT genliklerinin eğitim boyunca tutarlı bir dağılımda kalmasını güvence altına almaktadır. Bu adımın çıkarım hattında birebir tekrarlanması zorunludur; aksi hâlde modelin etkinlikleri eğitilmemiş bir çalışma bölgesine kaymaktadır. Nitekim v2.0 sürümünde, çıkarım hattının ham (normalize edilmemiş) sesi modele vermesi, STFT genliklerinin eğitim dağılımına göre üç ila on kat küçük kalmasına ve modelin tüm sınıflarda işlevsiz hâle gelmesine yol açmıştır. Bu hata, çıkarım hattında tam dosya genliğinin tepe normalizasyonuyla giderilmiş ve eğitim ile çıkarım dağılımlarının tutarlılığı sağlanmıştır.

## 3.5 FiLM-Koşullu U-Net Mimarisi

Önerilen modelin çekirdeği, iki girişli ve FiLM ile koşullandırılmış iki boyutlu bir U-Net mimarisidir. Birinci giriş, $(256, 128, 1)$ boyutlu logaritmik genlik spektrogramı; ikinci giriş ise sözcük dağarcığı üzerindeki $(N,)$ boyutlu tek-sıcak sınıf sorgusudur. Ağın çıkışı, sorgulanan sınıf için $[0, 1]$ aralığında değer alan tek bir yumuşak maskedir. İlk kodlayıcı bloğunun kanal sayısı (`base_filters`) $32$ alındığında model yaklaşık $8{,}3$ milyon parametre içermektedir. Mimarinin genel yapısı Şekil 3.3'te gösterilmiştir.

![Şekil 3.3](../thesis_figures_v3.0/architecture.png)

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

### 3.5.4 Sorgu Gömme ve Çok Seviyeli Projeksiyon

Tek-sıcak sınıf sorgusu, önce paylaşılan bir gömme katmanından geçirilmektedir; bu katman, $128$ boyutlu bir yoğun (Dense) gömme üretmekte ve ReLU ile etkinleştirilmektedir. Paylaşılan gömme, her FiLM seviyesi için ayrı ölçek ve öteleme projeksiyonlarına beslenmektedir: her kodlayıcı seviyesi ve darboğaz, kendi $\gamma$ ve $\beta$ parametrelerini üreten bağımsız yoğun katmanlara sahiptir. FiLM koşullandırması yalnızca darboğazda değil, beş ayrı noktada — birinci ila dördüncü kodlayıcı seviyeleri (e1–e4) ve darboğaz — uygulanmaktadır. Koşullandırmanın tüm kodlayıcı seviyelerinde uygulanması, kodlayıcının kendisinin sınıfa özgü özellikler kurmasını zorlamakta; böylece kod çözücüye giren atlama bağlantıları da sınıfa özgü etkinlikler taşımaktadır. Bu çok seviyeli koşullandırmanın maske kesinliğine katkısı, dördüncü bölümde niceliksel olarak gösterilmiştir.

### 3.5.5 Maske Üretimi ve Float32 Çıkış Sabitleme

Kod çözücünün son etkinlik haritası, $1 \times 1$ evrişim ve sigmoid etkinleştirme ile tek kanallı, $[0, 1]$ aralığında bir yumuşak maskeye dönüştürülmektedir. Eğitim sırasında bu maske, ağ grafiğinin içinde doğrudan doğrusal genlik girişiyle çarpılarak kestirilen stem genliğini üretmektedir. Bu amaçla eğitim modeli, üç giriş ($[\,$logaritmik genlik, sınıf sorgusu, doğrusal genlik$\,]$) ve bir çıkış (kestirilen stem genliği) ile sarmalanmıştır. Maskenin uygulanması, bir çarpma katmanıyla gerçekleştirildiğinden ve özel (Lambda) katman içermediğinden, kaydedilen model dosyası özel nesne tanımına gerek kalmadan yeniden yüklenebilmektedir.

Sayısal kararlılık açısından kritik bir tasarım kararı, hem maske çıkış katmanının hem de maske uygulama çarpımının tam hassasiyette (float32) sabitlenmesidir. Eğitim, hesaplama hızı için karma hassasiyet (mixed_float16) politikası altında yürütülmesine karşın (Alt Başlık 3.7.2), sigmoid maske ve onu izleyen L1 kaybı yarı hassasiyette (float16) sayısal doygunluğa ve aşırı yuvarlamaya açıktır. Çıkış katmanının float32'ye sabitlenmesi, ağın gövdesi yarı hassasiyette çalışırken dahi maskenin ve kaybın tam hassasiyette hesaplanmasını güvence altına almaktadır.

### 3.5.6 Tespit Başı

Sınıf varlığının kestirimi için, FiLM ile koşullandırılmış darboğaz temsili üzerine hafif bir tespit başı eklenmiştir. Bu baş, darboğaz etkinliklerini küresel ortalama havuzlama (Global Average Pooling) ile $512$ boyutlu sınıf-duyarlı bir vektöre indirgemekte; ardından $128$ nöronlu bir ReLU yoğun katmanı ve tek nöronlu bir sigmoid katmanıyla, sorgulanan sınıfın varlık olasılığı $P(\text{sorgulanan sınıf mevcut} \mid \text{karışım})$ değerini üretmektedir. Darboğaz halihazırda sınıf sorgusuyla koşullandırıldığından, tespit başı hangi sınıfı değerlendirdiğini örtük olarak bilmektedir.

Tespit başı etkinleştirildiğinde, model iki çıkışlı hâle gelmektedir: kestirilen stem genliği ve sınıf varlık olasılığı. Varlık etiketleri, hedef stem genliğinden otomatik olarak türetilmektedir: hedefin mutlak değerinin en büyüğü bir eşiği ($10^{-6}$) aşıyorsa varlık $1{,}0$, aksi hâlde (negatif örneklerde) $0{,}0$ olarak atanmaktadır. Geriye dönük uyumluluk için, değerlendirme ve uygulama kodu modelin çıkış sayısını denetlemekte; tek çıkışlı eski sürümlerde maske-enerjisi sezgiseline, iki çıkışlı sürümlerde ise tespit başının olasılığına dayanmaktadır. Bu kurgu sayesinde eski ve yeni model sürümleri, kod değişikliği gerektirmeden aynı çıkarım hattıyla sunulabilmektedir.

## 3.6 Kayıp Fonksiyonları

Modelin eğitiminde, ayrıştırma ve tespit görevleri için iki ayrı kayıp fonksiyonu tanımlanmış ve ağırlıklı bir toplamla birleştirilmiştir.

### 3.6.1 Çok Çözünürlüklü L1 Kaybı

Ayrıştırma görevi, kestirilen stem genliği ile gerçek stem genliği arasındaki L1 (mutlak fark) kaybıyla denetlenmektedir. L1 kaybı, L2 (karesel) kaybına kıyasla aykırı değerlere karşı daha az duyarlıdır ve genlik spektrogramlarında daha keskin kestirimler üretmektedir. Bu çalışmada L1 kaybı, tek çözünürlükte değil, çok çözünürlüklü bir biçimde uygulanmıştır:

$$\mathcal{L}_{\text{ayr}} = \sum_{i=0}^{2} \left(\frac{1}{2}\right)^{i} \big\lVert y^{(i)} - \hat{y}^{(i)} \big\rVert_{1},$$

burada $y^{(0)}$ ve $\hat{y}^{(0)}$ tam çözünürlüklü gerçek ve kestirilen genlikleri; $y^{(i)}$ ve $\hat{y}^{(i)}$ ise $i$ kez $2\times 2$ ortalama havuzlama uygulanmış (yarı ve çeyrek çözünürlüklü) karşılıklarını göstermektedir. Çözünürlük seviyeleri sırasıyla $1{,}0$, $0{,}5$ ve $0{,}25$ katsayılarıyla ağırlıklandırılmaktadır. Daha kaba çözünürlükler, ince bin değerleri eniyilenmeden önce genel spektral biçimin oturmasını sağlamakta; bu da maske kalitesini ve eğitim kararlılığını artırmaktadır.

### 3.6.2 İkili Çapraz Entropi ve Odak Kaybı Karşılaştırması

Tespit görevi, sınıf varlık olasılığı üzerinde ikili çapraz entropi (Binary Cross-Entropy, BCE) kaybıyla denetlenmektedir. Eğitim sürecinde, sınıf dengesizliğini ele almak amacıyla odak kaybının (focal loss) BCE yerine kullanılması da denenmiştir. Odak kaybı, her örneğin BCE değerini bir $\alpha_t (1 - p_t)^{\gamma}$ çarpanıyla ölçekleyerek iyi sınıflandırılmış örneklerin katkısını azaltmaktadır. Ancak rastgele başlangıçta ($p \approx 0{,}5$), $\alpha = 0{,}25$ ve $\gamma = 2$ değerleri için bu çarpan

$$\alpha_t (1 - p_t)^{\gamma} = 0{,}25 \times 0{,}5^{2} = 0{,}0625$$

değerini almakta; aynı başlangıçta BCE değeri ise $-\log(0{,}5) \approx 0{,}693$ olmaktadır. Dolayısıyla odak kaybının gradyanı, başlangıçta BCE'ye göre yaklaşık on kat küçüktür. Dördüncü bölümde ayrıntılandırıldığı üzere, bu durum v2.7 sürümünde tespit başının gradyan yetersizliğinden ötürü hiç öğrenememesine ve her giriş için yaklaşık $0{,}5$ üretecek biçimde çökmesine yol açmıştır. Bu gözlem doğrultusunda, modelin son sürümlerinde tespit kaybı olarak BCE benimsenmiştir.

### 3.6.3 Çok Görevli Kayıp Ağırlıklandırması

Tespit başı etkin olduğunda, toplam kayıp, ayrıştırma ve tespit kayıplarının ağırlıklı toplamı olarak tanımlanmaktadır:

$$\mathcal{L} = \mathcal{L}_{\text{ayr}} \cdot w_{\text{ayr}} + \mathcal{L}_{\text{tespit}} \cdot w_{\text{tespit}}.$$

Modelin son sürümünde ayrıştırma ağırlığı $w_{\text{ayr}} = 1{,}0$ ve tespit ağırlığı $w_{\text{tespit}} = 0{,}5$ alınmıştır. Tespit ağırlığının ayrıştırma ağırlığından düşük tutulması, ana görev olan ayrıştırmanın baskın gradyan sinyalini korurken tespit başının da yeterli denetim almasını sağlamaktadır. Tespit için gereken varlık etiketleri, eğitim hattının içinde hedef stem genliğinden otomatik olarak türetilmektedir (Alt Başlık 3.5.6).

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

### 3.7.2 Karma Hassasiyet Eğitimi ve Kayıp Ölçekleme

Eğitim, hesaplama hızı ve bellek kullanımı açısından karma hassasiyet (mixed precision) politikası altında yürütülmüştür. Bu politikada ileri ve geri yayılım hesaplamaları yarı hassasiyette (float16) yapılırken, ana ağırlıklar tam hassasiyette (float32) tutulmaktadır [36]. Yarı hassasiyet, Tensor Çekirdeği (Tensor Core) donanımına sahip grafik işlemcilerde (T4, A100, L4) işlem hızını yaklaşık $1{,}5$–$2$ kat artırmakta ve etkinlik belleğini yarıya indirmektedir.

Yarı hassasiyetin temel riski, küçük gradyanların float16 alt-taşma sınırının altında kalarak sıfıra yuvarlanmasıdır. Bu sorun, kayıp ölçekleme (loss scaling) ile giderilmiştir: kayıp, geri yayılımdan önce büyük bir katsayıyla ölçeklenmekte, gradyanlar hesaplandıktan sonra aynı katsayıyla geri ölçeklenmektedir. Böylece gradyanlar, ara hesaplama boyunca float16'nın temsil edilebilir aralığında tutulmaktadır. Bu işlev, Adam optimize edicisinin bir kayıp-ölçekleme optimize edicisiyle (LossScaleOptimizer) sarmalanmasıyla sağlanmış; sarmalama, herhangi bir başarısızlık ya da tam hassasiyet politikası durumunda yalın Adam'a güvenli biçimde geri dönecek şekilde korumaya alınmıştır. Maske çıkış katmanı ve maske uygulama çarpımının tam hassasiyette sabitlenmesiyle (Alt Başlık 3.5.5) birlikte, karma hassasiyet doğruluk kaybı olmaksızın uygulanmaktadır.

### 3.7.3 XLA (JIT) Derlemesi

Eğitim adımı, TensorFlow çerçevesinin [37] XLA (Accelerated Linear Algebra) tam-zamanında (Just-In-Time, JIT) derleyicisiyle derlenmiştir. XLA, hesaplama grafiğindeki ardışık işlemleri tek bir eniyilenmiş çekirdekte birleştirerek (operator fusion) çekirdek başlatma yükünü ve ara sonuçların bellek üzerinden taşınmasını azaltmaktadır. Bu birleştirme, özellikle yüksek verimli grafik işlemcilerde (A100, L4) belirgin bir hızlanma sağlamakta; T4 üzerinde de katkı sunmaktadır. JIT derleme, eğitim betiğinde bir ortam değişkeniyle denetlenebilmekte ve herhangi bir işlemin derlenememesi durumunda devre dışı bırakılabilmektedir.

### 3.7.4 Öğrenme Oranı Çizelgeleme ve Erken Durdurma

Eğitim, üç geri çağırma (callback) işleviyle denetlenmiştir. Birincisi, doğrulama kaybının (Doğrulama Kaybı) dört dönem boyunca iyileşmemesi durumunda öğrenme oranını yarıya indiren ve alt sınırı $10^{-6}$ olan uyarlanır öğrenme oranı çizelgeleyicisidir (ReduceLROnPlateau). İkincisi, Doğrulama Kaybının on dönem boyunca iyileşmemesi durumunda eğitimi durduran ve en iyi ağırlıkları geri yükleyen erken durdurma (EarlyStopping) işlevidir. Üçüncüsü, Doğrulama Kaybı en düşük olan modeli diske kaydeden model kontrol noktası (ModelCheckpoint) işlevidir. Bu üçlü, hem aşırı uyumu (overfitting) sınırlandırmakta hem de eğitim sonunda en başarılı kontrol noktasının korunmasını sağlamaktadır.

### 3.7.5 Paralel Veri Hattı

Tek bir Python üreteci üzerinde çalışan ve örnek başına iki librosa STFT hesaplayan veri hattının, grafik işlemcisini aç bıraktığı gözlemlenmiştir; bir A100 üzerinde yaklaşık $20$ ms'de çalışması beklenen bir eğitim adımı, üretecin darboğazı nedeniyle yaklaşık $100$ ms sürmüştür. Bu darboğaz, birden çok bağımsız karışım üretecinin işçi iş parçacıkları arasında dönüşümlü olarak işletilmesiyle (tf.data interleave) giderilmiştir. Varsayılan olarak dört paralel üretici kullanılmakta; her üretici ayrı bir tohum (seed) değeriyle başlatılmaktadır. Çözülmüş veri kümesi, tüm üreticiler ve doğrulama üreteci tarafından paylaşılan tek bir bellek içi sözlükte tutulmaktadır; böylece dört veri işçisi, çok gigabaytlık veri kümesinin dört kopyası yerine yalnızca dört iş parçacığı maliyeti getirmektedir. Hat, ön getirme (prefetch) ile tamamlanarak veri hazırlamanın hesaplama ile örtüşmesi sağlanmıştır.

### 3.7.6 Hiperparametreler ve Donanım Ortamı

Modelin son sürümünde kullanılan başlıca hiperparametreler Tablo 3.1'de özetlenmiştir.

**Tablo 3.1:** v3.0 sürümünün eğitim hiperparametreleri.

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

Eğitim, Tensor Çekirdeği destekli grafik işlemcileri (Google Colab ortamında A100 80 GB ve T4) üzerinde, karma hassasiyet ve XLA etkin biçimde yürütülmüştür. Eğitim süreci boyunca Doğrulama Kaybı izlenmiş; modelin son sürümlerinde altmış dönem boyunca kararlı bir yakınsama gözlemlenmiştir. Eğitim, çıkarım ve değerlendirme betikleri, GPU bulunmayan ortamlar için Colab not defterleriyle de yinelenebilir kılınmıştır.

## 3.8 Çıkarım Hattı

Eğitilmiş model, bir web uygulaması üzerinden uçtan uca bir gürültü temizleme hattıyla işlevsel kılınmıştır. Çıkarım hattı, kullanıcının yüklediği dosyada bulunan sınıfların tespit edilmesi ve seçilen sınıfların dosyadan çıkarılması olmak üzere iki ana aşamadan oluşmaktadır.

### 3.8.1 Ses Olayı Tespiti

Tespit aşamasında, yüklenen dosyadan hız için en çok sekiz saniyelik bir bölüm bir saniyelik pencerelere ayrılmaktadır. Her pencere, eğitimle tutarlılığı korumak için tepe genliği $1{,}0$ olacak biçimde normalize edilmekte ve modele beslenmektedir. Tespit başına sahip modellerde, her aday sınıf için pencereler üzerinden ortalama varlık olasılığı bir tespit puanı olarak kullanılmaktadır. Tespit başının yakınsamadığı ve tüm sınıflar için yaklaşık $0{,}5$ değerinde neredeyse düzgün bir olasılık ürettiği durumlara karşı bir koruma tanımlanmıştır: puan aralığı $0{,}15$ eşiğinin altında kalırsa, hat maske-enerjisi sezgiseline geri dönmektedir.

Aday sınıflar, varsa modele ait tespit izin listesiyle (allow-list) sınırlandırılmakta; böylece yerel verinin doğrulayamadığı sınıflar aday havuzundan çıkarılmaktadır. Yüzeye çıkarılacak sınıflar, iki ölçütle belirlenmektedir: mutlak bir taban ($0{,}05$) ve kazanan sınıf puanının göreli bir oranı ($0{,}80 \times$ kazanan). Bu iki eşiğin büyüğü kesme değeri olarak alınmakta ve eşiği aşan en çok beş sınıf kullanıcıya sunulmaktadır. Bu çalışma noktası ($\text{taban} = 0{,}05$, $\text{göreli kesme} = 0{,}80$, $k = 5$), dördüncü bölümde sunulan eşik taramasında en az sahte yüzeyleme ile en yüksek kesinliği veren ayar olarak seçilmiştir.

### 3.8.2 Oran Maskesi ile Kaynak Çıkarma

Çıkarma aşamasında, her seçili sınıf için model bir genlik kestirimi üretmekte ve bu kestirimden bir bastırma maskesi türetilmektedir. Bir zaman-frekans hücresinde, kestirilen stem genliği $\hat{e}_c$ ile karışım genliği $|X|$ arasındaki genlik oranı

$$r_c = \mathrm{clip}\!\left(\frac{\hat{e}_c}{|X| + \varepsilon},\, 0,\, 1\right)$$

biçiminde hesaplanmaktadır. Genlik oranının (güç oranı $\hat{e}_c^2/|X|^2$ yerine) tercih edilmesi, modelin tutucu (küçük) genlik kestirimleri ürettiği durumlarda daha yüksek bir tepkisellik sağlamasındandır. Seçilen tüm sınıflar için bastırma maskeleri çarpımsal olarak birleştirilmekte; bir kullanıcı denetimli *çıkarma kuvveti* katsayısı $\lambda \in [0, 1]$ ile bastırma şiddeti ayarlanmaktadır:

$$M = \prod_{c \in \text{seçili}} \mathrm{clip}\!\big(1 - \lambda\, r_c,\, 0,\, 1\big).$$

$\lambda = 1$ tam bastırmaya, $\lambda = 0$ ise hiç bastırma yapılmamasına karşılık gelmektedir. Çarpımsal birleştirme, birden çok sınıfın aynı hücredeki katkılarının kümülatif olarak bastırılmasını sağlamaktadır.

### 3.8.3 Hanning Pencereli Örtüşmeli Toplama

Çıkarma işlemi, tüm dosya boyunca, model girişinin sabit zaman boyutuna ($128$ çerçeve) uyacak biçimde parçalar hâlinde uygulanmaktadır. Parçalar arası adım, $\text{TIME\_FRAMES}/4 = 32$ çerçeve (yaklaşık $0{,}25$ s) olarak belirlenmiş; bu, ardışık parçalar arasında $\%75$ örtüşme sağlamaktadır. Her parça, bir Hann penceresiyle ağırlıklandırılıp örtüşmeli toplama (overlap-add) ile birleştirilmekte ve birikmiş ağırlıklara bölünerek normalize edilmektedir. Örtüşme oranının $\%50$'den $\%75$'e çıkarılması, parça sınırlarında düzenli aralıklarla duyulan ve dördüncü bölümde belgelenen darbeli yapaylığı (boundary pulsing) gidermiştir. Ayrıca, müzikal gürültüyü bastırmak için maske, zaman ekseninde beş çerçevelik (yaklaşık $40$ ms) bir ortalama çekirdeğiyle düzleştirilmektedir.

### 3.8.4 Faz Yeniden Kullanımı ve Ters STFT

Model yalnızca genlik düzleminde çalıştığından, faz bilgisi kestirilmemekte; bunun yerine karışımın özgün fazı yeniden kullanılmaktadır. Tüm dosya için tek bir STFT hesaplanmakta ve böylece faz, parça sınırları boyunca küresel olarak tutarlı kalmaktadır. Maskelenmiş genlik, düşürülen Nyquist bini sıfırla yeniden eklendikten sonra karışımın karmaşık fazıyla çarpılarak ters STFT'ye verilmekte ve zaman düzlemine geri döndürülmektedir. Yeniden sentezlenen dalga biçimi, başlangıçtaki tepe normalizasyonu tersine çevrilerek özgün ölçeğine döndürülmekte; gerekirse kırpılmayı önlemek için yeniden ölçeklenmektedir. Genlikten faz kestiren yinelemeli Griffin–Lim türü yöntemler [38] daha yüksek bir yeniden sentez kalitesi sunabilmekle birlikte, ek hesaplama maliyeti ve yakınsama belirsizliği nedeniyle bu çalışmada faz yeniden kullanımı tercih edilmiştir.

### 3.8.5 Web Uygulaması ve Video Entegrasyonu

Çıkarım hattı, bir Gradio web uygulaması üzerinden sunulmaktadır. Kullanıcı bir ses ya da video dosyası yüklemekte, tespit edilen sınıfları işaretlemekte, çıkarma kuvvetini bir kaydırma çubuğuyla ayarlamakta ve temizlenmiş çıktıyı indirmektedir. Yalnızca ses içeren girişlerde, işlem öncesi ve sonrası karşılaştırma için bir "önce/sonra" ses çifti sunulmaktadır. Ek olarak, her seçili sınıfın model tarafından çıkarılan stem'i ayrı ayrı oynatılabilmekte; böylece her sınıfın hangi içerik olarak algılandığı işitsel olarak doğrulanabilmektedir. Video girişlerinde, temizlenen ses izi `ffmpeg` aracılığıyla özgün görüntü izinin üzerine yeniden bindirilerek video çıktısı üretilmektedir. Bu uçtan uca hat, üçüncü bölümde açıklanan tüm bileşenleri (ön işleme, koşullu model, tespit ve çıkarma) tek bir kullanıcı arayüzünde birleştirmektedir.
