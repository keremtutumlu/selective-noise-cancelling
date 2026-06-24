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

Bu tez çalışmasında, hedef sınıfın genlik spektrogramı için $[0,1]$ aralığında değer alan yumuşak bir maske kestirilmekte ve yeniden sentez aşamasında karışımın özgün fazı korunmaktadır. Dolayısıyla benimsenen kurgu, oran maskesi ve genlik maskesi ailesine girmekte; faz duyarlı ve karmaşık maske türevleri, ilgili başarım–maliyet ödünleşimi gerekçesiyle kapsam dışında bırakılmaktadır. Maskenin oracle bir işaret-gürültü oranından hesaplanması yerine, bir derin sinir ağı tarafından doğrudan öğrenilmesi söz konusudur; bu ağların mimari gelişimi Alt Başlık 2.4'te ele alınmıştır.
