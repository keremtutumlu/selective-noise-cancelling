# Interim Report — Revision Guide

A working checklist for editing **`Bitirme Projesi Mayıs Ara Rapor Taslak.docx`**.

> **How to use this file:** Each item below points to a section by its heading
> and the approximate paragraph index in the current draft. Work top-to-bottom
> through *Part B* (priority fixes) first, then sweep *Part C* (section map).
> Paragraph indices refer to the draft as reviewed on 2026-05-31 (315 paragraphs,
> 3 tables) — they will drift as you edit, so treat them as anchors, not addresses.
>
> **Language note:** the instructions/rationale here are in English (repo
> convention). Every block of text meant to be **pasted into the report** is given
> in Turkish inside a quote block, ready to adapt.

---

## Part A — The one decision that fixes everything

The draft has a **split identity**. It describes two different systems as if
they were one continuous pipeline:

| | System A (Nov–Mar work) | System B (current / May work) |
|---|---|---|
| Task | Sound **classification** + digital ANC | Query-conditioned **source separation** |
| Model | MobileNetV2 (Log-Mel, transfer learning) | FiLM-conditioned 2-D U-Net |
| "Cleaning" mechanism | Anti-phase wave (`target = input × −1`) | Soft spectrogram mask × magnitude |
| Headline metric | **84.38 %** accuracy, **75.10 dB** digital cancellation | **SI-SDRi**, **Detection F1** |
| Deployment idea | MCU / Edge AI / quantization / pruning | Gradio **web app** |

The draft currently lets System A's metrics (84.38 %, 75.10 dB) stand in as proof
that the whole project works, while System B's honest metrics (SI-SDRi still
negative) are explained away. That is the core credibility problem.

**The fix is a single framing decision, not a teardown.** Because this is an
*interim* report, keep the chronological story (Kasım → Ocak → Mart → Mayıs) and
all prior work. Just reframe the spine as:

> **Sınıflandırma tek başına bir sesi karışımdan cerrahi olarak çıkaramaz; bu
> yüzden proje, sorgu-koşullu kaynak ayrıştırmaya (source separation) yöneldi.
> Web tabanlı seçici ses temizleme sistemi bu raporun kapsamıdır; gömülü/Edge AI
> entegrasyonu ve gerçek-zamanlı akustik ANC ise projenin sonraki fazıdır.**

Everything in Part B serves that one sentence.

---

## Part B — Priority fixes (do these in order)

### P1 — Rewrite the Abstract / Özet  *(ABSTRACT ≈ para 82–87; Turkish Özet ≈ para 73–81)*

**Problem.** The current English abstract (para 84–85) is the *proposal* abstract:
it sells FxLMS replacement, MobileNetV2, 8-bit quantization, structural pruning,
embedded headphone hardware. None of that is the May deliverable. Date also reads
"January, 2026" (para 86).

**Action.** Replace the body so the *system actually built* leads, and edge AI is
one closing sentence. Fix the date to **May, 2026 / Mayıs 2026**.

Suggested English replacement (ABSTRACT body):

> This project develops a **selective sound-removal system**: given a short audio
> (or video) recording, it detects which sound classes are present and lets the
> user choose which ones to remove, returning the cleaned output. The core is a
> **FiLM-conditioned U-Net** that is told *which* class to extract and emits a soft
> spectrogram mask for it; a Gradio web application drives the full
> detect → select → render pipeline. Unlike conventional FxLMS-based Active Noise
> Cancellation, which cannot tell one sound source from another, this
> query-conditioned approach targets a single chosen class while preserving the
> rest. This interim report covers the web-based separation system and its
> iterative evaluation (v2.0–v2.4). Deployment on embedded/Edge-AI hardware and
> real-time acoustic ANC are planned for the project's next phase.

Then **mirror the same content in the Turkish Özet** (para 73–81). Suggested
Turkish text:

> Bu proje, bir **seçici ses temizleme sistemi** geliştirmektedir: kısa bir ses
> (veya video) kaydı verildiğinde sistem, kayıtta hangi ses sınıflarının
> bulunduğunu tespit eder, kullanıcının hangilerini kaldıracağını seçmesine izin
> verir ve temizlenmiş çıktıyı döndürür. Sistemin çekirdeği, *hangi* sınıfı
> çıkaracağı kendisine söylenen ve o sınıf için yumuşak bir spektrogram maskesi
> üreten **FiLM-koşullu bir U-Net** mimarisidir; bir Gradio web uygulaması
> tespit → seçim → işleme akışının tamamını yürütür. Bir ses kaynağını
> diğerinden ayıramayan klasik FxLMS tabanlı Aktif Gürültü Engellemenin (ANC)
> aksine, bu sorgu-koşullu yaklaşım yalnızca seçilen tek bir sınıfı hedeflerken
> geri kalanı korur. Bu ara rapor, web tabanlı ayrıştırma sistemini ve onun
> yinelemeli değerlendirmesini (v2.0–v2.4) kapsar. Gömülü/Uç-YZ (Edge AI)
> donanımına yerleştirme ve gerçek zamanlı akustik ANC, projenin sonraki fazına
> bırakılmıştır.

---

### P2 — Label 75.10 dB and 84.38 % honestly  *(para 208–211, 218–223, 250–255)*

These two numbers are real, but they belong to **System A**, not to the
separation system, and one of them is an upper bound, not an achievement.

**75.10 dB** (para 210, 255) is the RMS energy drop when a signal is added to its
own exact digital inverse (`x + (−x) ≈ 0`). That is an algebraic identity computed
inside a simulator — not an acoustic measurement in air.

**Action.** Wherever 75.10 dB appears, append a clause marking it as a digital
upper bound. Suggested Turkish insert:

> Bu 75.10 dB değeri, bir sinyal ile onun **birebir dijital ters fazının** üst üste
> bindirilmesiyle (x + (−x)) elde edilen, **dijital simülasyondaki teorik üst
> sınırdır**; mikrofon–hoparlör gecikmesi, oda akustiği ve donanım kısıtlarını
> içeren gerçek bir akustik ortamda ölçülmemiştir. Gerçek-ortam akustik doğrulama,
> projenin gömülü fazına aittir.

**84.38 %** (para 251) is MobileNetV2 *classification* accuracy. Keep it, but stop
using it as evidence that *removal/separation* works. In **"Bulguların Proje
Hedefleriyle İlişkisi"** (para 222–223), the draft chains
"84.38 % + 75.10 dB ⇒ system validated." Break that chain. Suggested Turkish
reframing for that paragraph:

> %84.38 sınıflandırma doğruluğu, sistemin **"hangi ses var?"** sorusunu (tanıma
> aşaması) güvenilir biçimde yanıtladığını gösterir. Ancak bir sesi karışımdan
> *çıkarmak*, sınıflandırmadan ayrı bir yetenektir ve bunun nicel ölçütü
> **SI-SDRi** ile **Detection F1**'dir. Dolayısıyla projenin ayrıştırma başarımı
> bu iki metrikle raporlanır; sınıflandırma doğruluğu ve dijital ANC sönümlemesi
> ise sistemin **erken faz (System A) PoC** bulgularıdır.

Also fix the stray double period in para 223 ("...değiştirmemektedir. . Zira...").

---

### P3 — Add the v2.3 results and the v2.4 fix  *(append inside "Nicel Değerlendirme Bulguları", after para 220; and into the Mayıs section ≈ para 263)*

This is the **most scientifically valuable** part of the May work and it is
entirely missing — the draft stops at v2.2 and only mentions v2.3 as a *plan*
(para 223). Add the actual v2.3 outcome and the v2.4 response.

Suggested Turkish text (new subsection):

> **v2.3 — FSD50K entegrasyonu ve yanlış-pozitif çöküşü.** `negative_prob` 0.15'e
> düşürüldü ve FSD50K veri seti (yükleyici hatası giderildikten sonra) doğru
> şekilde yüklendi; model söz dağarcığı **56 sınıftan 235 sınıfa** çıktı. SI-SDRi
> marjinal olarak iyileşti (**−22.79 → −21.49 dB, +1.30 dB**), ancak Detection F1
> **0.13'ten 0.02'ye çöktü** (TP/FP/FN = 13 / 1092 / 475). Neden: yerelde sesi
> bulunmayan 179 FSD50K-özel sınıf, hiçbir gerçek karışımda bulunamadığından
> yalnızca yanlış-pozitif üretebiliyor; `purr`, `bass_guitar`, `ringtone`, `boom`
> gibi az ve gürültülü örnekle öğrenilen sınıflar **dağınık (diffuse) maskeler**
> üretip alakasız her seste yüksek skor veriyor. Bu, "her dosyada bass_guitar
> çıkıyor" gözleminin nicel karşılığıdır.
>
> **v2.4 — Minimum klip-sayısı eşiği.** Çözüm olarak `load_all_datasets` içine
> **`min_clips_per_class = 40`** birleşim-sonrası (post-merge) eşiği eklendi. Eşik,
> FSD50K uzun-kuyruğunu budar; ESC-50 (sınıf başına 40 klip) ve UrbanSound8K
> (yüzlerce klip) etkilenmez. Eşik birleşimden *sonra* uygulandığından, çapraz-veri
> aliasları önce havuzlanır (FSD50K `bark` klipleri ESC-50 `dog`'a sayılır) ve
> yalnızca gerçekten yetersiz desteklenen FSD50K-özel etiketler düşürülür.
> **Tek-değişkenli** bir müdahale olması, etkisinin temiz biçimde atfedilebilmesini
> sağlar. *(v2.4 eğitimi henüz tamamlanmamıştır; sonuçlar eğitim sonrası eklenecektir.)*

> ⚠️ **Do not invent v2.4 numbers.** v2.4 is *designed* but not yet trained. Present
> it as the fix and its rationale; add metrics only after the Colab run.

A compact iteration table is the clearest single artifact you can add. Suggested
Turkish table:

| Versiyon | `negative_prob` | Veri seti | SI-SDRi | Detection F1 | Durum |
|---|---|---|---|---|---|
| v2.0 | 0.45 | ESC-50 + US8K (~56) | — | — | Başarısız (eğitim çöküşü) |
| v2.1 | 0.30 | ~56 | −22.18 dB | 0.21 | Kısmen çalışıyor |
| v2.2 | 0.30 | ~56 (FSD50K=0 klip, hata) | −22.79 dB | 0.13 | Sınır titremesi giderildi |
| v2.3 | 0.15 | 235 (FSD50K yüklendi) | −21.49 dB | **0.02** | YP çöküşü tespit edildi |
| v2.4 | 0.15 + `min_clips=40` | budanmış | *beklemede* | *beklemede* | Eğitim bekliyor |

**Scientific message to state explicitly:** the v2.3→v2.4 episode shows that
**per-class data volume/quality can matter more than model capacity or the
negative-sampling rate** — the same positive:negative ratio that ESC-50 classes
handle fine still produced systematic false positives for under-supported FSD50K
classes.

---

### P4 — Fix the dataset section  *(VERİ SETİNİN HAZIRLIK İŞLEMLERİ ≈ para 133–164; and "Veri Seti" ≈ para 238)*

Three inaccuracies here:

1. **LibriSpeech (para 141) is listed but not used** by the current pipeline.
   Either remove it, or move it to future work with an honest note.
   Suggested Turkish note if you keep it:
   > LibriSpeech, konuşma-koruma senaryosu için değerlendirilmiş ancak mevcut
   > ayrıştırma pipeline'ına (ESC-50 + UrbanSound8K + FSD50K) henüz dahil
   > edilmemiştir; konuşma-hedefli ayrıştırma sonraki faza bırakılmıştır.

2. **FSD50K is used (v2.2+) but missing** from the dataset list. Add it:
   > **FSD50K:** AudioSet hiyerarşisiyle etiketlenmiş, çok-etiketli (multi-label)
   > geniş bir ses olayı veri setidir. Sınıf söz dağarcığını genişletmek için v2.3
   > itibarıyla havuza eklenmiştir; çok-etiketli kliplerde ilk etiket kanonik sınıf
   > olarak alınır ve `CLASS_ALIASES` ile çakışan sınıflar ESC-50 kanonik adlarına
   > eşlenir.

3. **Phase-inversion target (para 156)** — "hedef veri = giriş × −1" — describes
   **System A** (ANC), not the current separation pipeline, whose target is the
   isolated source stem and whose output is a soft mask. Don't delete the
   paragraph (it is true history); add a clause marking it as the earlier approach:
   > *(Not: Ters-faz hedefleme, projenin erken faz ANC yaklaşımına aittir. Mevcut
   > kaynak-ayrıştırma sistemi hedef olarak izole edilmiş kaynak stem'ini kullanır
   > ve çıktı olarak yumuşak bir spektrogram maskesi üretir — ters faz dalgası
   > değil.)*

Also align the resample rate story: the draft says 16 kHz "MCU için" (para 146,
154) — keep 16 kHz, but the *current* motivation is the model's spectrogram
contract (n_fft=512, hop=128, 256×128 input), not MCU constraints. MCU framing is
future work.

---

## Part C — Section-by-section map

Walk these in order; most are small once Part B is done.

| Heading (≈para) | Action |
|---|---|
| Title / running headers (3–4, 73) | Title may stay (official TÜBİTAK project name). Ensure body frames Edge AI as future, not present. |
| ÖZET / ABSTRACT (73–87) | **P1.** Rewrite + fix date to Mayıs 2026. |
| KISALTMALAR (88–103) | Fine. Optionally add: SI-SDR, FiLM, U-Net, STFT, OLA, FP/TP. |
| GİRİŞ (106–117) | Add the bridge sentence from Part A: classification → separation → (future) edge ANC. |
| TEKNİK ARKA PLAN → Edge AI (118–132) | Keep, but add one line: "Edge AI optimizasyonu projenin sonraki fazıdır." |
| VERİ SETİ HAZIRLIK (133–164) | **P4.** LibriSpeech, FSD50K, phase-inversion note, resample motivation. |
| MODEL VE MİMARİ KARARLAR (166–181) | Frame 1D→2D/MobileNetV2 as System A history; point forward to the FiLM U-Net as the current model. |
| SİSTEM GERÇEKLEŞTİRİM (182–224) | **P2 + P3.** Separate System A metrics (84.38 %, 75.10 dB) from System B metrics (SI-SDRi, F1); add v2.3/v2.4. |
| Akustik Doğrulama (208–213) | **P2.** Mark 75.10 dB as digital upper bound. |
| Model Geliştirme / Nicel Bulgular (215–221) | **P3.** Add v2.3 actual results + iteration table. |
| Bulguların Proje Hedefleriyle İlişkisi (222–223) | **P2.** Break the "84.38 % + 75.10 dB ⇒ validated" chain; fix double period. |
| PROJE YÖNETİMİ → Mayıs (256–276) | Add v2.3/v2.4 to the May narrative; this is the most current work. |
| Sıradaki Adımlar (271–276) | Make sure it lists: train v2.4, re-evaluate, then (future) edge deployment. |
| Mevcut Sistem Sınırları / Gelecek (277–296) | **Part D.** Consolidate all edge/MCU/ANC as explicitly deferred. |
| Çalışma Takvimi (291–296) | Keep; mark MCU/quantization rows as next-phase. |
| KAYNAKLAR (297+) | Add refs you now cite: FiLM (Perez et al. 2018), SI-SDR (Le Roux et al. 2019), ESC-50, UrbanSound8K, FSD50K. |

---

## Part D — Consolidate the future-work story

Right now Edge AI / MCU / quantization / pruning / real-time ANC appear scattered
through the abstract, technical background, and goals sections as if they are part
of the current deliverable. Pull them into **one clearly-labeled future-work
block** under "Mevcut Sistem Sınırları ve Gelecek Dönem Hedefleri" (para 277).

Suggested Turkish framing paragraph:

> Bu rapor kapsamında geliştirilen web tabanlı seçici ses temizleme sistemi,
> projenin **ilk fazını** oluşturur. Aşağıdaki hedefler **sonraki faza**
> bırakılmıştır: (1) modelin 8-bit niceleme (quantization) ve yapısal budama
> (pruning) ile sıkıştırılarak mikrodenetleyici (MCU) üzerinde çalıştırılması,
> (2) mikrofon–hoparlör zincirinde <50 ms gecikmeli gerçek-zamanlı çıkarım,
> (3) dijital simülasyonda doğrulanan ters-faz sönümlemenin gerçek akustik
> ortamda (oda + donanım gecikmesi) doğrulanması. TÜBİTAK 2209-A başvurusundaki
> gömülü-sistem hedefleri bu fazı tanımlamaktadır.

This keeps the TÜBİTAK goals (good for motivation/scope) while making clear they
are not claimed as done.

---

## Part E — Minor fixes

- **Date:** "January, 2026" → "May, 2026 / Mayıs 2026" (para 86).
- **Double period:** para 223, "...değiştirmemektedir. . Zira..." → single period.
- **Class count consistency:** use "~56" for v2.0–v2.2 (ESC-50 + UrbanSound8K),
  "235" for v2.3 (FSD50K added), "budanmış (~56 + eşiği geçen FSD50K sınıfları)"
  for v2.4. Don't let an unqualified "56" sit next to the FSD50K discussion.
- **v2.0 claim:** the draft (para 217) correctly calls v2.0 a failure — keep that;
  it makes the iteration story credible.
- **Figures:** "Görsel 3.1/3.2" still depict the ANC phase-inversion pipeline. If
  you keep them, caption them as System A history; ideally add one spectrogram
  mask figure for System B.

---

## Suggested order of work

1. **P1** Abstract/Özet (sets the frame for everything else).
2. **P2** 75.10 dB + 84.38 % honesty (biggest credibility fix).
3. **P3** v2.3/v2.4 + iteration table (biggest scientific gain).
4. **P4** Dataset section accuracy.
5. **Part D** future-work consolidation.
6. **Part C** section sweep + **Part E** minor fixes.

When v2.4 finishes training, come back and fill its SI-SDRi/F1 into the Part B
table and the v2.4 paragraph.
