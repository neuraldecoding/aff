# Kerangka Disertasi: Adaptive Feature Fusion for EEG-Based Neural Decoding

## 1. Pemilihan Judul dan Ruang Lingkup Penelitian

Bagian ini memvalidasi posisi penelitian dan membatasi *scope* agar fokus pada algoritma inti, bukan pada sistem *end-to-end*.

### Mengapa "Neural Decoding" dan bukan "BCI"?

| Parameter | Brain-Computer Interface (BCI) | Neural Decoding |
| --- | --- | --- |
| **Definisi** | Sistem lengkap: akuisisi → decoding → feedback | Proses interpretasi sinyal otak → intent/command |
| **Cakupan** | Hardware, software, user loop, aplikasi | Model, algoritma, representasi fitur |
| **Fokus** | End-to-end aplikasi | Komponen algoritmik |

**Kesimpulan:** Judul saat ini sudah tepat menggunakan *neural decoding*. Ini menunjukkan kepastian di *layer* mana kontribusi berada dan menghindari *overclaim* terhadap sistem BCI secara keseluruhan.

### Analisis Penggunaan Kata "Architecture"

| Parameter | Menggunakan "Architecture" | Tanpa "Architecture" |
| --- | --- | --- |
| **Beban Pembuktian** | Harus merancang sistem lengkap yang terdokumentasi | Cukup membuktikan mekanisme fusi adaptifnya |
| **Fleksibilitas** | Lebih rigid (engineering-heavy) | Seimbang antara metode dan aplikasi |

**Verdict:** Judul **"Adaptive Feature Fusion for EEG-Based Neural Decoding"** adalah pilihan terbaik. Tujuh kata yang padat, presisi, dan kuat tanpa ada kata yang sia-sia.

---

## 2. Definisi Kontribusi Utama

Kontribusi utama penelitian ini adalah **bagaimana** fitur-fitur EEG digabungkan, bukan sekadar menggunakan model yang sudah ada.

### Framing Novelty

* **Bukan Kontribusi Utama:** Metode ekstraksi fitur (CCA, HHT, LDA) atau *neural decoding model* itu sendiri (sudah ada).
* **Kontribusi Utama:** Mekanisme pembobotan dan penggabungan ketiga fitur tersebut secara **adaptif**.

### Hierarki Posisi Penelitian (Metode vs Algoritma vs Model)

Output kontribusi mencakup ketiganya, namun dengan hierarki yang jelas:

* **Kontribusi Utama (Metode):** *Adaptive feature fusion* yang mencakup prosedur dan justifikasi logis.
* **Formulasi (Algoritma):** Langkah adaptasi dan pembobotan fitur secara matematis.
* **Implementasi (Model):** Dilatih pada dataset EEG dan diuji performanya.

> **Argumentasi Sidang:**
> *"Kontribusi utama saya adalah sebuah **metode** adaptive feature fusion — yang secara operasional diformulasikan sebagai algoritma pembobotan adaptif, dan divalidasi melalui implementasi model yang diuji pada dataset EEG benchmark."*

*(Catatan Batasan: Siapkan jawaban untuk bab limitasi mengenai apakah metode fusi ini secara eksklusif hanya untuk sinyal EEG, atau dapat digeneralisasi ke fMRI/fNIRS).*

---

## 3. Implikasi pada Bab Metodologi

Dengan berbekal tinjauan literatur (*Systematic Literature Review*) mengenai model *State Space* modern, metodologi penelitian harus secara eksplisit menjawab matriks berikut:

| Pertanyaan Kunci | Elemen yang Harus Ada di Disertasi |
| --- | --- |
| **Adaptif terhadap apa?** | Variabilitas input signal, task, atau antar-subjek. |
| **Bagaimana adaptasinya?** | Penggunaan attention weights, gating, atau learned combination. |
| **Mengapa CCA + HHT + LDA?** | Justifikasi dari literatur dan SLR sebagai fondasi argumen. |
| **Seberapa baik hasilnya?** | Eksperimen komparatif melawan metode *static fusion*. |

---

## 4. Tiga Artefak Inti Penelitian

Ketiga artefak ini adalah "tulang punggung" operasional yang membuktikan kebaruan dari judul penelitian.

### Artefak 1: Definisi Formal Operator Fusi Adaptif

Fusi tidak menggunakan konkatenasi statis, melainkan pembobotan yang bergantung pada input (*input-dependent*). Diberikan representasi multi-domain $h_t$ (temporal), $h_s$ (spasial), dan $h_f$ (spektral).

Proses fusi adaptif menghasilkan representasi $z$ yang dirumuskan secara dinamis sebagai:

$$z = \sum_{m \in \{t,s,f\}} \alpha_m(x) \cdot g_m(h_m)$$

Dengan bobot $\alpha_m(x)$ dihitung per-sampel melalui mekanisme *gating* atau *cross-attention*, di mana total bobot harus memenuhi:

$$\sum_m \alpha_m = 1$$

### Artefak 2: Kesatuan Arsitektur Pipeline

Arsitektur usulan menyatukan 4 blok utama (Transformer, Diffusion, CLIP, Mamba) ke dalam satu *pipeline* pemrosesan yang terintegrasi.

```text
EEG ─┬─ Temporal encoder (Mamba/TCN) ─┐
     ├─ Spatial encoder (graph/attn)  ├─► [ADAPTIVE FUSION MODULE] ─► z
     └─ Spectral encoder (wavelet/PSD)┘     (gating + cross-attn)     │
                                                                      │
                                 ├─► Contrastive align (CLIP) → konsistensi semantik
                                 ├─► Classification head      → task klasifikasi
                                 └─► Diffusion decoder        → task rekonstruksi

```

### Artefak 3: Adaptasi Lintas-Subjek

Penerapan *Parameter-Efficient Fine-Tuning* (PEFT) atau *adapter* secara spesifik pada **FUSION MODULE**. Pendekatan ini bertujuan untuk merancang sebuah *"fusion foundation model"* yang secara konsisten tangguh mengatasi tingginya variabilitas data antar-subjek.
