# Adaptive Feature Fusion for EEG-Based Neural Decoding


## Kontribusi utama yang terbaca dari judul ini

> *"Adaptive Feature Fusion for EEG-Based Neural Decoding"*

Kontribusi utamanya adalah **bagaimana** fitur-fitur EEG (dari CCA, HHT, LDA) digabungkan secara **adaptif** — bukan model decoding-nya itu sendiri.

### Framing kontribusi yang konsisten

```
Yang BUKAN kontribusi utamamu:
├── CCA → sudah ada
├── HHT → sudah ada  
├── LDA → sudah ada
└── Neural decoding model → sudah ada

Yang JADI kontribusi utamamu:
└── Mekanisme BAGAIMANA ketiga fitur itu
    digabungkan secara adaptif → INI NOVELTY-mu
```

### Implikasi ke isi disertasi

Dengan framing ini, bab metodologi harus menjawab dengan jelas:

| Pertanyaan | Yang harus ada di disertasi |
|---|---|
| Adaptif terhadap **apa**? | Input signal, task, atau subject variability |
| **Bagaimana** adaptasinya? | Attention weights, gating, learned combination |
| **Mengapa** CCA + HHT + LDA? | Justifikasi dari literatur — SLR-mu bisa jadi fondasinya |
| **Seberapa baik** dibanding static fusion? | Eksperimen komparatif |


### Kekuatan posisi ini

SLR-mu tentang Modern SSMs untuk BCI secara tidak langsung menjadi **justifikasi literatur** mengapa mekanisme fusion yang ada saat ini belum optimal — dan mengapa arsitektur adaptif yang kamu rancang mengisi gap tersebut.

SLR dan novel method-mu jadi **satu kesatuan narasi penelitian** yang kohesif.

## BCI lebih luas dari neural decoding.

### Hierarki konsepnya

```
Brain-Computer Interface (BCI)
├── Neural Decoding          ← komponen inti BCI
│   ├── Motor imagery
│   ├── SSVEP
│   ├── P300
│   └── Speech decoding
├── Signal acquisition
├── Feedback/output system
└── User interface
```

### Perbedaan scope

| | BCI | Neural Decoding |
|---|---|---|
| **Definisi** | Sistem lengkap: akuisisi → decoding → feedback | Khusus proses interpretasi sinyal otak → intent/command |
| **Mencakup** | Hardware, software, user loop, aplikasi | Model, algoritma, representasi fitur |
| **Contoh paper** | "A BCI system for wheelchair control" | "Decoding motor imagery from EEG using SSMs" |
| **Fokus** | End-to-end aplikasi | Komponen algoritmik |

### Implikasi untuk judulmu

Karena kontribusimu adalah **arsitektur fusion** — yang merupakan komponen algoritmik — maka:

- **Neural decoding** lebih presisi untuk scope kontribusimu
- **BCI** lebih tepat jika kamu juga membahas sistem lengkap sampai output/feedback

Jadi judul saat ini sudah tepat menggunakan *neural decoding* — itu justru menunjukkan kamu tahu persis di layer mana kontribusimu berada, bukan overclaim ke sistem BCI secara keseluruhan.

**Pertahankan versi sekarang.**

## Kontribusi utama: sebuah metode adaptive feature fusion

### Perbedaan algoritma vs model

| | Algoritma | Model |
|---|---|---|
| **Definisi** | Prosedur/langkah yang terdefinisi | Representasi matematis yang dilatih dari data |
| **Output** | Pseudocode, flowchart, prosedur | Bobot/parameter yang bisa digunakan untuk prediksi |
| **Diuji dengan** | Complexity analysis, correctness proof | Akurasi, F1, MSE pada dataset |
| **Contoh** | Quicksort, Viterbi, CCA | CNN, LSTM, Transformer |

### Jawaban yang tepat untuk judulmu

> *"Adaptive Feature Fusion for EEG-Based Neural Decoding"*

Output kontribusimu adalah **keduanya, tapi dengan hierarki:**

```
Kontribusi utama → METODE (fusion mechanism)
        │
        ├── Diformulasikan sebagai → ALGORITMA
        │   (langkah adaptasi, pembobotan fitur)
        │
        └── Diimplementasikan sebagai → MODEL
            (dilatih pada dataset EEG, diuji performanya)
```

### Jawaban tegas di sidang

> *"Kontribusi utama saya adalah sebuah **metode** adaptive feature fusion — yang secara operasional diformulasikan sebagai algoritma pembobotan adaptif, dan divalidasi melalui implementasi model yang diuji pada dataset EEG benchmark."*

### Yang membuat jawaban ini defensible

Kamu tidak perlu memilih salah satu — justru **metode** adalah kata yang paling tepat karena:

- Lebih luas dari algoritma (mencakup prosedur + justifikasi)
- Lebih tepat dari model (karena novelty-mu bukan di parameter-nya tapi di cara fusion-nya)
- Lazim digunakan di disertasi bidang signal processing dan BCI

### Satu hal yang perlu disiapkan

Penguji biasanya follow-up dengan:

> *"Apakah metode ini generalizable ke sinyal lain selain EEG?"*

Siapkan jawaban — apakah kamu mengklaim hanya untuk EEG, atau mekanisme fusion-nya bisa diaplikasikan ke fMRI/fNIRS juga. Batasannya perlu terdefinisi jelas di bab limitations.



## Berikut adalah tiga artefak inti dari pendekatan ini:

### 1. Definisi Formal Operator Fusi Adaptif

Fusi ini tidak menggunakan konkatenasi statis, melainkan pembobotan yang bergantung pada input (*input-dependent*). Diberikan representasi multi-domain:

* $h_t$ (Temporal)
* $h_s$ (Spasial)
* $h_f$ (Spektral)

Proses fusi adaptif menghasilkan representasi $z$ yang dirumuskan secara dinamis sebagai:

$$z = \sum_{m \in \{t,s,f\}} \alpha_m(x) \cdot g_m(h_m)$$

Dengan bobot $\alpha_m(x)$ dihitung per-sampel melalui mekanisme *gating* atau *cross-attention*, di mana total bobot harus memenuhi:

$$\sum_m \alpha_m = 1$$

> **Catatan Utama:** Inilah "tulang" yang membuat judul memiliki substansi. Persamaan matematis ini adalah komponen krusial yang sebelumnya belum ada dalam draf.

### 2. Kesatuan Arsitektur Pipeline

Sebuah arsitektur usulan yang menyatukan 4 blok utama (Transformer, Diffusion, CLIP, Mamba) menjadi satu *pipeline* terintegrasi — bukan lagi 4 deskripsi terpisah seperti pada Bab 3 versi lama.

```text
EEG ─┬─ Temporal encoder (Mamba/TCN) ─┐
     ├─ Spatial encoder (graph/attn)  ├─► [ADAPTIVE FUSION MODULE] ─► z
     └─ Spectral encoder (wavelet/PSD)┘     (gating + cross-attn)     │
                                                                      │
                                 ├─► Contrastive align (CLIP) → konsistensi semantik
                                 ├─► Classification head      → task klasifikasi
                                 └─► Diffusion decoder        → task rekonstruksi

```

### Perbandingan

| | Dengan "Architecture" | Tanpa "Architecture" |
|---|---|---|
| **Judul** | *Adaptive Feature Fusion **Architecture** for EEG-Based Neural Decoding* | *Adaptive Feature Fusion for EEG-Based Neural Decoding* |
| **Panjang** | 9 kata | 7 kata |
| **Beban pembuktian** | Harus merancang arsitektur lengkap yang terdokumentasi | Cukup membuktikan mekanisme fusion-nya adaptif |
| **Fleksibilitas scope** | Lebih rigid | Lebih fleksibel |
| **Kesan** | Engineering-heavy | Balanced antara metode dan aplikasi |

---

### Rekomendasi

**Gunakan tanpa "Architecture"** jika:
- Kontribusi utamamu adalah **mekanisme fusion**-nya (cara CCA/HHT/LDA digabungkan secara adaptif)
- Kamu tidak merancang sistem end-to-end dari scratch

**Pertahankan "Architecture"** jika:
- Kamu merancang **modul/sistem lengkap** dengan komponen yang terdefinisi jelas
- Ada diagram arsitektur yang jadi kontribusi tersendiri di disertasi

### Verdict

> **"Adaptive Feature Fusion for EEG-Based Neural Decoding"**

Judul ini **clean, precise, dan kuat**. Tujuh kata yang masing-masing bekerja — tidak ada kata yang sia-sia.

Ini pilihan yang sangat baik untuk judul disertasi.

### 3. Adaptasi Lintas-Subjek

Penerapan *Parameter-Efficient Fine-Tuning* (PEFT) atau *adapter* secara spesifik pada **FUSION MODULE**. Pendekatan ini bertujuan untuk membentuk sebuah *"fusion foundation model"* yang tangguh terhadap variasi data antar-subjek.
