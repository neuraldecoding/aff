# Adaptive Feature Fusion for EEG-Based Neural Decoding


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

### 3. Adaptasi Lintas-Subjek

Penerapan *Parameter-Efficient Fine-Tuning* (PEFT) atau *adapter* secara spesifik pada **FUSION MODULE**. Pendekatan ini bertujuan untuk membentuk sebuah *"fusion foundation model"* yang tangguh terhadap variasi data antar-subjek.
