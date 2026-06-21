# Adaptive Feature Fusion for EEG-Based Neural Decoding

Berikut adalah tiga artefak inti dari pendekatan ini:

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
