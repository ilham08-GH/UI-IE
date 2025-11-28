import streamlit as st
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# ==========================================
# 1. KONFIGURASI HALAMAN & TAMPILAN (CSS)
# ==========================================
st.set_page_config(
    page_title="Ekstraksi Informasi Pidana",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk Tampilan Hitam (Dark Mode Style)
st.markdown("""
<style>
    /* Kotak untuk Entitas (Background Hitam) */
    .entity-box {
        background-color: #000000; /* Hitam Pekat */
        color: #00FF41;            /* Hijau Neon (Matrix Style) */
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        display: inline-block;
        font-family: 'Courier New', monospace; /* Font Terminal */
        font-weight: bold;
        border: 1px solid #333;    /* Garis tepi abu tua */
        box-shadow: 2px 2px 0px #222;
    }
    
    /* Label Kecil di sebelah kata (misal: B_PUNI) */
    .entity-label {
        font-size: 0.7em;
        color: #ffffff;            /* Putih */
        margin-left: 6px;
        background-color: #333333; /* Abu-abu untuk label tag */
        padding: 1px 4px;
        border-radius: 3px;
        text-transform: uppercase;
    }

    /* Teks Biasa (Non-Entitas) */
    .normal-text {
        font-family: sans-serif;
        line-height: 2.0;
        color: #333; /* Warna teks biasa (sesuaikan tema Streamlit user) */
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KONSTANTA (WAJIB SAMA DENGAN TRAINING)
# ==========================================
MAX_LEN = 100       # Panjang sequence maksimal
VECTOR_SIZE = 10    # Dimensi embedding CBOW

# ==========================================
# 3. FUNGSI LOAD RESOURCES (MODEL & MAPPING)
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # A. Load Model BiLSTM
        model = tf.keras.models.load_model('ner_bilstm_cbow.keras')
        
        # B. Load Model Word2Vec (CBOW)
        cbow_model = Word2Vec.load("cbow_embedding.model")
        
        # C. Load Mapping Label (tag2idx.pkl)
        # Kita perlu membalik mapping dari {Label: Angka} menjadi {Angka: Label}
        # agar bisa menerjemahkan prediksi model (angka) kembali ke teks.
        try:
            with open('tag2idx.pkl', 'rb') as f:
                loaded_map = pickle.load(f)
            
            # Deteksi format mapping dan balik kuncinya
            sample_key = next(iter(loaded_map))
            if isinstance(sample_key, str): 
                # Jika format aslinya {'B_ADVO': 0, ...}, kita balik jadi {0: 'B_ADVO', ...}
                idx2tag = {v: k for k, v in loaded_map.items()}
            else:
                idx2tag = loaded_map
        except:
            st.warning("⚠️ File tag2idx.pkl tidak ditemukan. Menggunakan mapping default (urutan alfabet).")
            # Fallback jika file hilang
            tags_sorted = sorted(['O', 'B_ADVO', 'B_ARTV', 'B_CRIA', 'B_DEFN', 'B_JUDG', 'B_JUDP', 
                                  'B_PENA', 'B_PROS', 'B_PUNI', 'B_REGI', 'B_TIMV', 'B_VERN', 
                                  'I_ADVO', 'I_ARTV', 'I_CRIA', 'I_DEFN', 'I_JUDG', 'I_JUDP', 
                                  'I_PENA', 'I_PROS', 'I_PUNI', 'I_REGI', 'I_TIMV', 'I_VERN'])
            idx2tag = {i: tag for i, tag in enumerate(tags_sorted)}
            
        return model, cbow_model, idx2tag

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None, None

# Load semua resource saat aplikasi mulai
model, cbow_model, idx2tag = load_resources()

# ==========================================
# 4. PREPROCESSING (PENTING!)
# ==========================================
def clean_text(text):
    """Membersihkan teks agar formatnya sama persis dengan saat training Word2Vec."""
    # 1. Lowercase (huruf kecil semua)
    text = text.lower()
    
    # 2. Ganti angka dengan token <X> (Sesuai PDF Hal 3)
    # Contoh: "Pasal 362" -> "pasal <X>"
    # Tanpa ini, model akan bingung melihat angka
    text = re.sub(r'\d+', '<X>', text)
    return text

def prepare_input(text, cbow_model, max_len):
    # Bersihkan teks
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    
    # Konversi Kata ke Vektor Embedding
    X = []
    processed_words = [] # Menyimpan kata asli untuk ditampilkan
    
    for word in words:
        if word in cbow_model.wv:
            X.append(cbow_model.wv[word])
        else:
            # Jika kata tidak dikenal (OOV), gunakan vektor nol
            X.append(np.zeros(cbow_model.vector_size))
        processed_words.append(word)
            
    # Tambahkan dimensi batch: (jumlah_kata, 10) -> (1, jumlah_kata, 10)
    X = [X]
    
    # Padding agar panjangnya pas 100
    X_padded = pad_sequences(maxlen=max_len, sequences=X, padding="post", dtype='float32')
    
    return X_padded, words

# ==========================================
# 5. USER INTERFACE (UI) UTAMA
# ==========================================
st.title("⚖️ NER Putusan Pidana")
st.markdown("""
Aplikasi ini mengekstrak entitas penting (Terdakwa, Hakim, Hukuman, dll) dari teks putusan pengadilan.
**Visualisasi:** Kotak Hitam = Entitas Terdeteksi.
""")

# Input Area
default_text = "Menyatakan Terdakwa SABAR BIN SAWIJO terbukti secara sah melanggar Pasal 362 KUHP dan dijatuhi hukuman 2 tahun penjara."
input_text = st.text_area("Masukkan Teks Putusan:", height=150, value=default_text)

# Tombol Prediksi
if st.button("🔍 Analisis Teks", type="primary"):
    if model is not None:
        # A. Preprocessing & Prediksi
        X_new, tokenized_words = prepare_input(input_text, cbow_model, MAX_LEN)
        
        # Prediksi Probabilitas
        y_pred = model.predict(X_new) 
        
        # Ambil kelas dengan probabilitas tertinggi (Argmax)
        y_pred_indices = np.argmax(y_pred, axis=-1)[0]
        
        st.subheader("Hasil Ekstraksi:")
        st.caption("Entitas yang terdeteksi ditandai dengan kotak hitam.")
        
        # B. Pembuatan Visualisasi HTML
        html_output = "<div style='line-height: 2.5; background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>"
        
        found_entities = []
        
        for i, word in enumerate(tokenized_words):
            if i < MAX_LEN:
                tag_idx = y_pred_indices[i]
                tag_label = idx2tag.get(tag_idx, "O") # Default 'O' jika index error
                
                # Kita kembalikan token <X> ke bentuk angka/kata asli jika memungkinkan
                # (Disini kita pakai kata hasil tokenisasi clean_text)
                display_word = word
                
                if tag_label != "O": # Jika BUKAN 'O' (Other), berarti Entitas
                    # Render Kotak Hitam
                    html_output += f"<span class='entity-box'>{display_word}<span class='entity-label'>{tag_label}</span></span> "
                    found_entities.append({"Kata": display_word, "Tipe Entitas": tag_label})
                else:
                    # Render Teks Biasa
                    html_output += f"<span class='normal-text'>{display_word} </span>"
        
        html_output += "</div>"
        
        # C. Tampilkan ke Layar
        st.markdown(html_output, unsafe_allow_html=True)
        
        # D. Tampilkan Tabel Ringkasan
        if found_entities:
            st.write("### 📋 Daftar Entitas")
            import pandas as pd
            df_res = pd.DataFrame(found_entities)
            st.dataframe(df_res, use_container_width=True)
        else:
            st.info("Tidak ada entitas hukum spesifik yang ditemukan dalam teks ini.")
            
    else:
        st.error("Model gagal dimuat. Pastikan file .keras, .model, dan .pkl sudah diupload ke GitHub.")

# Footer
st.markdown("---")
st.caption("Dikembangkan menggunakan BiLSTM + Word2Vec (CBOW)")
