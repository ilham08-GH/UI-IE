import streamlit as st
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Entitas Pidana (BiLSTM + CBOW)",
    layout="centered"
)

# --- Konstanta (Sesuai PDF Anda) ---
MAX_LEN = 100        # Halaman 2 PDF
VECTOR_SIZE = 10     # Halaman 38 PDF (vector_size=10)

# --- Fungsi Load Model & Resources ---
@st.cache_resource
def load_resources():
    # 1. Load Model BiLSTM
    try:
        model = tf.keras.models.load_model('ner_bilstm_cbow.keras')
    except Exception as e:
        st.error(f"Error memuat model .keras: {e}")
        return None, None, None

    # 2. Load Model Word2Vec (CBOW)
    try:
        cbow_model = Word2Vec.load("cbow_embedding.model")
    except Exception as e:
        st.error(f"Error memuat model Word2Vec: {e}")
        return None, None, None

    # 3. Load Mapping Label (tag2idx/idx2tag)
    try:
        # Kita butuh idx2tag (angka ke label)
        with open('tag2idx.pkl', 'rb') as f:
            idx2tag_loaded = pickle.load(f)
            
        # Cek apakah formatnya {label: angka} atau {angka: label}
        # Jika formatnya {label: angka}, kita balik dulu
        sample_key = next(iter(idx2tag_loaded))
        if isinstance(sample_key, str): 
            idx2tag = {v: k for k, v in idx2tag_loaded.items()}
        else:
            idx2tag = idx2tag_loaded
            
    except FileNotFoundError:
        st.warning("File tag2idx.pkl tidak ditemukan. Menggunakan mapping default (Mungkin tidak akurat).")
        # List tag default jika file tidak ada
        tags_list = ['O', 'B_ADVO', 'B_ARTV', 'B_CRIA', 'B_DEFN', 'B_JUDG', 'B_JUDP', 
                     'B_PENA', 'B_PROS', 'B_PUNI', 'B_REGI', 'B_TIMV', 'B_VERN', 
                     'I_ADVO', 'I_ARTV', 'I_CRIA', 'I_DEFN', 'I_JUDG', 'I_JUDP', 
                     'I_PENA', 'I_PROS', 'I_PUNI', 'I_REGI', 'I_TIMV', 'I_VERN']
        idx2tag = {i: tag for i, tag in enumerate(tags_list)}
        
    return model, cbow_model, idx2tag

# Memuat resources
model, cbow_model, idx2tag = load_resources()

# --- Fungsi Preprocessing (Sesuai PDF Halaman 24 & 39) ---
def get_word_embedding(word, cbow_model):
    """Mengambil vektor embedding untuk satu kata."""
    if word in cbow_model.wv:
        return cbow_model.wv[word]
    else:
        return np.zeros(cbow_model.vector_size)

def prepare_input(text, cbow_model, max_len):
    """
    Mengubah kalimat input menjadi matriks vektor embedding.
    Logika: Tokenize -> Get Embedding -> Padding
    """
    # 1. Tokenisasi sederhana (split spasi) & lowercase
    words = text.lower().split()
    
    # 2. Ubah kata jadi vektor
    # X shape: (1, jumlah_kata, vector_size)
    X = [[get_word_embedding(word, cbow_model) for word in words]]
    
    # 3. Padding agar panjangnya sama dengan MAX_LEN (100)
    # padding='post' artinya nol ditambahkan di belakang
    X_padded = pad_sequences(maxlen=max_len, sequences=X, padding="post", dtype='float32')
    
    return X_padded, words

# --- UI Aplikasi Utama ---
st.title("⚖️ Deteksi Entitas Pidana")
st.write("Model: BiLSTM + Word2Vec (CBOW)")

# Input Text Area
input_text = st.text_area("Masukkan teks putusan:", height=150, 
                          placeholder="Contoh: Terdakwa Sabar bin Sawijo terbukti melanggar Pasal 362 KUHP...")

if st.button("Prediksi"):
    if input_text and model and cbow_model:
        # 1. Preprocessing
        X_new, tokenized_words = prepare_input(input_text, cbow_model, MAX_LEN)
        
        # 2. Prediksi Model
        y_pred = model.predict(X_new)
        
        # 3. Ambil hasil argmax (index dengan probabilitas tertinggi)
        # y_pred shape: (1, 100, jumlah_tag) -> ambil batch pertama
        y_pred_indices = np.argmax(y_pred, axis=-1)[0]
        
        # 4. Tampilkan Hasil
        st.subheader("Hasil Ekstraksi:")
        
        # Format hasil agar enak dibaca
        results = []
        for i, word in enumerate(tokenized_words):
            if i < MAX_LEN:
                tag_index = y_pred_indices[i]
                tag_name = idx2tag.get(tag_index, "Unknown")
                
                # Hanya simpan jika bukan padding atau mapping valid
                results.append({"Kata": word, "Label": tag_name})

        # Tampilkan sebagai Dataframe berwarna
        import pandas as pd
        df = pd.DataFrame(results)
        
        # Mewarnai baris yang bukan 'O' (Other) agar entitas terlihat jelas
        def highlight_entity(row):
            if row['Label'] != 'O':
                return ['background-color: #fffdc1'] * len(row)
            else:
                return [''] * len(row)

        st.dataframe(df.style.apply(highlight_entity, axis=1), use_container_width=True)
        
        # Tampilan Raw Tags (Optional)
        with st.expander("Lihat Raw Tags"):
            st.write(results)

    elif not input_text:
        st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        st.error("Model belum siap. Cek file .keras, .model, dan .pkl Anda.")

# Footer info
st.markdown("---")
st.caption("Dideploy menggunakan Streamlit Community Cloud")
