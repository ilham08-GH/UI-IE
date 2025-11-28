import streamlit as st
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from collections import Counter

# ==========================================
# 1. KONFIGURASI HALAMAN & TAMPILAN (CSS)
# ==========================================
st.set_page_config(
    page_title="Ekstraksi Informasi Pidana",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk Tampilan Hitam (Dark Mode Style)
st.markdown("""
<style>
    /* Kotak untuk Entitas (Background Hitam) */
    .entity-box {
        background-color: #000000; /* Hitam Pekat */
        color: #00FF41;            /* Hijau Neon */
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        display: inline-block;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        border: 1px solid #333;
        box-shadow: 2px 2px 0px #222;
    }
    
    /* Label Kecil di sebelah kata */
    .entity-label {
        font-size: 0.6em;
        color: #ddd;
        margin-left: 6px;
        background-color: #444;
        padding: 1px 4px;
        border-radius: 3px;
        text-transform: uppercase;
        vertical-align: middle;
    }

    /* Teks Biasa (Non-Entitas) */
    .normal-text {
        font-family: sans-serif;
        line-height: 2.0;
        color: #333;
        padding: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KONSTANTA
# ==========================================
MAX_LEN = 100       
VECTOR_SIZE = 10    

# ==========================================
# 3. FUNGSI LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('ner_bilstm_cbow.keras')
        cbow_model = Word2Vec.load("cbow_embedding.model")
        
        # Load Mapping Label
        try:
            with open('tag2idx.pkl', 'rb') as f:
                loaded_map = pickle.load(f)
            
            # Balik mapping {Label: Angka} -> {Angka: Label}
            sample_key = next(iter(loaded_map))
            if isinstance(sample_key, str): 
                idx2tag = {v: k for k, v in loaded_map.items()}
            else:
                idx2tag = loaded_map
        except:
            st.warning("⚠️ Menggunakan mapping default (alfabetis).")
            tags_sorted = sorted(['O', 'B_ADVO', 'B_ARTV', 'B_CRIA', 'B_DEFN', 'B_JUDG', 'B_JUDP', 
                                  'B_PENA', 'B_PROS', 'B_PUNI', 'B_REGI', 'B_TIMV', 'B_VERN', 
                                  'I_ADVO', 'I_ARTV', 'I_CRIA', 'I_DEFN', 'I_JUDG', 'I_JUDP', 
                                  'I_PENA', 'I_PROS', 'I_PUNI', 'I_REGI', 'I_TIMV', 'I_VERN'])
            idx2tag = {i: tag for i, tag in enumerate(tags_sorted)}
            
        return model, cbow_model, idx2tag

    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

model, cbow_model, idx2tag = load_resources()

# ==========================================
# 4. PREPROCESSING
# ==========================================
def clean_text(text):
    """Membersihkan teks untuk model (angka jadi <X>)"""
    text = text.lower()
    text = re.sub(r'\d+', '<X>', text)
    return text

def prepare_input(text, cbow_model, max_len):
    # 1. Simpan kata asli untuk display (sebelum diubah jadi <X>)
    original_tokens = text.split() # Tokenisasi spasi sederhana
    
    # 2. Proses teks untuk model
    cleaned_text = clean_text(text)
    model_tokens = cleaned_text.split()
    
    # Konversi ke Vektor
    X = []
    
    # Pastikan panjang original_tokens dan model_tokens sinkron untuk display
    # Kita pakai model_tokens untuk embedding loop
    
    for word in model_tokens:
        if word in cbow_model.wv:
            X.append(cbow_model.wv[word])
        else:
            X.append(np.zeros(cbow_model.vector_size))
            
    # Tambah dimensi batch
    X = [X]
    
    # Padding
    X_padded = pad_sequences(maxlen=max_len, sequences=X, padding="post", dtype='float32')
    
    return X_padded, model_tokens, original_tokens

# ==========================================
# 5. UI UTAMA
# ==========================================
st.title("⚖️ NER Putusan Pidana")

# --- SIDEBAR KALIBRASI (SOLUSI UTAMA MASALAH ANDA) ---
st.sidebar.header("🔧 Kalibrasi Label")
st.sidebar.info("Gunakan ini jika output salah (misal semua terdeteksi sebagai B_REGI).")

manual_o_index = st.sidebar.selectbox(
    "Index untuk Label 'O' (Bukan Entitas):",
    options=list(range(len(idx2tag))) if idx2tag else [],
    format_func=lambda x: f"Index {x} (Saat ini: {idx2tag.get(x, 'Unknown')})",
    help="Ubah ini sampai teks biasa tidak lagi memiliki kotak hitam."
)

# Input Area
default_text = "Menyatakan Terdakwa SABAR BIN SAWIJO terbukti secara sah melanggar Pasal 362 KUHP dan dijatuhi hukuman 2 tahun penjara."
input_text = st.text_area("Masukkan Teks Putusan:", height=100, value=default_text)

if st.button("🔍 Analisis Teks", type="primary"):
    if model:
        # A. Preprocessing
        X_new, model_tokens, original_tokens = prepare_input(input_text, cbow_model, MAX_LEN)
        
        # B. Prediksi
        y_pred = model.predict(X_new) 
        y_pred_indices = np.argmax(y_pred, axis=-1)[0]
        
        # --- LOGIKA AUTO-FIX MAPPING ---
        # Kita buat mapping sementara berdasarkan pilihan user di sidebar
        # Tujuannya: Memaksa Index yang dipilih user menjadi 'O'
        
        current_idx2tag = idx2tag.copy()
        
        # Cari label 'O' yang asli ada di index mana
        old_o_index = -1
        for k, v in current_idx2tag.items():
            if v == 'O':
                old_o_index = k
                break
        
        # Lakukan SWAP (Tukar Posisi)
        # Label di manual_o_index menjadi 'O'
        # Label 'O' yang lama pindah ke manual_o_index
        if manual_o_index in current_idx2tag:
            label_at_target = current_idx2tag[manual_o_index] # Misal: B_REGI
            
            # Tukar
            if old_o_index != -1:
                current_idx2tag[old_o_index] = label_at_target
            current_idx2tag[manual_o_index] = 'O'
            
        # --- VISUALISASI ---
        st.subheader("Hasil Ekstraksi:")
        
        # Hitung statistik index yang muncul
        detected_indices = y_pred_indices[:len(model_tokens)]
        counts = Counter(detected_indices)
        most_common_idx, _ = counts.most_common(1)[0]
        
        # Tampilkan saran kalibrasi jika 'O' bukan mayoritas
        if most_common_idx != manual_o_index:
            st.warning(f"⚠️ **Saran Kalibrasi:** Index `{most_common_idx}` muncul paling sering. Coba ganti sidebar 'Index untuk Label O' ke `{most_common_idx}` agar hasil lebih rapi.")
        
        html_output = "<div style='line-height: 2.5; background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #ddd;'>"
        
        found_entities = []
        
        # Loop token
        limit = min(len(model_tokens), MAX_LEN)
        for i in range(limit):
            tag_idx = y_pred_indices[i]
            
            # Gunakan mapping yang sudah dikalibrasi
            tag_label = current_idx2tag.get(tag_idx, "Unknown")
            
            # Tampilkan kata asli (bukan <X>) jika panjang array sama
            display_word = original_tokens[i] if i < len(original_tokens) else model_tokens[i]
            
            if tag_label != "O": 
                # Entitas (Kotak Hitam)
                html_output += f"<span class='entity-box'>{display_word}<span class='entity-label'>{tag_label}</span></span> "
                found_entities.append({"Kata": display_word, "Tipe Entitas": tag_label})
            else:
                # Teks Biasa
                html_output += f"<span class='normal-text'>{display_word} </span>"
        
        html_output += "</div>"
        
        st.markdown(html_output, unsafe_allow_html=True)
        
        if found_entities:
            st.write("### 📋 Tabel Entitas")
            import pandas as pd
            st.dataframe(pd.DataFrame(found_entities), use_container_width=True)
            
    else:
        st.error("Model gagal dimuat.")
