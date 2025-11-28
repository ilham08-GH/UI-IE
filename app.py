import streamlit as st
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
from collections import Counter

# ==========================================
# 1. KONFIGURASI HALAMAN & TAMPILAN
# ==========================================
st.set_page_config(
    page_title="Ekstraksi Informasi Pidana",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom (Background Hitam, Teks Hijau)
st.markdown("""
<style>
    .entity-box {
        background-color: #000000;
        color: #00FF41;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        display: inline-block;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        border: 1px solid #333;
        box-shadow: 2px 2px 0px #222;
    }
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
    .normal-text {
        font-family: sans-serif;
        line-height: 2.0;
        color: #333;
        padding: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KONSTANTA (WAJIB SAMA DENGAN TRAINING)
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
    text = text.lower()
    # Hapus tanda baca (titik, koma, dll) agar sesuai dengan Word2Vec
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Ganti angka dengan <X>
    text = re.sub(r'\d+', '<X>', text)
    return text

def prepare_input(text, cbow_model, max_len):
    # Token display (pertahankan tanda baca untuk visualisasi)
    original_tokens = text.split() 
    
    # Token model (bersih tanpa tanda baca)
    cleaned_text = clean_text(text)
    model_tokens = cleaned_text.split()
    
    X = []
    # Loop menggunakan model_tokens untuk lookup vector
    for word in model_tokens:
        if word in cbow_model.wv:
            X.append(cbow_model.wv[word])
        else:
            X.append(np.zeros(cbow_model.vector_size))
            
    X = [X]
    X_padded = pad_sequences(maxlen=max_len, sequences=X, padding="post", dtype='float32')
    
    return X_padded, model_tokens, original_tokens

# ==========================================
# 5. UI UTAMA
# ==========================================
st.title("⚖️ NER Putusan Pidana")

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("⚙️ Pengaturan")
auto_correct = st.sidebar.checkbox("✅ Auto-Correct Label 'O'", value=True, 
                                   help="Otomatis mendeteksi index yang paling sering muncul sebagai 'O' (Bukan Entitas).")

manual_o_index = 0
if not auto_correct:
    st.sidebar.warning("Mode Manual Aktif")
    manual_o_index = st.sidebar.number_input("Index Manual untuk 'O':", min_value=0, max_value=25, value=0)

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
        
        # C. LOGIKA AUTO-CORRECT MAPPING (The Fix!)
        current_idx2tag = idx2tag.copy()
        
        # Hitung statistik index yang muncul
        detected_indices = y_pred_indices[:len(model_tokens)]
        counts = Counter(detected_indices)
        
        if detected_indices.size > 0:
            most_common_idx, _ = counts.most_common(1)[0]
        else:
            most_common_idx = 0 # Default fallback
            
        target_o_index = most_common_idx if auto_correct else manual_o_index
        
        # --- SWAP MAPPING ---
        # Kita paksa index terbanyak (target_o_index) menjadi label 'O'
        # Cari label 'O' yang asli ada di index mana di mapping kita
        old_o_keys = [k for k, v in current_idx2tag.items() if v == 'O']
        old_o_index = old_o_keys[0] if old_o_keys else -1
        
        # Label apa yang saat ini menempati target_o_index? (Misal: B_REGI)
        label_at_target = current_idx2tag.get(target_o_index, "Unknown")
        
        # Lakukan Penukaran (Swap) jika targetnya bukan sudah 'O'
        if label_at_target != 'O':
            # 1. Pindahkan label yang tergusur ke tempat 'O' yang lama
            if old_o_index != -1:
                current_idx2tag[old_o_index] = label_at_target
            # 2. Set target index jadi 'O'
            current_idx2tag[target_o_index] = 'O'
            
            if auto_correct:
                st.success(f"🤖 Auto-Correct: Mendeteksi Index **{target_o_index}** sebagai 'O' (Bukan Entitas). Mapping dikoreksi otomatis.")

        # --- VISUALISASI ---
        st.subheader("Hasil Ekstraksi:")
        
        html_output = "<div style='line-height: 2.5; background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #ddd;'>"
        
        found_entities = []
        # Gunakan panjang terpendek antara tokens asli, tokens model, dan max_len untuk menghindari error
        limit = min(len(original_tokens), len(model_tokens), MAX_LEN)
        
        for i in range(limit):
            tag_idx = y_pred_indices[i]
            tag_label = current_idx2tag.get(tag_idx, "Unknown")
            
            # Tampilkan kata asli (yang masih ada tanda bacanya)
            display_word = original_tokens[i]
            
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
            st.info("Tidak ada entitas hukum spesifik yang ditemukan.")
            
    else:
        st.error("Model gagal dimuat.")
