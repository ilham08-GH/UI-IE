import streamlit as st
import joblib
from annotated_text import annotated_text
import warnings

# Mengabaikan warning dari joblib saat memuat model sklearn
warnings.filterwarnings("ignore", category=UserWarning)

# ====================================================================
# 1. FUNGSI HELPER (Disalin dari Notebook Anda)
# Kita salin fungsi feature engineering CRF Anda [cite: 602-680]
# ====================================================================

def word2features(sent, i):
    word = sent[i][0]
    prev = sent[i][1]
    nextt = sent[i][2]
    
    # Fitur-fitur ini persis seperti di notebook Anda [cite: 608-634]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'prev.lower()': prev.lower(),
        'prev[-3:]': prev[-3:],
        'prev[-2:]': prev[-2:],
        'prev.isupper()': prev.isupper(),
        'prev.istitle()': prev.istitle(),
        'prev.isdigit()': prev.isdigit(),
        'nextt.lower()': nextt.lower(),
        'nextt[-3:]': nextt[-3:],
        'nextt[-2:]': nextt[-2:],
        'nextt.isupper()': nextt.isupper(),
        'nextt.istitle()': nextt.istitle(),
        'nextt.isdigit()': nextt.isdigit(),
    }
    
    # Fitur untuk kata sebelumnya [cite: 635-652]
    if i > 0:
        word1 = sent[i-1][0]
        prev1 = sent[i-1][1]
        nextt1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:prev.lower()': prev1.lower(),
            '-1:prev.istitle()': prev1.istitle(),
            '-1:prev.isupper()': prev1.isupper(),
            '-1:nextt.lower()': nextt1.lower(),
            '-1:nextt.istitle()': nextt1.istitle(),
            '-1:nextt.isupper()': nextt1.isupper(),
        })
    else:
        # Menandakan Awal Kalimat (Beginning of Sentence) [cite: 653]
        features['BOS'] = True
        
    # Fitur untuk kata sesudahnya [cite: 654-672]
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        prev1 = sent[i+1][1]
        nextt1 = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:prev.lower()': prev1.lower(),
            '+1:prev.istitle()': prev1.istitle(),
            '+1:prev.isupper()': prev1.isupper(),
            '+1:nextt.lower()': nextt1.lower(),
            '+1:nextt.istitle()': nextt1.istitle(),
            '+1:nextt.isupper()': nextt1.isupper(),
        })
    else:
        # Menandakan Akhir Kalimat (End of Sentence) [cite: 673]
        features['EOS'] = True
        
    return features

# Fungsi ini juga disalin dari notebook Anda [cite: 675-677]
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# ====================================================================
# 2. FUNGSI PREPROCESSING BARU
# Fungsi ini adalah "lem" yang mengubah teks mentah dari user
# menjadi format yang dipahami `word2features`.
# ====================================================================

def format_raw_text(raw_text):
    """
    Mengubah teks input mentah (string) menjadi format list of tuples 
    (word, prev, nextt, dummy_label) yang dibutuhkan oleh `sent2features`.
    """
    words = raw_text.split()
    sentence_tuples = []
    
    for i in range(len(words)):
        word = words[i]
        # Logika `prev` dan `next` sederhana, bisa disesuaikan jika Anda punya
        # logika yang lebih kompleks saat membuat CSV.
        prev = words[i-1] if i > 0 else ""
        nextt = words[i+1] if i < len(words) - 1 else ""
        
        # 'O' adalah dummy label. Tidak digunakan saat prediksi,
        # tapi dibutuhkan agar struktur datanya sama.
        sentence_tuples.append((word, prev, nextt, 'O')) 
        
    # Model CRF Anda di-train pada *list of sentences* [cite: 681, 707]
    # Jadi kita bungkus dia dalam list, seolah-olah ini adalah 1 kalimat.
    return [sentence_tuples]

# ====================================================================
# 3. FUNGSI UTAMA APLIKASI STREAMLIT
# ====================================================================

# @st.cache_resource digunakan agar model hanya di-load sekali
@st.cache_resource
def load_model():
    """Memuat model CRF yang sudah disimpan."""
    try:
        model = joblib.load('crf_model.joblib')
        return model
    except FileNotFoundError:
        return None

# ----- Tampilan UI -----
st.set_page_config(page_title="Ekstraksi Informasi", layout="wide")
st.title("🔎 Aplikasi Ekstraksi Informasi Teks Hukum")
st.markdown("Aplikasi ini menggunakan model **CRF** untuk mengekstrak entitas dari teks.")

# Memuat model
crf = load_model()

if crf is None:
    st.error("File model 'crf_model.joblib' tidak ditemukan. Pastikan Anda sudah menyimpan model dari notebook.")
else:
    # ----- Area Input -----
    st.header("Masukkan Teks untuk Dianalisis")
    
    # Contoh teks dari data Anda untuk memudahkan [cite: 97]
    default_text = "Pwd Sabar Bin Sawijo TUNTUTAN Noviana, S.H. Terdakwa menghadap sendiri"
    text_input = st.text_area("Teks:", value=default_text, height=150,
                              help="Masukkan teks putusan atau dokumen hukum di sini.")

    # ----- Tombol Proses -----
    if st.button("🚀 Proses Teks", type="primary"):
        if text_input.strip():
            
            # 1. Ubah teks mentah jadi format yang benar
            sentence_data = format_raw_text(text_input)
            
            # 2. Ekstrak fitur (menggunakan fungsi dari notebook) [cite: 681]
            X_test_features = [sent2features(s) for s in sentence_data]
            
            # 3. Lakukan prediksi
            y_pred_tags = crf.predict(X_test_features)
            
            # y_pred_tags akan berbentuk [[tag1, tag2, ...]]
            # Kita ambil hasil prediksi untuk kalimat pertama (indeks 0)
            tags = y_pred_tags[0]
            words = text_input.split()

            # 4. Tampilkan hasil dengan highlight
            st.header("Hasil Ekstraksi (Named Entity Recognition)")
            
            if len(words) == len(tags):
                # Siapkan data untuk `annotated_text`
                annotated_result = []
                for word, tag in zip(words, tags):
                    # Tag '0' adalah "Other" (bukan entitas), kita biarkan 
                    if tag != '0':
                        # Tambahkan sebagai tuple (kata, tag)
                        annotated_result.append((word, tag))
                    else:
                        # Tambahkan sebagai string biasa
                        annotated_result.append(word + " ")
                
                # Tampilkan!
                annotated_text(*annotated_result)
            
            else:
                st.error("Terjadi kesalahan: Jumlah kata dan tag hasil prediksi tidak cocok.")
                st.write("Words:", words)
                st.write("Tags:", tags)

        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")
