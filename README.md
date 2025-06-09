# Spam Detection
Proyek deteksi spam berbasis teks yang dikembangkan oleh Tim SADAR (CC25-CF265) sebagai bagian dari fitur CekAjaDulu. Proyek ini menggunakan model deep learning untuk mengklasifikasikan teks sebagai **Spam** atau **Not Spam**, lengkap dengan API dan penjelasan model menggunakan LIME.

## Fitur
- Deteksi spam teks dengan model RNN.
- Penjelasan prediksi menggunakan [LIME (Local Interpretable Model-agnostic Explanations)](https://github.com/marcotcr/lime).
- FastAPI backend untuk REST API.
- Docker-ready untuk deployment.
- Preprocessing modular: cleaning, case folding, tokenizing, slang removal, dan stopword filtering.

## Sumber Data
Dataset berasal dari berbagai sumber publik, antara lain:
- [Data SMS Spam](https://gist.github.com/agtbaskara/a1a7017027cc1df9d35cf06e1e5575b7)
- Data sintesis yang dikompilasi dan disesuaikan sendiri oleh tim SADAR.

## Preprocessing
Langkah preprocessing meliputi:
- Cleaning: Menghapus simbol, angka, URL, dan whitespace.
- Case folding: Mengubah semua huruf ke huruf kecil
- Slang handling: Mengganti kata tidak baku dengan padanan formal.
- Tokenizing: Memisah teks menjadi kata-kata.
- Stopword removal: Menghapus kata tidak bermakna.

Semua disatukan dalam fungsi preprocess_text(text) di utils/spam_preprocess_text.py.

## Rule-Based Spam Filter
Sebelum masuk ke model machine learning, input teks akan terlebih dahulu dicek menggunakan aturan berbasis logika (rule-based), yang kami sebut sebagai Rule Breaker. Jika teks mengandung lebih dari 30% karakter non-alfabet, maka langsung dikategorikan sebagai SPAM, tanpa diproses oleh model. Jika teks tidak melanggar aturan, maka dilanjutkan ke model deep learning untuk diprediksi secara probabilistik.

## Model
Model yang digunakan adalah Recurrent Neural Network (RNN) yang dilatih dengan data SMS spam dan non-spam berbahasa Indonesia, menggunakan TF-IDF sebagai representasi fitur.

## Tools & Teknologi
- Python 3.10+
- FastAPI
- TensorFlow / Keras
- scikit-learn
- LIME
- Docker (optional)
- GitHub for version control
