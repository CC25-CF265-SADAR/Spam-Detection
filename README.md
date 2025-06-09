# Detektor Pesan Spam (Proyek CC25-CF265-SADAR)
Repositori ini berisi proyek deteksi spam berbasis teks yang dikembangkan oleh Tim SADAR (CC25-CF265) sebagai bagian dari fitur CekAjaDulu. Proyek ini menggunakan model _deep learning_ untuk mengklasifikasikan teks sebagai "Spam" atau "Not Spam". Proyek ini juga dilengkapi dengan API untuk kemudahan integrasi dan penjelasan model menggunakan LIME (Local Interpretable Model-agnostic Explanations) untuk transparansi.

## Daftar Isi
- [Fitur Utama](#fitur)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Alur Kerja](#alur-kerja)
- [Sumber Data](#sumber-data)
- [Setup dan Instalasi Lokal](#setup-dan-instalasi-lokal)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [Endpoint API](#endpoint-api)
- [Struktur Proyek](#struktur-proyek)
- [Deployment](#deployment)

## Fitur
- Deteksi spam teks dengan model RNN.
- Penjelasan prediksi menggunakan [LIME (Local Interpretable Model-agnostic Explanations)](https://github.com/marcotcr/lime) untuk memahami mengapa sebuah teks diklasifikasikan sebagai spam.
- FastAPI backend untuk REST API.
- Docker-ready untuk deployment.
- Preprocessing modular: cleaning, case folding, tokenizing, slang removal, dan stopword filtering.
- Rule-Based Filtering: Sistem filter awal untuk menyaring pesan dengan lebih dari 30% karakter non-alfabet.

## Teknologi yang Digunakan
- **Bahasa Pemrograman:** Python 3.10+
- **Framework API:** FastAPI
- **Machine Learning:** TensorFlow/Keras, Scikit-learn
- **Penjelasan Model:** LIME
- **Deployment (Opsional):** Docker, Railway
- **Version Control:** Git & GitHub

## Alur Kerja
1. **Input Teks:** Pengguna mengirimkan teks melalui endpoint API. 
2. **Rule-Based Filter:** Sistem akan melakukan pengecekan awal. Jika teks mengandung lebih dari 30% karakter non-alfabet, maka akan langsung diklasifikasikan sebagai SPAM. 
3. **Pra-pemrosesan:** Teks yang lolos dari filter akan melalui beberapa tahapan:
   - Pembersihan dari karakter yang tidak perlu.
   - Case folding (mengubah semua huruf menjadi huruf kecil).
   - Penghapusan kata-kata slang.
   - Tokenisasi.
   - Penghapusan stopword.
4. **Prediksi Model:** Teks yang sudah diproses akan dimasukkan ke dalam model RNN untuk klasifikasi. 
5. **Output:** Hasil prediksi ("Spam" atau "Not Spam").

## Sumber Data
Model ini dilatih dengan menggunakan dataset berikut:
- [Data SMS Spam](https://gist.github.com/agtbaskara/a1a7017027cc1df9d35cf06e1e5575b7)
- Data sintesis yang dikompilasi dan disesuaikan sendiri oleh tim SADAR.

## Setup dan Instalasi Lokal
Untuk menjalankan proyek ini di lingkungan lokal, ikuti langkah-langkah berikut. Pastikan Anda sudah memiliki Python 3.11+ dan Git terinstal.
### 1. Clone Repositori
```plaintext
git clone https://github.com/CC25-CF265-SADAR/Spam-Detection.git
cd Spam-Detection
```
### 2. Buat Lingkungan Virtual
Sangat disarankan untuk membuat lingkungan virtual agar dependensi proyek tidak bercampur dengan instalasi Python global.
```plaintext
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Instalasi Dependensi
Instal semua pustaka Python yang diperlukan yang tercantum dalam file requirements.txt.
```plaintext
pip install -r requirements.txt
```

## Menjalankan Aplikasi
Setelah semua dependensi terinstal, jalankan server aplikasi menggunakan Uvicorn.
### 1. Jalankan Server
Dari direktori utama proyek, jalankan perintah berikut di terminal:
```plaintext
uvicorn python_backend.spam_app:app --reload --port 8000
```
### 2. Akses Aplikasi
Setelah server berjalan, Anda dapat mengakses dokumentasi API interaktif (Swagger UI) melalui browser di alamat: http://127.0.0.1:8080/docs.

## Endpoint API 
Aplikasi menyediakan endpoint utama untuk melakukan prediksi pesan spam melalui metode POST.

Endpoint
URL: /predict
Metode: POST
Content-Type: application/json

API ini juga telah tersedia secara publik pada url https://spam-detection-sadar.up.railway.app

## Struktur Proyek
```plaintext

```

## Deployment
