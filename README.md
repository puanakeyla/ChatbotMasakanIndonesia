# Chatbot Asisten Memasak Indonesia 👨‍🍳

Chatbot asisten memasak berbasis **Retrieval-Augmented Generation (RAG)** dan **Large Language Model (LLM)** untuk resep masakan Indonesia.

## 📋 Deskripsi

Sistem ini mengintegrasikan:
- **Retrieval-Augmented Generation (RAG)** - Menggabungkan pencarian informasi dengan generasi bahasa alami
- **Vector Database (ChromaDB)** - Penyimpanan dan pencarian semantik resep
- **Large Language Model (Gemini 2.5)** - Generasi jawaban yang kontekstual dan akurat
- **Sentence Transformers** - Model embedding multilingual untuk bahasa Indonesia
- **Streamlit** - Interface web yang interaktif

## 🏗️ Arsitektur Sistem

```
┌─────────────┐
│   Pengguna  │
└──────┬──────┘
       │ Query
       ▼
┌─────────────────────────────────┐
│    Interface Chatbot (Web)      │
│         (Streamlit)              │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Pemrosesan Bahasa Alami       │
│   - Normalisasi                 │
│   - Text Embedding              │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│      Vector Store (ChromaDB)    │
│   - Semantic Search             │
│   - Retrieval Top-K             │
└──────────────┬──────────────────┘
               │ Retrieved Docs
               ▼
┌─────────────────────────────────┐
│    RAG Mechanism                │
│   - Context Enrichment          │
│   - Prompt Engineering          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   LLM (Gemini 2.5)        │
│   - Answer Generation           │
└──────────────┬──────────────────┘
               │ Response
               ▼
┌─────────────────────────────────┐
│        Pengguna                 │
└─────────────────────────────────┘
```

## 🚀 Fitur Utama

1. **Pencarian Semantik** - Mencari resep berdasarkan makna, bukan hanya kata kunci
2. **Respons Berbasis Data** - Jawaban berdasarkan resep asli Indonesia yang terkurasi
3. **Kontekstual** - Memahami konteks pertanyaan dan memberikan jawaban yang relevan
4. **Minimalisasi Hallucination** - Menggunakan RAG untuk memastikan akurasi informasi
5. **Interface Interaktif** - Web interface yang mudah digunakan

## 📦 Instalasi

### Prasyarat

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Gemini API Key

### Langkah Instalasi

1. **Clone atau download repository ini**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup environment variables**
   
   Copy file `.env.example` menjadi `.env`:
   ```bash
   copy .env.example .env
   ```
   
   Edit file `.env` dan tambahkan Gemini API key Anda:
   ```
   GEMINI_API_KEY=sk-your-api-key-here
   ```

4. **Setup database (Vector Store)**
   ```bash
   python setup_database.py
   ```
   
   Script ini akan:
   - Memuat data resep dari `data/resep_indonesia.json`
   - Memproses dan membersihkan data
   - Membuat embedding untuk setiap resep
   - Menyimpan ke ChromaDB vector store

5. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```
   
   Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📂 Struktur Proyek

```
Chatbot/
├── src/
│   ├── data_processor.py      # Preprocessing dan cleaning data
│   ├── embedding.py            # Text embedding dengan Sentence Transformers
│   ├── vector_store.py         # Manajemen ChromaDB vector store
│   ├── retriever.py            # Retrieval dokumen relevan
│   └── rag_chatbot.py          # RAG mechanism dan LLM integration
├── data/
│   └── resep_indonesia.json    # Dataset resep masakan Indonesia (15 resep)
├── chroma_db/                  # Vector database (generated)
├── app.py                      # Streamlit web application
├── setup_database.py           # Script setup database
├── requirements.txt            # Python dependencies
├── .env.example                # Template environment variables
├── .gitignore                  # Git ignore rules
└── README.md                   # Dokumentasi ini
```

## 🔧 Komponen Sistem

### 1. Data Processor (`data_processor.py`)
- Pembersihan data resep
- Normalisasi format bahan dan langkah
- Konversi resep ke format teks terstruktur

### 2. Embedding (`embedding.py`)
- Menggunakan Sentence Transformers
- Model: `paraphrase-multilingual-mpnet-base-v2`
- Mendukung bahasa Indonesia

### 3. Vector Store (`vector_store.py`)
- ChromaDB sebagai basis data vektor
- Penyimpanan embedding dan metadata
- Pencarian similarity dengan cosine distance

### 4. Retriever (`retriever.py`)
- Semantic search berdasarkan query
- Filter berdasarkan kategori
- Formatting context untuk LLM

### 5. RAG Chatbot (`rag_chatbot.py`)
- Integrasi retriever dan generator
- Prompt engineering untuk LLM
- Manajemen conversation history

### 6. Streamlit App (`app.py`)
- Interface web interaktif
- Chat interface
- Statistik dan visualisasi

## 📊 Dataset

Dataset berisi **15 resep masakan Indonesia** dengan kategori:
- Makanan Utama (Nasi Goreng, Rendang, Sate, dll)
- Makanan Berkuah (Soto, Rawon, Bakso, dll)
- Sayuran (Gado-gado, Sayur Asem)
- Makanan Tradisional (Gudeg, Pempek)
- Makanan Ringan (Martabak)

Setiap resep mencakup:
- Nama masakan
- Kategori
- Porsi
- Waktu memasak
- Tingkat kesulitan
- Bahan-bahan lengkap
- Langkah-langkah detail
- Tips memasak

## 💡 Cara Penggunaan

### Menjalankan Chatbot

1. Pastikan sudah menjalankan `setup_database.py`
2. Jalankan: `streamlit run app.py`
3. Buka browser di `http://localhost:8501`
4. Ketik pertanyaan tentang resep masakan Indonesia

### Contoh Pertanyaan

- "Bagaimana cara membuat nasi goreng yang enak?"
- "Apa saja bahan-bahan untuk membuat rendang?"
- "Berapa lama waktu memasak soto ayam?"
- "Tips agar bakso kenyal?"
- "Resep masakan Indonesia yang mudah dibuat?"

### Pengaturan

Di sidebar, Anda dapat:
- Mengatur jumlah resep yang diambil (Top-K)
- Toggle tampilan sumber resep
- Melihat statistik database
- Menghapus riwayat chat

## 🔑 Environment Variables

File `.env` berisi konfigurasi:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (dengan default values)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LLM_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7
VECTOR_STORE_TYPE=chroma
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=3
```

## 🧪 Testing Komponen Individual

Setiap modul dapat ditest secara independen:

```bash
# Test data processor
python src/data_processor.py

# Test embedding
python src/embedding.py

# Test vector store
python src/vector_store.py

# Test retriever
python src/retriever.py

# Test RAG chatbot
python src/rag_chatbot.py
```

## 📈 Metodologi RAG

### Tahapan Proses:

1. **Input Processing**
   - Pengguna memasukkan pertanyaan
   - Normalisasi dan cleaning teks

2. **Embedding**
   - Query diubah menjadi vektor
   - Menggunakan model yang sama dengan database

3. **Retrieval**
   - Pencarian semantic di vector store
   - Mengambil top-K dokumen paling relevan
   - Ranking berdasarkan cosine similarity

4. **Context Enrichment**
   - Dokumen hasil retrieval diformat
   - Digabungkan dengan query original
   - Membentuk prompt terstruktur

5. **Generation**
   - LLM menerima prompt + context
   - Generate jawaban berbasis data
   - Minimalisasi hallucination

6. **Response**
   - Jawaban ditampilkan ke pengguna
   - Include sumber resep (opsional)
   - Metadata retrieval

## 🎯 Keunggulan RAG vs Pure LLM

| Aspek | Pure LLM | RAG |
|-------|----------|-----|
| Akurasi | Rentan hallucination | Berbasis data real |
| Update Data | Perlu retraining | Tinggal update database |
| Transparansi | Black box | Dapat trace sumber |
| Cost | Tinggi (token besar) | Lebih efisien |
| Domain Specific | Terbatas | Sangat baik |

## 🛠️ Troubleshooting

### Error: "GEMINI_API_KEY tidak ditemukan"
- Pastikan file `.env` sudah dibuat
- Pastikan API key sudah diisi dengan benar
- Restart aplikasi setelah mengubah `.env`

### Error: "Vector store belum disetup"
- Jalankan `python setup_database.py` terlebih dahulu
- Pastikan folder `chroma_db` terbentuk

### Embedding terlalu lambat
- Model akan di-download saat pertama kali dijalankan
- Setelah download selesai, proses akan lebih cepat
- Ukuran model: ~400MB

### Response tidak relevan
- Coba tambah jumlah Top-K di sidebar
- Periksa kualitas data resep
- Pastikan query dalam bahasa Indonesia yang baik

## 📝 Menambah Data Resep

Untuk menambah resep baru:

1. Edit file `data/resep_indonesia.json`
2. Tambahkan resep dengan format yang sama
3. Jalankan ulang `python setup_database.py`
4. Pilih opsi untuk reload database

Format resep:
```json
{
  "nama": "Nama Masakan",
  "kategori": "Kategori",
  "porsi": "X porsi",
  "waktu_masak": "X menit/jam",
  "tingkat_kesulitan": "Mudah/Sedang/Sulit",
  "bahan": ["bahan 1", "bahan 2"],
  "langkah": ["langkah 1", "langkah 2"],
  "tips": "Tips memasak"
}
```

## 🔄 Update Model

Untuk menggunakan model LLM atau embedding berbeda:

1. Edit file `.env`:
   ```env
   LLM_MODEL=gpt-4  # atau model lain
   EMBEDDING_MODEL=nama-model-lain
   ```

2. Restart aplikasi

## 📚 Referensi

### Libraries
- [LangChain](https://python.langchain.com/) - Framework untuk LLM applications
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [Streamlit](https://streamlit.io/) - Web framework
- [OpenAI API](https://platform.openai.com/) - LLM provider

### Papers
- RAG: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Sentence Transformers: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

## 👥 Kontributor

Proyek ini dibuat sebagai implementasi sistem chatbot berbasis RAG untuk Kapita Selekta.

## 📄 Lisensi

Proyek ini dibuat untuk keperluan edukasi dan penelitian.

## 🙏 Acknowledgments

- Data resep dikumpulkan dari berbagai sumber kuliner Indonesia
- Menggunakan open-source libraries dan frameworks
- OpenAI untuk LLM API

---

**Selamat mencoba! 🎉**

Jika ada pertanyaan atau issues, silakan buka issue di repository ini.
