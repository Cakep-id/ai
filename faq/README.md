# FAQ NLP System

Sistem FAQ berbasis Natural Language Processing dengan Python Flask, MySQL, dan TF-IDF untuk pencarian pertanyaan yang mirip.

## 🚀 Fitur

### Database MySQL
- **Tabel `faq_dataset`**: Menyimpan pertanyaan utama dan jawaban
- **Tabel `faq_variations`**: Menyimpan variasi pertanyaan dengan similarity score
- **Tabel `search_logs`**: Menyimpan log pencarian untuk analisis
- **Full-text search** dan indexing untuk performa optimal
- **Timestamp tracking** untuk semua data

### Backend Python (Flask)
- **RESTful API** untuk semua operasi CRUD
- **NLP Processing** menggunakan TF-IDF dan cosine similarity
- **Preprocessing text** dengan tokenization, stopwords removal, dan stemming
- **Similarity matching** dengan threshold yang dapat disesuaikan
- **Search logging** untuk tracking dan analisis
- **Error handling** yang komprehensif

### Frontend Web (Vanilla JS)
- **Interface responsif** tanpa framework eksternal
- **Form input** untuk menambah FAQ dan variasi
- **Pencarian real-time** dengan similarity threshold
- **Management panel** untuk CRUD operations
- **Dashboard statistik** pencarian
- **Toast notifications** untuk user feedback

### NLP Capabilities
- **Text preprocessing** dengan support bahasa Indonesia
- **TF-IDF vectorization** untuk feature extraction
- **Cosine similarity** untuk matching pertanyaan
- **Multi-language stopwords** (Indonesia/English)
- **Fallback ke string matching** jika TF-IDF gagal

## 📋 Persyaratan

### Software
- Python 3.8+
- MySQL 5.7+ atau MariaDB 10.3+
- Web browser modern

### Python Packages
```
Flask==2.3.3
mysql-connector-python==8.1.0
SQLAlchemy==2.0.21
Flask-SQLAlchemy==3.0.5
python-dotenv==1.0.0
scikit-learn==1.3.0
numpy==1.24.3
pandas==2.0.3
nltk==3.8.1
Flask-CORS==4.0.0
```

## 🛠️ Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd faq-nlp-system
```

### 2. Setup Virtual Environment
```bash
# Windows
setup.bat

# Manual setup
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. Konfigurasi Database
Buat database MySQL:
```sql
CREATE DATABASE faq_nlp_system;
```

Edit file `.env`:
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=faq_nlp_system

# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your_secret_key_here

# NLP Configuration
MIN_SIMILARITY_THRESHOLD=0.5
MAX_RESULTS=10
DEFAULT_SEARCH_METHOD=tfidf

# Server Configuration
HOST=127.0.0.1
PORT=5000
```

### 4. Initialize Database
```bash
python database/init_db.py
```

### 5. Jalankan Aplikasi
```bash
# Windows
run.bat

# Manual
python app.py
```

Akses aplikasi di: http://127.0.0.1:5000

## 📁 Struktur Proyek

```
faq/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables
├── setup.bat                  # Windows setup script
├── run.bat                    # Windows run script
├── README.md                  # Documentation
│
├── database/
│   ├── __init__.py
│   ├── connection.py          # Database models & connection
│   ├── schema.sql            # Database schema
│   └── init_db.py            # Database initialization
│
├── services/
│   ├── __init__.py
│   └── faq_service.py        # Business logic service
│
├── nlp/
│   ├── __init__.py
│   └── processor.py          # NLP processing module
│
└── frontend/
    ├── index.html            # Main HTML template
    └── static/
        ├── css/
        │   └── style.css     # Styling
        └── js/
            └── app.js        # Frontend JavaScript
```

## 🔌 API Endpoints

### Health Check
- `GET /api/health` - Status aplikasi dan database

### FAQ Management
- `POST /api/faq` - Tambah FAQ baru
- `GET /api/faq` - Get semua FAQ
- `GET /api/faq/{id}` - Get FAQ by ID
- `PUT /api/faq/{id}` - Update FAQ
- `DELETE /api/faq/{id}` - Delete FAQ

### Search
- `POST /api/search` - Cari FAQ dengan NLP
- `GET /api/search?q={query}` - Cari via GET parameter

### Utilities
- `GET /api/categories` - Get semua kategori
- `GET /api/statistics` - Get statistik pencarian
- `POST /api/train` - Trigger model training

### Contoh Request

#### Tambah FAQ
```json
POST /api/faq
{
    "question": "Bagaimana cara reset password?",
    "answer": "Klik 'Lupa Password' dan ikuti instruksi email.",
    "category": "account",
    "variations": [
        "Cara mengubah password",
        "Lupa kata sandi",
        "Reset kata sandi"
    ]
}
```

#### Search FAQ
```json
POST /api/search
{
    "query": "lupa password",
    "threshold": 0.3,
    "max_results": 10
}
```

#### Response Format
```json
{
    "success": true,
    "message": "Operation successful",
    "data": {...},
    "results": [...],
    "total": 5
}
```

## 🧠 NLP Features

### Text Preprocessing
- **Lowercasing**: Konversi ke huruf kecil
- **Punctuation removal**: Hapus tanda baca
- **Tokenization**: Pemisahan kata menggunakan NLTK
- **Stopwords removal**: Hapus kata umum (Bahasa Indonesia)
- **Basic stemming**: Untuk Bahasa Indonesia (suffix removal)

### Similarity Calculation
- **TF-IDF Vectorization**: Convert text ke numerical features
- **Cosine Similarity**: Hitung kemiripan antar vektor
- **Threshold-based filtering**: Filter hasil berdasarkan similarity score
- **Fallback mechanism**: Simple string matching jika TF-IDF gagal

### Search Process
1. **Query preprocessing**: Bersihkan dan normalize query
2. **Document retrieval**: Ambil semua FAQ dan variasi
3. **Vectorization**: Convert query dan documents ke TF-IDF vectors
4. **Similarity calculation**: Hitung cosine similarity
5. **Ranking**: Urutkan berdasarkan similarity score
6. **Filtering**: Filter berdasarkan threshold
7. **Logging**: Simpan search log untuk analisis

## 📊 Database Schema

### faq_dataset
```sql
CREATE TABLE faq_dataset (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'general',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_category (category),
    INDEX idx_created_at (created_at),
    FULLTEXT(question, answer)
);
```

### faq_variations
```sql
CREATE TABLE faq_variations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    faq_id INT NOT NULL,
    variation_question TEXT NOT NULL,
    similarity_score DECIMAL(5,4) DEFAULT 1.0000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (faq_id) REFERENCES faq_dataset(id) ON DELETE CASCADE,
    INDEX idx_faq_id (faq_id),
    INDEX idx_similarity (similarity_score),
    FULLTEXT(variation_question)
);
```

### search_logs
```sql
CREATE TABLE search_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    search_query TEXT NOT NULL,
    result_faq_id INT,
    similarity_score DECIMAL(5,4),
    user_ip VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (result_faq_id) REFERENCES faq_dataset(id) ON DELETE SET NULL,
    INDEX idx_created_at (created_at),
    INDEX idx_result_faq_id (result_faq_id)
);
```

## 🎯 Penggunaan

### 1. Tambah FAQ
- Buka tab "Tambah FAQ"
- Isi pertanyaan utama dan jawaban
- Pilih kategori
- Tambahkan variasi pertanyaan (opsional)
- Klik "Simpan FAQ"

### 2. Pencarian FAQ
- Buka tab "Pencarian FAQ"
- Ketik pertanyaan di search box
- Sesuaikan threshold similarity (0.1 - 1.0)
- Klik "Cari" atau tekan Enter
- Lihat hasil dengan similarity score

### 3. Kelola FAQ
- Buka tab "Kelola FAQ"
- Lihat daftar semua FAQ
- Filter berdasarkan kategori
- Edit atau hapus FAQ yang ada
- Lihat variasi pertanyaan

### 4. Statistik
- Buka tab "Statistik"
- Lihat total pencarian dan success rate
- Monitor pencarian terbaru
- Analisis FAQ populer

## 🔧 Konfigurasi

### Similarity Threshold
- **0.1 - 0.3**: Hasil lebih banyak, tapi mungkin kurang relevan
- **0.3 - 0.5**: Balance antara recall dan precision
- **0.5 - 0.8**: Hasil lebih akurat, tapi mungkin sedikit
- **0.8 - 1.0**: Hanya exact match atau very similar

### NLP Parameters
Edit di `nlp/processor.py`:
```python
self.vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=list(self.stop_words),
    ngram_range=(1, 2),        # Unigram + Bigram
    max_features=5000          # Max vocabulary size
)
```

## 🐛 Troubleshooting

### Database Connection Error
```bash
# Check MySQL service
net start mysql

# Test connection
mysql -u root -p
```

### NLTK Data Error
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Port Already in Use
```bash
# Change port in .env
PORT=5001

# Or kill process
netstat -ano | findstr :5000
taskkill /PID <process_id> /F
```

### Performance Issues
- **Add database indexes** untuk query yang sering
- **Optimize TF-IDF parameters** untuk dataset besar
- **Implement caching** untuk frequent queries
- **Use connection pooling** untuk high traffic

## 📈 Pengembangan Lanjutan

### Improvements
1. **Vector Database**: Migrate ke Pinecone/Weaviate untuk better performance
2. **Advanced NLP**: Integrate BERT/Sentence Transformers
3. **Caching**: Implement Redis untuk search results
4. **API Rate Limiting**: Add throttling untuk production
5. **User Authentication**: Add user management system
6. **Analytics Dashboard**: Advanced search analytics
7. **Export/Import**: Bulk FAQ management features
8. **Multi-language**: Support multiple languages
9. **Auto-categorization**: ML-based category prediction
10. **Feedback System**: User feedback untuk improve accuracy

### Production Deployment
1. **Use production WSGI server** (Gunicorn/uWSGI)
2. **Setup reverse proxy** (Nginx)
3. **Implement SSL/HTTPS**
4. **Database optimization** dan backup strategy
5. **Monitoring** dengan logging dan metrics
6. **Docker containerization**
7. **CI/CD pipeline** setup

## 📄 License

MIT License - Feel free to use and modify for your projects.

## 👥 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

**FAQ NLP System** - Intelligent FAQ management with Natural Language Processing capabilities.
