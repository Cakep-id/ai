# AgentV2 - Advanced AI Asset Inspection System

## Overview

AgentV2 adalah sistem inspeksi aset berbasis AI yang canggih dengan arsitektur 3-role (User, Admin, Trainer) yang menyediakan evaluasi YOLO tingkat lanjut, perhitungan risiko berbasis geometri, dan pembelajaran yang dipandu manusia.

## Fitur Utama

### 🔍 Deteksi AI Canggih
- **YOLO Advanced Evaluation**: mAP@0.5, mAP@[.5:.95], F1 scores, confusion matrices
- **Temperature Scaling**: Kalibrasi confidence untuk prediksi yang lebih akurat
- **Monte Carlo Dropout**: Quantifikasi uncertainty untuk estimasi keandalan
- **Per-Class Performance**: Tracking performa detail per kategori kerusakan

### 📊 Risk Assessment Berbasis Geometri
- **Physics-Based Damage Assessment**: Perhitungan stress concentration factors
- **Multi-Dimensional Risk Analysis**: Area, aspect ratio, proximity clustering
- **Automated Maintenance Scheduling**: Prioritas berdasarkan tingkat risiko
- **Real-time Risk Monitoring**: Dashboard pemantauan risiko secara real-time

### 🎓 Human-Driven Learning
- **Interactive Training Interface**: Antarmuka trainer untuk upload dan anotasi data
- **Real-time Training Monitoring**: Progress tracking dengan metrics visualization
- **Advanced Data Augmentation**: Augmentasi adaptif untuk peningkatan performa
- **Model Evaluation & Deployment**: Sistem evaluasi komprehensif dan deployment otomatis

### 🏗️ Arsitektur 3-Role
- **User Interface**: Submit inspeksi, lihat hasil, download laporan
- **Admin Dashboard**: Validasi laporan, maintenance scheduling, system monitoring
- **Trainer Platform**: Data management, model training, performance evaluation

## Struktur Proyek

```
agentv2/
├── backend/
│   ├── main.py                 # FastAPI backend utama
│   ├── db_manager.py          # Database connection & management
│   ├── yolo_service.py        # Advanced YOLO service
│   ├── risk_engine.py         # Geometry-based risk assessment
│   ├── training_service.py    # AI training management
│   └── evaluation_service.py  # Model evaluation & metrics
├── frontend/
│   ├── user.html             # User interface
│   ├── admin.html            # Admin dashboard
│   └── trainer.html          # Trainer platform
├── database/
│   └── schema.sql            # Advanced database schema
├── models/                   # AI models storage
├── uploads/                  # File uploads
├── logs/                     # System logs
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Instalasi

### Prerequisites
- Python 3.9+
- MySQL 8.0+
- CUDA (opsional, untuk GPU acceleration)

### Setup Database
1. Buat database MySQL:
```sql
CREATE DATABASE agentv2_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. Import schema:
```bash
mysql -u root -p agentv2_db < database/schema.sql
```

### Setup Python Environment
1. Buat virtual environment:
```bash
python -m venv agentv2_env
source agentv2_env/bin/activate  # Linux/Mac
# atau
agentv2_env\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLO model:
```bash
# Model akan otomatis didownload saat pertama kali dijalankan
# Atau download manual ke folder models/
```

### Configuration
1. Copy dan edit file konfigurasi:
```bash
cp config.py.example config.py
```

2. Set environment variables:
```bash
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=your_password
export DB_NAME=agentv2_db
export DEBUG=false
```

## Menjalankan Sistem

### 1. Start Backend API
```bash
python backend/main.py
```
Backend akan berjalan di `http://localhost:8000`

### 2. Akses Frontend
- **User Interface**: `http://localhost:8000/user.html`
- **Admin Dashboard**: `http://localhost:8000/admin.html`
- **Trainer Platform**: `http://localhost:8000/trainer.html`

### 3. API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Penggunaan

### User Workflow
1. **Upload Gambar**: Drag-drop atau pilih gambar aset
2. **Isi Form**: Deskripsi aset, lokasi, kondisi
3. **Submit Inspeksi**: Sistem akan memproses otomatis
4. **Lihat Hasil**: Real-time progress dan hasil deteksi
5. **Download Laporan**: PDF report dengan analisis lengkap

### Admin Workflow
1. **Dashboard Overview**: Monitor sistem dan statistik
2. **Validation Queue**: Review dan validasi laporan
3. **Maintenance Schedule**: Kelola jadwal pemeliharaan
4. **System Settings**: Konfigurasi sistem dan threshold

### Trainer Workflow
1. **Data Upload**: Upload gambar training baru
2. **Data Annotation**: Annotate gambar untuk training
3. **Model Training**: Konfigurasi dan start training session
4. **Evaluation**: Analisis performa model
5. **Deployment**: Deploy model terbaik ke production

## API Endpoints

### User Endpoints
- `POST /api/user/submit` - Submit inspection
- `GET /api/user/report/{report_id}` - Get report details
- `GET /api/user/reports` - List user reports

### Admin Endpoints
- `GET /api/admin/dashboard` - Dashboard statistics
- `POST /api/admin/validate/{report_id}` - Validate report
- `GET /api/admin/maintenance/schedule` - Maintenance schedule

### Trainer Endpoints
- `POST /api/trainer/upload` - Upload training data
- `POST /api/trainer/train` - Start training session
- `GET /api/trainer/sessions` - Training history

## Advanced Features

### Model Evaluation Metrics
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@[.5:.95]**: Mean Average Precision across IoU thresholds
- **Precision/Recall Curves**: Per-class performance analysis
- **Confusion Matrix**: Classification accuracy visualization
- **Calibration Analysis**: Confidence reliability assessment

### Risk Assessment Algoritma
```python
# Geometry-based severity calculation
severity = (
    damage_type_factor * 0.3 +
    size_factor * 0.25 +
    geometry_factor * 0.2 +
    clustering_factor * 0.15 +
    location_factor * 0.1
)

# Stress concentration factor
stress_factor = base_factors[damage_type] * geometry_multiplier
```

### Training Configuration
```python
training_config = {
    "epochs": 100,
    "batch_size": 16,
    "image_size": 640,
    "learning_rate": 0.01,
    "augmentation": {
        "flip": True,
        "rotate": True,
        "scale": True,
        "crop": True
    }
}
```

## Monitoring & Debugging

### Health Check
```bash
curl http://localhost:8000/health
```

### Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```

### Logs
```bash
tail -f logs/agentv2.log
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Pastikan MySQL berjalan
   - Check kredensial database di config
   - Pastikan database sudah dibuat

2. **CUDA Out of Memory**
   - Kurangi batch size
   - Gunakan model yang lebih kecil (YOLOv8n)
   - Set device ke "cpu" di config

3. **Model Loading Error**
   - Download ulang model YOLO
   - Check path model di config
   - Pastikan format model compatible

4. **File Upload Error**
   - Check file size limit
   - Pastikan format file supported
   - Check permissions folder uploads

### Performance Optimization

1. **Database Optimization**
   - Index pada kolom yang sering diquery
   - Connection pooling
   - Query optimization

2. **Model Inference**
   - GPU acceleration dengan CUDA
   - Model quantization
   - Batch processing

3. **File Handling**
   - Image compression
   - Async file operations
   - CDN untuk static files

## Development

### Setup Development Environment
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
isort .
flake8 .
```

## Deployment

### Production Setup
1. **Environment Variables**
```bash
export DEBUG=false
export DB_HOST=production_db_host
export SECRET_KEY=your_super_secret_key
```

2. **SSL Certificate**
```bash
# Setup SSL dengan Let's Encrypt atau certificate lainnya
```

3. **Reverse Proxy (Nginx)**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "backend/main.py"]
```

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

Untuk pertanyaan atau dukungan:
- Email: support@agentv2.local
- Documentation: `http://localhost:8000/docs`
- Issues: GitHub Issues

## Changelog

### v2.0.0 (Current)
- Advanced YOLO evaluation with mAP@0.5 and mAP@[.5:.95]
- Geometry-based risk assessment
- Human-driven training system
- 3-role architecture (User/Admin/Trainer)
- Temperature scaling for confidence calibration
- Monte Carlo dropout for uncertainty quantification
- Real-time training monitoring
- Advanced data augmentation
- Comprehensive evaluation metrics

### v1.0.0
- Basic YOLO detection
- Simple risk categorization
- Single-role interface
- Basic training functionality

---

**AgentV2** - Advanced AI Asset Inspection System
Developed with ❤️ for industrial asset management
