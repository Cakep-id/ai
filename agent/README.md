# CAKEP.id Early Warning System AI Module

Sistem AI terintegrasi untuk Early Warning System pemeliharaan aset industri dengan capabilities:
- **Computer Vision** menggunakan YOLO untuk damage detection
- **NLP Analysis** menggunakan Groq AI untuk text analysis  
- **Risk Assessment** yang menggabungkan CV + NLP results
- **Automated Scheduling** berdasarkan risk levels
- **Admin Interface** untuk upload dan retrain models

## 🏗️ Architecture

```
agent/
├── main.py                 # FastAPI application entry point
├── config.py               # Configuration management
├── requirements.txt        # Dependencies
├── .env.example           # Environment variables template
├── services/              # Core business logic
│   ├── db.py             # Database operations
│   ├── yolo_service.py   # Computer vision with YOLO
│   ├── groq_service.py   # NLP analysis with Groq AI
│   ├── risk_engine.py    # Risk assessment engine
│   └── scheduler.py      # Automated scheduling
├── api/                   # FastAPI endpoints
│   ├── cv_endpoints.py   # Computer vision APIs
│   ├── nlp_endpoints.py  # NLP analysis APIs
│   ├── risk_endpoints.py # Risk assessment APIs
│   ├── schedule_endpoints.py # Scheduling APIs
│   └── admin_endpoints.py    # Admin management APIs
└── ml/                    # ML models and datasets
    ├── models/           # Trained models
    └── datasets/         # Training datasets
```

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.8+
- MySQL Database
- Groq API Key
- CUDA (optional, untuk GPU acceleration)

### 2. Installation

```bash
# Clone atau copy agent folder
cd agent/

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env dengan konfigurasi Anda
```

### 3. Database Setup

```sql
-- Create database
CREATE DATABASE cakep_ews CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create user (optional)
CREATE USER 'ews_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON cakep_ews.* TO 'ews_user'@'localhost';
FLUSH PRIVILEGES;
```

### 4. Environment Configuration

Edit file `.env`:

```env
# Database
DB_HOST=localhost
DB_PORT=3306
DB_NAME=cakep_ews
DB_USER=ews_user
DB_PASSWORD=your_password

# Groq AI API
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=mixtral-8x7b-32768

# YOLO Configuration
YOLO_MODEL_PATH=models/yolo_damage_detection.pt
YOLO_CONFIDENCE_THRESHOLD=0.5

# Risk Engine
RISK_VISUAL_WEIGHT=0.6
RISK_TEXT_WEIGHT=0.4

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

### 5. Initialize Database

```bash
python -c "from services.db import init_database; init_database()"
```

### 6. Run Application

```bash
# Development mode
python main.py

# Production mode dengan uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📖 API Documentation

Setelah aplikasi berjalan, akses:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc  
- **Health Check**: http://localhost:8000/health

### API Endpoints Overview

#### Computer Vision API (`/api/cv`)
- `POST /detect` - Damage detection dari image
- `POST /retrain` - Retrain YOLO model
- `GET /model/info` - Model information
- `GET /stats` - Detection statistics

#### NLP Analysis API (`/api/nlp`)
- `POST /analyze` - Text analysis dengan Groq AI
- `POST /analyze/batch` - Batch text analysis
- `GET /categories` - Damage categories
- `GET /stats` - Analysis statistics

#### Risk Assessment API (`/api/risk`)
- `POST /assess` - Risk assessment untuk report
- `POST /assess/manual` - Manual risk calculation
- `POST /assess/bulk` - Bulk risk assessment
- `GET /report/{id}` - Get risk assessment
- `GET /stats` - Risk statistics

#### Scheduling API (`/api/schedule`)
- `POST /generate` - Generate maintenance schedule
- `POST /generate/bulk` - Bulk schedule generation
- `PUT /update` - Update existing schedule
- `GET /list` - List schedules dengan filtering
- `GET /calendar` - Calendar view
- `GET /resource-planning` - Resource planning

#### Admin API (`/api/admin`)
- `POST /upload/image` - Upload images
- `POST /upload/annotations` - Upload training annotations
- `POST /train/yolo` - Train YOLO model
- `GET /training/status/{id}` - Training status
- `GET /system/status` - System status
- `POST /backup/create` - Create backup

## 🔄 Usage Flow

### 1. Basic Detection Flow

```python
import requests

# 1. Upload image
files = {'file': open('damage_image.jpg', 'rb')}
response = requests.post('http://localhost:8000/api/cv/detect', files=files)
detection_result = response.json()

# 2. Analyze description text
data = {'text': 'Kerusakan pada pipa dengan korosi berat'}
response = requests.post('http://localhost:8000/api/nlp/analyze', json=data)
nlp_result = response.json()

# 3. Risk assessment
risk_data = {
    'visual_score': detection_result['damage_score'],
    'text_score': nlp_result['text_score']
}
response = requests.post('http://localhost:8000/api/risk/assess/manual', json=risk_data)
risk_result = response.json()

# 4. Generate schedule
schedule_data = {'report_id': 1}
response = requests.post('http://localhost:8000/api/schedule/generate', json=schedule_data)
schedule_result = response.json()
```

### 2. Admin Upload & Retrain Flow

```python
# Upload training images
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb'))
]
data = {'dataset_name': 'new_damage_dataset'}
response = requests.post('http://localhost:8000/api/admin/upload/annotations', 
                        files=files, data=data)

# Start YOLO retraining
train_data = {
    'epochs': 100,
    'batch_size': 16,
    'dataset_path': 'uploads/datasets/new_damage_dataset'
}
response = requests.post('http://localhost:8000/api/admin/train/yolo', data=train_data)
training_id = response.json()['training_log_id']

# Monitor training
response = requests.get(f'http://localhost:8000/api/admin/training/status/{training_id}')
training_status = response.json()
```

## 🎯 Key Features

### Computer Vision (YOLO)
- Real-time damage detection
- Custom damage categories mapping
- Confidence scoring
- Model retraining capabilities
- Batch processing

### NLP Analysis (Groq AI)
- Text sentiment analysis
- Damage category classification  
- Keyword extraction
- Confidence scoring
- Batch text processing

### Risk Assessment Engine
- Kombinasi visual + text scores
- Configurable weights (default: 60% visual, 40% text)
- Multi-level risk classification (CRITICAL, HIGH, MEDIUM, LOW)
- Repair procedure recommendations
- Cost estimations

### Automated Scheduling
- SLA-based scheduling
- Priority-based resource allocation
- Calendar view
- Work order generation
- Resource planning

### Admin Capabilities
- Image upload untuk detection/training
- Annotation management
- Model retraining
- System monitoring
- Backup/restore

## 🔧 Configuration

### Risk Engine Tuning

```python
# Update risk weights
config_data = {
    'visual_weight': 0.7,
    'text_weight': 0.3,
    'risk_thresholds': {
        'CRITICAL': 0.8,
        'HIGH': 0.6,
        'MEDIUM': 0.4,
        'LOW': 0.0
    }
}
response = requests.post('http://localhost:8000/api/risk/config/update', json=config_data)
```

### YOLO Model Configuration

Edit `.env`:
```env
YOLO_MODEL_PATH=models/custom_yolo_v8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.6
YOLO_DEVICE=cuda  # atau 'cpu'
```

## 🚀 Production Deployment

### 1. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. Systemd Service

```ini
[Unit]
Description=CAKEP EWS AI Module
After=network.target

[Service]
Type=simple
User=ews
WorkingDirectory=/opt/cakep-ews
ExecStart=/opt/cakep-ews/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### 3. Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name ews-api.cakep.id;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoring & Maintenance

### Health Monitoring

```bash
# Check system health
curl http://localhost:8000/health

# Check individual services
curl http://localhost:8000/api/cv/test-connection
curl http://localhost:8000/api/nlp/test-connection
```

### Log Monitoring

```bash
# View logs
tail -f logs/ews_ai.log

# Filter by level
grep "ERROR" logs/ews_ai.log
```

### Performance Monitoring

```python
# Get system stats
response = requests.get('http://localhost:8000/api/admin/system/status')
system_status = response.json()

# Database stats
response = requests.get('http://localhost:8000/api/cv/stats')
cv_stats = response.json()
```

## 🔒 Security Considerations

1. **API Keys**: Jangan expose Groq API key di logs
2. **Database**: Gunakan user dengan minimal privileges
3. **File Upload**: Validasi file types dan sizes
4. **CORS**: Configure properly untuk production
5. **Authentication**: Implement auth untuk admin endpoints

## 🐛 Troubleshooting

### Common Issues

**1. Database Connection Error**
```bash
# Check database connectivity
mysql -h localhost -u ews_user -p cakep_ews
```

**2. YOLO Model Not Found**
```bash
# Check model path
ls -la models/
# Download default model if needed
```

**3. Groq API Error**
```bash
# Verify API key
curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/v1/models
```

**4. Memory Issues**
```bash
# Monitor memory usage
free -h
# Adjust batch sizes in config
```

### Debug Mode

```env
DEBUG=true
LOG_LEVEL=DEBUG
```

## 📈 Performance Optimization

1. **Database**: Add proper indexes, connection pooling
2. **YOLO**: Use GPU acceleration, optimize model size
3. **API**: Implement caching, rate limiting
4. **Background Tasks**: Use Celery untuk heavy processing

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests untuk new features
4. Submit pull request

## 📄 License

Proprietary - CAKEP.id Project

## 📞 Support

Untuk support teknis, hubungi tim development CAKEP.id.

---
**CAKEP.id Early Warning System AI Module v1.0.0**
