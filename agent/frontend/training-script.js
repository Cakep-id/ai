/**
 * Training Interface JavaScript
 * Handles dataset training management
 */

const API_BASE = 'http://localhost:8000';

// Global variables
let currentDatasets = [];
let damageClasses = [];
let annotations = [];
let isDrawing = false;
let startX, startY;

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadDamageClasses();
    loadDatasets();
    loadDatasetList();
    loadTrainingSessions();
});

function initializeEventListeners() {
    // Create Dataset Form
    document.getElementById('createDatasetForm').addEventListener('submit', handleCreateDataset);
    
    // Upload Image Form
    document.getElementById('uploadImageForm').addEventListener('submit', handleUploadImage);
    
    // Start Training Form
    document.getElementById('startTrainingForm').addEventListener('submit', handleStartTraining);
    
    // File upload area
    const fileUploadArea = document.getElementById('fileUploadArea');
    const imageFile = document.getElementById('imageFile');
    
    fileUploadArea.addEventListener('click', () => imageFile.click());
    fileUploadArea.addEventListener('dragover', handleDragOver);
    fileUploadArea.addEventListener('drop', handleDrop);
    imageFile.addEventListener('change', handleFileSelect);
    
    // Canvas for annotations
    const canvas = document.getElementById('annotationCanvas');
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', drawRectangle);
    canvas.addEventListener('mouseup', stopDrawing);
}

function switchTab(event, tabName) {
    // Prevent default if event is provided
    if (event) {
        event.preventDefault();
    }
    
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Add active class to clicked button
    if (event && event.target) {
        event.target.classList.add('active');
    }
    
    // Load data based on tab
    switch(tabName) {
        case 'upload-images':
            loadDatasets();
            break;
        case 'manage-datasets':
            loadDatasetList();
            break;
        case 'start-training':
            loadDatasets();
            loadTrainingSessions();
            break;
    }
}

// Create Dataset Functions
async function handleCreateDataset(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    
    // Show loading
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'inline-block';
    
    try {
        const formData = new FormData(form);
        const data = {
            dataset_name: formData.get('dataset_name'),
            description: formData.get('description'),
            uploaded_by: formData.get('uploaded_by')
        };
        
        const response = await fetch(`${API_BASE}/api/training/dataset/create`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('success', `Dataset "${result.data.dataset_name}" berhasil dibuat!`);
            form.reset();
            loadDatasets(); // Refresh dataset list
        } else {
            showAlert('error', result.message || 'Gagal membuat dataset');
        }
        
    } catch (error) {
        console.error('Error creating dataset:', error);
        showAlert('error', 'Terjadi kesalahan saat membuat dataset');
    } finally {
        // Hide loading
        submitBtn.disabled = false;
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
    }
}

// Load Functions
async function loadDatasets() {
    try {
        const response = await fetch(`${API_BASE}/api/training/datasets`);
        const result = await response.json();
        
        if (result.success) {
            currentDatasets = result.data;
            updateDatasetSelects();
        }
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}

async function loadDamageClasses() {
    try {
        const response = await fetch(`${API_BASE}/api/training/damage-classes`);
        const result = await response.json();
        
        if (result.success) {
            damageClasses = result.data;
            updateDamageClassSelects();
        }
    } catch (error) {
        console.error('Error loading damage classes:', error);
    }
}

function updateDatasetSelects() {
    const selects = ['targetDataset', 'trainingDataset'];
    
    selects.forEach(selectId => {
        const select = document.getElementById(selectId);
        if (select) {
            // Clear existing options except first
            select.innerHTML = '<option value="">-- Pilih Dataset --</option>';
            
            currentDatasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset.dataset_id;
                option.textContent = `${dataset.dataset_name} (${dataset.total_images} gambar) - ${dataset.status}`;
                select.appendChild(option);
            });
        }
    });
}

function updateDamageClassSelects() {
    const selects = ['damageType', 'annotationClass'];
    
    selects.forEach(selectId => {
        const select = document.getElementById(selectId);
        if (select) {
            // Clear existing options except first
            select.innerHTML = '<option value="">-- Pilih Jenis Kerusakan --</option>';
            
            damageClasses.forEach(cls => {
                const option = document.createElement('option');
                option.value = cls.class_name;
                option.textContent = `${cls.class_name} - ${cls.description}`;
                option.dataset.color = cls.color;
                select.appendChild(option);
            });
        }
    });
}

// File Upload Functions
function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    handleFiles(files);
}

function handleFileSelect(event) {
    const files = event.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            showAlert('error', 'Format file tidak didukung. Gunakan: JPG, PNG, BMP, atau TIFF');
            return;
        }
        
        // Show preview
        showImagePreview(file);
    }
}

function showImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const preview = document.getElementById('imagePreview');
        const canvas = document.getElementById('annotationCanvas');
        const container = document.getElementById('imagePreviewContainer');
        
        preview.src = e.target.result;
        container.style.display = 'block';
        
        // Setup canvas
        preview.onload = function() {
            const maxWidth = 600;
            const maxHeight = 400;
            let { width, height } = calculateImageSize(preview.naturalWidth, preview.naturalHeight, maxWidth, maxHeight);
            
            canvas.width = width;
            canvas.height = height;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            
            // Draw image on canvas
            const ctx = canvas.getContext('2d');
            ctx.drawImage(preview, 0, 0, width, height);
        };
    };
    
    reader.readAsDataURL(file);
}

function calculateImageSize(naturalWidth, naturalHeight, maxWidth, maxHeight) {
    let width = naturalWidth;
    let height = naturalHeight;
    
    if (width > maxWidth) {
        height = (height * maxWidth) / width;
        width = maxWidth;
    }
    
    if (height > maxHeight) {
        width = (width * maxHeight) / height;
        height = maxHeight;
    }
    
    return { width: Math.round(width), height: Math.round(height) };
}

// Annotation Functions
function startDrawing(event) {
    const rect = event.target.getBoundingClientRect();
    startX = event.clientX - rect.left;
    startY = event.clientY - rect.top;
    isDrawing = true;
}

function drawRectangle(event) {
    if (!isDrawing) return;
    
    const canvas = event.target;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    const currentX = event.clientX - rect.left;
    const currentY = event.clientY - rect.top;
    
    // Clear canvas and redraw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const preview = document.getElementById('imagePreview');
    ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);
    
    // Draw existing annotations
    drawExistingAnnotations(ctx);
    
    // Draw current rectangle
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
}

function stopDrawing(event) {
    if (!isDrawing) return;
    
    const rect = event.target.getBoundingClientRect();
    const endX = event.clientX - rect.left;
    const endY = event.clientY - rect.top;
    
    const width = Math.abs(endX - startX);
    const height = Math.abs(endY - startY);
    
    // Only create annotation if rectangle is big enough
    if (width > 10 && height > 10) {
        const className = document.getElementById('annotationClass').value;
        if (className) {
            const annotation = {
                x: Math.min(startX, endX),
                y: Math.min(startY, endY),
                width: width,
                height: height,
                class_name: className,
                confidence: 1.0
            };
            
            annotations.push(annotation);
            updateAnnotationList();
            redrawCanvas();
        } else {
            showAlert('error', 'Pilih class annotation terlebih dahulu');
        }
    }
    
    isDrawing = false;
}

function drawExistingAnnotations(ctx) {
    annotations.forEach((ann, index) => {
        ctx.strokeStyle = getClassColor(ann.class_name);
        ctx.lineWidth = 2;
        ctx.strokeRect(ann.x, ann.y, ann.width, ann.height);
        
        // Draw label
        ctx.fillStyle = getClassColor(ann.class_name);
        ctx.fillRect(ann.x, ann.y - 20, ann.class_name.length * 8, 20);
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(ann.class_name, ann.x + 2, ann.y - 5);
    });
}

function getClassColor(className) {
    const cls = damageClasses.find(c => c.class_name === className);
    return cls ? cls.color : '#ff0000';
}

function redrawCanvas() {
    const canvas = document.getElementById('annotationCanvas');
    const ctx = canvas.getContext('2d');
    const preview = document.getElementById('imagePreview');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);
    drawExistingAnnotations(ctx);
}

function addAnnotation() {
    const className = document.getElementById('annotationClass').value;
    if (!className) {
        showAlert('error', 'Pilih class annotation terlebih dahulu');
        return;
    }
    
    showAlert('info', 'Klik dan drag pada gambar untuk membuat bounding box');
}

function clearAnnotations() {
    annotations = [];
    updateAnnotationList();
    redrawCanvas();
}

function removeAnnotation(index) {
    annotations.splice(index, 1);
    updateAnnotationList();
    redrawCanvas();
}

function updateAnnotationList() {
    const container = document.getElementById('annotations');
    
    if (annotations.length === 0) {
        container.innerHTML = '<p>Belum ada annotation</p>';
        return;
    }
    
    container.innerHTML = annotations.map((ann, index) => `
        <div class="annotation-item">
            <span>${ann.class_name} (${Math.round(ann.width)}x${Math.round(ann.height)})</span>
            <button class="btn btn-danger" onclick="removeAnnotation(${index})">Hapus</button>
        </div>
    `).join('');
}

// Upload Image Function
async function handleUploadImage(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    
    // Validation
    const datasetId = document.getElementById('targetDataset').value;
    const imageFile = document.getElementById('imageFile').files[0];
    
    if (!datasetId) {
        showAlert('error', 'Pilih dataset target terlebih dahulu');
        return;
    }
    
    if (!imageFile) {
        showAlert('error', 'Pilih file gambar terlebih dahulu');
        return;
    }
    
    // Show loading
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'inline-block';
    
    try {
        const formData = new FormData();
        formData.append('image_file', imageFile);
        formData.append('damage_type', document.getElementById('damageType').value);
        formData.append('damage_severity', document.getElementById('damageSeverity').value);
        formData.append('damage_description', document.getElementById('damageDescription').value);
        formData.append('annotations', JSON.stringify(annotations));
        
        const response = await fetch(`${API_BASE}/api/training/dataset/${datasetId}/upload-image`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('success', `Gambar berhasil diupload dengan ${result.data.annotations_count} annotations!`);
            
            // Reset form
            form.reset();
            document.getElementById('imagePreviewContainer').style.display = 'none';
            annotations = [];
            
            // Refresh datasets
            loadDatasets();
        } else {
            showAlert('error', result.message || 'Gagal mengupload gambar');
        }
        
    } catch (error) {
        console.error('Error uploading image:', error);
        showAlert('error', 'Terjadi kesalahan saat mengupload gambar');
    } finally {
        // Hide loading
        submitBtn.disabled = false;
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
    }
}

// Dataset Management Functions
async function loadDatasetList() {
    try {
        const response = await fetch(`${API_BASE}/api/training/datasets`);
        const result = await response.json();
        
        if (result.success) {
            displayDatasetList(result.data);
        }
    } catch (error) {
        console.error('Error loading dataset list:', error);
    }
}

function displayDatasetList(datasets) {
    const container = document.getElementById('datasetList');
    
    if (datasets.length === 0) {
        container.innerHTML = '<p>Belum ada dataset</p>';
        return;
    }
    
    container.innerHTML = datasets.map(dataset => `
        <div class="dataset-card">
            <div class="dataset-header">
                <div class="dataset-info">
                    <h3>${dataset.dataset_name}</h3>
                    <p>${dataset.description}</p>
                    <span class="status-badge status-${dataset.status}">${dataset.status}</span>
                </div>
                <div class="dataset-actions">
                    <button class="btn btn-primary" onclick="viewDatasetImages(${dataset.dataset_id})">
                        👁️ Lihat Images
                    </button>
                    <button class="btn btn-danger" onclick="deleteDataset(${dataset.dataset_id})">
                        🗑️ Hapus
                    </button>
                </div>
            </div>
            <div class="dataset-stats">
                <p><strong>Total Images:</strong> ${dataset.total_images}</p>
                <p><strong>Upload Date:</strong> ${new Date(dataset.upload_date).toLocaleDateString('id-ID')}</p>
                <p><strong>Uploaded By:</strong> ${dataset.uploaded_by}</p>
                <p><strong>Version:</strong> ${dataset.dataset_version}</p>
            </div>
        </div>
    `).join('');
}

async function deleteDataset(datasetId) {
    if (!confirm('Yakin ingin menghapus dataset ini?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/training/dataset/${datasetId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('success', 'Dataset berhasil dihapus');
            loadDatasetList();
            loadDatasets();
        } else {
            showAlert('error', result.message || 'Gagal menghapus dataset');
        }
    } catch (error) {
        console.error('Error deleting dataset:', error);
        showAlert('error', 'Terjadi kesalahan saat menghapus dataset');
    }
}

// Training Functions
async function handleStartTraining(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    
    // Show loading
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'inline-block';
    
    try {
        const formData = new FormData(form);
        
        const response = await fetch(`${API_BASE}/api/training/dataset/${formData.get('dataset_id')}/start-training`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('success', `Training session "${result.data.session_id}" berhasil dimulai!`);
            form.reset();
            loadTrainingSessions();
        } else {
            showAlert('error', result.message || 'Gagal memulai training');
        }
        
    } catch (error) {
        console.error('Error starting training:', error);
        showAlert('error', 'Terjadi kesalahan saat memulai training');
    } finally {
        // Hide loading
        submitBtn.disabled = false;
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
    }
}

async function loadTrainingSessions() {
    try {
        const response = await fetch(`${API_BASE}/api/training/training-sessions`);
        const result = await response.json();
        
        if (result.success) {
            displayTrainingSessions(result.data);
        }
    } catch (error) {
        console.error('Error loading training sessions:', error);
    }
}

function displayTrainingSessions(sessions) {
    const container = document.getElementById('sessionsList');
    
    if (sessions.length === 0) {
        container.innerHTML = '<p>Belum ada training sessions</p>';
        return;
    }
    
    container.innerHTML = sessions.map(session => `
        <div class="dataset-card">
            <div class="dataset-header">
                <div class="dataset-info">
                    <h3>${session.session_name}</h3>
                    <p>Dataset: ${session.dataset_name}</p>
                    <span class="status-badge status-${session.status}">${session.status}</span>
                </div>
            </div>
            <div class="dataset-stats">
                <p><strong>Started By:</strong> ${session.started_by}</p>
                <p><strong>Start Time:</strong> ${new Date(session.start_time).toLocaleString('id-ID')}</p>
                <p><strong>Epochs:</strong> ${session.epochs}</p>
                ${session.training_accuracy ? `<p><strong>Training Accuracy:</strong> ${(session.training_accuracy * 100).toFixed(2)}%</p>` : ''}
                ${session.validation_accuracy ? `<p><strong>Validation Accuracy:</strong> ${(session.validation_accuracy * 100).toFixed(2)}%</p>` : ''}
            </div>
        </div>
    `).join('');
}

// Utility Functions
function showAlert(type, message) {
    const container = document.getElementById('alertContainer');
    const alertId = 'alert-' + Date.now();
    
    const alertElement = document.createElement('div');
    alertElement.id = alertId;
    alertElement.className = `alert alert-${type}`;
    alertElement.innerHTML = `
        ${message}
        <button onclick="closeAlert('${alertId}')" style="float: right; background: none; border: none; font-size: 18px; cursor: pointer;">&times;</button>
    `;
    
    container.appendChild(alertElement);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        closeAlert(alertId);
    }, 5000);
}

function closeAlert(alertId) {
    const alert = document.getElementById(alertId);
    if (alert) {
        alert.remove();
    }
}
