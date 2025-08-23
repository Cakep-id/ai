// CAKEP.id EWS Admin Dashboard Script - Fixed

// Global variables
let currentTab = 'validation';
let validationQueue = [];
let reportsQueue = [];
let trainingInProgress = false;
let systemStats = {};
let selectedDetection = null;
let editingReportId = null;

// API Base URL
const API_BASE = '/api';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Admin Dashboard initializing...');
    initializeApp();
    setupEventListeners();
    loadInitialData();
});

// Initialize application
function initializeApp() {
    console.log('Initializing app...');
    showTab('validation');
    updateSystemStatus();
    setInterval(updateSystemStatus, 30000); // Update every 30 seconds
    
    // Test click functionality
    setTimeout(() => {
        console.log('Testing click functionality...');
        const tabs = document.querySelectorAll('.nav-tab');
        tabs.forEach((tab, index) => {
            console.log(`Tab ${index}: ${tab.textContent.trim()}, data-tab: ${tab.dataset.tab}`);
        });
    }, 1000);
}

// Event Listeners Setup
function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Simple tab click test
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('nav-tab') || e.target.closest('.nav-tab')) {
            const tab = e.target.classList.contains('nav-tab') ? e.target : e.target.closest('.nav-tab');
            console.log('Nav tab clicked:', tab.dataset.tab);
            e.preventDefault();
            e.stopPropagation();
            showTab(tab.dataset.tab);
        }
    });
    
    // Tab handlers
    const tabs = document.querySelectorAll('.nav-tab');
    console.log('Found tabs:', tabs.length);
    
    tabs.forEach((tab, index) => {
        console.log(`Setting up tab ${index}:`, tab.dataset.tab);
        tab.addEventListener('click', (e) => {
            console.log('Tab clicked:', tab.dataset.tab);
            e.preventDefault();
            e.stopPropagation();
            const targetTab = tab.dataset.tab;
            showTab(targetTab);
        });
    });

    // Training data handlers
    const refreshTrainingBtn = document.getElementById('refreshTrainingBtn');
    if (refreshTrainingBtn) {
        refreshTrainingBtn.addEventListener('click', () => {
            loadTrainingData();
        });
    }

    // Filter handlers for training data
    const filters = ['validationFilter', 'riskFilter'];
    filters.forEach(filterId => {
        const filterElement = document.getElementById(filterId);
        if (filterElement) {
            filterElement.addEventListener('change', () => {
                loadTrainingData();
            });
        }
    });

    // Other existing handlers
    setupFormHandlers();
    setupModalHandlers();
    setupFileUploads(); // Add file upload handlers
}

// Tab management
function showTab(tabId) {
    console.log('Showing tab:', tabId);
    
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from nav tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    const targetTab = document.getElementById(tabId + 'Tab');
    if (targetTab) {
        targetTab.classList.add('active');
        console.log('Tab activated:', tabId + 'Tab');
    } else {
        console.error('Tab not found:', tabId + 'Tab');
    }
    
    // Add active class to nav tab
    const navTab = document.querySelector(`[data-tab="${tabId}"]`);
    if (navTab) {
        navTab.classList.add('active');
        console.log('Nav tab activated:', tabId);
    } else {
        console.error('Nav tab not found:', tabId);
    }
    
    currentTab = tabId;
    
    // Load tab-specific data
    switch(tabId) {
        case 'validation':
            loadValidationQueue();
            break;
        case 'reports':
            loadReportsQueue();
            break;
        case 'training':
            loadTrainingData();
            break;
        case 'monitoring':
            updateSystemStatus();
            break;
    }
}

// Load initial data
function loadInitialData() {
    loadValidationQueue();
    loadDashboardStats();
    loadTrainingData(); // Load training data when page loads
}

// Validation Queue Management
async function loadValidationQueue() {
    try {
        showLoading('validationQueue');
        
        // Use admin validation endpoint
        const response = await fetch(`${API_BASE}/admin/validation-queue`);
        const data = await response.json();
        
        if (data.success) {
            validationQueue = data.queue;
            renderValidationQueue(data.queue);
        } else {
            throw new Error(data.detail || 'Gagal memuat antrian validasi');
        }
    } catch (error) {
        console.error('Error loading validation queue:', error);
        showError('validationQueue', 'Gagal memuat antrian validasi: ' + error.message);
    } finally {
        hideLoading('validationQueue');
    }
}

function renderValidationQueue(queue) {
    const container = document.getElementById('validationQueue');
    
    if (!container) {
        console.error('Container validationQueue tidak ditemukan');
        return;
    }
    
    if (queue.length === 0) {
        container.innerHTML = `
            <div class="text-center" style="padding: 2rem;">
                <i class="fas fa-check-circle" style="font-size: 3rem; color: #2ecc71; margin-bottom: 1rem;"></i>
                <h3>Semua Validasi Selesai</h3>
                <p class="text-muted">Tidak ada laporan yang menunggu validasi</p>
            </div>
        `;
        return;
    }

    container.innerHTML = queue.map(report => `
        <div class="validation-item" data-id="${report.report_id}">
            <div class="validation-item-header">
                <div>
                    <span class="validation-item-type">Laporan Kerusakan</span>
                    <span class="validation-item-priority priority-${report.ai_risk_level?.toLowerCase() || 'medium'}">${report.ai_risk_level || 'MEDIUM'}</span>
                    ${report.urgency_status ? `<span class="urgency-status ${report.urgency_status.toLowerCase()}">${report.urgency_status}</span>` : ''}
                </div>
                <small class="text-muted">${formatDateTime(report.reported_at)} • ${report.hours_since_report || 0} jam lalu</small>
            </div>
            <div class="validation-item-content">
                <h4>${report.asset_name} - ${report.ai_detected_damage || 'Kerusakan Terdeteksi'}</h4>
                <p class="text-muted mb-2">${report.description}</p>
                <p><strong>Lokasi:</strong> ${report.asset_location}</p>
                <p><strong>Pelapor:</strong> ${report.reporter_name}</p>
                <p><strong>AI Confidence:</strong> ${Math.round((report.ai_confidence || 0) * 100)}%</p>
            </div>
            <div class="validation-item-actions">
                <button class="btn btn-primary" onclick="openValidationModal(${report.report_id})">
                    <i class="fas fa-eye"></i> Review & Validasi
                </button>
            </div>
        </div>
    `).join('');
}

// Modal Management
async function openValidationModal(reportId) {
    try {
        // Find report in validation queue
        const report = validationQueue.find(r => r.report_id === reportId);
        
        if (report) {
            selectedDetection = report;
            displayValidationModal(report);
        } else {
            showNotification('Gagal memuat detail laporan', 'error');
        }
    } catch (error) {
        console.error('Error loading report details:', error);
        showNotification('Gagal memuat detail laporan', 'error');
    }
}

function displayValidationModal(report) {
    const modalHTML = `
        <div class="modal" id="validationModal" style="display: flex;">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Validasi Laporan Kerusakan</h3>
                    <button class="modal-close" onclick="closeModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="report-details">
                        <div class="detail-section">
                            <h4>Informasi Aset</h4>
                            <p><strong>Nama Aset:</strong> ${report.asset_name}</p>
                            <p><strong>Lokasi:</strong> ${report.asset_location}</p>
                            <p><strong>Kritikalitas:</strong> ${report.criticality || 'MEDIUM'}</p>
                        </div>
                        
                        <div class="detail-section">
                            <h4>Detail Laporan</h4>
                            <p><strong>Tanggal:</strong> ${formatDateTime(report.reported_at)}</p>
                            <p><strong>Pelapor:</strong> ${report.reporter_name}</p>
                            <p><strong>Deskripsi:</strong> ${report.description}</p>
                        </div>
                        
                        <div class="detail-section">
                            <h4>Hasil Analisis AI</h4>
                            <p><strong>Jenis Kerusakan:</strong> ${report.ai_detected_damage}</p>
                            <p><strong>Tingkat Risiko:</strong> <span class="risk-${report.ai_risk_level?.toLowerCase()}">${report.ai_risk_level}</span></p>
                            <p><strong>Confidence:</strong> ${Math.round((report.ai_confidence || 0) * 100)}%</p>
                        </div>
                        
                        <div class="detail-section">
                            <h4>Validasi Admin</h4>
                            <form id="validationForm">
                                <div class="form-group">
                                    <label>Keputusan:</label>
                                    <div class="validation-actions">
                                        <label class="radio-label">
                                            <input type="radio" name="action" value="approve" required>
                                            <span class="approve">✓ Setujui</span>
                                        </label>
                                        <label class="radio-label">
                                            <input type="radio" name="action" value="reject" required>
                                            <span class="reject">✗ Tolak</span>
                                        </label>
                                    </div>
                                </div>
                                
                                <div class="form-group">
                                    <label for="adminNotes">Catatan Admin:</label>
                                    <textarea id="adminNotes" name="admin_notes" rows="3" placeholder="Berikan catatan untuk keputusan Anda..."></textarea>
                                </div>
                                
                                <div class="correction-section" id="correctionSection" style="display: none;">
                                    <h5>Koreksi Hasil AI (Opsional)</h5>
                                    <div class="form-group">
                                        <label for="correctedDamage">Jenis Kerusakan yang Benar:</label>
                                        <input type="text" id="correctedDamage" name="corrected_damage" placeholder="Jika AI salah, masukkan jenis kerusakan yang benar">
                                    </div>
                                    
                                    <div class="form-group">
                                        <label for="correctedRisk">Tingkat Risiko yang Benar:</label>
                                        <select id="correctedRisk" name="corrected_risk">
                                            <option value="">-- Gunakan hasil AI --</option>
                                            <option value="CRITICAL">CRITICAL</option>
                                            <option value="HIGH">HIGH</option>
                                            <option value="MEDIUM">MEDIUM</option>
                                            <option value="LOW">LOW</option>
                                        </select>
                                    </div>
                                    
                                    <div class="form-group">
                                        <label for="correctedProcedures">Prosedur yang Benar:</label>
                                        <textarea id="correctedProcedures" name="corrected_procedures" rows="4" placeholder="Jika perlu, berikan prosedur perbaikan yang lebih tepat..."></textarea>
                                    </div>
                                </div>
                                
                                <div class="modal-actions">
                                    <button type="button" class="btn btn-secondary" onclick="closeModal()">Batal</button>
                                    <button type="submit" class="btn btn-primary">Simpan Validasi</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal if any
    const existingModal = document.getElementById('validationModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Setup form handlers
    setupFormHandlers();
}

function setupFormHandlers() {
    const form = document.getElementById('validationForm');
    if (!form) return;
    
    const actionRadios = form.querySelectorAll('input[name="action"]');
    const correctionSection = document.getElementById('correctionSection');
    
    actionRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            if (correctionSection) {
                if (radio.value === 'approve') {
                    correctionSection.style.display = 'block';
                } else {
                    correctionSection.style.display = 'none';
                }
            }
        });
    });
    
    form.addEventListener('submit', handleValidationSubmit);
}

async function handleValidationSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const action = formData.get('action');
    
    if (!action) {
        showNotification('Pilih keputusan validasi terlebih dahulu', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/admin/validate-report/${selectedDetection.report_id}`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification(data.message, 'success');
            closeModal();
            loadValidationQueue(); // Refresh queue
        } else {
            throw new Error(data.detail || 'Gagal menyimpan validasi');
        }
    } catch (error) {
        console.error('Error submitting validation:', error);
        showNotification('Gagal menyimpan validasi: ' + error.message, 'error');
    }
}

function closeModal() {
    const modal = document.getElementById('validationModal');
    if (modal) {
        modal.remove();
    }
    selectedDetection = null;
}

// Training Data Management
async function loadTrainingData() {
    try {
        const response = await fetch(`${API_BASE}/admin/training-data`);
        const data = await response.json();
        
        if (data.success) {
            renderTrainingData(data.training_data, data.statistics);
        }
    } catch (error) {
        console.error('Error loading training data:', error);
    }
}

function renderTrainingData(trainingData, statistics) {
    const container = document.getElementById('trainingData');
    if (!container) return;
    
    container.innerHTML = `
        <div class="training-stats">
            <h4>Statistik Data Training</h4>
            <p>Total Sampel: ${trainingData.length}</p>
            <div class="stats-grid">
                ${statistics.map(stat => `
                    <div class="stat-item">
                        <strong>${stat.damage_label}</strong><br>
                        ${stat.risk_level}: ${stat.count} sampel
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// Dashboard Statistics
// Load dashboard statistics from database
async function loadDashboardStats() {
    try {
        showLoading('dashboardStats');
        
        const response = await fetch(`${API_BASE}/admin/dashboard-stats`);
        const data = await response.json();
        
        if (data.success) {
            updateDashboardDisplay(data.stats);
        } else {
            console.error('Failed to load dashboard stats:', data.message);
        }
    } catch (error) {
        console.error('Error loading dashboard stats:', error);
    } finally {
        hideLoading('dashboardStats');
    }
}

function updateDashboardDisplay(stats) {
    // Update AI Performance Metrics
    const modelAccuracy = document.getElementById('modelAccuracy');
    if (modelAccuracy) {
        modelAccuracy.textContent = `${(stats.model_accuracy * 100).toFixed(1)}%`;
    }
    
    const totalPredictions = document.getElementById('totalPredictions');
    if (totalPredictions) {
        totalPredictions.textContent = stats.total_predictions || 0;
    }
    
    const correctPredictions = document.getElementById('correctPredictions');
    if (correctPredictions) {
        correctPredictions.textContent = stats.correct_predictions || 0;
    }
    
    // Update Risk Distribution
    const lowRiskCount = document.getElementById('lowRiskCount');
    if (lowRiskCount) {
        lowRiskCount.textContent = stats.low_risk_reports || 0;
    }
    
    const mediumRiskCount = document.getElementById('mediumRiskCount');
    if (mediumRiskCount) {
        mediumRiskCount.textContent = stats.medium_risk_reports || 0;
    }
    
    const highRiskCount = document.getElementById('highRiskCount');
    if (highRiskCount) {
        highRiskCount.textContent = stats.high_risk_reports || 0;
    }
    
    // Update Training Progress
    updateTrainingProgress(stats);
    
    console.log('Dashboard stats updated with real data:', stats);
}

function updateTrainingProgress(stats) {
    // Update training data progress bar
    const trainingDataProgressBar = document.getElementById('trainingDataProgress');
    if (trainingDataProgressBar) {
        const progressPercentage = Math.min(100, (stats.total_training_samples / 100) * 100);
        trainingDataProgressBar.style.width = `${progressPercentage}%`;
    }
    
    // Update training data progress text
    const trainingDataText = document.getElementById('trainingDataText');
    if (trainingDataText) {
        trainingDataText.textContent = `${stats.total_training_samples}/100 samples`;
    }
    
    // Update model training progress bar
    const modelTrainingProgressBar = document.getElementById('modelTrainingProgress');
    if (modelTrainingProgressBar) {
        const epochsProgress = Math.min(100, (stats.model_epochs / 100) * 100);
        modelTrainingProgressBar.style.width = `${epochsProgress}%`;
    }
    
    // Update model epochs text
    const modelTrainingText = document.getElementById('modelTrainingText');
    if (modelTrainingText) {
        modelTrainingText.textContent = `${stats.model_epochs} epochs completed`;
    }
}

// File Upload Management
function setupFileUploads() {
    console.log('Setting up file uploads...');
    
    // Simple direct approach
    document.addEventListener('DOMContentLoaded', function() {
        initFileUpload();
    });
    
    // Also try immediately in case DOMContentLoaded already fired
    initFileUpload();
}

function initFileUpload() {
    console.log('Initializing file upload...');
    
    const imageUploadArea = document.getElementById('imageUploadArea');
    const imageUpload = document.getElementById('imageUpload');
    
    console.log('Found upload area:', !!imageUploadArea);
    console.log('Found upload input:', !!imageUpload);
    
    if (imageUploadArea && imageUpload) {
        console.log('Setting up click handler...');
        
        // Clear any existing handlers
        imageUploadArea.onclick = null;
        
        // Simple click handler
        imageUploadArea.onclick = function(event) {
            console.log('Upload area clicked!');
            event.preventDefault();
            event.stopPropagation();
            
            try {
                console.log('About to trigger file input...');
                imageUpload.click();
                console.log('File input triggered successfully');
            } catch (error) {
                console.error('Error triggering file input:', error);
            }
        };
        
        // File change handler
        imageUpload.onchange = function(event) {
            console.log('File input changed, files:', event.target.files.length);
            handleImageUpload(event);
        };
        
        // Make sure the area is clickable
        imageUploadArea.style.cursor = 'pointer';
        imageUploadArea.style.pointerEvents = 'auto';
        
        console.log('File upload setup complete');
        
        // Setup drag and drop
        setupDragAndDrop(imageUploadArea, imageUpload);
    } else {
        console.error('Could not find upload elements');
        // Try again after a short delay
        setTimeout(initFileUpload, 500);
    }
    
    // Risk category selection
    setupRiskCategorySelector();
    
    // Upload with risk button
    const uploadBtn = document.getElementById('uploadWithRiskBtn');
    if (uploadBtn) {
        uploadBtn.addEventListener('click', handleRiskBasedUpload);
    }
    
    // Load training data stats
    loadTrainingStats();
}

function setupRiskCategorySelector() {
    const riskButtons = document.querySelectorAll('.risk-btn');
    const uploadBtn = document.getElementById('uploadWithRiskBtn');
    
    riskButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active from all buttons
            riskButtons.forEach(b => b.classList.remove('active'));
            // Add active to clicked button
            btn.classList.add('active');
            
            // Enable upload button
            if (uploadBtn) {
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = `<i class="fas fa-upload"></i> Upload sebagai ${btn.dataset.risk} RISK`;
            }
        });
    });
}

async function handleRiskBasedUpload() {
    const imageUpload = document.getElementById('imageUpload');
    const selectedRiskBtn = document.querySelector('.risk-btn.active');
    
    if (!imageUpload.files.length) {
        showNotification('Pilih gambar terlebih dahulu!', 'error');
        return;
    }
    
    if (!selectedRiskBtn) {
        showNotification('Pilih kategori risiko terlebih dahulu!', 'error');
        return;
    }
    
    const riskLevel = selectedRiskBtn.dataset.risk;
    const formData = new FormData();
    
    // Add all selected files
    Array.from(imageUpload.files).forEach(file => {
        formData.append('images', file);
    });
    formData.append('risk_category', riskLevel);
    
    try {
        const response = await fetch(`${API_BASE}/admin/upload-training-data`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification(`${data.uploaded_count || imageUpload.files.length} gambar berhasil disimpan ke database sebagai ${riskLevel} RISK`, 'success');
            
            // Reset form
            imageUpload.value = '';
            document.querySelector('.risk-btn.active')?.classList.remove('active');
            document.getElementById('uploadWithRiskBtn').disabled = true;
            document.getElementById('uploadWithRiskBtn').innerHTML = '<i class="fas fa-upload"></i> Upload & Kategorikan';
            
            // Update upload area text
            const uploadArea = document.getElementById('imageUploadArea');
            uploadArea.querySelector('p').textContent = 'Seret & lepas gambar di sini atau pilih file';
            
            // Reload all relevant data
            loadTrainingStats(); // Update stats
            loadTrainingData(); // Update training data table
            loadDashboardStats(); // Update dashboard metrics
            
            console.log('Upload successful:', data.message);
            
            // Show auto-training notification
            setTimeout(() => {
                showNotification('Model AI sedang dilatih ulang dengan data baru...', 'info');
            }, 1000);
            
        } else {
            showNotification('Error: ' + (data.message || 'Upload gagal'), 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('Error upload: ' + error.message, 'error');
    }
}

async function loadTrainingStats() {
    try {
        const response = await fetch(`${API_BASE}/admin/training-stats`);
        const data = await response.json();
        
        if (data.success && data.stats) {
            // Update individual risk counters for upload tab
            document.getElementById('lowRiskCount').textContent = data.stats.low_risk || 0;
            document.getElementById('mediumRiskCount').textContent = data.stats.medium_risk || 0;
            document.getElementById('highRiskCount').textContent = data.stats.high_risk || 0;
            
            console.log('Training stats loaded from database:', data.stats);
        } else {
            console.error('Failed to load training stats:', data.message);
        }
    } catch (error) {
        console.error('Error loading training stats:', error);
    }
}

// Training Data Management Functions
async function loadTrainingData() {
    try {
        showLoading('loadingTraining');
        
        // Get filter values
        const validationFilter = document.getElementById('validationFilter')?.value || '';
        const riskFilter = document.getElementById('riskFilter')?.value || '';
        
        // Build query params
        const params = new URLSearchParams();
        if (validationFilter) params.append('status', validationFilter);
        if (riskFilter) params.append('risk_level', riskFilter);
        
        const response = await fetch(`${API_BASE}/admin/training-data?${params.toString()}`);
        const data = await response.json();
        
        if (data.success) {
            updateTrainingStats(data.stats);
            renderTrainingDataTable(data.datasets);
            console.log('Training data loaded from database:', data.datasets.length, 'records');
        } else {
            showError('datasetTableBody', 'Gagal memuat data training: ' + (data.message || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error loading training data:', error);
        showError('datasetTableBody', 'Error saat memuat data training: ' + error.message);
    } finally {
        hideLoading('loadingTraining');
    }
}

function updateTrainingStats(stats) {
    document.getElementById('totalDataset').textContent = stats.total_dataset || 0;
    document.getElementById('validatedData').textContent = stats.validated_data || 0;
    document.getElementById('pendingData').textContent = stats.pending_data || 0;
}

function renderTrainingDataTable(datasets) {
    const tableBody = document.getElementById('datasetTableBody');
    if (!tableBody) return;
    
    if (!datasets || datasets.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="6" class="empty-state">
                    <i class="fas fa-database fa-2x"></i>
                    <p>Tidak ada data training yang tersedia</p>
                </td>
            </tr>
        `;
        return;
    }
    
    tableBody.innerHTML = datasets.map(dataset => `
        <tr>
            <td>
                ${dataset.image_path ? 
                    `<img src="/${dataset.image_path}" alt="Dataset" class="dataset-thumbnail" style="width: 60px; height: 60px; object-fit: cover; border-radius: 4px;">` :
                    '<i class="fas fa-image"></i>'
                }
            </td>
            <td>${dataset.filename || 'Unknown'}</td>
            <td>
                <span class="risk-badge risk-${dataset.risk_level?.toLowerCase() || 'unknown'}">
                    ${dataset.risk_level || 'Unknown'}
                </span>
            </td>
            <td>
                <span class="status-badge status-${dataset.validation_status?.toLowerCase() || 'pending'}">
                    ${dataset.validation_status || 'Pending'}
                </span>
            </td>
            <td>${formatDate(dataset.upload_date)}</td>
            <td>
                <div class="action-buttons">
                    <button class="btn btn-sm btn-primary" onclick="viewDataset('${dataset.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-sm btn-warning" onclick="editDataset('${dataset.id}')">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="deleteDataset('${dataset.id}')">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('id-ID', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

async function viewDataset(datasetId) {
    // Implementation for viewing dataset details
    console.log('Viewing dataset:', datasetId);
}

async function editDataset(datasetId) {
    // Implementation for editing dataset
    console.log('Editing dataset:', datasetId);
}

async function deleteDataset(datasetId) {
    if (confirm('Apakah Anda yakin ingin menghapus dataset ini?')) {
        try {
            const response = await fetch(`${API_BASE}/admin/training-data/${datasetId}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                showNotification('Dataset berhasil dihapus', 'success');
                loadTrainingData(); // Reload the table
                loadTrainingStats(); // Reload stats
            } else {
                showNotification('Gagal menghapus dataset: ' + (data.message || 'Unknown error'), 'error');
            }
        } catch (error) {
            console.error('Error deleting dataset:', error);
            showNotification('Error saat menghapus dataset: ' + error.message, 'error');
        }
    }
}

function setupDragAndDrop(dropArea, fileInput) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropArea.style.borderColor = 'var(--primary-color)';
        dropArea.style.backgroundColor = '#f0f9ff';
    }

    function unhighlight(e) {
        dropArea.style.borderColor = 'var(--border-color)';
        dropArea.style.backgroundColor = '';
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        // Set files to input
        fileInput.files = files;
        
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }
}

async function handleTrainingUpload(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    
    try {
        const response = await fetch(`${API_BASE}/admin/upload-training-data`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('Data training berhasil diupload!', 'success');
            event.target.reset();
            loadTrainingData();
        } else {
            throw new Error(data.detail || 'Gagal upload data training');
        }
    } catch (error) {
        console.error('Error uploading training data:', error);
        showNotification('Gagal upload data training: ' + error.message, 'error');
    }
}

function handleImageUpload(event) {
    const files = event.target.files;
    const preview = document.getElementById('imagePreview');
    
    if (files.length > 0 && preview) {
        preview.innerHTML = '';
        Array.from(files).forEach(file => {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.style.maxWidth = '100px';
            img.style.margin = '5px';
            preview.appendChild(img);
        });
    }
    
    // Update upload area text
    const uploadArea = document.getElementById('imageUploadArea');
    if (uploadArea && files.length > 0) {
        const fileText = files.length === 1 ? files[0].name : `${files.length} file terpilih`;
        uploadArea.querySelector('p').textContent = `File terpilih: ${fileText}`;
    }
}

function handleAnnotationUpload(event) {
    const files = event.target.files;
    const uploadArea = document.getElementById('annotationUploadArea');
    
    if (uploadArea && files.length > 0) {
        const fileText = files.length === 1 ? files[0].name : `${files.length} file terpilih`;
        uploadArea.querySelector('p').textContent = `File terpilih: ${fileText}`;
    }
}

// System Status
async function updateSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/admin/dashboard-stats`);
        const data = await response.json();
        
        if (data.success) {
            updateStatusIndicators(data.stats);
        }
    } catch (error) {
        console.error('Error updating system status:', error);
    }
}

function updateStatusIndicators(stats) {
    // Update various status indicators in the UI
    const statusEl = document.getElementById('systemStatus');
    if (statusEl) {
        statusEl.innerHTML = `
            <div class="status-item">
                <span class="status-label">Pending Validasi:</span>
                <span class="status-value">${stats.pending_validations || 0}</span>
            </div>
            <div class="status-item">
                <span class="status-label">Training Sampel:</span>
                <span class="status-value">${stats.training_samples || 0}</span>
            </div>
        `;
    }
}

// Utility Functions
function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('id-ID') + ' ' + date.toLocaleTimeString('id-ID');
}

function showLoading(target) {
    const element = typeof target === 'string' ? document.getElementById(target) : target;
    if (element) {
        element.classList.add('loading');
    }
}

function hideLoading(target) {
    const element = typeof target === 'string' ? document.getElementById(target) : target;
    if (element) {
        element.classList.remove('loading');
    }
}

function showError(target, message) {
    const element = typeof target === 'string' ? document.getElementById(target) : target;
    if (element) {
        element.innerHTML = `<div class="error-message">${message}</div>`;
    }
}

function showNotification(message, type = 'info') {
    // Create notification
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Add notification styles if not exist
const existingStyles = document.getElementById('notificationStyles');
if (!existingStyles) {
    const style = document.createElement('style');
    style.id = 'notificationStyles';
    style.textContent = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            z-index: 1001;
            display: flex;
            align-items: center;
            gap: 1rem;
            max-width: 300px;
            border-left: 4px solid #3498db;
        }
        
        .notification-success {
            border-left-color: #2ecc71;
            background: #d4edda;
            color: #155724;
        }
        
        .notification-error {
            border-left-color: #e74c3c;
            background: #f8d7da;
            color: #721c24;
        }
        
        .notification button {
            background: none;
            border: none;
            font-size: 1.2rem;
            cursor: pointer;
            opacity: 0.7;
        }
        
        .loading {
            opacity: 0.5;
            pointer-events: none;
        }
        
        .error-message {
            color: #e74c3c;
            padding: 1rem;
            text-align: center;
        }
        
        .radio-label {
            display: inline-flex;
            align-items: center;
            margin-right: 1rem;
            cursor: pointer;
        }
        
        .radio-label input {
            margin-right: 0.5rem;
        }
        
        .approve {
            color: #2ecc71;
            font-weight: bold;
        }
        
        .reject {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .correction-section {
            border: 2px dashed #bdc3c7;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        
        .detail-section {
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .detail-section:last-child {
            border-bottom: none;
        }
        
        .risk-critical { color: #e74c3c; font-weight: bold; }
        .risk-high { color: #f39c12; font-weight: bold; }
        .risk-medium { color: #f1c40f; font-weight: bold; }
        .risk-low { color: #2ecc71; font-weight: bold; }
        
        .urgency-status.overdue { 
            background: #e74c3c; 
            color: white; 
            padding: 0.25rem 0.5rem; 
            border-radius: 4px; 
            font-size: 0.8rem;
        }
        
        .urgency-status.urgent { 
            background: #f39c12; 
            color: white; 
            padding: 0.25rem 0.5rem; 
            border-radius: 4px; 
            font-size: 0.8rem;
        }
        `;
    document.head.appendChild(style);
}

// Reports Management Functions
async function loadReportsQueue() {
    try {
        console.log('Loading reports queue...');
        const response = await fetch(`${API_BASE}/admin/reports/pending`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        reportsQueue = data.reports || [];
        
        console.log('Reports loaded:', reportsQueue.length);
        renderReportsQueue();
        
    } catch (error) {
        console.error('Error loading reports:', error);
        showNotification('Error memuat laporan: ' + error.message, 'error');
    }
}

function renderReportsQueue() {
    const container = document.getElementById('reportsQueue');
    if (!container) {
        console.error('Reports container not found');
        return;
    }
    
    if (reportsQueue.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox fa-3x"></i>
                <h3>Tidak ada laporan yang perlu direview</h3>
                <p>Semua laporan telah diproses atau tidak ada laporan baru.</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = reportsQueue.map(report => createReportCard(report)).join('');
}

function createReportCard(report) {
    const timeAgo = formatTimeAgo(report.reported_at);
    const riskClass = report.ai_risk_level ? report.ai_risk_level.toLowerCase() : 'unknown';
    
    return `
        <div class="report-item" data-report-id="${report.report_id}">
            <div class="report-header">
                <div class="report-info">
                    <h4 class="report-title">${report.description || 'Laporan Kerusakan'}</h4>
                    <div class="report-meta">
                        <span><i class="fas fa-user"></i> ${report.reporter_name || 'User'}</span>
                        <span><i class="fas fa-clock"></i> ${timeAgo}</span>
                        <span><i class="fas fa-map-marker-alt"></i> ${report.asset_name || 'Unknown Asset'}</span>
                        <span class="risk-${riskClass}"><i class="fas fa-exclamation-triangle"></i> ${report.ai_risk_level || 'UNKNOWN'} Risk</span>
                    </div>
                </div>
                <div class="report-actions">
                    <button class="btn btn-sm btn-primary" onclick="toggleRiskEditor(${report.report_id})">
                        <i class="fas fa-edit"></i> Edit Risk
                    </button>
                    ${report.ai_procedures && Array.isArray(report.ai_procedures) && report.ai_procedures.length > 0 ? `
                        <button class="btn btn-sm btn-info" onclick="showProcedures(${report.report_id})">
                            <i class="fas fa-tools"></i> Prosedur Perbaikan
                        </button>
                    ` : ''}
                    ${report.image_path ? `<img src="/${report.image_path}" class="report-image" alt="Damage image">` : ''}
                </div>
            </div>
            
            <div class="ai-analysis">
                <h4><i class="fas fa-robot"></i> Analisis AI</h4>
                <p><strong>Kerusakan Terdeteksi:</strong> ${report.ai_detected_damage || 'Tidak terdeteksi'}</p>
                <p><strong>Tingkat Keyakinan:</strong> ${report.ai_confidence ? (report.ai_confidence * 100).toFixed(1) + '%' : 'N/A'}</p>
            </div>
            
            <!-- Prosedur Perbaikan Modal -->
            <div id="proceduresModal_${report.report_id}" class="procedures-modal" style="display: none;">
                <div class="procedures-content">
                    <div class="procedures-header">
                        <h4><i class="fas fa-tools"></i> Prosedur Perbaikan</h4>
                        <button class="close-procedures" onclick="closeProcedures(${report.report_id})">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="procedures-list">
                        ${report.ai_procedures && Array.isArray(report.ai_procedures) ? 
                            report.ai_procedures.map((proc, index) => `
                                <div class="procedure-item">
                                    <span class="procedure-number">${index + 1}</span>
                                    <span class="procedure-text">${proc.description || proc}</span>
                                </div>
                            `).join('') :
                            '<p>Tidak ada prosedur perbaikan tersedia</p>'
                        }
                    </div>
                </div>
            </div>
            
            <div id="riskEditor_${report.report_id}" class="risk-level-editor" style="display: none;">
                <h4><i class="fas fa-sliders-h"></i> Edit Risk Level</h4>
                <div class="risk-level-controls">
                    <button type="button" class="risk-level-btn low ${report.ai_risk_level === 'LOW' ? 'active' : ''}" 
                            data-risk="LOW" onclick="selectRiskLevel(${report.report_id}, 'LOW')">
                        <i class="fas fa-check-circle"></i> Low
                    </button>
                    <button type="button" class="risk-level-btn medium ${report.ai_risk_level === 'MEDIUM' ? 'active' : ''}" 
                            data-risk="MEDIUM" onclick="selectRiskLevel(${report.report_id}, 'MEDIUM')">
                        <i class="fas fa-exclamation-triangle"></i> Medium
                    </button>
                    <button type="button" class="risk-level-btn high ${report.ai_risk_level === 'HIGH' ? 'active' : ''}" 
                            data-risk="HIGH" onclick="selectRiskLevel(${report.report_id}, 'HIGH')">
                        <i class="fas fa-times-circle"></i> High
                    </button>
                </div>
                <div class="risk-edit-actions">
                    <button class="btn-save-risk" onclick="saveRiskLevel(${report.report_id})">
                        <i class="fas fa-save"></i> Simpan
                    </button>
                    <button class="btn-cancel-edit" onclick="cancelRiskEdit(${report.report_id})">
                        <i class="fas fa-times"></i> Batal
                    </button>
                </div>
            </div>
        </div>
    `;
}

function toggleRiskEditor(reportId) {
    const editor = document.getElementById(`riskEditor_${reportId}`);
    const isVisible = editor.style.display !== 'none';
    
    // Hide all other editors
    document.querySelectorAll('.risk-level-editor').forEach(el => {
        el.style.display = 'none';
    });
    
    if (!isVisible) {
        editor.style.display = 'block';
        editingReportId = reportId;
    } else {
        editingReportId = null;
    }
}

function selectRiskLevel(reportId, riskLevel) {
    // Remove active class from all buttons in this editor
    const editor = document.getElementById(`riskEditor_${reportId}`);
    editor.querySelectorAll('.risk-level-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Add active class to selected button
    const selectedBtn = editor.querySelector(`[data-risk="${riskLevel}"]`);
    if (selectedBtn) {
        selectedBtn.classList.add('active');
    }
}

async function saveRiskLevel(reportId) {
    try {
        // Get selected risk level
        const editor = document.getElementById(`riskEditor_${reportId}`);
        const activeBtn = editor.querySelector('.risk-level-btn.active');
        
        if (!activeBtn) {
            showNotification('Pilih risk level terlebih dahulu', 'error');
            return;
        }
        
        const newRiskLevel = activeBtn.getAttribute('data-risk');
        
        // Send update request
        const response = await fetch(`${API_BASE}/admin/update-risk-level/${reportId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                new_risk_level: newRiskLevel,
                notes: `Admin mengubah risk level ke ${newRiskLevel}`
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showNotification(`Risk level berhasil diubah ke ${newRiskLevel}`, 'success');
            
            // Hide editor
            editor.style.display = 'none';
            editingReportId = null;
            
            // Reload reports to show updated data
            loadReportsQueue();
        } else {
            throw new Error(result.message || 'Failed to update risk level');
        }
        
    } catch (error) {
        console.error('Error saving risk level:', error);
        showNotification('Error mengubah risk level: ' + error.message, 'error');
    }
}

function cancelRiskEdit(reportId) {
    const editor = document.getElementById(`riskEditor_${reportId}`);
    editor.style.display = 'none';
    editingReportId = null;
}

// Fungsi untuk menampilkan prosedur perbaikan
function showProcedures(reportId) {
    const modal = document.getElementById(`proceduresModal_${reportId}`);
    if (modal) {
        modal.style.display = 'block';
    }
}

// Fungsi untuk menutup prosedur perbaikan
function closeProcedures(reportId) {
    const modal = document.getElementById(`proceduresModal_${reportId}`);
    if (modal) {
        modal.style.display = 'none';
    }
}

function formatTimeAgo(timestamp) {
    const now = new Date();
    const reportTime = new Date(timestamp);
    const diffMs = now - reportTime;
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffDays > 0) {
        return `${diffDays} hari yang lalu`;
    } else if (diffHours > 0) {
        return `${diffHours} jam yang lalu`;
    } else {
        return 'Baru saja';
    }
}
