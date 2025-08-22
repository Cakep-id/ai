// User Interface JavaScript untuk CAKEP.id EWS

class UserDashboard {
    constructor() {
        this.init();
        this.setupEventListeners();
    }

    init() {
        // Initialize tabs
        this.showTab('report');
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.showTab(tabName);
            });
        });

        // Form submission
        const form = document.getElementById('damageReportForm');
        form.addEventListener('submit', (e) => this.handleReportSubmit(e));

        // Image upload
        const imageUpload = document.getElementById('imageUpload');
        const uploadArea = document.getElementById('uploadArea');
        
        imageUpload.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                imageUpload.files = files;
                this.handleImageUpload({ target: { files } });
            }
        });

        // Remove image
        document.getElementById('removeImage').addEventListener('click', () => {
            this.removeImage();
        });

        // Refresh buttons
        document.getElementById('refreshReportsBtn').addEventListener('click', () => {
            this.loadMyReports();
        });

        // Modal close
        document.getElementById('modalClose').addEventListener('click', () => {
            this.closeModal();
        });
    }

    showTab(tabName) {
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Remove active class from nav tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Show selected tab
        document.getElementById(tabName + 'Tab').classList.add('active');
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Load data for specific tabs
        if (tabName === 'my-reports') {
            this.loadMyReports();
        }
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showNotification('File harus berupa gambar', 'error');
            return;
        }

        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showNotification('Ukuran file maksimal 10MB', 'error');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('previewImage').src = e.target.result;
            document.getElementById('previewContainer').style.display = 'block';
            document.querySelector('.upload-content').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    removeImage() {
        document.getElementById('imageUpload').value = '';
        document.getElementById('previewContainer').style.display = 'none';
        document.querySelector('.upload-content').style.display = 'block';
    }

    async handleReportSubmit(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        
        // Validate required fields
        if (!formData.get('image')) {
            this.showNotification('Upload foto kerusakan', 'error');
            return;
        }
        
        if (!formData.get('description')) {
            this.showNotification('Berikan deskripsi kerusakan', 'error');
            return;
        }

        // Show loading state
        this.showLoading(true);
        document.getElementById('submitBtn').disabled = true;

        try {
            const response = await fetch('/api/user/report-damage', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Show AI analysis results
                this.showAIResults(data.ai_analysis);
                
                // Reset form
                form.reset();
                this.removeImage();
                
                this.showNotification('Laporan berhasil dikirim dan dianalisis AI!', 'success');
            } else {
                throw new Error(data.detail || 'Gagal mengirim laporan');
            }
        } catch (error) {
            console.error('Error submitting report:', error);
            this.showNotification(`Gagal mengirim laporan: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
            document.getElementById('submitBtn').disabled = false;
        }
    }

    showLoading(show) {
        const loadingState = document.getElementById('loadingState');
        const resultCard = document.getElementById('resultCard');
        
        if (show) {
            loadingState.style.display = 'block';
            resultCard.style.display = 'none';
        } else {
            loadingState.style.display = 'none';
        }
    }

    showAIResults(analysis) {
        const resultCard = document.getElementById('resultCard');
        const resultContent = document.getElementById('resultContent');
        
        const riskClass = `risk-${analysis.risk_level.toLowerCase()}`;
        
        resultContent.innerHTML = `
            <div class="ai-result">
                <div class="ai-metric">
                    <h4>Aset Terdeteksi</h4>
                    <div class="value">${analysis.detected_asset || 'Equipment'}</div>
                </div>
                <div class="ai-metric">
                    <h4>Tingkat Risiko</h4>
                    <div class="value ${riskClass}">${analysis.risk_level}</div>
                </div>
                <div class="ai-metric">
                    <h4>Tingkat Kepercayaan</h4>
                    <div class="value">${(analysis.confidence * 100).toFixed(0)}%</div>
                </div>
            </div>
            
            ${analysis.procedures && analysis.procedures.length > 0 ? `
                <div class="procedures">
                    <h4><i class="fas fa-tools"></i> Prosedur Perbaikan yang Disarankan:</h4>
                    <ol>
                        ${analysis.procedures.map(step => `<li>${step.description || step}</li>`).join('')}
                    </ol>
                </div>
            ` : ''}
            
            <div class="status-info">
                <p><i class="fas fa-info-circle"></i> Laporan Anda sedang menunggu validasi admin. Anda akan mendapat notifikasi setelah divalidasi.</p>
            </div>
        `;
        
        resultCard.style.display = 'block';
        resultCard.scrollIntoView({ behavior: 'smooth' });
    }

    async loadMyReports() {
        const loadingState = document.getElementById('reportsLoading');
        const reportsList = document.getElementById('reportsList');
        
        loadingState.style.display = 'block';
        reportsList.innerHTML = '';
        
        try {
            const response = await fetch('/api/user/my-reports');
            const data = await response.json();
            
            if (data.success) {
                this.renderReports(data.reports);
            } else {
                throw new Error('Gagal memuat laporan');
            }
        } catch (error) {
            console.error('Error loading reports:', error);
            reportsList.innerHTML = `
                <div class="error-state">
                    <p>Gagal memuat laporan. <button onclick="userDashboard.loadMyReports()">Coba lagi</button></p>
                </div>
            `;
        } finally {
            loadingState.style.display = 'none';
        }
    }

    renderReports(reports) {
        const reportsList = document.getElementById('reportsList');
        
        if (reports.length === 0) {
            reportsList.innerHTML = `
                <div class="empty-state">
                    <p>Belum ada laporan. <a href="#" onclick="userDashboard.showTab('report')">Buat laporan pertama</a></p>
                </div>
            `;
            return;
        }
        
        reportsList.innerHTML = reports.map(report => `
            <div class="report-item status-${report.admin_status}" onclick="userDashboard.showReportDetail(${report.report_id})">
                <div class="report-header">
                    <div>
                        <div class="report-title">${report.asset_name}</div>
                        <div class="report-meta">
                            ${new Date(report.reported_at).toLocaleDateString('id-ID')} • 
                            ${report.location_details || report.asset_location}
                        </div>
                    </div>
                    <span class="report-status status-${report.admin_status}">
                        ${this.getStatusText(report.admin_status)}
                    </span>
                </div>
                
                <div class="report-description">
                    ${report.description}
                </div>
                
                ${report.ai_detected_damage ? `
                    <div class="report-ai-info">
                        <span class="ai-damage">${report.ai_detected_damage}</span>
                        <span class="ai-risk risk-${report.ai_risk_level?.toLowerCase()}">${report.ai_risk_level}</span>
                    </div>
                ` : ''}
            </div>
        `).join('');
    }

    getStatusText(status) {
        const statusMap = {
            'pending': 'Menunggu Validasi',
            'approved': 'Disetujui',
            'rejected': 'Ditolak'
        };
        return statusMap[status] || status;
    }

    async showReportDetail(reportId) {
        try {
            const response = await fetch(`/api/user/report/${reportId}`);
            const data = await response.json();
            
            if (data.success) {
                this.renderReportModal(data.report);
                this.showModal();
            } else {
                throw new Error('Gagal memuat detail laporan');
            }
        } catch (error) {
            console.error('Error loading report detail:', error);
            this.showNotification('Gagal memuat detail laporan', 'error');
        }
    }

    renderReportModal(report) {
        const modalBody = document.getElementById('modalBody');
        
        modalBody.innerHTML = `
            <div class="report-detail">
                <div class="detail-section">
                    <h4>Informasi Aset</h4>
                    <p><strong>Nama:</strong> ${report.asset_name}</p>
                </div>
                
                <div class="detail-section">
                    <h4>Detail Laporan</h4>
                    <p><strong>Tanggal:</strong> ${new Date(report.reported_at).toLocaleString('id-ID')}</p>
                    <p><strong>Deskripsi:</strong> ${report.description}</p>
                </div>
                
                ${report.ai_detected_damage ? `
                    <div class="detail-section">
                        <h4>Hasil Analisis AI</h4>
                        <p><strong>Tingkat Risiko:</strong> <span class="risk-${report.ai_risk_level?.toLowerCase()}">${report.ai_risk_level}</span></p>
                        <p><strong>Kepercayaan:</strong> ${(report.ai_confidence * 100).toFixed(0)}%</p>
                        
                        ${report.ai_procedures && report.ai_procedures.length > 0 ? `
                            <div class="procedures">
                                <h5>Prosedur Perbaikan:</h5>
                                <ol>
                                    ${report.ai_procedures.map(step => `<li>${step.description || step}</li>`).join('')}
                                </ol>
                            </div>
                        ` : ''}
                    </div>
                ` : ''}
                
                <div class="detail-section">
                    <h4>Status Validasi</h4>
                    <p><strong>Status:</strong> <span class="status-${report.admin_status}">${this.getStatusText(report.admin_status)}</span></p>
                    ${report.validated_at ? `<p><strong>Divalidasi:</strong> ${new Date(report.validated_at).toLocaleString('id-ID')}</p>` : ''}
                    ${report.validated_by_name ? `<p><strong>Validator:</strong> ${report.validated_by_name}</p>` : ''}
                    ${report.admin_notes ? `<p><strong>Catatan Admin:</strong> ${report.admin_notes}</p>` : ''}
                </div>
                
                ${report.next_maintenance ? `
                    <div class="detail-section">
                        <h4>Jadwal Pemeliharaan</h4>
                        <p><strong>Tanggal:</strong> ${new Date(report.next_maintenance).toLocaleDateString('id-ID')}</p>
                    </div>
                ` : ''}
            </div>
        `;
    }

    showModal() {
        document.getElementById('reportModal').style.display = 'flex';
    }

    closeModal() {
        document.getElementById('reportModal').style.display = 'none';
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">×</button>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }
}

// Initialize when page loads
let userDashboard;
document.addEventListener('DOMContentLoaded', () => {
    userDashboard = new UserDashboard();
});

// Add notification styles
const style = document.createElement('style');
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
    
    .error-state, .empty-state {
        text-align: center;
        padding: 3rem;
        color: #7f8c8d;
    }
    
    .procedures {
        margin-top: 1rem;
    }
    
    .procedures h4, .procedures h5 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .procedures ol {
        padding-left: 1.5rem;
    }
    
    .procedures li {
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    
    .detail-section {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #ecf0f1;
    }
    
    .detail-section:last-child {
        border-bottom: none;
    }
    
    .detail-section h4 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    .detail-section p {
        margin-bottom: 0.5rem;
        color: #555;
    }
    
    .status-info {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .status-info p {
        margin: 0;
        color: #2c3e50;
    }
`;
document.head.appendChild(style);
