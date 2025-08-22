// Admin Panel JavaScript
class AdminPanel {
    constructor() {
        this.currentUser = null;
        this.currentTab = 'dashboard';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAuthStatus();
    }

    setupEventListeners() {
        // Login form
        document.getElementById('login-form').addEventListener('submit', (e) => this.handleLogin(e));
        
        // Logout button
        document.getElementById('logout-btn').addEventListener('click', () => this.handleLogout());
        
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // FAQ management
        document.getElementById('admin-add-faq-form').addEventListener('submit', (e) => this.handleAddFAQ(e));
        document.getElementById('admin-train-btn').addEventListener('click', () => this.handleTrainModel());
        document.getElementById('admin-refresh-faqs').addEventListener('click', () => this.loadFAQs());
        
        // User interactions
        document.getElementById('refresh-interactions').addEventListener('click', () => this.loadUserInteractions());
        
        // Settings
        document.getElementById('save-settings').addEventListener('click', () => this.saveSettings());
        document.getElementById('default-threshold').addEventListener('input', (e) => {
            document.getElementById('threshold-display').textContent = e.target.value;
        });
    }

    async checkAuthStatus() {
        const token = localStorage.getItem('admin_token');
        if (token) {
            try {
                const response = await fetch('/api/auth/verify', {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    if (data.user.role === 'admin') {
                        this.currentUser = data.user;
                        this.showAdminPanel();
                        this.loadDashboard();
                        return;
                    }
                }
            } catch (error) {
                console.error('Auth verification failed:', error);
            }
        }
        
        this.showLoginModal();
    }

    showLoginModal() {
        document.getElementById('login-modal').classList.add('show');
        document.getElementById('admin-container').style.display = 'none';
    }

    showAdminPanel() {
        document.getElementById('login-modal').classList.remove('show');
        document.getElementById('admin-container').style.display = 'block';
        document.getElementById('admin-name').textContent = `Welcome, ${this.currentUser.username}`;
    }

    async handleLogin(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        
        this.showLoading();
        
        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                if (data.user.role !== 'admin') {
                    this.showToast('Access denied. Admin privileges required.', 'error');
                    return;
                }
                
                localStorage.setItem('admin_token', data.token);
                this.currentUser = data.user;
                this.showAdminPanel();
                this.loadDashboard();
                this.showToast('Login successful!', 'success');
            } else {
                this.showToast(data.message || 'Login failed', 'error');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showToast('Login failed. Please try again.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    handleLogout() {
        localStorage.removeItem('admin_token');
        this.currentUser = null;
        this.showLoginModal();
        this.showToast('Logged out successfully', 'success');
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        this.currentTab = tabName;
        
        // Load tab-specific content
        switch (tabName) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'faq-management':
                this.loadFAQs();
                break;
            case 'user-interactions':
                this.loadUserInteractions();
                break;
            case 'language-analysis':
                this.loadLanguageAnalysis();
                break;
        }
    }

    async loadDashboard() {
        try {
            const token = localStorage.getItem('admin_token');
            const response = await fetch('/api/admin/statistics', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateDashboardStats(data);
                this.loadRecentSearches();
            }
        } catch (error) {
            console.error('Failed to load dashboard:', error);
            this.showToast('Failed to load dashboard data', 'error');
        }
    }

    updateDashboardStats(data) {
        document.getElementById('total-faqs').textContent = data.total_faqs || 0;
        document.getElementById('total-searches').textContent = data.total_searches || 0;
        document.getElementById('active-users').textContent = data.active_users || 0;
        document.getElementById('success-rate').textContent = `${data.success_rate || 0}%`;
    }

    async loadRecentSearches() {
        try {
            const token = localStorage.getItem('admin_token');
            const response = await fetch('/api/admin/recent-searches', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (response.ok) {
                const searches = await response.json();
                this.displayRecentSearches(searches);
            }
        } catch (error) {
            console.error('Failed to load recent searches:', error);
        }
    }

    displayRecentSearches(searches) {
        const container = document.getElementById('recent-searches');
        
        if (searches.length === 0) {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-search"></i><p>No recent searches</p></div>';
            return;
        }
        
        container.innerHTML = searches.map(search => `
            <div class="search-item">
                <div class="search-query">${this.escapeHtml(search.query)}</div>
                <div class="search-result">${search.found ? 'Found' : 'Not Found'}</div>
                <div class="search-time">${new Date(search.created_at).toLocaleString()}</div>
            </div>
        `).join('');
    }

    async handleAddFAQ(e) {
        e.preventDefault();
        
        const question = document.getElementById('admin-question').value.trim();
        const answer = document.getElementById('admin-answer').value.trim();
        const category = document.getElementById('admin-category').value;
        const variations = document.getElementById('admin-variations').value.trim();
        
        if (!question || !answer) {
            this.showToast('Question and answer are required', 'error');
            return;
        }
        
        this.showLoading();
        
        try {
            const token = localStorage.getItem('admin_token');
            const response = await fetch('/api/admin/add-faq', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    question,
                    answer,
                    category,
                    variations: variations ? variations.split('\n').filter(v => v.trim()) : []
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.showToast('FAQ added successfully!', 'success');
                document.getElementById('admin-add-faq-form').reset();
                this.loadFAQs();
            } else {
                this.showToast(data.message || 'Failed to add FAQ', 'error');
            }
        } catch (error) {
            console.error('Add FAQ error:', error);
            this.showToast('Failed to add FAQ', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async handleTrainModel() {
        this.showLoading();
        
        try {
            const token = localStorage.getItem('admin_token');
            const response = await fetch('/api/admin/train', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.showToast('Model trained successfully!', 'success');
            } else {
                this.showToast(data.message || 'Training failed', 'error');
            }
        } catch (error) {
            console.error('Training error:', error);
            this.showToast('Training failed', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async loadFAQs() {
        try {
            const token = localStorage.getItem('admin_token');
            const response = await fetch('/api/admin/faqs', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (response.ok) {
                const faqs = await response.json();
                this.displayFAQs(faqs);
            }
        } catch (error) {
            console.error('Failed to load FAQs:', error);
            this.showToast('Failed to load FAQs', 'error');
        }
    }

    displayFAQs(faqs) {
        const container = document.getElementById('admin-faq-list');
        
        if (faqs.length === 0) {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-question-circle"></i><h3>No FAQs found</h3><p>Start by adding your first FAQ</p></div>';
            return;
        }
        
        container.innerHTML = faqs.map(faq => `
            <div class="faq-item">
                <div class="faq-header">
                    <h3 class="faq-question">${this.escapeHtml(faq.question)}</h3>
                    <span class="faq-category">${faq.category || 'General'}</span>
                </div>
                <p class="faq-answer">${this.escapeHtml(faq.answer)}</p>
                ${faq.variations && faq.variations.length > 0 ? `
                    <div class="faq-variations">
                        <h4>Variasi Pertanyaan:</h4>
                        <ul>
                            ${faq.variations.map(v => `<li>${this.escapeHtml(v.variation_text)}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                <div class="faq-actions">
                    <button class="btn btn-sm btn-primary" onclick="adminPanel.editFAQ(${faq.id})">
                        <i class="fas fa-edit"></i> Edit
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="adminPanel.deleteFAQ(${faq.id})">
                        <i class="fas fa-trash"></i> Delete
                    </button>
                </div>
            </div>
        `).join('');
    }

    async deleteFAQ(faqId) {
        if (!confirm('Are you sure you want to delete this FAQ?')) {
            return;
        }
        
        try {
            const token = localStorage.getItem('admin_token');
            const response = await fetch(`/api/admin/faq/${faqId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (response.ok) {
                this.showToast('FAQ deleted successfully', 'success');
                this.loadFAQs();
            } else {
                this.showToast('Failed to delete FAQ', 'error');
            }
        } catch (error) {
            console.error('Delete FAQ error:', error);
            this.showToast('Failed to delete FAQ', 'error');
        }
    }

    async loadUserInteractions() {
        try {
            const token = localStorage.getItem('admin_token');
            const response = await fetch('/api/admin/user-interactions', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (response.ok) {
                const interactions = await response.json();
                this.displayUserInteractions(interactions);
            }
        } catch (error) {
            console.error('Failed to load user interactions:', error);
            this.showToast('Failed to load user interactions', 'error');
        }
    }

    displayUserInteractions(interactions) {
        const container = document.getElementById('user-interactions-list');
        
        if (interactions.length === 0) {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-comments"></i><h3>No interactions found</h3><p>User interactions will appear here</p></div>';
            return;
        }
        
        container.innerHTML = interactions.map(interaction => `
            <div class="interaction-item">
                <div class="interaction-header">
                    <span class="interaction-user">User: ${this.escapeHtml(interaction.user_id || 'Anonymous')}</span>
                    <span class="interaction-time">${new Date(interaction.created_at).toLocaleString()}</span>
                </div>
                <div class="interaction-query">"${this.escapeHtml(interaction.query)}"</div>
                <div class="interaction-response">${this.escapeHtml(interaction.response)}</div>
                ${interaction.language_style ? `
                    <div class="language-style">
                        Style: ${this.escapeHtml(JSON.stringify(interaction.language_style))}
                    </div>
                ` : ''}
            </div>
        `).join('');
    }

    async loadLanguageAnalysis() {
        try {
            const token = localStorage.getItem('admin_token');
            const response = await fetch('/api/admin/language-analysis', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (response.ok) {
                const analysis = await response.json();
                this.displayLanguageAnalysis(analysis);
            }
        } catch (error) {
            console.error('Failed to load language analysis:', error);
            this.showToast('Failed to load language analysis', 'error');
        }
    }

    displayLanguageAnalysis(analysis) {
        const container = document.getElementById('language-analysis');
        
        if (!analysis || Object.keys(analysis).length === 0) {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-brain"></i><h3>No language data</h3><p>Language analysis will appear here as users interact</p></div>';
            return;
        }
        
        container.innerHTML = `
            <div class="language-analysis">
                <div class="analysis-card">
                    <h3><i class="fas fa-chart-bar"></i> Formality Distribution</h3>
                    <div class="style-metric">
                        <span class="metric-label">Formal</span>
                        <span class="metric-value">${analysis.formality?.formal || 0}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${analysis.formality?.formal || 0}%"></div>
                    </div>
                    <div class="style-metric">
                        <span class="metric-label">Informal</span>
                        <span class="metric-value">${analysis.formality?.informal || 0}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${analysis.formality?.informal || 0}%"></div>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3><i class="fas fa-heart"></i> Emotion Analysis</h3>
                    <div class="style-metric">
                        <span class="metric-label">Positive</span>
                        <span class="metric-value">${analysis.emotion?.positive || 0}%</span>
                    </div>
                    <div class="style-metric">
                        <span class="metric-label">Neutral</span>
                        <span class="metric-value">${analysis.emotion?.neutral || 0}%</span>
                    </div>
                    <div class="style-metric">
                        <span class="metric-label">Negative</span>
                        <span class="metric-value">${analysis.emotion?.negative || 0}%</span>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3><i class="fas fa-language"></i> Common Patterns</h3>
                    ${analysis.patterns ? analysis.patterns.map(pattern => `
                        <div class="style-metric">
                            <span class="metric-label">${this.escapeHtml(pattern.pattern)}</span>
                            <span class="metric-value">${pattern.frequency}</span>
                        </div>
                    `).join('') : '<p>No patterns detected yet</p>'}
                </div>
            </div>
        `;
    }

    async saveSettings() {
        const settings = {
            default_threshold: parseFloat(document.getElementById('default-threshold').value),
            max_results: parseInt(document.getElementById('max-results').value),
            language_learning_enabled: document.getElementById('language-learning-enabled').checked
        };
        
        try {
            const token = localStorage.getItem('admin_token');
            const response = await fetch('/api/admin/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(settings)
            });
            
            if (response.ok) {
                this.showToast('Settings saved successfully!', 'success');
            } else {
                this.showToast('Failed to save settings', 'error');
            }
        } catch (error) {
            console.error('Save settings error:', error);
            this.showToast('Failed to save settings', 'error');
        }
    }

    showLoading() {
        document.getElementById('loading').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas ${type === 'success' ? 'fa-check' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
                <span>${message}</span>
            </div>
            <button class="toast-close">&times;</button>
        `;
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
        
        // Manual close
        toast.querySelector('.toast-close').addEventListener('click', () => {
            toast.remove();
        });
        
        // Animate in
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize admin panel when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.adminPanel = new AdminPanel();
});

// Additional helper functions
function formatDateTime(dateString) {
    return new Date(dateString).toLocaleString('id-ID', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function truncateText(text, maxLength = 100) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}
