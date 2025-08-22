// FAQ NLP System JavaScript
class FAQApp {
    constructor() {
        this.baseUrl = window.location.origin;
        this.currentTab = 'search';
        this.init();
    }

    init() {
        this.initEventListeners();
        this.initTabs();
        this.loadCategories();
        this.updateThresholdValue();
        
        // Load initial data
        if (this.currentTab === 'manage') {
            this.loadFAQs();
        }
    }

    initEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Live Chat
        document.getElementById('send-message').addEventListener('click', () => {
            this.sendChatMessage();
        });

        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });

        document.getElementById('clear-chat').addEventListener('click', () => {
            this.clearChat();
        });

        // Chat suggestions
        document.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const text = e.target.dataset.text;
                document.getElementById('chat-input').value = text;
                this.sendChatMessage();
            });
        });

        // Search form
        document.getElementById('search-btn').addEventListener('click', () => {
            this.searchFAQ();
        });

        document.getElementById('search-query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchFAQ();
            }
        });

        // Threshold slider
        document.getElementById('similarity-threshold').addEventListener('input', () => {
            this.updateThresholdValue();
        });

        // Add FAQ form
        document.getElementById('add-faq-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addFAQ();
        });

        // Train button
        document.getElementById('train-btn').addEventListener('click', () => {
            this.trainModel();
        });

        // Refresh buttons
        document.getElementById('refresh-faqs').addEventListener('click', () => {
            this.loadFAQs();
        });

        document.getElementById('refresh-stats').addEventListener('click', () => {
            this.loadStatistics();
        });

        // Category filter
        document.getElementById('filter-category').addEventListener('change', () => {
            this.loadFAQs();
        });

        // Edit modal
        document.getElementById('edit-faq-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.updateFAQ();
        });

        // Modal close
        document.querySelector('.close').addEventListener('click', () => {
            this.closeModal();
        });

        document.querySelector('.cancel-btn').addEventListener('click', () => {
            this.closeModal();
        });

        // Close modal on outside click
        document.getElementById('edit-modal').addEventListener('click', (e) => {
            if (e.target.id === 'edit-modal') {
                this.closeModal();
            }
        });
    }

    initTabs() {
        // Show initial tab
        this.switchTab('chat');
    }

    switchTab(tabName) {
        // Update button states
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update content visibility
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        this.currentTab = tabName;

        // Load data for specific tabs
        if (tabName === 'manage') {
            this.loadFAQs();
        } else if (tabName === 'stats') {
            this.loadStatistics();
        }
    }

    // Live Chat Functions
    async sendChatMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;

        // Add user message to chat
        this.addMessageToChat(message, 'user');
        
        // Clear input
        input.value = '';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Search for FAQ
            const data = await this.apiRequest('/search', {
                method: 'POST',
                body: JSON.stringify({
                    query: message,
                    threshold: 0.2,
                    max_results: 1
                })
            });

            // Hide typing indicator
            this.hideTypingIndicator();

            if (data.results && data.results.length > 0) {
                const result = data.results[0];
                
                // Add bot response
                this.addMessageToChat(result.answer, 'bot', {
                    question: result.question,
                    similarity: result.similarity_score,
                    matchType: result.match_type
                });

                // Add suggestions for related questions
                this.addChatSuggestions([
                    'Apakah ada pertanyaan lain?',
                    'Bisakah dijelaskan lebih detail?',
                    'Ada informasi tambahan?'
                ]);
                
            } else {
                // No FAQ found, provide helpful response
                this.addMessageToChat(
                    'Maaf, saya belum memiliki informasi tentang pertanyaan tersebut. ' +
                    'Apakah Anda bisa menggunakan kata kunci yang berbeda atau menghubungi customer service untuk bantuan lebih lanjut?',
                    'bot'
                );

                this.addChatSuggestions([
                    'Bagaimana cara menghubungi customer service?',
                    'Coba tanyakan hal lain',
                    'Lihat daftar FAQ tersedia'
                ]);
            }

        } catch (error) {
            this.hideTypingIndicator();
            this.addMessageToChat(
                'Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi atau hubungi customer service.',
                'bot'
            );
            console.error('Chat error:', error);
        }
    }

    addMessageToChat(text, sender, metadata = null) {
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const currentTime = new Date().toLocaleTimeString('id-ID', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        let metadataHtml = '';
        if (metadata) {
            metadataHtml = `
                <div class="message-metadata">
                    <small>
                        Berdasarkan: "${this.escapeHtml(metadata.question)}" 
                        (${(metadata.similarity * 100).toFixed(1)}% match)
                    </small>
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(text)}</div>
                ${metadataHtml}
                <div class="message-time">${currentTime}</div>
            </div>
        `;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-text">
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                    Sedang mengetik...
                </div>
            </div>
        `;

        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    addChatSuggestions(suggestions) {
        const chatMessages = document.getElementById('chat-messages');
        const suggestionsDiv = document.createElement('div');
        suggestionsDiv.className = 'bot-suggestions';
        
        suggestionsDiv.innerHTML = suggestions.map(suggestion => 
            `<span class="bot-suggestion" onclick="app.handleSuggestionClick('${suggestion}')">${suggestion}</span>`
        ).join('');

        // Add to last bot message
        const lastBotMessage = chatMessages.querySelector('.bot-message:last-child .message-content');
        if (lastBotMessage) {
            lastBotMessage.appendChild(suggestionsDiv);
        }
    }

    handleSuggestionClick(suggestion) {
        document.getElementById('chat-input').value = suggestion;
        this.sendChatMessage();
    }

    clearChat() {
        const chatMessages = document.getElementById('chat-messages');
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">
                        Halo! Saya FAQ Assistant. Silakan tanyakan apapun yang ingin Anda ketahui. 
                        Saya akan membantu mencari jawaban yang tepat untuk Anda.
                    </div>
                    <div class="message-time">Sekarang</div>
                </div>
            </div>
        `;
    }

    updateThresholdValue() {
        const slider = document.getElementById('similarity-threshold');
        const valueDisplay = document.getElementById('threshold-value');
        valueDisplay.textContent = slider.value;
    }

    showLoading() {
        document.getElementById('loading').classList.add('show');
    }

    hideLoading() {
        document.getElementById('loading').classList.remove('show');
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };

        toast.innerHTML = `
            <i class="${icons[type]}"></i>
            <div class="toast-message">${message}</div>
            <span class="toast-close">&times;</span>
        `;

        // Add close functionality
        toast.querySelector('.toast-close').addEventListener('click', () => {
            toast.remove();
        });

        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }

    async apiRequest(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.baseUrl}/api${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || 'API request failed');
            }

            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    async searchFAQ() {
        const query = document.getElementById('search-query').value.trim();
        const threshold = parseFloat(document.getElementById('similarity-threshold').value);

        if (!query) {
            this.showToast('Masukkan pertanyaan untuk dicari', 'warning');
            return;
        }

        this.showLoading();

        try {
            const data = await this.apiRequest('/search', {
                method: 'POST',
                body: JSON.stringify({
                    query: query,
                    threshold: threshold,
                    max_results: 10
                })
            });

            this.displaySearchResults(data);
            
            if (data.results.length > 0) {
                this.showToast(`Ditemukan ${data.results.length} hasil`, 'success');
            } else {
                this.showToast('Tidak ada hasil yang ditemukan. Coba turunkan threshold atau gunakan kata kunci berbeda.', 'info');
            }

        } catch (error) {
            this.showToast(`Error pencarian: ${error.message}`, 'error');
            this.displaySearchResults({ results: [] });
        } finally {
            this.hideLoading();
        }
    }

    displaySearchResults(data) {
        const container = document.getElementById('search-results');
        
        if (!data.results || data.results.length === 0) {
            container.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-search"></i>
                    <h3>Tidak ada hasil ditemukan</h3>
                    <p>Coba gunakan kata kunci yang berbeda atau turunkan threshold similarity</p>
                </div>
            `;
            return;
        }

        container.innerHTML = data.results.map(result => `
            <div class="search-result">
                <div class="search-result-question">
                    <strong>Q:</strong> ${this.escapeHtml(result.question)}
                </div>
                <div class="search-result-answer">
                    <strong>A:</strong> ${this.escapeHtml(result.answer)}
                </div>
                <div class="search-result-meta">
                    <div>
                        <span class="faq-category">${result.category}</span>
                        <span class="match-type">${result.match_type === 'main' ? 'Pertanyaan Utama' : 'Variasi'}</span>
                    </div>
                    <div class="similarity-score">
                        ${(result.similarity_score * 100).toFixed(1)}% Match
                    </div>
                </div>
                ${result.matched_question !== result.question ? `
                    <div style="margin-top: 10px; font-size: 0.9rem; color: #718096;">
                        <em>Cocok dengan: "${this.escapeHtml(result.matched_question)}"</em>
                    </div>
                ` : ''}
            </div>
        `).join('');
    }

    async addFAQ() {
        const question = document.getElementById('question').value.trim();
        const answer = document.getElementById('answer').value.trim();
        const category = document.getElementById('category').value;
        const variations = document.getElementById('variations').value.trim();

        if (!question || !answer) {
            this.showToast('Pertanyaan dan jawaban harus diisi', 'warning');
            return;
        }

        this.showLoading();

        try {
            const variationList = variations ? 
                variations.split('\n').map(v => v.trim()).filter(v => v) : [];

            const data = await this.apiRequest('/faq', {
                method: 'POST',
                body: JSON.stringify({
                    question: question,
                    answer: answer,
                    category: category,
                    variations: variationList
                })
            });

            this.showToast(data.message, 'success');
            
            // Reset form
            document.getElementById('add-faq-form').reset();
            
            // Refresh FAQ list if on manage tab
            if (this.currentTab === 'manage') {
                this.loadFAQs();
            }

        } catch (error) {
            this.showToast(`Error menambah FAQ: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async loadFAQs() {
        this.showLoading();

        try {
            const category = document.getElementById('filter-category').value;
            const params = new URLSearchParams({
                include_variations: 'true'
            });
            
            if (category) {
                params.append('category', category);
            }

            const data = await this.apiRequest(`/faq?${params}`);
            this.displayFAQs(data.data || []);

        } catch (error) {
            this.showToast(`Error memuat FAQ: ${error.message}`, 'error');
            this.displayFAQs([]);
        } finally {
            this.hideLoading();
        }
    }

    displayFAQs(faqs) {
        const container = document.getElementById('faq-list');
        
        if (faqs.length === 0) {
            container.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-list"></i>
                    <h3>Belum ada FAQ</h3>
                    <p>Tambahkan FAQ pertama Anda untuk memulai</p>
                </div>
            `;
            return;
        }

        container.innerHTML = faqs.map(faq => `
            <div class="faq-item">
                <div class="faq-header">
                    <div class="faq-question">${this.escapeHtml(faq.question)}</div>
                    <div class="faq-actions">
                        <button class="btn btn-success" onclick="app.editFAQ(${faq.id})">
                            <i class="fas fa-edit"></i> Edit
                        </button>
                        <button class="btn btn-danger" onclick="app.deleteFAQ(${faq.id})">
                            <i class="fas fa-trash"></i> Hapus
                        </button>
                    </div>
                </div>
                <div class="faq-answer">${this.escapeHtml(faq.answer)}</div>
                <div class="faq-meta">
                    <span class="faq-category">${faq.category}</span>
                    <span>Dibuat: ${new Date(faq.created_at).toLocaleDateString('id-ID')}</span>
                </div>
                ${faq.variations && faq.variations.length > 0 ? `
                    <div class="faq-variations">
                        <h4>Variasi Pertanyaan:</h4>
                        <div class="variation-list">
                            ${faq.variations.map(v => `
                                <span class="variation-item">${this.escapeHtml(v.variation_question)}</span>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `).join('');
    }

    async editFAQ(faqId) {
        try {
            const data = await this.apiRequest(`/faq/${faqId}`);
            const faq = data.data;

            // Fill edit form
            document.getElementById('edit-faq-id').value = faq.id;
            document.getElementById('edit-question').value = faq.question;
            document.getElementById('edit-answer').value = faq.answer;
            document.getElementById('edit-category').value = faq.category;
            
            // Fill variations
            const variations = faq.variations ? 
                faq.variations.map(v => v.variation_question).join('\n') : '';
            document.getElementById('edit-variations').value = variations;

            // Show modal
            this.showModal();

        } catch (error) {
            this.showToast(`Error memuat FAQ: ${error.message}`, 'error');
        }
    }

    async updateFAQ() {
        const faqId = document.getElementById('edit-faq-id').value;
        const question = document.getElementById('edit-question').value.trim();
        const answer = document.getElementById('edit-answer').value.trim();
        const category = document.getElementById('edit-category').value;
        const variations = document.getElementById('edit-variations').value.trim();

        if (!question || !answer) {
            this.showToast('Pertanyaan dan jawaban harus diisi', 'warning');
            return;
        }

        this.showLoading();

        try {
            const variationList = variations ? 
                variations.split('\n').map(v => v.trim()).filter(v => v) : [];

            const data = await this.apiRequest(`/faq/${faqId}`, {
                method: 'PUT',
                body: JSON.stringify({
                    question: question,
                    answer: answer,
                    category: category,
                    variations: variationList
                })
            });

            this.showToast(data.message, 'success');
            this.closeModal();
            this.loadFAQs();

        } catch (error) {
            this.showToast(`Error update FAQ: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async deleteFAQ(faqId) {
        if (!confirm('Apakah Anda yakin ingin menghapus FAQ ini?')) {
            return;
        }

        this.showLoading();

        try {
            const data = await this.apiRequest(`/faq/${faqId}`, {
                method: 'DELETE'
            });

            this.showToast(data.message, 'success');
            this.loadFAQs();

        } catch (error) {
            this.showToast(`Error hapus FAQ: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async loadCategories() {
        try {
            const data = await this.apiRequest('/categories');
            const categories = data.data || [];
            
            // Update category selects
            const selects = [
                document.getElementById('category'),
                document.getElementById('edit-category'),
                document.getElementById('filter-category')
            ];

            selects.forEach((select, index) => {
                if (index === 2) { // filter dropdown
                    select.innerHTML = '<option value="">Semua Kategori</option>';
                } else {
                    select.innerHTML = '';
                }

                categories.forEach(cat => {
                    const option = document.createElement('option');
                    option.value = cat;
                    option.textContent = this.capitalizeFirst(cat);
                    select.appendChild(option);
                });

                // Set default for add form
                if (index === 0) {
                    select.value = 'general';
                }
            });

        } catch (error) {
            console.error('Error loading categories:', error);
        }
    }

    async loadStatistics() {
        this.showLoading();

        try {
            const data = await this.apiRequest('/statistics');
            this.displayStatistics(data.data);

        } catch (error) {
            this.showToast(`Error memuat statistik: ${error.message}`, 'error');
            this.displayStatistics(null);
        } finally {
            this.hideLoading();
        }
    }

    displayStatistics(stats) {
        const container = document.getElementById('statistics');
        
        if (!stats) {
            container.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-chart-bar"></i>
                    <h3>Error memuat statistik</h3>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-value">${stats.total_searches}</div>
                    <div class="stat-label">Total Pencarian</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.successful_searches}</div>
                    <div class="stat-label">Pencarian Berhasil</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.success_rate}%</div>
                    <div class="stat-label">Tingkat Keberhasilan</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.popular_faq_ids.length}</div>
                    <div class="stat-label">FAQ Populer</div>
                </div>
            </div>
            
            <div class="recent-searches">
                <h3>Pencarian Terbaru</h3>
                ${stats.recent_searches.slice(0, 10).map(search => `
                    <div class="search-log">
                        <div class="search-query">${this.escapeHtml(search.query)}</div>
                        <div class="search-status ${search.found_result ? 'found' : 'not-found'}">
                            ${search.found_result ? 'Ditemukan' : 'Tidak Ditemukan'}
                        </div>
                        <div style="font-size: 0.8rem; color: #718096;">
                            ${new Date(search.created_at).toLocaleString('id-ID')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    async trainModel() {
        if (!confirm('Apakah Anda yakin ingin melatih ulang model?')) {
            return;
        }

        this.showLoading();

        try {
            const data = await this.apiRequest('/train', {
                method: 'POST'
            });

            this.showToast(data.message, 'success');

        } catch (error) {
            this.showToast(`Error training model: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    showModal() {
        document.getElementById('edit-modal').classList.add('show');
    }

    closeModal() {
        document.getElementById('edit-modal').classList.remove('show');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FAQApp();
});

// Global functions for inline event handlers
window.editFAQ = (id) => window.app.editFAQ(id);
window.deleteFAQ = (id) => window.app.deleteFAQ(id);
