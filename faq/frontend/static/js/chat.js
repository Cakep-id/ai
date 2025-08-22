// Chat Interface JavaScript
class ChatInterface {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.messageHistory = [];
        this.isTyping = false;
        this.settings = {
            fontSize: 'medium',
            theme: 'light',
            soundNotifications: true
        };
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSettings();
        this.setupAutoResize();
        this.setupScrollToBottom();
        this.startSession();
    }

    generateSessionId() {
        return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    setupEventListeners() {
        // Chat form submission
        document.getElementById('chat-form').addEventListener('submit', (e) => this.handleSendMessage(e));
        
        // Input field events
        const inputField = document.getElementById('user-input');
        inputField.addEventListener('keydown', (e) => this.handleKeyDown(e));
        inputField.addEventListener('input', (e) => this.handleInputChange(e));
        
        // Quick action buttons
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleQuickAction(e));
        });
        
        // Chat actions
        document.getElementById('clear-chat').addEventListener('click', () => this.clearChat());
        document.getElementById('chat-settings').addEventListener('click', () => this.toggleSettings());
        
        // Settings
        document.getElementById('font-size-setting')?.addEventListener('change', (e) => this.changeFontSize(e.target.value));
        document.getElementById('theme-setting')?.addEventListener('change', (e) => this.changeTheme(e.target.value));
        document.getElementById('sound-notifications')?.addEventListener('change', (e) => this.toggleSound(e.target.checked));
        
        // Sidebar
        document.getElementById('close-sidebar')?.addEventListener('click', () => this.closeSidebar());
        
        // Scroll to bottom
        document.getElementById('scroll-to-bottom').addEventListener('click', () => this.scrollToBottom());
        
        // Window events
        window.addEventListener('beforeunload', () => this.endSession());
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    handleInputChange(e) {
        const charCount = e.target.value.length;
        document.querySelector('.char-count').textContent = `${charCount}/500`;
        
        // Auto resize textarea
        e.target.style.height = 'auto';
        e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
        
        // Enable/disable send button
        const sendBtn = document.getElementById('send-btn');
        sendBtn.disabled = charCount === 0 || this.isTyping;
    }

    setupAutoResize() {
        const inputField = document.getElementById('user-input');
        inputField.style.height = 'auto';
        inputField.style.height = inputField.scrollHeight + 'px';
    }

    setupScrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.addEventListener('scroll', () => {
            const scrollBtn = document.getElementById('scroll-to-bottom');
            const isNearBottom = messagesContainer.scrollTop + messagesContainer.clientHeight >= messagesContainer.scrollHeight - 100;
            scrollBtn.style.display = isNearBottom ? 'none' : 'block';
        });
    }

    async startSession() {
        try {
            const response = await fetch('/api/chat/start-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    user_agent: navigator.userAgent
                })
            });
            
            if (response.ok) {
                console.log('Chat session started successfully');
            }
        } catch (error) {
            console.error('Failed to start session:', error);
        }
    }

    async endSession() {
        try {
            await fetch('/api/chat/end-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });
        } catch (error) {
            console.error('Failed to end session:', error);
        }
    }

    handleSendMessage(e) {
        e.preventDefault();
        this.sendMessage();
    }

    handleQuickAction(e) {
        const question = e.target.closest('.quick-btn').dataset.question;
        if (question) {
            document.getElementById('user-input').value = question;
            this.sendMessage();
        }
    }

    async sendMessage() {
        const inputField = document.getElementById('user-input');
        const message = inputField.value.trim();
        
        if (!message || this.isTyping) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.messageHistory.push({ role: 'user', content: message });
        
        // Clear input and reset height
        inputField.value = '';
        inputField.style.height = 'auto';
        document.querySelector('.char-count').textContent = '0/500';
        document.getElementById('send-btn').disabled = true;
        
        // Hide quick actions after first message
        const quickActions = document.querySelector('.quick-actions');
        if (quickActions) {
            quickActions.style.display = 'none';
        }
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: message,
                    session_id: this.sessionId,
                    history: this.messageHistory.slice(-10) // Last 10 messages for context
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Show language learning indicator if AI is learning
                if (data.learning_active) {
                    this.showLearningIndicator();
                }
                
                this.addMessage(data.response, 'bot', data.confidence);
                this.messageHistory.push({ role: 'assistant', content: data.response });
                
                // Play notification sound
                if (this.settings.soundNotifications) {
                    this.playNotificationSound();
                }
            } else {
                this.addMessage('Maaf, saya mengalami masalah dalam memproses pertanyaan Anda. Silakan coba lagi.', 'bot');
                console.error('Chat error:', data.message);
            }
        } catch (error) {
            console.error('Network error:', error);
            this.addMessage('Maaf, terjadi kesalahan koneksi. Silakan periksa koneksi internet Anda dan coba lagi.', 'bot');
        } finally {
            this.hideTypingIndicator();
            document.getElementById('send-btn').disabled = false;
        }
    }

    addMessage(content, type, confidence = null) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const currentTime = new Date().toLocaleTimeString('id-ID', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        const avatar = type === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-robot"></i>';
        
        // Format content for better display
        const formattedContent = this.formatMessage(content);
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                ${avatar}
            </div>
            <div class="message-content">
                <div class="message-text">
                    ${formattedContent}
                    ${confidence ? `<div class="confidence-indicator">Confidence: ${Math.round(confidence * 100)}%</div>` : ''}
                </div>
                <div class="message-time">${currentTime}</div>
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Add entrance animation
        setTimeout(() => {
            messageDiv.classList.add('animate-in');
        }, 50);
    }

    formatMessage(content) {
        // Convert line breaks to <br>
        content = content.replace(/\n/g, '<br>');
        
        // Convert URLs to links
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        content = content.replace(urlRegex, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
        
        // Convert **text** to bold
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert *text* to italic
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        return content;
    }

    showTypingIndicator() {
        this.isTyping = true;
        document.getElementById('typing-indicator').style.display = 'block';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        document.getElementById('typing-indicator').style.display = 'none';
    }

    showLearningIndicator() {
        const indicator = document.getElementById('learning-indicator');
        indicator.style.display = 'flex';
        
        // Hide after 3 seconds
        setTimeout(() => {
            indicator.style.display = 'none';
        }, 3000);
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    clearChat() {
        if (confirm('Apakah Anda yakin ingin menghapus riwayat chat?')) {
            const messagesContainer = document.getElementById('chat-messages');
            // Keep welcome message and quick actions
            const welcomeMessage = messagesContainer.querySelector('.welcome-message').parentNode;
            const quickActions = messagesContainer.querySelector('.quick-actions');
            
            messagesContainer.innerHTML = '';
            messagesContainer.appendChild(welcomeMessage);
            if (quickActions) {
                messagesContainer.appendChild(quickActions);
                quickActions.style.display = 'block';
            }
            
            this.messageHistory = [];
            this.showToast('Riwayat chat telah dihapus', 'success');
        }
    }

    toggleSettings() {
        const sidebar = document.getElementById('chat-sidebar');
        sidebar.classList.toggle('open');
    }

    closeSidebar() {
        document.getElementById('chat-sidebar').classList.remove('open');
    }

    loadSettings() {
        const savedSettings = localStorage.getItem('chat-settings');
        if (savedSettings) {
            this.settings = { ...this.settings, ...JSON.parse(savedSettings) };
            this.applySettings();
        }
    }

    saveSettings() {
        localStorage.setItem('chat-settings', JSON.stringify(this.settings));
    }

    applySettings() {
        document.body.className = `chat-body theme-${this.settings.theme} font-${this.settings.fontSize}`;
        
        // Update UI controls
        if (document.getElementById('font-size-setting')) {
            document.getElementById('font-size-setting').value = this.settings.fontSize;
        }
        if (document.getElementById('theme-setting')) {
            document.getElementById('theme-setting').value = this.settings.theme;
        }
        if (document.getElementById('sound-notifications')) {
            document.getElementById('sound-notifications').checked = this.settings.soundNotifications;
        }
    }

    changeFontSize(size) {
        this.settings.fontSize = size;
        this.applySettings();
        this.saveSettings();
    }

    changeTheme(theme) {
        this.settings.theme = theme;
        this.applySettings();
        this.saveSettings();
    }

    toggleSound(enabled) {
        this.settings.soundNotifications = enabled;
        this.saveSettings();
    }

    playNotificationSound() {
        if (this.settings.soundNotifications) {
            // Create a simple notification sound
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0, audioContext.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.01);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.1);
        }
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
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 3000);
        
        // Manual close
        toast.querySelector('.toast-close').addEventListener('click', () => {
            toast.remove();
        });
        
        // Animate in
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
    }

    // Utility methods
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Accessibility features
    setupAccessibility() {
        // Add keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.altKey && e.key === 's') {
                e.preventDefault();
                this.toggleSettings();
            }
            if (e.altKey && e.key === 'c') {
                e.preventDefault();
                document.getElementById('user-input').focus();
            }
        });
        
        // Add ARIA labels
        document.getElementById('user-input').setAttribute('aria-label', 'Type your message here');
        document.getElementById('send-btn').setAttribute('aria-label', 'Send message');
    }

    // Error handling
    handleError(error, context = 'Unknown') {
        console.error(`Error in ${context}:`, error);
        this.showToast(`Terjadi kesalahan: ${error.message}`, 'error');
    }

    // Performance optimization
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Initialize chat interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
    
    // Setup accessibility features
    chatInterface.setupAccessibility();
    
    // Add CSS for animations and themes
    const style = document.createElement('style');
    style.textContent = `
        .message.animate-in {
            animation: messageSlideIn 0.4s ease-out;
        }
        
        .confidence-indicator {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 0.5rem;
            font-style: italic;
        }
        
        .bot-message .confidence-indicator {
            color: var(--text-light);
        }
        
        .theme-dark {
            filter: invert(1) hue-rotate(180deg);
        }
        
        .font-small {
            font-size: 0.9rem;
        }
        
        .font-large {
            font-size: 1.1rem;
        }
        
        .toast {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            display: flex;
            align-items: center;
            justify-content: space-between;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }
        
        .toast.show {
            transform: translateX(0);
        }
        
        .toast-success {
            border-left: 4px solid #10b981;
        }
        
        .toast-error {
            border-left: 4px solid #ef4444;
        }
        
        .toast-info {
            border-left: 4px solid var(--primary-color);
        }
        
        .toast-content {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .toast-close {
            background: none;
            border: none;
            font-size: 1.2rem;
            cursor: pointer;
            color: var(--text-light);
        }
    `;
    document.head.appendChild(style);
});

// Error handling for the entire application
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    if (window.chatInterface) {
        chatInterface.handleError(e.error, 'Global');
    }
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    if (window.chatInterface) {
        chatInterface.handleError(e.reason, 'Promise');
    }
});

// Export for testing purposes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatInterface;
}
