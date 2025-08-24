// AgentV2 Frontend JavaScript Utilities

// Guidebook Management
class GuidebookManager {
    constructor() {
        this.isVisible = false;
        this.init();
    }

    init() {
        // Add smooth scrolling to guidebook section
        this.addSmoothScrolling();
        
        // Add intersection observer for animations
        this.setupAnimations();
        
        // Add copy to clipboard functionality
        this.setupCopyFunctionality();
    }

    addSmoothScrolling() {
        // Add scroll to guidebook button if it doesn't exist
        const scrollButton = document.getElementById('scroll-to-guide');
        if (scrollButton) {
            scrollButton.addEventListener('click', () => {
                const guidebook = document.getElementById('guidebook-section') || 
                                  document.getElementById('trainer-guidebook');
                if (guidebook) {
                    guidebook.scrollIntoView({ 
                        behavior: 'smooth',
                        block: 'start' 
                    });
                }
            });
        }
    }

    setupAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    
                    // Stagger animation for card grids
                    if (entry.target.classList.contains('guide-grid')) {
                        const cards = entry.target.querySelectorAll('.guide-card');
                        cards.forEach((card, index) => {
                            setTimeout(() => {
                                card.classList.add('slide-in-left');
                            }, index * 100);
                        });
                    }
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        // Observe guidebook sections
        const guidebookSections = document.querySelectorAll(
            '.guidebook-section, .guide-grid, .workflow-section, .best-practices'
        );
        
        guidebookSections.forEach(section => {
            observer.observe(section);
        });
    }

    setupCopyFunctionality() {
        // Add copy buttons to code snippets or important text
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('copy-btn')) {
                const textToCopy = e.target.getAttribute('data-copy');
                if (textToCopy) {
                    this.copyToClipboard(textToCopy);
                    this.showCopySuccess(e.target);
                }
            }
        });
    }

    copyToClipboard(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                console.log('Text copied to clipboard');
            });
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
        }
    }

    showCopySuccess(button) {
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i> Copied!';
        button.style.background = '#27ae60';
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.style.background = '';
        }, 2000);
    }

    toggleGuidebook() {
        const guidebook = document.getElementById('guidebook-section') || 
                          document.getElementById('trainer-guidebook');
        
        if (guidebook) {
            this.isVisible = !this.isVisible;
            guidebook.style.display = this.isVisible ? 'block' : 'none';
            
            if (this.isVisible) {
                guidebook.scrollIntoView({ behavior: 'smooth' });
            }
        }
    }
}

// Search Functionality for Guidebook
class GuideSearch {
    constructor() {
        this.setupSearch();
    }

    setupSearch() {
        // Add search input if it doesn't exist
        this.createSearchInput();
        
        // Setup search functionality
        this.setupSearchLogic();
    }

    createSearchInput() {
        const guidebookSection = document.getElementById('guidebook-section') || 
                                 document.getElementById('trainer-guidebook');
        
        if (guidebookSection && !document.getElementById('guide-search')) {
            const searchContainer = document.createElement('div');
            searchContainer.style.cssText = `
                text-align: center;
                margin-bottom: 2rem;
                padding: 1rem;
            `;
            
            searchContainer.innerHTML = `
                <div style="max-width: 400px; margin: 0 auto; position: relative;">
                    <input type="text" id="guide-search" placeholder="Search guidebook..." 
                           style="width: 100%; padding: 12px 45px 12px 15px; border: 2px solid #ddd; 
                                  border-radius: 25px; font-size: 14px; outline: none; transition: all 0.3s ease;">
                    <i class="fas fa-search" style="position: absolute; right: 15px; top: 50%; 
                                                    transform: translateY(-50%); color: #666;"></i>
                </div>
            `;
            
            const title = guidebookSection.querySelector('h2');
            if (title) {
                title.insertAdjacentElement('afterend', searchContainer);
            }
        }
    }

    setupSearchLogic() {
        const searchInput = document.getElementById('guide-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.performSearch(e.target.value.toLowerCase());
            });
        }
    }

    performSearch(query) {
        const guideCards = document.querySelectorAll('.guide-card');
        
        guideCards.forEach(card => {
            const text = card.textContent.toLowerCase();
            const isMatch = text.includes(query) || query === '';
            
            card.style.display = isMatch ? 'block' : 'none';
            card.style.transition = 'all 0.3s ease';
            
            if (isMatch && query !== '') {
                // Highlight matching text
                this.highlightText(card, query);
            } else {
                // Remove highlights
                this.removeHighlight(card);
            }
        });
    }

    highlightText(element, query) {
        // Simple highlighting implementation
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const textNodes = [];
        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }

        textNodes.forEach(textNode => {
            const text = textNode.textContent;
            const regex = new RegExp(`(${query})`, 'gi');
            if (regex.test(text)) {
                const span = document.createElement('span');
                span.innerHTML = text.replace(regex, '<mark style="background: #ffeb3b; padding: 2px;">$1</mark>');
                textNode.parentNode.replaceChild(span, textNode);
            }
        });
    }

    removeHighlight(element) {
        const marks = element.querySelectorAll('mark');
        marks.forEach(mark => {
            mark.outerHTML = mark.innerHTML;
        });
    }
}

// Progress Tracking for Training/Validation
class ProgressTracker {
    constructor() {
        this.setupProgressBars();
    }

    setupProgressBars() {
        // Setup progress tracking for admin validation
        this.trackValidationProgress();
        
        // Setup progress tracking for trainer sessions
        this.trackTrainingProgress();
    }

    trackValidationProgress() {
        const progressContainer = document.getElementById('validation-progress');
        if (progressContainer) {
            this.updateValidationMetrics();
            setInterval(() => this.updateValidationMetrics(), 30000); // Update every 30 seconds
        }
    }

    trackTrainingProgress() {
        const trainingContainer = document.getElementById('training-progress');
        if (trainingContainer) {
            this.updateTrainingMetrics();
            setInterval(() => this.updateTrainingMetrics(), 10000); // Update every 10 seconds
        }
    }

    updateValidationMetrics() {
        // Fetch validation metrics from API
        fetch('/api/admin/dashboard')
            .then(response => response.json())
            .then(data => {
                // Update progress indicators
                this.updateProgressBar('validation-rate', data.validation_rate || 0);
                this.updateMetric('pending-reports', data.pending_reports || 0);
                this.updateMetric('approved-today', data.approved_today || 0);
            })
            .catch(error => console.log('Validation metrics update failed:', error));
    }

    updateTrainingMetrics() {
        // Fetch training metrics from API
        fetch('/api/trainer/training-sessions')
            .then(response => response.json())
            .then(data => {
                const activeSessions = data.training_sessions?.filter(s => s.status === 'running') || [];
                if (activeSessions.length > 0) {
                    const session = activeSessions[0];
                    this.updateProgressBar('training-progress', session.progress || 0);
                    this.updateMetric('current-epoch', session.current_epoch || 0);
                    this.updateMetric('training-loss', session.current_loss || 0);
                }
            })
            .catch(error => console.log('Training metrics update failed:', error));
    }

    updateProgressBar(elementId, percentage) {
        const progressBar = document.getElementById(elementId);
        if (progressBar) {
            progressBar.style.width = `${Math.min(percentage, 100)}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
        }
    }

    updateMetric(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize guidebook manager
    window.guidebookManager = new GuidebookManager();
    
    // Initialize search functionality
    window.guideSearch = new GuideSearch();
    
    // Initialize progress tracking
    window.progressTracker = new ProgressTracker();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('guide-search');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to clear search
        if (e.key === 'Escape') {
            const searchInput = document.getElementById('guide-search');
            if (searchInput && searchInput === document.activeElement) {
                searchInput.value = '';
                window.guideSearch.performSearch('');
            }
        }
    });
});

// Export for external use
window.AgentV2Utils = {
    GuidebookManager,
    GuideSearch,
    ProgressTracker
};
