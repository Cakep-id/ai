-- FAQ NLP System Database Schema
-- Created: August 22, 2025

CREATE DATABASE IF NOT EXISTS faq_nlp_system;
USE faq_nlp_system;

-- Tabel untuk menyimpan dataset FAQ utama
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

-- Tabel untuk menyimpan variasi pertanyaan
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

-- Tabel untuk menyimpan log pencarian
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

-- Insert contoh data
INSERT INTO faq_dataset (question, answer, category) VALUES
('Bagaimana cara reset password?', 'Untuk reset password, klik "Lupa Password" di halaman login, masukkan email Anda, dan ikuti instruksi yang dikirim ke email.', 'account'),
('Apa itu machine learning?', 'Machine Learning adalah cabang dari artificial intelligence yang memungkinkan komputer belajar dan membuat keputusan tanpa diprogram secara eksplisit.', 'technology'),
('Bagaimana cara menghubungi customer service?', 'Anda dapat menghubungi customer service melalui email: support@company.com atau telepon: 0800-1234-5678 (Senin-Jumat, 09:00-17:00).', 'support');

-- Insert variasi pertanyaan
INSERT INTO faq_variations (faq_id, variation_question, similarity_score) VALUES
(1, 'Cara mengubah password', 0.8500),
(1, 'Lupa kata sandi', 0.9000),
(1, 'Reset kata sandi', 0.9500),
(2, 'Pengertian machine learning', 0.9200),
(2, 'Definisi ML', 0.7800),
(2, 'Apa yang dimaksud dengan machine learning', 0.9100),
(3, 'Kontak customer service', 0.8800),
(3, 'Hubungi CS', 0.7500),
(3, 'Customer support', 0.8200);
