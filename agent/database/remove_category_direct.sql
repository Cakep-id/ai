-- Script untuk menghapus kolom kategori dari admin_training_data
-- Jalankan di phpMyAdmin atau MySQL client

USE cakep_ews;

-- Lihat struktur tabel saat ini
DESCRIBE admin_training_data;

-- Hapus foreign key constraints dulu (jika ada)
-- Ganti nama constraint sesuai dengan yang tampil di SHOW CREATE TABLE
SHOW CREATE TABLE admin_training_data;

-- Script untuk menghapus constraint dan kolom
-- Uncomment dan jalankan satu per satu setelah melihat nama constraint yang benar

-- ALTER TABLE admin_training_data DROP FOREIGN KEY admin_training_data_ibfk_2;
-- ALTER TABLE admin_training_data DROP FOREIGN KEY admin_training_data_ibfk_3;

-- Hapus kolom kategori
-- ALTER TABLE admin_training_data DROP COLUMN asset_category_id;
-- ALTER TABLE admin_training_data DROP COLUMN damage_type_id;

-- Verifikasi hasil
-- DESCRIBE admin_training_data;
