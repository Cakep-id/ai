-- LANGKAH 1: Jalankan query ini untuk melihat struktur tabel
DESCRIBE admin_training_data;

-- LANGKAH 2: Jalankan query ini untuk melihat foreign key constraints
SHOW CREATE TABLE admin_training_data;

-- LANGKAH 3: Setelah melihat hasil di atas, jalankan command berikut satu per satu
-- (ganti nama constraint sesuai hasil SHOW CREATE TABLE)

-- Contoh: jika constraint bernama admin_training_data_ibfk_2 dan admin_training_data_ibfk_3
ALTER TABLE admin_training_data DROP FOREIGN KEY admin_training_data_ibfk_2;
ALTER TABLE admin_training_data DROP FOREIGN KEY admin_training_data_ibfk_3;

-- Hapus kolom kategori
ALTER TABLE admin_training_data DROP COLUMN asset_category_id;
ALTER TABLE admin_training_data DROP COLUMN damage_type_id;

-- LANGKAH 4: Verifikasi hasil akhir
DESCRIBE admin_training_data;
