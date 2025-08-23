-- Migration untuk menghapus kategori dari admin_training_data
-- Untuk MySQL yang kompatibel dengan versi lama

USE cakep_ews;

-- Step 1: Backup existing data (uncomment if needed)
-- CREATE TABLE admin_training_data_backup AS SELECT * FROM admin_training_data;

-- Step 2: Add new columns (dengan pengecekan apakah sudah ada)
SET sql_mode = '';

-- Check dan add filename column jika belum ada
SET @col_exists = (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
                   WHERE TABLE_SCHEMA = 'cakep_ews' 
                   AND TABLE_NAME = 'admin_training_data' 
                   AND COLUMN_NAME = 'filename');

SET @sql = IF(@col_exists = 0, 
              'ALTER TABLE admin_training_data ADD COLUMN filename VARCHAR(255) DEFAULT "";',
              'SELECT "Column filename already exists" as info;');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Check dan add validation_status column jika belum ada
SET @col_exists = (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
                   WHERE TABLE_SCHEMA = 'cakep_ews' 
                   AND TABLE_NAME = 'admin_training_data' 
                   AND COLUMN_NAME = 'validation_status');

SET @sql = IF(@col_exists = 0, 
              'ALTER TABLE admin_training_data ADD COLUMN validation_status ENUM("pending", "validated", "rejected") DEFAULT "pending";',
              'SELECT "Column validation_status already exists" as info;');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Step 3: Update filename dari image_path
UPDATE admin_training_data 
SET filename = SUBSTRING_INDEX(image_path, '/', -1)
WHERE filename = '' OR filename IS NULL;

-- Step 4: Show current table structure dan constraints
SELECT 'Current table structure:' as info;
DESCRIBE admin_training_data;

SELECT 'Current foreign key constraints:' as info;
SELECT 
    CONSTRAINT_NAME,
    COLUMN_NAME,
    REFERENCED_TABLE_NAME,
    REFERENCED_COLUMN_NAME
FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
WHERE TABLE_SCHEMA = 'cakep_ews' 
AND TABLE_NAME = 'admin_training_data' 
AND REFERENCED_TABLE_NAME IS NOT NULL;

-- Step 5: Manual steps (uncomment dan jalankan satu per satu)
-- Ganti 'constraint_name' dengan nama constraint sebenarnya dari query di atas

-- ALTER TABLE admin_training_data DROP FOREIGN KEY constraint_name_for_asset_category;
-- ALTER TABLE admin_training_data DROP FOREIGN KEY constraint_name_for_damage_type;
-- ALTER TABLE admin_training_data DROP COLUMN asset_category_id;
-- ALTER TABLE admin_training_data DROP COLUMN damage_type_id;

SELECT 'Migration script completed. Please run the manual steps based on your constraint names.' as info;
