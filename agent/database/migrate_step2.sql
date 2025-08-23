-- Migration script yang disederhanakan
-- Karena kolom filename sudah ada, langsung ke penghapusan kategori

USE cakep_ews;

-- Step 1: Update filename jika masih kosong
UPDATE admin_training_data 
SET filename = SUBSTRING_INDEX(image_path, '/', -1)
WHERE filename = '' OR filename IS NULL;

-- Step 2: Update validation_status jika masih NULL
UPDATE admin_training_data 
SET validation_status = 'pending'
WHERE validation_status IS NULL;

-- Step 3: Show current table structure
SELECT 'Current table structure:' as info;
DESCRIBE admin_training_data;

-- Step 4: Show foreign key constraints yang perlu dihapus
SELECT 'Foreign key constraints to remove:' as info;
SELECT 
    CONSTRAINT_NAME,
    COLUMN_NAME,
    REFERENCED_TABLE_NAME
FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
WHERE TABLE_SCHEMA = 'cakep_ews' 
AND TABLE_NAME = 'admin_training_data' 
AND REFERENCED_TABLE_NAME IS NOT NULL;

-- Step 5: Manual commands untuk copy-paste (ganti nama constraint sesuai output di atas)
SELECT 'Run these commands one by one (replace constraint names):' as info;
SELECT 'ALTER TABLE admin_training_data DROP FOREIGN KEY your_constraint_name_1;' as command;
SELECT 'ALTER TABLE admin_training_data DROP FOREIGN KEY your_constraint_name_2;' as command;
SELECT 'ALTER TABLE admin_training_data DROP COLUMN asset_category_id;' as command;
SELECT 'ALTER TABLE admin_training_data DROP COLUMN damage_type_id;' as command;
