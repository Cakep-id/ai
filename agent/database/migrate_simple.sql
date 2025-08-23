-- Migration sederhana untuk menghapus kategori dari admin_training_data
-- Untuk MySQL versi lama yang tidak support IF EXISTS

USE cakep_ews;

-- Step 1: Add new columns
ALTER TABLE admin_training_data ADD COLUMN filename VARCHAR(255) DEFAULT '';
ALTER TABLE admin_training_data ADD COLUMN validation_status ENUM('pending', 'validated', 'rejected') DEFAULT 'pending';

-- Step 2: Update filename from image_path
UPDATE admin_training_data 
SET filename = SUBSTRING_INDEX(image_path, '/', -1)
WHERE filename = '' OR filename IS NULL;

-- Step 3: Check existing foreign keys
SHOW CREATE TABLE admin_training_data;

-- Step 4: Remove foreign keys manually (run these one by one based on your constraint names)
-- Replace 'actual_constraint_name' with the real constraint names from SHOW CREATE TABLE above

-- For asset_category_id constraint:
-- ALTER TABLE admin_training_data DROP FOREIGN KEY actual_constraint_name_1;

-- For damage_type_id constraint:  
-- ALTER TABLE admin_training_data DROP FOREIGN KEY actual_constraint_name_2;

-- Step 5: Drop the columns
-- ALTER TABLE admin_training_data DROP COLUMN asset_category_id;
-- ALTER TABLE admin_training_data DROP COLUMN damage_type_id;

-- Step 6: Verify final structure
-- DESCRIBE admin_training_data;
