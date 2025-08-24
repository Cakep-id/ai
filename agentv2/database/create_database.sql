-- AgentV2 Database Setup Script
-- Run this first to create the database and user

-- Create database
CREATE DATABASE IF NOT EXISTS cakep_ews_v2 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- Use the database
USE cakep_ews_v2;

-- Create user (optional, for security)
-- CREATE USER IF NOT EXISTS 'cakep_ews_user'@'localhost' IDENTIFIED BY 'cakep_ews_password';
-- GRANT ALL PRIVILEGES ON cakep_ews_v2.* TO 'cakep_ews_user'@'localhost';
-- FLUSH PRIVILEGES;

-- Show confirmation
SELECT 'cakep_ews_v2 database created successfully!' as message;
