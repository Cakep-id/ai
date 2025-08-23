-- phpMyAdmin SQL Dump
-- version 5.2.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Waktu pembuatan: 23 Agu 2025 pada 06.10
-- Versi server: 8.0.30
-- Versi PHP: 8.2.27

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `cakep_ews`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `admin_training_data`
--

CREATE TABLE `admin_training_data` (
  `training_id` int NOT NULL,
  `uploaded_by_admin` int NOT NULL,
  `image_path` varchar(500) COLLATE utf8mb4_unicode_ci NOT NULL,
  `damage_description` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `risk_level` enum('CRITICAL','HIGH','MEDIUM','LOW') COLLATE utf8mb4_unicode_ci NOT NULL,
  `annotations` json DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT '1',
  `uploaded_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `filename` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT '',
  `validation_status` enum('pending','validated','rejected') COLLATE utf8mb4_unicode_ci DEFAULT 'pending'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `ai_learning_history`
--

CREATE TABLE `ai_learning_history` (
  `learning_id` int NOT NULL,
  `source_type` enum('user_report','admin_training') COLLATE utf8mb4_unicode_ci NOT NULL,
  `source_id` int NOT NULL,
  `image_path` varchar(500) COLLATE utf8mb4_unicode_ci NOT NULL,
  `risk_level` enum('CRITICAL','HIGH','MEDIUM','LOW') COLLATE utf8mb4_unicode_ci NOT NULL,
  `confidence_score` float DEFAULT NULL,
  `model_version` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `learned_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `is_validated` tinyint(1) DEFAULT '1'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data untuk tabel `ai_learning_history`
--

INSERT INTO `ai_learning_history` (`learning_id`, `source_type`, `source_id`, `image_path`, `risk_level`, `confidence_score`, `model_version`, `learned_at`, `is_validated`) VALUES
(1, 'user_report', 4, 'uploads/user_reports/ebb23b6f-353e-4b9b-9ff9-9f60f06da044.JPG', 'LOW', 0.584, 'yolo_v1.0', '2025-08-22 16:16:19', 1),
(2, 'user_report', 6, 'uploads/user_reports/48a542fa-4c0d-408c-8efc-06287bd61419.JPG', 'HIGH', 0.708, 'yolo_v1.0', '2025-08-22 16:29:07', 1),
(3, 'user_report', 5, 'uploads/user_reports/eb6742b0-86ae-4bed-a790-ef6c5d5dbdc7.JPG', 'LOW', 0.744, 'yolo_v1.0', '2025-08-22 16:29:26', 1);

-- --------------------------------------------------------

--
-- Struktur dari tabel `assets`
--

CREATE TABLE `assets` (
  `asset_id` int NOT NULL,
  `asset_name` varchar(100) COLLATE utf8mb4_unicode_ci NOT NULL,
  `description` text COLLATE utf8mb4_unicode_ci,
  `installation_date` date DEFAULT NULL,
  `last_maintenance` date DEFAULT NULL,
  `next_maintenance` date DEFAULT NULL,
  `criticality` enum('CRITICAL','HIGH','MEDIUM','LOW') COLLATE utf8mb4_unicode_ci DEFAULT 'MEDIUM',
  `status` enum('active','maintenance','inactive') COLLATE utf8mb4_unicode_ci DEFAULT 'active',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data untuk tabel `assets`
--

INSERT INTO `assets` (`asset_id`, `asset_name`, `description`, `installation_date`, `last_maintenance`, `next_maintenance`, `criticality`, `status`, `created_at`, `updated_at`) VALUES
(1, 'Pompa Utama', 'Pompa utama sistem air', NULL, NULL, NULL, 'HIGH', 'active', '2025-08-22 15:46:24', '2025-08-22 15:46:24'),
(2, 'Pipa Distribusi A', 'Pipa distribusi air utama', NULL, NULL, NULL, 'MEDIUM', 'active', '2025-08-22 15:46:24', '2025-08-22 15:46:24'),
(3, 'Tangki Air Utama', 'Tangki penyimpanan air utama', NULL, NULL, NULL, 'CRITICAL', 'active', '2025-08-22 15:46:24', '2025-08-22 15:46:24'),
(4, 'Motor Ventilasi', 'Motor sistem ventilasi gedung', NULL, NULL, NULL, 'MEDIUM', 'active', '2025-08-22 15:46:24', '2025-08-22 15:46:24');

-- --------------------------------------------------------

--
-- Struktur dari tabel `maintenance_schedules`
--

CREATE TABLE `maintenance_schedules` (
  `schedule_id` int NOT NULL,
  `asset_id` int NOT NULL,
  `report_id` int DEFAULT NULL,
  `maintenance_type` enum('preventive','corrective','emergency') COLLATE utf8mb4_unicode_ci NOT NULL,
  `scheduled_date` date NOT NULL,
  `priority` enum('CRITICAL','HIGH','MEDIUM','LOW') COLLATE utf8mb4_unicode_ci NOT NULL,
  `estimated_duration_hours` int DEFAULT '2',
  `assigned_technician` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `procedure_steps` text COLLATE utf8mb4_unicode_ci,
  `status` enum('scheduled','in_progress','completed','cancelled') COLLATE utf8mb4_unicode_ci DEFAULT 'scheduled',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `completed_at` timestamp NULL DEFAULT NULL,
  `completion_notes` text COLLATE utf8mb4_unicode_ci
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `model_metrics`
--

CREATE TABLE `model_metrics` (
  `metric_id` int NOT NULL,
  `model_version` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL,
  `accuracy` float DEFAULT NULL,
  `precision_score` float DEFAULT NULL,
  `recall_score` float DEFAULT NULL,
  `f1_score` float DEFAULT NULL,
  `total_predictions` int DEFAULT '0',
  `correct_predictions` int DEFAULT '0',
  `false_positives` int DEFAULT '0',
  `false_negatives` int DEFAULT '0',
  `calculated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `system_config`
--

CREATE TABLE `system_config` (
  `config_id` int NOT NULL,
  `config_key` varchar(100) COLLATE utf8mb4_unicode_ci NOT NULL,
  `config_value` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `description` text COLLATE utf8mb4_unicode_ci,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `updated_by` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data untuk tabel `system_config`
--

INSERT INTO `system_config` (`config_id`, `config_key`, `config_value`, `description`, `updated_at`, `updated_by`) VALUES
(1, 'ai_confidence_threshold', '0.75', 'Threshold minimum confidence untuk AI prediction', '2025-08-22 15:46:24', NULL),
(2, 'auto_schedule_enabled', 'true', 'Enable automatic maintenance scheduling', '2025-08-22 15:46:24', NULL),
(3, 'training_batch_size', '20', 'Jumlah data minimum untuk retrain model', '2025-08-22 15:46:24', NULL),
(4, 'high_risk_alert_hours', '4', 'Jam maksimal response untuk high risk', '2025-08-22 15:46:24', NULL),
(5, 'critical_risk_alert_hours', '1', 'Jam maksimal response untuk critical risk', '2025-08-22 15:46:24', NULL);

-- --------------------------------------------------------

--
-- Struktur dari tabel `users`
--

CREATE TABLE `users` (
  `user_id` int NOT NULL,
  `username` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL,
  `email` varchar(100) COLLATE utf8mb4_unicode_ci NOT NULL,
  `password_hash` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `role` enum('admin','user') COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'user',
  `full_name` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `is_active` tinyint(1) DEFAULT '1'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data untuk tabel `users`
--

INSERT INTO `users` (`user_id`, `username`, `email`, `password_hash`, `role`, `full_name`, `created_at`, `updated_at`, `is_active`) VALUES
(1, 'admin', 'admin@cakep.id', '$2b$12$example_hash', 'admin', 'Administrator', '2025-08-22 15:46:24', '2025-08-22 15:46:24', 1),
(2, 'user_demo', 'user@cakep.id', '$2b$12$example_hash', 'user', 'Demo User', '2025-08-22 15:46:24', '2025-08-22 15:46:24', 1);

-- --------------------------------------------------------

--
-- Struktur dari tabel `user_reports`
--

CREATE TABLE `user_reports` (
  `report_id` int NOT NULL,
  `reported_by_user_id` int NOT NULL,
  `image_path` varchar(500) COLLATE utf8mb4_unicode_ci NOT NULL,
  `description` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `location_details` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `reported_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `ai_detected_damage` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ai_risk_level` enum('CRITICAL','HIGH','MEDIUM','LOW') COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ai_confidence` float DEFAULT NULL,
  `ai_procedures` text COLLATE utf8mb4_unicode_ci,
  `ai_analyzed_at` timestamp NULL DEFAULT NULL,
  `admin_status` enum('pending','approved','rejected') COLLATE utf8mb4_unicode_ci DEFAULT 'pending',
  `validated_by` int DEFAULT NULL,
  `validated_at` timestamp NULL DEFAULT NULL,
  `admin_corrected_damage` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `admin_corrected_risk` enum('CRITICAL','HIGH','MEDIUM','LOW') COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `admin_corrected_procedures` text COLLATE utf8mb4_unicode_ci,
  `admin_notes` text COLLATE utf8mb4_unicode_ci,
  `is_used_for_training` tinyint(1) DEFAULT '0',
  `training_added_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data untuk tabel `user_reports`
--

INSERT INTO `user_reports` (`report_id`, `reported_by_user_id`, `image_path`, `description`, `location_details`, `reported_at`, `ai_detected_damage`, `ai_risk_level`, `ai_confidence`, `ai_procedures`, `ai_analyzed_at`, `admin_status`, `validated_by`, `validated_at`, `admin_corrected_damage`, `admin_corrected_risk`, `admin_corrected_procedures`, `admin_notes`, `is_used_for_training`, `training_added_at`) VALUES
(1, 1, 'uploads/user_reports/816c6eda-b046-4732-86b1-1b5a40b9b439.JPG', 'terjadi korosi pipa siang tadi saat saya ingin makan siang', 'Lokasi terdeteksi dari analisis gambar', '2025-08-22 15:47:02', 'Equipment', NULL, NULL, NULL, '2025-08-22 15:47:02', 'rejected', 1, '2025-08-22 15:49:32', NULL, NULL, NULL, '', 0, NULL),
(2, 1, 'uploads/user_reports/c545e20e-dd0b-47f9-b019-77daa4637e30.JPG', 'terjadi korosi pipa siang tadi saat saya ingin makan siang', 'Lokasi terdeteksi dari analisis gambar', '2025-08-22 15:47:03', 'Equipment', NULL, NULL, NULL, '2025-08-22 15:47:03', 'rejected', 1, '2025-08-22 15:55:25', NULL, NULL, NULL, '', 0, NULL),
(3, 1, 'uploads/user_reports/98ef7311-820c-424c-acb9-83a207cf473f.JPG', 'terjadi issue atau kerusakan pada siang hari tadi sebelum saya makan', 'Lokasi terdeteksi dari analisis gambar', '2025-08-22 15:47:58', 'Equipment', NULL, NULL, NULL, '2025-08-22 15:47:58', 'rejected', 1, '2025-08-22 15:55:33', NULL, NULL, NULL, '', 0, NULL),
(4, 1, 'uploads/user_reports/ebb23b6f-353e-4b9b-9ff9-9f60f06da044.JPG', 'terjadi kerusakan tadi pada saat saya ingin makan', 'Lokasi terdeteksi dari analisis gambar', '2025-08-22 15:54:13', 'Unknown', 'LOW', 0.584, '[{\"step\": 1, \"description\": \"1. Matikan sistem dan pastikan keamanan area kerja (15 menit)\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 2, \"description\": \"2. Siapkan tools: kunci pas, seal baru, pembersih (5 menit)\", \"estimated_time_minutes\": 5, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 3, \"description\": \"3. Pembersih area kerja dari debu dan material yang tidak diperlukan (10 menit) **Langkah 2: Lepas Komponen yang Rusak**\", \"estimated_time_minutes\": 10, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 4, \"description\": \"1. Lepas komponen yang rusak dengan hati-hati dan pastikan tidak ada kabel atau pipa yang terikat (30 menit)\", \"estimated_time_minutes\": 30, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 5, \"description\": \"2. Identifikasi komponen yang rusak dan catat informasi spesifikasi dan kode part (15 menit)\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 6, \"description\": \"3. Siapkan komponen ganti yang sesuai dengan spesifikasi dan kode part (15 menit) **Langkah 3: Perbaikan dan Penggantian Komponen**\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 7, \"description\": \"1. Perbaiki komponen yang rusak dengan menggunakan tools yang sesuai (30 menit)\", \"estimated_time_minutes\": 30, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 8, \"description\": \"2. Pasang komponen ganti yang sesuai dengan spesifikasi dan kode part (20 menit)\", \"estimated_time_minutes\": 20, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 9, \"description\": \"3. Pastikan komponen yang baru telah terpasang dengan benar dan tidak ada kebocoran (15 menit) **Langkah 4: Uji Coba dan Pengujian**\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 10, \"description\": \"1. Lakukan uji coba sistem untuk memastikan bahwa komponen yang baru telah berfungsi dengan baik (30 menit)\", \"estimated_time_minutes\": 30, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 11, \"description\": \"2. Periksa sistem untuk memastikan tidak ada kebocoran atau kerusakan (15 menit)\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 12, \"description\": \"3. Catat informasi hasil uji coba dan pengujian (10 menit) **Waktu Estimasi: 2 jam** **Material/Tools Needed:** * Kunci pas * Seal baru * Pembersih * Kompresor udara (optional) * Kabel atau pipa yang sesuai dengan spesifikasi **Safety Notes:** * Pastikan area kerja telah dijamin keamanan dan tidak ada material berbahaya yang tercebur * Gunakan tools dengan hati-hati dan sesuai dengan spesifikasi * Pastikan komponen yang baru telah terpasang dengan benar dan tidak ada kebocoran * Lakukan uji coba sistem dengan hati-hati dan sesuai dengan spesifikasi **Catatan:** * Waktu estimasi dapat berbeda-beda tergantung pada kondisi kerusakan dan teknisi yang melakukan perbaikan * Material dan tools yang diperlukan dapat berbeda-beda tergantung pada spesifikasi dan kode part yang digunakan * Safety notes adalah kewajiban teknisi untuk memastikan keamanan area kerja dan teknisi sendiri\", \"estimated_time_minutes\": 10, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}]', '2025-08-22 15:54:15', 'approved', 1, '2025-08-22 16:16:19', NULL, 'LOW', NULL, '', 1, '2025-08-22 16:16:19'),
(5, 1, 'uploads/user_reports/eb6742b0-86ae-4bed-a790-ef6c5d5dbdc7.JPG', 'terjadi kerusakan pipa pada bagian badan', 'Lokasi terdeteksi dari analisis gambar', '2025-08-22 16:17:44', 'Unknown', 'MEDIUM', 0.744, '[{\"step\": 1, \"description\": \"1. Matikan sistem pipa dan pastikan keamanan area kerja (15 menit) * Estimasi waktu: 15 menit * Material/tools needed: tidak ada * Safety notes: Pastikan area kerja bebas dari gangguan dan kondisi yang berbahaya. **Langkah 2: Siapkan Tools dan Material**\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 2, \"description\": \"1. Siapkan tools: kunci pas, seal baru, pembersih, dan pelindung tangan (5 menit) * Estimasi waktu: 5 menit * Material/tools needed: kunci pas, seal baru, pembersih, pelindung tangan * Safety notes: Pastikan tools dan material yang digunakan aman dan sesuai untuk penggunaan. **Langkah 3: Lepas Komponen yang Rusak**\", \"estimated_time_minutes\": 5, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 3, \"description\": \"1. Lepas komponen yang rusak dengan hati-hati menggunakan kunci pas (30 menit) * Estimasi waktu: 30 menit * Material/tools needed: kunci pas * Safety notes: Pastikan area kerja bebas dari gangguan dan kondisi yang berbahaya. Hindari mengikat atau menghentakkan komponen yang rusak. **Langkah 4: Membersihkan Area Kerja**\", \"estimated_time_minutes\": 30, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 4, \"description\": \"1. Membersihkan area kerja menggunakan pembersih dan pelindung tangan (15 menit) * Estimasi waktu: 15 menit * Material/tools needed: pembersih, pelindung tangan * Safety notes: Pastikan area kerja bebas dari material yang berbahaya dan kondisi yang berbahaya. **Langkah 5: Instalasi Seal Baru**\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 5, \"description\": \"1. Instalasi seal baru menggunakan kunci pas dan pelindung tangan (30 menit) * Estimasi waktu: 30 menit * Material/tools needed: seal baru, kunci pas, pelindung tangan * Safety notes: Pastikan area kerja bebas dari gangguan dan kondisi yang berbahaya. Hindari mengikat atau menghentakkan seal baru. **Langkah 6: Uji Coba Sistem**\", \"estimated_time_minutes\": 30, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 6, \"description\": \"1. Uji coba sistem pipa untuk memastikan bahwa kerusakan telah diperbaiki (30 menit) * Estimasi waktu: 30 menit * Material/tools needed: tidak ada * Safety notes: Pastikan area kerja bebas dari gangguan dan kondisi yang berbahaya. **Total Estimasi Waktu:** 1 jam 45 menit **Total Material/Tools Needed:** * Kunci pas * Seal baru * Pembersih * Pelindung tangan **Safety Notes:** * Pastikan area kerja bebas dari gangguan dan kondisi yang berbahaya. * Hindari mengikat atau menghentakkan komponen yang rusak. * Pastikan tools dan material yang digunakan aman dan sesuai untuk penggunaan.\", \"estimated_time_minutes\": 30, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}]', '2025-08-22 16:17:46', 'approved', 1, '2025-08-22 16:29:26', NULL, 'LOW', NULL, '', 1, '2025-08-22 16:29:26'),
(6, 1, 'uploads/user_reports/48a542fa-4c0d-408c-8efc-06287bd61419.JPG', 'terjadi karat pada pipa pada saat saya mau berangkat kerja tadi', 'Lokasi terdeteksi dari analisis gambar', '2025-08-22 16:27:18', 'Unknown', 'MEDIUM', 0.708, '[{\"step\": 1, \"description\": \"1. Matikan sistem pipa dan pastikan keamanan area kerja (15 menit)\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 2, \"description\": \"2. Siapkan tools: kunci pas, seal baru, pembersih, masker, dan sarung tangan (5 menit)\", \"estimated_time_minutes\": 5, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 3, \"description\": \"3. Lengkapi sarung tangan dengan karet untuk mencegah terjadinya gesekan dan mengurangi risiko kemedangan **Langkah 2: Pembersihan Area Kerja**\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 4, \"description\": \"1. Bersihkan area kerja dari debu dan kotoran (10 menit)\", \"estimated_time_minutes\": 10, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 5, \"description\": \"2. Gunakan pembersih untuk menghilangkan sisa-sisa karat dan kotoran (10 menit)\", \"estimated_time_minutes\": 10, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 6, \"description\": \"3. Pastikan area kerja kering dan tidak berbahaya untuk pekerjaan perbaikan **Langkah 3: Penghapusan Karat**\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 7, \"description\": \"1. Gunakan kunci pas untuk menghapus karat yang telah terbentuk (30 menit)\", \"estimated_time_minutes\": 30, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 8, \"description\": \"2. Hati-hati dalam menghapus karat agar tidak menyebabkan kerusakan pada pipa\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 9, \"description\": \"3. Pastikan karat telah dihapus sepenuhnya sebelum melanjutkan ke langkah berikutnya **Langkah 4: Penggantian Seal**\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 10, \"description\": \"1. Siapkan seal baru yang sesuai dengan ukuran pipa (5 menit)\", \"estimated_time_minutes\": 5, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 11, \"description\": \"2. Gunakan kunci pas untuk menggantikan seal yang rusak (20 menit)\", \"estimated_time_minutes\": 20, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 12, \"description\": \"3. Pastikan seal terpasang dengan sempurna dan tidak ada kebocoran **Langkah 5: Pengujian Sistem**\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 13, \"description\": \"1. Jalankan sistem pipa untuk memastikan tidak ada kebocoran atau kerusakan (15 menit)\", \"estimated_time_minutes\": 15, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}, {\"step\": 14, \"description\": \"2. Pastikan sistem berfungsi normal dan tidak ada tanda-tanda kerusakan **Waktu yang Diperkirakan: 1,5 jam (90 menit)** **Material/Tools Needed:** * Kunci pas * Seal baru * Pembersih * Masker * Sarung tangan * Karet untuk sarung tangan **Safety Notes:** * Pastikan area kerja kering dan tidak berbahaya * Gunakan masker dan sarung tangan untuk mencegah terjadinya kemedangan dan mengurangi risiko kesehatan * Hati-hati dalam menghapus karat agar tidak menyebabkan kerusakan pada pipa * Pastikan seal terpasang dengan sempurna dan tidak ada kebocoran\", \"estimated_time_minutes\": 90, \"safety_level\": \"MEDIUM\", \"required_tools\": [], \"materials\": [], \"safety_notes\": \"\"}]', '2025-08-22 16:27:20', 'approved', 1, '2025-08-22 16:29:07', NULL, 'HIGH', NULL, '', 1, '2025-08-22 16:29:07');

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `admin_training_data`
--
ALTER TABLE `admin_training_data`
  ADD PRIMARY KEY (`training_id`),
  ADD KEY `uploaded_by_admin` (`uploaded_by_admin`),
  ADD KEY `idx_training_data_active` (`is_active`);

--
-- Indeks untuk tabel `ai_learning_history`
--
ALTER TABLE `ai_learning_history`
  ADD PRIMARY KEY (`learning_id`),
  ADD KEY `idx_ai_learning_source` (`source_type`,`source_id`);

--
-- Indeks untuk tabel `assets`
--
ALTER TABLE `assets`
  ADD PRIMARY KEY (`asset_id`);

--
-- Indeks untuk tabel `maintenance_schedules`
--
ALTER TABLE `maintenance_schedules`
  ADD PRIMARY KEY (`schedule_id`),
  ADD KEY `asset_id` (`asset_id`),
  ADD KEY `report_id` (`report_id`),
  ADD KEY `idx_maintenance_schedules_date` (`scheduled_date`),
  ADD KEY `idx_maintenance_schedules_status` (`status`);

--
-- Indeks untuk tabel `model_metrics`
--
ALTER TABLE `model_metrics`
  ADD PRIMARY KEY (`metric_id`);

--
-- Indeks untuk tabel `system_config`
--
ALTER TABLE `system_config`
  ADD PRIMARY KEY (`config_id`),
  ADD UNIQUE KEY `config_key` (`config_key`),
  ADD KEY `updated_by` (`updated_by`);

--
-- Indeks untuk tabel `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`user_id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indeks untuk tabel `user_reports`
--
ALTER TABLE `user_reports`
  ADD PRIMARY KEY (`report_id`),
  ADD KEY `reported_by_user_id` (`reported_by_user_id`),
  ADD KEY `validated_by` (`validated_by`),
  ADD KEY `idx_user_reports_status` (`admin_status`),
  ADD KEY `idx_user_reports_ai_risk` (`ai_risk_level`),
  ADD KEY `idx_user_reports_date` (`reported_at`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `admin_training_data`
--
ALTER TABLE `admin_training_data`
  MODIFY `training_id` int NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT untuk tabel `ai_learning_history`
--
ALTER TABLE `ai_learning_history`
  MODIFY `learning_id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT untuk tabel `assets`
--
ALTER TABLE `assets`
  MODIFY `asset_id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT untuk tabel `maintenance_schedules`
--
ALTER TABLE `maintenance_schedules`
  MODIFY `schedule_id` int NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT untuk tabel `model_metrics`
--
ALTER TABLE `model_metrics`
  MODIFY `metric_id` int NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT untuk tabel `system_config`
--
ALTER TABLE `system_config`
  MODIFY `config_id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT untuk tabel `users`
--
ALTER TABLE `users`
  MODIFY `user_id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT untuk tabel `user_reports`
--
ALTER TABLE `user_reports`
  MODIFY `report_id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- Ketidakleluasaan untuk tabel pelimpahan (Dumped Tables)
--

--
-- Ketidakleluasaan untuk tabel `admin_training_data`
--
ALTER TABLE `admin_training_data`
  ADD CONSTRAINT `admin_training_data_ibfk_1` FOREIGN KEY (`uploaded_by_admin`) REFERENCES `users` (`user_id`);

--
-- Ketidakleluasaan untuk tabel `maintenance_schedules`
--
ALTER TABLE `maintenance_schedules`
  ADD CONSTRAINT `maintenance_schedules_ibfk_1` FOREIGN KEY (`asset_id`) REFERENCES `assets` (`asset_id`),
  ADD CONSTRAINT `maintenance_schedules_ibfk_2` FOREIGN KEY (`report_id`) REFERENCES `user_reports` (`report_id`);

--
-- Ketidakleluasaan untuk tabel `system_config`
--
ALTER TABLE `system_config`
  ADD CONSTRAINT `system_config_ibfk_1` FOREIGN KEY (`updated_by`) REFERENCES `users` (`user_id`);

--
-- Ketidakleluasaan untuk tabel `user_reports`
--
ALTER TABLE `user_reports`
  ADD CONSTRAINT `user_reports_ibfk_2` FOREIGN KEY (`reported_by_user_id`) REFERENCES `users` (`user_id`),
  ADD CONSTRAINT `user_reports_ibfk_3` FOREIGN KEY (`validated_by`) REFERENCES `users` (`user_id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
