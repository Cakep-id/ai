"""
Insert sample data untuk testing admin interface
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from services import db_service
from datetime import datetime

def insert_sample_data():
    """Insert sample training data dan asset categories"""
    
    try:
        # Insert asset categories jika belum ada
        categories = [
            ('Generator', 'Generator listrik industrial'),
            ('Motor Listrik', 'Motor listrik untuk industri'),
            ('Kompresor', 'Kompresor udara industrial'),
            ('Pompa', 'Pompa air dan fluida'),
            ('Lainnya', 'Aset industrial lainnya')
        ]
        
        for cat_name, cat_desc in categories:
            try:
                query = "INSERT IGNORE INTO asset_categories (category_name, description) VALUES (%s, %s)"
                db_service.execute_query(query, (cat_name, cat_desc))
                print(f"Added category: {cat_name}")
            except Exception as e:
                print(f"Category {cat_name} might already exist: {e}")
        
        # Insert damage types jika belum ada
        damage_types = [
            ('Kerusakan Umum', 'Kerusakan umum pada aset'),
            ('Korosi', 'Karat dan korosi'),
            ('Keausan', 'Keausan material'),
            ('Kebocoran', 'Kebocoran fluida'),
            ('Keretakan', 'Retak pada struktur')
        ]
        
        for dmg_name, dmg_desc in damage_types:
            try:
                query = "INSERT IGNORE INTO damage_types (damage_name, description) VALUES (%s, %s)"
                db_service.execute_query(query, (dmg_name, dmg_desc))
                print(f"Added damage type: {dmg_name}")
            except Exception as e:
                print(f"Damage type {dmg_name} might already exist: {e}")
        
        # Insert admin user jika belum ada
        try:
            query = """
            INSERT IGNORE INTO users (username, email, password_hash, role, full_name) 
            VALUES (%s, %s, %s, %s, %s)
            """
            values = ('admin', 'admin@cakep.id', 'hashed_password', 'admin', 'Administrator')
            db_service.execute_query(query, values)
            print("Added admin user")
        except Exception as e:
            print(f"Admin user might already exist: {e}")
        
        # Insert sample training data
        sample_training_data = [
            (1, 'uploads/training/high/sample_generator_damage.jpg', 1, 1, 'Generator dengan kerusakan tinggi', 'HIGH'),
            (1, 'uploads/training/medium/sample_motor_wear.jpg', 2, 3, 'Motor dengan keausan sedang', 'MEDIUM'),
            (1, 'uploads/training/low/sample_pump_normal.jpg', 4, 1, 'Pompa dalam kondisi normal', 'LOW'),
            (1, 'uploads/training/high/sample_compressor_leak.jpg', 3, 4, 'Kompresor dengan kebocoran', 'HIGH'),
            (1, 'uploads/training/medium/sample_generator_corrosion.jpg', 1, 2, 'Generator dengan korosi sedang', 'MEDIUM'),
        ]
        
        for data in sample_training_data:
            try:
                query = """
                INSERT INTO admin_training_data 
                (uploaded_by_admin, image_path, asset_category_id, damage_type_id, damage_description, risk_level, is_active, uploaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = data + (True, datetime.now())
                result = db_service.execute_query(query, values)
                print(f"Added training data: {data[4]} (ID: {result})")
            except Exception as e:
                print(f"Error adding training data: {e}")
        
        # Insert sample user reports
        sample_reports = [
            ('Kerusakan Generator', 'generator_detected', 'Generator', 'MEDIUM', 0.85, 'uploads/user_reports/sample1.jpg'),
            ('Pompa Bocor', 'pump_damage', 'Pompa', 'HIGH', 0.92, 'uploads/user_reports/sample2.jpg'),
            ('Motor Normal', 'motor_normal', 'Motor Listrik', 'LOW', 0.78, 'uploads/user_reports/sample3.jpg'),
        ]
        
        for report in sample_reports:
            try:
                query = """
                INSERT INTO user_reports 
                (description, ai_detected_damage, asset_name, ai_risk_level, ai_confidence, image_path, reported_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                values = report + (datetime.now(),)
                result = db_service.execute_query(query, values)
                print(f"Added user report: {report[0]} (ID: {result})")
            except Exception as e:
                print(f"Error adding user report: {e}")
        
        print("Sample data insertion completed!")
        
    except Exception as e:
        print(f"Error inserting sample data: {e}")

if __name__ == "__main__":
    insert_sample_data()
