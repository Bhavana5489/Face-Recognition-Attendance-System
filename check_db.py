import os
import sqlite3

def check_db(db_path):
    print(f"\nChecking {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in c.fetchall()]
        print("Tables:", tables)
        if "biometric_templates" in tables:
            c.execute("SELECT COUNT(*) FROM biometric_templates")
            print("Templates count:", c.fetchone()[0])
            c.execute("SELECT student_id, template_status, quality_score FROM biometric_templates")
            print("Templates:", c.fetchall())
        if "students" in tables:
            c.execute("SELECT * FROM students")
            print("Students:", c.fetchall())
        conn.close()
    except Exception as e:
        print("Error:", e)

for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.db'):
            check_db(os.path.join(root, f))
