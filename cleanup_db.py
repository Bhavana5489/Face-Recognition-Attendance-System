import sqlite3
conn = sqlite3.connect('gazepass/gazepass.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("DELETE FROM students WHERE name LIKE '%<img%'")
deleted = cur.rowcount
print('Deleted %d corrupted student entries.' % deleted)

cur.execute("DELETE FROM biometric_templates WHERE student_id NOT IN (SELECT id FROM students)")
cur.execute("DELETE FROM attendance_records WHERE student_id NOT IN (SELECT id FROM students)")

conn.commit()

cur.execute('SELECT id, name FROM students')
print('Remaining students:')
for r in cur.fetchall():
    print('  ' + r['name'])
conn.close()
print('Done.')
