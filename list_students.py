import sqlite3

conn = sqlite3.connect('gazepass/gazepass.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute('SELECT id, name FROM students')
rows = cur.fetchall()
print('All students:')
for r in rows:
    print('  id=%s  name=%s' % (r['id'], repr(r['name'])))
conn.close()
cur = conn.cursor()
cur.execute('SELECT id, name FROM students')
rows = cur.fetchall()
print('All students:')
for r in rows:
    print('  id=%s  name=%s' % (r['id'], repr(r['name'])))
conn.close()
