import sqlite3
import re

conn = sqlite3.connect('cat_chat_index.db')
c = conn.cursor()

c.execute('''
    SELECT f.rel_path, co.content
    FROM files f
    JOIN content co ON f.id = co.file_id
    WHERE f.rel_path LIKE '%TODO%'
''')

todos = c.fetchall()
total_open = sum(len(re.findall(r'\[ \]', content)) for _, content in todos)

print(f'Total: {total_open} open items\n')
for path, content in todos:
    open_count = len(re.findall(r'\[ \]', content))
    print(f'{path}: {open_count} open')

conn.close()
