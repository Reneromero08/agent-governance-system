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
combined_todos = "\n\n## Refactored Backlog (Extracted from Legacy TODOs)\n"

for path, content in sorted(todos):
    phase = re.search(r'PHASE\d', path)
    phase_name = phase.group(0) if phase else path
    combined_todos += f"\n### Pending from {phase_name}\n"
    
    lines = content.split('\n')
    for line in lines:
        if '[ ]' in line:
            combined_todos += f"{line.strip()}\n"

with open('CAT_CHAT_ROADMAP.md', 'a', encoding='utf-8') as f:
    f.write(combined_todos)

conn.close()
print("Success: Appended 76 items to CAT_CHAT_ROADMAP.md")
