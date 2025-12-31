import sqlite3, os

candidates = [
    os.path.join("CORTEX", "_generated", "system3.db"),
    os.path.join("CORTEX", "_generated", "system1.db"),
    "system3.db",
    "system1.db",
]

db = next((p for p in candidates if os.path.exists(p)), None)
print("DB:", db)

con = sqlite3.connect(db)
con.execute("PRAGMA foreign_keys=ON;")

triggers = con.execute(
    "SELECT name, tbl_name FROM sqlite_master WHERE type='trigger' ORDER BY name"
).fetchall()

print("Triggers:")
for t in triggers:
    print(" ", t)
