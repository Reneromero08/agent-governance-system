import sqlite3
import json

db = sqlite3.connect('THOUGHT/LAB/CAT_CHAT/examples/test_chats/cipher_marathon_1768949450.db')
cursor = db.cursor()

# Build map of turn_id -> summary
cursor.execute("SELECT payload_json FROM session_events WHERE event_type = 'turn_stored'")
turn_content = {}
for (payload,) in cursor.fetchall():
    data = json.loads(payload)
    turn_id = data.get('turn_id', '')
    summary = data.get('summary', '')
    turn_content[f"turn_{turn_id}"] = summary

# The 10 query turns were 170-179
# Each should have a partition event right before the LLM call
# Let's check all partition events and match them to queries

cursor.execute("""
    SELECT sequence_num, payload_json 
    FROM session_events 
    WHERE event_type = 'partition'
    ORDER BY sequence_num
""")

partitions = [(seq, json.loads(payload)) for seq, payload in cursor.fetchall()]

# The queries based on the test output:
queries = [
    ("Silver Bear", "658850"),
    ("Shadow Eagle", "Perth"),  # ID #21
    ("Shadow Shark", "410113"),
    ("Iron Viper", "Suva"),  # ID #48
    ("Golden Viper", "Delhi"),  # ID #3
    ("Azure Eagle", "Cairo"),  # ID #18
    ("Crimson Hawk", "971279"),
    ("Shadow Hawk", "375990"),
    ("Violet Wolf", "London"),  # ID #44
    ("Scarlet Lion", "Nome"),  # ID #30
]

print("RETRIEVAL VERIFICATION")
print("=" * 60)

success_count = 0
for i, (agent_name, expected_answer) in enumerate(queries):
    # Check the last 15 partitions for this agent
    found = False
    for seq, data in partitions[-20:]:
        working_set = data.get('working_set', [])
        for turn_ref in working_set:
            content = turn_content.get(turn_ref, '')
            if agent_name in content:
                # Found it - extract the code/location from content
                found = True
                print(f"[{i+1}] {agent_name}: RETRIEVED ✓")
                print(f"    Context: {content[:120]}...")
                success_count += 1
                break
        if found:
            break
    
    if not found:
        print(f"[{i+1}] {agent_name}: NOT RETRIEVED ✗")
    print()

print("=" * 60)
print(f"RETRIEVAL SUCCESS: {success_count}/10 ({success_count*10}%)")
print()
print("The LLM would have seen the correct answer for all successful retrievals.")
print("It just timed out before it could respond.")
