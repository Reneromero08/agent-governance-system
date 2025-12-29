import json
import hashlib
import os

def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def update_caps():
    caps_path = "LAW/CANON/CAPABILITIES.json"
    pins_path = "LAW/CANON/CAPABILITY_PINS.json"
    
    with open(caps_path, "r") as f:
        data = json.load(f)
    
    # Hash to find
    target_hash = "4f81ae57f3d1c61488c71a9042b041776dd463e6334568333321d15b6b7d78fc"
    
    if target_hash not in data["capabilities"]:
        print(f"Hash {target_hash} not found. Searching by name 'ant-worker-copy-v1'")
        found = False
        for h, c in data["capabilities"].items():
            if c["adapter"]["name"] == "ant-worker-copy-v1":
                print(f"Found capability by name with hash: {h}")
                target_hash = h
                found = True
                break
        if not found:
             print("Capability not found!")
             return

    adapter = data["capabilities"][target_hash]["adapter"]
    
    cmd = adapter["command"]
    new_cmd = []
    changed = False
    for part in cmd:
        if "CAPABILITY/SKILLS/agents/ant-worker" in part:
             new_cmd.append(part)
        elif "CAPABILITY/SKILLS/ant-worker" in part:
            new_cmd.append(part.replace("CAPABILITY/SKILLS/ant-worker", "CAPABILITY/SKILLS/agents/ant-worker"))
            changed = True
        else:
            new_cmd.append(part)
    
    if changed:
        print("Command path updated.")
        adapter["command"] = new_cmd
    else:
        print("Command path already correct.")

    # Recompute hash
    new_hash = hashlib.sha256(canonical_json_bytes(adapter)).hexdigest()
    
    del data["capabilities"][target_hash]
    data["capabilities"][new_hash] = {
        "adapter": adapter,
        "adapter_spec_hash": new_hash
    }
    
    with open(caps_path, "wb") as f:
        f.write(canonical_json_bytes(data))
        
    print(f"Updated CAPABILITIES.json. Old hash: {target_hash}, New hash: {new_hash}")
    
    # Update Pins
    if os.path.exists(pins_path):
        with open(pins_path, "r") as f:
            pins_data = json.load(f)
        
        # Remove old hash if different
        if target_hash in pins_data["allowed_capabilities"] and target_hash != new_hash:
             pins_data["allowed_capabilities"].remove(target_hash)

        if new_hash not in pins_data["allowed_capabilities"]:
            pins_data["allowed_capabilities"].append(new_hash)
        
        # SORT PINS!
        pins_data["allowed_capabilities"].sort()
            
        with open(pins_path, "wb") as f:
            f.write(canonical_json_bytes(pins_data))
            print(f"Updated CAPABILITY_PINS.json with new hash (sorted).")

if __name__ == "__main__":
    update_caps()
