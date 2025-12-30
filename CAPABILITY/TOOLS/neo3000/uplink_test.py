import requests
import json
import sys

URL = "http://localhost:8000/api/bridge/uplink"

def send_command(action, text):
    payload = {
        "action": action,
        "target": "chatgpt.com",
        "text": text
    }
    try:
        response = requests.post(URL, json=payload)
        if response.status_code == 200:
            print(f">>> COMMAND DISPATCHED: {action} -> {text}")
        else:
            print(f">>> FAILED TO DISPATCH: {response.text}")
    except Exception as e:
        print(f">>> ERROR: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python uplink_test.py 'Your question here'")
        sys.exit(1)
    
    text = " ".join(sys.argv[1:])
    send_command("query", text)
