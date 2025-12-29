
import subprocess
import sys

def main():
    try:
        # Run cleanup script
        subprocess.run([sys.executable, "cleanup_lines.py"], check=True)
        print("Cleanup complete.")
    except Exception as e:
        print(f"Cleanup failed: {e}")

if __name__ == "__main__":
    main()
