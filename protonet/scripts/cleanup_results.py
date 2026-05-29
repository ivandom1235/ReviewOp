import shutil
import os
from pathlib import Path

def main():
    output_dir = Path("protonet/output")
    keep = "study_v1"
    
    if not output_dir.exists():
        print("Output directory not found.")
        return

    for item in output_dir.iterdir():
        if item.name == keep:
            continue
        
        try:
            if item.is_dir():
                shutil.rmtree(item)
                print(f"Deleted directory: {item.name}")
            else:
                item.unlink()
                print(f"Deleted file: {item.name}")
        except Exception as e:
            print(f"Error deleting {item.name}: {e}")

if __name__ == "__main__":
    main()
