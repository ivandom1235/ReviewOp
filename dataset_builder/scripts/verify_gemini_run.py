import os
import subprocess
from pathlib import Path

# Load environment variables for the subprocess
env = os.environ.copy()
if "GOOGLE_CLOUD_LOCATION" not in env:
    env["GOOGLE_CLOUD_LOCATION"] = "global"
env["GEMINI_MODEL"] = "gemini-3.1-flash-lite-preview"
env["REVIEWOP_DEFAULT_LLM_PROVIDER"] = "gemini"

input_file = "dataset_builder/input/Laptop_train.csv"
output_dir = "dataset_builder/output_gemini_verify"

cmd = [
    "python",
    "dataset_builder/scripts/build_benchmark.py",
    input_file,
    "--output-dir", output_dir,
    "--llm", "gemini",
    "--sample-size", "10",
    "--max-workers", "2",
    "--overwrite"
]

print(f"Running command: {' '.join(cmd)}")
result = subprocess.run(cmd, env=env)

if result.returncode == 0:
    print("\nVerification SUCCESSFUL!")
    print(f"Check output in {output_dir}")
else:
    print("\nVerification FAILED!")
    exit(result.returncode)
