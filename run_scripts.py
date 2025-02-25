import os
import time
import subprocess

# Define the folder path
folder_path = "./python_ollama_code"

# Get all Python files in the folder
python_files = [f for f in os.listdir(folder_path) if f.endswith(".py")]

# Sort files (optional, to ensure order)
python_files.sort()

# Execute each Python file with a 5-second delay
for file in python_files:
    file_path = os.path.join(folder_path, file)
    print(f"Running {file}...")
    subprocess.run(["python", file_path])
    print(f"Finished {file}, waiting 5 seconds...")
    time.sleep(5)

print("All scripts executed.")
