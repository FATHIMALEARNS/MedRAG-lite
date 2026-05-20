import zipfile
import os

source_dir = r"c:\Projects\MedRAG-lite"
zip_path = r"c:\Projects\MedRAG-lite.zip"

excludes = ["venv", ".git", "__pycache__", "medrag.db", "uploads", "MedRAG-lite-temp", "MedRAG-lite.zip"]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(source_dir):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in excludes]
        for file in files:
            if file not in excludes and not file.endswith('.zip'):
                file_path = os.path.join(root, file)
                # Keep folder structure starting from MedRAG-lite
                arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                zipf.write(file_path, arcname)
print("Zipping completed successfully.")
