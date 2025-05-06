# reset_pipeline.py

import os
import shutil

folders_to_clean = ["models", "outputs"]

for folder in folders_to_clean:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"🧹 Deleted: {file_path}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")
    else:
        print(f"📂 Folder '{folder}' does not exist. Skipping.")

print("✅ Project directories cleaned. You can now re-run `main.py`.")
