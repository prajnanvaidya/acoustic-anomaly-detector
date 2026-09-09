import os
import io
import re
import csv
import shutil
import zipfile
import requests

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DATA_DIR = os.path.join(BASE_DIR, "data_2021")
NORMAL_DIR = os.path.join(TARGET_DATA_DIR, "normal")
ANOMALOUS_DIR = os.path.join(TARGET_DATA_DIR, "anomalous")

DEV_ZIP = "dev_data_valve.zip"
EVAL_ZIP = "eval_data_valve_train.zip"

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(ANOMALOUS_DIR, exist_ok=True)

def fetch_github_labels():
    """Fetches and merges all valve ground truth labels directly from the DCASE GitHub repository."""
    print("\n🌐 Fetching ground truth labels directly from DCASE 2021 GitHub...")
    base_url = "https://raw.githubusercontent.com/y-kawagu/dcase2021_task2_evaluator/main/ground_truth_data/"
    sections = ['03', '04', '05', '06']
    domains = ['source', 'target']
    label_map = {}

    for sec in sections:
        for dom in domains:
            csv_name = f"ground_truth_valve_section_{sec}_{dom}_test.csv"
            url = base_url + csv_name
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Parse CSV content directly from memory
                    reader = csv.reader(response.text.strip().split('\n'))
                    for row in reader:
                        if not row or "filename" in row[0]: continue
                        fname = row[0].strip()
                        label = int(row[1].strip())
                        label_map[fname] = label
            except Exception as e:
                print(f"   ⚠️ Failed to fetch {csv_name}: {e}")
                
    print(f"✅ Successfully compiled {len(label_map)} ground truth labels from GitHub.")
    return label_map

def extract_and_sort_zip(zip_path, label_map=None):
    """Extracts data and routes it to normal/anomalous folders based on filename or GitHub label map."""
    if not os.path.exists(zip_path):
        print(f"⚠️ Archive {zip_path} not found locally. Skipping extraction.")
        return

    print(f"\n📂 Extracting and sorting content from: {zip_path}")
    matched_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.wav'):
                fname = os.path.basename(file_info.filename)
                
                # 1. Check if the class is explicitly in the filename (Dev Zip & Train Zip behavior)
                if "normal" in file_info.filename.lower():
                    dest = os.path.join(NORMAL_DIR, fname)
                elif "anomaly" in file_info.filename.lower() or "anomalous" in file_info.filename.lower():
                    dest = os.path.join(ANOMALOUS_DIR, fname)
                
                # 2. If unlabeled, check our GitHub dictionary map (Eval Test Zip behavior)
                elif label_map and fname in label_map:
                    is_anomaly = label_map[fname] == 1
                    dest = os.path.join(ANOMALOUS_DIR if is_anomaly else NORMAL_DIR, fname)
                
                else:
                    continue
                
                with zip_ref.open(file_info) as source, open(dest, "wb") as target:
                    shutil.copyfileobj(source, target)
                matched_count += 1
                
    print(f"✅ Successfully processed {matched_count} audio files from {zip_path}.")

if __name__ == "__main__":
    print("⚡ Starting End-to-End DCASE 2021 Data Processing Pipeline ⚡")
    
    # 1. Check for existing files
    for zip_file in [DEV_ZIP, EVAL_ZIP]:
        if os.path.exists(zip_file):
            print(f"📦 {zip_file} located safely on disk.")
        else:
            print(f"❌ Missing {zip_file}. Please ensure files are in the root directory.")
            
    # 2. Fetch Labels from GitHub (Bypassing Zenodo entirely)
    github_labels = fetch_github_labels()
    
    # 3. Extract and Sort both archives using the smart routing function
    extract_and_sort_zip(DEV_ZIP)
    extract_and_sort_zip(EVAL_ZIP, label_map=github_labels)
    
    # 4. Clean up
    print("\n🧹 Cleaning up compressed zip files...")
    for zip_f in [DEV_ZIP, EVAL_ZIP]:
        if os.path.exists(zip_f):
            os.remove(zip_f)
            
    print(f"\n🎉 SUCCESS: Data pipeline complete. Target data structured cleanly at: {TARGET_DATA_DIR}")
    print(f"   📊 Normal folder file count: {len(os.listdir(NORMAL_DIR))}")
    print(f"   📊 Anomalous folder file count: {len(os.listdir(ANOMALOUS_DIR))}")