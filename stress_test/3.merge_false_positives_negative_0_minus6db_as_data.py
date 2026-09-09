import os
import shutil

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- DESTINATIONS (Your Original Training Folders) ---
MAIN_NORMAL = os.path.join(BASE_DIR, "data_stress_merged", "normal")
MAIN_ANOMALOUS = os.path.join(BASE_DIR, "data_stress_merged", "anomalous")

# --- SOURCES (The mistakes your model made) ---
zero_db_false_alarms = os.path.join(BASE_DIR, "false_negatives_0db")
minus_6db_false_alarms = os.path.join(BASE_DIR, "false_alarms_minus6db")
minus_6db_missed = os.path.join(BASE_DIR, "missed_anomalies_minus6db")

def copy_files(source_dir, dest_dir, description):
    if not os.path.exists(source_dir):
        print(f"⚠️ Skipping {description} - folder not found.")
        return 0
        
    files = os.listdir(source_dir)
    count = 0
    for f in files:
        if f.endswith('.wav'):
            src_path = os.path.join(source_dir, f)
            dest_path = os.path.join(dest_dir, f)
            # copy2 preserves the original file metadata
            shutil.copy2(src_path, dest_path)
            count += 1
    print(f"✅ Copied {count} files from {description} -> {os.path.basename(dest_dir)}")
    return count

print("\n🚀 MERGING TRICK FILES INTO MAIN DATASET...")

# 1. Move all False Alarms (False Positives) into the NORMAL folder
copy_files(zero_db_false_alarms, MAIN_NORMAL, "0dB False Alarms")
copy_files(minus_6db_false_alarms, MAIN_NORMAL, "-6dB False Alarms")

# 2. Move all Missed Anomalies (False Negatives) into the ANOMALOUS folder
copy_files(minus_6db_missed, MAIN_ANOMALOUS, "-6dB Missed Anomalies")

print("\n🎉 Merge complete! Your dataset is locked and loaded.")