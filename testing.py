import os

root_dir = "/Users/emiz/Desktop/ct sweep datasets/testing"  # <-- change this to your actual path

for patient_folder in os.listdir(root_dir):
    patient_path = os.path.join(root_dir, patient_folder)

    if not os.path.isdir(patient_path):
        continue

    old_path = os.path.join(patient_path, "mitral-tricuspid-rotation")
    new_path = os.path.join(patient_path, "pitch")

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} → {new_path}")
    else:
        print(f"Skipped: {patient_folder} (no 'apex-base')")