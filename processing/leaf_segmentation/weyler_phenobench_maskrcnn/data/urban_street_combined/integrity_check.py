import os
import shutil

# Define the folder names
folders = ["images", "leaf_instances", "plant_instances", "semantics"]

def main():
    for folder in folders:
        os.makedirs(folder + "_disc", exist_ok=True)

    # Get a list of all files in each folder
    folder_files = {folder: set(os.listdir(folder)) for folder in folders}

    # Find the intersection of all files (files that exist in all folders)
    common_files = set.intersection(*folder_files.values())

    # Find and move files that are not in the intersection to the discarded folder
    for folder in folders:
        for file in folder_files[folder] - common_files:
            src_path = os.path.join(folder, file)
            dst_path = os.path.join(folder + "_disc", file)
            shutil.move(src_path, dst_path)
            print(f"Moved {file} from {folder} to {folder}_disc")

    print("Integrity check complete. Discrepant files moved to the discarded folder.")

if __name__ == '__main__':
    main()
