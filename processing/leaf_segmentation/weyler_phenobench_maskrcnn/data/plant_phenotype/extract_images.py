import os
import shutil

INPUT_DIR = "phenotyping_data"

def main():
    for path, subdirs, files in os.walk(os.path.join(os.getcwd(), INPUT_DIR)):
        print("Traversing", path)
        for name in files:
            if "label" in name:
                rgb_name = name.replace("label", "rgb")
                label_file = os.path.join(path, name)
                rgb_file = os.path.join(path, rgb_name)

                shutil.copy(label_file, os.path.join("labels", name))
                shutil.copy(rgb_file, os.path.join("images", rgb_name))
                

if __name__ == "__main__":
    main()