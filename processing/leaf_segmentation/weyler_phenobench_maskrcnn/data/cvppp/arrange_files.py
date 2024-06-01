import os
import shutil

INPUT_DIR = 'training'
OUTPUT_DIR = '.'

def main():
    for a in os.listdir(INPUT_DIR):
        for file in os.listdir(os.path.join(INPUT_DIR, a)):
            if 'fg' in file or 'rgb' in file:
                shutil.copy(os.path.join(INPUT_DIR, a, file), os.path.join(OUTPUT_DIR, 'masks' if 'fg' in file else 'images', file.replace(".png", "") + a + ".png"))


if __name__ == '__main__':
    main()