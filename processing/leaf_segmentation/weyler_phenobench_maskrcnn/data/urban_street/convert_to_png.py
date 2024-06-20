import cv2
import os
import numpy as np
import click
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import shutil

def main():
    os.mkdir("images_jpg", exists_ok=True)
    shutil.move("images/*", "images_jpg")
    os.mkdir("images", exists_ok=True)
    files = [file for file in os.listdir(os.path.join("images_jpg")) if ".jpg" in file]
    print("Converting {} files".format(len(files)))

    with mp.Pool(mp.cpu_count()) as pool:
        p = list(tqdm(pool.imap_unordered(generate, files), total=len(files)))
        
def generate(file):
    img = cv2.imread(os.path.join("images_jpg", file))
    cv2.imwrite(os.path.join("images", file.replace(".jpg", ".png")), img)

if __name__ == '__main__':
    main()