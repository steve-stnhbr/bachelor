import cv2
import os
import numpy as np
import click
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import shutil

def main():
    g = partial(generate)
    shutil.move("images", "images_jpg")
    files = [file for file in os.listdir(os.path.join("images_jpg")) if ".jpg" in file]

    with mp.Pool(mp.cpu_count()) as pool:
        p = list(tqdm(pool.imap_unordered(g, files), total=len(files)))
        
def generate(file):
    img = cv2.imread(os.path.join("images_jpg", file))
    cv2.imwrite(os.path.join("images", file))

if __name__ == '__main__':
    main()