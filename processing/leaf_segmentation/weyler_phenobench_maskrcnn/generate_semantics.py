import cv2
import os
import numpy as np
import click
from tqdm import tqdm
from functools import partial
import multiprocessing as mp

@click.command()
@click.option("-i", 
              "--input_path",
              )
@click.option("--jpeg", "-j", default=False, is_flag=True)
def main(input_path, jpeg):
    g = partial(generate, input_path, jpeg)
    files = [file for file in os.listdir(os.path.join(input_path, "images")) if ".jpg" in file or ".png" in file]
    print("Generating instances for {} files".format(len(files)))

    os.makedirs(os.path.join(input_path, "semantics"), exist_ok=True)

    with mp.Pool(mp.cpu_count()) as pool:
        p = list(tqdm(pool.imap_unordered(g, files), total=len(files)))
        
def generate(input_path, jpeg, file):
    if jpeg:
        file = file.replace(".jpg", ".png")
    img = cv2.imread(input_path + "/masks/" + file)
    cv2.imwrite(input_path + "/semantics/" + file, np.where(img != 0, np.ones(img.shape) * 3, np.zeros(img.shape)))

if __name__ == '__main__':
    main()