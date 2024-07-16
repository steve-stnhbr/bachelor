import os
import click
import shutil
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

def check_file(file):
    image = os.path.join(input_dir, "images", file)
    instances = os.path.join(input_dir, "leaf_instances", file)

    disc_image = os.path.join(input_dir, "disc_images", file)
    disc_instances = os.path.join(input_dir, "disc_leaf_instances", file)
    im = Image.open(instances).convert("L")
    if 255 in np.unique(im):
        # remove images
        shutil.move(image, disc_image)
        shutil.move(instanes, disc_instances)

@click.command("-i", "--input-dir", type=str)
def main(input_dir):
    files = os.listdir(os.path.join(input_dir, "images"))
    
    with Pool(cpu_coun() * .8) as p:
        _ = list(tqdm.tqdm(pool.imap(check_file, files), total = len(files)))
    