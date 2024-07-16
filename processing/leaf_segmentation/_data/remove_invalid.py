import os
import tqdm
import click
import shutil
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool, cpu_count

def check_file(file, input_dir):
    image = os.path.join(input_dir, "images", file)
    instances = os.path.join(input_dir, "leaf_instances", file)

    disc_image = os.path.join(input_dir, "disc_images", file)
    disc_instances = os.path.join(input_dir, "disc_leaf_instances", file)
    im = Image.open(instances).convert("L")
    values = np.unique(im)
    if 255 in values or len(values) == 1:
        # remove images
        shutil.move(image, disc_image)
        shutil.move(instances, disc_instances)
        

@click.command()
@click.argument("input_dir", type=str)
def main(input_dir):
    files = [file for file in os.listdir(os.path.join(input_dir, "images")) if os.path.isfile(os.path.join(input_dir, "images", file))]
    os.makedirs(os.path.join(input_dir, "disc_images"), exist_ok=True)
    os.makedirs(os.path.join(input_dir, "disc_leaf_instances"), exist_ok=True)
    
    with Pool(int(cpu_count() * .8)) as pool:
        _ = list(tqdm.tqdm(pool.imap(partial(check_file, input_dir=input_dir), files), total = len(files)))
        
if __name__ == '__main__':
    main()
    