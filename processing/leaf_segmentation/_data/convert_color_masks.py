import os
import re
import tqdm
import click
import shutil
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool, cpu_count

def process_images(image_file, mask_file, output_dir, image_subdir, mask_subdir):
    file_name = os.path.basename(image_file)
    
    mask = Image.open(mask_file)
    gray_mask = np.zeros(mask[:2])
    
    unique_values = np.unique(mask)
    for i, value in unique_values:
        gray_mask[mask == value] = i
    
    gray_mask_image = Image.fromarray(gray_mask.astype(np.uint8))
    gray_mask_image.save(os.path.join(mask_subdir, file_name))
    
    shutil.move(image_file, os.path.join(image_subdir, file_name))
    

@click.command()
@click.argument("input_dir")
@click.argument("output_dir")
@click.argument("--image-pattern", type=str)
@click.argument("--mask-pattern", type=str)
@click.argument("--image-subdir", type=str)
@click.argument("--mask-subdir", type=str)
def main(input_dir, output_dir, image_pattern, mask_pattern, image_subdir, mask_subdir):
    use_pattern = (image_pattern is None) or (mask_pattern is None)
    if use_pattern and (image_pattern is None) ^ (mask_pattern is None):
        print("Both patterns need to be defined")
    
    if use_pattern:
        mask_files = [os.path.join(input_dir, file) for file os.listdir(input_dir) if re.match(mask_pattern) is not None]
        image_files = [os.path.join(input_dir, file) for file os.listdir(input_dir) if re.match(image_pattern) is not None]
    else:
        mask_files = [os.path.join(input_dir, mask_subdir, file) for file in os.listdir(os.path.join(input_dir, mask_subdir))]
        image_files = [os.path.join(input_dir, image_subdir, file) for file in os.listdir(os.path.join(input_dir, image_subdir))]
    
    files = zip(image_files, mask_files)
        
    image_output_path = os.path.join(output_dir, image_subdir)
    mask_output_path = os.path.join(output_dir, mask_subdir)
    os.mkdirs(image_output_path, exist_ok=True)
    os.mkdirs(mask_output_path, exist_ok=True)
    
    with Pool(int(cpu_count() * .8)) as p:
        _ = list(tqdm.tqdm(p.imap(partial(process_images, output_dir=output_dir, image_subdir=image_output_path, mask_subdir=mask_output_path), files), total=len(files)))
        
        
if __name__ == '__main__':
    main()
    
    