import os
import re
import tqdm
import click
import shutil
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool, cpu_count

def process_images(files, output_dir, image_subdir, mask_subdir):
    image_file, mask_file = files
    file_name = os.path.basename(image_file)
    
    mask = Image.open(mask_file).convert("RGB")
    mask = np.array(mask)
    gray_mask = np.zeros(mask.shape[:2])
    
    unique_values = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0)
    for i, value in enumerate(unique_values):
        gray_mask[np.all(mask == value, axis=-1)] = i
    
    gray_mask_image = Image.fromarray(gray_mask.astype(np.uint8))
    gray_mask_image.save(os.path.join(mask_subdir, file_name))
    
    shutil.move(image_file, os.path.join(image_subdir, file_name))
    

@click.command()
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--image-pattern", type=str)
@click.option("--mask-pattern", type=str)
@click.option("--image-subdir", type=str)
@click.option("--mask-subdir", type=str)
@click.option("--scan-subdirs", is_flag=True)
def main(input_dir, output_dir, image_pattern, mask_pattern, image_subdir, mask_subdir, scan_subdirs):
    use_pattern = (image_pattern is not None) or (mask_pattern is not None)
    if use_pattern and (image_pattern is None) ^ (mask_pattern is None):
        print("Both patterns need to be defined")
        return
    
    if use_pattern:
        print("Utilizing patterns!")
        if scan_subdirs:
            mask_files = [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files if re.search(mask_pattern, file)]
            image_files = [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files if re.search(image_pattern, file)]
        else:
            mask_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if re.search(mask_pattern, file)]
            image_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if re.search(image_pattern, file)]
    else:
        mask_files = [os.path.join(input_dir, mask_subdir, file) for file in os.listdir(os.path.join(input_dir, mask_subdir))]
        image_files = [os.path.join(input_dir, image_subdir, file) for file in os.listdir(os.path.join(input_dir, image_subdir))]
    
    files = list(zip(image_files, mask_files))
        
    image_output_path = os.path.join(output_dir, image_subdir)
    mask_output_path = os.path.join(output_dir, mask_subdir)
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)
    
    with Pool(int(cpu_count() * .8)) as p:
        list(
             tqdm.tqdm(
                p.imap(
                    partial(process_images, output_dir=output_dir, image_subdir=image_output_path, mask_subdir=mask_output_path),
                    files
                ),
                total=len(files)
             )
        )
        
        
if __name__ == '__main__':
    main()
    
    