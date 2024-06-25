import os
import click
import random
import shutil

INPUT_PATH = "data"

@click.command()
@click.option("-i", "--input", default=INPUT_PATH)
@click.option("-n", "--amount", default=250)
@click.option("-o", "--output", default="data/test")
@click.option("-c", "--classes", is_flag=True)
def main(input, amount, output, classes=False):
    files = []
    for dir_path, dir_names, file_names in os.walk(input):
        files.extend([{'file': file, 'path': dir_path} for file in file_names])
    
    sampled = random.sample(files, amount)

    for i, sample in enumerate(sampled):
        if classes:
            class_dir = os.path.basename(os.path.normpath(sample['path']))
            dest = os.path.join(output, class_dir)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(os.path.join(sample['path'], sample['file']), dest)
        else:
            _, ext = os.path.splitext(sample['file'])
            shutil.copyfile(os.path.join(sample['path'], sample['file']), os.path.join(output, f"{os.path.basename(os.path.normpath(sample['path']))}_{i}.{ext}"))

if __name__ == '__main__':
    main()