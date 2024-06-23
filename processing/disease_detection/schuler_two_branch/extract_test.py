import os
import click
import random
import shutil

INPUT_PATH = "data"

@click.command()
@click.option("-i", "--input", default=INPUT_PATH)
@click.option("-n", "--amount", default=250)
@click.option("-o", "--output", default="data/test")
def main(input, amount, output):
    files = []
    for dir_path, dir_names, file_names in os.walk(input):
        files.extend([{'file': file, 'path': dir_path} for file in file_names])
    
    sampled = random.sample(files, amount)

    for sample in sampled:
        shutil.copyfile(os.path.join(sample.path, sample.file), os.path.join(output, sample.file))

    