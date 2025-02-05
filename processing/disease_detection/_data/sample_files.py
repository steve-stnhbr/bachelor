import os
import click
import random
import shutil

@click.command()
@click.option("-i", "--input")
@click.option('-o', '--output')
@click.option('-s', '--split', default=None)
@click.option('-r', '--ratio', type=float)
def main(input, output, split, ratio):
    
    files = []
    for root, dirs, dir_files in os.walk(input):
        files.extend([os.path.join(os.path.relpath(root, input), file) for file in dir_files])

    sampled = random.sample(files, (int(len(files) * ratio)))

    if split is not None:
        files = list(set(files).difference(set(sampled)))
        for file in files:
            os.makedirs(os.path.join(split, os.path.dirname(file)), exist_ok=True)
            shutil.copy(os.path.join(input, file), os.path.join(split, file))

    for file in sampled:
        os.makedirs(os.path.join(output, os.path.dirname(file)), exist_ok=True)
        shutil.copy(os.path.join(input, file), os.path.join(output, file))
    
if __name__ == '__main__':
    main()
