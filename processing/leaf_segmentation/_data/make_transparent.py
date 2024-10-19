import os
import cv2
import click
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

def process(name, images_dir, masks_dir, output_dir):
    image_file = os.path.join(images_dir, name)
    mask_file = os.path.join(masks_dir, name)
    output_file = os.path.join(output_dir, name)

    image = cv2.imread(image_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Image {name} not found")

    if mask is None:
        print(f"mask {name} not found")


    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    bgra_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    bgra_image[:, :, 3] = mask

    cv2.imwrite(output_file, bgra_image)

@click.command()
@click.argument("images_dir")
@click.argument("masks_dir")
@click.argument("output_dir")
@click.option("-p", "--processors", type=int, default=8)
def main(images_dir, masks_dir, output_dir, processors):
    os.makedirs(output_dir, exist_ok=True)

    run = partial(process, images_dir=images_dir, masks_dir=masks_dir, output_dir=output_dir)

    with Pool(processors) as p:
        _ = list(tqdm(p.imap(run, os.listdir(images_dir))))

if __name__ == '__main__':
    main()
