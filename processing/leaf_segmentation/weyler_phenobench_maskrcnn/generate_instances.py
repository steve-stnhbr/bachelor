import cv2
import os
import numpy as np
import click


@click.command()
@click.option("-i", 
              "--input_path",
              )
@click.option("-j", default=False)
def main(path, jpg_images):
    for file in os.listdir(path + "/images"):
        if jpg_images:
            file = file.replace(".jpg", ".png")
        img = cv2.imread(path + "/masks/" + file)
        cv2.imwrite(path + "/plant_instances/" + file, np.where(img != 0, np.ones(img.shape), np.zeros(img.shape)))
        cv2.imwrite(path + "/leaf_instances/" + file, np.where(img != 0, np.ones(img.shape), np.zeros(img.shape)))

if __name__ == '__main__':
    main()