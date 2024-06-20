import cv2
import os
import numpy as np
import click

@click.command()
@click.option("-i", 
              "--input_path",
              )
@click.option("--jpeg", "-j", default=False, is_flag=True)
def main(input_path, jpeg):
    for file in os.listdir(input_path + "/images"):
        if jpeg:
            file = file.replace(".jpg", ".png")
        img = cv2.imread(input_path + "/masks/" + file)
        cv2.imwrite(input_path + "/plant_instances/" + file, np.where(img != 0, np.ones(img.shape), np.zeros(img.shape)))
        cv2.imwrite(input_path + "/leaf_instances/" + file, np.where(img != 0, np.ones(img.shape), np.zeros(img.shape)))

if __name__ == '__main__':
    main()