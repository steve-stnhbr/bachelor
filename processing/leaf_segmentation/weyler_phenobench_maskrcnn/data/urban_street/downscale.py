import cv2
import os
from multiprocessing import Pool
from tqdm import tqdm

FIELDS = ['images', 'leaf_instances', 'plant_instances', 'semantics']
RESOLUTIONS = [0.5, 0.25, 0.1]
RES_DIRS = [os.path.join("..", os.getcwd() + str(res).replace(".", '_')) for res in RESOLUTIONS]

def main():
    for dir in RES_DIRS:
        for field in FIELDS:
            os.makedirs(os.path.join(dir, field), exist_ok=True)

    files = os.listdir(FIELDS[0])

    with Pool(12) as p:
        r = list(tqdm(p.imap(resize_file, files), total=len(files)))
    
def resize_field(field):
    for file in os.listdir(field):
        img = cv2.imread(os.path.join(field, file), cv2.IMREAD_UNCHANGED)
        for i, res in enumerate(RESOLUTIONS):
            img_resized = cv2.resize(img, (int(float(img.shape[0]) * res), int(float(img.shape[1]) * res)))
            cv2.imwrite(os.path.join(RES_DIRS[i], field, file), img_resized)

def resize_file(file):
    for field in FIELDS:
        img = cv2.imread(os.path.join(field, file), cv2.IMREAD_UNCHANGED)
        for i, res in enumerate(RESOLUTIONS):
            img_resized = cv2.resize(img, (int(float(img.shape[0]) * res), int(float(img.shape[1]) * res)))
            cv2.imwrite(os.path.join(RES_DIRS[i], field, file), img_resized)


if __name__ == '__main__':
    main()

