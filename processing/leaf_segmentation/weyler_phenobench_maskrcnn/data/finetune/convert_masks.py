import cv2
import os

INPUT_DIR = "masks_red"
OUTPUT_DIR = "masks"

def main():
    for file in os.listdir(INPUT_DIR):
        img = cv2.imread(os.path.join(INPUT_DIR, file))
        non_black_mask = (img != [0, 0, 0]).any(axis=-1)
    
        # Setze alle nicht-schwarzen Pixel auf Weiß
        img[non_black_mask] = [255, 255, 255]
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, file), img)

if __name__ == '__main__':
    main()