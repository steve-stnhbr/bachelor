import os 
import cv2

def main():
    images = os.listdir("semantics")
    img = cv2.imread(os.path.join("semantics", images[0]), cv2.IMREAD_UNCHANGED)
    print(img.shape)

if __name__ == '__main__':
    main()