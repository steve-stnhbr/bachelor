import os

def main():
    for file in os.listdir("images"):
        os.rename(os.path.join("images", file), os.path.join("images", file.replace("_rgb", "")))

if __name__ == '__main__':
    main()