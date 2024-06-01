import os

def main():
    for file in os.listdir("images"):
        try:
            for field in ["plant_instances", "leaf_instances", "semantics"]:
                if not os.path.isfile(os.path.join(field, file.replace("rgb", "label"))):
                    print("Purged", file)
                    os.rename(os.path.join("images", file), os.path.join("images_purged", file))
                    raise ConnectionRefusedError()
        except ConnectionRefusedError:
            continue

if __name__ == '__main__':
    main()