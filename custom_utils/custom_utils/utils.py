import os
def get_classes(paths):
    return [os.path.normpath(path).split(os.sep)[-2] for path in paths]