from keras.utils import PyDataset, to_categorical
import os
from utils import get_classes

class PlantLeafsDataGen(PyDataset):
    def __init__(self, path, batch_size=32, shuffle=True, transforms=None, **kwargs):
        super().__init__(**kwargs)
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

        # self.classes = os.listdir(path)
        with open(os.path.join(path, "labels.txt")) as f: label_text = f.read()
        self.classes = label_text.split(os.linesep)
        self.num_classes = len(self.classes)

        self.file_paths = []
        for class_dir in self.classes:
            self.file_paths.extend(os.path.join(path, class_dir, file) for file in os.listdir(os.path.join(path, class_dir)))
        
        if self.shuffle:
            import random
            random.shuffle(self.file_paths)
        
        self.transforms = transforms

    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.file_paths) // self.batch_size)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        paths = self.file_paths[index*self.batch_size:(index+1)*self.batch_size]
        class_names = get_classes(paths)
        classes = [self.classes.index(clazz) for clazz in class_names]
        if self.transforms is not None and len(self.transforms) > 0:
            for transform in self.transforms:
                paths = transform(paths)

        return paths, to_categorical(classes, num_classes=self.num_classes)
        

class PlantLeafsDataGenBinary(PyDataset):
    def __init__(self, path, batch_size=32, shuffle=True, transforms=None, determine_healthy=None, **kwargs):
        super().__init__(**kwargs)
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

        self.classes = os.listdir(path)
        self.num_classes = 2

        self.file_paths = []
        for class_dir in self.classes:
            self.file_paths.extend(os.path.join(path, class_dir, file) for file in os.listdir(os.path.join(path, class_dir)))
        
        if self.shuffle:
            import random
            random.shuffle(self.file_paths)
        
        if determine_healthy is not None:
            self.determine_healthy = self.is_healthy
        else:
            self.determine_healthy = determine_healthy
        
        self.transforms = transforms

    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.file_paths) // self.batch_size)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        paths = self.file_paths[index*self.batch_size:(index+1)*self.batch_size]
        class_names = get_classes(paths)
        classes = [1 if self.determine_healthy(clazz) else 0 for clazz in class_names]
        if self.transforms is not None and len(self.transforms) > 0:
            for transform in self.transforms:
                paths = transform(paths)

        return paths, to_categorical(classes, num_classes=self.num_classes)
    
    def is_healthy(clazz):
        return "healthy" in clazz
        