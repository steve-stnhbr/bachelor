from model import ResNet9
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchsummary import summary
import random 
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm

VERBOSE = False

# for moving data into GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# for loading in the device (GPU if available else CPU)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
print("Cuda {}available".format("" if torch.cuda.is_available() else "un"))
    
device = get_default_device()
batch_size = 1

train_dir = '../_data/test'
diseases = os.listdir(train_dir)


# datasets for validation and training
train = ImageFolder(train_dir, transform=transforms.ToTensor())
train_dl = DataLoader(train, batch_size, shuffle=True)
#train_dl = DeviceDataLoader(train_dl, device=device)

model = ResNet9(3, len(train.classes))

print("Loading model with pth file")
model.load_state_dict(torch.load('data/model/pretrained.pth'))
model.eval()

# getting summary of the model
INPUT_SHAPE = (3, 256, 256)
print(summary(model.cuda(), (INPUT_SHAPE)))

model.to(device)

# saving model
#PATH = 'data/model/model.pt'
#print("Saving model to {}".format(PATH))
#torch.save(model, PATH)

num_correct_pred = 0
num_wrong_pred = 0
data = map(lambda x: [x, 0, 0], diseases)
df = pd.DataFrame(data, columns=['disease', 'total', 'correct'])
print(df)

try:
    for image, label in tqdm(train_dl):
        image.to(device)
        if VERBOSE:
            print("Image size: {}".format(image.size()))
        inference = model(image.cuda())
        _, predicted_label = torch.max(inference, dim=1)
        if VERBOSE:
            print("Prediction: {}".format(predicted_label[0]))
            print("Actual: {}".format(label[0]))
        df.at[int(label[0]), 'total'] += 1
        if predicted_label[0] != label[0]:
            num_wrong_pred += 1
        else:
            num_correct_pred += 1
            df.at[int(label[0]), 'correct'] += 1
except KeyboardInterrupt:
    print("Accuracy: ", num_correct_pred / (num_correct_pred + num_wrong_pred))
    df.to_csv("./out/result.csv")


print("Accuracy: ", num_correct_pred / (num_correct_pred + num_wrong_pred))
df.to_csv("./out/result.csv")
print(df)