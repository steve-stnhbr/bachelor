import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class SingleClassDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Since it's a single class, we can use a constant label (e.g., 0)
        label = 0
        
        return image, label

# Usage example:
folder_path = '_data/urban_street0_25/images'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = SingleClassDataset(folder_path, transform=transform)

# Create a DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

data_test = '_data/urban_street0_25/images_test'
dataset_test = datasets.ImageFolder(data_test, transform=transform)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(actual, predicted):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Assuming actual and predicted are numpy arrays or lists of the same shape
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Calculate metrics
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average='weighted')
    recall = recall_score(actual, predicted, average='weighted')
    f1 = f1_score(actual, predicted, average='weighted')

    # Return metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

class Autoencoder:
    class AutoencoderNetwork(nn.Module):
        def __init__(self):
            super(self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 7)
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 7),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def __init__(self):
        self.model = self.AutoencoderNetwork()
    
    def train(self):
        self.model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        print("starting training")
        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            for data in dataloader_train:
                img, _ = data
                img = img.to(device)  # Move input data to GPU
                
                recon = self.model(img)
                loss = criterion(recon, img)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            torch.save(self.model, "out/autoencoder_latest.pth")
    
    def predict(self, img):
        return self.model(img) #TODO

    def test(self):
        y_true = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for data in dataloader_test:
                img, label = data
                img = img.to(device)
                recon = self.predict(img)
                y_true.append(label)
                y_pred.append(recon)

        return calculate_metrics(y_true, y_pred)
        
class OneClassSVM:

    def __init__(self):
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.oc_svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')

    def train(self):
        imgs, labels = dataloader_train
        X_features = extract_features(imgs)
        X_train_scaled = self.scaler.fit_transform(X_features)
        self.oc_svm.fit(X_train_scaled)

    def test(self):
        imgs, labels = dataloader_test
        X_features = extract_features(imgs)
        X_test_scaled = self.scaler.transform(X_features)
        y_pred_test = self.oc_svm.predict(X_test_scaled)

        return calculate_metrics(labels, y_pred_test)


def extract_features(img, return_nodes=None):
    import torch
    import torchvision.models as models
    from torchvision.models.feature_extraction import create_feature_extractor
    # Load a pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Define the layers you want to extract features from
    if return_nodes is None: 
        return_nodes = {
            'layer1': 'feature1',
            'layer2': 'feature2',
            'layer3': 'feature3',
            'layer4': 'feature4'
        }

    # Create a feature extractor
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    with torch.no_grad():
        features = feature_extractor(img)
    
    return features