from torch.utils import data
from torchvision import transforms
import cv2

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, root):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.root = root

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]

        X = cv2.imread(self.root + ID)
        y = self.labels[index]
        
        data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        X = data_transform(X)    
        return X, y
    
    
class Dataset_to_sent(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, root):
        'Initialization'
        self.list_IDs = list_IDs
        self.root = root

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        X = cv2.imread(self.root + ID)
        
        data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        X = data_transform(X)    
            
        return X