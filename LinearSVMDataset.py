import torch
import uuid
import numpy as np
from torch.utils.data import Dataset, DataLoader

class LinearSVMDataset(Dataset):
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return {
            'id': self.id[index], 
            'x': self.x[:,index],
            'y': self.y[index]
        }

def datasets_generate_linear_svm(features=6, examples=5):

    # Set random seed for reproducibility
    torch.manual_seed(1)

    # Initialize x, y, w, b
    w_true = torch.randn(1, features) 
    b_true = torch.randn(1)
    x = torch.randn(features, examples)
    y = (w_true.matmul(x) + b_true).sign()
    id = np.array([uuid.uuid4() for _ in range(examples)])

    dataset = LinearSVMDataset(id, x, y)
    
    return dataset

def custom_collate(batch):
   id = [item['id'] for item in batch]  
   x = torch.stack([item['x'] for item in batch])
   y = np.array([item['y'] for item in batch])
   y = torch.from_numpy(y).long()
   return {'id': id, 'x': x, 'y': y}



# Example usage
dataset = datasets_generate_linear_svm()
# 使用自定义collate_fn
dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)

for data in dataloader:
    print(data)