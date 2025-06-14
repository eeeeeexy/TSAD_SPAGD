
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset


class TimeDataset(Dataset):
    def __init__(self,
                 dataset_name:str='swat',
                 mode:str='train',
                 slide_win:int=15,
                 slide_stride:int=5,
                 window_size=50,
                 step=50,
                 path_prefix:str='dataset/processed_data/'):
        super(TimeDataset, self).__init__()
        self.dataset_name = dataset_name
        self.mode = mode
        self.slide_win = slide_win
        self.slide_stride = slide_stride
        self.path_prefix = path_prefix
        self.dataset, self.labels = self.load_raw_data()
        self.x, self.y, self.labels = self.slicing()
        self.number_nodes = self.x[0].size(0)
        self.input_dim = self.x[0].size(1)

        self.window_size = window_size
        self.step = step

        self.x_win = self.create_windows(self.x)
        self.y_win = self.create_windows(self.y)
        self.label_win = self.create_windows(self.labels)

    # Function to create windows from the data
    def create_windows(self, data):
        windows = []

        for i in range(0, data.shape[0] - self.window_size + 1, self.step):
            windows.append(data[i:i + self.window_size])
        kk = [windows[i].shape for i in range(len(windows))]
        
        return torch.stack(windows, dim=0).contiguous()
    
    def load_raw_data(self):
        data_load_path = os.path.join(self.path_prefix,
                                      self.dataset_name,
                                      self.mode+".csv")
        data = pd.read_csv(data_load_path)
        
        name_load_path = os.path.join(self.path_prefix,
                                      self.dataset_name,
                                      "list.txt")
        
        with open(name_load_path, 'r') as f:
            names = [name.strip() for name in f]
        
        dataset = [data.loc[:, name].values.tolist() for name in names]
        dataset = torch.Tensor(dataset) # size [num_nodes, total_time_len]
        
        labels = data.loc[:, 'attack'].values.tolist()
        labels = torch.Tensor(labels) # size [total_time_len]
        
        return dataset, labels
        

    def slicing(self):
        x, y, labels = [], [], []

        total_time_len = self.dataset.size(1)
        slide_stride = self.slide_stride if self.mode == 'train' else 1
        for i in range(self.slide_win, total_time_len, slide_stride):
            x.append(self.dataset[:, i-self.slide_win : i]) # size [num_nodes, slide_win]
            y.append(self.dataset[:, i]) # size [num_nodes]
            labels.append(self.labels[i]) # size [1]

        x = torch.stack(x, dim=0).contiguous()
        y = torch.stack(y, dim=0).contiguous()
        labels = torch.Tensor(labels).contiguous()

        
        return x, y, labels

    def __len__(self):
        return len(self.x_win)
    
    def __getitem__(self, idx):

        x = self.x_win[idx]
        y = self.y_win[idx]
        label = self.label_win[idx]

        # flip list
        # label_init = self.label_win[idx]
        # flipped_list = [int(1 - x) if x in (0, 1) else x for x in label_init.tolist()]
        # label = torch.tensor(flipped_list)

        return x, y, label, idx


############################# Functions #############################
    
def get_loaders(train_dataset: TimeDataset,
                test_dataset: TimeDataset,
                batch_size: int,
                val_ratio: float=0.1):
    
    # get train, val and test dataloader
    train_dataset_len = int(len(train_dataset))
    train_len = int(train_dataset_len * (1 - val_ratio))

    # import pdb; pdb.set_trace()
    
    random_indices = torch.randperm(train_dataset_len)
    train_indices = random_indices[:train_len]
    valid_indices = random_indices[train_len:]
    
    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(train_dataset, valid_indices)
    
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader