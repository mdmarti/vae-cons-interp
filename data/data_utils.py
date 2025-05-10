from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import os



def toy_dataset(Dataset):

    def __init__(self,data):

        self.data = data

    def __len__(self):

        return self.data.shape[0]
    
    def __getitem__(self, idx):

        return torch.from_numpy(self.data[idx])


def get_loaders(data,test_size = 0.4,seed=None,batch_size=256,num_workers=1):


    X_train, X_cv = train_test_split(data,test_size=test_size,random_state=seed)
    X_val,X_test = train_test_split(X_cv,test_size=0.5,seed=seed)

    DS_train,DS_val,DS_test = toy_dataset(X_train),toy_dataset(X_val),toy_dataset(X_test)

    loaders = {
        'train': DataLoader(DS_train,batch_size=batch_size,shuffle=True,num_workers=num_workers),
        'val': DataLoader(DS_val,batch_size=batch_size,shuffle=False,num_workers=num_workers),
        'test':DataLoader(DS_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    }

    return loaders
