from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import numpy as np



class toy_dataset(Dataset):

    def __init__(self,data):

        self.data = data

    def __len__(self):

        return self.data.shape[0]
    
    def __getitem__(self, idx):

        return torch.from_numpy(self.data[idx])


def get_loaders(data,labels = [],test_size = 0.4,seed=None,batch_size=256,num_workers=1):

    if len(labels) == 0:
        X_train, X_cv = train_test_split(data,test_size=test_size,random_state=seed)
        X_val,X_test = train_test_split(X_cv,test_size=0.5,random_state=seed)
        ret_labels = ([],[],[])
    else:
        X_train,X_val,X_test = [],[],[]
        labs_train,labs_val,labs_test = [],[],[]
        classes = np.unique(labels)
        for label in classes:
            d_class = data[labels ==label]
            
            train_class,cv_class = train_test_split(d_class,test_size=test_size,random_state=seed)
            val_class,test_class= train_test_split(cv_class,test_size=0.5,random_state=seed)
            X_train.append(train_class)
            X_val.append(val_class)
            X_test.append(test_class)
            labs_train.append(np.array([d_class]*len(train_class)))
            labs_test.append(np.array([d_class]*len(test_class)))
            labs_val.append(np.array([d_class]*len(val_class)))

        X_train = np.vstack(X_train)
        X_val = np.vstack(X_val)
        X_test = np.vstack(X_test)
        labs_train,labs_val,labs_test = np.hstack(labs_train),np.hstack(labs_val),np.hstack(labs_test)
        ret_labels = (labs_train,labs_val,labs_test)


    DS_train,DS_val,DS_test = toy_dataset(X_train),toy_dataset(X_val),toy_dataset(X_test)

    loaders = {
        'train': DataLoader(DS_train,batch_size=batch_size,shuffle=True,num_workers=num_workers),
        'val': DataLoader(DS_val,batch_size=batch_size,shuffle=False,num_workers=num_workers),
        'test':DataLoader(DS_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    }

    return loaders,ret_labels
