#for reading data
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#for visualizing
import plotly.express as px

#for model building
import torch
import torch.nn as nn
import torch.nn.functional as F

#building first class
class DataCleaning(Dataset):
    def __init__(self, file):
        #reading raw data from hosting URL
        self.raw_data = pd.read_xml(file)

    #returning length of complete data set to be used in internal calculations for pytorch
    def __len__(self):
        return self.raw_data.shape[0]

    #retrieve single record based on index position 'idx'
    def __getitem__(self, idx):
        #extract the image separate from the label
        image = self.raw_data.iloc[idx, 1:].values.reshape(1, 28, 28)
        #specify dtype to align with default dtype used by weiht matrices
        image = torch.tensor(image, dtype=torch.float32)
        #extract the label
        label = self.raw_data.iloc[idx, 0]
        #return the image and its label
        return image, label

#loading data into memory
train_data = DataCleaning("https://uofnebraska-my.sharepoint.com/:f:/r/personal/00904883_nebraska_edu/Documents/Baseball%20Detection%20Videos/Annotations?csf=1&web=1&e=OHLkZQ")
#create data feed pipelines for modeling
train_dataloader = DataLoader(train_data, batch_size=64)


#check that data looks right when sample
idx = 1