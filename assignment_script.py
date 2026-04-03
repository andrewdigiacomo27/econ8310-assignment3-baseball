#for reading data

import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#for visualizing
import plotly.express as px

#for model building
import torch
import torch.nn as nn
import torch.nn.functional as F

#reference file from folder
import os

#building first class
class DataCleaning(Dataset):
    def __init__(self, folder):
        #reading raw data from hosting URL
        #in this case would be folder from site (annotations or actual videos)
        self.folder = folder
        self.data = []
        self.readData = []

        for current, subfolder, files in os.walk(folder):   #variables for folder, subfolder, and files in each folder
            for file in files:                                  #cycling through files in folder to add to list
                data.append(os.path.join(current, file))

        for i in data:                                      #read the information in each file into new list
            readData.append(open(i).read())

    #returning length of complete data set to be used in internal calculations for pytorch
    #use it to see how many files are in the folder you are wanting to extract
    def __len__(self):
        return len(self.readData)


        #complete up until here



    #retrieve single record based on index position 'idx'
    #need to create a process that extracts each file(frame) from the list
    def __getitem__(self, idx):
        #extract the image separate from the label
        #figure out what the parts of the frame are to separate
        frame = self.raw_data.iloc[idx, 1:].values.reshape(1, 28, 28)
        #specify dtype to align with default dtype used by weight matrices
        frame = torch.tensor(frame, dtype=torch.float32)
        #extract the label
        #figure out where the label is for baseball
        label = self.raw_data.iloc[idx, 0]
        #add in a section that labels the baseball as moving?
        moving = self.raw_data.iloc[idx, "moving"]      #----unsure about this one
        #return the image and its label, and moving?
        return frame, label, moving

#loading data into memory
train_data = DataCleaning("C:/Users/andre/Downloads/baseballannotated")
#create data feed pipelines for modeling
train_dataloader = DataLoader(train_data, batch_size=64)

class CustomMNIST(Dataset):
    def __init__(self, url):
        #read in our raw data from the hosting URL
        self.raw_data = pd.read_csv(url)
    
    #return the length of the complete data set
    #   to be used in internal calculations for pytorch
    def __len__(self):
        return self.raw_data.shape[0]









#saving files from downloaded folder into list
#running through list, and then adding the read xml file into a new list
folder = "C:/Users/andre/Downloads/baseballannotated"
data = []

for current, subfolder, files in os.walk(folder):
    for file in files:
        data.append(os.path.join(current, file))

readData = []

for i in data:
    readData.append(open(i).read())