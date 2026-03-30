#week 9 - neural networks - LSTMs part 2

#libraries from part 1
import requests
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#for visualizing
import plotly.express as px

#for model building
import torch
import torch.nn as nn
import torch.nn.functional as F

#import helpers
import nnhelpers as nnh
from d2l import torch as d2l

# Time series models inspired by 
#   https://machinelearningmastery.com/
#     lstm-for-time-series-prediction-in-pytorch/

import pandas as pd
import plotly.express as px
import numpy as np
from torch.utils.data import TensorDataset

# Read in data, grab relevant column
#weather data from Omaha
temp = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/omahaNOAA.csv")
temp = temp['HOURLYDRYBULBTEMPF'].fillna(0).replace(0, method='pad').values[-(365*24):]

px.line(temp)

# train-test split for time series
train_size = int(len(temp) * 0.67)
test_size = len(temp) - train_size
train, test = temp[:train_size], temp[train_size:]

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

class TempLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=lookback,
             hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.LazyLinear(1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = TempLSTM()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(X_train, y_train),
     shuffle=True, batch_size=32)
 
n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 10 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(temp) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1]
    train_plot[lookback:train_size] = model(X_train)[:, -1]
    # shift test predictions for plotting
    test_plot = np.ones_like(temp) * np.nan
    test_plot[train_size+lookback:len(temp)] = model(X_test)[:, -1]

# Build plotting data
plot_data = pd.DataFrame([temp, test_plot]).T
plot_data.columns = ['truth', 'forecast']

px.line(plot_data, y = ['truth', 'forecast'])