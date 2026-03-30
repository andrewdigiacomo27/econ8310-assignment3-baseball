#week 7 - neural networks

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

class CustomMNIST(Dataset):
    def __init__(self, url):
        #read in our raw data from the hosting URL
        self.raw_data = pd.read_csv(url)
    
    #return the length of the complete data set
    #   to be used in internal calculations for pytorch
    def __len__(self):
        return self.raw_data.shape[0]

    #retrieve a single record based on index position 'idx'
    def __getitem__(self, idx):
        #extract the image separate from the label
        image = self.raw_data.iloc[idx, 1:].values.reshape(1, 28, 28)
        #specify dtype to align with default dtype used by weight matrices
        image = torch.tensor(image, dtype=torch.float32)
        #extract the label
        label = self.raw_data.iloc[idx, 0]

        #return the image and its corresponding label
        return image, label

#load our data into memory
train_data = CustomMNIST("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/mnistTrain.csv")
test_data = CustomMNIST("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/mnistTest.csv")

#create data feed pipelines for modeling
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#check that our data look right when we sample
idx=1
print(f"This image is labeled a {train_data.__getitem__(idx)[1]}")
px.imshow(train_data.__getitem__(idx)[0].reshape(28,28))

class FirstNet(nn.Module):
    def __init__(self):
        #we define the components of our model here
        super(FirstNet, self).__init__()
        #function to flatten our image
        self.flatten = nn.Flatten()
        #create the sequence of our network
        self.linear_relu_model = nn.Sequential(
            #add a linear output layer w/ 10 perceptrons
            nn.LazyLinear(10),
        )
    def forward(self, x):
        #we construct the sequencing of our model here
        x = self.flatten(x)
        #pass flattened images through our sequence
        output = self.linear_relu_model(x)

        #return the evaluations of our ten classes as a 10-dimensional vector
        return output

#create an instance of our model
model = FirstNet()

#define some training parameters
learning_rate = 1e-2
batch_size = 64
epochs = 20

#define our loss function
#   this one works for multiclass problems
loss_fn = nn.CrossEntropyLoss()

#build our optimizer with the paramters from the model we defined,
#   and the learning rate that we picked
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #set the model to training mode
    #important for batch normalization and dropout layers
    #unnecessary in this situation but added for best practices
    model.train()
    #loop over batches via the dataloader
    for batch, (X, y) in enumerate(dataloader):
        #compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropogation and looking for improved gradients
        loss.backward()
        optimizer.step()
        #zeroing out the gradient (otherwise they are summed)
        #   in preparation for next round
        optimizer.zero_grad()

        #print progress update every few loops
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5f}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    #set the model to evaluation mode
    #important for batch normalization and dropout layers
    #unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    #evaluating the model with torch.no_grad() ensures
    #   that no gradients are computed during test mode
    #also serves to reduce unnecessary gradient computations
    #and memory usage for tensors with requires_grad=True
    with torch.no_grad():
            for X, y in dataloader:
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    #printing some output after a testing round
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#need to repeat the training process for each epoch.
#   In each epoch, the model will eventually see EVERY
#   observations in the data
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

#to make more predictions
#decide if we are loading for predictions or more training
model.eval()
# - or -
#model.train()

#make predictions
pred = model(test_data.__getitem__(1)[0]).argmax()
truth = test_data.__getitem__(1)[1]
print(f"This image is predicted to be a {pred}, and is labeled as {truth}")



#save our model for later, so we can train more or make predictions

EPOCH = epochs
#we use the .pt file extension by convention for saving pytorch models
PATH = "model.pt"
#the save function creates a binary sotring all our data for us
torch.save({
    'epoch': EPOCH,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }, PATH)

# ----- pretending that this is a new file:
# this is the code to reload the data back into python in a new file

#specify our path
PATH = "model.pt"
#create a new "blank" model to load our information into
model = FirstNet()
#recreate our optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#load back all of our data from the file
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
EPOCH = checkpoint['epoch']