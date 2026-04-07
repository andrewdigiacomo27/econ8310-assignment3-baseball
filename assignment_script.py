#switching to pretrained model
#https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html?utm_source=chatgpt.com

# importing XML files. video found online using xml.tree.ElementTree on importing and parsing
#https://www.youtube.com/watch?v=5SlemSWGD1g

# importing video files. video found online using openCV module
#https://video.search.yahoo.com/video/play;_ylt=AwrFFynW5c5p2cwB3kf7w8QF;_ylu=c2VjA3NyBHNsawN2aWQEZ3BvcwMx?p=importing+videos+and+separating+video+frames+in+python&vid=c9ce7d4e05e775c0f1823d9d27196bfa&turl=https%3A%2F%2Ftse3.mm.bing.net%2Fth%2Fid%2FOVP.AjTplKi-isv8hx3pXAcsrQHgFo%3Fpid%3DApi%26h%3D360%26w%3D480%26c%3D7%26rs%3D1&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DECCHN1j_lis&tit=Extracting+%3Cb%3EVideo%3C%2Fb%3E+%3Cb%3EFrames%3C%2Fb%3E+Using+OpenCV+%3Cb%3Ein%3C%2Fb%3E+%3Cb%3EPython%3C%2Fb%3E+%7C+%3Cb%3EPython%3C%2Fb%3E+Project&c=0&sigr=qIUGT8oAANaB&sigt=0rHo6zFK58vl&sigi=jNYR9_QI3F_K&fr2=p%3As%2Cv%3Av&h=360&w=480&l=244&age=1656509403&fr=mcafee&type=E210US1357G0&tt=b

import os
import requests
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import cv2
import xml.etree.ElementTree as ET
import plotly.express as px

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# parse and extract labels from xml file

folder = "/content/videos/IMG"
xmlFiles = []
for file in os.listdir(folder):

    filePath = os.path.join(folder, file)
    tree = ET.parse(filePath)
    root = tree.getroot()

    for track in root.findall("track"):  #need a for loop for each id within xml file
        id = track.attrib["id"]
        label = track.attrib["label"]           #baseball label

        for child in track.findall("box"):               #getting label values from binding boxes
            frame = int(child.attrib["frame"])
            outside = int(child.attrib["outside"])
            xtl = float(child.attrib["xtl"])
            ytl = float(child.attrib["ytl"])
            xbr = float(child.attrib["xbr"])
            ybr = float(child.attrib["ybr"])

            for tag in child:
                if tag.text == "true":
                    moving = 1
                else:
                    moving = 0

            # items = [file, id, label, frame, outside, xtl, ytl, xbr, ybr, moving]
            xmlFiles.append({
                "file": os.path.splitext(file)[0],
                "id": id,
                "label": label,
                "frame": frame,
                "outside": outside,
                "boundbox": [xtl, ytl, xbr, ybr]
            })

print("Files parsed.")

# import and separate frames from videos

import cv2
import os

videoFolder = "/content/videos/Movies"
outputFolder = "/content/videos/captures"

# # creating folder (for google collab)
# outputFolder = "videos/Captures"
# os.makedirs(outputFolder, exist_ok=True)

#for loop to go through videos
for videoFile in os.listdir(videoFolder):
  path = os.path.join(videoFolder, videoFile)

  #creating folder for each video
  videoName = os.path.splitext(videoFile)[0]
  outFolder = os.path.join(outputFolder, videoName)
  os.makedirs(outFolder, exist_ok=True)

  #capture video
  capture = cv2.VideoCapture(path)

  #initialize frame number to 0
  f = 0

  #to store the frames of the video
  while(capture.isOpened()):
    ret, frame = capture.read()
    if ret == False:
      break

    # to save the frame to a specified file
    filename = os.path.join(outFolder, f"frame_{f}.jpg")
    cv2.imwrite(filename, frame)
    f += 1

  capture.release()

print("Frames extracted.")

# match labels from xml file to video frames

data = []
filePath = "/content/videos/captures"

for item in xmlFiles:

  if item["outside"] == 1:
    continue

  video = item["file"]
  frame = item["frame"]

  path1 = os.path.join(filePath, video, f"frame_{frame}.jpg")
  if not os.path.exists(path1):
    continue

  data.append({
      "frame" : path1,
      "label": item["label"],
      "boundbox": item["boundbox"]
  })

print("Dataset size:", len(data))

# convert to pytorch tensors (in baseball class - done above)

#final framing format should be (frame - video image, target - bounding box)

# baseball neural network

# pretrained COCO model - found online after struggling to create own NN
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

variables = 2     #baseball and frame

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        variables
    )

class CustomBaseball(Dataset):
  def __init__(self, data):
    self.data = data
    self.transform = torchvision.transforms.Resize((800, 800))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):      #extract and separate label, frame, boundbox
    item = self.data[idx]   #grabbing one item
    #image
    frame = read_image(item["frame"]).float()/255.0    #creating image #need to read up on this !!!!!
    frame = self.transform(frame)               #resize image
    #specify dtype to align with default dtype used by weight matrices
    boundbox = torch.tensor([item["boundbox"]], dtype=torch.float32)
    #label to tensor
    label = torch.tensor([1], dtype=torch.int64)

    target = {
        "box": boundbox,
        "label": label
        }

    return frame, target

#dataset and loader

def collate_fn(batch):          #batch function
    return tuple(zip(*batch))

dataset = CustomBaseball(data)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

#pretrained model

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 2   #frame and baseball

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #don't think I have gpu with collab
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)   #may need to change - ai recommendation


#training loop

def train_loop(train_loader, model, optimizer, device):
    size = len(train_loader.dataset)
    #set the model to training mode
    #important for batch normalization and dropout layers
    #unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(train_loader):

      # move to device
      X = X.to(device)
      y = {k: v.to(device) for k, v in y.items()}

      # REQUIRED format for Faster R-CNN
      loss_dict = model([X[0]], [y])

      # sum all losses
      loss = sum(loss for loss in loss_dict.values())

      # backprop
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      # print progress like your professor style
      if batch % 10 == 0:
          loss_value = loss.item()
          current = (batch + 1) * len(X)

          print(f"loss: {loss_value:>7f} [{current:>5d}/{size:>5d}]")


#need to repeat the training process for each epoch.
#   In each epoch, the model will eventually see EVERY
#   observations in the data
epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------")
    train_loop(train_loader, model, optimizer, device)
print("Done!")

#to make more predictions
#decide if we are loading for predictions or more training
model.eval()
# - or -
#model.train()

#make predictions
sample, test = dataset[1]
with torch.no_grad():
  pred = model([sample.to(device)])[0]
print("Prediction:", pred)
print("Test:", test)



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

#reload


# specify our path
PATH = "model.pt"

# referencing pretrained model
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 2  # baseball and frame
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# recreating optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# reload checkpoint
checkpoint = torch.load(PATH, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
EPOCH = checkpoint["epoch"]


