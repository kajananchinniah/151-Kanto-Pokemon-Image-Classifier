'''
-------------------------------------------------------------
test.py
-------------------------------------------------------------
Script that does the following:
    - Intalizes some constants
    - Applys transforms
    - Loads data and gets the iterator
    - Gets list of all classes (in terms of name)
    - Intializes and loads the model. If CUDA is avaliable, uses that
    - Gets tensor containing list of images and labels
    - Makes predictions
    - Visualizes result:
        - Non-bracketed part on plot is what machine thought the image was
        - Bracketed part is what it actually is
        - Green: machine was correct, red: machine was wrong
-------------------------------------------------------------

'''
import torch
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import image_visual

#Constants
batch_size = 32
num_workers = 0
data_dir = 'predict/'
remove_from_class_string = 'predict\\'
train_on_gpu = torch.cuda.is_available()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
file_load_from = '151_Pokemon_image_classifier.pt'
im_len = 224
im_wid = 224


#Applying transform
transform = transforms.Compose([transforms.Resize(im_len), transforms.CenterCrop(im_len),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

#Loading prediction/test set and getting iterator
dataset = datasets.ImageFolder(data_dir, transform = transform)

predict_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle= True, 
                                             num_workers = num_workers)


#Classes contains list of all classes (based on directory labels)
classes = []
p = Path(data_dir)
dirs = p.glob('*')
for folder in dirs:
    label = str(folder).split('/')[-1]
    label = label.replace(remove_from_class_string, '')
    classes.append(label)

#Initiliazing and loading model
model = network.network()
model.load_state_dict(torch.load(file_load_from))
if train_on_gpu == True:
    model = model.cuda()

#Getting tensor with images and labels 
dataiter = iter(predict_loader)
images, labels = dataiter.next()
print(images.shape)

if train_on_gpu == True:
    images = images.cuda()
    labels = labels.cuda()

#Making a prediction and getting a tensor of top predictions for the batch  
output = model(images)
top_prob, top_class = output.topk(1, dim = 1)

#Visualizing data
image_visual.visualizeData(images, mean, std, classes, top_class, labels)
