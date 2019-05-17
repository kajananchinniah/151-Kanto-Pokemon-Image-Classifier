import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler

#Constants
data_dir = 'dataset/'
num_workers = 0
batch_size = 32
validation_percent = 0.2 
n_epochs = 25
cuda_avaliability = torch.cuda.is_available()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
model = models.vgg16(pretrained = True)
classifier_input_size = 1024
classifier_output_size = 151
classifier_h1_size = int( (classifier_input_size + classifier_output_size)/2 )

#Defining train model    
def train_model(model, n_epochs, train_loader, valid_loader, cuda_avaliability):
    file_save_to = '151_Pokemon_image_classifier.pt'
    if model == None:
        return False #Error case
    
    if cuda_avaliability == True:
        model = model.cuda()
            
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.03)
    min_valid_loss = 99999 #Arbiturary big number
    
    #Training
    for e in range(0, n_epochs, 1):
        train_loss = 0  
        valid_loss = 0
        accuracy = 0
        model.train()
        for images, labels in train_loader:
            if cuda_avaliability == True:
                images = images.cuda()
                labels = labels.cuda()
                
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + loss.item()
          
        #Validating
        model.eval()
        for images, labels in valid_loader:
            if cuda_avaliability == True:
                images = images.cuda()
                labels = labels.cuda()
        
            output = model.forward(images)
            loss = criterion(output, labels)
        
            #Measuring validation loss
            valid_loss = valid_loss + loss.item()
            
            #Measuring accuracy
            top_prob, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view((*top_class.shape))
            if cuda_avaliability == True:
                accuracy = accuracy + torch.mean(equals.type(torch.cuda.FloatTensor))
            else:
                accuracy = accuracy + torch.mean(equals.type(torch.FloatTensor))
        
        #Computing averages and printing                    
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        
        print('\n.... EPOCH #', e, ' ....')
        print('train_loss = ', train_loss)
        print('valid_loss = ', valid_loss)
        print('test accuracy = ', accuracy.item()/len(valid_loader))
        
        #Saving if current valid loss is less than the min valid loss ever obtained
        if valid_loss <= min_valid_loss:
            print('Saving model...')
            torch.save(model.state_dict(), file_save_to)
            min_valid_loss = valid_loss
            
    return True


#Creating transform
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean,std)])
    
#Loading dataset
dataset = datasets.ImageFolder(data_dir, transform = transform)

#Splitting data into train and valid set
num_train = len(dataset)
indices = list(range(num_train))
numpy.random.shuffle(indices)
split = int(numpy.floor(validation_percent * num_train))
train_ind = indices[split:]
valid_ind = indices[:split]
train_sampler = SubsetRandomSampler(train_ind)
valid_sampler = SubsetRandomSampler(valid_ind)

#Loading iterators for both
train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                           sampler = valid_sampler, num_workers = num_workers)

#Creating classes
classes = []
p = Path(data_dir)
dirs = p.glob('*')
for folder in dirs:
    label = str(folder).split('/')[-1]
    label = label.replace("dataset\\", "")
    classes.append(label)

model = models.densenet121(pretrained = True)
#Freezing parameters
for p in model.parameters():
    p.requires_grad = False 
    
#Defining classifier
classifier = nn.Sequential(nn.Linear(classifier_input_size, classifier_h1_size), 
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(classifier_h1_size, classifier_output_size))
model.classifier = classifier

print(model)

train_model(model, n_epochs, train_loader, valid_loader, cuda_avaliability)


