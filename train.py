# Imports here
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models
import torchvision.transforms as transforms

from collections import OrderedDict

from PIL import Image, ImageFilter
import tensorflow as f
import numpy as np

# import argparse
import argparse
import sys
# parser.add_arguement(--'', type= , default='', help='')

# define CLI parameters
parser = argparse.ArgumentParser()
parser.add_argument('--sav_dir', type=str, default='save_directory', help='Directory to save checkpoint')
parser.add_argument('--arch', type=str, default='vgg16', help='Choose architecture - vgg16 or densenet121')
parser.add_argument('--lr', type=float, default=0.01, help='enter learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='enter # of hidden units')
parser.add_argument('--epochs', type=int, default=2, help='enter number of epochs')
parser.add_argument('--gpu_cpu', type=str, default='cuda', help='specify cuda or cpu')
args = parser.parse_args()

# assign parameters to passed arguments
in_args = parser.parse_args()
learning_rate = in_args.lr
hidden_units = in_args.hidden_units
epochs = in_args.epochs
gpu_cpu = in_args.gpu_cpu
arch = in_args.arch
sav_dir = in_args.sav_dir

# Set data directory for image data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.Resize(225),
                                      transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255)
                                     ,transforms.CenterCrop(224)
                                     ,transforms.ToTensor()
                                     ,transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
validating_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=(60), shuffle=True)
validloader = torch.utils.data.DataLoader(validating_data, batch_size =(30),shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=(25), shuffle=True)

# Import JSON file for category mapping
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Load the pre-trained network
if gpu_cpu == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

#Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)

    # Only train the classifier parameters, feature parameters are frozen
    for param in model.parameters():
        param.requires_grad=False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('drop_out', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(4096, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop_out', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

elif arch == 'densenet121':
    model = models.densenet121(pretrained=True)

     # Only train the classifier parameters, feature parameters are frozen
    for param in model.parameters():
        param.requires_grad=False

        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop_out', nn.Dropout()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

else:
    raise Exception("Architecture not accepted")


print("Begining training the model using the pretrained network: " + arch)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

torch.cuda.is_available()
model.to(device)

sys.stdout.write("\n")

# Train the classifier layers using backpropagation using the pre-trained network to get the features
# Track the loss and accuracy on the validation set to determine the best hyperparameters
steps = 0
running_loss = 0
valid_loss = 0
print_every = 40
running_losses, valid_losses = [], []
model.to(device)

for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        log_ps = model.forward(inputs)
        loss = criterion(log_ps, labels) # Loss
        loss.backward() # Calculate gradients (backpropagation)
        optimizer.step() # Adjust parametes based on gradient

        running_loss += loss.item() # Adjust training loss

    else:
        test_loss = 0
        accuracy = 0

    if steps % print_every == 0:
        with torch.no_grad():
            model.eval()
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)

                log_ps = model.forward(inputs)
                valid_loss = criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        running_losses.append(running_loss/len(trainloader))
        valid_losses.append(valid_loss/len(validloader))

        print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Training loss: {running_loss/len(trainloader):.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")

    model.train()

# Do validation on the test set
test_losses = []
model.eval()

for epoch in range(epochs):
    test_loss = 0
    for inputs, labels in testloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        log_ps = model(inputs)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        test_loss += loss.item()

    else:
        test_loss = 0
        accuracy = 0

    if steps % print_every == 0:
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                log_ps = model.forward(inputs)
                batch_loss = criterion(log_ps, labels)

                test_loss += batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                test_losses.append(test_loss/len(testloader))

        print(f"Epoch {epoch+1}/{epochs}.. " f"Test accuracy: {accuracy/len(testloader):.3f}")

# Save the checkpoint
# TODO: Save the checkpoint
print("Saving checkpoint")

model.class_to_idx = train_data.class_to_idx

checkpoint = {'arch': arch,
              'learning_rate': learning_rate,
              'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'epochs': epochs,
              'optimizer_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
