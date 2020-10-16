import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
import argparse
from os.path import isdir


#Args

ap = argparse.ArgumentParser()

ap.add_argument("--save_dir", type=str, required=True,
                help="directory to save checkpoint")

ap.add_argument("--arch", type=str, default="vgg16",
                help="define what architecture to use")

ap.add_argument("--lr", type=int, default=0.001,
                help="define learning rate")

ap.add_argument("--hidden_units", type=int, default=4096,
                help="define amount of hidden units")

ap.add_argument("--epochs", type=int, default=4,
                help="define amount of epochs")

ap.add_argument("--gpu", action="store_true",
                help="will use the gpu instead of cpu")

args = ap.parse_args()

#Define transforms
def train_transform(train_dir):
    train_transforms = transforms.Compose([transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.RandomRotation(30),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data
    
def valid_transform(valid_dir):
    valid_transforms = transforms.Compose([transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    return valid_data
    
def test_transform(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    
#Define dataloaders
def data_loader(loader):
    if loader == 'train_data':
        data_load = torch.utils.data.DataLoader(loader, batch_size=32, shuffle=True)
    elif loader == 'valid_data':
        data_load = torch.utils.data.DataLoader(loader, batch_size=32)
    else:
        data_load = torch.utils.data.DataLoader(loader, batch_size=32)
        
    return data_load

#Check for GPU
def gpu_check(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("Using CPU since GPU not available.")
    
    return device
        

#Define loader model
def loader_model(architecture):
    model = eval("models.{}(pretrained=True)".format(architecture))
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

#Classifier Function

def classifier(model, hidden_units):
        
    in_features_ = model.classifier[0].in_features
    
    print(in_features_)
    
    classifier_ = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features_, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier_
    
    return model.classifier

#Validation function
def validation(model, testLoader, criterion, device):
    test_loss = 0
    accuracy = 0
    for inputs, labels in testLoader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

#Training Function

def data_train(model, epochs, criterion, trainLoader, testLoader, optimizer, device):
    
    print("Starting training:")
    with active_session():
        print_every = 40
        steps = 0
        running_loss = 0
    
        for e in range(epochs):
            model.train()
            for images, labels in iter(trainLoader):
                steps += 1
            
                images, labels = images.to(device), labels.to(device)
            
                optimizer.zero_grad()
            
                output = model.forward(images)
                loss = criterion(output, labels)
            
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
            
                if steps % print_every == 0:
                    model.eval()
                
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, testLoader, criterion, device)
                    
                        print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(testLoader)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(testLoader)))
                    
                        running_loss = 0
                    
                        model.train()
                  
    print('Training finished') 
    
    return Model

#Testing network
def network_validate(model, testLoader, device):
    with active_session:
        correct = 0
        total = 0
    
        with torch.no_grad():
            model.eval()
            for data in testLoader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs,data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))
    
#Saving model to checkpoint
def checkpoint_save(model, save_dir, train_data):
    
    if isdir(save_dir):
        model.class_to_idx = train_data.class_to_idx
    
        checkpoint = {
            'architecture': model.name,
            'class_to_idx': model.class_to_idx,
            'classifier': model.classifier,
            'state_dict': model.state_dict()
        }
    
        torch.save(checkpoint, 'my_checkpoint.pth')
        
    else:
        print('Filepath does not exist. Please choose a valid path.')
        
#Check GPU
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA wasn't found. Will use CPU.")
    return device
    
##Main function to be run

def main():
    
    #Define directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define datasets
    train_data = train_transform(train_dir)
    valid_data = train_transform(valid_dir)
    test_data = train_transform(test_dir)
    
    #Define dataloaders
    trainLoader = data_loader(train_data) 
    validLoader = data_loader(valid_data)
    testLoader = data_loader(test_data)

    #Load model
    ##model = loader_model(architecture=args.arch)
    model = loader_model(args.arch)
    
    #Build classifier
    model.classifier = classifier(model, args.hidden_units)
    
    #Check for GPU
    device = check_gpu(args.gpu)
    
    #Switch to GPU
    model.to(device)
    
    #Load learning rate
    learning_rate = args.lr
    
    #Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    #Train classifier
    trained_model = data_train(model, args.epochs, criterion, trainLoader, testLoader, optimizer, device)
    
    #Validate model
    network_validate(trained_model, testLoader, device)
    
    #Save model
    checkpoint_save(trained_model, args.save_dir, train_data)

    
    #Run application
    
if __name__ == '__main__': main()
    
