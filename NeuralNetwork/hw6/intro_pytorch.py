import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):    
    transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    if training == True:
        train_set=datasets.FashionMNIST('./data', train=training, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
        return train_loader
    else:
        test_set=datasets.FashionMNIST('./data', train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle=False)
        return test_loader
    

def build_model():
    untrained_model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_features= 28*28, out_features=120),
              nn.ReLU(),
             nn.Linear(in_features=120, out_features=60),
              nn.ReLU(),
             nn.Linear(in_features=60, out_features=10))
    return untrained_model


def train_model(model, train_loader, criterion, T):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    n = len(train_loader.dataset)
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()*50

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
        print(f'Train Epoch: {epoch}\tAccuracy: {correct}/{n}({correct/n*100:.2f}%)\tLoss: {running_loss/n:.3f}')    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    n = len(test_loader.dataset)
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()*50
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    if not show_loss:
        print(f'Accuracy: {(correct/n)*100:.2f}%')
    else:
        print(f'Average loss: {(running_loss/n):.4f}')
        print(f'Accuracy: {(correct/n)*100:.2f}%')
    
    

def predict_label(model, test_images, index):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    logits = model(test_images)
    prob = F.softmax(logits,dim=1)
    probs,labels = torch.topk(prob[index],3)
    
    for i in range(3):
        print(f'{class_names[labels[i]]}: {probs[i]*100:.2f}%')
    


if __name__ == '__main__':
    train_loader = get_data_loader()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    test_loader = get_data_loader(False)
    test_images = test_loader.dataset[0][0]
    test_images = test_images[None]
    test_images = torch.cat((x, (test_loader.dataset[1][0])[None]),0)
    
    train_model(model, train_loader, criterion, 5)
    print()
    evaluate_model(model, test_loader, criterion, True)
    print()
    evaluate_model(model, test_loader, criterion, False)
    print()
    predict_label(model, test_images, 1)
    
