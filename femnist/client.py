import warnings

from tqdm import tqdm
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Process
import numpy as np
import json
import random

warnings.filterwarnings("ignore", category=UserWarning)


class FemnistDataset(Dataset):
    def __init__(self, dataset, transform):
        self.x = dataset['x']
        self.y = dataset['y']
        self.transform = transform

    def __getitem__(self, index):
        input_data = np.array(self.x[index]).reshape(28, 28, 1)
        if self.transform:
            input_data = self.transform(input_data)
        target_data = self.y[index]
        
        return input_data, target_data

    def __len__(self):
        return len(self.y)


class femnist_network(nn.Module):
    def __init__(self) -> None:
        super(femnist_network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = nn.Linear(7*7*64, 2048)
        self.linear2 = nn.Linear(2048, 62)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu((self.linear1(x)))
        x = self.linear2(x)
        return x



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = femnist_network().to(DEVICE)

def load_data():
    """Load FEMNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    number = random.randint(0, 35)
    if number == 35:
        subject_number = random.randint(0, 96)
    else:
        subject_number = random.randint(0, 99)

    print('number : {}, subject number : {}'.format(number, subject_number))

    with open("./data/data/train/all_data_"+str(number)+"_niid_0_keep_0_train_9.json","r") as f:
        train_json = json.load(f)
    with open("./data/data/test/all_data_"+str(number)+"_niid_0_keep_0_test_9.json","r") as f:
        test_json = json.load(f)
    
    train_user = train_json['users'][subject_number]
    train_data = train_json['user_data'][train_user]
    
    test_user = test_json['users'][subject_number]
    test_data = test_json['user_data'][test_user]
    
    trainset = FemnistDataset(train_data, transform)
    testset = FemnistDataset(test_data, transform)
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64)
    
    return trainloader, testloader


class FemnistClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainloader, _ = load_data()
        train(net, trainloader, epochs=20)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        _, testloader = load_data()
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0003)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE).float(), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


def client_fn():
    return FemnistClient()