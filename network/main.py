import numpy as np
from mnistloader import mnist
from network import Network

train, test = mnist(False)

net = Network()
net.epoch(100, train, test, display=True)   

net.save_parameters()
#saved parameters currently have accuracy of 97% on deskewing data



'''
- Sigmoid | No deskewing
    50 HU | Batch size = 50
    Learning rate = 1e-4 Adam
    Accuracy: 95.14% after 100 epochs

- Sigmoid | No deskewing
    50 HU | Batch size = 50
    Learning rate = 1e-4
    Accuracy: 97.03% after 100 epochs

- Sigmoid | No deskewing
    300 HU | Batch size = 50
    Learning rate = 1e-4
    Accuracy: 98.15%

- Sigmoid | No deskewing
    300 HU | Batch size = 50
    Learning rate = 1e-4 Adam
    Accuracy: 96%

- Sigmoid | No deskewing
    300 HU | Batch size = 50
    Learning rate = 1e-4 Adam with 1.1x when delta accuracy < 0.5
    Accuracy: 97.87%

- Sigmoid | Deskewing
    300 HU | Batch size = 600
    Learning rate = 1e-4 Adam with 1.1x when delta accuracy < 0.5
    Accuracy: 98.7%
'''