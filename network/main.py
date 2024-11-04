from mnistloader import mnist
from network import Network

train, test = mnist(True)

net = Network(300, batch_size=2048, learning_rate=1e-3)
net.epoch(100, train, test, fixed_learning_rate=False, display=True)   

net.save_parameters()
