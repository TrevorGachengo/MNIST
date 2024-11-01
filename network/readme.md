# How to use the network
## Loading the dataset
Start by running the deskewer.py. This file will extract the training and test from the entire dataset file mnist.pkl.gz. It saves copies of the default data and deskewed data.
In your main.py call the mnist function from the mnistloader to load the data that you want. True for deskewed and false for default.

## Initializing and Training
The network class takes in 3 parameters:
- Number of hidden units
- Learning rate
- Batch size
> These values are defaulted to Hidden units = 50, Learning rate = 1e-4 and Batch size = 50

To train the network use the epoch function which takes in 5 parameters:
- Epochs: number of iterations to train the network
- Train data: the data to train the network with
- Test data: the data to test the network against
- Fixed learning rate: boolean to decide to change the learning rate or keep it constant.
- Display: boolean to decide to display the network training or only have it output the final accuracy 


## Saving and Loading parameters
To use the network without having to train it again, use save_parameters() and load_parameters() accordingly.

save_parameters() will create a folder of numpy arrays under "\network\data\" called trained_parameters.

## Tested Parameters
No deskewing: 
  - 300 HU,  Batch size = 50, Learning rate = 1e-4, Accuracy: 98.15%

  - 300 HU, Batch size = 50, Learning rate = 1e-4 Adam, Accuracy: 96%

  - 300 HU, Batch size = 50, Learning rate = 1e-4 Adam with 1.1x when delta accuracy < 0.5, Accuracy: 97.87%

Deskewing:
  - 300 HU, Batch size = 600, Learning rate = 1e-4 + Adam with 1.1x when delta accuracy < 0.5, Accuracy: 98.7%

  - 300 HU, Batch size = 2048, Learning rate = 1e-3 + Adam with 1.5x when delta accuracy < 0.5, Accuracy: 98.63% 
