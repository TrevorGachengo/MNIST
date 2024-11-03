#  2-Layer Neural Network From Scratch  | MNIST Database 

## Contents
### Interfacer
- This module opens a canvas that allows users to draw digits, which the network then attempts to classify based on the training it received on the MNIST dataset.

### Activation functions
- Hidden layer: ReLU
- Output layer: Sigmoid

### Optimizer
- Adam, for efficient gradient descent.


## Motivation for project
I initially aimed to create a network capable of solving a Rubik's cube, but I realized I had no foundational knowledge of how neural networks actually worked. This led me to the idea of building a multiclass classification model from scratch, without using libraries like PyTorch or TensorFlow. This approach has allowed me to grasp the core fundamentals of neural networks and further develop my problem-solving skills.

  
## References
- [**The Complete Mathematics of Neural Networks and Deep Learning**](https://www.youtube.com/watch?v=Ixl3nykKG9M)

- [Neural Networks and Deep Learning Book](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Digit Classifier Reference Code](https://github.com/kdexd/digit-classifier/blob/master/network.py)

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

## Problems and Limitations 
When using the Interfacer the accuracy doesn't quite match with the value you get when the network is trained. This discrepancy is due to the difference in image properties between the images used to train the network and the images from the Interfacer. The images used to train the network have the number centered in the middle of the image, when drawing on the canvas the number may not be perfectly in the center. In addition, the method used to reshape the image may not be 100% accurate leading to a lower accuracy.
