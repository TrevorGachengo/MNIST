# to do
- create missing folders when doing anything directory related

## Tested Parameters
- Sigmoid, No deskewing, 300 HU,  Batch size = 50, Learning rate = 1e-4, Accuracy: 98.15%

- Sigmoid, No deskewing, 300 HU, Batch size = 50, Learning rate = 1e-4 Adam, Accuracy: 96%

- Sigmoid, No deskewing, 300 HU, Batch size = 50, Learning rate = 1e-4 Adam with 1.1x when delta accuracy < 0.5, Accuracy: 97.87%

- Sigmoid, Deskewing, 300 HU, Batch size = 600, Learning rate = 1e-4 + Adam with 1.1x when delta accuracy < 0.5, Accuracy: 98.7%

- Sigmoid, Deskewing, 300 HU, Batch size = 2048, Learning rate = 1e-3 + Adam with 1.5x when delta accuracy < 0.5, Accuracy: 98.63% 
