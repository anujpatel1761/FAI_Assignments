import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a feedforward neural network. 

Please only complete the model definition and do not include any training code.

The model should be a feedforward neural network, that accepts 784 inputs (each image is 28x28, and is flattened for input to the network)
and the output size is 10. Whether you need to normalize outputs using softmax depends on your choice of loss function.

PyTorch documentation is available at https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton 
requires normalized outputs or not.

'''


# class FF_Net(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # Fully Connected Layer 1
#         self.fc1 = nn.Linear(28 * 28, 256)
#         self.bn1 = nn.BatchNorm1d(256)
        
#         # Fully Connected Layer 2
#         self.fc2 = nn.Linear(256, 128)
#         self.bn2 = nn.BatchNorm1d(128)
        
#         # New Hidden Layer
#         self.fc3 = nn.Linear(128, 64)
#         self.bn3 = nn.BatchNorm1d(64)
        
#         # Final Output Layer
#         self.fc4 = nn.Linear(64, 10)
        
#         # Dropout Layer
#         self.dropout = nn.Dropout(0.3)  # Reduced dropout rate for better learning

#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))  # First hidden layer
#         x = self.dropout(x)               # Dropout
        
#         x = F.relu(self.bn2(self.fc2(x)))  # Second hidden layer
#         x = self.dropout(x)                # Dropout
        
#         x = F.relu(self.bn3(self.fc3(x)))  # Third hidden layer
#         x = self.dropout(x)                # Dropout
        
#         x = self.fc4(x)  # Final output layer
#         return x
class FF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Increased number of neurons and added layers
        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.fc5 = nn.Linear(64, 10)
        
        self.dropout = nn.Dropout(0.5)  # Increased dropout to prevent overfitting

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01)
        x = self.dropout(x)
        
        x = self.fc5(x)  # Output layer without activation
        return x