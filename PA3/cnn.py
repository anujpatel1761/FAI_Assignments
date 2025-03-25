import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a convolutional neural network. 

Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, and convolutional layers in PyTorch:

    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


Whether you need to normalize outputs using softmax depends on your choice of loss function. PyTorch documentation is available at
https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton requires normalized outputs or not.

'''
# # add comments
# class Conv_Net(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # First Convolutional Layer
#         # Input channels = 1 (grayscale pict), Output channels = 32
#         # Kernel size = 3x3, no padding (default padding=0), stride = 1 (default)
#         # Output size after this layer: (28 - 3 + 1) x (28 - 3 + 1) = 26 x 26
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        
#         # Max Pooling Layer
#         # Pooling operation with a 2x2 window and stride = 2
#         # Reduces spatial dimensions by half. After pooling, output size: 26 / 2 = 13 x 13
#         self.pool = nn.MaxPool2d(2, 2)

#         # Second Convolutional Layer
#         # Input channels = 32, Output channels = 64
#         # Kernel size = 3x3, no padding (default padding=0), stride = 1 (default)
#         # Output size after this layer: (13 - 3 + 1) x (13 - 3 + 1) = 11 x 11
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
#         # Another Max Pooling Layer
#         # Pooling reduces spatial dimensions by half. After pooling, output size: 11 / 2 = 5 x 5
#         # Output tensor size from this layer: 64 channels x 5 x 5 = 64 * 5 * 5 = 1600 (flattened later)
        
#         # Fully Connected Layer 1
#         # Input size = 64 * 5 * 5 (flattened tensor from previous layer)
#         # Output size = 128 (arbitrary choice, can be adjusted)
#         self.fc1 = nn.Linear(64 * 5 * 5, 128)

#         # Fully Connected Layer 2 (Output Layer)
#         # Input size = 128 (from previous layer), Output size = 10 (number of classes in Fashion-MNIST)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         # Pass input through the first convolutional layer and apply ReLU activation
#         x = F.relu(self.conv1(x))  # Output size: 26 x 26

#         # Apply max pooling to reduce spatial dimensions
#         x = self.pool(x)  # Output size: 13 x 13

#         # Pass through the second convolutional layer and apply ReLU activation
#         x = F.relu(self.conv2(x))  # Output size: 11 x 11

#         # Apply max pooling to further reduce spatial dimensions
#         x = self.pool(x)  # Output size: 5 x 5

#         # Flatten the tensor to prepare it for the fully connected layers
#         # Each image now becomes a 1D vector of size 64 * 5 * 5 = 1600
#         x = x.view(-1, 64 * 5 * 5)

#         # Pass through the first fully connected layer and apply ReLU activation
#         x = F.relu(self.fc1(x))  # Output size: 128

#         # Pass through the final fully connected layer (output layer)
#         # Produces raw logits for the 10 classes
#         x = self.fc2(x)  # Output size: 10

#         return x

class Conv_Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional Layers with BatchNorm and Activation
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.pool(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.pool(x)
        
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = self.pool(x)
        
        x = self.dropout_conv(x)
        
        # Flatten for Fully Connected Layers
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully Connected Layers with Dropout
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout_fc(x)
        
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout_fc(x)
        
        x = self.fc3(x)  # Output layer without activation
        return x
