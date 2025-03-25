import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
################################
import matplotlib.pyplot as plt

conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer of the network
weights = conv_net.conv1.weight.data  # Shape: (out_channels, in_channels, kernel_size, kernel_size)

#Remove in channels dimension  1 for grayscale images)
kernels = weights.squeeze(1)  # Shape: (out_channels, kernel_size, kernel_size)

# Normalize the kernels to be between 0 and 1
min_k = kernels.min()
max_k = kernels.max()
kernels_normalized = (kernels - min_k) / (max_k - min_k)

# Add channel dimension back for plotting
kernels_normalized = kernels_normalized.unsqueeze(1)  # Shape: (out_channels, 1, kernel_size, kernel_size)

# Create a grid of kernel images
grid_kernels = torchvision.utils.make_grid(kernels_normalized, nrow=8, padding=1)






# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.
plt.figure(figsize=(10, 10))
np_grid = grid_kernels.numpy()
plt.imshow(np.transpose(np_grid, (1, 2, 0)))
plt.title('First Conv Layer Kernels')
plt.axis('off')
plt.savefig('kernel_grid.png')



# Apply the kernel to the provided sample image.
img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0					# Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image

output = conv_net.conv1(img)

# Convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.

output = output.squeeze(0)
output_normalized = (output - output.min()) / (output.max() - output.min())
output_normalized = output_normalized.unsqueeze(1)

# Create a grid of feature maps
grid_output = torchvision.utils.make_grid(output_normalized, nrow=8, padding=1)




# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.
plt.figure(figsize=(10, 10))
np_grid_output = grid_output.numpy()
plt.imshow(np.transpose(np_grid_output, (1, 2, 0)))
plt.title('Feature Maps from First Conv Layer')
plt.axis('off')
plt.savefig('image_transform_grid.png')
