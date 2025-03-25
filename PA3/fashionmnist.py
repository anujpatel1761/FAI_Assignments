import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *
##############chnages#####################
import matplotlib.pyplot as plt
import numpy as np

'''

In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.

'''

'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([                            
    transforms.ToTensor(),                                  
    transforms.Normalize(mean=[0.5], std=[0.5])             
])

batch_size = 128   

'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

trainset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)



testset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False)


'''
PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.
'''

feedforward_net = FF_Net()
conv_net = Conv_Net()

'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

criterion = nn.CrossEntropyLoss()  

optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=0.001)  
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001)  

'''

PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)

'''

num_epochs_ffn = 5

 
train_loss_ffn = []

for epoch in range(num_epochs_ffn): 
    running_loss_ffn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Flatten inputs for ffn
        inputs = inputs.view(-1, 28*28)  

        # zero the parameter gradients
        optimizer_ffn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, labels) 
        loss.backward()
        optimizer_ffn.step()
        running_loss_ffn += loss.item()

    train_loss_ffn.append(running_loss_ffn / len(trainloader))
    print(f"Epoch {epoch+1}, Training loss: {running_loss_ffn / len(trainloader)}")


print('Finished Training FFN')
torch.save(feedforward_net.state_dict(), 'ffn.pth') 

################cnn######################################

num_epochs_cnn = 5 
train_loss_cnn = []

for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
    running_loss_cnn = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # zero the parameter gradients
        optimizer_cnn.zero_grad()

        # forward + backward + optimize
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)  # YOUR CODE HERE
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()

    train_loss_cnn.append(running_loss_cnn / len(trainloader))
    print(f"Epoch {epoch+1}, Training loss: {running_loss_cnn / len(trainloader)}")

print('Finished Training CNN')

torch.save(conv_net.state_dict(), 'cnn.pth') 

'''

PART 6:
Evaluate your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.

'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

correct_ffn = 0
total_ffn = 0

correct_cnn = 0
total_cnn = 0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Variables to store images for plotting
found_correct_ffn = False
found_incorrect_ffn = False

found_correct_cnn = False
found_incorrect_cnn = False

with torch.no_grad():  # since we're not training, we don't need to calculate the gradients for our outputs
    for data in testloader:
        images, labels = data

        # FFN Evaluation
        images_flat = images.view(-1, 28*28)
        outputs_ffn = feedforward_net(images_flat)
        _, predicted_ffn = torch.max(outputs_ffn.data, 1)
        total_ffn += labels.size(0)
        correct_ffn += (predicted_ffn == labels).sum().item()

        # CNN Evaluation
        outputs_cnn = conv_net(images)
        _, predicted_cnn = torch.max(outputs_cnn.data, 1)
        total_cnn += labels.size(0)
        correct_cnn += (predicted_cnn == labels).sum().item()

        # Find correct and incorrect examples for plotting
        for i in range(len(labels)):
            if not found_correct_ffn:
                if predicted_ffn[i] == labels[i]:
                    correct_image_ffn = images[i]
                    correct_label_ffn = labels[i]
                    correct_pred_ffn = predicted_ffn[i]
                    found_correct_ffn = True
            if not found_incorrect_ffn:
                if predicted_ffn[i] != labels[i]:
                    incorrect_image_ffn = images[i]
                    incorrect_label_ffn = labels[i]
                    incorrect_pred_ffn = predicted_ffn[i]
                    found_incorrect_ffn = True
            if not found_correct_cnn:
                if predicted_cnn[i] == labels[i]:
                    correct_image_cnn = images[i]
                    correct_label_cnn = labels[i]
                    correct_pred_cnn = predicted_cnn[i]
                    found_correct_cnn = True
            if not found_incorrect_cnn:
                if predicted_cnn[i] != labels[i]:
                    incorrect_image_cnn = images[i]
                    incorrect_label_cnn = labels[i]
                    incorrect_pred_cnn = predicted_cnn[i]
                    found_incorrect_cnn = True
            if (found_correct_ffn and found_incorrect_ffn and
                found_correct_cnn and found_incorrect_cnn):
                break
        if (found_correct_ffn and found_incorrect_ffn and
            found_correct_cnn and found_incorrect_cnn):
            break

print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
print('Accuracy for convolutional network: ', correct_cnn/total_cnn)

'''

PART 7:

Check the instructions PDF. You need to generate some plots.

'''

'''

YOUR CODE HERE

'''

# Function to unnormalize and display images
def imshow(img):
    img = img * 0.5 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis('off')

#FFN Correct Classification
plt.figure()
imshow(correct_image_ffn)
plt.title(f"FFN Correct Classification\nPredicted: {class_names[correct_pred_ffn]}\nTrue: {class_names[correct_label_ffn]}")
plt.savefig('ffn_correct.png')

#FFN Incorrect
plt.figure()
imshow(incorrect_image_ffn)
plt.title(f"FFN Incorrect Classification\nPredicted: {class_names[incorrect_pred_ffn]}\nTrue: {class_names[incorrect_label_ffn]}")
plt.savefig('ffn_incorrect.png')

#CNNCorrect
plt.figure()
imshow(correct_image_cnn)
plt.title(f"CNN Correct Classification\nPredicted: {class_names[correct_pred_cnn]}\nTrue: {class_names[correct_label_cnn]}")
plt.savefig('cnn_correct.png')

#CNN Incorrect
plt.figure()
imshow(incorrect_image_cnn)
plt.title(f"CNN Incorrect Classification\nPredicted: {class_names[incorrect_pred_cnn]}\nTrue: {class_names[incorrect_label_cnn]}")
plt.savefig('cnn_incorrect.png')

#FFN Training Loss
plt.figure()
plt.plot(range(1, num_epochs_ffn+1), train_loss_ffn, label='FFN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('FFN Training Loss over Epochs')
plt.legend()
plt.savefig('ffn_training_loss.png')

#CNN Training Loss
plt.figure()
plt.plot(range(1, num_epochs_cnn+1), train_loss_cnn, label='CNN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training Loss over Epochs')
plt.legend()
plt.savefig('cnn_training_loss.png')
