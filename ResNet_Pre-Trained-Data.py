#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNetModel, self).__init__()
        # Load the pre-trained ResNet50 model from torchvision models
        self.resnet = models.resnet50(pretrained=True)
         
        # If you want to fine-tune the model, you can freeze the weights here
        # for param in self.resnet.parameters():
        # param.requires_grad = False
        
        # Modify the last fully connected layer to match the number of classes in your task
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Instantiate the ResNetModel
resnet_model = ResNetModel()


# In[7]:


resnet_model


# In[9]:


def extract_pretrained_weights(model):
    pretrained_weights = {}
    for name, param in model.named_parameters():
        if "fc" not in name:  # Exclude the last fully connected layer weights
            pretrained_weights[name] = param.data.clone()
    return pretrained_weights


# In[11]:


pretrained_weights = extract_pretrained_weights(resnet_model)

print(pretrained_weights)


# In[12]:


import torch
import torch.nn as nn

def reshape_and_svd(weight_matrix):
    # Reshape the weight matrix into a 2D matrix
    # The first dimension will be the product of all the other dimensions
    reshaped_matrix = weight_matrix.view(weight_matrix.size(0), -1)
    
    # Perform Singular Value Decomposition (SVD)
    # The function torch.svd returns three tensors: U, S, and Vt
    U, S, Vt = torch.svd(reshaped_matrix)
    
    return U, S, Vt
# Assuming you have extracted the pretrained_weights from the ResNet model 
# For example, let's extract the first convolutional layer weight matrix
conv1_weight_matrix = pretrained_weights['resnet.conv1.weight']

# Reshape and perform SVD
U, S, Vt = reshape_and_svd(conv1_weight_matrix)

# Print the shapes of the resulting matrices
print("U shape:", U.shape)
print("S shape:", S.shape)
print("Vt shape:", Vt.shape)


# In[48]:


import matplotlib.pyplot as plt

def plot_singular_values(S):
    # Convert the singular values to a numpy array for plotting
    singular_values = S.numpy()
      
    # Plot the singular values using a linear scale
    plt.plot(singular_values, marker='o')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Values Distribution')
    plt.grid(True)
    plt.show()

conv1_weight_matrix = pretrained_weights['resnet.conv1.weight']
U, S, Vt = reshape_and_svd(conv1_weight_matrix)

# Visualize the singular values
plot_singular_values(S)


# In[16]:


import matplotlib.pyplot as plt

def plot_singular_values(S, layer_name):
    # Convert the singular values to a numpy array for plotting
    singular_values = S.numpy()
    
    # Plot the singular values using a linear scale
    plt.plot(singular_values, marker='o', label=layer_name)
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Values Distribution')
    plt.grid(True)
    plt.legend()
    plt.show()

# Iterate through layers and plot singular values
for layer_name, weight_matrix in pretrained_weights.items():
    U, S, Vt = reshape_and_svd(weight_matrix)
    plot_singular_values(S, layer_name)


# In[ ]:




