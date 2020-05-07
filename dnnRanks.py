import numpy as np
from random import randint
import collections
import pickle
from torch.utils import data
import matplotlib.pyplot as pp
from pylab import *   
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
from functools import reduce
from softRank import softRank
#from MyReLU import MyReLU


def mayberand(a,b,TF):
    if TF==True:
        return randint(a,b)
    else:
        return b
    

n_cols = 5
n_rows = 10000  #need 10,000

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = n_cols      # n_jobs * n_tasksperjob * 2    (machine ID, duration)
hidden_size = n_cols     # math.ceil( 2.0/3.0 * input_size )
num_classes = 10
num_epochs = 100
batch_size = 100
learning_rate = 0.001

def ssum(l):
    c = 0
    for i in range(0,len(l)):
        c = c+l[i]
    return c

def ranks(l):
    l = -1.0*np.array(l)
    sigma = np.argsort(l)
    sigma_inv = np.argsort(sigma)+1
    return sigma_inv.tolist()

#[4 ,1 ,6 ,3 ,2 , 0, 10,5 ,7 ,8 , 9]
#this = [  55,999,23,344,567,1001, 1,43,22,9,4 ]
#print("x = ", this)
#print("ranks = ",  ranks(this) )
#input("waiting")

ranklabel = ranks( np.random.rand(n_cols) )
x = [ sorted( np.random.rand(n_cols) ) for i in range(0,n_rows) ]
#x = [ ( np.random.rand(n_cols) ) for i in range(0,n_rows) ]
y = [ ranklabel for i in x ]  # same label for every sample
print("rank label = ", ranklabel)

split_ind = math.ceil( 0.8*n_rows )

# create training data
training_x = x[:split_ind]    #[  x[i] for i in range(0,split_ind)  ]
training_y = y[:split_ind]


# convert to tensor
training_x = torch.Tensor(training_x)
training_y = torch.Tensor(training_y)


# create testing data
testing_x  = x[split_ind:]
testing_y  = y[split_ind:]

# convert to tensor
testing_x = torch.Tensor(testing_x)
testing_y = torch.Tensor(testing_y)


train_tensor = data.TensorDataset( training_x, training_y )
test_tensor  = data.TensorDataset(  testing_x,  testing_y )

train_loader = data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)
test_loader  = data.DataLoader(dataset = test_tensor,  batch_size = batch_size, shuffle = False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1  = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc2  = nn.Linear(hidden_size, hidden_size)
        self.srank  = softRank()
        #self.myrelu = MyReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        #out = self.myrelu.apply(out)
        #out = self.relu(out)
        #out = self.fc2(out)
        out = self.srank.apply(out)
        
        return out

    
model = NeuralNet(input_size, input_size, num_classes).to(device)   #JK the model lives on the device

# Loss and optimizer
#criterion = nn.CrossEntropyLoss()     #JK-when ready:
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
loss_list =[] #JK
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, n_cols).to(device)     # JK -1 means figure out #rows 
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) 
        
        # Backward and optimize
        optimizer.zero_grad()  #JK reset the gradient, the optimizer accumulates them if not reset (usage is for RNN apparently)
        loss.backward()
        #input("waiting")
        optimizer.step()
        loss_list.append(loss.item()) #JK
        if (i+1) % 40 == 0:
            # Test the model 
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:

                    outputs = model(images)   
                    total += labels.size(0)
                    correct += torch.all( (outputs == labels).bool(),1 ).sum().item()
            
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item(),100 * correct / total))


         


# Save the model checkpoint
#torch.save(model.state_dict(), 'softRankDNN_state.ckpt')  #99.7%
torch.save(model.state_dict(), 'zeroDNN_state.ckpt')


print("printing one output: \n")
print(model( torch.Tensor([221,222,223,224,225])[None,:] ))



