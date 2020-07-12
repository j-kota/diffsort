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

# !! Last left off - not training, very high starting test loss, training loss starts out increasing


# Fully connected neural network with one hidden layer
class multiRankNet(nn.Module):
    def __init__(self, input_size, n_cols, num_classes):
        super(multiRankNet, self).__init__()
        
        hidden_size = input_size
        self.fc1  = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc2  = nn.Linear(hidden_size, hidden_size)
        self.srank  = softRank()

        # for testing: a collection of linears to use on different segments of the input
        self.partialLin = {i: nn.Linear(n_cols, n_cols) for i in range(0,howManyRanks) }
        
        #self.myrelu = MyReLU()
        self.splitLin = {i: nn.Linear(input_size, n_cols) for i in range(0,howManyRanks) }
        self.splitRank = {i: softRank() for i in range(0,howManyRanks) }

        
    def forward(self, x):

    
        x_in = {i: x[:,i*n_cols:(i+1)*n_cols] for i in range(0,howManyRanks)}
        
        #print("x_in.items() = ",  x_in.items() )

        h = dict()
        for i in range(0,howManyRanks):
            h[i] = self.partialLin[i]( x_in[i] )

        r = dict()
        for i in range(0,howManyRanks):
            r[i] = self.splitRank[i].apply( h[i] )

        #print("r.items() = ",  r.items() )
       

        out = torch.cat( tuple(r.values()), 1 )
        return out

              
        """
        out = self.fc1(x)

        h = dict()
        for i in range(0,howManyRanks):
            h[i] = self.splitLin[i](out)

        r = dict()
        for i in range(0,howManyRanks):
            r[i] = self.splitRank[i].apply( out )#h[i] )

        out = torch.cat( tuple(r.values()), 1 )
        return out
        """





        
def catenate(l):
    return reduce( (lambda l,r: l+r), l+[] )


def mayberand(a,b,TF):
    if TF==True:
        return randint(a,b)
    else:
        return b
    



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

howManyRanks = 1
n_cols = 5
n_rows = 10000 

ranklabel = catenate(   [ ranks( np.random.rand(n_cols) ) for i in range(0,howManyRanks) ]   )
x = [ sorted( np.random.rand(n_cols*howManyRanks) ) for i in range(0,n_rows) ]
#x = [ ( np.random.rand(n_cols) ) for i in range(0,n_rows) ]
y = [ ranklabel for i in x ]  # same label for every sample


input_size  = n_cols*howManyRanks 
num_classes = 10
num_epochs = 200
batch_size = 400
learning_rate = 0.001



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

train_loader = data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = False)
test_loader  = data.DataLoader(dataset = test_tensor,  batch_size = batch_size, shuffle = False)


    
model = multiRankNet(input_size, n_cols, num_classes).to(device)   #JK the model lives on the device

# Loss and optimizer
#criterion = nn.CrossEntropyLoss()     #JK-when ready:
criterion = nn.MSELoss()
test_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
loss_list = [] #JK
accu_list = []
trainloss_list = []
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) 
        
        # Backward and optimize
        optimizer.zero_grad()  #JK reset the gradient, the optimizer accumulates them if not reset (usage is for RNN apparently)
        loss.backward()
        #input("waiting")
        optimizer.step()
        #loss_list.append(loss.item()) #JK
        if True:  #(i+1) % 80 == 0:
            # Test the model 
            with torch.no_grad():
                correct = 0
                
                total = 0
                testloss = 0
                for images, labels in test_loader:

                    outputs = model(images)   
                    total += labels.size(0)
                    testloss += test_criterion(outputs,labels).item() 
                    correct += torch.all( (outputs == labels).bool(),1 ).sum().item()
            
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, TestLoss: {:.4f}, Accuracy: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item(), testloss, 100 * correct / total))

                loss_list.append(loss.item())
                accu_list.append(100*correct/total)
                trainloss_list.append( testloss / total )
                


         


# Save the model checkpoint
#torch.save(model.state_dict(), 'softRankDNN_state.ckpt')  #99.7%
torch.save(model.state_dict(), 'relu_rank_len15_state.ckpt')

pickle.dump( (loss_list, trainloss_list, accu_list) , open("relu_rank_len15_plot.p", "wb")) 

plt.plot( range(1,len(loss_list)+1), loss_list, 'r' )
plt.show()

#print("printing one output: \n")
#print(model( torch.Tensor([221,222,223,224,225])[None,:] ))




