import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

EPOCHS = 1000 #Number of forward pass and backward pass of all the training examples
BATCH_SIZE = 5 #The number of training examples in one forward/backward pass
VAL_PCT = 0.2 #Percentage of validation sample from training data
LEARNING_COEF = 0.1 #Learning rate of nn


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f'Running on {torch.cuda.get_device_name(device)}')
    print(f'CUDA Device count: {torch.cuda.device_count()}')
else:
    device = torch.device('cpu')
    print('Running on CPU')


#Class for Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #4 Node Input Layer  --> 5 Node Hidden Layer ---> 3 Node Output Layer
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        #Hidden layer
        x = self.fc1(x)
        #Output layer
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


#Training function
def train(input, target, batch, epochs, net, optimizer):
    for epoch in range(epochs):
        for i in tqdm(range(0, len(input), batch)):
            batch_X = input[i:i+batch]
            batch_y = target[i:i+batch]

            optimizer.zero_grad()
            outputs = net(batch_X.type(torch.float).to(device))

            loss = euclidean_loss(outputs, batch_y.to(device))
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} loss: {loss}')


#Validation function, returns accuracy
def validate(testX, testy, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in zip(testX, testy):
            prediction = net(X.type(torch.float).to(device))
            prediction = torch.argmax(prediction)
            if torch.argmax(y) == prediction:
                correct += 1
                #print("Correct")
            total += 1

    return round(correct/total, 3)


#For shuffling data
def randomize(X, y):
    perm = np.random.permutation(X.shape[0])
    shuffled_X = X[perm]; shuffled_y = y[perm]
    return shuffled_X, shuffled_y


#Euclidean loss
def euclidean_loss(input, target):
    return ((input-target)**2).sum() / len(input)


path = 'data/irisData.mat'
mat = scipy.io.loadmat(path) #Loads irisData.mat

data_X = np.array(mat.get('X')) #Gets 'X' and puts it on a np array
data_y = np.array(mat.get('y')) #Gets 'y' and puts it on a np array
data_y = np.reshape(data_y, (150, 1)) #To understand the data more easily I reshaped y to (150,1)


data_X, data_y = randomize(data_X, data_y) #Shuffles X and y

labels = np.eye(3) #Labels [1,0,0],[0,1,0],[0,0,1]
target = np.zeros((150, 3)) #Target outputs

#To pass the y to neural network output must match the nn's output
for i in range(len(data_y)):
    target[i] = labels[data_y[i]-1]


#Validation sample size
val_size = int(len(data_X)*VAL_PCT)
print(f'Validation sample size: {val_size}')

#Convert np array to torch
data_X = torch.tensor([i for i in data_X]).view(-1, 4)
target = torch.tensor([i for i in target])

#Training sample
train_X = data_X[:-val_size]
train_y = target[:-val_size]

#Validation sample
test_X = data_X[-val_size:]
test_y = target[-val_size:]

#Initializing nn
net = Net().to(device)
print(f'Type of fc1 weights: {net.fc1.weight.type()}')
print(f'Type of fc2 weights: {net.fc2.weight.type()}')

#Optimizer initialization, for this nn we chose stochastic gradient descent as optimizer
optimizer = optim.SGD(net.parameters(), lr=LEARNING_COEF)

#training nn
train(train_X, train_y, BATCH_SIZE, EPOCHS, net, optimizer)
accuracy = validate(test_X, test_y, net)
print(f'Accuracy is : {accuracy}')
