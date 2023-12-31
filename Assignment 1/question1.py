# -*- coding: utf-8 -*-
"""Question1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13RpDEI6B20ihXLTyjngG_RywO6nCPO4y

## Libraries
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as func
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math

"""## Device Config"""

device = torch.device("cude" if torch.cuda.is_available() else "cpu")
device

"""## Hyperparameters define"""

num_epochs= 5
batch_size = 100
lr = 0.001
input_size = 3*32*32 # 3*32*32 pixel size
hidden_size1 = 100
hidden_size2 = 100
hidden_size3 = 100
# num_classes = 10
output_size = 10

"""## Composed Tranform"""

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""## Import CIFAR10 Dataset"""

train_dataset = torchvision.datasets.CIFAR10(root='./data', train= True, download=True, transform = transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size,)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train= False, transform = transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,)

"""## Verify"""

example = iter(train_loader)
# print(type(example))
sample, label = next(example)
print(sample.shape, label.shape)
sample = sample.view(-1, 3*32*32)
print(sample)
# for i in range(6):
#   plt.subplot(2, 3, i+1)
#   plt.imshow(sample[i][0])
#   print(label[i])

# plt.show()

"""#-----------------------------------------------------------------------------------------------------------------------------

# Question 1: Utilize various activation functions like sigmoid, tanh and critique the performance in each case.

## Create a Neural Network with Sigmoid Activation funtion without vanish gradient problem

### Neural Netwrok With Sigmoid
"""

class NeuralNetSigmoid(nn.Module):
  def __init__(self, input_size, hidden_size1, output_size):
    super(NeuralNetSigmoid, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size1)
    self.sigmoid = nn.Sigmoid()
    self.l2 = nn.Linear(hidden_size1, output_size)


  def forward(self, x):
    x = x.view(-1, 32*32*3)
    out = self.l1(x)
    out = self.sigmoid(out)
    out = self.l2(out)
    return out


sigmoid_model = NeuralNetSigmoid(input_size=input_size, hidden_size1=hidden_size1, output_size=output_size)
sigmoid_model
# print(sigmoid_model.state_dict())

"""### Loss function for sigmoid"""

loss_sigmoid = nn.CrossEntropyLoss()
loss_sigmoid

"""### Optimizer function for sigmoid"""

optimizer_sigmoid = torch.optim.SGD(sigmoid_model.parameters(), lr=lr)
optimizer_sigmoid

"""### Training Loop for sigmoid function"""

n_step = len(train_loader)
sig_loss = []
sig_acc = []
sig_epoch = []

for epoch in range(num_epochs):
  sig_epoch.append(epoch+1)

  epoch_loss = 0.0
  epoch_acc = 0

  n_correct = 0 # number of correct prediction
  n_samples = 0 # number of total sample

  for i ,(image, label) in enumerate(train_loader):
    
    # image = image.reshape(-1,1024).to(device)
    image = image.to(device)
    label = label.to(device)

    # Forward pass
    label_predict = sigmoid_model(image)


    # Tracking correct predictions
    _, predict = torch.max(label_predict,1)
    n_samples += label.shape[0]
    n_correct += (predict == label).sum().item()

    # loss
    l = loss_sigmoid(label_predict, label)

    # backward
    optimizer_sigmoid.zero_grad()
    l.backward()

    # Upate weights
    optimizer_sigmoid.step()

    if (i+1)%100 == 0:
      print(f"epoch : {epoch+1}/{num_epochs} | step : {i+1}/{n_step} | loss : {l.item(): .4f}")


    # Store each epoch total loss
    epoch_loss += l.item()


    # Store each epoch total loss into empty list
  sig_loss.append(epoch_loss)

  # Accuracy calculation
  acc = 100.0*(n_correct/n_samples)
  sig_acc.append(acc)
  


print(sig_loss)

print(sig_acc)

"""### Accuracy Calculation on test dataset"""

with torch.no_grad():
  n_correct = 0
  n_samples = 0

  for image, label in test_loader:
    image = image.to(device)
    label = label.to(device)

    output = sigmoid_model(image)

    # value, index
    _, predictions = torch.max(output, 1)
    n_samples += label.shape[0]
    n_correct += (predictions == label).sum().item()


  acc = 100.0 * (n_correct/n_samples)

print("Accuracy of Sigmoid : ",acc)

"""### Plot Graph"""

# Convert list to numpy array
x = np.array(sig_loss)
y = np.array(sig_acc)
z = np.array(sig_epoch)
# chart show

# set chart Size
fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10)  

#plot 1: Loss vs Accuracy
plt.title("Loss vs Accuracy")
plt.xlabel("Loss")
plt.ylabel("Accuracy")
# plt.subplot(1, 4, 1)
plt.plot(x,y)
plt.show()

#plot 2: Accuracy vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Accuracy vs Epoch")
plt.xlabel("Accuracy")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 2)
plt.plot(y,z)
plt.show()

#plot 3: Loss vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Loss vs Epoch")
plt.xlabel("Loss")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 3)
plt.plot(x,z)

plt.show()

"""##-------------------------------------------------------------------------------------------------------------

## Neural Netrwork using tanh activation function without vanishing gradient problem

### Neural Network with tanh function
"""

class NeuralNetTanh(nn.Module):
  def __init__(self, input_size, hidden_size1, output_size):
    super(NeuralNetTanh, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size1)
    self.tanh = nn.Tanh()
    self.l2 = nn.Linear(hidden_size1, output_size)

  def forward(self, x):
    x = x.view(-1, 3*32*32)
    out = self.l1(x)
    out = self.tanh(out)
    out = self.l2(out)
    
    return out


tanh_model = NeuralNetTanh(input_size=input_size, hidden_size1=hidden_size1, output_size=output_size)
tanh_model

# print(tanh_model.state_dict())

"""### Loss function for tanh"""

loss_tanh = nn.CrossEntropyLoss()
loss_tanh

"""### Optimizer function for tanh"""

optimizer_tanh = torch.optim.SGD(tanh_model.parameters(), lr= lr)
optimizer_tanh

"""### Taining Loop for tanh"""

n_steps = len(train_loader)
tanh_loss = []
tanh_acc = []
tanh_epoch = []


for epoch in range(num_epochs):
  tanh_epoch.append(epoch+1)
  
  epoch_loss = 0.0
  epoch_acc = 0


  n_correct = 0 # number of correct prediction
  n_samples = 0 # number of total sample

  for i, (image, label) in enumerate(train_loader):

    # If device uses GPU then store data to GPU
    image = image.to(device)
    label = label.to(device)
    
    # foraward pass
    output = tanh_model(image)

    # Tracking correct predictions
    _, predict = torch.max(output,1)
    n_samples += label.shape[0]
    n_correct += (predict == label).sum().item()


    # loss
    l = loss_tanh(output, label)

    # backward pass
    optimizer_tanh.zero_grad()

    l.backward()

    optimizer_tanh.step()


    if (i+1)%100 == 0:
      print(f"epoch : {epoch+1}/{num_epochs} | step : {i+1}/{n_step} | loss : {l.item(): .4f}")
    
    # Store each epoch total loss
    epoch_loss += l.item()

  # Store each epoch total loss into empty list
  tanh_loss.append(epoch_loss)

  # Accuracy calculation
  acc = 100.0*(n_correct/n_samples)
  tanh_acc.append(acc)
  


print(tanh_loss)

print(tanh_acc)

"""### Plot Graph"""

# Convert list to numpy array
x = np.array(tanh_loss)
y = np.array(tanh_acc)
z = np.array(tanh_epoch)
# chart show

# set chart Size
fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10)  

#plot 1: Loss vs Accuracy
plt.title("Loss vs Accuracy")
plt.xlabel("Loss")
plt.ylabel("Accuracy")
# plt.subplot(1, 4, 1)
plt.plot(x,y)
plt.show()

#plot 2: Accuracy vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Accuracy vs Epoch")
plt.xlabel("Accuracy")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 2)
plt.plot(y,z)
plt.show()

#plot 3: Loss vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Loss vs Epoch")
plt.xlabel("Loss")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 3)
plt.plot(x,z)

plt.show()

"""### Accuracy calculation"""

with torch.no_grad():
  n_correct = 0
  n_samples = 0

  for image,label in test_loader:
    image = image.to(device)
    label = label.to(device)

    output = tanh_model(image)

    _,predictions = torch.max(output, 1)
    n_samples += label.shape[0]
    n_correct += (predictions == label).sum().item()

  acc = 100.0*(n_correct/n_samples)

print("Accuracy of Tanh : ",acc)

"""#-----------------------------------------------------------------------------------------------------------------------------

# Question 2:Increase the depth of the given network by adding more Fully-Connected layers till the point you encounter the vanishing gradient problem. With the help of the results, mention how to identify it.

## Neural Network with Sigmoid activation function Which has vanish gradient problem

### Create Neural Network with Sigmoid function
"""

class NeuralNetSigmoidVanish(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
    super(NeuralNetSigmoidVanish, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size1)
    self.sigmoid = nn.Sigmoid()
    self.l2 = nn.Linear(hidden_size1, hidden_size2)
    self.l3 = nn.Linear(hidden_size2, hidden_size3)
    self.l4 = nn.Linear(hidden_size3, output_size)


  def forward(self, x):
    x = x.view(-1, 32*32*3)
    out = self.l1(x)
    out = self.sigmoid(out)
    out = self.l2(out)
    out = self.sigmoid(out)
    out = self.l3(out)
    out = self.sigmoid(out)
    out = self.l4(out)

    return out


sigmoid_model_vanish = NeuralNetSigmoidVanish(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3, output_size=output_size)
sigmoid_model_vanish
# print(sigmoid_model.state_dict())

"""### Loss function for sigmoid"""

loss_sigmoid = nn.CrossEntropyLoss()
loss_sigmoid

"""### Optimizer function for sigmoid"""

optimizer_sigmoid = torch.optim.SGD(sigmoid_model_vanish.parameters(), lr=lr)
optimizer_sigmoid

"""### Training Loop for sigmoid function"""

n_step = len(train_loader)
sig_loss = []
sig_acc = []
sig_epoch = []

for epoch in range(num_epochs):
  sig_epoch.append(epoch+1)

  epoch_loss = 0.0
  epoch_acc = 0

  n_correct = 0 # number of correct prediction
  n_samples = 0 # number of total sample

  for i ,(image, label) in enumerate(train_loader):
    
    # image = image.reshape(-1,1024).to(device)
    image = image.to(device)
    label = label.to(device)

    # Forward pass
    label_predict = sigmoid_model_vanish(image)


    # Tracking correct predictions
    _, predict = torch.max(label_predict,1)
    n_samples += label.shape[0]
    n_correct += (predict == label).sum().item()

    # loss
    l = loss_sigmoid(label_predict, label)

    # backward
    optimizer_sigmoid.zero_grad()
    l.backward()

    # Upate weights
    optimizer_sigmoid.step()

    if (i+1)%100 == 0:
      print(f"epoch : {epoch+1}/{num_epochs} | step : {i+1}/{n_step} | loss : {l.item(): .4f}")


    # Store each epoch total loss
    epoch_loss += l.item()


    # Store each epoch total loss into empty list
  sig_loss.append(epoch_loss)

  # Accuracy calculation
  acc = 100.0*(n_correct/n_samples)
  sig_acc.append(acc)
  


print(sig_loss)

print(sig_acc)

"""### Accuracy Calculation on test dataset"""

with torch.no_grad():
  n_correct = 0
  n_samples = 0

  for image, label in test_loader:
    image = image.to(device)
    label = label.to(device)

    output = sigmoid_model(image)

    # value, index
    _, predictions = torch.max(output, 1)
    n_samples += label.shape[0]
    n_correct += (predictions == label).sum().item()


  acc = 100.0 * (n_correct/n_samples)

print("Accuracy of Sigmoid : ",acc)

"""### Plot Graph"""

# Convert list to numpy array
x = np.array(sig_loss)
y = np.array(sig_acc)
z = np.array(sig_epoch)
# chart show

# set chart Size
fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10)  

#plot 1: Loss vs Accuracy
plt.title("Loss vs Accuracy")
plt.xlabel("Loss")
plt.ylabel("Accuracy")
# plt.subplot(1, 4, 1)
plt.plot(x,y)
plt.show()

#plot 2: Accuracy vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Accuracy vs Epoch")
plt.xlabel("Accuracy")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 2)
plt.plot(y,z)
plt.show()

#plot 3: Loss vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Loss vs Epoch")
plt.xlabel("Loss")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 3)
plt.plot(x,z)

plt.show()

"""##-------------------------------------------------------------------------------------------------------------

## Neural Network using **tanh activation** function with vanishing gradient problem

### Neural Network with tanh
"""

class NeuralNetTanhVanish(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
    super(NeuralNetTanhVanish, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size1)
    self.tanh = nn.Tanh()
    self.l2 = nn.Linear(hidden_size1, hidden_size2)
    self.l3 = nn.Linear(hidden_size2, hidden_size3)
    self.l4 = nn.Linear(hidden_size3, output_size)

  def forward(self, x):
    x = x.view(-1, 3*32*32)
    out = self.l1(x)
    out = self.tanh(out)
    out = self.l2(out)
    out = self.tanh(out)
    out = self.l3(out)
    out = self.tanh(out)
    out = self.l4(out)
    return out


tanh_model_vanish = NeuralNetTanhVanish(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3, output_size=output_size)
tanh_model_vanish

# print(tanh_model.state_dict())

"""### Tanh Loss with vanish gradient decent """

loss = nn.CrossEntropyLoss()
loss

"""### Optimizer with tanh function"""

optimizer = torch.optim.SGD(tanh_model_vanish.parameters(), lr=lr)
optimizer

n_steps = len(train_loader)
tanh_loss = []
tanh_acc = []
tanh_epoch = []


for epoch in range(num_epochs):
  tanh_epoch.append(epoch+1)
  
  epoch_loss = 0.0
  epoch_acc = 0


  n_correct = 0 # number of correct prediction
  n_samples = 0 # number of total sample

  for i, (image, label) in enumerate(train_loader):

    # If device uses GPU then store data to GPU
    image = image.to(device)
    label = label.to(device)
    
    # foraward pass
    output = tanh_model_vanish(image)

    # Tracking correct predictions
    _, predict = torch.max(output,1)
    n_samples += label.shape[0]
    n_correct += (predict == label).sum().item()


    # loss
    l = loss_tanh(output, label)

    # backward pass
    optimizer_tanh.zero_grad()

    l.backward()

    optimizer_tanh.step()


    if (i+1)%100 == 0:
      print(f"epoch : {epoch+1}/{num_epochs} | step : {i+1}/{n_step} | loss : {l.item(): .4f}")
    
    # Store each epoch total loss
    epoch_loss += l.item()

  # Store each epoch total loss into empty list
  tanh_loss.append(epoch_loss)

  # Accuracy calculation
  acc = 100.0*(n_correct/n_samples)
  tanh_acc.append(acc)
  


print(tanh_loss)

print(tanh_acc)

"""### Accuracy calculation"""

with torch.no_grad():
  n_correct = 0
  n_samples = 0

  for image,label in test_loader:
    image = image.to(device)
    label = label.to(device)

    output = tanh_model(image)

    _,predictions = torch.max(output, 1)
    n_samples += label.shape[0]
    n_correct += (predictions == label).sum().item()

  acc = 100.0*(n_correct/n_samples)

print("Accuracy of Tanh : ",acc)



"""### Plot Graph"""

# Convert list to numpy array
x = np.array(tanh_loss)
y = np.array(tanh_acc)
z = np.array(tanh_epoch)
# chart show

# set chart Size
fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10)  

#plot 1: Loss vs Accuracy
plt.title("Loss vs Accuracy")
plt.xlabel("Loss")
plt.ylabel("Accuracy")
# plt.subplot(1, 4, 1)
plt.plot(x,y)
plt.show()

#plot 2: Accuracy vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Accuracy vs Epoch")
plt.xlabel("Accuracy")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 2)
plt.plot(y,z)
plt.show()

#plot 3: Loss vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Loss vs Epoch")
plt.xlabel("Loss")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 3)
plt.plot(x,z)

plt.show()

"""#-----------------------------------------------------------------------------------------------------------------------------

# Question 3:Suggest and implement methods to overcome the above problem.

## We can remove by using relu function

### Relu Neural Network
"""

class NeuralNetRelu(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
    super(NeuralNetRelu, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size1)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size1, hidden_size2)
    self.l3 = nn.Linear(hidden_size2, hidden_size3)
    self.l4 = nn.Linear(hidden_size3, output_size)


  def forward(self, x):
    x = x.view(-1, 32*32*3)
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)
    out = self.l3(out)
    out = self.relu(out)
    out = self.l4(out)

    return out


relu_model_vanish = NeuralNetRelu(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3, output_size=output_size)
relu_model_vanish
# print(sigmoid_model.state_dict())

"""### Loss function for Relu"""

loss_relu = nn.CrossEntropyLoss()
loss_relu

"""### Optimizer function for sigmoid"""

optimizer_relu = torch.optim.SGD(relu_model_vanish.parameters(), lr=lr)
optimizer_relu

"""### Training Loop for relu function"""

n_step = len(train_loader)
relu_loss = []
relu_acc = []
relu_epoch = []

for epoch in range(num_epochs):
  relu_epoch.append(epoch+1)

  epoch_loss = 0.0
  epoch_acc = 0

  n_correct = 0 # number of correct prediction
  n_samples = 0 # number of total sample

  for i ,(image, label) in enumerate(train_loader):
    
    # image = image.reshape(-1,1024).to(device)
    image = image.to(device)
    label = label.to(device)

    # Forward pass
    label_predict = relu_model_vanish(image)


    # Tracking correct predictions
    _, predict = torch.max(label_predict,1)
    n_samples += label.shape[0]
    n_correct += (predict == label).sum().item()

    # loss
    l = loss_relu(label_predict, label)

    # backward
    optimizer_relu.zero_grad()
    l.backward()

    # Upate weights
    optimizer_relu.step()

    if (i+1)%100 == 0:
      print(f"epoch : {epoch+1}/{num_epochs} | step : {i+1}/{n_step} | loss : {l.item(): .4f}")


    # Store each epoch total loss
    epoch_loss += l.item()


    # Store each epoch total loss into empty list
  relu_loss.append(epoch_loss)

  # Accuracy calculation
  acc = 100.0*(n_correct/n_samples)
  relu_acc.append(acc)
  


print(relu_loss)

print(relu_acc)

"""### Accuracy Calculation on test dataset"""

with torch.no_grad():
  n_correct = 0
  n_samples = 0

  for image, label in test_loader:
    image = image.to(device)
    label = label.to(device)

    output = relu_model_vanish(image)

    # value, index
    _, predictions = torch.max(output, 1)
    n_samples += label.shape[0]
    n_correct += (predictions == label).sum().item()


  acc = 100.0 * (n_correct/n_samples)

print("Accuracy of Relu : ",acc)

"""### Plot Graph"""

# Convert list to numpy array
x = np.array(relu_loss)
y = np.array(relu_acc)
z = np.array(relu_epoch)
# chart show

# set chart Size
fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10)  

#plot 1: Loss vs Accuracy
plt.title("Loss vs Accuracy")
plt.xlabel("Loss")
plt.ylabel("Accuracy")
# plt.subplot(1, 4, 1)
plt.plot(x,y)
plt.show()

#plot 2: Accuracy vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Accuracy vs Epoch")
plt.xlabel("Accuracy")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 2)
plt.plot(y,z)
plt.show()

#plot 3: Loss vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Loss vs Epoch")
plt.xlabel("Loss")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 3)
plt.plot(x,z)

plt.show()

"""## We can also remove vanishing gradient problem by using Adam Optimizer instead of SGD

### Create Neural Network with Sigmoid function
"""

class NeuralNetSigmoidVanish(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
    super(NeuralNetSigmoidVanish, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size1)
    self.sigmoid = nn.Sigmoid()
    self.l2 = nn.Linear(hidden_size1, hidden_size2)
    self.l3 = nn.Linear(hidden_size2, hidden_size3)
    self.l4 = nn.Linear(hidden_size3, output_size)


  def forward(self, x):
    x = x.view(-1, 32*32*3)
    out = self.l1(x)
    out = self.sigmoid(out)
    out = self.l2(out)
    out = self.sigmoid(out)
    out = self.l3(out)
    out = self.sigmoid(out)
    out = self.l4(out)

    return out


sigmoid_model_vanish = NeuralNetSigmoidVanish(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3, output_size=output_size)
sigmoid_model_vanish
# print(sigmoid_model.state_dict())

"""### Loss function for sigmoid"""

loss_sigmoid = nn.CrossEntropyLoss()
loss_sigmoid

"""### Optimizer function for sigmoid"""

optimizer_sigmoid = torch.optim.Adam(sigmoid_model_vanish.parameters(), lr=lr)
optimizer_sigmoid

"""### Training Loop for sigmoid function"""

n_step = len(train_loader)
sig_loss = []
sig_acc = []
sig_epoch = []

for epoch in range(num_epochs):
  sig_epoch.append(epoch+1)

  epoch_loss = 0.0
  epoch_acc = 0

  n_correct = 0 # number of correct prediction
  n_samples = 0 # number of total sample

  for i ,(image, label) in enumerate(train_loader):
    
    # image = image.reshape(-1,1024).to(device)
    image = image.to(device)
    label = label.to(device)

    # Forward pass
    label_predict = sigmoid_model_vanish(image)


    # Tracking correct predictions
    _, predict = torch.max(label_predict,1)
    n_samples += label.shape[0]
    n_correct += (predict == label).sum().item()

    # loss
    l = loss_sigmoid(label_predict, label)

    # backward
    optimizer_sigmoid.zero_grad()
    l.backward()

    # Upate weights
    optimizer_sigmoid.step()

    if (i+1)%100 == 0:
      print(f"epoch : {epoch+1}/{num_epochs} | step : {i+1}/{n_step} | loss : {l.item(): .4f}")


    # Store each epoch total loss
    epoch_loss += l.item()


    # Store each epoch total loss into empty list
  sig_loss.append(epoch_loss)

  # Accuracy calculation
  acc = 100.0*(n_correct/n_samples)
  sig_acc.append(acc)
  


print(sig_loss)

print(sig_acc)

"""### Accuracy Calculation on test dataset"""

with torch.no_grad():
  n_correct = 0
  n_samples = 0

  for image, label in test_loader:
    image = image.to(device)
    label = label.to(device)

    output = sigmoid_model(image)

    # value, index
    _, predictions = torch.max(output, 1)
    n_samples += label.shape[0]
    n_correct += (predictions == label).sum().item()


  acc = 100.0 * (n_correct/n_samples)

print("Accuracy of Sigmoid : ",acc)

"""### Plot Graph"""

# Convert list to numpy array
x = np.array(sig_loss)
y = np.array(sig_acc)
z = np.array(sig_epoch)
# chart show

# set chart Size
fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10)  

#plot 1: Loss vs Accuracy
plt.title("Loss vs Accuracy")
plt.xlabel("Loss")
plt.ylabel("Accuracy")
# plt.subplot(1, 4, 1)
plt.plot(x,y)
plt.show()

#plot 2: Accuracy vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Accuracy vs Epoch")
plt.xlabel("Accuracy")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 2)
plt.plot(y,z)
plt.show()

#plot 3: Loss vs Epoch

fig = plt.figure()  
  
fig.set_figheight(5)  
fig.set_figwidth(10) 


plt.title("Loss vs Epoch")
plt.xlabel("Loss")
plt.ylabel("Epoch")
# plt.subplot(1, 4, 3)
plt.plot(x,z)

plt.show()