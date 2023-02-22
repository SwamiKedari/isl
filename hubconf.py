#!/usr/bin/env python
# coding: utf-8

# In[59]:


get_ipython().system('python3 -c "import torch; print(torch.__version__)"')


# In[60]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

#Convert a PIL Image to tensor
from torchvision.transforms import ToTensor


# In[61]:


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# In[45]:





# In[62]:


y=list(test_data)
for i,lis in enumerate(y):
  z=y[i]
  # print(z)
  x=list(z)
  # print()

  

  # print(x[1])
  if x[1]==9:
    x[1]=3
    # print("yeyy")
  elif x[1]==0 or x[1]==3 or x[1]==4 or x[1]==6:
    x[1]=0
  elif x[1]==1 or x[1]==2:
    x[1]=1
  elif x[1]==5 or x[1]==7 or x[1]==8:
    x[1]=2
  t=tuple(x)
  y[i]=t
  
test_data=tuple(y)


# In[63]:


y=list(training_data)
for i,lis in enumerate(y):
  z=y[i]
  # print(z)
  x=list(z)
  # print()

  

  # print(x[1])
  if x[1]==9:
    x[1]=3
    # print("yeyy")
  elif x[1]==0 or x[1]==3 or x[1]==4 or x[1]==6:
    x[1]=0
  elif x[1]==1 or x[1]==2:
    x[1]=1
  elif x[1]==5 or x[1]==7 or x[1]==8:
    x[1]=2
  t=tuple(x)
  y[i]=t
  
training_data=tuple(y)


# In[64]:


batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    #batch, channel, height, width (NCHW)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# In[65]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# In[66]:


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        '''
        input layer neurons : output layer neurons
        784 : 512
        512 : 512
        512 : 10
        '''
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# In[67]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# In[68]:


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        #set the gradients to zero before backpropragation as PyTorch accumulates the gradients on subsequent backward passes
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# In[69]:


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[70]:


epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# In[71]:


classes = [
    "Upper",
    "Lower",
    "Feet",
    "Bag"
]

model.eval()
x, y = test_data[1][0], test_data[1][1]
with torch.no_grad():
    pred = model(x.to(device))
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


# In[ ]:





