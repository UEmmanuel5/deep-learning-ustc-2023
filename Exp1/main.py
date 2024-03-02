# Import necessary libraries
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

# Define a neural network class
class Net(nn.Module):
    def __init__(self, activation=torch.sigmoid, layers=[1, 20, 1]):
        super(Net, self).__init__()
        self.activation = activation
        self.num_layers = len(layers) - 1
        self.fctions = nn.ModuleList()
        for i in range(self.num_layers):
            self.fctions.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.activation(self.fctions[i](x))
        x = self.fctions[-1](x)
        return x

# Define a lambda function for learning rate scheduling
def lr_lambda(current_step):
    warmup_steps = 1000
    if current_step < warmup_steps:
        return float(current_step / warmup_steps)
    else:
        return 1.0

# Define a training function
def Training(traindata, valdata, layers, activation, lr):
    # Create the neural network, optimizer, and loss function
    net = Net(activation=activation, layers=layers)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    loss_func = nn.MSELoss()

    # Create a learning rate scheduler that uses the lambda function
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Training loop
    x_train, y_train = traindata
    x_val, y_val = valdata
    loss_train = []
    loss_val = []
    for i in range(10000):
        optimizer.zero_grad()
        y_pred = net(x_train)
        loss_t = loss_func(y_pred, y_train)
        loss_t.backward()
        optimizer.step()
        scheduler.step()
        loss_train.append(loss_t.item())
        y_vpred = net(x_val)
        loss_v = loss_func(y_vpred, y_val)
        loss_val.append(loss_v.item())

        t = i + 1
        if t % 1000 == 0:
            print(f"\t\t Epoch: {t}, Loss: {loss_v.item()}")
        if loss_v.item() < 1e-6:
            print(f"\t\t Epoch: {t}, Loss: {loss_v.item()}")
            break

    loss_curve = [loss_train, loss_val]
    return net, loss_curve

# Define a function to create data with a sine function
def func(x):
    return np.sin(x)

# Generate data
a = 0
b = 2 * np.pi
n = 3000
t = int(0.8 * n)
v = int(0.9 * n)

x = np.random.uniform(a, b, size=(n, 1))
y = func(x)

x_train = torch.from_numpy(x[0: t]).float()
x_val = torch.from_numpy(x[t: v]).float()
x_test = torch.from_numpy(x[v: n]).float()

y_train = torch.from_numpy(y[0: t]).float()
y_val = torch.from_numpy(y[t: v]).float()
y_test = torch.from_numpy(y[v: n]).float()

# Specify hyperparameters
activation = torch.relu
depth = 5
width = 17
lr = 0.1

# Train the neural network
print('Validation set MSE:')
layers = [1]
for i in range(depth):
    layers.append(width)
layers.append(1)
net, loss_curve = Training((x_train, y_train), (x_val, y_val), layers, activation, lr)
t = len(loss_curve[0])

# Plot training and validation loss curves
plt.figure(figsize=(8, 4), dpi=300)
plt.title('Loss Curves')
plt.plot(range(t), loss_curve[0])
plt.plot(range(t), loss_curve[1])
plt.legend(['Train', 'Validation'])
plt.show()

# Test the trained model and calculate test MSE
y_tpred = net(x_test)
mse = nn.MSELoss()(y_tpred, y_test)
print(f"Test set MSE: {mse.item()}")

# Plot the ground truth and predicted values
plt.figure(figsize=(8, 4), dpi=300)
plt.scatter(x_test, y_test, label='Truth', s=1)
plt.scatter(x_test, y_tpred.detach().numpy(), label='Prediction', s=1)
plt.legend()
plt.show()
