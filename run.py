#!/home/janginkyu/anaconda3/bin/python

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from simulation import run_simulation

class PhysNet(torch.nn.Module):

    def __init__(self):
        qdof = 1
        ctrldof = 1

        super(PhysNet, self).__init__()
        self.fc1 = torch.nn.Linear(ctrldof + qdof * 2 + qdof, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 20)
        self.fc4 = torch.nn.Linear(20, qdof)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

net = PhysNet()
#net.cuda()
print(net)

input = torch.rand(1, 4)
output = torch.rand(1, 1)
print(net(input).size())

criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adagrad(net.parameters(), lr=1e-2)

data = run_simulation()

for it in range(500):
    loss = torch.tensor(0.0)
    optimizer.zero_grad()
    for t in range(data[:,1].size):
        output = torch.tensor([[data[t, 2]]])
        input = torch.tensor([[math.cos(data[t, 0]), math.sin(data[t, 0]), data[t, 1], data[t, 3]]])
        output_pred = net(input)
        loss += criterion(output_pred, output)
    loss.backward()
    optimizer.step()
    print(str(it) + " : " + str(loss))

out = np.array(data[:,2])
for t in range(data[:,2].size):
    out[t] = net(torch.tensor([[math.cos(data[t, 0]), math.sin(data[t, 0]), data[t, 1], data[t, 3]]]))
plt.plot(data[:,4], data[:,2])
plt.plot(data[:,4], out)
plt.show()