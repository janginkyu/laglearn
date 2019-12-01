#!/home/janginkyu/anaconda3/bin/python

import torch

class PhysNet(torch.nn.Module):

    def __init__(self):
        qdof = 1
        ctrldof = 1

        super(PhysNet, self).__init__()
        self.fc1 = torch.nn.Linear(ctrldof + qdof * 2 + qdof, 60)
        self.fc2 = torch.nn.Linear(60, 60)
        self.fc3 = torch.nn.Linear(60, 60)
        self.fc4 = torch.nn.Linear(60, qdof)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

net = PhysNet()
print(net)

input = torch.rand(1, 8)
print(net(input))
