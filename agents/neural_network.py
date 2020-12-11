import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Function approximator that maps states to actions 
Simple neural network
"""
class Net(nn.Module):

    def __init__(self, states, actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32, actions)

    def forward(self, x):
        episodes = x.size()[0]
        horizon_length = x.size()[1]
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        out = torch.nn.functional.softmax(out, dim=1)
        out = out.view(episodes, horizon_length, -1)
        return out

    def num_flat_features(self, x):
        size = x.size()[2:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features







