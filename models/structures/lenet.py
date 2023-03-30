'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, c_dim=8):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc_plus = nn.Linear(16*13*13, 16*5*5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, c_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_plus(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def get_fc_outputs(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = fc_plus = F.relu(self.fc_plus(out))#400
        out = fc1 = F.relu(self.fc1(out))#120
        out = fc2 = F.relu(self.fc2(out))#84
        return fc_plus, fc1, fc2


class SmplCNN(nn.Module):
    def __init__(self, c_dim=8):
        super(SmplCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc_plus = nn.Linear(16*13*13, 16*5*5)
        self.fc1   = nn.Linear(16*5*5, 64)
        self.fc2   = nn.Linear(64, c_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_plus(out))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class attacker_mia(nn.Module):
    def __init__(self, nch_info, nch_output=1):

        super(attacker_mia, self).__init__()
        self.nch_info = nch_info
        self.nch_output = nch_output

        self.fc1 = nn.Linear(self.nch_info, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, max(16, 2*self.nch_output))
        self.fc4 = nn.Linear(max(16, 2*self.nch_output), self.nch_output)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x


class attacker_mia1(nn.Module):
    def __init__(self, nch_info, nch_output=1):

        super(attacker_mia1, self).__init__()
        self.nch_info = nch_info
        self.nch_output = nch_output

        self.fc1 = nn.Linear(self.nch_info, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, max(16, 2*self.nch_output))
        self.fc4 = nn.Linear(max(16, 2*self.nch_output), self.nch_output)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.Softmax(self.fc4(x))

        return x

class attacker_mia2(nn.Module):
    def __init__(self, nch_info, nch_output=1):

        super(attacker_mia2, self).__init__()
        self.nch_info = nch_info
        self.nch_output = nch_output

        self.fc1 = nn.Linear(self.nch_info, 64)
        self.fc3 = nn.Linear(64, max(16, 2*self.nch_output))
        self.fc4 = nn.Linear(max(16, 2*self.nch_output), self.nch_output)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.Softmax(self.fc4(x))

        return x

class attacker_real(nn.Module):
    def __init__(self, nch_info, nch_output=40):

        super(attacker_real, self).__init__()
        self.nch_info = nch_info
        self.nch_output = nch_output

        self.fc1 = nn.Linear(self.nch_info, min(64, 2*self.nch_output))
        self.fc2 = nn.Linear(min(64, 2*self.nch_output), self.nch_output)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def embedding_(self, x):
        x = nn.functional.relu(self.fc1(x))

        return x
