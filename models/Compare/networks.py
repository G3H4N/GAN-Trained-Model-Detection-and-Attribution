import torch
import torch.nn as nn
import torch.nn.functional as F


class attacker_siamese1(nn.Module):
    def __init__(self, nch_info):
        super(attacker_siamese1, self).__init__()
        self.nch_info = nch_info

        self.fc1 = nn.Sequential(
            nn.Linear(self.nch_info, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(),
        )

    def forward(self, x1, x2):

        x1 = self.fc1(x1)
        embedding1 = self.fc2(x1)
        x2 = self.fc1(x2)
        embedding2 = self.fc2(x2)

        distances = (embedding2 - embedding1).pow(2)#.sum(1)
        output = self.fc3(distances)
        return [embedding1, embedding2], output



class attacker_siamese2(nn.Module):
    def __init__(self, nch_info):

        super(attacker_siamese2, self).__init__()
        self.nch_info = nch_info
        self.fc1 = nn.Sequential(
            nn.Linear(self.nch_info, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        embedding1 = self.fc2(x1)
        x2 = self.fc1(x2)
        embedding2 = self.fc2(x2)

        distances = (embedding2 - embedding1).pow(2)#.sum(1)
        output = self.fc3(distances)
        return [embedding1, embedding2], output


