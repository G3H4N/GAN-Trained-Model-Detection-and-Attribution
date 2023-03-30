'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import resnet
import lenet
import pickle

import sys


class Logger(object):
    def __init__(self, fileN="Default.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

train_dir = '/data/gehan/PytorchProjects/MIamgGAN/MIG/CelebA/data/GenbyTargetGAN'
test_dir = '/data/gehan/PytorchProjects/MIamgGAN/MIG/CelebA/data/Split_align_Attr_37_32_22_sets/target_test'
save_dir = './byTargetGAN_8/'
save_name = 'LeNet_'

def loadData(filename):
    with open(filename, 'rb') as file:
        print('Open file', filename, '......ok')
        obj = pickle.load(file, encoding='latin1')
        file.close()
        return obj

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
sys.stdout = Logger("%s/TrainLog.txt" % save_dir)

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=float, help='batch size')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--resume', default=-1, action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testset = loadData(test_dir)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

classes = ('000', '001', '010', '011',
           '100', '101', '110', '111')

# Model
print('==> Building model..')



net = lenet.LeNet()#resnet.ResNet18()



net = net.to(device)
if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume != -1:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no model for resumption!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 ==0:
            print('batch_idx %d, len(trainloader) %d, Loss: %.6f | Acc: %.3f%% (%d/%d)'
                     % (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    global best_acc
    net.eval()

    test_loss_train = 0
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx * args.batch_size > 5000:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss_train += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

    acc_train = 100. * correct_train / total_train


    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total

    print('>> Test: current accuracy on train set %.3f%%, current accuracy on real data %.3f%%, best accuracy %.3f%%'
          % (acc_train, acc, best_acc))

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save(state, '%s/%s_E%04d_%.2f_%.2f.pth' % (save_dir, save_name, epoch, acc_train, acc))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
