
import torch
import os
import sys
import argparse

from dataset import SiameseTensorDataset
from networks import attacker_siamese1, attacker_siamese2
from losses import ContrastiveLoss
from trainer1 import fit
from utils import loadData, saveData, ConcatDataset2Data, Logger


parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
#/data/gehan/PytorchProjects/GANInference/models/FM_outputs
#/data/gehan/PytorchProjects/GANInference/models/SVHN_outputs/
parser.add_argument('--dataset_addr', dest='dataset_addr', type=str, help='target and shadow model save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/SVHN_outputs/')
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='./outputs/')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='SVHN_compare')
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=10)
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.0001)
parser.add_argument('--n_epochs', dest='n_epochs', type=int, help='number of epochs',
                    default=2000)
parser.add_argument('--margin', dest='margin', type=int, help='margin',
                    default=0.5)

args = parser.parse_args()
dataset_addr = args.dataset_addr
save_addr = args.save_addr
save_name = args.save_name
batch_size = args.batch_size
lr = args.lr
n_epochs = args.n_epochs
margin = args.margin

log_interval = 500
loss_siamese = ContrastiveLoss(margin)
loss_binary = torch.nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Save print log
logger_addr = os.path.join(save_addr, save_name)
if not os.path.exists(logger_addr):
    os.makedirs(logger_addr)
sys.stdout = Logger("%s/posterior_correct_TrainLog.txt" % (logger_addr))

print(args)
# Datasets preparation
trainset_path = os.path.join(dataset_addr, 'real_GANship_Compare_Trainset_unfolded')
if not os.path.exists(os.path.join(dataset_addr, 'real_GANship_Compare_Trainset_unfolded')):
    trainset_path = os.path.join(dataset_addr, 'real_GANship_Compare_noreal_Trainset_AllStrctr')
    trainset_folded = loadData(trainset_path)
    #trainset = ConcatDataset2Data(trainset_folded)
    trainset_data = ConcatDataset2Data(trainset_folded)
    trainset = torch.utils.data.TensorDataset(*trainset_data)
    #saveData(dataset_addr, 'real_GANship_Compare_Trainset_unfolded', trainset)
else:
    trainset = loadData(trainset_path)

testset_path = os.path.join(dataset_addr, 'real_GANship_Compare_Testset_unfolded')
if not os.path.exists(os.path.join(dataset_addr, 'real_GANship_Compare_Testset_unfolded')):
    testset_path = os.path.join(dataset_addr, 'real_GANship_Compare_Testset_AllStrctr')
    testset_folded = loadData(testset_path)
    #testset = ConcatDataset2Data(testset_folded)
    testset_data = ConcatDataset2Data(testset_folded)
    testset = torch.utils.data.TensorDataset(*testset_data)
    #saveData(dataset_addr, 'real_GANship_Compare_Testset_unfolded', testset)
else:
    testset = loadData(testset_path)

content_list = ['posterior_sorted', 'posterior_label', 'posterior_correct', 'GANship']
information_list = ['posterior_correct']#'posterior_sorted', 'posterior_label', 'posterior_correct']
target_name = 'GANship'

target_index = content_list.index(target_name)
for i in information_list:
    information_index = content_list.index(i)

    # Step 1
    siamese_train_dataset = SiameseTensorDataset(trainset, information_index, target_index)
    siamese_test_dataset = SiameseTensorDataset(testset, information_index, target_index)
    siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True)#, **kwargs)
    siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False)#, **kwargs)

    # Step 2
    if i == 'posterior_sorted':
        # Building model
        model = attacker_siamese1(nch_info=1000).to(device)
    elif i == 'posterior_label':
        # Building model
        model = attacker_siamese1(nch_info=2000).to(device)
    elif i == 'posterior_correct':
        # Building model
        model = attacker_siamese1(nch_info=1100).to(device)

    # Step 3
    fit(siamese_train_loader, siamese_test_loader, model, loss_siamese, loss_binary, lr, n_epochs, device, log_interval)