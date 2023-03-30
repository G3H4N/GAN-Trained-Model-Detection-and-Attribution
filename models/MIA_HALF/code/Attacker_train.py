import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from numpy import *
import os
import argparse
import pickle
import sys
import PIL.Image as Image
import shutil
sys.path.append("../..")
from models import structures
import models.GAN_structures

from sklearn.metrics import roc_auc_score

# ==============================================================================
# =                                  Settings                                  =
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
# Train setting
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.000015)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=200)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=400)
parser.add_argument('--data_addr', dest='data_addr', type=str, help='dataset address',
                    default='/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/')
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/attackers_all/')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='same')
args = parser.parse_args()

# Train settings
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
data_addr = args.data_addr
# Save settings
save_addr = args.save_addr
save_name = args.save_name

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# =                                    Utils                                   =
# ==============================================================================

class Logger(object):
    def __init__(self, fileN="Default.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def loadData(filename):
    with open(filename, 'rb') as file:
        print('Open file', filename, '......ok')
        obj = pickle.load(file, encoding='latin1')
        file.close()
        return obj

def saveData(dir, filename, obj):
    path = os.path.join(dir, filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(path):
        # 注意字符串中含有空格，所以有r' '；touch指令创建新的空文件
        os.system(r'touch {}'.format(path))

    with open(path, 'wb') as file:
        # pickle.dump(obj,file[,protocol])，将obj对象序列化存入已经打开的file中，file必须以二进制可写模式打开（“wb”），可选参数protocol表示高职pickler使用的协议，支持的协议有0，1，2，3，默认的协议是添加在python3中的协议3。
        pickle.dump(obj, file)
        file.close()
    print('Save data', path, '......ok')

    return

def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    #print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_path)
    return ckpt

def save_checkpoint(obj, save_path, is_best=False, max_keep=None):
    # save checkpoint
    torch.save(obj, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    save_name = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_name + '\n'] + ckpt_list
    else:
        ckpt_list = [save_name + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))

def Denormalize(data):
    return (data + 1) / 2

def test(net, i_index, t_index, test_loader):
    # global best_acc
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data[i_index].to(device)
            targets = data[t_index].to(device)
            targets = targets.squeeze(-1)
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total

    print('>> Test\nAccuracy: %.5f%%' % (acc))

    return acc

def test_binary(net, i_index, t_index, test_loader):
    # global best_acc
    net.eval()

    test_loss = 0
    outputs_test = []
    members_test = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data[i_index].to(device)
            targets = data[t_index].to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            outputs_test.extend(outputs)
            members_test.extend(targets)

    auc = roc_auc_score(torch.BoolTensor(members_test), outputs_test)

    print('>> Test\nAUC: %.5f' % (auc))

    return auc


# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

# Save print log
logger_addr = os.path.join(save_addr, save_name)
if not os.path.exists(logger_addr):
    os.makedirs(logger_addr)
sys.stdout = Logger("%s/TrainLog.txt" % (logger_addr))

# Datasets preparation
trainset_path = os.path.join(data_addr, 'Trainset_'+save_name)
trainset = loadData(trainset_path)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

testset_path = os.path.join(data_addr, 'Testset')
testset = loadData(testset_path)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

content_list = ['posterior_sorted', 'posterior_label', 'posterior_correct', 'Membership']
information_list = ['posterior_sorted', 'posterior_label', 'posterior_correct']
t = 'Membership'

# ====== Train and Test a model for each of information ====== #
for i in information_list:

    information_index = content_list.index(i)

    print('\n--* %s --> %s *--' % (i, t))
    target_index = content_list.index(t)

    # Directory to save model
    model_dir = os.path.join(os.path.join(save_addr, save_name), i)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if i == 'posterior_sorted':
        # Building model
        net = structures.lenet.attacker_mia(nch_info=10, nch_output=1).to(device)
    elif i == 'posterior_label':
        # Building model
        net = structures.lenet.attacker_mia(nch_info=20, nch_output=1).to(device)
    elif i == 'posterior_correct':
        # Building model
        net = structures.lenet.attacker_mia(nch_info=11, nch_output=1).to(device)

    criterion = nn.BCELoss()
    net_optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)

    # Load checkpoint
    model_ckpt_dir = os.path.join(model_dir, 'checkpoints')

    try:
        ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
        start_epoch = ckpt_best['epoch']
        best_acc = ckpt_best['acc']
        net.load_state_dict(ckpt_best['net'])
        net_optimizer.load_state_dict(ckpt_best['optimizer'])
    except:
        try:
            ckpt = load_checkpoint(model_ckpt_dir, load_best=False)
            start_epoch = ckpt['epoch']
            best_acc = ckpt['acc']
            net.load_state_dict(ckpt['net'])
            net_optimizer.load_state_dict(ckpt['optimizer'])
        except:
            print(' [*] No checkpoint!\nTrain from beginning.')
            start_epoch = 0
            best_acc = 0

    for ep in range(start_epoch, epochs):

        # TRAIN
        print('\n>> TRAIN Epoch: %d' % ep)
        net.train()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(train_loader):
            inputs = data[information_index].to(device)
            targets = data[target_index].to(device)
            #targets = targets.squeeze(-1)

            net_optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            net_optimizer.step()

            train_loss += loss.item()
            #_, predicted = outputs.max(1)
            #total += targets.size(0)
            #correct += predicted.eq(targets).sum().item()

            if (batch_idx % 20 == 0) or (batch_idx == (len(train_loader)-1)):
                print('batch_idx %d, len(trainloader) %d, Loss: %.6f'
                         % (batch_idx, len(train_loader), train_loss/(batch_idx+1)))
            #break

        # after all batch-iterations in current epoch

        # test during training
        acc = test_binary(net, information_index, target_index, test_loader)

        if acc > best_acc:
            print('Saving best...')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': ep + 1,
                'optimizer': net_optimizer.state_dict()
            }
            if not os.path.isdir(model_ckpt_dir):
                os.mkdir(model_ckpt_dir)
            save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep), max_keep=2,
                            is_best=True)
            best_acc = acc

        # Save checkpoint every 10 epochs
        elif ep % 5 == 0:
            print('Saving...')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': ep + 1,
                'optimizer': net_optimizer.state_dict()
            }
            if not os.path.isdir(model_ckpt_dir):
                os.mkdir(model_ckpt_dir)
            save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep), max_keep=2,
                            is_best=False)
        #break

    # after all epochs
    acc = test_binary(net, information_index, target_index, test_loader)
    if start_epoch != epochs:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': ep + 1,
            'optimizer': net_optimizer.state_dict()
        }
        if not os.path.isdir(model_ckpt_dir):
            os.mkdir(model_ckpt_dir)
        save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep), max_keep=2, is_best=False)
    else:
        acc = test_binary(net, information_index, target_index, test_loader)
    print('Finished!')# best_acc is %.5f%%\n\n' % best_acc)
    # release current network
    del net, net_optimizer
