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
from models import structures
import models.GAN_structures

# ==============================================================================
# =                                  Settings                                  =
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--batch_size', dest='batch_size', type=int, help='size of batch',
                    default=10)
parser.add_argument('--c_dim', dest='c_dim', type=int, help='number of classes',
                    default=10)
parser.add_argument('--model_addr', dest='model_addr', type=str, help='model address',
                    default='/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/VGG11/')
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/')
args = parser.parse_args()

batch_size = args.batch_size
c_dim = args.c_dim
model_addr = args.model_addr
save_addr = args.save_addr

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
        #print('Open file', filename, '......ok')
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
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_path)
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


# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

# Check save address
if not os.path.exists(save_addr):
    os.makedirs(save_addr)
sys.stdout = Logger("%s/DatasetGenerationLog_forAttacker.txt" % (save_addr))

model_list = ['TargetModel',
              'ShadowModel_same', 'ShadowModel_other', 'ShadowModel_real',
              'ShadowModel_ALLsame', 'ShadowModel_ALLother', 'ShadowModel_SVHN']

for model_name in model_list:
    # Initiate the Testset of current GAN-strctr
    posterior_sorted = torch.tensor([]).to(device)
    posterior_label = torch.tensor([]).to(device)  # un-sorted
    posterior_correct = torch.tensor([]).to(device)  # sorted
    Membership = torch.tensor([]).to(device)

    if model_name.startswith('Target'):
        Din = loadData(save_addr+'Din_t')
        Dout = loadData(save_addr+'Dout_t')
        dataset_name = 'Testset'

    elif model_name == 'ShadowModel_same':
        Din_1 = loadData(save_addr+'Din_real1')
        Din_2 = loadData(save_addr+'Din_same')
        Din = torch.utils.data.ConcatDataset([Din_1, Din_2])
        Dout_1 = loadData(save_addr+'Dout_real')
        Dout_2 = loadData(save_addr+'Dout_same')
        Dout = torch.utils.data.ConcatDataset([Dout_1, Dout_2])
        dataset_name = 'Trainset_same'

    elif model_name == 'ShadowModel_other':
        Din_1 = loadData(save_addr + 'Din_real1')
        Din_2 = loadData(save_addr + 'Din_other')
        Din = torch.utils.data.ConcatDataset([Din_1, Din_2])
        Dout_1 = loadData(save_addr + 'Dout_real')
        Dout_2 = loadData(save_addr + 'Dout_other')
        Dout = torch.utils.data.ConcatDataset([Dout_1, Dout_2])
        dataset_name = 'Trainset_other'

    elif model_name == 'ShadowModel_real':
        Din_1 = loadData(save_addr + 'Din_real1')
        Din_2 = loadData(save_addr + 'Din_real2')
        Din = torch.utils.data.ConcatDataset([Din_1, Din_2])
        Dout = loadData(save_addr + 'Dout_ALLreal')
        dataset_name = 'Trainset_real'

    elif model_name == 'ShadowModel_ALLsame':
        Din = loadData(save_addr + 'Din_ALLsame')
        Dout = loadData(save_addr + 'Dout_ALLsame')
        dataset_name = 'Trainset_ALLsame'

    elif model_name == 'ShadowModel_ALLother':
        Din = loadData(save_addr + 'Din_ALLother')
        Dout = loadData(save_addr + 'Dout_ALLother')
        dataset_name = 'Trainset_ALLother'

    elif model_name == 'ShadowModel_SVHN':
        Din = loadData(save_addr + 'Din_svhn')
        Dout = loadData(save_addr + 'Dout_svhn')
        dataset_name = 'Trainset_SVHN'

    Din_loader = torch.utils.data.DataLoader(Din, batch_size=batch_size, shuffle=True)
    Dout_loader = torch.utils.data.DataLoader(Dout, batch_size=batch_size, shuffle=True)

    print("\n>> ", model_name)

    model_dir = os.path.join(model_addr, model_name)
    assert os.path.isdir(model_dir), 'No such trained model!'

    # Building model
    net = structures.vgg.VGG('VGG11', c_dim=c_dim).to(device)

    net.eval()
    # Load checkpoint
    model_ckpt_dir = os.path.join(model_dir, 'checkpoints')
    ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
    if ckpt_best:
        best_acc = ckpt_best['acc']
        net.load_state_dict(ckpt_best['net'])
    else:
        ckpt = load_checkpoint(model_ckpt_dir)
        if ckpt:
            best_acc = ckpt['acc']
            net.load_state_dict(ckpt['net'])
        else:
            print(' [Wrong] No checkpoint!')
            continue
    print('Model loaded, acc = %.2f' % (best_acc))


    with torch.no_grad():
        for prob_loader in [Din_loader, Dout_loader]:

            information0 = torch.tensor([]).to(device)
            information1 = torch.tensor([]).to(device)
            information2 = torch.tensor([]).to(device)

            for batch_idx, (inputs, targets) in enumerate(prob_loader):

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)

                # one-hot true labels (generated by GAN)
                labels = torch.zeros(outputs.shape).to(device)
                for i in range(len(targets)):
                    labels[i][targets[i]] = 1

                # correctness of classification
                _, predicted = outputs.max(1)
                predicted = predicted.to(device)
                corrects = predicted.eq(targets).float().unsqueeze(-1)

                outputs_sorted = outputs.sort(dim=1)[0]
                # concatenate UN-SORTED posterior and true label
                post_l = torch.cat([outputs, labels], 1)
                # concatenate SORTED posterior and correct bit
                post_c = torch.cat([outputs_sorted, corrects], 1)

                information0 = torch.cat((information0, outputs_sorted), 0)
                information1 = torch.cat((information1, post_l), 0)
                information2 = torch.cat((information2, post_c), 0)

            # update dataset
            posterior_sorted = torch.cat((posterior_sorted, information0), 0)
            posterior_label = torch.cat((posterior_label, information1), 0)
            posterior_correct = torch.cat((posterior_correct, information2), 0)
            if prob_loader == Din_loader:
                Membership = torch.cat((Membership, torch.ones((information0.shape[0], 1)).to(device)), 0)
            else:
                Membership = torch.cat((Membership, torch.zeros((information0.shape[0], 1)).to(device)), 0)


    dataset = torch.utils.data.TensorDataset(posterior_sorted, posterior_label, posterior_correct, Membership)
    saveData(save_addr, dataset_name, dataset)
    print('%s saved, length %d.' % (dataset_name, len(dataset)))
    del dataset