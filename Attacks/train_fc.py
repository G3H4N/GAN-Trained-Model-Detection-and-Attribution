import numpy as np
import torch
import torch.nn as nn
from model import *
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import argparse
import pickle
from sklearn.metrics import roc_auc_score
#import pytorch_ssim
import matplotlib.pyplot as plt



def loadData(filename):
    with open(filename, 'rb') as file:
        print('Open file', filename, '......ok')
        obj = pickle.load(file, encoding='latin1')
        file.close()
        return obj


# Plot auc
def plot_auc(shadow_auc, target_auc, epoch, num_epoch, save, save_dir, show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epoch)
    ax.set_ylim(0, max(np.max(shadow_auc), np.max(target_auc))*1.1)
    plt.xlabel('Epoch {0}'.format(epoch + 1))
    plt.ylabel('AUC')
    plt.plot(shadow_auc, label='AUC tested on shadow medel')
    plt.plot(target_auc, label='AUC tested on target model')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(save_dir + '/AUC/'):
            os.mkdir(save_dir + '/AUC/')
        save_fn = save_dir + '/AUC/epoch_{:03d}'.format(epoch) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epoch', type=int, default=100, help='num of epoch')
    parser.add_argument('--lr_G', type=float, default=0.0001, help='learning rate of Generator')
    parser.add_argument('--lr_D', type=float,default=0.0001, help='learning rate of Discriminator')

    parser.add_argument('--num_freq_save', type=int, default=10, help='frequency of save model')
    parser.add_argument('--num_freq_disp', type=int, default=1, help='frequency of sample image')

    parser.add_argument('--save_dir', type=str, default='./GAN_PosteriorLabel_fc', help='directory for saving model')
    parser.add_argument('--result_dir', type=str, default='./GAN_PosteriorLabel_fc', help='directory for saving result')
    parser.add_argument('--train_dir', type=str, default='/data/gehan/PytorchProjects/GANInference/CelebA/cDCGANs_jpg/gen/forAttack_GAN_realONLY/shadow_18000_36000_Train', help='directory for train dataset')
    parser.add_argument('--test_dir', type=str, default='/data/gehan/PytorchProjects/GANInference/CelebA/cDCGANs_jpg/gen/forAttack_GAN_realONLY/shadow_2000_2000_Test', help='directory for test dataset')
    parser.add_argument('--testTarget_dir', type=str, default='/data/gehan/PytorchProjects/GANInference/CelebA/cDCGANs_jpg/gen/forAttack_GAN_realONLY/target_2000_2000_Test', help='directory for test dataset')
    parser.add_argument('--load_epoch', type=int, default=-1, help='number of loading model')

    parser.add_argument('--nch_image', type=int, default=3, help='the number of channels for input')
    parser.add_argument('--nch_info', type=int, default=16, help='the number of channels for outputs')
    parser.add_argument('--n_class', type=int, default=8, help='the number of class')

    args = parser.parse_args()

    batch_size = args.batch_size
    image_size = args.image_size
    num_epoch = args.num_epoch
    lr_G = args.lr_G
    lr_D = args.lr_D
    num_freq_save = args.num_freq_save
    num_freq_disp = args.num_freq_disp
    save_dir = args.save_dir
    result_dir = args.result_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    testTarget_dir = args.testTarget_dir
    model_dir = args.save_dir
    load_epoch = args.load_epoch
    nch_image = args.nch_image
    nch_info = args.nch_info
    n_class = args.n_class

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = DCGAN_R_simple_fc(nch_image=nch_image, nch_info=nch_info).to(device)

    init_weights(netG)

    m = nn.Sigmoid()
    #bn = nn.BatchNorm1d()

    loss_function = nn.BCELoss().to(device)

    optimG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))
    '''
        train = dset.ImageFolder(root=train_dir,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.CenterCrop(image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                 ]))
    '''
    train = loadData(train_dir)
    loader_train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)

    testShadow = loadData(test_dir)
    loader_testShadow = torch.utils.data.DataLoader(testShadow, batch_size=batch_size, shuffle=True, num_workers=0)

    testTarget = loadData(testTarget_dir)
    loader_testTarget = torch.utils.data.DataLoader(testTarget, batch_size=batch_size, shuffle=True, num_workers=0)

    num_train = len(train)
    num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

    def model_save(dir_chck, netG, optimG, epoch, auc_target):

        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(),
                    'optimG': optimG.state_dict()},
                   '%s/model_epoch%04d_%04d.pth' % (dir_chck, epoch, auc_target*10000))

        print('Saved %dth network' % epoch)

    def model_load( dir_chck, netG, optimG=[], epoch=-1):
        if epoch == -1:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        netG.load_state_dict(dict_net['netG'])
        if optimG != []:
            optimG.load_state_dict(dict_net['optimG'])

        return netG, optimG, epoch

    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]

        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requries_grad = requires_grad


    def Denormalize(data):
        return (data + 1) / 2

    start_epoch = 0
    if load_epoch != -1:
        netG, _, start_epoch = model_load(model_dir, netG, [], load_epoch)

    shadow_auc = []
    target_auc = []
    best_auc = 0

    for epoch in range(start_epoch+1, num_epoch + 1):
        netG.train()

        dist_train = []
        loss_train = []

        for batch, (images, info, labels) in enumerate(loader_train):
            netG.train()
            info_input = torch.zeros(info.shape[0], info.shape[1], images.shape[-2], images.shape[-1]).to(device)
            for i in range(info.shape[0]):
                for j in range(info.shape[1]):
                    info_input[i][j] = info[i][j].data

            set_requires_grad(netG, True)
            optimG.zero_grad()

            output = netG(info).view(info.shape[0])
            labels = labels.type(torch.FloatTensor).to(device)
            output = m(output)
            loss = loss_function(output, labels)

            loss.backward()
            optimG.step()

            #dist_train += [distance.mean().item()]
            loss_train += [loss.item()]

            if (batch % 100) ==0:
                print('TRAIN: EPOCH %d/%d: BATCH %04d/%04d: '
                      'average loss: %.4f' %
                      (epoch, num_epoch, batch, num_batch_train, np.mean(loss_train)))

        with torch.no_grad():
            netG.eval()

            dist_test = []
            loss_test = []
            output_test = []
            members_test = []
            for batch, (images, info, labels) in enumerate(loader_testShadow):
                info_input = torch.zeros(info.shape[0], info.shape[1], images.shape[-2], images.shape[-1]).to(
                    device)
                for i in range(info.shape[0]):
                    for j in range(info.shape[1]):
                        info_input[i][j] = info[i][j].data

                output = netG(info).view(info.shape[0])
                labels = labels.type(torch.FloatTensor).to(device)
                output = m(output)
                loss = loss_function(output, labels)
                loss_test += [loss.item()]

                output_test.extend(output)
                members_test.extend(labels)

            auc_shadow = roc_auc_score(torch.BoolTensor(members_test), output_test)
            shadow_auc.append(auc_shadow)
            print('TEST ON SHADOW: EPOCH %d/%d: average loss: %.4f, AUC: %.4f' %
                  (epoch, num_epoch, np.mean(loss_test), auc_shadow))

            dist_test = []
            loss_test = []
            output_test = []
            members_test = []

            for batch, (images, info, labels) in enumerate(loader_testTarget):
                info_input = torch.zeros(info.shape[0], info.shape[1], images.shape[-2], images.shape[-1]).to(
                    device)
                for i in range(info.shape[0]):
                    for j in range(info.shape[1]):
                        info_input[i][j] = info[i][j].data

                output = netG(info).view(info.shape[0])
                labels = labels.type(torch.FloatTensor).to(device)
                output = m(output)
                loss = loss_function(output, labels)
                loss_test += [loss.item()]


                output_test.extend(output)
                members_test.extend(labels)

            auc_target = roc_auc_score(torch.BoolTensor(members_test), output_test)
            target_auc.append(auc_target)
            print('TEST ON TARGET: EPOCH %d/%d: average loss: %.4f, AUC: %.4f' %
                  (epoch, num_epoch, np.mean(loss_test), auc_target))

            if auc_target > best_auc:
                print('Saving..')
                model_save(save_dir, netG, optimG, epoch, auc_target)
                best_auc = auc_target

        if (epoch % num_freq_save) == 0:
            plot_auc(shadow_auc, target_auc, epoch, num_epoch, save=True, save_dir=save_dir, show=False)
        '''
        if (epoch % num_freq_disp) == 0:
            outputs = Denormalize(outputs)
            for i in range(outputs.shape[0]):
                if i > 5:
                    break
                im = outputs[i].cpu().clone()
                im = im.squeeze(0)
                torchvision.utils.save_image(im, '%s/image_E%03d_%04d_l%d.png' % (result_dir, epoch, i, labels[i]))
        '''


if __name__=="__main__":
    main()

