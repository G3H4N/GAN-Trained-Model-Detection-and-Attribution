import torch
from torch.nn import init
from layer import *


class DCGAN_R_simple_fc(nn.Module):
    def __init__(self, nch_image, nch_info, nch_ker=128):

        super(DCGAN_R_simple_fc, self).__init__()

        self.nch_image = nch_image
        self.nch_ker = nch_ker
        self.nch_info = nch_info

        self.fc1 = nn.Linear(self.nch_info, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class DCGAN_R_simple(nn.Module):
    def __init__(self, nch_image, nch_info, nch_ker=128):

        super(DCGAN_R_simple, self).__init__()

        self.nch_image = nch_image
        self.nch_ker = nch_ker
        self.nch_info = nch_info

        self.CBR1_2 = CBR2D(1 * nch_info, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR1 = CBR2D(1 * nch_ker, 2 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR2 = CBR2D(2 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR3 = CBR2D(4 * nch_ker, 8 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR4 = CBR2D(8 * nch_ker, 16 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR5 = CBR2D(16 * nch_ker, 1, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, info):
        x = self.CBR1_2(info) # 128*32*32
        x = self.CBR1(x) # 256*16*16
        x = self.CBR2(x) # 512*8*8
        x = self.CBR3(x) # 1024*4*4
        x = self.CBR4(x) # 2048*2*2
        x = self.CBR5(x) # 1*1*1

        return x


class DCGAN_R_xyp(nn.Module):
    def __init__(self, nch_image, nch_info, nch_ker=128):

        super(DCGAN_R_xyp, self).__init__()

        self.nch_image = nch_image
        self.nch_ker = nch_ker
        self.nch_info = nch_info

        # DCGAN_D
        self.CBR1_1 = CBR2D(1 * nch_image, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR1_2 = CBR2D(1 * nch_info, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR2 = CBR2D(2 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR3 = CBR2D(4 * nch_ker, 8 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR4 = CBR2D(8 * nch_ker, 16 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR5 = CBR2D(16 * nch_ker, 1,           kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)

        # DCGAN_G
        self.DECBR1_1 = DECBR2D(1 * self.nch_image, 8 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False)
        self.DECBR1_2 = DECBR2D(1 * self.nch_info, 4 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False)
        self.DECBR2 = DECBR2D(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR3 = DECBR2D(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR4 = DECBR2D(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR5 = DECBR2D(1 * self.nch_ker, 1 * self.nch_image, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)


        self.DECBR1_info = nn.Sequential(
            DECBR2D(self.nch_info, 1 * self.nch_image, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False),
            DECBR2D(1 * self.nch_image, 2 * self.nch_image, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False),
            DECBR2D(2 * self.nch_image, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False),
            DECBR2D(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        )
        self.CBR1 = CBR2D(1 * self.nch_image, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.half_nch_image = int(self.nch_image / 2)
        #self.DECBR6 = CBR2D(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        #self.DECBR7 = DECBR2D(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)
        self.DECBR8 = DECBR2D(4 * self.nch_ker, 1 * self.nch_image, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)


    def forward(self, image, info):
        x_1 = self.CBR1(image)
        x_2 = self.DECBR1_info(info)
        x = torch.cat([x_1, x_2], 1)
        x = self.DECBR8(x)

        x = torch.tanh(x)
        return x

class DCGAN_R_yp(nn.Module):
    def __init__(self, nch_image, nch_info, nch_ker=128):

        super(DCGAN_R_yp, self).__init__()

        self.nch_image = nch_image
        self.nch_ker = nch_ker
        self.nch_info = nch_info

        # DCGAN_D
        self.CBR1_1 = CBR2D(1 * nch_image, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR1_2 = CBR2D(1 * nch_info, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR2 = CBR2D(2 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR3 = CBR2D(4 * nch_ker, 8 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR4 = CBR2D(8 * nch_ker, 16 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR5 = CBR2D(16 * nch_ker, 1,           kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)

        # DCGAN_G
        self.DECBR1_1 = DECBR2D(1 * self.nch_image, 8 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False)
        self.DECBR1_2 = DECBR2D(1 * self.nch_info, 4 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False)
        self.DECBR2 = DECBR2D(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR3 = DECBR2D(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR4 = DECBR2D(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR5 = DECBR2D(1 * self.nch_ker, 1 * self.nch_image, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)


        self.DECBR1_info = nn.Sequential(
            DECBR2D(self.nch_info, 1 * self.nch_image, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False),
            DECBR2D(1 * self.nch_image, 2 * self.nch_image, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False),
            DECBR2D(2 * self.nch_image, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False),
            DECBR2D(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        )
        self.CBR1 = CBR2D(1 * self.nch_image, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        #self.DECBR6 = CBR2D(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        #self.DECBR7 = DECBR2D(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)
        self.DECBR8 = DECBR2D(2 * self.nch_ker, 1 * self.nch_image, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)


    def forward(self, info):
        x = self.DECBR1_info(info)
        #x = self.DECBR6(x)
        #x = self.DECBR7(x)
        x = self.DECBR8(x)

        x = torch.tanh(x)
        return x

class DCGAN_C(nn.Module):
    def __init__(self, nch_image, nch_info, nch_ker=128):

        super(DCGAN_C, self).__init__()

        self.nch_image = nch_image
        self.nch_ker = nch_ker
        self.nch_info = nch_info

        # DCGAN_D
        self.CBR1 = CBR2D(1 * nch_image, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR2 = CBR2D(2 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR3 = CBR2D(4 * nch_ker, 8 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR4 = CBR2D(8 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.fc1 = nn.Linear(4 * nch_ker * 4 * 4, 4 * nch_ker)
        self.fc2 = nn.Linear(4 * nch_ker, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, image1, image2):
        x_1 = self.CBR1(image1)
        x_2 = self.CBR1(image2)
        x = torch.cat([x_1, x_2], 1) # 256*32*32
        x = self.CBR2(x) # 512*16*16
        x = self.CBR3(x) # 1024*8*8
        x = self.CBR4(x) # 512*4*4
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = torch.sigmoid(x)

        return x

'''
class DCGAN_G(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=128):
        super(DCGAN_G, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker

        self.DECBR1_1 = DECBR2D(1 * self.nch_in, 8 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False)
        #self.DECBR1_2 = DECBR2D(1 * self.n_class, 4 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False)
        self.DECBR2 = DECBR2D(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR3 = DECBR2D(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR4 = DECBR2D(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR5 = DECBR2D(1 * self.nch_ker, 1 * self.nch_out, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)

    def forward(self, input):
        #x_1 = self.DECBR1_1(input)
        #x_2 = self.DECBR1_2(label)
        #x = torch.cat([x_1,x_2],1)
        x = self.DECBR1_1(input)
        x = self.DECBR2(x)
        x = self.DECBR3(x)
        x = self.DECBR4(x)
        x = self.DECBR5(x)

        x = torch.tanh(x)
        return x

'''
class DCGAN_D(nn.Module):
    def __init__(self, nch_in,n_class, nch_ker=64):
        super(DCGAN_D, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.n_class = n_class

        self.CBR1_1 = CBR2D(1 * nch_in, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR1_2 = CBR2D(1 * n_class, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR2 = CBR2D(2 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR3 = CBR2D(4 * nch_ker, 8 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR4 = CBR2D(8 * nch_ker, 16 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR5 = CBR2D(16 * nch_ker, 8 * nch_ker, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)
        self.CBR6 = CBR2D(8 * nch_ker, 1, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)

    def forward(self, input, label):

        x_1 = self.CBR1_1(input)
        x_2 = self.CBR1_2(label)
        x = torch.cat([x_1,x_2],1)
        x = self.CBR2(x)
        x = self.CBR3(x)
        x = self.CBR4(x)
        x = self.CBR5(x)
        x = self.CBR6(x)

        x = torch.sigmoid(x)

        return x

class DCGAN_DforR(nn.Module):
    def __init__(self, nch_in,n_class, nch_ker=64):
        super(DCGAN_DforR, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.n_class = n_class

        self.CBR1_1 = CBR2D(1 * nch_in, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR1_2 = CBR2D(1 * nch_ker, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR2 = CBR2D(1 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR3 = CBR2D(4 * nch_ker, 8 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR4 = CBR2D(8 * nch_ker, 16 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR5 = CBR2D(16 * nch_ker, 1,           kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)

    def forward(self, input):

        x = self.CBR1_1(input)
        x = self.CBR1_2(x)
        x = self.CBR2(x)
        x = self.CBR3(x)
        x = self.CBR4(x)
        x = self.CBR5(x)

        x = torch.sigmoid(x)

        return x

def init_weights(net, init_gain=0.02):

    def  init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
