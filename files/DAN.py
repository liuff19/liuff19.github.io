import torch
import sys
import os
import torch
import torch.nn as nn
import torch.autograd as ta
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import h5py

import fbresnet

from scipy import misc

CNNmodel = fbresnet.resnet50(pretrained=False)
CNNmodel.fc = nn.Linear(2048, 3380)
CNNmodel = torch.nn.DataParallel(CNNmodel).cuda()
print(CNNmodel)
CNNmodel.eval()
checkpoint = torch.load('./output/ReID_resnet50_triplet_80_checkpoint.pth') # recognition model
CNNmodel.load_state_dict(checkpoint['state_dict'])
cudnn.benchmark = True

aggN = 5 # number of images to aggregate


def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight)
    elif classname.find('Linear') != -1:
        init.xavier_uniform(m.weight)


class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(aggN*3, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU())
        self.conv6_p = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(64, 3, 3, stride=1, padding=1, bias=False), nn.Sigmoid())
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')



    def forward(self, f, training=False):
        f1 = self.conv1(f)
        f2 = self.conv2(self.downsample(f1))
        f3 = self.conv3(self.downsample(f2))
        f4 = self.conv4(self.upsample(f3))
        f5 = self.conv5(self.upsample(f4))
        f_mid = f1 + f5
        f6 = self.conv6(f_mid)
        f6 = self.conv6_p(f6)
        f7 = self.conv7(f6)
        return f7


netG = G_Net()
netG.apply(weights_init)
print(netG)

class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.fc1 = nn.Linear(1*4*128, 256, bias=True)
        self.fc2 = nn.Linear(256, 1, bias=True)
        self.downsample = nn.MaxPool2d(2, 2)

    def forward(self, f, training=False):
        f = self.conv1(f)
        f = self.conv2(self.downsample(f))
        f = self.conv3(self.downsample(f))
        f = self.conv4(self.downsample(f))
        f = self.conv5(self.downsample(f))
        f = self.downsample(f)
        #print(f.size())
        f = f.view(f.size(0), -1)
        f = F.relu(self.fc1(f))
        p = F.sigmoid(self.fc2(f))
        return p

netD = D_Net()
netD.apply(weights_init)
print(netD)



def log(x):
    return torch.log(x + 1e-8)

criterion = nn.BCELoss()
L2criterion = nn.MSELoss()

batchSize = 16

real_label = 1
fake_label = 0

netD.cuda()
netG.cuda()
criterion.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

def clean_all():
    netD.zero_grad()
    netG.zero_grad()
    CNNmodel.zero_grad()

niter = 100000

class BatchLoader(object):

    def __init__(self):
        self.metricinput = np.load('./result/train_features_MARS.npy') # features
        self.imageinput = h5py.File('./MARS_train.h5', 'r') # images
        f_label = open('MARS_train.txt','r')
        self.labels = {}
        self.keys = []
        for line in f_label.readlines():
            key, nlabel = line.strip().split(' ')
            if not nlabel in self.labels:
                self.labels[nlabel] = []
                self.keys.append(nlabel)
            self.labels[nlabel].append(key)
        print '####dataset info####'
        for key in self.keys:
            #print key, len(self.labels[key])
            if len(self.labels[key]) < 2:
                print 'error', key
                self.keys.remove(key)
                self.labels.pop(key)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        totensor = transforms.ToTensor()
        self.transform= totensor

    def load_next_tuple(self, early_return=False):
        PosOrNeg = np.random.randint(2)
        if PosOrNeg == 0:
            label = np.random.randint(len(self.labels))
            label = self.keys[label]
            idx = np.random.permutation(len(self.labels[label]))
            key1 = self.labels[label][idx[0]]
            key2 = self.labels[label][idx[1]]
            label = 1
        else:
            label = np.random.randint(len(self.labels))
            label = self.keys[label]
            idx = np.random.permutation(len(self.labels[label]))
            key1 = self.labels[label][idx[0]]

            label2 = np.random.randint(len(self.labels))
            while label2 == label:
                label2 = np.random.randint(len(self.labels))
            label = self.keys[label2]
            idx = np.random.permutation(len(self.labels[label]))
            key2 = self.labels[label][idx[0]]
            label = 0

        idx = np.random.randint(self.imageinput[key1].shape[0], size=aggN)

        img = torch.zeros((aggN,3,144,56))
        for i in range(aggN):
            img[i,...] = self.transform(self.imageinput[key1][idx[i],...])
        inputdata = img.view(aggN*3, 144, 56)

        if early_return:
            return inputdata, img[np.random.randint(aggN),...]

        target = self.metricinput[int(key2)]

        tpidx = 0

        if label == 1:
            mindis = np.sqrt(np.sum((target - self.metricinput[int(key1)])**2)+1e-8)
            #print label, mindis
        else:
            mindis = 20
            #print label, np.sqrt(np.sum((target - self.metricinput[int(key1)])**2)+1e-8)
        tpidx = np.random.randint(aggN)
            
        lfwimg = img[tpidx,...]
        center = self.metricinput[int(key1)]
        return inputdata, target, mindis, label*2.0 - 1.0, lfwimg, center

batchLoader = BatchLoader()

ganfactor = 1

L1Loss = nn.L1Loss()


# init netG using mse loss
for epoch in range(10000):
        clean_all()
        frames = torch.zeros((batchSize, aggN*3, 144, 56))
        target = np.zeros((batchSize, 2048), np.float32)
        margin = np.zeros((batchSize, 1), np.float32)
        label = np.zeros((batchSize, 1), np.float32)
        real_img = torch.zeros((batchSize, 3, 144, 56))
        center = np.zeros((batchSize, 2048), np.float32)

        for i in range(batchSize):
            x0, x1, x2, x3, x4, x5= batchLoader.load_next_tuple()          
            frames[i,...] = x0
            target[i,...] = x1
            margin[i,...] = x2
            label[i,...] = x3
            real_img[i,...] = x4
            center[i,...] = x5
        #print(center.mean())

        #frames = torch.from_numpy(frames)
        target = torch.from_numpy(target)
        margin = torch.from_numpy(margin)
        label = torch.from_numpy(label)
        #real_img = torch.from_numpy(real_img)
        center = torch.from_numpy(center)

        frames = Variable(frames.cuda())
        real_img = Variable(real_img.cuda())

        fake = netG(frames)

        loss = L2criterion(fake, real_img)
        loss.backward()
        optimizerG.step()

        if epoch%100 == 0:
            print('pre-train', epoch, loss.data[0])


torch.save(netG.state_dict(), 'output_stn/netG_init_N5' + '.pth')


#checkpoint = torch.load('output_stn/netG_init' + '.pth')
#netG.load_state_dict(checkpoint)


# GAN training
for epoch in range(niter):

        frames = torch.zeros((batchSize, aggN*3, 144,56))
        real_img = torch.zeros((batchSize, 3, 144,56))

        for i in range(batchSize):
            x0, x1 = batchLoader.load_next_tuple(early_return=True)          
            frames[i,...] = x0
            real_img[i,...] = x1

        img_mean=np.array([0.485, 0.456, 0.406]).astype(np.float32).reshape((1,3,1,1))
        img_std=np.array([0.229, 0.224, 0.225]).astype(np.float32).reshape((1,3,1,1))
        img_mean = Variable(torch.from_numpy(img_mean).cuda())
        img_std = Variable(torch.from_numpy(img_std).cuda())        
        
        # train D
        clean_all()

        # train with real
        real_gpu = real_img.cuda()
        #input.copy_(real_gpu)
        inputv = Variable(real_gpu)
        output = netD(inputv)
        errD_real = -ganfactor*torch.mean(log(output))
        errD_real.backward()
        optimizerD.step()
        D_x = output.data.mean()

        # train with fake
        clean_all()
        frames = Variable(frames.cuda())
        fake = netG(frames)
        output = netD(fake)
        GD_x = output.data.mean()


        errD_fake = -ganfactor*torch.mean(log(1-output))
        errD_fake.backward(retain_graph=True)
        optimizerD.step()

        frames = torch.zeros((batchSize, aggN*3, 144,56))
        real_img = torch.zeros((batchSize, 3, 144,56))
        target = np.zeros((batchSize, 2048), np.float32)
        margin = np.zeros((batchSize, 1), np.float32)
        label = np.zeros((batchSize, 1), np.float32)
        center = np.zeros((batchSize, 2048), np.float32)

        for i in range(batchSize):
            x0, x1, x2, x3, x4, x5= batchLoader.load_next_tuple()          
            frames[i,...] = x0
            target[i,...] = x1
            margin[i,...] = x2
            label[i,...] = x3
            real_img[i,...] = x4
            center[i,...] = x5

        #frames = torch.from_numpy(frames)
        target = torch.from_numpy(target)
        margin = torch.from_numpy(margin)
        label = torch.from_numpy(label)
        #real_img = torch.from_numpy(real_img)
        center = torch.from_numpy(center)

        clean_all()
        frames = Variable(frames.cuda())
        fake = netG(frames)
        output = netD(fake)
        errG_fake = -ganfactor*torch.mean(log(output))

        norm_fake = (fake - img_mean)/img_std
        
        _, features = CNNmodel(fake)

        target = Variable(target.cuda())
        margin = Variable(margin.cuda())
        label = Variable(label.cuda())
        center = Variable(center.cuda())
        real_img = Variable(real_img.cuda())

        diff = features - target
        tp = F.relu(label * (torch.sqrt(torch.sum(diff*diff, dim=1, keepdim=True)) - margin))

        metric_loss = torch.sum(tp)/(batchSize)

        diff2 = torch.sqrt(torch.sum(torch.pow(features - center,2), dim=1))

        recon_loss =  0.1*torch.sum(diff2)/(batchSize)

        L1_recons = L1Loss(fake, real_img)


        loss = errG_fake + metric_loss + recon_loss 

        loss.backward()
        
        optimizerG.step()


        if epoch % 10 == 0:
            print('[%d/%d] D(x): %.4f D(G(x)): %.4f  recon_loss: %.4f metric_loss: %.4f L1_loss: %.4f'
                    % (epoch, niter, D_x, GD_x, recon_loss.data[0], metric_loss.data[0], L1_recons.data[0]))
            fake_img = fake.data[0,...].cpu().numpy().transpose((1,2,0))
        if epoch % 100 == 0:
            fake_img = (fake_img*255).astype(int)
            misc.imsave('output_stn/'+ str(epoch) + '.jpg', fake_img)
        if epoch % 1000 == 0:
            torch.save(netG.state_dict(), 'output_stn/netG_'+ str(epoch) + '.pth')


