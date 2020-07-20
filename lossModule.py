from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torch
import torch.nn as nn
from torchvision.models import vgg16 as pre_vgg16
import torchvision.transforms as transforms
import transform_network

#TODO:(02.06.2020)learn how to use pre trained vgg16 network and get the inter-feature-map for calculation
#to use the pre-trained vggnet the input images are required to be mini-batches of 3-channel RGB images of shape (3 x H x W)
#, where H and W are expected to be at least 224.
# The images have to be loaded in to a range of [0, 1] and then normalized
# using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(pre_vgg16(pretrained=True).features)[:23]#down load the pre-traind vgg16
        # features' 3，8，15，22 layers are : relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)
        return results
#TODO:(03.06.2020)To calculate the loss function
#here we already konwn the intermediate layers output the index of list is layers, the ferst dimension of very tensor is
#batch index. Now all data are on GPU.
#very element of the lisot is a 4-dimension tensor


class Loss_Net(torch.nn.Module):
    def __init__(self,alpha=9.0,beta=1.0):
        super(Loss_Net,self).__init__()
        self.alpha=alpha
        self.beta=beta

    def forward(self,x,vgg16net):#x cat at N :ys[1,3,H,W],y[N,3,H,W],yc[N,3,H,W]
        n,_,_,_ = x.shape#the firt dimension is batch size
        lossvalue = 0
        batchsize = int((n-1)/2)
        for i in range(1,batchsize+1):
            tempx = torch.stack((x[0,:,:,:],x[i,:,:,:],x[i+batchsize,:,:,:]),dim=0)
            lossvalue += self.loss_one_sample(tempx,vgg16net)
        return lossvalue/batchsize

    def loss_one_sample(self,x,vgg16net):#the input is the x(list of 4-d-tensor) of self-vgg16,along the first dimension are ys,y,yc
        x.requires_grad_(True)
        alpha = self.alpha
        beta = self.beta
        n,_,_,_ = x.shape
        input = vgg16net.forward(x)
        lossvalue = 0
        for j, featuremapj in enumerate(input, start=0):
            k, c, h, w = featuremapj.size()
            featuremapj = torch.reshape(featuremapj, (k, c, -1,))

            # vgg jth-layer output for ys,y,yc
            fs = featuremapj[0, :, :]
            f = featuremapj[1, :, :]
            fc = featuremapj[2, :, :]

            # lossj for style target
            #phi_s = f - fs
            # tphi_s =torch.transpose(torch.tensor(phi_s),0,1)
            #gramj = torch.mm(phi_s, phi_s.T) / (c * h * w)
            #gramj = torch.mm(phi_s, phi_s.T) / (c * h * w)
            gramjfs = torch.mm(fs,fs.T)/(c*h*w)
            gramjf = torch.mm(f,f.T)/(c*h*w)
            lossj = torch.sum((gramjfs-gramjf)**2)
            #lossj = torch.norm(gramj, p='fro')
            lossvalue += alpha*lossj

            # lossj for content target
            if j == 2:
                phi_c = f - fc
                #gramj = torch.mm(phi_c, phi_c.T) / (c * h * w)
                lossj = torch.sum(phi_c**2) / (c * h * w)
                #lossj = torch.norm(gramj, p='fro')
                lossvalue += beta*lossj

        return lossvalue