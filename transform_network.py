import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    # now i did'nt do resnet but use the equavlante block
    # bcause the dimension problem
    def __init__(self,inchannels,outchannels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannels,outchannels,3)#before conv1 128 × 84 × 84, after conv1 128 × 82 × 82
        self.bn1 = nn.InstanceNorm2d(outchannels)#do normalize in each feature map
        self.conv2 = nn.Conv2d(outchannels,outchannels,3)#after conv1 128 × 80 × 80
        self.bn2 = nn.InstanceNorm2d(outchannels)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = F.relu(self.bn1(o1))
        o3 = self.conv2(o2)
        o4 = self.bn2(o3)
        #o5 = o4+x
        o5 = F.relu(self.bn2(o4))
        return o5

class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()
        self.refpad = nn.ReflectionPad2d(40)# input 3 × 256 × 256 output 3 × 336 × 336
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=9,stride=1,padding=4)#32 × 9 × 9 conv, stride 1 32 × 336 × 336
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)#64 × 3 × 3 conv, stride 2 64 × 168 × 168
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)#128 × 3 × 3 conv, stride 2 128 × 84 × 84
        self.block1 = ResidualBlock(inchannels=128,outchannels=128)# Residual block, 128 filters 128 × 80 × 80
        self.block2 = ResidualBlock(inchannels=128,outchannels=128)# Residual block, 128 filters 128 × 76 × 76
        self.block3 = ResidualBlock(inchannels=128,outchannels=128)# Residual block, 128 filters 128 × 72 × 72
        self.block4 = ResidualBlock(inchannels=128, outchannels=128)# Residual block, 128 filters 128 × 68 × 68
        self.block5 = ResidualBlock(inchannels=128, outchannels=128)# Residual block, 128 filters 128 × 64 × 64
        self.conv4 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1)#64 × 3 × 3 conv, stride 1/2 64 × 128 × 128
        self.conv5 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,padding=1,output_padding=1)#32 × 3 × 3 conv, stride 1/2 32 × 256 × 256
        self.conv6 = nn.Conv2d(in_channels=32,out_channels=3,kernel_size=9,stride=1,padding=4)#3 × 9 × 9 conv, stride 1 3 × 256 × 256

    def forward(self, x):# input: 3 × 256 × 256
        o1 = self.refpad(x)
        o2 = F.relu(F.instance_norm(self.conv1(o1)))
        o3 = F.relu(F.instance_norm(self.conv2(o2)))
        o4 = F.relu(F.instance_norm(self.conv3(o3)))
        o5 = self.block1(o4)
        o6 = self.block2(o5)
        o7 = self.block3(o6)
        o8 = self.block4(o7)
        o9 = self.block5(o8)
        o10 = F.relu(F.instance_norm(self.conv4(o9)))
        o11 = F.relu(F.instance_norm(self.conv5(o10)))
        o12 = torch.tanh(F.instance_norm(self.conv6(o11)))
        return o12