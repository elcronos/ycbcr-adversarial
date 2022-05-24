import torch
import torch.nn as nn
import torchvision.models as m
from  torch.nn.functional import interpolate
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from math import log10
import random

random.seed = 2019
BASE_DIR     = '/your_directory_here'
DATASET = BASE_DIR + 'Datasets/'


def get_image(model, attack, index):
    bs = 50

    if index < 50:
        path = DATASET + '{}/{}/0.pt'.format(model, attack)
        x = torch.load(path)[index].cuda()
        y = 0
        return x,y

    else:
        batch = index // 50
        idx = index - (50 * batch)
        path = DATASET + '{}/{}/{}.pt'.format(model, attack, batch)
        x = torch.load(path)[idx].cuda()
        return x, batch


def get_image_clean(index):
    bs = 50

    if index < 50:
        path = DATASET + 'clean/0.pt'
        x = torch.load(path)[index].cuda()
        y = 0
        return x,y

    else:
        batch = index // 50
        idx = index - (50 * batch)
        path = DATASET + 'clean/{}.pt'.format(batch)
        x = torch.load(path)[idx].cuda()
        return x, batch

class Encoder(nn.Module):

    def __init__(self, model):
        super(Encoder, self).__init__()
        self.network    = model
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.fc = None


        self.network.layer1.name = 'layer1'
        self.network.layer2.name = 'layer2'
        self.network.layer3.name = 'layer3'
        self.network.layer4.name = 'layer4'
        self.network.fc.name     = 'fc'

        self.network.layer1.register_forward_hook(self.get_output_hidden)
        self.network.layer2.register_forward_hook(self.get_output_hidden)
        self.network.layer3.register_forward_hook(self.get_output_hidden)
        self.network.layer4.register_forward_hook(self.get_output_hidden)
        self.network.fc.register_forward_hook(self.get_input_hidden)

    def get_input_hidden(self,m,i,o):
        setattr(self,m.name, torch.stack(list(i)))

    def get_output_hidden(self,m,i,o):
        setattr(self,m.name, torch.stack(list(o)))

    def forward(self, x):
        result = self.network(x)
        return result, self.layer1, self.layer2, self.layer3, self.layer4, self.fc

# Define Residual block
class ResidualBlock(nn.Module):
    def __init__(self, channel_num, kernel=3):
        super(ResidualBlock, self).__init__()

        # the input and output channel number is channel_num
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel, padding=1),
            nn.BatchNorm2d(channel_num),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)

        return out



# Define Residual block
class ResidualBlock(nn.Module):
    def __init__(self, channel_num, kernel=3):
        super(ResidualBlock, self).__init__()

        # the input and output channel number is channel_num
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel, padding=1),
            nn.BatchNorm2d(channel_num),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)

        return out



class UpsampleNetwork(nn.Module):

    def __init__(self, model):
        super(UpsampleNetwork, self).__init__()

        # Encoder
        self.encoder = Encoder(model).cuda().eval()

        # Upsampling Architecture
        self.y_conv0 = nn.Conv2d(784,512,2,1)
        self.y_conv1 = self.ResidualUpsamplingConv(1024, 256, scale_factor=2, kernel=3)
        self.y_conv2 = self.ResidualUpsamplingConv(512,  128, scale_factor=2, kernel=3)
        self.y_conv3 = self.ResidualUpsamplingConv(256,   64, scale_factor=2, kernel=3)
        self.y_conv4 = self.ResidualUpsamplingConv(128,    64, scale_factor=2, kernel=3)
        self.y_conv5 = self.ResidualUpsamplingConv(64,      32, scale_factor=2, kernel=3)
        self.y_conv6 = self.ResidualUpsamplingConv(32,      1, scale_factor=1, kernel=3)


    def ResidualUpsamplingConv(self, in_ch, out_ch, scale_factor=2, kernel=3):
        upsample_dim = out_ch * (scale_factor ** 2)
        return nn.Sequential(
            ResidualBlock(in_ch),
            ResidualBlock(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel ,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, upsample_dim, kernel,1,1),
            nn.PixelShuffle(scale_factor)
        )

    def to_rgb(self,ycbcr):
            ycbcr = ycbcr.float()
            rgb = torch.zeros(tuple(ycbcr.shape))
            y   = ycbcr[:,0,:,:]
            cb  = ycbcr[:,1,:,:] - 128
            cr  = ycbcr[:,2,:,:] - 128
            # R
            rgb[:,0,:,:] = y + 1.402 * cr
            # G
            rgb[:,1,:,:] = y - .34414 * cb - .71414 * cr
            # B
            rgb[:,2,:,:] = y + 1.772 * cb
            #Quantization
            return rgb.float() / 255.

    def to_ycbcr(self,rgb):
        rgb = rgb.float()
        cbcr = torch.zeros(tuple(rgb.shape)).cuda()
        r    = rgb[:,:,:,0]
        g    = rgb[:,:,:,1]
        b    = rgb[:,:,:,2]
        # Y
        cbcr[:,:,:,0] = .299 * r + .587 * g + .114 * b
        # Cb
        cbcr[:,:,:,1] = 128 - .169 * r - .331 * g + .5 * b
        # Cr
        cbcr[:,:,:,2] = 128 + .5 * r - .419 * g - .081 * b
        #Quantization
        return cbcr.int()

    def initialise_features(self, x0):
        ######################################################################
        #    INITIAL FEATURES
        ######################################################################

        # YCbCr Color space conversion
        rgb = x0.permute(0, 2, 3, 1).cuda()
        self.ycbcr = (self.to_ycbcr(rgb*255.).float() / 255.).permute(0,3,1,2)

        #Downsampling
        #self.y_4x = interpolate(ycbcr[:,:,:,:], scale_factor=0.25, mode='bilinear', align_corners=False)[:,0].cuda()

        # Y channel
        self.y = self.ycbcr[:,0]
        #self.y = x0

        self.result, self.layer1, self.layer2, self.layer3, self.layer4, self.fc = self.encoder(x0)



    def forward(self, x):
        self.initialise_features(x)

        ###########################################################################
        # First Upconv
        ###########################################################################
        # In: [batch_size, 1, 224, 224]

        # Random noise
        noise = torch.rand((1,224,224)).cuda() * 0.005
        # Original input
        y = torch.clamp(self.y + noise,0,1).view(x.shape[0],784,8,8)
        # Feature Vector
        #x1 = self.fc.squeeze(0).view(x.shape[0],8,8,8)
        # Out: [batch_size, 8,8,8]
        #y = torch.cat((x0,x1), dim=1)
        # Out: [batch_size, 792,8,8]


        # In: [batch_size, 8,8,8]
        y = self.y_conv0(y)
        # Out: [batch_size, 512,7,7]

        # Skip Connection
        y = torch.cat((y , self.layer4), dim=1)
        # Out: [batch_size, 1024, 7, 7]

        # In: [batch_size, 1024, 7,7]
        y = self.y_conv1(y)
        # Out: [batch_size, 256, 14, 14]

        ###########################################################################
        # Second Upconv
        ###########################################################################

        # Skip Connection
        # In: [batch_size, 512, 14, 14]
        y = torch.cat((y, self.layer3), dim=1)
        y = self.y_conv2(y)
        #print(y.shape)
        # Out: [batch_size, 128, 28, 28]

        ###########################################################################
        # Third Upconv
        ###########################################################################

        # Skip Connection
        # In: [batch_size, 256, 28, 28]
        y = torch.cat((y, self.layer2), dim=1)
        y = self.y_conv3(y)
        #print(y.shape)
        # Out: [batch_size, 64, 56, 56]


        # In: [batch_size, 128, 56, 56]
        y = torch.cat((y, self.layer1), dim=1)
        y = self.y_conv4(y)
        # Out: [batch_size, 64, 112, 112]


        # In: [batch_size, 64, 112, 112]
        y = self.y_conv5(y)
        # Out: [batch_size, 32, 224, 224]

        ###########################################################################
        # Last Conv
        ###########################################################################

        # In: [batch_size, 32, 224, 224]
        y = self.y_conv6(y)
        # Out: [batch_size, 1, 224, 224

        # Replace y channel in original YCbCr Image
        self.ycbcr[:,0,:,:] = y

        # Convert back to rgb
        x1 = self.to_rgb(self.ycbcr*255.)
        # Clipping
        x1[x1 < 0] = 0
        x1[x1 > 1] = 1

        return x1

def get_model(checkpoint='./arch5_noise_2.pth'):
    resnet18 = m.resnet18(pretrained=True).eval().cuda()
    model = UpsampleNetwork(resnet18).cuda()
    model.load_state_dict(torch.load(checkpoint))
    return model.eval()
