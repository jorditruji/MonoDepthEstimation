from torchvision import models
# check keras-like model summary using torchsummary
import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):

    def __init__(self, n_class = 1):
        super().__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        
        self.base_layers = list(self.base_model.children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu(64, 64, 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)  

        self.layer4_e_loss = convrelu(512, 512, 1, 0)  


        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.upsample_v2 = nn.Upsample(size =(15, 20), mode='bilinear', align_corners=True)




        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Sequential(nn.Conv2d(64, n_class, 1),
        								nn.Sigmoid())

        self.x_sobel, self.y_sobel = self.make_sobel_filters()
        self.x_sobel = self.x_sobel.cuda()
        self.y_sobel = self.y_sobel.cuda()

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        # Down pass
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)

        # loss L_emb
        embedding_loss = self.layer4_e_loss(layer4)


        layer4 = self.layer4_1x1(embedding_loss)


        x = self.upsample_v2(layer4)
        layer3 = self.layer3_1x1(layer3)

        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
        
        return out, self.imgrad(out), embedding_loss, x
    
    def make_sobel_filters(self):
        ''' Returns sobel filters as part of the network'''

        a = torch.Tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

        # Add dims to fit batch_size, n_filters, filter shape
        a = a.view((1,1,3,3))
        a = Variable(a)

        # Repeat for vertical contours
        b = torch.Tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

        b = b.view((1,1,3,3))
        b = Variable(b)

        return a,b

    
    def imgrad(self,img):
        # Filter horizontal contours
        G_x = F.conv2d(img, self.x_sobel)
        
        # Filter vertical contrours
        G_y = F.conv2d(img, self.y_sobel)

        G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
        return G


if __name__ == "__main__":
	model = ResNetUNet(5)
	print(model.layer4)
	summary(model, input_size=(3, 240, 320))