from torchvision import models
# check keras-like model summary using torchsummary
import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class RGBDepth_Depth(nn.Module):
    """
    Double encoder consisting on residual blocks (one for RGB and one for depth) and shared decoder which uses a depth manifold plus RGB multiscale information to 
    predict depth maps at the original image resolution.

    :ivar depth_names (list): List of the depth images paths 
    :ivar is_train (boolean): Load images for train/inference 
    :ivar transforms (albumentation or str): Loads augmentator config from path if str and sets it to attr transforms
    """
    def __init__(self, n_class = 1, dropout = True):
        self.dropout = dropout
        super().__init__()
        # Pretrained resnet
        self.base_model = models.resnet18(pretrained=True)
        
        self.base_layers = list(self.base_model.children())

        # Encoder RGB
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

        del self.base_model
        del self.base_layers
        self.base_model = models.resnet18(pretrained=True)
        
        self.base_layers = list(self.base_model.children())
        # Encoder Depth
        self.depth_input_cnn = convrelu(1, 3, 1, 0) # size=(N, 3, x.H, x.W)
        self.depth_layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.depth_layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.depth_layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.depth_layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.depth_layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)



        # Decoder RGB-DEPTH
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.upsample_v2 = nn.Upsample(size =(15, 20), mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.depth_conv_original_size0 = convrelu(3, 64, 3, 1)
        self.depth_conv_original_size1 = convrelu(64, 64, 3, 1)
        
        self.conv_last = nn.Sequential(nn.Conv2d(64, n_class, 1),
        								nn.Sigmoid())

        self.x_sobel, self.y_sobel = self.make_sobel_filters()
        self.x_sobel = self.x_sobel.cuda() if torch.cuda.is_available() else self.x_sobel
        self.y_sobel = self.y_sobel.cuda() if torch.cuda.is_available() else self.y_sobel
        self.base_layers = None # Avoid unnecessary memory
        self.drop_1 = nn.Dropout2d(p=0.35)
        self.drop_2 = nn.Dropout2d(p=1.)

    def forward(self, input, outputs):
        # Intermediate channels
        #start_time = time.time()
        #input = self.drop_1(input)
        #output = self.drop_2(outputs)	
        ground_truth = outputs.clone()
        #ground_truth = self.drop_2(ground_truth)
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        # Down pass RGB
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)

        #mani_RGB =  Variable(layer4.data.clone(), requires_grad=True)
        # Down pass depth
        depth_3channel = self.depth_input_cnn(ground_truth)
        depth_original = self.depth_conv_original_size0(depth_3channel)
        depth_original = self.depth_conv_original_size1(depth_original)
        depth_layer0 = self.depth_layer0(depth_3channel)            
        depth_layer1 = self.depth_layer1(depth_layer0)
        depth_layer2 = self.depth_layer2(depth_layer1)
        depth_layer3 = self.depth_layer3(depth_layer2)        
        depth_layer4 = self.depth_layer4(depth_layer3)        
        #mani_depth = Variable(depth_layer4.data.clone(), requires_grad=True)

        
        '''
        # Encoder - decoder connections
        #layer4 = self.layer4_1x1(layer4)
        layer3 = self.drop_1(self.layer3_1x1(layer3))
        layer2 = self.drop_1(self.layer2_1x1(layer2))
        layer1 = self.drop_1(self.layer1_1x1(layer1))
        layer0 = self.drop_1(layer0)
        x_original = self.drop_1(x_original)
        # Decoder RGB
        layer4 = self.layer4_1x1(layer4)
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
        '''
        # Decoder depth
        
        layer4 = self.layer4_1x1(depth_layer4)
        layer4 = self.drop_1(layer4)

        depth = self.upsample_v2(layer4)

        depth = torch.cat([depth, layer3], dim=1)
        depth = self.conv_up3(depth)
        depth = self.drop_1(depth)

        depth = self.upsample(depth)
        depth = torch.cat([depth, layer2], dim=1)
        depth = self.conv_up2(depth)
        depth = self.drop_1(depth)
        depth = self.upsample(depth)
        depth = torch.cat([depth, layer1], dim=1)
        depth = self.conv_up1(depth)
        depth = self.drop_1(depth)

        depth = self.upsample(depth)
        depth = torch.cat([depth, layer0], dim=1)
        depth = self.conv_up0(depth)
        depth = self.drop_1(depth)
        
        depth = self.upsample(depth)
        depth = torch.cat([depth, x_original], dim=1)
        depth = self.conv_original_size2(depth)        
        
        out_depth = self.conv_last(depth)
        #print("{} seconds for forward pass.".format(time.time()-start_time))

        return out_depth, self.imgrad(out_depth)

        # Retornem 2 reconstruccions, els gradients de les reconstruccions i els manifolds
        #return (out,out_depth), (self.imgrad(out),self.imgrad(out_depth)),(mani_RGB, mani_depth)
        #return (out,None), (self.imgrad(out),None),(mani_RGB, mani_depth)
        
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
