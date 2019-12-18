from Data_management.dataset import NYUDataset
import torch
from torch.utils import data
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import sys
import copy
from Models.resunet import RGBDepth_Depth
import torch.nn.functional as F
import matplotlib 
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
from matplotlib import path, rcParams
import matplotlib.pyplot as plt
import albumentations as A
from albumentations import (Resize, Normalize,
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, RandomCrop
)
from albumentations.pytorch import ToTensor
import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from torch.utils.tensorboard import SummaryWriter
import argparse

class DepthScale(BasicTransform):
    """Transform applied to mask only."""
    def __init__(self, always_apply = True, p = 1.):
        super(DepthScale, self).__init__(always_apply, p)

    @property
    def targets(self):
        return {"mask": self.apply_to_mask}

    def apply_to_mask(self, img, **params):
        #print(img.shape)
        return (img-np.min(img))/(np.max(img)-np.min(img))



class RGB_dropper(BasicTransform):
    """Transform applied to image only."""
    def __init__(self, always_apply = True, p = 1., size = (240,320,3), drop_p = 1.):
        super(RGB_dropper, self).__init__(always_apply, p)
        self.matrix = np.random.choice([0.,1.], size = size, p = [drop_p, 1-drop_p])

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img, **params):
        #print(img.shape)
        return img[:,:,:]*self.matrix[:,:,:]


class Depth_dropper(BasicTransform):
    """Transform applied to image only."""
    def __init__(self, always_apply = True, p = 1., size = (240,320), drop_p = 1.):
        super(Depth_dropper, self).__init__(always_apply, p)
        self.matrix = np.random.choice([0.,1.], size = size, p = [drop_p, 1-drop_p])

    @property
    def targets(self):
        return {"mask": self.apply_to_mask}

    def apply_to_mask(self, img, **params):
        #print(img.shape)
        return img[:,:]*self.matrix[:,:]




# Save predictions
def save_predictions(prediction, rgb, depth, name = 'test'):
    # Matplotlib style display = channels last
    inp = rgb.numpy().transpose((1, 2, 0))
    print(inp.shape)
    mean = np.array([0.48958883, 0.41837043, 0.39797969])
    std = np.array([0.26429949, 0.2728771,  0.28336788])
    inp = std * inp + mean
    plt.subplot(3,1,1)
    plt.axis('off')
    plt.imshow(inp)
    plt.title("RGB")
    #Depth
    plt.subplot(3,1,2)
    plt.axis('off')

    plt.imshow(np.squeeze(depth.cpu().numpy()), 'gray', interpolation='nearest')
    plt.title("Ground truth")

    plt.subplot(3,1,3)
    plt.axis('off')

    plt.imshow(np.squeeze(prediction.cpu().numpy()), 'gray', interpolation='nearest')
    plt.title("Prediction")

    plt.show()
    plt.savefig(name+'.png')
    return


#LR decay:
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#Losses:
class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        #print("Calculing loss")
        #print(torch.min(fake), torch.min(real))
        #print(torch.max(fake), torch.max(real))
        loss = torch.sqrt( torch.mean(torch.abs(torch.log(real+1e-6)-torch.log(fake+1e-6)) ** 2 ) )
        #print(loss)
        return loss


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = ( grad_fake[:,:,None,:] @ grad_real[:,:,:,None] ).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt( torch.sum( grad_fake**2, dim=-1 ) )
        real_norm = torch.sqrt( torch.sum( grad_real**2, dim=-1 ) )

        return 1 - torch.mean( prod/(fake_norm*real_norm) )

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.sqrt( torch.mean( (10.*real-10.*fake) ** 2 ) )
        return loss


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    
    # L1 norm
    def forward(self, grad_fake, grad_real):
        
        return torch.sum( torch.mean( torch.abs(grad_real-grad_fake) ) )


def make_train_transforms(drop_p = 1.):
    return Compose([RandomCrop(360,480),
        Resize(240, 320),
        DepthScale(),
        HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, 
            val_shift_limit=15, p=0.5),
        HorizontalFlip(p=0.5),
        Normalize(
         mean=[0.48958883,0.41837043, 0.39797969],
            std=[0.26429949, 0.2728771,  0.28336788]),
        RGB_dropper(drop_p = 1.),
        #Depth_dropper(drop_p = drop_p),
        ToTensor()]
    )

def make_test_transforms(drop_p = 1.):
    return  Compose([Resize(240, 320),
        Normalize(
         mean=[0.48958883,0.41837043, 0.39797969],
            std=[0.26429949, 0.2728771,  0.28336788]),
        DepthScale(),
        RGB_dropper(drop_p = 1.),
        #Depth_dropper(drop_p = drop_p),
        ToTensor()]
    )

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default=None, type=str, required=True,
                        help="Options: NYUDataset, ScanNet.")

    args = parser.parse_args()

    # Instantiate a model and dataset
    net = RGBDepth_Depth()



    depths_list = np.load('Data_management/NYU_partitions0.npy', allow_pickle=True).item()
    #depths = ['Test_samples/frame-000000.depth.pgm','Test_samples/frame-000025.depth.pgm','Test_samples/frame-000050.depth.pgm','Test_samples/frame-000075.depth.pgm']

    train_depths = [depth for depth in depths_list['train'] if 'NYUstudy_0002_out/study_00026depth' not in depth]



    # dataset = Dataset(np.load('Data_management/dataset.npy').item()['train'][1:20])
    # Parameters
    params = {'batch_size': 18 ,
              'shuffle': True,
              'num_workers': 8,
              'pin_memory': True}
    params_test = {'batch_size': 18 ,
              'shuffle': False,
              'num_workers': 8,
              'pin_memory': True}




    net.train()
    print(net)


    writer = SummaryWriter(args.experiment_name)

    # Loss
    depth_criterion = RMSE_log()
    grad_loss = GradLoss()
    normal_loss = NormalLoss()
    mani_loss = RMSE()
    # Use gpu if possible and load model there
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = net.to(device)

    # Optimizer
    optimizer_ft = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    #scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    best_loss = 50
    iter_train = 0
    iter_val = 0
    n_epoch = 10
    for epoch in range(n_epoch):
        # Train
        net.train()
        cont = 0
        loss_train = 0.0
        grads_loss = 0.0
        RGB_drops = np.array([1]*n_epoch)# + list(range(5)) + [5]*(n_epoch-10))/5
        #RGB_drops = np.array([0]*n_epoch + list(range(5)) + [5]*(n_epoch-10))/5
        # flip
        #RGB_drops = RGB_drops[::-1]
        # Transforms train
        train_trans = make_train_transforms(drop_p = RGB_drops[epoch])
        test_trans =  make_test_transforms(drop_p = RGB_drops[epoch])
        print("{} ratio of RGB zeroed pixels".format(RGB_drops[epoch]))

        # Create datasets
        dataset = NYUDataset(train_depths,  transforms=train_trans)
        dataset_val = NYUDataset(depths_list['val'],  transforms=test_trans)
        training_generator = data.DataLoader(dataset,**params)
        val_generator = data.DataLoader(dataset_val,**params_test)

        for _i, (depths, rgbs, filename) in enumerate(training_generator):
            #cont+=1
            iter_train+=1
            # Get items from generator
            inputs, outputs = rgbs.cuda(), depths.cuda()
            writer.add_scalar('Others/train_RGB_information', 1-RGB_drops[epoch],iter_train)

            #print(torch.max(outputs.view(input.size(0), -1)))

            # Clean grads
            optimizer_ft.zero_grad()

            #Forward
            predicts, grads = net(inputs,outputs)
            #print("inputs: depth {} rgb {}".format(inputs.size(), outputs.size()))
            #print("outputs: depth {} grad {}".format(predicts.size(), grads.size()))
               
            #Backward+update weights
            depth_loss = depth_criterion(predicts, outputs) #+ depth_criterion(predicts[1], outputs)
            writer.add_scalar('Loss/train_RMSE_log', depth_loss.item(),iter_train)

            # Grad loss
            gradie_loss = 0.
            if epoch > 1:
                real_grad = net.imgrad(outputs)
                gradie_loss = grad_loss(grads, real_grad)#+ grad_loss(grads[1], real_grad)
                writer.add_scalar('Loss/train_MAE_grad_log', gradie_loss.item(), iter_train)

            #normal_loss = normal_loss(predict_grad, real_grad) * (epoch>7)
            #cont+=1
            # Manifold loss
            embed_lose = 0#mani_loss(manifolds[0],manifolds[1])

            loss = depth_loss + 12*gradie_loss #+0.045*embed_lose# + normal_loss
            writer.add_scalar('Loss/train_real_loss', loss.item(),  iter_train)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer_ft.step()
            loss_train+=loss.item()*inputs.size(0)
            if cont%250 == 0:
                #loss.append(depth_loss.item())
                print("TRAIN: [epoch %2d][iter %4d] loss: %.4f" \
                % (epoch, cont, depth_loss.item()))
                print("Mani: {}, depth:{}, gradient{}".format(0.075*embed_lose, depth_loss, gradie_loss))
            cont+=1    
        if epoch%1==0:
            predict_depth = predicts[0].detach().cpu()
            #np.save('pspnet'+str(epoch), saver)
            save_predictions(predict_depth[0].detach(), rgbs[0], outputs[0],name ="{}train_unet1_epoch_{}".format(args.experiment_name, str(epoch)))
            #predict_depth = predicts[1].detach().cpu()
            #np.save('pspnet'+str(epoch), saver)
            save_predictions(predicts[1][0].detach().cpu(), rgbs[1], outputs[1],name ="{}train_unet2_epoch_{}".format(args.experiment_name, str(epoch)))
       

        loss_train = loss_train/dataset.__len__()

     
        print("\n FINISHED TRAIN epoch %2d with loss: %.4f " % (epoch, loss_train ))
        # Val
        #net.eval()
        loss_val = 0.0
        cont = 0

        # We dont need to track gradients here, so let's save some memory and time
        with torch.no_grad():
            for _i, (depths, rgbs, filename) in enumerate(val_generator):
                cont+=1
                iter_val+=1
                # Get items from generator
                inputs = rgbs.cuda()
                # Non blocking so computation can start while labels are being loaded
                outputs = depths.cuda()
                
                #Forward
                predicts, grads= net(inputs,outputs)
                writer.add_scalar('Others/val_RGB_information', 1-RGB_drops[epoch],iter_train)
                #Sobel grad estimates:
                real_grad = net.imgrad(outputs)

                depth_loss = depth_criterion(predicts, outputs)#+depth_criterion(predict_grad, real_grad)
                writer.add_scalar('Loss/val_RMSE_log', depth_loss.item(), iter_val)

                loss_val+=depth_loss.item()*inputs.size(0)
                if cont%250 == 0:
                    print("VAL: [epoch %2d][iter %4d] loss: %.4f" \
                    % (epoch, cont, depth_loss))   
                #scheduler.step()
            if epoch%1==0:
                predict_depth = predicts[0].detach().cpu()
                #np.save('pspnet'+str(epoch), saver)
                save_predictions(predict_depth[0].detach(), rgbs[0], outputs[0],name ="{}unet1_epoch_{}".format(args.experiment_name, str(epoch)))
                #predict_depth = predicts[1].detach().cpu()
                #np.save('pspnet'+str(epoch), saver)
                save_predictions(predicts[1][0].detach().cpu(), rgbs[1], outputs[1],name ="{}unet2_epoch_{}".format(args.experiment_name, str(epoch)))


            loss_val = loss_val/dataset_val.__len__()
            print("\n FINISHED VAL epoch %2d with loss: %.4f " % (epoch, loss_val ))

        if loss_val< best_loss and epoch>2:
            best_loss = loss_val
            best_model_wts = copy.deepcopy(net.state_dict())
            torch.save({'model': best_model_wts, 'optim':optimizer_ft.state_dict() }, 'pesosmultioencoder')




