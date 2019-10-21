from Data_management.dataset import NYUDataset
from Trainer import trainer
import albumentations as A
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
from albumentations.pytorch import ToTensor
from Models.resunet import RGBDepth_Depth


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
        loss = torch.sqrt( torch.mean(torch.abs(torch.log(real+1e-3)-torch.log(fake+1e-3)) ** 2 ) )
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

def strong_aug(p=0.5):
    return Compose([
        #RandomRotate90(),
        Flip(),
        #Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


augm = strong_aug(0.9)

depths = ['../sample_images/classroom_000310depth.png','../sample_images/classroom_000350depth.png',
          '../sample_images/classroom_000329depth.png']

# Instantiate a model and dataset
dataset = NYUDataset(depths, is_train = False, transforms=  augm)
# Parameters
params = {'batch_size': 16 ,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True}
params_test = {'batch_size': 16 ,
          'shuffle': False,
          'num_workers': 12,
          'pin_memory': True}

training_generator = data.DataLoader(dataset,**params)
val_generator = data.DataLoader(dataset_val,**params_test)

model = RGBDepth_Depth()

trainer = Trainer(training_generator, 
					val_generator,
					model, 
					optimizer)

