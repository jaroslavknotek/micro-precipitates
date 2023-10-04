import torch
from torch import nn
import precipitates.img_tools as it
import numpy as np

from torch.functional import F


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int = 2,dropout = .2):
        super().__init__()

        layers = []
        for i in range(n):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, 
                    out_channels,
                    kernel_size=3, 
                    padding=1, 
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ELU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        
        layers.pop() # remove last dropout            
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DownSamplingLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,dropout=.2):
        super().__init__()

        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ConvLayer(in_channels, out_channels,dropout=dropout)
        )

    def forward(self, x):
        return self.layer(x)


class UpSamplingLayer(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mode: str = 'transposed',
        dropout = .2
    ):
        """
        :param mode: 'transposed' for transposed convolution, or 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        """
        super().__init__()

        if mode == 'transposed':
            self.up = nn.ConvTranspose2d(
                in_channels, 
                in_channels // 2, 
                kernel_size=2, 
                stride=2
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(
                    in_channels, 
                    in_channels // 2, 
                    kernel_size=1,
                    dropout = dropout
                )
            )

        self.conv = ConvLayer(in_channels, out_channels,dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 1, 
        depth: int = 3, 
        start_filters: int = 16, 
        up_mode: str = 'transposed',
        dropout = .2
    ):
        super().__init__()

        self.inc = ConvLayer(in_channels, start_filters,dropout=dropout)

        # Contracting path
        self.down = nn.ModuleList(
            [
                DownSamplingLayer(
                    start_filters * 2 ** i, 
                    start_filters * 2 ** (i + 1),
                    dropout=dropout
                )
                for i in range(depth)
            ]
        )
        
        self.drop = nn.Dropout(dropout)
        # Expansive path
        self.up = nn.ModuleList(
            [
                UpSamplingLayer(
                    start_filters * 2 ** (i + 1), 
                    start_filters * 2 ** i, 
                    up_mode,
                    dropout=dropout
                )
                for i in range(depth - 1, -1, -1)
            ]
        )

        self.outc = nn.Sequential(
            nn.Conv2d(start_filters, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.inc(x)

        outputs = []

        for module in self.down:
            outputs.append(x)
            x = module(x)
        x = self.drop(x)

        for module, output in zip(self.up, outputs[::-1]):
            x = module(x, output)

        return self.outc(x)

# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha = .8,gamma = 2,reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)       
        
        bce = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        bce_exp = torch.exp(-bce)
        return self.alpha * (1-bce_exp)**self.gamma * bce
                       
    
        
class DiceLoss(nn.Module):
    def __init__(self, smooth = 1,reduction = 'none'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        #intersection = (inputs * targets).sum()                            
        intersection = inputs * targets
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        loss = 1 - dice
        if self.reduction == 'none':
            return loss
        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError()
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    


def predict(model,x,crop_size,device='cpu'):
    stride = crop_size//4
    crops = it.cut_to_squares(x,crop_size,stride)
    crops_3d = np.stack([crops]*3,axis=1) 
    with torch.no_grad():
        test = torch.from_numpy(crops_3d).to(device)
        res  = model(test)
        crops_predictions = np.squeeze(res.cpu().detach().numpy())
    denoise,fg,bg,borders = [
        it.decut_squares(crops_predictions[:,i],stride ,x.shape) 
        for i in range(crops_predictions.shape[1])
    ]
    
    return {
        "x":x,
        "denoise":denoise,
        "foreground":fg,
        "background":bg,
        "borders":borders
    }


def resolve_loss(loss_name):
    losses = {
        'dice':lambda:DiceLoss(reduction='none'),
        'bc':lambda:nn.BCEWithLogitsLoss(reduction='none'),
        'fl':lambda:FocalLoss(reduction='none')
    }
    return losses[loss_name]()
    
