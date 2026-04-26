import torch
from torchvision.models import vgg19
import torch.nn as nn

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        self.vgg = vgg.features[:35]
        self.vgg.eval().to(device)

        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        input = self.vgg(x)
        target = self.vgg(y)
        return self.loss(input, target)
        
