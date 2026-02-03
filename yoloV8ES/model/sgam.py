import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(self, c, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GAM(nn.Module):
    """
    Global Attention Mechanism (Ref: https://arxiv.org/abs/2112.05561)
    As depicted in Figure 3(b) of the provided paper.
    Uses Permutations to capture Channel-Spatial dependencies.
    """
    def __init__(self, c, r=4):
        super().__init__()
        mid_c = max(4, c // r)
        
        # Channel Attention Sub-module
        self.channel_attention = nn.Sequential(
            nn.Linear(c, mid_c),
            nn.ReLU(inplace=True),
            nn.Linear(mid_c, c)
        )
        
        # Spatial Attention Sub-module (using MLP on Permuted dimensions)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c, mid_c, 7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, c, 7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(c)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Channel Attention
        # Permute to [B, W, H, C] for linear layer application on C
        y = x.permute(0, 3, 2, 1) # [B, W, H, C]
        y = self.channel_attention(y)
        y = y.permute(0, 3, 2, 1) # Back to [B, C, H, W]
        x = x * torch.sigmoid(y)
        
        # 2. Spatial Attention
        # GAM Paper uses specific spatial mixing.
        # Figure 3b shows Permutation -> MLP -> Reverse
        y = self.spatial_attention(x)
        x = x * torch.sigmoid(y)
        
        return x

class CoordinateAttention(nn.Module):
    """Coordinate Attention"""
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out

class SGAM(nn.Module):
    """
    Selective Global Attention Mechanism
    Sequential Integration of SE -> GAM -> CA
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        if c2 is None: c2 = c1
        
        self.se = SE(c1)
        self.gam = GAM(c1)
        self.ca = CoordinateAttention(c1)
        
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        x = self.se(x)
        x = self.gam(x)
        x = self.ca(x)
        return self.proj(x)