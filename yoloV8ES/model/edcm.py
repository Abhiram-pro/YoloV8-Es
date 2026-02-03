import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Helper for ODConv attention branches"""
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        # Omni-Dimensional branches
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
        self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
        self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EDCM(nn.Module):
    """
    Enhanced Dynamic Convolution Module (EDCM)
    Combines ODConv logic with PSA structure.
    """
    def __init__(self, c1, c2=None, k=3, s=1, g=1, kernel_num=1):
        super().__init__()
        if c2 is None: c2 = c1
        self.in_planes = c1
        self.out_planes = c2
        self.kernel_num = kernel_num # Paper uses 4 dimensions, but n=1 is standard for ODConv single-dynamic
        self.kernel_size = k
        self.groups = g

        # Attention calculator
        self.attention = Attention(c1, c2, k, g)
        
        # Static Weight: [kernel_num, Out, In/g, k, k]
        self.weight = nn.Parameter(torch.randn(kernel_num, c2, c1//g, k, k), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(c2), requires_grad=True)
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Calculate Attentions
        attn_feats = self.attention(x)
        
        # 1. Channel Attention [B, In, 1, 1]
        channel_attn = torch.sigmoid(self.attention.channel_fc(attn_feats)).view(batch_size, 1, channels, 1, 1)
        
        # 2. Filter Attention [B, Out, 1, 1]
        filter_attn = torch.sigmoid(self.attention.filter_fc(attn_feats)).view(batch_size, self.out_planes, 1, 1, 1)
        
        # 3. Spatial Attention [B, 1, 1, k, k]
        spatial_attn = torch.sigmoid(self.attention.spatial_fc(attn_feats)).view(batch_size, 1, 1, 1, self.kernel_size, self.kernel_size)
        
        # 4. Kernel Attention [B, n, 1, 1] (if n > 1)
        kernel_attn = torch.softmax(self.attention.kernel_fc(attn_feats), dim=1).view(batch_size, self.kernel_num, 1, 1, 1, 1)

        # Modulate Weights
        # W' = alpha_w * (W * alpha_c * alpha_f * alpha_s)
        weight = self.weight.unsqueeze(0) # [1, n, Out, In, k, k]
        
        # Apply attentions
        # Note: We combine inputs carefully to avoid massive memory usage
        weight = weight * channel_attn.unsqueeze(1) # Broadcast over kernels
        weight = weight * filter_attn.unsqueeze(1).unsqueeze(3)
        weight = weight * spatial_attn.unsqueeze(1).unsqueeze(2)
        
        # Sum over kernel dimension (Standard dynamic conv reduction)
        weight = (weight * kernel_attn).sum(dim=1) # [B, Out, In, k, k]
        
        # Reshape for group convolution
        # Reshape x to [1, B*C, H, W]
        x = x.reshape(1, -1, height, width)
        # Reshape weight to [B*Out, In, k, k]
        weight = weight.reshape(batch_size * self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size)
        
        # Apply Conv
        out = F.conv2d(x, weight, bias=None, stride=1, padding=self.kernel_size//2, groups=batch_size * self.groups)
        out = out.view(batch_size, self.out_planes, height, width)
        
        # Add bias (broadcast)
        out = out + self.bias.view(1, -1, 1, 1)
        
        return self.act(self.bn(out))