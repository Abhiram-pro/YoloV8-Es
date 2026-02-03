import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Helper for ODConv attention branches"""
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        
        # Calculate attention channel safely
        attention_channel = max(int(in_planes * reduction), min_channel)
        
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

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
    Uses Lazy Initialization to adapt to input channels automatically.
    """
    def __init__(self, c1, c2=None, k=3, s=1, g=1, kernel_num=1):
        super().__init__()
        if c2 is None: c2 = c1
        
        # Store parameters for lazy build
        self.c2 = c2
        self.k = k
        self.s = s
        self.g = g
        self.kernel_num = kernel_num
        
        # We will build layers on the first forward pass
        self.initialized = False
        
        # Placeholder for submodules
        self.attention = None
        self.weight = None
        self.bias = None
        self.bn = None
        self.act = None

    def _init_layers(self, x):
        """Initialize layers based on actual input tensor x"""
        c1 = x.shape[1] # Detect actual input channels
        device = x.device
        
        self.in_planes = c1
        self.out_planes = self.c2
        
        self.attention = Attention(c1, self.c2, self.k, self.g).to(device)
        
        self.weight = nn.Parameter(torch.randn(self.kernel_num, self.c2, c1//self.g, self.k, self.k).to(device))
        self.bias = nn.Parameter(torch.zeros(self.c2).to(device))
        
        self.bn = nn.BatchNorm2d(self.c2).to(device)
        self.act = nn.SiLU().to(device)
        
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init_layers(x)
            
        batch_size, channels, height, width = x.size()
        
        # Calculate Attentions
        attn_feats = self.attention(x)
        
        channel_attn = torch.sigmoid(self.attention.channel_fc(attn_feats)).view(batch_size, 1, channels, 1, 1)
        filter_attn = torch.sigmoid(self.attention.filter_fc(attn_feats)).view(batch_size, self.out_planes, 1, 1, 1)
        spatial_attn = torch.sigmoid(self.attention.spatial_fc(attn_feats)).view(batch_size, 1, 1, 1, self.k, self.k)
        kernel_attn = torch.softmax(self.attention.kernel_fc(attn_feats), dim=1).view(batch_size, self.kernel_num, 1, 1, 1, 1)

        weight = self.weight.unsqueeze(0) 
        weight = weight * channel_attn.unsqueeze(1) 
        weight = weight * filter_attn.unsqueeze(1).unsqueeze(3)
        weight = weight * spatial_attn.unsqueeze(1).unsqueeze(2)
        weight = (weight * kernel_attn).sum(dim=1)
        
        x = x.reshape(1, -1, height, width)
        weight = weight.reshape(batch_size * self.out_planes, self.in_planes // self.g, self.k, self.k)
        
        out = F.conv2d(x, weight, bias=None, stride=1, padding=self.k//2, groups=batch_size * self.g)
        out = out.view(batch_size, self.out_planes, height, width)
        
        out = out + self.bias.view(1, -1, 1, 1)
        return self.act(self.bn(out))