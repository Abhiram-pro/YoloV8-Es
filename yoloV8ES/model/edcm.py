import torch
import torch.nn as nn
import torch.nn.functional as F


class EDCM(nn.Module):
    """
    Enhanced Dynamic Convolution Module (EDCM)
    
    From: "Efficient and accurate road crack detection technology based on YOLOv8-ES"
    Section 3.2 - Combines ODConv (Omni-Dimensional Dynamic Convolution) with PSA
    
    Key characteristics:
    - Dynamic kernels across 4 dimensions (spatial, channel, filter, kernel)
    - Always uses stride=1 (no downsampling)
    - Per-sample adaptive convolution weights
    
    Args:
        c1: Number of input channels (already scaled by width_multiple)
        c2: Number of output channels (already scaled, optional, defaults to c1)
        k: Kernel size (default: 3)
        s: Stride (ignored, always 1 per paper)
        g: Number of groups for grouped convolution
    """

    def __init__(self, c1=None, c2=None, k=3, s=1, g=1):
        super().__init__()
        
        # Handle case where c1 might not be passed
        if c1 is None:
            raise ValueError("EDCM requires c1 (input channels) to be specified")
        
        # c2 defaults to c1 if not specified
        if c2 is None:
            c2 = c1

        self.k = k
        self.groups = g
        self.out_channels = c2
        self.in_channels = c1
        
        # Ultralytics compatibility attributes
        self.f = -1  # From previous layer
        self.i = -1  # Layer index (will be set by model)
        self.type = 'EDCM'

        # 4 parallel dynamic convolution kernels (ODConv)
        self.weight = nn.Parameter(
            torch.randn(4, c2, c1 // g, k, k)
        )

        # Global average pooling for attention
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Attention branches will be created lazily on first forward pass
        # This allows us to handle the actual runtime channel dimensions
        self._attention_initialized = False
        
        self.bn = nn.BatchNorm2d(c2)
    
    def _init_attention_branches(self, actual_channels, device):
        """Initialize attention branches with actual runtime channel dimensions"""
        mid_channels = max(actual_channels // 4, 4)
        
        self.fc_s = nn.Sequential(
            nn.Linear(actual_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 4, bias=True),
        ).to(device)  # spatial attention
        
        self.fc_c = nn.Sequential(
            nn.Linear(actual_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 4, bias=True),
        ).to(device)  # channel attention
        
        self.fc_f = nn.Sequential(
            nn.Linear(actual_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 4, bias=True),
        ).to(device)  # filter attention
        
        self.fc_w = nn.Sequential(
            nn.Linear(actual_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 4, bias=True),
        ).to(device)  # kernel attention
        
        self._attention_initialized = True
        self._actual_channels = actual_channels

    def forward(self, x):
        b, c, _, _ = x.shape

        # Initialize attention branches on first forward pass with actual channel dimensions
        if not self._attention_initialized:
            self._init_attention_branches(c, x.device)

        # Compute attention weights for each dimension
        pooled = self.gap(x).view(b, c)

        alpha_s = torch.softmax(self.fc_s(pooled), dim=1)
        alpha_c = torch.softmax(self.fc_c(pooled), dim=1)
        alpha_f = torch.softmax(self.fc_f(pooled), dim=1)
        alpha_w = torch.softmax(self.fc_w(pooled), dim=1)

        # Combine attention weights (element-wise multiplication)
        alpha = (alpha_s * alpha_c * alpha_f * alpha_w).view(
            b, 4, 1, 1, 1, 1
        )

        # Generate dynamic weights per sample
        weight = self.weight.unsqueeze(0)
        dyn_weight = (alpha * weight).sum(dim=1)
        
        # dyn_weight: [B, out_c, in_c//groups, k, k]
        # Apply per-sample convolution using grouped conv trick
        b, out_c, in_c_per_group, kh, kw = dyn_weight.shape
        
        x_reshaped = x.reshape(1, b * c, x.shape[2], x.shape[3])
        weight_reshaped = dyn_weight.reshape(b * out_c, in_c_per_group, kh, kw)
        
        out = F.conv2d(
            x_reshaped,
            weight_reshaped,
            stride=1,  # Always 1 per YOLOv8-ES paper
            padding=self.k // 2,
            groups=b * self.groups,
        )
        
        out = out.reshape(b, out_c, out.shape[2], out.shape[3])

        return self.bn(out)
