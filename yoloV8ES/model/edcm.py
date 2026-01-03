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
        c1: Number of input channels
        c2: Number of output channels (optional, defaults to c1)
        k: Kernel size (default: 3)
        s: Stride (ignored, always 1 per paper)
        g: Number of groups for grouped convolution
    """

    def __init__(self, c1, c2=None, k=3, s=1, g=1):
        super().__init__()
        
        # Handle various argument patterns from YAML parser
        # c1 is always input channels (from previous layer)
        # c2 can be output channels, or None (default to c1)
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

        # 4 attention branches for dynamic kernel selection (PSA)
        def make_branch():
            mid_channels = max(c1 // 4, 4)
            return nn.Sequential(
                nn.Linear(c1, mid_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channels, 4, bias=True),
            )

        self.fc_s = make_branch()  # spatial attention
        self.fc_c = make_branch()  # channel attention
        self.fc_f = make_branch()  # filter attention
        self.fc_w = make_branch()  # kernel attention

        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        b, c, _, _ = x.shape

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
