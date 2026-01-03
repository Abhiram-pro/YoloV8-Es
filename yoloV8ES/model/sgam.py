import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Channel attention mechanism that recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels.
    
    Args:
        c: Number of channels
        r: Reduction ratio (default: 16)
    """
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global average pooling: [B, C, H, W] → [B, C, 1, 1] → [B, C]
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Channel attention: [B, C] → [B, C] → [B, C, 1, 1]
        y = self.fc(y).view(b, c, 1, 1)
        # Apply attention
        return x * y


class GAM(nn.Module):
    """
    Global Attention Mechanism
    
    Spatial attention that captures global context through channel reduction
    and expansion with sigmoid gating.
    
    Args:
        c: Number of channels
    """
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c // 2, 1)
        self.conv2 = nn.Conv2d(c // 2, c, 1)

    def forward(self, x):
        # Channel reduction and expansion: [B, C, H, W] → [B, C//2, H, W] → [B, C, H, W]
        y = self.conv1(x)
        y = F.relu(y, inplace=True)
        y = self.conv2(y)
        # Apply spatial attention with sigmoid gating
        return x * torch.sigmoid(y)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention
    
    Encodes spatial information along horizontal and vertical directions separately,
    enabling the network to capture long-range dependencies with precise positional information.
    
    Args:
        c: Number of channels
        r: Reduction ratio (default: 32)
    """
    def __init__(self, c, r=32):
        super().__init__()
        # Intermediate channels with minimum of 8
        m = max(8, c // r)
        self.conv1 = nn.Conv2d(c, m, 1)
        self.conv_h = nn.Conv2d(m, c, 1)
        self.conv_w = nn.Conv2d(m, c, 1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Directional pooling: preserve spatial information in each direction
        # Height pooling: [B, C, H, W] → [B, C, H, 1]
        h_pool = F.adaptive_avg_pool2d(x, (h, 1))
        # Width pooling: [B, C, H, W] → [B, C, 1, W] → [B, C, W, 1]
        w_pool = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)

        # Concatenate along spatial dimension: [B, C, H+W, 1]
        y = torch.cat([h_pool, w_pool], dim=2)
        y = F.relu(self.conv1(y), inplace=True)

        # Split back into height and width attention
        h_att, w_att = torch.split(y, [h, w], dim=2)
        w_att = w_att.permute(0, 1, 3, 2)

        # Apply coordinate-wise attention
        return x * torch.sigmoid(self.conv_h(h_att)) * torch.sigmoid(self.conv_w(w_att))


class SGAM(nn.Module):
    """
    Selective Global Attention Mechanism (SGAM)
    
    From: "Efficient and accurate road crack detection technology based on YOLOv8-ES"
    Section 3.3 - Combines three attention mechanisms sequentially
    
    Architecture: SE → GAM → CA
    - SE: Channel attention for feature recalibration
    - GAM: Global spatial attention for context modeling
    - CA: Coordinate attention for position-sensitive feature enhancement
    
    Used in the neck of YOLOv8-ES to improve feature fusion for crack detection.
    
    Args:
        c1: Number of input channels
        c2: Number of output channels (optional, defaults to c1)
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        # Default output channels to input channels if not specified
        if c2 is None:
            c2 = c1
        # If c2 looks like it should be c1 (YAML parser issue)
        elif isinstance(c2, int) and c2 > c1:
            c2 = c1
        
        # Ultralytics compatibility attributes
        self.f = -1  # From previous layer
        self.i = -1  # Layer index (will be set by model)
        self.type = 'SGAM'
        
        self.se = SE(c1)
        self.gam = GAM(c1)
        self.ca = CoordinateAttention(c1)
        
        # If c1 != c2, add a 1x1 conv to match dimensions
        self.conv = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # Sequential attention application
        x = self.se(x)   # Channel attention
        x = self.gam(x)  # Global spatial attention
        x = self.ca(x)   # Coordinate attention
        x = self.conv(x)  # Match output channels if needed
        return x
