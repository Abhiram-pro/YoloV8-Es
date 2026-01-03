"""
YOLOv8-ES Model Package

Core modules for YOLOv8-ES:
- EDCM: Enhanced Dynamic Convolution Module
- SGAM: Selective Global Attention Mechanism
- WIoUv3Loss: Wise-IoU v3 Loss Function
"""

from .edcm import EDCM
from .sgam import SGAM, SE, GAM, CoordinateAttention
from .loss_wiou import WIoUv3Loss, wiou_v3_loss

__all__ = [
    'EDCM',
    'SGAM',
    'SE',
    'GAM',
    'CoordinateAttention',
    'WIoUv3Loss',
    'wiou_v3_loss',
]

__version__ = '1.0.0'
