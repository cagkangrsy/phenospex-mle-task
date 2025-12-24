import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Convolutional block with two 3x3 convolutions.

    This block applies two consecutive 3x3 convolutions, each followed by
    batch normalization and ReLU activation. This is a standard building block
    used in both the encoder and decoder paths of the U-Net architecture.

    Architecture:
        Conv2d(3x3) -> BatchNorm -> ReLU -> Conv2d(3x3) -> BatchNorm -> ReLU
    """
    
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNetTiny(nn.Module):
    """
    Lightweight U-Net architecture for density map regression.

    This is a simplified U-Net with 3 encoder/decoder levels plus a bottleneck.
    The architecture follows the classic U-Net design with:
    - Encoder path: Progressive downsampling (32 -> 64 -> 128 -> 256 channels)
    - Bottleneck: Deepest feature representation
    - Decoder path: Progressive upsampling with skip connections
    - Output: Single-channel density heatmap

    The skip connections (concatenating encoder features with decoder features)
    help preserve fine-grained spatial information for accurate localization.

    Input shape: (B, in_ch, H, W)
    Output shape: (B, out_ch, H, W) with values in [0, 1] (via sigmoid)
    """
    
    def __init__(self, in_ch: int = 3, out_ch: int = 1) -> None:
        super().__init__()
        
        # Encoder path
        # Level 1: 32 channels
        self.enc1 = ConvBlock(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)  # Downsample by factor of 2

        # Level 2: 64 channels
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        # Level 3: 128 channels
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck: deepest features with 256 channels
        self.bottleneck = ConvBlock(128, 256)

        # Decoder path
        # Level 3: upsample from bottleneck and merge with enc3 features
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)  # 128 (upsampled) + 128 (skip) = 256 input

        # Level 2: upsample and merge with enc2 features
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)  # 64 (upsampled) + 64 (skip) = 128 input

        # Level 1: upsample and merge with enc1 features
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)  # 32 (upsampled) + 32 (skip) = 64 input

        # Output layer: 1x1 convolution to produce final heatmap
        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net architecture.

        The forward pass consists of:
        1. Encoder path: Extract features at multiple scales
        2. Bottleneck: Deepest feature representation
        3. Decoder path: Reconstruct full-resolution output with skip connections

        Args:
            x (torch.Tensor): Input image tensor of shape (B, in_ch, H, W)

        Returns:
            torch.Tensor: Density heatmap of shape (B, out_ch, H, W) with
                values in [0, 1] range (via sigmoid activation)
        """
        # Encoder path: extract multi-scale features
        # Level 1: Initial feature extraction
        e1 = self.enc1(x)          # (B, 32, H, W)
        p1 = self.pool1(e1)         # (B, 32, H/2, W/2) - downsample

        # Level 2: Deeper features at lower resolution
        e2 = self.enc2(p1)         # (B, 64, H/2, W/2)
        p2 = self.pool2(e2)         # (B, 64, H/4, W/4) - downsample

        # Level 3: Even deeper features
        e3 = self.enc3(p2)         # (B, 128, H/4, W/4)
        p3 = self.pool3(e3)         # (B, 128, H/8, W/8) - downsample

        # Bottleneck: Deepest feature representation
        b = self.bottleneck(p3)    # (B, 256, H/8, W/8)

        # Decoder path: reconstruct full resolution with skip connections
        # Level 3: Upsample and concatenate with encoder features (skip connection)
        u3 = self.up3(b)           # (B, 128, H/4, W/4) - upsample
        d3 = self.dec3(torch.cat([u3, e3], dim=1))  # Concatenate: (B, 256, H/4, W/4)

        # Level 2: Continue upsampling with skip connection
        u2 = self.up2(d3)          # (B, 64, H/2, W/2) - upsample
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # Concatenate: (B, 128, H/2, W/2)

        # Level 1: Final upsampling to original resolution
        u1 = self.up1(d2)          # (B, 32, H, W) - upsample
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # Concatenate: (B, 64, H, W)

        # Output layer: produce final density heatmap
        out = self.out_conv(d1)    # (B, out_ch, H, W)
        out = torch.sigmoid(out)
        return out
