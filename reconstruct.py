import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zlib
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import SSIM
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import warnings
import os
import lpips

from torchvision.models import vgg16

base_dir = os.getcwd()


class SimplerImprovedCNNAutoencoderWithSkip(nn.Module): # Using Transpose2d in decode process
    def __init__(self, latent_size=512):
        super(SimplerImprovedCNNAutoencoderWithSkip, self).__init__()
        self.latent_size = latent_size

        # --- Encoder ---
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 128, 128]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Dropout for regularization
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch_size, 128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Dropout for regularization
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch_size, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Dropout for regularization
        )
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [batch_size, 512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Dropout for regularization
        )

        # Latent Space
        self.encoder_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(512, latent_size),  # Latent space
        )

        # --- Decoder ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_size, 512 * 16 * 16),  # Expand to 512x16x16
            nn.ReLU(),
        )

        self.decoder_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [batch_size, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.25),  # Dropout for regularization
        )
        self.decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512 + 256, 128, kernel_size=4, stride=2, padding=1),
            # Input channels: 512 (from deconv1) + 256 (skip)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.25),  # Dropout for regularization
        )
        self.decoder_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 64, kernel_size=4, stride=2, padding=1),
            # Input channels: 256 (from deconv2) + 128 (skip)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),  # Dropout for regularization
        )
        self.decoder_deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 3, kernel_size=4, stride=2, padding=1),
            # Input channels: 128 (from deconv3) + 64 (skip)
            nn.Sigmoid(),  # Output in [0, 1] range
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_conv1(x)  # Shape: [batch_size, 64, 128, 128]
        x2 = self.encoder_conv2(x1)  # Shape: [batch_size, 128, 64, 64]
        x3 = self.encoder_conv3(x2)  # Shape: [batch_size, 256, 32, 32]
        x4 = self.encoder_conv4(x3)  # Shape: [batch_size, 512, 16, 16]

        # Latent Space
        latent_space = self.encoder_fc(x4)  # Shape: [batch_size, latent_size]

        # Decoder
        x = self.decoder_fc(latent_space)  # Shape: [batch_size, 512 × 16 × 16]
        x = x.view(x.size(0), 512, 16, 16)  # Reshape to [batch_size, 512, 16, 16]

        x = self.decoder_deconv1(x)  # Shape: [batch_size, 256, 32, 32]
        x4_resized = torch.nn.functional.interpolate(x4, size=x.shape[2:], mode='bilinear',
                                                     align_corners=False)  # Resize x4 to match x
        x = torch.cat([x, x4_resized], dim=1)  # Skip connection from x4 (Shape: [batch_size, 512 + 256, 32, 32])

        x = self.decoder_deconv2(x)  # Shape: [batch_size, 128, 64, 64]
        x3_resized = torch.nn.functional.interpolate(x3, size=x.shape[2:], mode='bilinear',
                                                     align_corners=False)  # Resize x3 to match x
        x = torch.cat([x, x3_resized], dim=1)  # Skip connection from x3 (Shape: [batch_size, 256 + 128, 64, 64])

        x = self.decoder_deconv3(x)  # Shape: [batch_size, 64, 128, 128]
        x2_resized = torch.nn.functional.interpolate(x2, size=x.shape[2:], mode='bilinear',
                                                     align_corners=False)  # Resize x2 to match x
        x = torch.cat([x, x2_resized], dim=1)  # Skip connection from x2 (Shape: [batch_size, 128 + 64, 128, 128])

        x = self.decoder_deconv4(x)  # Shape: [batch_size, 3, 256, 256]

        return x


class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


# class ImprovedCNNAutoencoderWithSkip(nn.Module):
#     def __init__(self, latent_size=2048):
#         super(ImprovedCNNAutoencoderWithSkip, self).__init__()
#         self.latent_size = latent_size
#
#         # --- Encoder (Unchanged) ---
#         self.encoder_conv1 = nn.Sequential(
#             nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),  # [B,128,256,256]
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.1),
#         )
#         self.encoder_conv2 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B,256,128,128]
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.1),
#         )
#         self.encoder_conv3 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B,512,64,64]
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.1),
#         )
#         self.encoder_conv4 = nn.Sequential(
#             nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # [B,1024,32,32]
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.1),
#         )
#         self.encoder_conv5 = nn.Sequential(
#             nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),  # [B,2048,16,16]
#             nn.BatchNorm2d(2048),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.1),
#         )
#         self.encoder_fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(2048, latent_size),
#         )
#
#         # --- Decoder (Fixed Channel Dimensions) ---
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(latent_size, 2048 * 16 * 16),
#             nn.ReLU(),
#         )
#
#         # PixelShuffle Blocks with Correct Channels
#         self.decoder_upsample1 = nn.Sequential(
#             nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
#             nn.PixelShuffle(upscale_factor=2),  # 2048 -> 512, 16x16 -> 32x32
#             nn.Conv2d(512, 1024, kernel_size=3, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#             nn.Dropout2d(0.1),
#         )
#
#         # Fixed decoder_upsample2 to expect 2048 input channels (1024 from upsample1 + 1024 from x5_skip)
#         self.decoder_upsample2 = nn.Sequential(
#             nn.Conv2d(2048, 4096, kernel_size=3, padding=1),  # Now expects 2048 input channels
#             nn.PixelShuffle(upscale_factor=2),  # 4096 -> 1024, 32x32 -> 64x64
#             nn.Conv2d(1024, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Dropout2d(0.1),
#         )
#
#         self.decoder_upsample3 = nn.Sequential(
#             nn.Conv2d(1024, 2048, kernel_size=3, padding=1),  # 512 from upsample2 + 512 from x4_skip
#             nn.PixelShuffle(upscale_factor=2),  # 2048 -> 512, 64x64 -> 128x128
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout2d(0.1),
#         )
#
#         self.decoder_upsample4 = nn.Sequential(
#             nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # 256 from upsample3 + 256 from x3_skip
#             nn.PixelShuffle(upscale_factor=2),  # 1024 -> 256, 128x128 -> 256x256
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Dropout2d(0.1),
#         )
#
#         self.decoder_upsample5 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 128 from upsample4 + 128 from x2_skip
#             nn.PixelShuffle(upscale_factor=2),  # 512 -> 128, 256x256 -> 512x512
#             nn.Conv2d(128, 3, kernel_size=3, padding=1),
#             nn.Sigmoid(),
#         )
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # Encoder
#         x1 = self.encoder_conv1(x)  # [B,128,256,256]
#         x2 = self.encoder_conv2(x1)  # [B,256,128,128]
#         x3 = self.encoder_conv3(x2)  # [B,512,64,64]
#         x4 = self.encoder_conv4(x3)  # [B,1024,32,32]
#         x5 = self.encoder_conv5(x4)  # [B,2048,16,16]
#         latent = self.encoder_fc(x5)  # [B,latent_size]
#
#         # Decoder
#         x = self.decoder_fc(latent).view(-1, 2048, 16, 16)  # [B,2048,16,16]
#
#         # Upsample with skip connections
#         x = self.decoder_upsample1(x)  # [B,1024,32,32]
#         x5_resized = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)  # [B,2048,32,32]
#         x5_skip = x5_resized[:, :1024, :, :]  # Take first 1024 channels to match
#         x = torch.cat([x, x5_skip], dim=1)  # [B,2048,32,32] (1024 + 1024)
#
#         x = self.decoder_upsample2(x)  # [B,512,64,64]
#         x4_resized = F.interpolate(x4, size=x.shape[2:], mode='bilinear', align_corners=False)  # [B,1024,64,64]
#         x4_skip = x4_resized[:, :512, :, :]  # Take first 512 channels
#         x = torch.cat([x, x4_skip], dim=1)  # [B,1024,64,64] (512 + 512)
#
#         x = self.decoder_upsample3(x)  # [B,256,128,128]
#         x3_resized = F.interpolate(x3, size=x.shape[2:], mode='bilinear', align_corners=False)  # [B,512,128,128]
#         x3_skip = x3_resized[:, :256, :, :]  # Take first 256 channels
#         x = torch.cat([x, x3_skip], dim=1)  # [B,512,128,128] (256 + 256)
#
#         x = self.decoder_upsample4(x)  # [B,128,256,256]
#         x2_resized = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=False)  # [B,256,256,256]
#         x2_skip = x2_resized[:, :128, :, :]  # Take first 128 channels
#         x = torch.cat([x, x2_skip], dim=1)  # [B,256,256,256] (128 + 128)
#
#         x = self.decoder_upsample5(x)  # [B,3,512,512]
#         return x


class ImprovedCNNAutoencoderWithSkip(nn.Module):
    def __init__(self, latent_size=2048):  # Increased latent size
        super(ImprovedCNNAutoencoderWithSkip, self).__init__()
        self.latent_size = latent_size

        # --- Encoder ---
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),  # Increased filters
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),  # Reduced dropout
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Increased filters
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Increased filters
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
        )
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # Increased filters
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
        )
        self.encoder_conv5 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),  # Additional layer
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
        )

        # Latent Space
        self.encoder_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(2048, latent_size),  # Increased latent size
        )

        # --- Decoder ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_size, 2048 * 16 * 16),  # Expand to 2048x16x16
            nn.ReLU(),
        )

        self.decoder_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # [batch_size, 1024, 32, 32]
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.attention1 = AttentionModule(2048 + 1024)  # Attention module for skip connection

        self.decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(2048 + 1024, 512, kernel_size=4, stride=2, padding=1),
            # Input channels: 2048 (from deconv1) + 1024 (skip)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.attention2 = AttentionModule(1024 + 512)  # Attention module for skip connection

        self.decoder_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(1024 + 512, 256, kernel_size=4, stride=2, padding=1),
            # Input channels: 1024 (from deconv2) + 512 (skip)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.attention3 = AttentionModule(512 + 256)  # Attention module for skip connection

        self.decoder_deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512 + 256, 128, kernel_size=4, stride=2, padding=1),
            # Input channels: 512 (from deconv3) + 256 (skip)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.attention4 = AttentionModule(256 + 128)  # Attention module for skip connection

        self.decoder_deconv5 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 3, kernel_size=4, stride=2, padding=1),
            # Input channels: 256 (from deconv4) + 128 (skip)
            nn.Sigmoid(),  # Output in [0, 1] range
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_conv1(x)  # Shape: [batch_size, 128, 256, 256]
        x2 = self.encoder_conv2(x1)  # Shape: [batch_size, 256, 128, 128]
        x3 = self.encoder_conv3(x2)  # Shape: [batch_size, 512, 64, 64]
        x4 = self.encoder_conv4(x3)  # Shape: [batch_size, 1024, 32, 32]
        x5 = self.encoder_conv5(x4)  # Shape: [batch_size, 2048, 16, 16]

        # Latent Space
        latent_space = self.encoder_fc(x5)  # Shape: [batch_size, latent_size]

        # Decoder
        x = self.decoder_fc(latent_space)  # Shape: [batch_size, 2048 × 16 × 16]
        x = x.view(x.size(0), 2048, 16, 16)  # Reshape to [batch_size, 2048, 16, 16]

        x = self.decoder_deconv1(x)  # Shape: [batch_size, 1024, 32, 32]
        x5_resized = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)  # Resize x5 to match x
        x = self.attention1(torch.cat([x, x5_resized], dim=1))  # Skip connection with attention

        x = self.decoder_deconv2(x)  # Shape: [batch_size, 512, 64, 64]
        x4_resized = F.interpolate(x4, size=x.shape[2:], mode='bilinear', align_corners=False)  # Resize x4 to match x
        x = self.attention2(torch.cat([x, x4_resized], dim=1))  # Skip connection with attention

        x = self.decoder_deconv3(x)  # Shape: [batch_size, 256, 128, 128]
        x3_resized = F.interpolate(x3, size=x.shape[2:], mode='bilinear', align_corners=False)  # Resize x3 to match x
        x = self.attention3(torch.cat([x, x3_resized], dim=1))  # Skip connection with attention

        x = self.decoder_deconv4(x)  # Shape: [batch_size, 128, 256, 256]
        x2_resized = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=False)  # Resize x2 to match x
        x = self.attention4(torch.cat([x, x2_resized], dim=1))  # Skip connection with attention

        x = self.decoder_deconv5(x)  # Shape: [batch_size, 3, 512, 512]

        return x


# class ImprovedCNNAutoencoderWithSkip(nn.Module):
#     def __init__(self, latent_size=512):
#         super(ImprovedCNNAutoencoderWithSkip, self).__init__()
#         self.latent_size = latent_size
#
#         # --- Encoder ---
#         # Layer 1: Input 256x256 -> Output 128x128
#         self.encoder_conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 128, 128]
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25),
#         )
#
#         # Layer 2: Input 128x128 -> Output 64x64
#         self.encoder_conv2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch_size, 128, 64, 64]
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25),
#         )
#
#         # Layer 3: Input 64x64 -> Output 32x32
#         self.encoder_conv3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch_size, 256, 32, 32]
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25),
#         )
#
#         # Layer 4: Input 32x32 -> Output 16x16
#         self.encoder_conv4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [batch_size, 512, 16, 16]
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25),
#         )
#
#         # Layer 5: Input 16x16 -> Output 8x8 (Additional layer for more complexity)
#         self.encoder_conv5 = nn.Sequential(
#             nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # [batch_size, 1024, 8, 8]
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25),
#         )
#
#         # Latent Space
#         self.encoder_fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
#             nn.Flatten(),
#             nn.Linear(1024, latent_size),  # Latent space (input size is 1024, matching encoder_conv5 output)
#
#         )
#
#         # --- Decoder ---
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(latent_size, 1024 * 8 * 8),  # Expand to 1024x8x8
#             nn.ReLU(),
#         )
#
#         # Layer 1: Input 8x8 -> Output 16x16
#         self.decoder_deconv1 = nn.Sequential(
#             nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # [batch_size, 512, 16, 16]
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Dropout2d(0.25),
#         )
#
#         # Layer 2: Input 16x16 -> Output 32x32
#         self.decoder_deconv2 = nn.Sequential(
#             nn.ConvTranspose2d(1024 + 512, 256, kernel_size=4, stride=2, padding=1),
#             # Input channels: 1024 (from deconv1) + 512 (skip)
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout2d(0.25),
#         )
#
#         # Layer 3: Input 32x32 -> Output 64x64
#         self.decoder_deconv3 = nn.Sequential(
#             nn.ConvTranspose2d(512 + 256, 128, kernel_size=4, stride=2, padding=1),
#             # Input channels: 512 (from deconv2) + 256 (skip)
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Dropout2d(0.25),
#         )
#
#         # Layer 4: Input 64x64 -> Output 128x128
#         self.decoder_deconv4 = nn.Sequential(
#             nn.ConvTranspose2d(256 + 128, 64, kernel_size=4, stride=2, padding=1),
#             # Input channels: 256 (from deconv3) + 128 (skip)
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Dropout2d(0.25),
#         )
#
#         # Layer 5: Input 128x128 -> Output 256x256
#         self.decoder_deconv5 = nn.Sequential(
#             nn.ConvTranspose2d(128 + 64, 3, kernel_size=4, stride=2, padding=1),
#             # Input channels: 128 (from deconv4) + 64 (skip)
#             nn.Sigmoid(),  # Output in [0, 1] range
#         )
#
#         # Initialize weights
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         """Initialize weights using Kaiming initialization."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # Encoder
#         x1 = self.encoder_conv1(x)  # Shape: [batch_size, 64, 128, 128]
#         x2 = self.encoder_conv2(x1)  # Shape: [batch_size, 128, 64, 64]
#         x3 = self.encoder_conv3(x2)  # Shape: [batch_size, 256, 32, 32]
#         x4 = self.encoder_conv4(x3)  # Shape: [batch_size, 512, 16, 16]
#         x5 = self.encoder_conv5(x4)  # Shape: [batch_size, 1024, 8, 8]
#
#         # Latent Space
#         latent_space = self.encoder_fc(x5)  # Shape: [batch_size, latent_size]
#
#         # Decoder
#         x = self.decoder_fc(latent_space)  # Shape: [batch_size, 1024 × 8 × 8]
#         x = x.view(x.size(0), 1024, 8, 8)  # Reshape to [batch_size, 1024, 8, 8]
#
#         x = self.decoder_deconv1(x)  # Shape: [batch_size, 512, 16, 16]
#         x5_resized = torch.nn.functional.interpolate(x5, size=x.shape[2:], mode='bilinear',
#                                                      align_corners=False)  # Resize x5 to match x
#         x = torch.cat([x, x5_resized], dim=1)  # Skip connection from x5 (Shape: [batch_size, 1024 + 512, 16, 16])
#
#         x = self.decoder_deconv2(x)  # Shape: [batch_size, 256, 32, 32]
#         x4_resized = torch.nn.functional.interpolate(x4, size=x.shape[2:], mode='bilinear',
#                                                      align_corners=False)  # Resize x4 to match x
#         x = torch.cat([x, x4_resized], dim=1)  # Skip connection from x4 (Shape: [batch_size, 512 + 256, 32, 32])
#
#         x = self.decoder_deconv3(x)  # Shape: [batch_size, 128, 64, 64]
#         x3_resized = torch.nn.functional.interpolate(x3, size=x.shape[2:], mode='bilinear',
#                                                      align_corners=False)  # Resize x3 to match x
#         x = torch.cat([x, x3_resized], dim=1)  # Skip connection from x3 (Shape: [batch_size, 256 + 128, 64, 64])
#
#         x = self.decoder_deconv4(x)  # Shape: [batch_size, 64, 128, 128]
#         x2_resized = torch.nn.functional.interpolate(x2, size=x.shape[2:], mode='bilinear',
#                                                      align_corners=False)  # Resize x2 to match x
#         x = torch.cat([x, x2_resized], dim=1)  # Skip connection from x2 (Shape: [batch_size, 128 + 64, 128, 128])
#
#         x = self.decoder_deconv5(x)  # Shape: [batch_size, 3, 256, 256]
#
#         return x


# Load the trained model
def load_model(model_path, model, optimizer, scheduler):
    """Loads the trained autoencoder model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(checkpoint)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    torch.set_rng_state(checkpoint['random_state'])
    np.random.set_state(checkpoint['numpy_random_state'])
    return model, optimizer, scheduler


def worker_init_fn(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def quantize(tensor, quantization_levels):
    tensor = torch.clamp(tensor, 0, 1)  # Ensure values are in [0, 1]
    Q = torch.round(tensor * (quantization_levels - 1))
    return Q


def dequantize(quantized_tensor, quantization_levels):
    Vq = quantized_tensor / (quantization_levels - 1)
    return Vq


# --- Lossless Compression Functions ---
def compress_latent_space(latent_space_quantized):
    # Convert tensor to a byte array.  Important: Ensure it's contiguous!
    latent_space_bytes = latent_space_quantized.cpu().detach().numpy().astype(
        np.float32).tobytes()  # Explicitly cast to float32
    # Compress using zlib (Deflate)
    compressed_bytes = zlib.compress(latent_space_bytes)
    return compressed_bytes


def decompress_latent_space(compressed_bytes, original_shape, quantization_levels, device):
    # Decompress using zlib
    latent_space_bytes = zlib.decompress(compressed_bytes)
    # Calculate the number of elements from the original shape
    num_elements = np.prod(original_shape)
    # Use numpy.frombuffer, explicitly defining the data type (e.g., float32)
    latent_space_quantized = torch.from_numpy(
        np.frombuffer(latent_space_bytes, dtype=np.float32)[:num_elements].reshape(original_shape)
    ).to(device)
    latent_space_dequantized = dequantize(latent_space_quantized, quantization_levels)  # Dequantize
    return latent_space_dequantized


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Adversarial Loss
class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()  # Define the discriminator network
        self.criterion = nn.BCELoss()  # Binary cross-entropy loss

    def forward(self, reconstructed, real):
        # Real and fake labels
        real_labels = torch.ones(real.size(0), 1, device=real.device)
        fake_labels = torch.zeros(reconstructed.size(0), 1, device=reconstructed.device)

        # Discriminator loss
        real_output = self.discriminator(real)
        fake_output = self.discriminator(reconstructed.detach())
        d_loss = self.criterion(real_output, real_labels) + self.criterion(fake_output, fake_labels)

        # Generator loss
        fake_output = self.discriminator(reconstructed)
        g_loss = self.criterion(fake_output, real_labels)

        return g_loss  # Return generator loss for the autoencoder


class LPIPSLoss(nn.Module):

    def __init__(self, net_type='vgg'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net_type)
        self.lpips.eval()

    def forward(self, output, target):
        output = (output - 0.5) * 2
        target = (target - 0.5) * 2
        return self.lpips(output, target).mean()


class PSNRLoss(nn.Module):

    def __init__(self, max_val=1.0, epsilon=1e-8):
        super().__init__()
        self.max_val = max_val
        self.epsilon = epsilon

    def forward(self, output, target):
        mse = torch.mean((output - target) ** 2)
        psnr = 10 * torch.log10(self.max_val ** 2 / (mse + self.epsilon))
        return -psnr


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.1, delta=0.1, debug=False):
        super().__init__()
        self.debug = debug
        self.alpha = alpha  # Weight for SSIM loss
        self.beta = beta  # Weight for L1 loss
        self.gamma = gamma  # Weight for perceptual loss
        self.delta = delta  # Weight for adversarial loss
        # self.l1_loss = nn.L1Loss()
        self.lpips_loss = LPIPSLoss(net_type='vgg')
        self.psnr_loss = PSNRLoss(max_val=1.0)

        # self.adversarial_loss = AdversarialLoss()
        # self.vgg = vgg16(pretrained=True).features[:16].eval()
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

    def perceptual_loss(self, output, target):
        # Extract features from VGG
        output_features = self.vgg(output)
        target_features = self.vgg(target)

        # Compute L1 loss between features
        return nn.L1Loss()(output_features, target_features)

    def forward(self, output, target):
        # Resize the output to match the target size (256x256)
        # output_resized = torch.nn.functional.interpolate(output, size=target.shape[2:], mode='bilinear',
        #                                                  align_corners=False)

        # Calculate L1 loss between resized output and target
        # l1_loss = self.l1_loss(output_resized, target)

        # Calculate SSIM loss
        # try:
        #     ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3).to("cpu")
        #     ssim_loss = 1 - ssim_loss(output_resized, target)
        # except Exception as e:
        #     print(f"Error calculating SSIM: {e}")
        #     ssim_loss = 0

        # Resize output and target to 224x224 for perceptual loss
        # output_resized_224 = torch.nn.functional.interpolate(output, size=(224, 224), mode='bilinear',
        #                                                      align_corners=False)
        # target_resized_224 = torch.nn.functional.interpolate(target, size=(224, 224), mode='bilinear',
        #                                                      align_corners=False)
        # #
        # # # Calculate perceptual loss on resized inputs
        # perceptual_loss = self.perceptual_loss(output_resized_224, target_resized_224)

        # Calculate adversarial loss
        # adversarial_loss = self.adversarial_loss(output, target)

        # Combine losses
        # total_loss = self.alpha * ssim_loss + self.beta * l1_loss + self.gamma * perceptual_loss + self.delta * adversarial_loss
        # total_loss = self.alpha * ssim_loss + self.beta * l1_loss + self.gamma * perceptual_loss

        lpips = self.lpips_loss(output, target)
        psnr = self.psnr_loss(output, target)

        total_loss = self.alpha * lpips + self.beta * psnr

        if self.debug:
            print(
                f"lpips loss: {lpips}, psnr loss: {psnr}"
            )
            # f"SSIM loss: {ssim_loss}, Adversarial loss: {adversarial_loss}")

        return total_loss


# 2. Load the state dictionary from the saved file
# model_path = f"{base_dir}/best_model.pth"  # Replace with the actual path to your saved model file
# loaded_autoencoder.load_state_dict(torch.load(model_path))

# # 3. Set the model to evaluation mode (Important if you're using it for inference)
# loaded_autoencoder.eval()  # Disable training-specific layers like dropout

class YOLOv5Dataset(Dataset):
    def __init__(self, image_dir, transform=None):  # Corrected init
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be
            applied on an image.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if
                            f.endswith(('.jpg', '.jpeg', '.bmp'))]

    def __len__(self):  # Corrected len
        return len(self.image_files)

    def __getitem__(self, idx):  # Corrected getitem
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Ensure
        # consistent color format

        if self.transform:
            image = self.transform(image)

        return image, img_name  # Return the image and a dummy label (0).  We
    # don't need the YOLO labels for autoencoder training


def reconstruct(model, train_dataloader):
    model.to("cpu")
    model.eval()

    criterion = CombinedLoss(debug=True)

    with torch.no_grad():
        for i, data in enumerate(train_dataloader, 0):
            inputs, _ = data
            # print(f"original image shape: {image.shape}")
            inputs = inputs.to("cpu")  # Apply the transformation and move to the device

            # print(f"Min pixel value: {original_image.min().item()}")
            # print(f"Max pixel value: {original_image.max().item()}")

            # If the image is grayscale, convert it to RGB
            if inputs.shape[0] != 3:
                inputs = inputs[:3, :, :]

            # original_image_size = original_image.element_size() * original_image.nelement() #bytes

            # f. Decode
            x1 = model.encoder_conv1(inputs)  # Shape: [batch_size, 128, 256, 256]
            x2 = model.encoder_conv2(x1)  # Shape: [batch_size, 256, 128, 128]
            x3 = model.encoder_conv3(x2)  # Shape: [batch_size, 512, 64, 64]
            x4 = model.encoder_conv4(x3)  # Shape: [batch_size, 1024, 32, 32]
            x5 = model.encoder_conv5(x4)  # Shape: [batch_size, 2048, 16, 16]

            # Latent Space
            latent_space = model.encoder_fc(x5)  # Shape: [batch_size, latent_size]

            # --- Quantization and Dequantization ---
            latent_space_quantized = quantize(latent_space, 256)  # Quantize
            compressed_bytes = compress_latent_space(latent_space_quantized)
            print(len(compressed_bytes))
            latent_space_dequantized = decompress_latent_space(compressed_bytes, latent_space_quantized.shape,
                                                               quantization_levels=256, device="cpu")
            # latent_space_dequantized = dequantize(latent_space_quantized, 256)
            # --- Decoder Forward Pass ---
            x = model.decoder_fc(latent_space)  # Shape: [batch_size, 1024 × 8 × 8]
            x = x.view(x.size(0), 1024, 8, 8)  # Reshape to [batch_size, 1024, 8, 8]

            x = model.decoder_deconv1(x)  # Shape: [batch_size, 512, 16, 16]
            x5_resized = torch.nn.functional.interpolate(x5, size=x.shape[2:], mode='bilinear',
                                                         align_corners=False)  # Resize x5 to match x
            x = torch.cat([x, x5_resized], dim=1)  # Skip connection from x5 (Shape: [batch_size, 1024 + 512, 16, 16])

            x = model.decoder_deconv2(x)  # Shape: [batch_size, 256, 32, 32]
            x4_resized = torch.nn.functional.interpolate(x4, size=x.shape[2:], mode='bilinear',
                                                         align_corners=False)  # Resize x4 to match x
            x = torch.cat([x, x4_resized], dim=1)  # Skip connection from x4 (Shape: [batch_size, 512 + 256, 32, 32])

            x = model.decoder_deconv3(x)  # Shape: [batch_size, 128, 64, 64]
            x3_resized = torch.nn.functional.interpolate(x3, size=x.shape[2:], mode='bilinear',
                                                         align_corners=False)  # Resize x3 to match x
            x = torch.cat([x, x3_resized], dim=1)  # Skip connection from x3 (Shape: [batch_size, 256 + 128, 64, 64])

            x = model.decoder_deconv4(x)  # Shape: [batch_size, 64, 128, 128]
            x2_resized = torch.nn.functional.interpolate(x2, size=x.shape[2:], mode='bilinear',
                                                         align_corners=False)  # Resize x2 to match x
            x = torch.cat([x, x2_resized], dim=1)  # Skip connection from x2 (Shape: [batch_size, 128 + 64, 128, 128])

            reconstructed_image = model.decoder_deconv5(x)  # Shape: [batch_size, 3, 256, 256]

            # reconstructed = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # reconstructed = (np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)
            # reconstructed = Image.fromarray(reconstructed_image)
            # reconstructed.save(f"{base_dir}/reconstructed_test_train.bmp")
            save_reconstructed_image(reconstructed_image, "reconstructed_train_valitaion_256", _,0)
            # inputs_resized = torch.nn.functional.interpolate(original_image, size=(256, 256), mode='bilinear',
            #                                                  align_corners=False)
            loss = criterion(reconstructed_image, inputs)


def train(model, train_dataloader, val_dataloader, optimizer, epochs: int, device="cpu", save_interval=500,
          save_path_prefix="autoencoder_model_epoch", best_val_loss=float('inf'), quantization_levels=256,
          best_model_save_path="", scheduler=None):
    model.to(device)  # Move the model to the appropriate device
    model.train()  # Set the model to training mode

    # Learning rate scheduler
    criterion = CombinedLoss(debug=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, _ = data
            # print(_)
            inputs = inputs.to(device)  # Move inputs to the appropriate device

            optimizer.zero_grad()  # Zero the gradients

            x1 = model.encoder_conv1(inputs)  # [B,128,256,256]
            x2 = model.encoder_conv2(x1)  # [B,256,128,128]
            x3 = model.encoder_conv3(x2)  # [B,512,64,64]
            x4 = model.encoder_conv4(x3)  # [B,1024,32,32]
            # x5 = model.encoder_conv5(x4)  # [B,2048,16,16]
            latent_space = model.encoder_fc(x4)  # [B,latent_size]  # [B, latent_size]  # Shape: [batch_size, 2048, 16, 16]

            # Latent Space

            # --- Quantization and Dequantization ---
            # latent_space_quantized = quantize(latent_space, quantization_levels)  # Quantize
            # latent_space_dequantized = decompress_latent_space(latent_space_quantized, quantization_levels)  # Dequantize
            latent_space_quantized = quantize(latent_space, 256)  # Quantize
            compressed_bytes = compress_latent_space(latent_space_quantized)
            print(len(compressed_bytes))
            latent_space_dequantized = decompress_latent_space(compressed_bytes, latent_space_quantized.shape,
                                                               quantization_levels=256, device="cpu")
            # print(latent_space_dequantized.shape)
            # --- Decoder Forward Pass ---
            x = model.decoder_fc(latent_space)  # Shape: [batch_size, 512 × 16 × 16]
            x = x.view(x.size(0), 512, 16, 16)  # Reshape to [batch_size, 512, 16, 16]

            x = model.decoder_deconv1(x)  # Shape: [batch_size, 256, 32, 32]
            x4_resized = torch.nn.functional.interpolate(x4, size=x.shape[2:], mode='bilinear',
                                                         align_corners=False)  # Resize x4 to match x
            x = torch.cat([x, x4_resized], dim=1)  # Skip connection from x4 (Shape: [batch_size, 512 + 256, 32, 32])

            x = model.decoder_deconv2(x)  # Shape: [batch_size, 128, 64, 64]
            x3_resized = torch.nn.functional.interpolate(x3, size=x.shape[2:], mode='bilinear',
                                                         align_corners=False)  # Resize x3 to match x
            x = torch.cat([x, x3_resized], dim=1)  # Skip connection from x3 (Shape: [batch_size, 256 + 128, 64, 64])

            x = model.decoder_deconv3(x)  # Shape: [batch_size, 64, 128, 128]
            x2_resized = torch.nn.functional.interpolate(x2, size=x.shape[2:], mode='bilinear',
                                                         align_corners=False)  # Resize x2 to match x
            x = torch.cat([x, x2_resized], dim=1)  # Skip connection from x2 (Shape: [batch_size, 128 + 64, 128, 128])

            reconstructed = model.decoder_deconv4(x)  # Shape: [batch_size, 3, 256, 256]

            # --- Resize Reconstructed Output to Match Input Size (256x256) ---
            reconstructed_resized = torch.nn.functional.interpolate(reconstructed, size=(256, 256), mode='bilinear',
                                                                    align_corners=False)

            # --- Compute Loss ---
            loss = criterion(reconstructed_resized, inputs)

            # --- Backward Pass and Optimization ---
            loss.backward()
            optimizer.step()

            # Gradient monitoring (optional)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    pass  # Optional: Print gradient statistics for debugging

            running_loss += loss.item()

            # Print loss every 100 iterations
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

            # Save the model periodically
            if (i + 1) % save_interval == 0:
                save_path = f"{base_dir}/{save_path_prefix}_epoch{epoch + 1}_iter{i + 1}.pth"
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'random_state': torch.get_rng_state(),
                    'numpy_random_state': np.random.get_state(),
                    'random_seed': 42,
                    'loss': loss
                }
                torch.save(checkpoint, save_path)
                print(f"Model saved to {save_path} at epoch {epoch + 1}, iteration {i + 1}")

        # --- Validation Loop ---
        val_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        print("start validation phase")
        with torch.no_grad():  # Disable gradient calculation during validation
            for i, data in enumerate(val_dataloader, 0):
                inputs, img_names = data
                inputs = inputs.to(device)

                # --- Encoder Forward Pass ---
                x1 = model.encoder_conv1(inputs)  # [B,128,256,256]
                x2 = model.encoder_conv2(x1)  # [B,256,128,128]
                x3 = model.encoder_conv3(x2)  # [B,512,64,64]
                x4 = model.encoder_conv4(x3)  # [B,1024,32,32]
                # x5 = model.encoder_conv5(x4)  # [B,2048,16,16]
                latent_space = model.encoder_fc(x4)  # [B,latent_size]  # [B, latent_size]  # Shape: [batch_size, 2048, 16, 16]

                # Latent Space

                # --- Quantization and Dequantization ---
                # latent_space_quantized = quantize(latent_space, quantization_levels)  # Quantize
                # latent_space_dequantized = decompress_latent_space(latent_space_quantized, quantization_levels)  # Dequantize
                latent_space_quantized = quantize(latent_space, 256)  # Quantize
                compressed_bytes = compress_latent_space(latent_space_quantized)
                print(len(compressed_bytes))
                latent_space_dequantized = decompress_latent_space(compressed_bytes, latent_space_quantized.shape,
                                                                   quantization_levels=256, device="cpu")
                # --- Decoder Forward Pass ---
                x = model.decoder_fc(latent_space)  # Shape: [batch_size, 512 × 16 × 16]
                x = x.view(x.size(0), 512, 16, 16)  # Reshape to [batch_size, 512, 16, 16]

                x = model.decoder_deconv1(x)  # Shape: [batch_size, 256, 32, 32]
                x4_resized = torch.nn.functional.interpolate(x4, size=x.shape[2:], mode='bilinear',
                                                             align_corners=False)  # Resize x4 to match x
                x = torch.cat([x, x4_resized],
                              dim=1)  # Skip connection from x4 (Shape: [batch_size, 512 + 256, 32, 32])

                x = model.decoder_deconv2(x)  # Shape: [batch_size, 128, 64, 64]
                x3_resized = torch.nn.functional.interpolate(x3, size=x.shape[2:], mode='bilinear',
                                                             align_corners=False)  # Resize x3 to match x
                x = torch.cat([x, x3_resized],
                              dim=1)  # Skip connection from x3 (Shape: [batch_size, 256 + 128, 64, 64])

                x = model.decoder_deconv3(x)  # Shape: [batch_size, 64, 128, 128]
                x2_resized = torch.nn.functional.interpolate(x2, size=x.shape[2:], mode='bilinear',
                                                             align_corners=False)  # Resize x2 to match x
                x = torch.cat([x, x2_resized],
                              dim=1)  # Skip connection from x2 (Shape: [batch_size, 128 + 64, 128, 128])

                reconstructed = model.decoder_deconv4(x)  # Shape: [batch_size, 3, 256, 256]
                # x2_resized = torch.nn.functional.interpolate(x2, size=x.shape[2:], mode='bilinear',
                #                                              align_corners=False)  # Resize x2 to match x
                # x = torch.cat([x, x2_resized],
                #               dim=1)  # Skip connection from x2 (Shape: [batch_size, 256 + 128, 256, 256])

                # reconstructed = model.decoder_deconv5(x)
                save_reconstructed_image(reconstructed, "reconstructed_train_valitaion_256", img_names,
                                         epoch)
                # Shape: [batch_size, 3, 512, 512]

                # --- Resize Reconstructed Output to Match Input Size (256x256) ---
                reconstructed_resized = torch.nn.functional.interpolate(reconstructed, size=(256, 256), mode='bilinear',
                                                                        align_corners=False)
                # inputs_resized = torch.nn.functional.interpolate(inputs, size=(512, 512), mode='bilinear',
                #                                                  align_corners=False)

                # --- Compute Loss ---
                loss = criterion(reconstructed_resized, inputs)
                val_loss += loss.item()
                # reconstructed = Image.fromarray(reconstructed)
                # reconstructed.save(f"{base_dir}/reconstructed_test_{i}.bmp")

        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}')
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'random_state': torch.get_rng_state(),
                'numpy_random_state': np.random.get_state(),
                'random_seed': 42
            }
            best_save_path = f"{base_dir}/{best_model_save_path}"
            torch.save(checkpoint, best_save_path)
            print(f"Best model saved to {best_save_path} with validation loss: {best_val_loss}")

        # if -avg_val_loss < -best_val_loss * 1.1:  # Stop if validation loss increases by 10%
        #     print("Early stopping triggered.")
        #     break

        # Update learning rate
        # scheduler.step()
        #
        # model.train()  # Set the model back to training mode

    return model


def save_reconstructed_image(reconstructed_batch, output_dir: str, image_names: list, epoch: int):
    bch_size = reconstructed_batch.size(0)
    if not os.path.exists(f"{base_dir}/{output_dir}"):
        os.mkdir(f"{base_dir}/{output_dir}")
    output_dir = f"{base_dir}/{output_dir}/epoch_{epoch}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for u in range(bch_size):
        file_path = f"{output_dir}/reconstructed_{image_names[u]}.jpg"
        save_image(reconstructed_batch[u], file_path)


def main(learning_rate, batch_size):
    # Device Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SimplerImprovedCNNAutoencoderWithSkip(512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # MODEL_PATH = f"{base_dir}/autoencoder_model_epoch_epoch1_iter1000.pth"
    # model, optimizer, scheduler = load_model(MODEL_PATH, model, optimizer, scheduler)
    # model.eval()
    # Loss Function and Optimizer
    # criterion = nn.MSELoss()

    # Data Loading
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Optional: Resize images to a
        # consistent size
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = YOLOv5Dataset(image_dir=f'{base_dir}/dataset/training/images',
                                  transform=transform)  # Replace with the actual path
    print(len(train_dataset.image_files))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True)

    val_dataset = YOLOv5Dataset(image_dir=f'{base_dir}/dataset/training/images',
                                transform=transform)  # Create a separate dataset for validation
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False)  # No need to shuffle validation data

    model = train(model, train_dataloader, val_dataloader, optimizer,
                  10, device, save_interval, quantization_levels=256,
                  best_model_save_path="best_model_lpips_psnr_loss_simple_256output.pth",
                  scheduler=scheduler)

    # train_mean, train_std = calculate_mean_std(train_dataloader)
    model.eval()
    # reconstruct(model, train_dataloader)


def set_seed():
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    base_dir = os.getcwd()
    learning_rate = 1e-3
    batch_size = 8
    set_seed()
    save_interval = 1000
    main(learning_rate, batch_size)
