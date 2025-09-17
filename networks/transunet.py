# transunet.py
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=scale_factor,
            stride=scale_factor,
        )

    def forward(self, x):
        return self.up(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        attn_out = self.self_attn(src, src, src)[0]
        src = self.norm1(src + self.dropout1(attn_out))
        ff = self.linear2(self.activation(self.linear1(src)))
        src = self.norm2(src + self.dropout2(ff))
        return src


class TransUNet3D(nn.Module):
    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        num_classes=14,
        base_channels=32,
        embed_dim=256,
        mlp_dim=1024,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = ConvBlock(in_channels, base_channels, stride=1)
        self.conv2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        self.conv5 = ConvBlock(base_channels * 8, base_channels * 16, stride=2)

        self.embed_conv = nn.Conv3d(base_channels * 16, embed_dim, kernel_size=1)
        self.flatten = nn.Flatten(2)
        num_patches = (img_size[0] // 16) * (img_size[1] // 16) * (img_size[2] // 16)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )

        self.upconv4 = UpConv(embed_dim, base_channels * 8, scale_factor=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.upconv3 = UpConv(base_channels * 8, base_channels * 4, scale_factor=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.upconv2 = UpConv(base_channels * 4, base_channels * 2, scale_factor=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.upconv1 = UpConv(base_channels * 2, base_channels, scale_factor=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        x_embed = self.embed_conv(c5)
        b, c, d, h, w = x_embed.shape
        x_flat = self.flatten(x_embed).permute(2, 0, 1)  # (N, B, C)
        x_flat = x_flat + self.position_embeddings.permute(1, 0, 2)

        for block in self.transformer_blocks:
            x_flat = block(x_flat)

        x_out = x_flat.permute(1, 2, 0).view(b, c, d, h, w)

        d4 = self.upconv4(x_out)
        d4 = self.dec4(torch.cat((d4, c4), dim=1))

        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat((d3, c3), dim=1))

        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat((d2, c2), dim=1))

        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat((d1, c1), dim=1))

        return self.final_conv(d1)
