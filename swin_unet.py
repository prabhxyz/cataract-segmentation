import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinUNet(nn.Module):
    """
    A 'Swin U‐Net' style architecture:
     - Swin Transformer Large as encoder (features_only=True),
     - upsample stages with skip connections for a UNet-like decoder,
     - final segmentation head with 'num_classes' output channels.
    """
    def __init__(self, num_classes=3, backbone="swin_large_patch4_window7_224", img_size=384):
        super(SwinUNet, self).__init__()

        # 1) Swin Large encoder
        self.encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=(0,1,2,3),  # 4 stages
            img_size=img_size
        )

        # 2) Get channel counts for each stage
        fea_info = self.encoder.feature_info
        self.out_channels = fea_info.channels()  # e.g. [192, 384, 768, 1536] for Swin-L

        # Because in the decoder we do:
        #   d3 = concat(up(f3), f2) => in_channels = out_channels[3] + out_channels[2]
        #   d2 = concat(up(d3), f1) => in_channels = out_channels[2] + out_channels[1]
        #   d1 = concat(up(d2), f0) => in_channels = out_channels[1] + out_channels[0]
        #   d0 = up(d1) => in_channels = out_channels[0] (no concat)
        self.dec3 = self._upsample_block(self.out_channels[3] + self.out_channels[2],
                                         self.out_channels[2])
        self.dec2 = self._upsample_block(self.out_channels[2] + self.out_channels[1],
                                         self.out_channels[1])
        self.dec1 = self._upsample_block(self.out_channels[1] + self.out_channels[0],
                                         self.out_channels[0])
        self.dec0 = self._upsample_block(self.out_channels[0], 128)  # final smaller block

        self.seg_head = nn.Conv2d(128, num_classes, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels):
        """A small decode block with two 3×3 conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: [B,3,H,W] (e.g. 384×384)
        Returns: [B,num_classes,H,W]
        """
        # 1) Encoder => 4 feature maps
        features = self.encoder(x)  # e.g. [f0, f1, f2, f3]

        # Potential channels-last => fix if needed:
        def fix_ch_last(t):
            if t.dim()==4 and t.shape[1] < t.shape[-1]:
                return t.permute(0,3,1,2).contiguous()
            return t
        f0, f1, f2, f3 = [fix_ch_last(f) for f in features]

        # 2) Decoder
        # dec3 => up(f3) + f2
        d3_up = F.interpolate(f3, size=(f2.shape[2], f2.shape[3]), mode='bilinear', align_corners=False)
        d3_cat= torch.cat([d3_up, f2], dim=1)  # => channels = 1536+768=2304
        d3    = self.dec3(d3_cat)              # => out_channels=768

        # dec2 => up(d3) + f1
        d2_up = F.interpolate(d3, size=(f1.shape[2], f1.shape[3]), mode='bilinear', align_corners=False)
        d2_cat= torch.cat([d2_up, f1], dim=1)  # => channels = 768+384=1152
        d2    = self.dec2(d2_cat)              # => out_channels=384

        # dec1 => up(d2) + f0
        d1_up = F.interpolate(d2, size=(f0.shape[2], f0.shape[3]), mode='bilinear', align_corners=False)
        d1_cat= torch.cat([d1_up, f0], dim=1)  # => channels=384+192=576
        d1    = self.dec1(d1_cat)              # => out_channels=192

        # dec0 => up(d1)
        d0_up = F.interpolate(d1, scale_factor=2.0, mode='bilinear', align_corners=False)
        d0    = self.dec0(d0_up)               # in_channels=192, out_channels=128

        # final up => match input
        out   = F.interpolate(d0, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        logits = self.seg_head(out)  # => [B,num_classes,H,W]
        return logits


if __name__=="__main__":
    # Simple test
    model = SwinUNet(num_classes=3, backbone="swin_large_patch4_window7_224", img_size=384)
    x = torch.randn(2,3,384,384)
    y = model(x)
    print("Output shape:", y.shape)  # => [2,3,384,384]
