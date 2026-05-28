import torch.nn as nn
import torch.nn.functional as F

from ..registry import NECKS


class _CausalTemporalBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=1,
        num_groups=8,
        dropout=0.0,
        zero_init=True,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if channels % num_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_groups={num_groups}"
            )
        self.left_padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if zero_init:
            nn.init.zeros_(self.pointwise.weight)
            nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        residual = x
        x = F.pad(x, (self.left_padding, 0))
        x = self.depthwise(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = self.pointwise(x)
        x = self.dropout(x)
        return residual + x


@NECKS.register_module
class CausalFeatureTCN(nn.Module):
    """Causal temporal neck for per-frame 3D radar features.

    Input shape:  (B, T, C, Z, Y, X)
    Output shape: (B, C, Z, Y, X)

    The TCN is applied independently at every spatial voxel over the temporal
    feature sequence. With zero-initialized residual blocks, the neck initially
    behaves like "use the current frame feature", which makes fine-tuning from a
    per-frame checkpoint stable.
    """

    def __init__(
        self,
        in_channels,
        kernel_size=3,
        dilations=(1, 2),
        num_groups=8,
        dropout=0.0,
        zero_init_residual=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.blocks = nn.ModuleList(
            [
                _CausalTemporalBlock(
                    channels=in_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    num_groups=num_groups,
                    dropout=dropout,
                    zero_init=zero_init_residual,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x):
        if x.dim() != 6:
            raise ValueError(
                f"CausalFeatureTCN expects (B,T,C,Z,Y,X), got shape {tuple(x.shape)}"
            )
        batch, time, channels, z, y, x_size = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {channels}"
            )

        # Treat each voxel as one temporal sequence: (B*Z*Y*X, C, T).
        x = x.permute(0, 3, 4, 5, 2, 1).contiguous()
        x = x.view(batch * z * y * x_size, channels, time)
        for block in self.blocks:
            x = block(x)

        # Causal prediction for the current frame uses only the last timestep.
        x = x[:, :, -1]
        x = x.view(batch, z, y, x_size, channels)
        return x.permute(0, 4, 1, 2, 3).contiguous()
