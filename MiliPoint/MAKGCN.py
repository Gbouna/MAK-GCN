import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_knn_indices(x, k):
    """
    Compute the k-nearest neighbors for each point in x.

    Args:
        x (torch.Tensor): Shape (B, C, N).
        k (int): Number of neighbors.

    Returns:
        torch.Tensor: The indices of the top-k neighbors, shape (B, N, k).
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx

def get_graph_feature(x, k=10, idx=None):
    """
    Compute the graph feature for input x.
    
    Args:
        x (torch.Tensor): Input of shape (B, C, N).
        k (int, optional): Number of nearest neighbors. Default is 10.
        idx (torch.Tensor, optional): Precomputed adjacency indices. If None, they will be computed.

    Returns:
        tuple: (feature, idx)
            - feature (torch.Tensor): Graph feature of shape (B, 2*C, N, k).
            - idx (torch.Tensor): The adjacency indices used (B*N*k).
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = compute_knn_indices(x, k=k)
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (B*N*k, C)
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # concat feature-x and x
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx

class MAK(nn.Module):
    """
    Multi-Head Adaptive Kernel (MAK) layer.

    This layer generates adaptive filters from 'y' and applies them to 'x'
    across multiple heads, then combines the outputs.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        feat_channels,
        use_residual=True,
        dropout=0.0,
        num_heads=None
    ):
        super(MAK, self).__init__()
        if num_heads is None:
            raise ValueError("'num_heads' must be specified.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.use_residual = use_residual
        self.dropout = dropout
        self.num_heads = num_heads
        if self.use_residual and (in_channels != out_channels):
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
        else:
            self.shortcut_conv = None

        # -- Deeper MLP for generating "base" adaptive kernel
        self.conv0 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False)
        self.bn0   = nn.BatchNorm2d(out_channels)
        self.conv_mid = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_mid   = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            out_channels,
            out_channels * in_channels * self.num_heads,
            kernel_size=1,
            bias=False
        )
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        """
        Forward pass for MAK.

        Args:
            x (torch.Tensor): Input features of shape (B, in_channels, N, k).
            y (torch.Tensor): Feature tensor used for generating kernels, shape (B, feat_channels, N, k).

        Returns:
            torch.Tensor: Output features of shape (B, out_channels, N, k).
        """
        identity = x
        B, _, N, K = x.shape

        # Create adaptive kernel weights from y
        y = self.conv0(y)                     # (B, out_channels, N, K)
        y = self.leaky_relu(self.bn0(y))
        y = self.conv_mid(y)                  # (B, out_channels, N, K)
        y = self.leaky_relu(self.bn_mid(y))

        # Generate (out_channels * in_channels * num_heads) => (B, out_channels*in_channels*num_heads, N, K)
        y = self.conv1(y)

        # Reshape for multi-head
        y = y.permute(0, 2, 3, 1).contiguous() # (B, N, K, out_channels*in_channels*num_heads)
        y = y.view(
            B, N, K, self.out_channels, self.in_channels, self.num_heads
        )

         # Rearrange x for multiplication: x => (B, N, K, in_channels, 1)
        x = x.permute(0, 2, 3, 1).unsqueeze(4)  

        # Apply each head's filter
        head_outputs = []
        for h in range(self.num_heads):
            w_h = y[..., h]  # (B, N, K, out_channels, in_channels)
            out_h = torch.matmul(w_h, x).squeeze(4)  # (B, N, K, out_channels)
            head_outputs.append(out_h)

        # Sum across heads => (B, N, K, out_channels)
        out = torch.stack(head_outputs, dim=-1).sum(dim=-1)

        # Re-permute => (B, out_channels, N, K)
        out = out.permute(0, 3, 1, 2).contiguous()
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        out = self.bn_out(out)
        if self.use_residual:
            if self.shortcut_conv is not None:
                identity = self.shortcut_conv(identity)
                identity = self.shortcut_bn(identity)
            out = out + identity
        out = self.leaky_relu(out)
        return out
    
class Net(nn.Module):
    def __init__(self, args, output_channels=49):
        super(Net, self).__init__()
        self.args = args
        self.k = args.k
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm1d(args.emb_dims)

        self.adapt_conv1 = MAK(
            in_channels=6, 
            out_channels=64, 
            feat_channels=6,
            use_residual=True,
            dropout=args.dropout * 0.5,
            num_heads=args.num_heads
        )
        self.adapt_conv2 = MAK(
            in_channels=6, 
            out_channels=64, 
            feat_channels=128,
            use_residual=True,
            dropout=args.dropout * 0.5,
            num_heads=args.num_heads
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        points = x

        # Compute adjacency once
        _, common_idx = get_graph_feature(points, k=self.k)  # store common_idx

        # First layer
        feat_x, _ = get_graph_feature(x, k=self.k, idx=common_idx)
        geo_x, _ = get_graph_feature(points, k=self.k, idx=common_idx)
        x = self.adapt_conv1(geo_x, feat_x)  # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0] # (B, 64, N)

        # Second layer
        feat_x, _ = get_graph_feature(x1, k=self.k, idx=common_idx)
        geo_x, _ = get_graph_feature(points, k=self.k, idx=common_idx)
        x = self.adapt_conv2(geo_x, feat_x)  # (B, 64, N, k)
        x2 = x.max(dim=-1, keepdim=False)[0] # (B, 64, N)

        # Third layer
        x3_feat, _ = get_graph_feature(x2, k=self.k, idx=common_idx)
        x3 = self.conv3(x3_feat)            # (B, 128, N, k)
        x3 = x3.max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        # Fourth layer
        x4_feat, _ = get_graph_feature(x3, k=self.k, idx=common_idx)
        x4 = self.conv4(x4_feat)            # (B, 256, N, k)
        x4 = x4.max(dim=-1, keepdim=False)[0]  # (B, 256, N)

        # Combine local features
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 64+64+128+256=512, N)
        x = self.conv5(x)                      # (B, emb_dims, N)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)             # (B, emb_dims*2)
        x = F.leaky_relu(self.bn4(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn5(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
