import torch
from torch import nn
from torch.nn import functional as F

from .sparse import SubmanifoldConv3d, mean_pool, nearest_upsample
from .sparse_mps import DEPTH_EXTENT, image_neighbor_cache


class SparseResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        algorithm = "masked_implicit_gemm" if channels >= 128 else "implicit_gemm"
        self.conv1 = SubmanifoldConv3d(channels, algorithm)
        self.conv2 = SubmanifoldConv3d(channels, algorithm)

    def forward(self, features, coordinates, shape, cache=None, *, sorted_coords=False):
        residual = F.silu(self.norm1(features).type_as(features))
        if features.device.type == "mps":
            residual, cache = self.conv1(
                residual,
                coordinates,
                shape,
                cache,
                activation="silu",
                sorted_coords=sorted_coords,
            )
            return self.conv2(
                residual,
                coordinates,
                shape,
                cache,
                residual=features,
                sorted_coords=sorted_coords,
            )
        residual, cache = self.conv1(residual, coordinates, shape, cache)
        residual, cache = self.conv2(
            F.silu(residual), coordinates, shape, cache
        )
        return features + residual, cache


class SparseDown(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)

    def forward(self, features, coordinates, shape):
        output_shape = torch.Size(
            (shape[0], *((dimension + 1) // 2 for dimension in shape[1:4]), shape[4])
        )
        features, coordinates, shape, cache = mean_pool(
            features, coordinates, shape, output_shape
        )
        features = self.linear(features)
        return (
            features,
            coordinates,
            torch.Size((*shape[:-1], features.shape[-1])),
            cache,
        )


class SparseUp(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)

    def forward(
        self, features, coordinates, shape, target_coordinates, target_shape, cache
    ):
        features = self.linear(features)
        shape = torch.Size((*shape[:-1], features.shape[-1]))
        features = nearest_upsample(
            features, coordinates, shape, target_coordinates, target_shape, cache
        )
        return features, target_coordinates, target_shape


class SparseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (32, 64, 128, 256, 512)
        self.input_proj = nn.Linear(3, 32)
        self.down_stages = nn.ModuleList(
            nn.ModuleList((SparseResidual(dimension),)) for dimension in channels
        )
        self.downsample_blocks = nn.ModuleList(
            SparseDown(first, second)
            for first, second in zip(channels[:-1], channels[1:])
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(1024, 512), nn.SiLU(), nn.Linear(512, 512)
        )
        self.bottleneck_stage = nn.ModuleList((SparseResidual(512),))
        self.upsample_blocks = nn.ModuleList(
            SparseUp(first, second)
            for first, second in zip(channels[:0:-1], channels[-2::-1])
        )
        self.up_stages = nn.ModuleList(
            nn.ModuleList((SparseResidual(dimension),))
            for dimension in channels[-2::-1]
        )
        self.out_proj = nn.Linear(32, 1)

    def forward(self, features, coordinates, shape, condition, initial_neighbors=None):
        features = self.input_proj(features)
        shape = torch.Size((*shape[:4], features.shape[-1]))
        caches = [initial_neighbors, None, None, None, None]
        down_caches, skips = [], []
        for index, stage in enumerate(self.down_stages):
            features, caches[index] = stage[0](
                features,
                coordinates,
                shape,
                caches[index],
                sorted_coords=index > 0,
            )
            if index < 4:
                skips.append((features, coordinates, shape))
                features, coordinates, shape, cache = self.downsample_blocks[index](
                    features, coordinates, shape
                )
                down_caches.append(cache)
        sampled = condition[
            coordinates[:, 0].long(),
            :,
            coordinates[:, 1].long(),
            coordinates[:, 2].long(),
        ]
        features = self.fuse_proj(torch.cat((features, sampled), dim=-1))
        features, caches[4] = self.bottleneck_stage[0](
            features, coordinates, shape, caches[4], sorted_coords=True
        )
        for index, (upsample, stage) in enumerate(
            zip(self.upsample_blocks, self.up_stages)
        ):
            level = 3 - index
            skip_features, skip_coordinates, skip_shape = skips[level]
            features, coordinates, shape = upsample(
                features,
                coordinates,
                shape,
                skip_coordinates,
                skip_shape,
                down_caches[level],
            )
            features, caches[level] = stage[0](
                features + skip_features,
                coordinates,
                shape,
                caches[level],
                sorted_coords=level > 0,
            )
        return self.out_proj(features)


class DepthRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_projector = nn.Sequential(
            nn.Conv2d(1026, 1024, 1), nn.GELU(), nn.Conv2d(1024, 512, 1)
        )
        self.network = SparseUNet()

    def forward(self, rays, log_depth, feature):
        condition = self.feature_projector(feature.float())
        batch_size, height, width = log_depth.shape
        batch, rows, columns = torch.meshgrid(
            torch.arange(batch_size, device=rays.device),
            torch.arange(height, device=rays.device),
            torch.arange(width, device=rays.device),
            indexing="ij",
        )
        for _ in range(3):
            bins = torch.round(log_depth.float() * 256).long()
            bins = bins - bins.amin(dim=(1, 2), keepdim=True)
            coordinates = (
                torch.stack((batch, rows, columns, bins), dim=-1).reshape(-1, 4).int()
            )
            features = (
                torch.cat((rays, log_depth[..., None]), dim=-1).reshape(-1, 3).float()
            )
            initial_neighbors = None
            if rays.device.type == "mps":
                extent = DEPTH_EXTENT
                initial_neighbors = image_neighbor_cache(coordinates, height, width)
            else:
                extent = int(bins.max().item()) + 1
            shape = torch.Size((batch_size, height, width, extent, 3))
            residual = self.network(
                features, coordinates, shape, condition, initial_neighbors
            )
            log_depth = log_depth + residual.float().reshape_as(log_depth)
        depth = log_depth.exp()[..., None]
        return torch.cat((rays * depth, depth), dim=-1)
