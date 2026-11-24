"""Sparse operations for the depth refiner on CUDA and Apple Silicon."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import sparse_mps


_GATHER_BLOCK_BYTES = 32 * 1024 * 1024


def _neighbor_indices(coordinates, shape):
    _, height, width, depth, _ = shape
    # A one-voxel border keeps out-of-bounds offsets from wrapping to an active key.
    column_stride = depth + 2
    row_stride = (width + 2) * column_stride
    batch_stride = (height + 2) * row_stride
    strides = coordinates.new_tensor(
        (batch_stride, row_stride, column_stride, 1), dtype=torch.int64
    )
    padded = coordinates.long() + coordinates.new_tensor((0, 1, 1, 1))
    keys = (padded * strides).sum(dim=1)
    sorted_keys, order = keys.sort()

    offsets = torch.arange(-1, 2, device=coordinates.device, dtype=torch.int64)
    offsets = (
        offsets[:, None, None] * row_stride
        + offsets[None, :, None] * column_stride
        + offsets[None, None, :]
    ).flatten()
    queries = keys[:, None] + offsets
    matched = torch.searchsorted(sorted_keys, queries, out_int32=True)
    matched.clamp_max_(len(keys) - 1)
    return torch.where(sorted_keys[matched] == queries, order.int()[matched], len(keys))


class SubmanifoldConv3d(nn.Module):
    def __init__(self, channels, algorithm):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 3, 3, 3, channels))
        self.bias = nn.Parameter(torch.empty(channels))
        self.algorithm = algorithm
        bound = 1 / math.sqrt(27 * channels)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        features,
        coordinates,
        shape,
        cache,
        *,
        activation=None,
        residual=None,
        sorted_coords=False,
    ):
        if features.is_cuda:
            from flex_gemm.ops.spconv.submanifold_conv import submanifold_conv3d

            output, cache = submanifold_conv3d(
                features,
                coordinates,
                shape,
                self.weight,
                self.bias,
                neighbor_cache=cache,
                algorithm=self.algorithm,
            )
            if activation == "silu":
                output = F.silu(output)
            elif residual is not None:
                output = output + residual
            return output, cache

        if len(features) == 0:
            return features.clone(), cache

        if (
            features.device.type == "mps"
            and features.dtype == torch.float32
            and features.shape[1] % 32 == 0
            and (cache is None or isinstance(cache, sparse_mps.MPSNeighborCache))
        ):
            if cache is None:
                cache = sparse_mps.neighbor_cache(
                    coordinates, shape, sorted_coords=sorted_coords
                )
            output = sparse_mps.submanifold_conv3d(
                features,
                self.weight,
                self.bias,
                cache,
                activation=activation,
                residual=residual,
            )
            return output, cache

        if cache is None:
            cache = _neighbor_indices(coordinates, shape)
        use_metal_gather = (
            features.device.type == "mps"
            and features.dtype == torch.float32
            and features.shape[1] % 4 == 0
        )
        weights = self.weight.flatten(1)
        weights_t = weights.T
        bytes_per_row = weights.shape[1] * features.element_size()
        block_rows = max(1, _GATHER_BLOCK_BYTES // bytes_per_row)
        if use_metal_gather:
            source = features.contiguous()
            workspace = features.new_empty(
                (min(len(features), block_rows), weights.shape[1])
            )
        else:
            source = F.pad(features, (0, 0, 0, 1))
        output = torch.empty_like(features)
        for start in range(0, len(features), block_rows):
            stop = min(start + block_rows, len(features))
            indices = cache[start:stop].reshape(-1)
            if use_metal_gather:
                neighbors = workspace[: stop - start]
                sparse_mps.gather_neighbors(source, indices, neighbors)
            else:
                neighbors = source.index_select(0, indices).reshape(stop - start, -1)
            torch.addmm(self.bias, neighbors, weights_t, out=output[start:stop])
        if activation == "silu":
            output = F.silu(output)
        elif residual is not None:
            output = output + residual
        return output, cache


def mean_pool(features, coordinates, shape, output_shape):
    if features.is_cuda:
        from flex_gemm.ops.pool.sparse_pool import sparse_pool

        return sparse_pool(
            features,
            coordinates,
            shape,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            reduce="mean",
            output_shape=output_shape,
        )

    if (
        features.device.type == "mps"
        and features.dtype == torch.float32
        and features.shape[1] % 4 == 0
        and len(coordinates) > 0
    ):
        return sparse_mps.mean_pool(features, coordinates, output_shape)

    _, height, width, depth, _ = output_shape
    batch = coordinates[:, 0].long()
    row, column, level = (coordinates[:, 1:].long() // 2).unbind(1)
    keys = ((batch * height + row) * width + column) * depth + level
    keys, parents, counts = torch.unique(
        keys, sorted=True, return_inverse=True, return_counts=True
    )
    pooled = features.new_zeros((len(keys), features.shape[1]))
    pooled.index_add_(0, parents, features)
    pooled.div_(counts[:, None])
    level = keys % depth
    keys = keys // depth
    column = keys % width
    keys = keys // width
    output_coords = torch.stack((keys // height, keys % height, column, level), dim=1)
    return pooled, output_coords.int(), output_shape, parents


def nearest_upsample(features, coordinates, shape, target_coords, target_shape, cache):
    if features.is_cuda:
        from flex_gemm.ops.sample.upsample import sparse_upsample

        output, _, _, _ = sparse_upsample(
            features,
            coordinates,
            shape,
            scale_factor=(2, 2, 2),
            mode="nearest",
            output_coords=target_coords,
            output_shape=target_shape,
            neighbor_cache=cache.T,
        )
        return output
    return features.index_select(0, cache)
