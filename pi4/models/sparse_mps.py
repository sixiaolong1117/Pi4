"""Metal indexing kernels for the MPS sparse depth refiner."""

from functools import lru_cache

import torch


# Logical extent only. Sparse ops never allocate the depth axis. Keeping a fixed
# extent for nonnegative int32 depth bins avoids reading their maximum on the CPU.
DEPTH_EXTENT = 1 << 31


@lru_cache(maxsize=1)
def _kernels():
    return torch.mps.compile_shader(
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void encode_coordinates(
            const device int4* coordinates,
            device long* keys,
            constant long& batch_stride,
            constant long& row_stride,
            constant long& column_stride,
            uint i [[thread_position_in_grid]]
        ) {
            int4 voxel = coordinates[i];
            // Pad every spatial axis by one voxel so boundary offsets cannot wrap.
            keys[i] = long(voxel.x) * batch_stride
                + (long(voxel.y) + 1) * row_stride
                + (long(voxel.z) + 1) * column_stride + long(voxel.w) + 1;
        }

        kernel void find_neighbors(
            const device long* keys,
            const device long* sorted_keys,
            const device long* order,
            device int* output,
            constant int& voxel_count,
            constant long& row_stride,
            constant long& column_stride,
            uint i [[thread_position_in_grid]]
        ) {
            uint offset = i % 27;
            long key = keys[i / 27]
                + (int(offset / 9) - 1) * row_stride
                + (int(offset / 3 % 3) - 1) * column_stride
                + int(offset % 3) - 1;
            int lo = 0;
            int hi = voxel_count;
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (sorted_keys[mid] < key) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            output[i] = lo < voxel_count && sorted_keys[lo] == key
                ? int(order[lo]) : voxel_count;
        }

        kernel void image_neighbors(
            const device int4* coordinates,
            device packed_int3* output,
            constant int& voxel_count,
            constant int& height,
            constant int& width,
            uint i [[thread_position_in_grid]]
        ) {
            int4 voxel = coordinates[i / 9];
            uint offset = i % 9;
            int row = voxel.y + int(offset / 3) - 1;
            int column = voxel.z + int(offset % 3) - 1;
            int3 neighbors = int3(voxel_count);
            if (row >= 0 && row < height && column >= 0 && column < width) {
                int pixel = (voxel.x * height + row) * width + column;
                int delta = coordinates[pixel].w - voxel.w;
                if (delta >= -1 && delta <= 1) {
                    neighbors[delta + 1] = pixel;
                }
            }
            output[i] = packed_int3(neighbors);
        }

        kernel void gather_neighbors(
            const device float4* features,
            const device int* indices,
            device float4* output,
            constant int& channel_vectors,
            constant int& voxel_count,
            uint i [[thread_position_in_grid]]
        ) {
            int neighbor = indices[i / channel_vectors];
            output[i] = neighbor == voxel_count
                ? float4(0.0f)
                : features[uint(neighbor) * channel_vectors + i % channel_vectors];
        }
        """
    )


def neighbor_indices(coordinates, shape):
    _, height, width, depth, _ = shape
    column_stride = depth + 2
    row_stride = (width + 2) * column_stride
    batch_stride = (height + 2) * row_stride
    count = len(coordinates)
    keys = coordinates.new_empty(count, dtype=torch.int64)
    kernels = _kernels()
    kernels.encode_coordinates(
        coordinates.contiguous(), keys, batch_stride, row_stride, column_stride,
        threads=count, group_size=128,
    )
    sorted_keys, order = keys.sort()
    indices = coordinates.new_empty((count, 27))
    kernels.find_neighbors(
        keys, sorted_keys, order, indices, count, row_stride, column_stride,
        threads=indices.numel(), group_size=128, arg_casts={4: "int32"},
    )
    return indices


def image_neighbor_indices(coordinates, height, width):
    """Build neighbors for one active depth voxel per pixel in raster order."""
    count = len(coordinates)
    indices = coordinates.new_empty((count, 27))
    _kernels().image_neighbors(
        coordinates, indices, count, height, width,
        threads=count * 9, group_size=128, arg_casts="int32",
    )
    return indices


def gather_neighbors(features, indices, output):
    _kernels().gather_neighbors(
        features, indices, output, features.shape[1] // 4, len(features),
        threads=output.numel() // 4, group_size=256, arg_casts="int32",
    )
