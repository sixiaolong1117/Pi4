"""Metal Masked Implicit GEMM, neighbor caching, and sparse pooling kernels for MPS."""

from functools import lru_cache
from typing import NamedTuple

import torch


# Logical extent only. Sparse ops never allocate the depth axis. Keeping a fixed
# extent for nonnegative int32 depth bins avoids reading their maximum on the CPU.
DEPTH_EXTENT = 1 << 31

_TILE_M = 32


class MPSNeighborCache(NamedTuple):
    """Bitmask-sorted tile neighbor layout for Masked Implicit GEMM."""

    sorted_idx: torch.Tensor
    tile_neighbors: torch.Tensor
    tile_masks: torch.Tensor
    num_tiles: int
    voxel_count: int


@lru_cache(maxsize=1)
def _kernels():
    return torch.mps.compile_shader(
        """
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;

        // -----------------------------------------------------------------
        // 1. Level-0 image-grid O(1) bitmask + direct tile cache builder
        //    (Eliminates sorting/hashing and the intermediate [N, 27] map.)
        // -----------------------------------------------------------------
        kernel void image_bitmasks(
            const device int4* coordinates,
            device int* sort_keys,
            device uint* raw_masks,
            constant int& count,
            constant int& height,
            constant int& width,
            uint i [[thread_position_in_grid]]
        ) {
            if (int(i) >= count) return;
            int4 voxel = coordinates[i];
            uint mask = 0u;
            int base_b = voxel.x * height;
            for (int dy = -1; dy <= 1; ++dy) {
                int r = voxel.y + dy;
                if (r < 0 || r >= height) continue;
                int row_off = (base_b + r) * width;
                for (int dx = -1; dx <= 1; ++dx) {
                    int c = voxel.z + dx;
                    if (c < 0 || c >= width) continue;
                    int pixel = row_off + c;
                    int delta = coordinates[pixel].w - voxel.w;
                    if (delta >= -1 && delta <= 1) {
                        int k = ((dy + 1) * 3 + (dx + 1)) * 3 + (delta + 1);
                        mask |= (1u << k);
                    }
                }
            }
            raw_masks[i] = mask;
            sort_keys[i] = int(mask ^ (mask >> 1));
        }

        kernel void build_image_tile_cache_bm32(
            const device int4* coordinates,
            const device long* sorted_order,
            const device uint* raw_masks,
            device int* sorted_idx_out,
            device int* tile_neighbors,
            device uint* tile_masks,
            constant int& count,
            constant int& height,
            constant int& width,
            uint tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]]
        ) {
            int m = int(tgid) * 32 + int(lane);
            int vox = (m < count) ? int(sorted_order[m]) : -1;
            sorted_idx_out[m] = vox;
            uint mask = (vox >= 0) ? raw_masks[vox] : 0u;
            uint tmask = simd_or(mask);
            if (lane == 0) {
                tile_masks[tgid] = tmask;
            }
            device int* dst = tile_neighbors + ulong(tgid) * (27 * 32) + lane;
            uint mm = tmask;
            while (mm != 0u) {
                int k = ctz(mm);
                mm &= (mm - 1u);
                dst[k * 32] = -1;
            }
            if (vox < 0) return;
            int4 voxel = coordinates[vox];
            int base_b = voxel.x * height;
            for (int dy = -1; dy <= 1; ++dy) {
                int r = voxel.y + dy;
                if (r < 0 || r >= height) continue;
                int row_off = (base_b + r) * width;
                for (int dx = -1; dx <= 1; ++dx) {
                    int c = voxel.z + dx;
                    if (c < 0 || c >= width) continue;
                    int pixel = row_off + c;
                    int delta = coordinates[pixel].w - voxel.w;
                    if (delta >= -1 && delta <= 1) {
                        int k = ((dy + 1) * 3 + (dx + 1)) * 3 + (delta + 1);
                        dst[k * 32] = pixel;
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // 2. General sparse 3D neighbor search (8 spatial binary searches +
        //    adjacent depth check + fused 27-bit Gray-code mask)
        // -----------------------------------------------------------------
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

        kernel void find_neighbors_and_masks(
            const device long* keys,
            const device long* sorted_keys,
            const device long* order,
            device int* neighbors,
            device int* sort_keys,
            device uint* raw_masks,
            constant int& voxel_count,
            constant long& row_stride,
            constant long& column_stride,
            constant int& identity_order,
            uint i [[thread_position_in_grid]]
        ) {
            if (int(i) >= voxel_count) return;
            long base_key = keys[i];
            device int* out = neighbors + ulong(i) * 27;
            uint mask = 0u;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int k_base = ((dy + 1) * 3 + (dx + 1)) * 3;
                    int n0 = -1, n1 = -1, n2 = -1;
                    if (dy == 0 && dx == 0 && identity_order != 0) {
                        n1 = int(i);
                        if (int(i) > 0 && sorted_keys[i - 1] == base_key - 1) {
                            n0 = int(i) - 1;
                        }
                        if (int(i) + 1 < voxel_count && sorted_keys[i + 1] == base_key + 1) {
                            n2 = int(i) + 1;
                        }
                    } else {
                        long target_min = base_key + dy * row_stride + dx * column_stride - 1;
                        int lo = 0;
                        int hi = voxel_count;
                        while (lo < hi) {
                            int mid = (lo + hi) >> 1;
                            if (sorted_keys[mid] < target_min) {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        for (int step = 0; step < 3 && lo + step < voxel_count; ++step) {
                            long diff = sorted_keys[lo + step] - target_min;
                            if (diff > 2) break;
                            int orig = (identity_order != 0) ? (lo + step) : int(order[lo + step]);
                            if (diff == 0) n0 = orig;
                            else if (diff == 1) n1 = orig;
                            else n2 = orig;
                        }
                    }
                    out[k_base + 0] = n0;
                    out[k_base + 1] = n1;
                    out[k_base + 2] = n2;
                    if (n0 >= 0) mask |= (1u << (k_base + 0));
                    if (n1 >= 0) mask |= (1u << (k_base + 1));
                    if (n2 >= 0) mask |= (1u << (k_base + 2));
                }
            }
            raw_masks[i] = mask;
            sort_keys[i] = int(mask ^ (mask >> 1));
        }

        kernel void build_tile_cache_bm32(
            const device int* neighbors,
            const device long* sorted_order,
            const device uint* raw_masks,
            device int* sorted_idx_out,
            device int* tile_neighbors,
            device uint* tile_masks,
            constant int& voxel_count,
            uint tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]]
        ) {
            int m = int(tgid) * 32 + int(lane);
            int vox = (m < voxel_count) ? int(sorted_order[m]) : -1;
            sorted_idx_out[m] = vox;
            uint mask = (vox >= 0) ? raw_masks[vox] : 0u;
            uint tmask = simd_or(mask);
            if (lane == 0) {
                tile_masks[tgid] = tmask;
            }
            device int* dst = tile_neighbors + ulong(tgid) * (27 * 32) + lane;
            uint mm = tmask;
            if (vox >= 0) {
                const device int* src = neighbors + ulong(vox) * 27;
                while (mm != 0u) {
                    int k = ctz(mm);
                    mm &= (mm - 1u);
                    dst[k * 32] = src[k];
                }
            } else {
                while (mm != 0u) {
                    int k = ctz(mm);
                    mm &= (mm - 1u);
                    dst[k * 32] = -1;
                }
            }
        }

        // -----------------------------------------------------------------
        // 3. Masked Implicit GEMM for 3D Submanifold Convolution
        //    - Uses simdgroup_float8x8 hardware matrix instructions
        //    - Skips empty 3x3x3 offsets via ctz(tile_mask)
        //    - Reads weights directly from [C, 27, C] (zero weight copy)
        //    - Grid swizzled: tgid.x = tile_m, tgid.y = tile_n for L2 weight reuse
        //    - Fused epilogue: act_mode 0 = bias, 1 = bias + SiLU, 2 = bias + residual
        // -----------------------------------------------------------------
        kernel void spconv_bm32_bn32_bk32_2sg(
            const device float* features,
            const device float* weight,
            const device float* bias,
            const device float* residual_in,
            const device int* sorted_idx,
            const device int* tile_neighbors,
            const device uint* tile_masks,
            device float* output,
            constant int& N,
            constant int& C,
            constant int& act_mode,
            uint3 tgid [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint sg_id [[simdgroup_index_in_threadgroup]]
        ) {
            constexpr int BM = 32, BN = 32, BK = 32;
            threadgroup float As[BM * BK];
            threadgroup float Bs[BN * BK];

            int tile_m = int(tgid.x);
            int col_base = int(tgid.y) * BN;
            int sg_m = int(sg_id) * 16;

            simdgroup_float8x8 acc[2][4];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                    acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            int r0 = int(tid >> 3);        // 0..7
            int c4 = int((tid & 7) << 2);  // 0,4..28
            ulong c_u = ulong(C);
            ulong w_n_stride = 27ul * c_u;

            uint mask = tile_masks[tile_m];
            const device int* tile_nb_base = tile_neighbors + ulong(tile_m) * (27 * BM);

            while (mask != 0u) {
                int k_off = ctz(mask);
                mask &= (mask - 1u);

                const device int* nb_ptr = tile_nb_base + k_off * BM;
                int nb0 = nb_ptr[r0];
                int nb1 = nb_ptr[r0 + 8];
                int nb2 = nb_ptr[r0 + 16];
                int nb3 = nb_ptr[r0 + 24];
                const device float* w_base = weight
                    + ulong(col_base + r0) * w_n_stride
                    + ulong(k_off) * c_u + c4;

                for (int k0 = 0; k0 < C; k0 += BK) {
                    *reinterpret_cast<threadgroup float4*>(As + (r0 +  0) * BK + c4) =
                        (nb0 >= 0) ? *reinterpret_cast<const device float4*>(features + ulong(nb0) * c_u + k0 + c4) : float4(0.0f);
                    *reinterpret_cast<threadgroup float4*>(As + (r0 +  8) * BK + c4) =
                        (nb1 >= 0) ? *reinterpret_cast<const device float4*>(features + ulong(nb1) * c_u + k0 + c4) : float4(0.0f);
                    *reinterpret_cast<threadgroup float4*>(As + (r0 + 16) * BK + c4) =
                        (nb2 >= 0) ? *reinterpret_cast<const device float4*>(features + ulong(nb2) * c_u + k0 + c4) : float4(0.0f);
                    *reinterpret_cast<threadgroup float4*>(As + (r0 + 24) * BK + c4) =
                        (nb3 >= 0) ? *reinterpret_cast<const device float4*>(features + ulong(nb3) * c_u + k0 + c4) : float4(0.0f);

                    const device float* wk = w_base + k0;
                    *reinterpret_cast<threadgroup float4*>(Bs + (r0 +  0) * BK + c4) = *reinterpret_cast<const device float4*>(wk);
                    *reinterpret_cast<threadgroup float4*>(Bs + (r0 +  8) * BK + c4) = *reinterpret_cast<const device float4*>(wk + 8ul * w_n_stride);
                    *reinterpret_cast<threadgroup float4*>(Bs + (r0 + 16) * BK + c4) = *reinterpret_cast<const device float4*>(wk + 16ul * w_n_stride);
                    *reinterpret_cast<threadgroup float4*>(Bs + (r0 + 24) * BK + c4) = *reinterpret_cast<const device float4*>(wk + 24ul * w_n_stride);

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    #pragma unroll
                    for (int kk = 0; kk < BK; kk += 8) {
                        simdgroup_float8x8 af[2], bf[4];
                        simdgroup_load(af[0], As + (sg_m + 0) * BK + kk, BK);
                        simdgroup_load(af[1], As + (sg_m + 8) * BK + kk, BK);
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            simdgroup_load(bf[j], Bs + (j * 8) * BK + kk, BK, 0, true);
                        }
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            simdgroup_multiply_accumulate(acc[0][j], af[0], bf[j], acc[0][j]);
                            simdgroup_multiply_accumulate(acc[1][j], af[1], bf[j], acc[1][j]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                    simdgroup_store(acc[i][j], As + (sg_m + i * 8) * BN + j * 8, BN);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float4 b4 = *reinterpret_cast<const device float4*>(bias + col_base + c4);
            const device int* s_ptr = sorted_idx + ulong(tile_m) * BM;
            #pragma unroll
            for (int s = 0; s < 32; s += 8) {
                int out_v = s_ptr[r0 + s];
                if (out_v >= 0) {
                    float4 v = *reinterpret_cast<threadgroup float4*>(As + (r0 + s) * BN + c4) + b4;
                    if (act_mode == 1) {
                        v = v / (1.0f + exp(-v));
                    } else if (act_mode == 2) {
                        v += *reinterpret_cast<const device float4*>(residual_in + ulong(out_v) * c_u + col_base + c4);
                    }
                    *reinterpret_cast<device float4*>(output + ulong(out_v) * c_u + col_base + c4) = v;
                }
            }
        }

        kernel void spconv_bm32_bn64_bk32_4sg(
            const device float* features,
            const device float* weight,
            const device float* bias,
            const device float* residual_in,
            const device int* sorted_idx,
            const device int* tile_neighbors,
            const device uint* tile_masks,
            device float* output,
            constant int& N,
            constant int& C,
            constant int& act_mode,
            uint3 tgid [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint sg_id [[simdgroup_index_in_threadgroup]]
        ) {
            constexpr int BM = 32, BN = 64, BK = 32;
            threadgroup float As[BM * BK];
            threadgroup float Bs[BN * BK];

            int tile_m = int(tgid.x);
            int col_base = int(tgid.y) * BN;
            int sg_m = int(sg_id >> 1) * 16;
            int sg_n = int(sg_id & 1) * 32;

            simdgroup_float8x8 acc[2][4];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                    acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            int r0 = int(tid >> 3);        // 0..15
            int c4 = int((tid & 7) << 2);  // 0,4..28
            ulong c_u = ulong(C);
            ulong w_n_stride = 27ul * c_u;

            uint mask = tile_masks[tile_m];
            const device int* tile_nb_base = tile_neighbors + ulong(tile_m) * (27 * BM);

            while (mask != 0u) {
                int k_off = ctz(mask);
                mask &= (mask - 1u);

                const device int* nb_ptr = tile_nb_base + k_off * BM;
                int nb0 = nb_ptr[r0];
                int nb1 = nb_ptr[r0 + 16];
                const device float* w_base = weight
                    + ulong(col_base + r0) * w_n_stride
                    + ulong(k_off) * c_u + c4;

                for (int k0 = 0; k0 < C; k0 += BK) {
                    *reinterpret_cast<threadgroup float4*>(As + (r0 +  0) * BK + c4) =
                        (nb0 >= 0) ? *reinterpret_cast<const device float4*>(features + ulong(nb0) * c_u + k0 + c4) : float4(0.0f);
                    *reinterpret_cast<threadgroup float4*>(As + (r0 + 16) * BK + c4) =
                        (nb1 >= 0) ? *reinterpret_cast<const device float4*>(features + ulong(nb1) * c_u + k0 + c4) : float4(0.0f);

                    const device float* wk = w_base + k0;
                    *reinterpret_cast<threadgroup float4*>(Bs + (r0 +  0) * BK + c4) = *reinterpret_cast<const device float4*>(wk);
                    *reinterpret_cast<threadgroup float4*>(Bs + (r0 + 16) * BK + c4) = *reinterpret_cast<const device float4*>(wk + 16ul * w_n_stride);
                    *reinterpret_cast<threadgroup float4*>(Bs + (r0 + 32) * BK + c4) = *reinterpret_cast<const device float4*>(wk + 32ul * w_n_stride);
                    *reinterpret_cast<threadgroup float4*>(Bs + (r0 + 48) * BK + c4) = *reinterpret_cast<const device float4*>(wk + 48ul * w_n_stride);

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    #pragma unroll
                    for (int kk = 0; kk < BK; kk += 8) {
                        simdgroup_float8x8 af[2], bf[4];
                        simdgroup_load(af[0], As + (sg_m + 0) * BK + kk, BK);
                        simdgroup_load(af[1], As + (sg_m + 8) * BK + kk, BK);
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            simdgroup_load(bf[j], Bs + (sg_n + j * 8) * BK + kk, BK, 0, true);
                        }
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            simdgroup_multiply_accumulate(acc[0][j], af[0], bf[j], acc[0][j]);
                            simdgroup_multiply_accumulate(acc[1][j], af[1], bf[j], acc[1][j]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // Reuse Bs (8 KB = 32x64 floats) as epilogue staging buffer.
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                    simdgroup_store(acc[i][j], Bs + (sg_m + i * 8) * BN + sg_n + j * 8, BN);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int br = int(tid >> 4);         // 0..7
            int bc4 = int((tid & 15) << 2); // 0,4..60
            float4 b4 = *reinterpret_cast<const device float4*>(bias + col_base + bc4);
            const device int* s_ptr = sorted_idx + ulong(tile_m) * BM;
            #pragma unroll
            for (int s = 0; s < 32; s += 8) {
                int out_v = s_ptr[br + s];
                if (out_v >= 0) {
                    float4 v = *reinterpret_cast<threadgroup float4*>(Bs + (br + s) * BN + bc4) + b4;
                    if (act_mode == 1) {
                        v = v / (1.0f + exp(-v));
                    } else if (act_mode == 2) {
                        v += *reinterpret_cast<const device float4*>(residual_in + ulong(out_v) * c_u + col_base + bc4);
                    }
                    *reinterpret_cast<device float4*>(output + ulong(out_v) * c_u + col_base + bc4) = v;
                }
            }
        }

        // -----------------------------------------------------------------
        // 4. Atomic-free fused sparse 2x2x2 mean pooling
        // -----------------------------------------------------------------
        kernel void encode_pool_keys(
            const device int4* coordinates,
            device long* keys,
            constant long& height,
            constant long& width,
            constant long& depth,
            uint i [[thread_position_in_grid]]
        ) {
            int4 v = coordinates[i];
            long r = long(v.y >> 1);
            long c = long(v.z >> 1);
            long l = long(v.w >> 1);
            keys[i] = ((long(v.x) * height + r) * width + c) * depth + l;
        }

        kernel void mark_unique_runs(
            const device long* sorted_keys,
            device int* is_new,
            uint i [[thread_position_in_grid]]
        ) {
            is_new[i] = (i == 0 || sorted_keys[i] != sorted_keys[i - 1]) ? 1 : 0;
        }

        kernel void build_pool_segments(
            const device long* order,
            const device int* parent_cumsum,
            const device int* is_new,
            const device int4* in_coords,
            device int* parents_out,
            device int* seg_starts,
            device int4* out_coords,
            constant int& N,
            constant int& M,
            uint i [[thread_position_in_grid]]
        ) {
            if (int(i) >= N) return;
            int orig = int(order[i]);
            int p = parent_cumsum[i] - 1;
            parents_out[orig] = p;
            if (is_new[i] != 0) {
                seg_starts[p] = int(i);
                int4 v = in_coords[orig];
                out_coords[p] = int4(v.x, v.y >> 1, v.z >> 1, v.w >> 1);
            }
            if (int(i) == N - 1) {
                seg_starts[M] = N;
            }
        }

        kernel void mean_pool_gather_f4(
            const device float4* features,
            const device long* order,
            const device int* seg_starts,
            device float4* pooled,
            constant int& M,
            constant int& c_vec,
            uint2 gid [[thread_position_in_grid]]
        ) {
            int c = int(gid.x);
            int p = int(gid.y);
            if (c >= c_vec || p >= M) return;
            int start = seg_starts[p];
            int end = seg_starts[p + 1];
            float4 sum = float4(0.0f);
            ulong cv = ulong(c_vec);
            for (int idx = start; idx < end; ++idx) {
                int child = int(order[idx]);
                sum += features[ulong(child) * cv + c];
            }
            pooled[ulong(p) * cv + c] = sum / float(end - start);
        }

        // -----------------------------------------------------------------
        // 5. Vectorized 2D gather fallback for non-32-aligned channel widths
        // -----------------------------------------------------------------
        kernel void gather_neighbors_2d(
            const device float4* features,
            const device int* indices,
            device float4* output,
            constant int& channel_vectors,
            constant int& voxel_count,
            uint2 gid [[thread_position_in_grid]]
        ) {
            int c = int(gid.x);
            int idx = int(gid.y);
            int neighbor = indices[idx];
            ulong cv = ulong(channel_vectors);
            output[ulong(idx) * cv + c] = neighbor == voxel_count
                ? float4(0.0f)
                : features[ulong(neighbor) * cv + c];
        }
        """
    )


def image_neighbor_cache(coordinates, height, width):
    """Build bitmask-sorted tile cache directly for raster-ordered image voxels."""
    coords = coordinates.contiguous()
    count = len(coords)
    num_tiles = (count + _TILE_M - 1) // _TILE_M
    kernels = _kernels()
    sort_keys = coords.new_empty(count, dtype=torch.int32)
    raw_masks = coords.new_empty(count, dtype=torch.uint32)
    kernels.image_bitmasks(
        coords, sort_keys, raw_masks, count, height, width,
        threads=count, group_size=256, arg_casts="int32",
    )
    _, order = sort_keys.sort()
    sorted_idx = coords.new_empty(num_tiles * _TILE_M, dtype=torch.int32)
    tile_neighbors = coords.new_empty((num_tiles, 27, _TILE_M), dtype=torch.int32)
    tile_masks = coords.new_empty(num_tiles, dtype=torch.uint32)
    kernels.build_image_tile_cache_bm32(
        coords, order, raw_masks, sorted_idx, tile_neighbors, tile_masks,
        count, height, width,
        threads=num_tiles * _TILE_M, group_size=_TILE_M, arg_casts="int32",
    )
    return MPSNeighborCache(sorted_idx, tile_neighbors, tile_masks, num_tiles, count)


def neighbor_cache(coordinates, shape, *, sorted_coords=False):
    """Build bitmask-sorted tile cache for general 3D sparse coordinates."""
    coords = coordinates.contiguous()
    _, height, width, depth, _ = shape
    column_stride = depth + 2
    row_stride = (width + 2) * column_stride
    batch_stride = (height + 2) * row_stride
    count = len(coords)
    num_tiles = (count + _TILE_M - 1) // _TILE_M
    kernels = _kernels()

    keys = coords.new_empty(count, dtype=torch.int64)
    kernels.encode_coordinates(
        coords, keys, batch_stride, row_stride, column_stride,
        threads=count, group_size=256,
    )
    if sorted_coords:
        sorted_keys = keys
        order_in = keys
        identity_order = 1
    else:
        sorted_keys, order_in = keys.sort()
        identity_order = 0

    neighbors = coords.new_empty((count, 27), dtype=torch.int32)
    sort_keys = coords.new_empty(count, dtype=torch.int32)
    raw_masks = coords.new_empty(count, dtype=torch.uint32)
    kernels.find_neighbors_and_masks(
        keys, sorted_keys, order_in, neighbors, sort_keys, raw_masks,
        count, row_stride, column_stride, identity_order,
        threads=count, group_size=256, arg_casts={6: "int32", 9: "int32"},
    )
    _, order = sort_keys.sort()
    sorted_idx = coords.new_empty(num_tiles * _TILE_M, dtype=torch.int32)
    tile_neighbors = coords.new_empty((num_tiles, 27, _TILE_M), dtype=torch.int32)
    tile_masks = coords.new_empty(num_tiles, dtype=torch.uint32)
    kernels.build_tile_cache_bm32(
        neighbors, order, raw_masks, sorted_idx, tile_neighbors, tile_masks,
        count,
        threads=num_tiles * _TILE_M, group_size=_TILE_M, arg_casts="int32",
    )
    return MPSNeighborCache(sorted_idx, tile_neighbors, tile_masks, num_tiles, count)


def submanifold_conv3d(features, weight, bias, cache, *, activation=None, residual=None):
    """Run Masked Implicit GEMM 3D submanifold convolution on MPS."""
    src = features.contiguous()
    w = weight.contiguous()
    b = bias.contiguous()
    n_voxels, channels = src.shape
    output = torch.empty_like(src)

    if activation == "silu":
        act_mode = 1
        res_buf = src
    elif residual is not None:
        act_mode = 2
        res_buf = residual.contiguous()
    else:
        act_mode = 0
        res_buf = src

    kernels = _kernels()
    if channels >= 128 and channels % 64 == 0:
        kernels.spconv_bm32_bn64_bk32_4sg(
            src, w, b, res_buf,
            cache.sorted_idx, cache.tile_neighbors, cache.tile_masks,
            output, n_voxels, channels, act_mode,
            threads=(cache.num_tiles * 128, channels // 64, 1),
            group_size=(128, 1, 1),
            arg_casts="int32",
        )
    else:
        kernels.spconv_bm32_bn32_bk32_2sg(
            src, w, b, res_buf,
            cache.sorted_idx, cache.tile_neighbors, cache.tile_masks,
            output, n_voxels, channels, act_mode,
            threads=(cache.num_tiles * 64, channels // 32, 1),
            group_size=(64, 1, 1),
            arg_casts="int32",
        )
    return output


def mean_pool(features, coordinates, output_shape):
    """Atomic-free 2x2x2 sparse mean pooling on MPS."""
    coords = coordinates.contiguous()
    src = features.contiguous()
    _, height, width, depth, _ = output_shape
    n_voxels, channels = src.shape
    kernels = _kernels()

    keys = coords.new_empty(n_voxels, dtype=torch.int64)
    kernels.encode_pool_keys(
        coords, keys, height, width, depth,
        threads=n_voxels, group_size=256,
    )
    sorted_keys, order = keys.sort()
    is_new = coords.new_empty(n_voxels, dtype=torch.int32)
    kernels.mark_unique_runs(sorted_keys, is_new, threads=n_voxels, group_size=256)
    parent_cumsum = is_new.cumsum(0, dtype=torch.int32)
    out_count = int(parent_cumsum[-1].item())

    parents = coords.new_empty(n_voxels, dtype=torch.int32)
    seg_starts = coords.new_empty(out_count + 1, dtype=torch.int32)
    out_coords = coords.new_empty((out_count, 4), dtype=torch.int32)
    kernels.build_pool_segments(
        order, parent_cumsum, is_new, coords,
        parents, seg_starts, out_coords, n_voxels, out_count,
        threads=n_voxels, group_size=256, arg_casts="int32",
    )

    pooled = src.new_empty((out_count, channels))
    c_vec = channels // 4
    kernels.mean_pool_gather_f4(
        src, order, seg_starts, pooled, out_count, c_vec,
        threads=(c_vec, out_count, 1),
        group_size=(min(c_vec, 32), min(out_count, 8), 1),
        arg_casts="int32",
    )
    return pooled, out_coords, output_shape, parents


def gather_neighbors(features, indices, output):
    """Vectorized 2D gather fallback for channel widths not divisible by 32."""
    c_vec = features.shape[1] // 4
    num_indices = indices.numel()
    _kernels().gather_neighbors_2d(
        features, indices, output, c_vec, len(features),
        threads=(c_vec, num_indices, 1),
        group_size=(min(c_vec, 32), min(num_indices, 8), 1),
        arg_casts="int32",
    )
