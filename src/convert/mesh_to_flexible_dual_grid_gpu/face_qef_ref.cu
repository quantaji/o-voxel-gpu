#include "api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;

        __device__ __forceinline__ float ref_min(float a, float b)
        {
            return a < b ? a : b;
        }

        __device__ __forceinline__ float ref_max(float a, float b)
        {
            return a > b ? a : b;
        }

        __device__ __forceinline__ void atomic_add_qef(float *dst, SymQEF10 q)
        {
            atomicAdd(dst + 0, q.q00);
            atomicAdd(dst + 1, q.q01);
            atomicAdd(dst + 2, q.q02);
            atomicAdd(dst + 3, q.q03);
            atomicAdd(dst + 4, q.q11);
            atomicAdd(dst + 5, q.q12);
            atomicAdd(dst + 6, q.q13);
            atomicAdd(dst + 7, q.q22);
            atomicAdd(dst + 8, q.q23);
            atomicAdd(dst + 9, q.q33);
        }

        __global__ void face_qef_ref_kernel(
            int64_t num_triangles,
            const float *__restrict__ triangles,
            GridSpec grid,
            BrickLookup lookup,
            float *__restrict__ out_qefs)
        {
            const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (tid >= num_triangles)
                return;

            const float *tri = triangles + tid * 9;
            const float3 v0 = make_float3(tri[0], tri[1], tri[2]);
            const float3 v1 = make_float3(tri[3], tri[4], tri[5]);
            const float3 v2 = make_float3(tri[6], tri[7], tri[8]);
            const float vs[3] = {grid.voxel_size.x, grid.voxel_size.y, grid.voxel_size.z};
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;

            const float e0x = v1.x - v0.x;
            const float e0y = v1.y - v0.y;
            const float e0z = v1.z - v0.z;
            const float e1x = v2.x - v1.x;
            const float e1y = v2.y - v1.y;
            const float e1z = v2.z - v1.z;
            const float e2x = v0.x - v2.x;
            const float e2y = v0.y - v2.y;
            const float e2z = v0.z - v2.z;

            float nx = e0y * e1z - e0z * e1y;
            float ny = e0z * e1x - e0x * e1z;
            float nz = e0x * e1y - e0y * e1x;
            const float inv_len = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-30f);
            nx *= inv_len;
            ny *= inv_len;
            nz *= inv_len;
            const SymQEF10 qef = qef_from_plane(make_float4(nx, ny, nz, -(nx * v0.x + ny * v0.y + nz * v0.z)));

            const float bb_min_f_x = ref_min(ref_min(v0.x, v1.x), v2.x) / vs[0];
            const float bb_min_f_y = ref_min(ref_min(v0.y, v1.y), v2.y) / vs[1];
            const float bb_min_f_z = ref_min(ref_min(v0.z, v1.z), v2.z) / vs[2];
            const float bb_max_f_x = ref_max(ref_max(v0.x, v1.x), v2.x) / vs[0];
            const float bb_max_f_y = ref_max(ref_max(v0.y, v1.y), v2.y) / vs[1];
            const float bb_max_f_z = ref_max(ref_max(v0.z, v1.z), v2.z) / vs[2];

            const int bb_min_x = max(static_cast<int>(bb_min_f_x), grid_min.x);
            const int bb_min_y = max(static_cast<int>(bb_min_f_y), grid_min.y);
            const int bb_min_z = max(static_cast<int>(bb_min_f_z), grid_min.z);
            const int bb_max_x = min(static_cast<int>(bb_max_f_x + 1.0f), grid_max.x);
            const int bb_max_y = min(static_cast<int>(bb_max_f_y + 1.0f), grid_max.y);
            const int bb_max_z = min(static_cast<int>(bb_max_f_z + 1.0f), grid_max.z);

            const float c_x = nx > 0.0f ? vs[0] : 0.0f;
            const float c_y = ny > 0.0f ? vs[1] : 0.0f;
            const float c_z = nz > 0.0f ? vs[2] : 0.0f;
            const float d1 = nx * (c_x - v0.x) + ny * (c_y - v0.y) + nz * (c_z - v0.z);
            const float d2 = nx * (vs[0] - c_x - v0.x) + ny * (vs[1] - c_y - v0.y) + nz * (vs[2] - c_z - v0.z);

            const int mul_xy = nz < 0.0f ? -1 : 1;
            const float n_xy_e0_x = -mul_xy * e0y;
            const float n_xy_e0_y = mul_xy * e0x;
            const float n_xy_e1_x = -mul_xy * e1y;
            const float n_xy_e1_y = mul_xy * e1x;
            const float n_xy_e2_x = -mul_xy * e2y;
            const float n_xy_e2_y = mul_xy * e2x;
            const float d_xy_e0 = -(n_xy_e0_x * v0.x + n_xy_e0_y * v0.y) + fmaxf(n_xy_e0_x, 0.0f) * vs[0] + fmaxf(n_xy_e0_y, 0.0f) * vs[1];
            const float d_xy_e1 = -(n_xy_e1_x * v1.x + n_xy_e1_y * v1.y) + fmaxf(n_xy_e1_x, 0.0f) * vs[0] + fmaxf(n_xy_e1_y, 0.0f) * vs[1];
            const float d_xy_e2 = -(n_xy_e2_x * v2.x + n_xy_e2_y * v2.y) + fmaxf(n_xy_e2_x, 0.0f) * vs[0] + fmaxf(n_xy_e2_y, 0.0f) * vs[1];

            const int mul_yz = nx < 0.0f ? -1 : 1;
            const float n_yz_e0_x = -mul_yz * e0z;
            const float n_yz_e0_y = mul_yz * e0y;
            const float n_yz_e1_x = -mul_yz * e1z;
            const float n_yz_e1_y = mul_yz * e1y;
            const float n_yz_e2_x = -mul_yz * e2z;
            const float n_yz_e2_y = mul_yz * e2y;
            const float d_yz_e0 = -(n_yz_e0_x * v0.y + n_yz_e0_y * v0.z) + fmaxf(n_yz_e0_x, 0.0f) * vs[1] + fmaxf(n_yz_e0_y, 0.0f) * vs[2];
            const float d_yz_e1 = -(n_yz_e1_x * v1.y + n_yz_e1_y * v1.z) + fmaxf(n_yz_e1_x, 0.0f) * vs[1] + fmaxf(n_yz_e1_y, 0.0f) * vs[2];
            const float d_yz_e2 = -(n_yz_e2_x * v2.y + n_yz_e2_y * v2.z) + fmaxf(n_yz_e2_x, 0.0f) * vs[1] + fmaxf(n_yz_e2_y, 0.0f) * vs[2];

            const int mul_zx = ny < 0.0f ? -1 : 1;
            const float n_zx_e0_x = -mul_zx * e0x;
            const float n_zx_e0_y = mul_zx * e0z;
            const float n_zx_e1_x = -mul_zx * e1x;
            const float n_zx_e1_y = mul_zx * e1z;
            const float n_zx_e2_x = -mul_zx * e2x;
            const float n_zx_e2_y = mul_zx * e2z;
            const float d_zx_e0 = -(n_zx_e0_x * v0.z + n_zx_e0_y * v0.x) + fmaxf(n_zx_e0_x, 0.0f) * vs[2] + fmaxf(n_zx_e0_y, 0.0f) * vs[0];
            const float d_zx_e1 = -(n_zx_e1_x * v1.z + n_zx_e1_y * v1.x) + fmaxf(n_zx_e1_x, 0.0f) * vs[2] + fmaxf(n_zx_e1_y, 0.0f) * vs[0];
            const float d_zx_e2 = -(n_zx_e2_x * v2.z + n_zx_e2_y * v2.x) + fmaxf(n_zx_e2_x, 0.0f) * vs[2] + fmaxf(n_zx_e2_y, 0.0f) * vs[0];

            for (int z = bb_min_z; z < bb_max_z; ++z)
            {
                for (int y = bb_min_y; y < bb_max_y; ++y)
                {
                    for (int x = bb_min_x; x < bb_max_x; ++x)
                    {
                        const float px = x * vs[0];
                        const float py = y * vs[1];
                        const float pz = z * vs[2];
                        const float n_dot_p = nx * px + ny * py + nz * pz;
                        if ((n_dot_p + d1) * (n_dot_p + d2) > 0.0f)
                            continue;
                        if (n_xy_e0_x * px + n_xy_e0_y * py + d_xy_e0 < 0.0f)
                            continue;
                        if (n_xy_e1_x * px + n_xy_e1_y * py + d_xy_e1 < 0.0f)
                            continue;
                        if (n_xy_e2_x * px + n_xy_e2_y * py + d_xy_e2 < 0.0f)
                            continue;
                        if (n_yz_e0_x * py + n_yz_e0_y * pz + d_yz_e0 < 0.0f)
                            continue;
                        if (n_yz_e1_x * py + n_yz_e1_y * pz + d_yz_e1 < 0.0f)
                            continue;
                        if (n_yz_e2_x * py + n_yz_e2_y * pz + d_yz_e2 < 0.0f)
                            continue;
                        if (n_zx_e0_x * pz + n_zx_e0_y * px + d_zx_e0 < 0.0f)
                            continue;
                        if (n_zx_e1_x * pz + n_zx_e1_y * px + d_zx_e1 < 0.0f)
                            continue;
                        if (n_zx_e2_x * pz + n_zx_e2_y * px + d_zx_e2 < 0.0f)
                            continue;

                        const int64_t row = lookup_voxel_row_in_bricks(x, y, z, grid, lookup);
                        if (row >= 0)
                            atomic_add_qef(out_qefs + 10 * row, qef);
                    }
                }
            }
        }

    } // namespace

    torch::Tensor face_qef_ref(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        const torch::Tensor &voxels,
        const torch::Tensor &brick_hash_keys,
        const torch::Tensor &brick_hash_vals,
        const torch::Tensor &brick_bits,
        const torch::Tensor &brick_base)
    {
        TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");
        TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
        TORCH_CHECK(brick_hash_keys.is_cuda(), "brick_hash_keys must be a CUDA tensor");
        TORCH_CHECK(brick_hash_vals.is_cuda(), "brick_hash_vals must be a CUDA tensor");
        TORCH_CHECK(brick_bits.is_cuda(), "brick_bits must be a CUDA tensor");
        TORCH_CHECK(brick_base.is_cuda(), "brick_base must be a CUDA tensor");
        static_assert(sizeof(SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");

        const c10::cuda::CUDAGuard guard(triangles.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(triangles.get_device()).stream();
        const torch::Device device = triangles.device();
        const auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const int64_t num_triangles = triangles.size(0);
        const int64_t num_voxels = voxels.size(0);
        auto out_qefs = torch::zeros({num_voxels, 10}, opts_f32);
        if (num_triangles == 0 || num_voxels == 0)
            return out_qefs;

        const float *voxel_size_ptr = voxel_size.data_ptr<float>();
        const int32_t *grid_range_ptr = grid_range.data_ptr<int32_t>();
        const GridSpec grid{
            float3{voxel_size_ptr[0], voxel_size_ptr[1], voxel_size_ptr[2]},
            Int3{grid_range_ptr[0], grid_range_ptr[1], grid_range_ptr[2]},
            Int3{grid_range_ptr[3], grid_range_ptr[4], grid_range_ptr[5]},
        };
        const BrickLookup lookup{
            brick_hash_keys.data_ptr<uint64_t>(),
            brick_hash_vals.data_ptr<uint32_t>(),
            brick_bits.data_ptr<uint32_t>(),
            brick_base.data_ptr<int64_t>(),
            static_cast<uint64_t>(brick_hash_keys.numel()),
        };

        const int blocks = static_cast<int>((num_triangles + kThreads - 1) / kThreads);
        face_qef_ref_kernel<<<blocks, kThreads, 0, stream>>>(
            num_triangles,
            triangles.data_ptr<float>(),
            grid,
            lookup,
            out_qefs.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out_qefs;
    }

} // namespace o_voxel::fdg
