#include "api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>

namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;

        __device__ __forceinline__ uint64_t pack_edge_key(int32_t a, int32_t b)
        {
            return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
                   static_cast<uint32_t>(b);
        }

        __device__ __forceinline__ void atomic_add_qef_scaled(float *dst, SymQEF10 q, float weight)
        {
            atomicAdd(dst + 0, q.q00 * weight);
            atomicAdd(dst + 1, q.q01 * weight);
            atomicAdd(dst + 2, q.q02 * weight);
            atomicAdd(dst + 3, q.q03 * weight);
            atomicAdd(dst + 4, q.q11 * weight);
            atomicAdd(dst + 5, q.q12 * weight);
            atomicAdd(dst + 6, q.q13 * weight);
            atomicAdd(dst + 7, q.q22 * weight);
            atomicAdd(dst + 8, q.q23 * weight);
            atomicAdd(dst + 9, q.q33 * weight);
        }

        __device__ __forceinline__ SymQEF10 boundary_qef_from_segment_ref(float3 v0, float3 v1)
        {
            float dx = v1.x - v0.x;
            float dy = v1.y - v0.y;
            float dz = v1.z - v0.z;
            const float len2 = dx * dx + dy * dy + dz * dz;
            if (len2 < 1e-12f)
                return qef_zero();

            const float inv_len = rsqrtf(len2);
            dx *= inv_len;
            dy *= inv_len;
            dz *= inv_len;

            const float a00 = 1.0f - dx * dx;
            const float a01 = -dx * dy;
            const float a02 = -dx * dz;
            const float a11 = 1.0f - dy * dy;
            const float a12 = -dy * dz;
            const float a22 = 1.0f - dz * dz;
            const float b0 = -(a00 * v0.x + a01 * v0.y + a02 * v0.z);
            const float b1 = -(a01 * v0.x + a11 * v0.y + a12 * v0.z);
            const float b2 = -(a02 * v0.x + a12 * v0.y + a22 * v0.z);
            const float av0_x = a00 * v0.x + a01 * v0.y + a02 * v0.z;
            const float av0_y = a01 * v0.x + a11 * v0.y + a12 * v0.z;
            const float av0_z = a02 * v0.x + a12 * v0.y + a22 * v0.z;
            const float c = v0.x * av0_x + v0.y * av0_y + v0.z * av0_z;
            return SymQEF10{a00, a01, a02, b0, a11, a12, b1, a22, b2, c};
        }

        __device__ __forceinline__ void add_boundary_voxel_ref(
            int x,
            int y,
            int z,
            GridSpec grid,
            BrickLookup lookup,
            SymQEF10 qef,
            float boundary_weight,
            float *out_qefs)
        {
            if (x < grid.grid_min.x || x >= grid.grid_max.x)
                return;
            if (y < grid.grid_min.y || y >= grid.grid_max.y)
                return;
            if (z < grid.grid_min.z || z >= grid.grid_max.z)
                return;

            const int64_t row = lookup_voxel_row_in_bricks(x, y, z, grid, lookup);
            if (row >= 0)
                atomic_add_qef_scaled(out_qefs + 10 * row, qef, boundary_weight);
        }

        __global__ void extract_edges_ref_kernel(
            int64_t num_faces,
            const int32_t *__restrict__ faces,
            uint64_t *__restrict__ edge_keys,
            uint64_t *__restrict__ edge_vals)
        {
            const int64_t fid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (fid >= num_faces)
                return;

            int32_t v[3] = {
                faces[3 * fid + 0],
                faces[3 * fid + 1],
                faces[3 * fid + 2],
            };
            for (int e = 0; e < 3; ++e)
            {
                int32_t a = v[e];
                int32_t b = v[(e + 1) % 3];
                if (a > b)
                {
                    const int32_t t = a;
                    a = b;
                    b = t;
                }
                const uint64_t key = pack_edge_key(a, b);
                const int64_t out = 3 * fid + e;
                edge_keys[out] = key;
                edge_vals[out] = key;
            }
        }

        __global__ void accumulate_boundary_edges_ref_kernel(
            int64_t num_edges,
            const uint64_t *__restrict__ edge_keys,
            const uint64_t *__restrict__ edge_vals,
            const float *__restrict__ vertices,
            GridSpec grid,
            float boundary_weight,
            BrickLookup lookup,
            float *__restrict__ out_qefs)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_edges)
                return;

            const uint64_t key = edge_keys[i];
            const bool left_diff = (i == 0) || (key != edge_keys[i - 1]);
            const bool right_diff = (i == num_edges - 1) || (key != edge_keys[i + 1]);
            if (!(left_diff && right_diff))
                return;

            const uint64_t val = edge_vals[i];
            const int32_t v0_id = static_cast<int32_t>(val >> 32);
            const int32_t v1_id = static_cast<int32_t>(val & 0xffffffffu);
            const float3 v0 = make_float3(
                vertices[3 * static_cast<int64_t>(v0_id) + 0],
                vertices[3 * static_cast<int64_t>(v0_id) + 1],
                vertices[3 * static_cast<int64_t>(v0_id) + 2]);
            const float3 v1 = make_float3(
                vertices[3 * static_cast<int64_t>(v1_id) + 0],
                vertices[3 * static_cast<int64_t>(v1_id) + 1],
                vertices[3 * static_cast<int64_t>(v1_id) + 2]);

            const SymQEF10 qef = boundary_qef_from_segment_ref(v0, v1);
            double dir_x = static_cast<double>(v1.x) - static_cast<double>(v0.x);
            double dir_y = static_cast<double>(v1.y) - static_cast<double>(v0.y);
            double dir_z = static_cast<double>(v1.z) - static_cast<double>(v0.z);
            const double seg_len = sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
            if (seg_len < 1e-6)
                return;
            dir_x /= seg_len;
            dir_y /= seg_len;
            dir_z /= seg_len;

            int cur_x = static_cast<int>(floorf(v0.x / grid.voxel_size.x));
            int cur_y = static_cast<int>(floorf(v0.y / grid.voxel_size.y));
            int cur_z = static_cast<int>(floorf(v0.z / grid.voxel_size.z));

            if (fabs(dir_x) + fabs(dir_y) + fabs(dir_z) < 1e-12)
                return;

            const int step_x = dir_x > 0.0 ? 1 : -1;
            const int step_y = dir_y > 0.0 ? 1 : -1;
            const int step_z = dir_z > 0.0 ? 1 : -1;

            double tmax_x;
            double tmax_y;
            double tmax_z;
            double tdelta_x;
            double tdelta_y;
            double tdelta_z;

            if (fabs(dir_x) < 1e-12)
            {
                tmax_x = 1e300;
                tdelta_x = 1e300;
            }
            else
            {
                const float border = grid.voxel_size.x * (cur_x + (step_x > 0 ? 1 : 0));
                tmax_x = (border - v0.x) / dir_x;
                tdelta_x = static_cast<double>(grid.voxel_size.x) / fabs(dir_x);
            }
            if (fabs(dir_y) < 1e-12)
            {
                tmax_y = 1e300;
                tdelta_y = 1e300;
            }
            else
            {
                const float border = grid.voxel_size.y * (cur_y + (step_y > 0 ? 1 : 0));
                tmax_y = (border - v0.y) / dir_y;
                tdelta_y = static_cast<double>(grid.voxel_size.y) / fabs(dir_y);
            }
            if (fabs(dir_z) < 1e-12)
            {
                tmax_z = 1e300;
                tdelta_z = 1e300;
            }
            else
            {
                const float border = grid.voxel_size.z * (cur_z + (step_z > 0 ? 1 : 0));
                tmax_z = (border - v0.z) / dir_z;
                tdelta_z = static_cast<double>(grid.voxel_size.z) / fabs(dir_z);
            }

            add_boundary_voxel_ref(cur_x, cur_y, cur_z, grid, lookup, qef, boundary_weight, out_qefs);

            while (true)
            {
                int axis;
                if (tmax_x < tmax_y)
                    axis = (tmax_x < tmax_z) ? 0 : 2;
                else
                    axis = (tmax_y < tmax_z) ? 1 : 2;

                if (axis == 0 && tmax_x > seg_len)
                    break;
                if (axis == 1 && tmax_y > seg_len)
                    break;
                if (axis == 2 && tmax_z > seg_len)
                    break;

                if (axis == 0)
                {
                    cur_x += step_x;
                    tmax_x += tdelta_x;
                }
                else if (axis == 1)
                {
                    cur_y += step_y;
                    tmax_y += tdelta_y;
                }
                else
                {
                    cur_z += step_z;
                    tmax_z += tdelta_z;
                }

                add_boundary_voxel_ref(cur_x, cur_y, cur_z, grid, lookup, qef, boundary_weight, out_qefs);
            }
        }

    } // namespace

    torch::Tensor boundary_qef_ref(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        float boundary_weight,
        const torch::Tensor &voxels,
        const torch::Tensor &brick_hash_keys,
        const torch::Tensor &brick_hash_vals,
        const torch::Tensor &brick_bits,
        const torch::Tensor &brick_base)
    {
        TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
        TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
        TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
        TORCH_CHECK(brick_hash_keys.is_cuda(), "brick_hash_keys must be a CUDA tensor");
        TORCH_CHECK(brick_hash_vals.is_cuda(), "brick_hash_vals must be a CUDA tensor");
        TORCH_CHECK(brick_bits.is_cuda(), "brick_bits must be a CUDA tensor");
        TORCH_CHECK(brick_base.is_cuda(), "brick_base must be a CUDA tensor");
        static_assert(sizeof(SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");

        const c10::cuda::CUDAGuard guard(vertices.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        const torch::Device device = vertices.device();
        const auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(device);
        const auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);

        const int64_t num_faces = faces.size(0);
        const int64_t num_voxels = voxels.size(0);
        auto out_qefs = torch::zeros({num_voxels, 10}, opts_f32);
        if (num_faces == 0 || num_voxels == 0 || boundary_weight <= 0.0f)
            return out_qefs;

        const int64_t num_edges = num_faces * 3;
        auto edge_keys = torch::empty({num_edges}, opts_u64);
        auto edge_vals = torch::empty({num_edges}, opts_u64);
        int blocks = static_cast<int>((num_faces + kThreads - 1) / kThreads);
        extract_edges_ref_kernel<<<blocks, kThreads, 0, stream>>>(
            num_faces,
            faces.data_ptr<int32_t>(),
            edge_keys.data_ptr<uint64_t>(),
            edge_vals.data_ptr<uint64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto edge_keys_sorted = torch::empty({num_edges}, opts_u64);
        auto edge_vals_sorted = torch::empty({num_edges}, opts_u64);
        size_t temp_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_bytes,
            edge_keys.data_ptr<uint64_t>(),
            edge_keys_sorted.data_ptr<uint64_t>(),
            edge_vals.data_ptr<uint64_t>(),
            edge_vals_sorted.data_ptr<uint64_t>(),
            static_cast<int>(num_edges),
            0,
            64,
            stream));
        auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            temp.data_ptr<uint8_t>(),
            temp_bytes,
            edge_keys.data_ptr<uint64_t>(),
            edge_keys_sorted.data_ptr<uint64_t>(),
            edge_vals.data_ptr<uint64_t>(),
            edge_vals_sorted.data_ptr<uint64_t>(),
            static_cast<int>(num_edges),
            0,
            64,
            stream));

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

        blocks = static_cast<int>((num_edges + kThreads - 1) / kThreads);
        accumulate_boundary_edges_ref_kernel<<<blocks, kThreads, 0, stream>>>(
            num_edges,
            edge_keys_sorted.data_ptr<uint64_t>(),
            edge_vals_sorted.data_ptr<uint64_t>(),
            vertices.data_ptr<float>(),
            grid,
            boundary_weight,
            lookup,
            out_qefs.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out_qefs;
    }

} // namespace o_voxel::fdg
