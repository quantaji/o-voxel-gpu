#include "api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;
        constexpr uint64_t kEmptyKey = UINT64_MAX;
        constexpr uint32_t kEmptyVal = UINT32_MAX;
        constexpr uint32_t kOverflowVal = UINT32_MAX - 1u;

        __device__ __forceinline__ uint64_t mix64(uint64_t x)
        {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53ULL;
            x ^= x >> 33;
            return x;
        }

        __device__ __forceinline__ uint32_t hash_slot(uint64_t key, uint32_t capacity)
        {
            return static_cast<uint32_t>(mix64(key) % capacity);
        }

        __device__ uint32_t get_or_create_voxel(
            uint64_t key,
            int x,
            int y,
            int z,
            uint64_t *hash_keys,
            uint32_t *hash_vals,
            uint32_t *voxel_count,
            int32_t *voxel_coords,
            uint32_t capacity,
            uint32_t max_voxels)
        {
            uint32_t slot = hash_slot(key, capacity);
            for (uint32_t probe = 0; probe < capacity; ++probe)
            {
                const uint64_t prev = atomicCAS(
                    reinterpret_cast<unsigned long long *>(hash_keys + slot),
                    static_cast<unsigned long long>(kEmptyKey),
                    static_cast<unsigned long long>(key));
                if (prev == kEmptyKey)
                {
                    const uint32_t idx = atomicAdd(voxel_count, 1u);
                    if (idx >= max_voxels)
                    {
                        __threadfence();
                        hash_vals[slot] = kOverflowVal;
                        return kEmptyVal;
                    }
                    voxel_coords[3 * idx + 0] = x;
                    voxel_coords[3 * idx + 1] = y;
                    voxel_coords[3 * idx + 2] = z;
                    __threadfence();
                    hash_vals[slot] = idx;
                    return idx;
                }
                if (prev == key)
                {
                    volatile uint32_t *val_ptr = hash_vals + slot;
                    uint32_t val = *val_ptr;
                    while (val == kEmptyVal)
                        val = *val_ptr;
                    return val == kOverflowVal ? kEmptyVal : val;
                }
                slot = (slot + 1u) % capacity;
            }
            return kEmptyVal;
        }

        __device__ __forceinline__ SymQEF10 make_ref_qef(const float *tri)
        {
            const float e0x = tri[3] - tri[0];
            const float e0y = tri[4] - tri[1];
            const float e0z = tri[5] - tri[2];
            const float e1x = tri[6] - tri[3];
            const float e1y = tri[7] - tri[4];
            const float e1z = tri[8] - tri[5];
            float nx = e0y * e1z - e0z * e1y;
            float ny = e0z * e1x - e0x * e1z;
            float nz = e0x * e1y - e0y * e1x;
            const float inv_len = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-30f);
            nx *= inv_len;
            ny *= inv_len;
            nz *= inv_len;
            return qef_from_plane(float4{nx, ny, nz, -(nx * tri[0] + ny * tri[1] + nz * tri[2])});
        }

        __device__ __forceinline__ void atomic_add_qef(float *dst, const SymQEF10 &q)
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

        template <typename Emit>
        __device__ void scan_triangle_events(
            const float *tri,
            int ax2,
            float3 voxel_size,
            Int3 grid_min,
            Int3 grid_max,
            Emit emit)
        {
            const int ax0 = (ax2 + 1) % 3;
            const int ax1 = (ax2 + 2) % 3;
            double t[3][3] = {
                {static_cast<double>(tri[ax0]), static_cast<double>(tri[ax1]), static_cast<double>(tri[ax2])},
                {static_cast<double>(tri[3 + ax0]), static_cast<double>(tri[3 + ax1]), static_cast<double>(tri[3 + ax2])},
                {static_cast<double>(tri[6 + ax0]), static_cast<double>(tri[6 + ax1]), static_cast<double>(tri[6 + ax2])},
            };
            int order[3] = {0, 1, 2};
            if (t[order[0]][1] > t[order[1]][1])
            {
                const int tmp = order[0];
                order[0] = order[1];
                order[1] = tmp;
            }
            if (t[order[1]][1] > t[order[2]][1])
            {
                const int tmp = order[1];
                order[1] = order[2];
                order[2] = tmp;
            }
            if (t[order[0]][1] > t[order[1]][1])
            {
                const int tmp = order[0];
                order[0] = order[1];
                order[1] = tmp;
            }

            const double *t0 = t[order[0]];
            const double *t1 = t[order[1]];
            const double *t2 = t[order[2]];
            const float vs[3] = {voxel_size.x, voxel_size.y, voxel_size.z};
            const int start = max(min(static_cast<int>(t0[1] / vs[ax1]), grid_max[ax1] - 1), grid_min[ax1]);
            const int mid = max(min(static_cast<int>(t1[1] / vs[ax1]), grid_max[ax1] - 1), grid_min[ax1]);
            const int end = max(min(static_cast<int>(t2[1] / vs[ax1]), grid_max[ax1] - 1), grid_min[ax1]);

            auto scan_half = [&](int row_start, int row_end, const double *a, const double *b, const double *c)
            {
                for (int y_idx = row_start; y_idx < row_end; ++y_idx)
                {
                    const double y = (static_cast<double>(y_idx) + 1.0) * vs[ax1];
                    const double ab = fabs(a[1] - b[1]) < 1e-12 ? 0.0 : (y - a[1]) / (b[1] - a[1]);
                    const double ac = fabs(a[1] - c[1]) < 1e-12 ? 0.0 : (y - a[1]) / (c[1] - a[1]);
                    double t3x = (1.0 - ab) * a[0] + ab * b[0];
                    double t3z = (1.0 - ab) * a[2] + ab * b[2];
                    double t4x = (1.0 - ac) * a[0] + ac * c[0];
                    double t4z = (1.0 - ac) * a[2] + ac * c[2];
                    if (t3x > t4x)
                    {
                        double tmp = t3x;
                        t3x = t4x;
                        t4x = tmp;
                        tmp = t3z;
                        t3z = t4z;
                        t4z = tmp;
                    }

                    const int line_start = max(min(static_cast<int>(t3x / vs[ax0]), grid_max[ax0] - 1), grid_min[ax0]);
                    const int line_end = max(min(static_cast<int>(t4x / vs[ax0]), grid_max[ax0] - 1), grid_min[ax0]);
                    for (int x_idx = line_start; x_idx < line_end; ++x_idx)
                    {
                        const double x = (static_cast<double>(x_idx) + 1.0) * vs[ax0];
                        const double alpha = fabs(t4x - t3x) < 1e-12 ? 0.0 : (x - t3x) / (t4x - t3x);
                        const double z = (1.0 - alpha) * t3z + alpha * t4z;
                        const int z_idx = static_cast<int>(z / vs[ax2]);
                        if (z_idx >= grid_min[ax2] && z_idx < grid_max[ax2])
                            emit(ax0, ax1, ax2, x_idx, y_idx, z_idx, x, y, z);
                    }
                }
            };
            scan_half(start, mid, t0, t1, t2);
            scan_half(mid, end, t2, t1, t0);
        }

        __global__ void intersect_ref_kernel(
            int64_t num_triangles,
            const float *triangles,
            float3 voxel_size,
            Int3 grid_min,
            Int3 grid_max,
            uint64_t *hash_keys,
            uint32_t *hash_vals,
            uint32_t *voxel_count,
            int32_t *voxel_coords,
            float *mean_sum,
            float *cnt,
            uint32_t *intersected,
            float *qefs,
            uint32_t capacity,
            uint32_t max_voxels)
        {
            const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (tid >= num_triangles)
                return;

            const float *tri = triangles + tid * 9;
            const SymQEF10 qef = make_ref_qef(tri);
            auto emit = [&](int ax0, int ax1, int ax2, int x_idx, int y_idx, int z_idx, double x, double y, double z)
            {
                for (int dx = 0; dx < 2; ++dx)
                {
                    for (int dy = 0; dy < 2; ++dy)
                    {
                        int coord[3];
                        coord[ax0] = x_idx + dx;
                        coord[ax1] = y_idx + dy;
                        coord[ax2] = z_idx;
                        if (coord[0] < grid_min.x || coord[0] >= grid_max.x)
                            continue;
                        if (coord[1] < grid_min.y || coord[1] >= grid_max.y)
                            continue;
                        if (coord[2] < grid_min.z || coord[2] >= grid_max.z)
                            continue;

                        const uint64_t key = pack_voxel_key(coord[0], coord[1], coord[2], grid_min, grid_max);
                        const uint32_t idx = get_or_create_voxel(
                            key, coord[0], coord[1], coord[2],
                            hash_keys, hash_vals, voxel_count, voxel_coords, capacity, max_voxels);
                        if (idx == kEmptyVal)
                            continue;

                        float p[3];
                        p[ax0] = static_cast<float>(x);
                        p[ax1] = static_cast<float>(y);
                        p[ax2] = static_cast<float>(z);
                        atomicAdd(mean_sum + 3 * idx + 0, p[0]);
                        atomicAdd(mean_sum + 3 * idx + 1, p[1]);
                        atomicAdd(mean_sum + 3 * idx + 2, p[2]);
                        atomicAdd(cnt + idx, 1.0f);
                        if (dx == 0 && dy == 0)
                            atomicOr(intersected + idx, 1u << ax2);
                        atomic_add_qef(qefs + 10 * idx, qef);
                    }
                }
            };
            scan_triangle_events(tri, 0, voxel_size, grid_min, grid_max, emit);
            scan_triangle_events(tri, 1, voxel_size, grid_min, grid_max, emit);
            scan_triangle_events(tri, 2, voxel_size, grid_min, grid_max, emit);
        }

        __global__ void copy_outputs_kernel(
            int64_t n,
            const int32_t *coords_in,
            const float *mean_in,
            const float *cnt_in,
            const uint32_t *intersected_in,
            const float *qef_in,
            int32_t *coords_out,
            float *mean_out,
            float *cnt_out,
            bool *intersected_out,
            float *qef_out)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            coords_out[3 * i + 0] = coords_in[3 * i + 0];
            coords_out[3 * i + 1] = coords_in[3 * i + 1];
            coords_out[3 * i + 2] = coords_in[3 * i + 2];
            mean_out[3 * i + 0] = mean_in[3 * i + 0];
            mean_out[3 * i + 1] = mean_in[3 * i + 1];
            mean_out[3 * i + 2] = mean_in[3 * i + 2];
            cnt_out[i] = cnt_in[i];
            const uint32_t mask = intersected_in[i];
            intersected_out[3 * i + 0] = (mask & 1u) != 0;
            intersected_out[3 * i + 1] = (mask & 2u) != 0;
            intersected_out[3 * i + 2] = (mask & 4u) != 0;
            for (int k = 0; k < 10; ++k)
                qef_out[10 * i + k] = qef_in[10 * i + k];
        }

    } // namespace

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    intersect_qef_ref(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        int64_t chunk_triangles)
    {
        TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");
        TORCH_CHECK(chunk_triangles > 0, "chunk_triangles must be positive");

        const c10::cuda::CUDAGuard guard(triangles.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(triangles.get_device()).stream();
        const torch::Device device = triangles.device();
        const auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
        const auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(device);
        const auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(device);
        const auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(device);

        const int64_t num_triangles = triangles.size(0);
        auto empty_voxels = torch::empty({0, 3}, opts_i32);
        auto empty_mean = torch::empty({0, 3}, opts_f32);
        auto empty_cnt = torch::empty({0}, opts_f32);
        auto empty_intersected = torch::empty({0, 3}, opts_bool);
        auto empty_qefs = torch::empty({0, 10}, opts_f32);
        auto empty_hash_keys = torch::empty({0}, opts_u64);
        auto empty_hash_vals = torch::empty({0}, opts_u32);
        if (num_triangles == 0)
            return std::make_tuple(
                empty_voxels,
                empty_mean,
                empty_cnt,
                empty_intersected,
                empty_qefs,
                empty_hash_keys,
                empty_hash_vals);

        const float *voxel_size_ptr = voxel_size.data_ptr<float>();
        const int32_t *grid_range_ptr = grid_range.data_ptr<int32_t>();
        const float3 voxel_size_h{voxel_size_ptr[0], voxel_size_ptr[1], voxel_size_ptr[2]};
        const Int3 grid_min{grid_range_ptr[0], grid_range_ptr[1], grid_range_ptr[2]};
        const Int3 grid_max{grid_range_ptr[3], grid_range_ptr[4], grid_range_ptr[5]};

        const int64_t total_cells =
            static_cast<int64_t>(grid_max.x - grid_min.x) *
            static_cast<int64_t>(grid_max.y - grid_min.y) *
            static_cast<int64_t>(grid_max.z - grid_min.z);
        int64_t max_voxels = std::min(total_cells, num_triangles * 64);
        max_voxels = std::max<int64_t>(max_voxels, 65536);
        const int64_t capacity = max_voxels * 2;
        TORCH_CHECK(capacity <= UINT32_MAX, "ref hash capacity exceeds uint32_t");

        auto hash_keys = torch::empty({capacity}, opts_u64);
        auto hash_vals = torch::empty({capacity}, opts_u32);
        auto voxel_count = torch::zeros({1}, opts_u32);
        auto voxel_coords = torch::empty({max_voxels * 3}, opts_i32);
        auto mean_sum = torch::zeros({max_voxels * 3}, opts_f32);
        auto cnt = torch::zeros({max_voxels}, opts_f32);
        auto intersected = torch::zeros({max_voxels}, opts_u32);
        auto qefs = torch::zeros({max_voxels * 10}, opts_f32);
        C10_CUDA_CHECK(cudaMemsetAsync(hash_keys.data_ptr<uint64_t>(), 0xff, capacity * sizeof(uint64_t), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(hash_vals.data_ptr<uint32_t>(), 0xff, capacity * sizeof(uint32_t), stream));

        int blocks = static_cast<int>((num_triangles + kThreads - 1) / kThreads);
        intersect_ref_kernel<<<blocks, kThreads, 0, stream>>>(
            num_triangles,
            triangles.data_ptr<float>(),
            voxel_size_h,
            grid_min,
            grid_max,
            hash_keys.data_ptr<uint64_t>(),
            hash_vals.data_ptr<uint32_t>(),
            voxel_count.data_ptr<uint32_t>(),
            voxel_coords.data_ptr<int32_t>(),
            mean_sum.data_ptr<float>(),
            cnt.data_ptr<float>(),
            intersected.data_ptr<uint32_t>(),
            qefs.data_ptr<float>(),
            static_cast<uint32_t>(capacity),
            static_cast<uint32_t>(max_voxels));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        uint32_t num_voxels_u32 = 0;
        C10_CUDA_CHECK(cudaMemcpyAsync(&num_voxels_u32, voxel_count.data_ptr<uint32_t>(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CHECK(num_voxels_u32 <= static_cast<uint32_t>(max_voxels), "ref hash table exceeded estimated capacity");
        const int64_t num_voxels = static_cast<int64_t>(num_voxels_u32);
        if (num_voxels == 0)
            return std::make_tuple(
                empty_voxels,
                empty_mean,
                empty_cnt,
                empty_intersected,
                empty_qefs,
                hash_keys,
                hash_vals);

        blocks = static_cast<int>((num_voxels + kThreads - 1) / kThreads);

        auto out_voxels = torch::empty({num_voxels, 3}, opts_i32);
        auto out_mean = torch::empty({num_voxels, 3}, opts_f32);
        auto out_cnt = torch::empty({num_voxels}, opts_f32);
        auto out_intersected = torch::empty({num_voxels, 3}, opts_bool);
        auto out_qefs = torch::empty({num_voxels, 10}, opts_f32);
        copy_outputs_kernel<<<blocks, kThreads, 0, stream>>>(
            num_voxels,
            voxel_coords.data_ptr<int32_t>(),
            mean_sum.data_ptr<float>(),
            cnt.data_ptr<float>(),
            intersected.data_ptr<uint32_t>(),
            qefs.data_ptr<float>(),
            out_voxels.data_ptr<int32_t>(),
            out_mean.data_ptr<float>(),
            out_cnt.data_ptr<float>(),
            out_intersected.data_ptr<bool>(),
            out_qefs.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return std::make_tuple(out_voxels, out_mean, out_cnt, out_intersected, out_qefs, hash_keys, hash_vals);
    }

} // namespace o_voxel::fdg
