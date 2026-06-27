#include "api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;

        __host__ __device__ __forceinline__ int64_t div_up_i64(int64_t n, int64_t d)
        {
            return (n + d - 1) / d;
        }

        int32_t read_i32(const torch::Tensor &t, cudaStream_t stream)
        {
            int32_t value = 0;
            C10_CUDA_CHECK(cudaMemcpyAsync(&value, t.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            return value;
        }

        int64_t read_compact_total_i32(const torch::Tensor &counts, const torch::Tensor &offsets, int64_t n, cudaStream_t stream)
        {
            int32_t tail[2] = {0, 0};
            C10_CUDA_CHECK(cudaMemcpyAsync(
                tail,
                counts.data_ptr<int32_t>() + n - 1,
                sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream));
            C10_CUDA_CHECK(cudaMemcpyAsync(
                tail + 1,
                offsets.data_ptr<int32_t>() + n - 1,
                sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            return static_cast<int64_t>(tail[0]) + static_cast<int64_t>(tail[1]);
        }

        __device__ __forceinline__ float3 sub3(float3 a, float3 b)
        {
            return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        }

        __device__ __forceinline__ float dot3(float3 a, float3 b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        __device__ __forceinline__ float3 cross3(float3 a, float3 b)
        {
            return make_float3(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
        }

        __device__ __forceinline__ float3 normalize3(float3 a)
        {
            const float n2 = dot3(a, a);
            if (n2 <= 0.0f)
                return a;
            const float n = sqrtf(n2);
            return make_float3(a.x / n, a.y / n, a.z / n);
        }

        struct QEFAdd
        {
            __host__ __device__ SymQEF10 operator()(const SymQEF10 &a, const SymQEF10 &b) const
            {
                return qef_add(a, b);
            }
        };

        __host__ __device__ __forceinline__ uint64_t pack_pair_key(int32_t voxel_id, int32_t face_id)
        {
            return (static_cast<uint64_t>(static_cast<uint32_t>(voxel_id)) << 32) |
                   static_cast<uint32_t>(face_id);
        }

        __host__ __device__ __forceinline__ int32_t unpack_pair_voxel_id(uint64_t key)
        {
            return static_cast<int32_t>(key >> 32);
        }

        __host__ __device__ __forceinline__ int32_t unpack_pair_face_id(uint64_t key)
        {
            return static_cast<int32_t>(key & 0xffffffffu);
        }

        __device__ int32_t lower_bound_u64(const uint64_t *data, int64_t n, uint64_t key)
        {
            int64_t lo = 0;
            int64_t hi = n;
            while (lo < hi)
            {
                const int64_t mid = (lo + hi) >> 1;
                if (data[mid] < key)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            return static_cast<int32_t>(lo);
        }

        __global__ void build_synth_faces_kernel(
            int64_t num_triangles,
            int32_t *__restrict__ faces)
        {
            const int64_t fid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (fid >= num_triangles)
                return;
            faces[3 * fid + 0] = static_cast<int32_t>(3 * fid + 0);
            faces[3 * fid + 1] = static_cast<int32_t>(3 * fid + 1);
            faces[3 * fid + 2] = static_cast<int32_t>(3 * fid + 2);
        }

        __global__ void build_surface_keys_kernel(
            const int32_t *__restrict__ voxels,
            int64_t num_voxels,
            Int3 grid_min,
            Int3 grid_max,
            uint64_t *__restrict__ keys,
            int32_t *__restrict__ ids)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_voxels)
                return;
            const int x = voxels[3 * i + 0];
            const int y = voxels[3 * i + 1];
            const int z = voxels[3 * i + 2];
            keys[i] = pack_voxel_key(x, y, z, grid_min, grid_max);
            ids[i] = static_cast<int32_t>(i);
        }

        __global__ void build_raw_voxel_keys_kernel(
            const int32_t *__restrict__ raw_voxels,
            int64_t num_pairs,
            Int3 grid_min,
            Int3 grid_max,
            uint64_t *__restrict__ raw_keys)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_pairs)
                return;
            const int x = raw_voxels[3 * i + 0] + grid_min.x;
            const int y = raw_voxels[3 * i + 1] + grid_min.y;
            const int z = raw_voxels[3 * i + 2] + grid_min.z;
            raw_keys[i] = pack_voxel_key(x, y, z, grid_min, grid_max);
        }

        __global__ void map_raw_pairs_kernel(
            const uint64_t *__restrict__ raw_voxel_keys,
            const int32_t *__restrict__ raw_face_ids,
            int64_t num_pairs,
            const uint64_t *__restrict__ surface_keys,
            const int32_t *__restrict__ surface_ids,
            int64_t num_voxels,
            uint64_t *__restrict__ pair_keys_all,
            int32_t *__restrict__ valid)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_pairs)
                return;
            const uint64_t key = raw_voxel_keys[i];
            const int32_t pos = lower_bound_u64(surface_keys, num_voxels, key);
            if (pos < num_voxels && surface_keys[pos] == key)
            {
                pair_keys_all[i] = pack_pair_key(surface_ids[pos], raw_face_ids[i]);
                valid[i] = 1;
            }
            else
            {
                pair_keys_all[i] = 0;
                valid[i] = 0;
            }
        }

        __global__ void compact_pair_keys_kernel(
            const uint64_t *__restrict__ pair_keys_all,
            const int32_t *__restrict__ valid,
            const int32_t *__restrict__ offsets,
            int64_t num_pairs,
            uint64_t *__restrict__ pair_keys)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_pairs || valid[i] == 0)
                return;
            pair_keys[offsets[i]] = pair_keys_all[i];
        }

        __global__ void build_face_qef_contrib_kernel(
            const uint64_t *__restrict__ pair_keys,
            int64_t num_pairs,
            const float *__restrict__ triangles,
            int32_t *__restrict__ voxel_ids,
            SymQEF10 *__restrict__ qefs)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_pairs)
                return;
            const uint64_t key = pair_keys[i];
            const int32_t voxel_id = unpack_pair_voxel_id(key);
            const int32_t face_id = unpack_pair_face_id(key);
            const float *tri = triangles + 9 * static_cast<int64_t>(face_id);
            const float3 v0 = make_float3(tri[0], tri[1], tri[2]);
            const float3 v1 = make_float3(tri[3], tri[4], tri[5]);
            const float3 v2 = make_float3(tri[6], tri[7], tri[8]);
            const float3 e0 = sub3(v1, v0);
            const float3 e1 = sub3(v2, v1);
            const float3 n = normalize3(cross3(e0, e1));
            const float4 plane = make_float4(n.x, n.y, n.z, -dot3(n, v0));
            voxel_ids[i] = voxel_id;
            qefs[i] = qef_from_plane(plane);
        }

        __global__ void scatter_reduced_qefs_kernel(
            const int32_t *__restrict__ voxel_ids,
            const SymQEF10 *__restrict__ reduced_qefs,
            int64_t n,
            float *__restrict__ out_qefs)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const int32_t voxel_id = voxel_ids[i];
            const SymQEF10 q = reduced_qefs[i];
            float *dst = out_qefs + 10 * static_cast<int64_t>(voxel_id);
            dst[0] = q.q00;
            dst[1] = q.q01;
            dst[2] = q.q02;
            dst[3] = q.q03;
            dst[4] = q.q11;
            dst[5] = q.q12;
            dst[6] = q.q13;
            dst[7] = q.q22;
            dst[8] = q.q23;
            dst[9] = q.q33;
        }

    } // namespace

    torch::Tensor face_qef_old(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        const torch::Tensor &voxels)
    {
        TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");
        TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
        static_assert(sizeof(SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");

        const c10::cuda::CUDAGuard guard(triangles.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(triangles.get_device()).stream();
        const torch::Device device = triangles.device();
        const auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
        const auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(device);
        const auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);

        const int64_t num_triangles = triangles.size(0);
        const int64_t num_voxels = voxels.size(0);
        auto out_qefs = torch::zeros({num_voxels, 10}, opts_f32);
        if (num_triangles == 0 || num_voxels == 0)
            return out_qefs;

        TORCH_CHECK(num_triangles <= std::numeric_limits<int32_t>::max(), "face count exceeds int32 range");
        TORCH_CHECK(num_voxels <= std::numeric_limits<int>::max(), "surface voxel count exceeds CUB int range");
        const float *voxel_size_ptr = voxel_size.data_ptr<float>();
        const int32_t *grid_range_ptr = grid_range.data_ptr<int32_t>();
        const Int3 grid_min{grid_range_ptr[0], grid_range_ptr[1], grid_range_ptr[2]};
        const Int3 grid_max{grid_range_ptr[3], grid_range_ptr[4], grid_range_ptr[5]};

        auto flat_vertices = triangles.reshape({num_triangles * 3, 3});
        auto faces_synth = torch::empty({num_triangles, 3}, opts_i32);
        int blocks = static_cast<int>(div_up_i64(num_triangles, kThreads));
        build_synth_faces_kernel<<<blocks, kThreads, 0, stream>>>(
            num_triangles,
            faces_synth.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto raw = voxelize_mesh_octree(flat_vertices, faces_synth, voxel_size, grid_range);
        torch::Tensor raw_face_ids = std::get<0>(raw);
        torch::Tensor raw_voxels = std::get<1>(raw);
        const int64_t num_raw_pairs = raw_face_ids.size(0);
        if (num_raw_pairs == 0)
            return out_qefs;
        TORCH_CHECK(num_raw_pairs <= std::numeric_limits<int>::max(), "face voxel pair count exceeds CUB int range");

        auto surface_keys = torch::empty({num_voxels}, opts_u64);
        auto surface_ids = torch::empty({num_voxels}, opts_i32);
        blocks = static_cast<int>(div_up_i64(num_voxels, kThreads));
        build_surface_keys_kernel<<<blocks, kThreads, 0, stream>>>(
            voxels.data_ptr<int32_t>(),
            num_voxels,
            grid_min,
            grid_max,
            surface_keys.data_ptr<uint64_t>(),
            surface_ids.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto surface_keys_sorted = torch::empty({num_voxels}, opts_u64);
        auto surface_ids_sorted = torch::empty({num_voxels}, opts_i32);
        size_t temp_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_bytes,
            surface_keys.data_ptr<uint64_t>(),
            surface_keys_sorted.data_ptr<uint64_t>(),
            surface_ids.data_ptr<int32_t>(),
            surface_ids_sorted.data_ptr<int32_t>(),
            static_cast<int>(num_voxels),
            0,
            64,
            stream));
        auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            temp.data_ptr<uint8_t>(),
            temp_bytes,
            surface_keys.data_ptr<uint64_t>(),
            surface_keys_sorted.data_ptr<uint64_t>(),
            surface_ids.data_ptr<int32_t>(),
            surface_ids_sorted.data_ptr<int32_t>(),
            static_cast<int>(num_voxels),
            0,
            64,
            stream));

        auto raw_voxel_keys = torch::empty({num_raw_pairs}, opts_u64);
        blocks = static_cast<int>(div_up_i64(num_raw_pairs, kThreads));
        build_raw_voxel_keys_kernel<<<blocks, kThreads, 0, stream>>>(
            raw_voxels.data_ptr<int32_t>(),
            num_raw_pairs,
            grid_min,
            grid_max,
            raw_voxel_keys.data_ptr<uint64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto pair_keys_all = torch::empty({num_raw_pairs}, opts_u64);
        auto valid = torch::empty({num_raw_pairs}, opts_i32);
        map_raw_pairs_kernel<<<blocks, kThreads, 0, stream>>>(
            raw_voxel_keys.data_ptr<uint64_t>(),
            raw_face_ids.data_ptr<int32_t>(),
            num_raw_pairs,
            surface_keys_sorted.data_ptr<uint64_t>(),
            surface_ids_sorted.data_ptr<int32_t>(),
            num_voxels,
            pair_keys_all.data_ptr<uint64_t>(),
            valid.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto offsets = torch::empty({num_raw_pairs}, opts_i32);
        temp_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            temp_bytes,
            valid.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            static_cast<int>(num_raw_pairs),
            stream));
        temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            temp.data_ptr<uint8_t>(),
            temp_bytes,
            valid.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            static_cast<int>(num_raw_pairs),
            stream));
        const int64_t num_valid_pairs = read_compact_total_i32(valid, offsets, num_raw_pairs, stream);
        if (num_valid_pairs == 0)
            return out_qefs;

        auto pair_keys = torch::empty({num_valid_pairs}, opts_u64);
        compact_pair_keys_kernel<<<blocks, kThreads, 0, stream>>>(
            pair_keys_all.data_ptr<uint64_t>(),
            valid.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            num_raw_pairs,
            pair_keys.data_ptr<uint64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto pair_keys_sorted = torch::empty({num_valid_pairs}, opts_u64);
        temp_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            nullptr,
            temp_bytes,
            pair_keys.data_ptr<uint64_t>(),
            pair_keys_sorted.data_ptr<uint64_t>(),
            static_cast<int>(num_valid_pairs),
            0,
            64,
            stream));
        temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            temp.data_ptr<uint8_t>(),
            temp_bytes,
            pair_keys.data_ptr<uint64_t>(),
            pair_keys_sorted.data_ptr<uint64_t>(),
            static_cast<int>(num_valid_pairs),
            0,
            64,
            stream));

        auto unique_pair_keys = torch::empty({num_valid_pairs}, opts_u64);
        auto num_unique_t = torch::empty({1}, opts_i32);
        temp_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceSelect::Unique(
            nullptr,
            temp_bytes,
            pair_keys_sorted.data_ptr<uint64_t>(),
            unique_pair_keys.data_ptr<uint64_t>(),
            num_unique_t.data_ptr<int32_t>(),
            static_cast<int>(num_valid_pairs),
            stream));
        temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
        C10_CUDA_CHECK(cub::DeviceSelect::Unique(
            temp.data_ptr<uint8_t>(),
            temp_bytes,
            pair_keys_sorted.data_ptr<uint64_t>(),
            unique_pair_keys.data_ptr<uint64_t>(),
            num_unique_t.data_ptr<int32_t>(),
            static_cast<int>(num_valid_pairs),
            stream));
        const int64_t num_unique_pairs = read_i32(num_unique_t, stream);
        if (num_unique_pairs == 0)
            return out_qefs;

        auto contrib_voxel_ids = torch::empty({num_unique_pairs}, opts_i32);
        auto contrib_qefs = torch::empty({num_unique_pairs, 10}, opts_f32);
        blocks = static_cast<int>(div_up_i64(num_unique_pairs, kThreads));
        build_face_qef_contrib_kernel<<<blocks, kThreads, 0, stream>>>(
            unique_pair_keys.data_ptr<uint64_t>(),
            num_unique_pairs,
            triangles.data_ptr<float>(),
            contrib_voxel_ids.data_ptr<int32_t>(),
            reinterpret_cast<SymQEF10 *>(contrib_qefs.data_ptr<float>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto reduced_voxel_ids = torch::empty({num_unique_pairs}, opts_i32);
        auto reduced_qefs = torch::empty({num_unique_pairs, 10}, opts_f32);
        auto num_reduced_t = torch::empty({1}, opts_i32);
        temp_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
            nullptr,
            temp_bytes,
            contrib_voxel_ids.data_ptr<int32_t>(),
            reduced_voxel_ids.data_ptr<int32_t>(),
            reinterpret_cast<SymQEF10 *>(contrib_qefs.data_ptr<float>()),
            reinterpret_cast<SymQEF10 *>(reduced_qefs.data_ptr<float>()),
            num_reduced_t.data_ptr<int32_t>(),
            QEFAdd(),
            static_cast<int>(num_unique_pairs),
            stream));
        temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
        C10_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
            temp.data_ptr<uint8_t>(),
            temp_bytes,
            contrib_voxel_ids.data_ptr<int32_t>(),
            reduced_voxel_ids.data_ptr<int32_t>(),
            reinterpret_cast<SymQEF10 *>(contrib_qefs.data_ptr<float>()),
            reinterpret_cast<SymQEF10 *>(reduced_qefs.data_ptr<float>()),
            num_reduced_t.data_ptr<int32_t>(),
            QEFAdd(),
            static_cast<int>(num_unique_pairs),
            stream));
        const int64_t num_reduced = read_i32(num_reduced_t, stream);
        if (num_reduced == 0)
            return out_qefs;

        blocks = static_cast<int>(div_up_i64(num_reduced, kThreads));
        scatter_reduced_qefs_kernel<<<blocks, kThreads, 0, stream>>>(
            reduced_voxel_ids.data_ptr<int32_t>(),
            reinterpret_cast<const SymQEF10 *>(reduced_qefs.data_ptr<float>()),
            num_reduced,
            out_qefs.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out_qefs;
    }

} // namespace o_voxel::fdg
