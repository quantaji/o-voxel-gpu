#include "api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;
        using Key = int64_t;

        struct D2
        {
            double x;
            double z;
        };

        struct D3
        {
            double x;
            double y;
            double z;
        };

        struct alignas(16) QEFEventValue
        {
            float mean_sum_x;
            float mean_sum_y;
            float mean_sum_z;
            float cnt;
            SymQEF10 qef;
            uint32_t intersected;
        };

        static_assert(std::is_trivially_copyable<QEFEventValue>::value, "QEFEventValue must be trivially copyable");

        struct OccChunk
        {
            torch::Tensor keys;
            int64_t size = 0;
        };

        struct QEFChunk
        {
            torch::Tensor keys;
            torch::Tensor values_storage;
            int64_t size = 0;
        };

        struct AddQEFEventValue
        {
            __host__ __device__ QEFEventValue operator()(const QEFEventValue &a, const QEFEventValue &b) const
            {
                QEFEventValue out;
                out.mean_sum_x = a.mean_sum_x + b.mean_sum_x;
                out.mean_sum_y = a.mean_sum_y + b.mean_sum_y;
                out.mean_sum_z = a.mean_sum_z + b.mean_sum_z;
                out.cnt = a.cnt + b.cnt;
                out.qef = qef_add(a.qef, b.qef);
                out.intersected = a.intersected | b.intersected;
                return out;
            }
        };

        __host__ __device__ __forceinline__ double lerp_scalar(double a, double b, double t, double va, double vb)
        {
            if (a == b)
                return va;
            const double alpha = (t - a) / (b - a);
            return (1.0 - alpha) * va + alpha * vb;
        }

        __host__ __device__ __forceinline__ D2 lerp_vec2(double a, double b, double t, D2 va, D2 vb)
        {
            if (a == b)
                return va;
            const double alpha = (t - a) / (b - a);
            return D2{(1.0 - alpha) * va.x + alpha * vb.x, (1.0 - alpha) * va.z + alpha * vb.z};
        }

        __host__ __device__ __forceinline__ int clamp_int(int x, int lo, int hi)
        {
            return x < lo ? lo : (x > hi ? hi : x);
        }

        __host__ __device__ __forceinline__ void sort_by_y(D3 &t0, D3 &t1, D3 &t2)
        {
            if (t0.y > t1.y)
            {
                const D3 t = t0;
                t0 = t1;
                t1 = t;
            }
            if (t1.y > t2.y)
            {
                const D3 t = t1;
                t1 = t2;
                t2 = t;
            }
            if (t0.y > t1.y)
            {
                const D3 t = t0;
                t0 = t1;
                t1 = t;
            }
        }

        __device__ __forceinline__ void normalize3(double &x, double &y, double &z)
        {
            const double n = sqrt(x * x + y * y + z * z);
            if (n > 0.0)
            {
                x /= n;
                y /= n;
                z /= n;
            }
        }

        __device__ __forceinline__ SymQEF10 make_plane_qef_from_triangle(
            const double v0[3],
            const double v1[3],
            const double v2[3])
        {
            const double e0x = v1[0] - v0[0];
            const double e0y = v1[1] - v0[1];
            const double e0z = v1[2] - v0[2];

            const double e1x = v2[0] - v1[0];
            const double e1y = v2[1] - v1[1];
            const double e1z = v2[2] - v1[2];

            double nx = e0y * e1z - e0z * e1y;
            double ny = e0z * e1x - e0x * e1z;
            double nz = e0x * e1y - e0y * e1x;
            normalize3(nx, ny, nz);

            const double d = -(nx * v0[0] + ny * v0[1] + nz * v0[2]);
            return SymQEF10{
                static_cast<float>(nx * nx),
                static_cast<float>(nx * ny),
                static_cast<float>(nx * nz),
                static_cast<float>(nx * d),
                static_cast<float>(ny * ny),
                static_cast<float>(ny * nz),
                static_cast<float>(ny * d),
                static_cast<float>(nz * nz),
                static_cast<float>(nz * d),
                static_cast<float>(d * d),
            };
        }

        template <typename Emit>
        __device__ __forceinline__ void scan_triangle_events(
            const float *tri,
            int ax2,
            const float voxel_size[3],
            Int3 grid_min,
            Int3 grid_max,
            Emit emit)
        {
            const double v0[3] = {static_cast<double>(tri[0]), static_cast<double>(tri[1]), static_cast<double>(tri[2])};
            const double v1[3] = {static_cast<double>(tri[3]), static_cast<double>(tri[4]), static_cast<double>(tri[5])};
            const double v2[3] = {static_cast<double>(tri[6]), static_cast<double>(tri[7]), static_cast<double>(tri[8])};

            const int ax0 = (ax2 + 1) % 3;
            const int ax1 = (ax2 + 2) % 3;

            D3 t0{v0[ax0], v0[ax1], v0[ax2]};
            D3 t1{v1[ax0], v1[ax1], v1[ax2]};
            D3 t2{v2[ax0], v2[ax1], v2[ax2]};
            sort_by_y(t0, t1, t2);

            const int start = clamp_int(static_cast<int>(t0.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            const int mid = clamp_int(static_cast<int>(t1.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            const int end = clamp_int(static_cast<int>(t2.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);

            auto scan_half = [&](int row_start, int row_end, D3 a, D3 b, D3 c)
            {
                for (int y_idx = row_start; y_idx < row_end; ++y_idx)
                {
                    const double y = (static_cast<double>(y_idx) + 1.0) * voxel_size[ax1];
                    D2 t3 = lerp_vec2(a.y, b.y, y, D2{a.x, a.z}, D2{b.x, b.z});
                    D2 t4 = lerp_vec2(a.y, c.y, y, D2{a.x, a.z}, D2{c.x, c.z});
                    if (t3.x > t4.x)
                    {
                        const D2 tmp = t3;
                        t3 = t4;
                        t4 = tmp;
                    }

                    const int line_start = clamp_int(static_cast<int>(t3.x / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
                    const int line_end = clamp_int(static_cast<int>(t4.x / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
                    for (int x_idx = line_start; x_idx < line_end; ++x_idx)
                    {
                        const double x = (static_cast<double>(x_idx) + 1.0) * voxel_size[ax0];
                        const double z = lerp_scalar(t3.x, t4.x, x, t3.z, t4.z);
                        const int z_idx = static_cast<int>(z / voxel_size[ax2]);
                        if (z_idx < grid_min[ax2] || z_idx >= grid_max[ax2])
                            continue;
                        emit(ax0, ax1, ax2, x_idx, y_idx, z_idx, x, y, z);
                    }
                }
            };

            scan_half(start, mid, t0, t1, t2);
            scan_half(mid, end, t2, t1, t0);
        }

        __global__ void count_events_kernel(
            const float *triangles,
            int64_t tri_begin,
            int64_t tri_count,
            float3 voxel_size,
            Int3 grid_min,
            Int3 grid_max,
            int64_t *counts)
        {
            const int64_t local_t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (local_t >= tri_count)
                return;

            const float *tri = triangles + (tri_begin + local_t) * 9;
            const float voxel_size_v[3] = {voxel_size.x, voxel_size.y, voxel_size.z};

            int64_t total = 0;
            auto count = [&](int, int, int, int, int, int, double, double, double)
            {
                total += 4;
            };
            scan_triangle_events(tri, 0, voxel_size_v, grid_min, grid_max, count);
            scan_triangle_events(tri, 1, voxel_size_v, grid_min, grid_max, count);
            scan_triangle_events(tri, 2, voxel_size_v, grid_min, grid_max, count);
            counts[local_t] = total;
        }

        __global__ void emit_occ_keys_kernel(
            const float *triangles,
            int64_t tri_begin,
            int64_t tri_count,
            float3 voxel_size,
            Int3 grid_min,
            Int3 grid_max,
            const int64_t *offsets,
            Key *event_keys)
        {
            const int64_t local_t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (local_t >= tri_count)
                return;

            const float *tri = triangles + (tri_begin + local_t) * 9;
            const float voxel_size_v[3] = {voxel_size.x, voxel_size.y, voxel_size.z};

            int64_t out = offsets[local_t];
            auto emit = [&](int ax0, int ax1, int ax2, int x_idx, int y_idx, int z_idx, double, double, double)
            {
                int coord[3];
                auto emit_one = [&](int vx, int vy, int vz)
                {
                    coord[ax0] = vx;
                    coord[ax1] = vy;
                    coord[ax2] = vz;
                    event_keys[out++] = static_cast<Key>(pack_voxel_key(coord[0], coord[1], coord[2], grid_min, grid_max));
                };

                emit_one(x_idx + 0, y_idx + 0, z_idx);
                emit_one(x_idx + 1, y_idx + 0, z_idx);
                emit_one(x_idx + 0, y_idx + 1, z_idx);
                emit_one(x_idx + 1, y_idx + 1, z_idx);
            };
            scan_triangle_events(tri, 0, voxel_size_v, grid_min, grid_max, emit);
            scan_triangle_events(tri, 1, voxel_size_v, grid_min, grid_max, emit);
            scan_triangle_events(tri, 2, voxel_size_v, grid_min, grid_max, emit);
        }

        __global__ void emit_qef_events_kernel(
            const float *triangles,
            int64_t tri_begin,
            int64_t tri_count,
            float3 voxel_size,
            Int3 grid_min,
            Int3 grid_max,
            const int64_t *offsets,
            Key *event_keys,
            QEFEventValue *event_values)
        {
            const int64_t local_t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (local_t >= tri_count)
                return;

            const float *tri = triangles + (tri_begin + local_t) * 9;
            const double v0[3] = {static_cast<double>(tri[0]), static_cast<double>(tri[1]), static_cast<double>(tri[2])};
            const double v1[3] = {static_cast<double>(tri[3]), static_cast<double>(tri[4]), static_cast<double>(tri[5])};
            const double v2[3] = {static_cast<double>(tri[6]), static_cast<double>(tri[7]), static_cast<double>(tri[8])};
            const float voxel_size_v[3] = {voxel_size.x, voxel_size.y, voxel_size.z};
            const SymQEF10 qef = make_plane_qef_from_triangle(v0, v1, v2);

            int64_t out = offsets[local_t];
            auto emit = [&](int ax0, int ax1, int ax2, int x_idx, int y_idx, int z_idx, double x, double y, double z)
            {
                auto emit_one = [&](int x_idx, int y_idx, int z_idx, double x, double y, double z, uint32_t mask)
                {
                    int coord[3];
                    coord[ax0] = x_idx;
                    coord[ax1] = y_idx;
                    coord[ax2] = z_idx;

                    event_keys[out] = static_cast<Key>(pack_voxel_key(coord[0], coord[1], coord[2], grid_min, grid_max));
                    event_values[out].mean_sum_x = static_cast<float>(ax0 == 0 ? x : (ax1 == 0 ? y : z));
                    event_values[out].mean_sum_y = static_cast<float>(ax0 == 1 ? x : (ax1 == 1 ? y : z));
                    event_values[out].mean_sum_z = static_cast<float>(ax0 == 2 ? x : (ax1 == 2 ? y : z));
                    event_values[out].cnt = 1.0f;
                    event_values[out].qef = qef;
                    event_values[out].intersected = mask;
                    ++out;
                };

                emit_one(x_idx + 0, y_idx + 0, z_idx, x, y, z, static_cast<uint32_t>(1u << ax2));
                emit_one(x_idx + 1, y_idx + 0, z_idx, x, y, z, 0u);
                emit_one(x_idx + 0, y_idx + 1, z_idx, x, y, z, 0u);
                emit_one(x_idx + 1, y_idx + 1, z_idx, x, y, z, 0u);
            };
            scan_triangle_events(tri, 0, voxel_size_v, grid_min, grid_max, emit);
            scan_triangle_events(tri, 1, voxel_size_v, grid_min, grid_max, emit);
            scan_triangle_events(tri, 2, voxel_size_v, grid_min, grid_max, emit);
        }

        __global__ void decode_occ_keys_kernel(
            const Key *keys,
            int64_t size,
            Int3 grid_min,
            Int3 grid_max,
            int32_t *out_voxels)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= size)
                return;

            const Int3 voxel = unpack_voxel_key(static_cast<uint64_t>(keys[i]), grid_min, grid_max);
            out_voxels[3 * i + 0] = voxel.x;
            out_voxels[3 * i + 1] = voxel.y;
            out_voxels[3 * i + 2] = voxel.z;
        }

        __global__ void decode_qef_values_kernel(
            const Key *keys,
            const QEFEventValue *values,
            int64_t size,
            Int3 grid_min,
            Int3 grid_max,
            int32_t *out_voxels,
            float *out_mean_sum,
            float *out_cnt,
            bool *out_intersected,
            float *out_qefs)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= size)
                return;

            const Int3 voxel = unpack_voxel_key(static_cast<uint64_t>(keys[i]), grid_min, grid_max);
            out_voxels[3 * i + 0] = voxel.x;
            out_voxels[3 * i + 1] = voxel.y;
            out_voxels[3 * i + 2] = voxel.z;

            const QEFEventValue v = values[i];
            out_mean_sum[3 * i + 0] = v.mean_sum_x;
            out_mean_sum[3 * i + 1] = v.mean_sum_y;
            out_mean_sum[3 * i + 2] = v.mean_sum_z;
            out_cnt[i] = v.cnt;
            out_intersected[3 * i + 0] = (v.intersected & (1u << 0)) != 0;
            out_intersected[3 * i + 1] = (v.intersected & (1u << 1)) != 0;
            out_intersected[3 * i + 2] = (v.intersected & (1u << 2)) != 0;

            out_qefs[10 * i + 0] = v.qef.q00;
            out_qefs[10 * i + 1] = v.qef.q01;
            out_qefs[10 * i + 2] = v.qef.q02;
            out_qefs[10 * i + 3] = v.qef.q03;
            out_qefs[10 * i + 4] = v.qef.q11;
            out_qefs[10 * i + 5] = v.qef.q12;
            out_qefs[10 * i + 6] = v.qef.q13;
            out_qefs[10 * i + 7] = v.qef.q22;
            out_qefs[10 * i + 8] = v.qef.q23;
            out_qefs[10 * i + 9] = v.qef.q33;
        }

        int64_t read_raw_size(const torch::Tensor &counts, const torch::Tensor &offsets, int64_t n, cudaStream_t stream)
        {
            if (n == 0)
                return 0;

            int64_t last_count = 0;
            int64_t last_offset = 0;
            C10_CUDA_CHECK(cudaMemcpyAsync(
                &last_count,
                counts.data_ptr<int64_t>() + n - 1,
                sizeof(int64_t),
                cudaMemcpyDeviceToHost,
                stream));
            C10_CUDA_CHECK(cudaMemcpyAsync(
                &last_offset,
                offsets.data_ptr<int64_t>() + n - 1,
                sizeof(int64_t),
                cudaMemcpyDeviceToHost,
                stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            return last_offset + last_count;
        }

        int read_i32(const torch::Tensor &t, cudaStream_t stream)
        {
            int value = 0;
            C10_CUDA_CHECK(cudaMemcpyAsync(&value, t.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            return value;
        }

        void cub_exclusive_sum(const torch::Tensor &in, const torch::Tensor &out, int64_t n, cudaStream_t stream)
        {
            if (n == 0)
                return;

            size_t temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                nullptr,
                temp_bytes,
                in.data_ptr<int64_t>(),
                out.data_ptr<int64_t>(),
                static_cast<int>(n),
                stream));

            torch::Tensor temp = torch::empty(
                {static_cast<int64_t>(temp_bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(in.device()));
            C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                temp.data_ptr<uint8_t>(),
                temp_bytes,
                in.data_ptr<int64_t>(),
                out.data_ptr<int64_t>(),
                static_cast<int>(n),
                stream));
        }

        torch::Tensor cub_sort_keys(const torch::Tensor &keys, int64_t n, cudaStream_t stream)
        {
            auto out = torch::empty({n}, keys.options());
            if (n == 0)
                return out;

            size_t temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
                nullptr,
                temp_bytes,
                keys.data_ptr<Key>(),
                out.data_ptr<Key>(),
                static_cast<int>(n),
                0,
                sizeof(Key) * 8,
                stream));

            torch::Tensor temp = torch::empty(
                {static_cast<int64_t>(temp_bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(keys.device()));
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
                temp.data_ptr<uint8_t>(),
                temp_bytes,
                keys.data_ptr<Key>(),
                out.data_ptr<Key>(),
                static_cast<int>(n),
                0,
                sizeof(Key) * 8,
                stream));
            return out;
        }

        OccChunk cub_unique_keys(const torch::Tensor &sorted_keys, int64_t n, cudaStream_t stream)
        {
            OccChunk out;
            out.keys = torch::empty({n}, sorted_keys.options());
            if (n == 0)
                return out;

            auto num_selected = torch::empty({1}, sorted_keys.options().dtype(torch::kInt32));
            size_t temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceSelect::Unique(
                nullptr,
                temp_bytes,
                sorted_keys.data_ptr<Key>(),
                out.keys.data_ptr<Key>(),
                num_selected.data_ptr<int>(),
                static_cast<int>(n),
                stream));

            torch::Tensor temp = torch::empty(
                {static_cast<int64_t>(temp_bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(sorted_keys.device()));
            C10_CUDA_CHECK(cub::DeviceSelect::Unique(
                temp.data_ptr<uint8_t>(),
                temp_bytes,
                sorted_keys.data_ptr<Key>(),
                out.keys.data_ptr<Key>(),
                num_selected.data_ptr<int>(),
                static_cast<int>(n),
                stream));
            out.size = read_i32(num_selected, stream);
            return out;
        }

        QEFChunk cub_sort_reduce_pairs(
            const torch::Tensor &keys,
            const torch::Tensor &values_storage,
            int64_t n,
            cudaStream_t stream)
        {
            QEFChunk out;
            out.keys = torch::empty({n}, keys.options());
            out.values_storage = torch::empty(
                {static_cast<int64_t>(sizeof(QEFEventValue)) * n},
                torch::TensorOptions().dtype(torch::kUInt8).device(keys.device()));
            if (n == 0)
                return out;

            torch::Tensor sorted_keys = torch::empty({n}, keys.options());
            torch::Tensor sorted_values_storage = torch::empty(
                {static_cast<int64_t>(sizeof(QEFEventValue)) * n},
                torch::TensorOptions().dtype(torch::kUInt8).device(keys.device()));
            const auto values_in = reinterpret_cast<const QEFEventValue *>(values_storage.data_ptr<uint8_t>());
            const auto sorted_values = reinterpret_cast<QEFEventValue *>(sorted_values_storage.data_ptr<uint8_t>());
            const auto values_out = reinterpret_cast<QEFEventValue *>(out.values_storage.data_ptr<uint8_t>());

            size_t sort_temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                nullptr,
                sort_temp_bytes,
                keys.data_ptr<Key>(),
                sorted_keys.data_ptr<Key>(),
                values_in,
                sorted_values,
                static_cast<int>(n),
                0,
                sizeof(Key) * 8,
                stream));

            torch::Tensor sort_temp = torch::empty(
                {static_cast<int64_t>(sort_temp_bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(keys.device()));
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                sort_temp.data_ptr<uint8_t>(),
                sort_temp_bytes,
                keys.data_ptr<Key>(),
                sorted_keys.data_ptr<Key>(),
                values_in,
                sorted_values,
                static_cast<int>(n),
                0,
                sizeof(Key) * 8,
                stream));

            auto num_runs = torch::empty({1}, keys.options().dtype(torch::kInt32));
            size_t reduce_temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
                nullptr,
                reduce_temp_bytes,
                sorted_keys.data_ptr<Key>(),
                out.keys.data_ptr<Key>(),
                sorted_values,
                values_out,
                num_runs.data_ptr<int>(),
                AddQEFEventValue{},
                static_cast<int>(n),
                stream));

            torch::Tensor reduce_temp = torch::empty(
                {static_cast<int64_t>(reduce_temp_bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(keys.device()));
            C10_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
                reduce_temp.data_ptr<uint8_t>(),
                reduce_temp_bytes,
                sorted_keys.data_ptr<Key>(),
                out.keys.data_ptr<Key>(),
                sorted_values,
                values_out,
                num_runs.data_ptr<int>(),
                AddQEFEventValue{},
                static_cast<int>(n),
                stream));
            out.size = read_i32(num_runs, stream);
            return out;
        }

        OccChunk build_occ_chunk(
            const float *triangles,
            int64_t tri_begin,
            int64_t tri_count,
            float3 voxel_size,
            Int3 grid_min,
            Int3 grid_max,
            const torch::Device &device,
            cudaStream_t stream)
        {
            auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
            auto counts = torch::empty({tri_count}, opts_i64);
            auto offsets = torch::empty({tri_count}, opts_i64);

            count_events_kernel<<<static_cast<int>((tri_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
                triangles,
                tri_begin,
                tri_count,
                voxel_size,
                grid_min,
                grid_max,
                counts.data_ptr<int64_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            cub_exclusive_sum(counts, offsets, tri_count, stream);
            const int64_t raw_size = read_raw_size(counts, offsets, tri_count, stream);
            if (raw_size == 0)
                return OccChunk{};

            auto raw_keys = torch::empty({raw_size}, opts_i64);
            emit_occ_keys_kernel<<<static_cast<int>((tri_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
                triangles,
                tri_begin,
                tri_count,
                voxel_size,
                grid_min,
                grid_max,
                offsets.data_ptr<int64_t>(),
                raw_keys.data_ptr<Key>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            torch::Tensor sorted_keys = cub_sort_keys(raw_keys, raw_size, stream);
            return cub_unique_keys(sorted_keys, raw_size, stream);
        }

        QEFChunk build_qef_chunk(
            const float *triangles,
            int64_t tri_begin,
            int64_t tri_count,
            float3 voxel_size,
            Int3 grid_min,
            Int3 grid_max,
            const torch::Device &device,
            cudaStream_t stream)
        {
            auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
            auto counts = torch::empty({tri_count}, opts_i64);
            auto offsets = torch::empty({tri_count}, opts_i64);

            count_events_kernel<<<static_cast<int>((tri_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
                triangles,
                tri_begin,
                tri_count,
                voxel_size,
                grid_min,
                grid_max,
                counts.data_ptr<int64_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            cub_exclusive_sum(counts, offsets, tri_count, stream);
            const int64_t raw_size = read_raw_size(counts, offsets, tri_count, stream);
            if (raw_size == 0)
                return QEFChunk{};

            auto raw_keys = torch::empty({raw_size}, opts_i64);
            auto raw_values_storage = torch::empty(
                {static_cast<int64_t>(sizeof(QEFEventValue)) * raw_size},
                torch::TensorOptions().dtype(torch::kUInt8).device(device));
            auto raw_values = reinterpret_cast<QEFEventValue *>(raw_values_storage.data_ptr<uint8_t>());

            emit_qef_events_kernel<<<static_cast<int>((tri_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
                triangles,
                tri_begin,
                tri_count,
                voxel_size,
                grid_min,
                grid_max,
                offsets.data_ptr<int64_t>(),
                raw_keys.data_ptr<Key>(),
                raw_values);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            return cub_sort_reduce_pairs(raw_keys, raw_values_storage, raw_size, stream);
        }

        OccChunk merge_occ_chunks(std::vector<OccChunk> chunks, const torch::Device &device, cudaStream_t stream)
        {
            int64_t total = 0;
            for (const OccChunk &chunk : chunks)
                total += chunk.size;
            if (total == 0)
                return OccChunk{};

            auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
            auto keys = torch::empty({total}, opts_i64);

            int64_t offset = 0;
            for (const OccChunk &chunk : chunks)
            {
                if (chunk.size == 0)
                    continue;
                C10_CUDA_CHECK(cudaMemcpyAsync(
                    keys.data_ptr<Key>() + offset,
                    chunk.keys.data_ptr<Key>(),
                    static_cast<size_t>(chunk.size) * sizeof(Key),
                    cudaMemcpyDeviceToDevice,
                    stream));
                offset += chunk.size;
            }

            if (chunks.size() == 1)
                return OccChunk{keys, total};

            torch::Tensor sorted_keys = cub_sort_keys(keys, total, stream);
            return cub_unique_keys(sorted_keys, total, stream);
        }

        QEFChunk merge_qef_chunks(std::vector<QEFChunk> chunks, const torch::Device &device, cudaStream_t stream)
        {
            int64_t total = 0;
            for (const QEFChunk &chunk : chunks)
                total += chunk.size;
            if (total == 0)
                return QEFChunk{};

            auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
            auto keys = torch::empty({total}, opts_i64);
            auto values_storage = torch::empty(
                {static_cast<int64_t>(sizeof(QEFEventValue)) * total},
                torch::TensorOptions().dtype(torch::kUInt8).device(device));

            int64_t offset = 0;
            QEFEventValue *values = reinterpret_cast<QEFEventValue *>(values_storage.data_ptr<uint8_t>());
            for (const QEFChunk &chunk : chunks)
            {
                if (chunk.size == 0)
                    continue;
                C10_CUDA_CHECK(cudaMemcpyAsync(
                    keys.data_ptr<Key>() + offset,
                    chunk.keys.data_ptr<Key>(),
                    static_cast<size_t>(chunk.size) * sizeof(Key),
                    cudaMemcpyDeviceToDevice,
                    stream));
                C10_CUDA_CHECK(cudaMemcpyAsync(
                    values + offset,
                    reinterpret_cast<const QEFEventValue *>(chunk.values_storage.data_ptr<uint8_t>()),
                    static_cast<size_t>(chunk.size) * sizeof(QEFEventValue),
                    cudaMemcpyDeviceToDevice,
                    stream));
                offset += chunk.size;
            }

            if (chunks.size() == 1)
                return QEFChunk{keys, values_storage, total};

            return cub_sort_reduce_pairs(keys, values_storage, total, stream);
        }

        torch::Tensor decode_occ_chunk(OccChunk chunk, Int3 grid_min, Int3 grid_max, const torch::Device &device, cudaStream_t stream)
        {
            auto voxels = torch::empty(
                {chunk.size, 3},
                torch::TensorOptions().dtype(torch::kInt32).device(device));
            if (chunk.size == 0)
                return voxels;

            decode_occ_keys_kernel<<<static_cast<int>((chunk.size + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
                chunk.keys.data_ptr<Key>(),
                chunk.size,
                grid_min,
                grid_max,
                voxels.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return voxels;
        }

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        decode_qef_chunk(QEFChunk chunk, Int3 grid_min, Int3 grid_max, const torch::Device &device, cudaStream_t stream)
        {
            auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
            auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
            auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(device);

            auto voxels = torch::empty({chunk.size, 3}, opts_i32);
            auto mean_sum = torch::empty({chunk.size, 3}, opts_f32);
            auto cnt = torch::empty({chunk.size}, opts_f32);
            auto intersected = torch::empty({chunk.size, 3}, opts_bool);
            auto qefs = torch::empty({chunk.size, 10}, opts_f32);
            if (chunk.size == 0)
                return std::make_tuple(voxels, mean_sum, cnt, intersected, qefs);

            decode_qef_values_kernel<<<static_cast<int>((chunk.size + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
                chunk.keys.data_ptr<Key>(),
                reinterpret_cast<const QEFEventValue *>(chunk.values_storage.data_ptr<uint8_t>()),
                chunk.size,
                grid_min,
                grid_max,
                voxels.data_ptr<int32_t>(),
                mean_sum.data_ptr<float>(),
                cnt.data_ptr<float>(),
                intersected.data_ptr<bool>(),
                qefs.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return std::make_tuple(voxels, mean_sum, cnt, intersected, qefs);
        }

    } // namespace

    torch::Tensor intersect_occ_old(
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

        const float *voxel_size_ptr = voxel_size.data_ptr<float>();
        const int32_t *grid_range_ptr = grid_range.data_ptr<int32_t>();
        const float3 voxel_size_h = float3{voxel_size_ptr[0], voxel_size_ptr[1], voxel_size_ptr[2]};
        const Int3 grid_min{grid_range_ptr[0], grid_range_ptr[1], grid_range_ptr[2]};
        const Int3 grid_max{grid_range_ptr[3], grid_range_ptr[4], grid_range_ptr[5]};

        const int64_t num_triangles = triangles.size(0);
        std::vector<OccChunk> chunks;
        chunks.reserve(static_cast<size_t>((num_triangles + chunk_triangles - 1) / chunk_triangles));

        for (int64_t tri_begin = 0; tri_begin < num_triangles; tri_begin += chunk_triangles)
        {
            const int64_t tri_count = std::min<int64_t>(chunk_triangles, num_triangles - tri_begin);
            OccChunk chunk = build_occ_chunk(
                triangles.data_ptr<float>(),
                tri_begin,
                tri_count,
                voxel_size_h,
                grid_min,
                grid_max,
                device,
                stream);
            if (chunk.size > 0)
                chunks.push_back(std::move(chunk));
        }

        return decode_occ_chunk(merge_occ_chunks(std::move(chunks), device, stream), grid_min, grid_max, device, stream);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    intersect_qef_old(
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

        const float *voxel_size_ptr = voxel_size.data_ptr<float>();
        const int32_t *grid_range_ptr = grid_range.data_ptr<int32_t>();
        const float3 voxel_size_h = float3{voxel_size_ptr[0], voxel_size_ptr[1], voxel_size_ptr[2]};
        const Int3 grid_min{grid_range_ptr[0], grid_range_ptr[1], grid_range_ptr[2]};
        const Int3 grid_max{grid_range_ptr[3], grid_range_ptr[4], grid_range_ptr[5]};

        const int64_t num_triangles = triangles.size(0);
        std::vector<QEFChunk> chunks;
        chunks.reserve(static_cast<size_t>((num_triangles + chunk_triangles - 1) / chunk_triangles));

        for (int64_t tri_begin = 0; tri_begin < num_triangles; tri_begin += chunk_triangles)
        {
            const int64_t tri_count = std::min<int64_t>(chunk_triangles, num_triangles - tri_begin);
            QEFChunk chunk = build_qef_chunk(
                triangles.data_ptr<float>(),
                tri_begin,
                tri_count,
                voxel_size_h,
                grid_min,
                grid_max,
                device,
                stream);
            if (chunk.size > 0)
                chunks.push_back(std::move(chunk));
        }

        return decode_qef_chunk(merge_qef_chunks(std::move(chunks), device, stream), grid_min, grid_max, device, stream);
    }

} // namespace o_voxel::fdg
