#include "api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;

        struct EdgeDesc
        {
            double segment_length;
            int32_t start_x;
            int32_t start_y;
            int32_t start_z;
            int8_t step_x;
            int8_t step_y;
            int8_t step_z;
            uint8_t valid;
            double tmax0_x;
            double tmax0_y;
            double tmax0_z;
            double tdelta_x;
            double tdelta_y;
            double tdelta_z;
        };

        struct QEFAdd
        {
            __host__ __device__ SymQEF10 operator()(const SymQEF10 &a, const SymQEF10 &b) const
            {
                return qef_add(a, b);
            }
        };

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
            if (n == 0)
                return 0;
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

        __host__ __device__ __forceinline__ uint64_t lex_voxel_key(int x, int y, int z, Int3 grid_min, Int3 grid_max)
        {
            const uint64_t sy = static_cast<uint64_t>(grid_max.y - grid_min.y);
            const uint64_t sz = static_cast<uint64_t>(grid_max.z - grid_min.z);
            const uint64_t rx = static_cast<uint64_t>(x - grid_min.x);
            const uint64_t ry = static_cast<uint64_t>(y - grid_min.y);
            const uint64_t rz = static_cast<uint64_t>(z - grid_min.z);
            return rx * sy * sz + ry * sz + rz;
        }

        __host__ __device__ __forceinline__ Int3 decode_lex_voxel_key(uint64_t key, Int3 grid_min, Int3 grid_max)
        {
            const uint64_t sy = static_cast<uint64_t>(grid_max.y - grid_min.y);
            const uint64_t sz = static_cast<uint64_t>(grid_max.z - grid_min.z);
            const uint64_t yz = sy * sz;
            const uint64_t x = key / yz;
            const uint64_t rem = key - x * yz;
            const uint64_t y = rem / sz;
            const uint64_t z = rem - y * sz;
            return Int3{
                static_cast<int>(x) + grid_min.x,
                static_cast<int>(y) + grid_min.y,
                static_cast<int>(z) + grid_min.z,
            };
        }

        __host__ __device__ __forceinline__ uint64_t pack_pair_key(int32_t voxel_id, int32_t edge_id)
        {
            return (static_cast<uint64_t>(static_cast<uint32_t>(voxel_id)) << 32) |
                   static_cast<uint32_t>(edge_id);
        }

        __host__ __device__ __forceinline__ int32_t unpack_pair_voxel_id(uint64_t key)
        {
            return static_cast<int32_t>(key >> 32);
        }

        __host__ __device__ __forceinline__ int32_t unpack_pair_edge_id(uint64_t key)
        {
            return static_cast<int32_t>(key & 0xffffffffu);
        }

        __device__ __forceinline__ bool in_bounds(int x, int y, int z, Int3 grid_min, Int3 grid_max)
        {
            return grid_min.x <= x && x < grid_max.x &&
                   grid_min.y <= y && y < grid_max.y &&
                   grid_min.z <= z && z < grid_max.z;
        }

        __device__ __forceinline__ int argmin_axis_strict(double tx, double ty, double tz)
        {
            if (tx < ty)
                return (tx < tz) ? 0 : 2;
            return (ty < tz) ? 1 : 2;
        }

        __device__ __forceinline__ EdgeDesc make_edge_desc(float3 v0, float3 v1, float3 voxel_size)
        {
            EdgeDesc desc{};
            const double dx = static_cast<double>(v1.x) - static_cast<double>(v0.x);
            const double dy = static_cast<double>(v1.y) - static_cast<double>(v0.y);
            const double dz = static_cast<double>(v1.z) - static_cast<double>(v0.z);
            const double segment_length = sqrt(dx * dx + dy * dy + dz * dz);
            if (segment_length < 1e-6)
                return desc;

            const double dir_x = dx / segment_length;
            const double dir_y = dy / segment_length;
            const double dir_z = dz / segment_length;
            const int32_t sx = static_cast<int32_t>(floor(static_cast<double>(v0.x) / static_cast<double>(voxel_size.x)));
            const int32_t sy = static_cast<int32_t>(floor(static_cast<double>(v0.y) / static_cast<double>(voxel_size.y)));
            const int32_t sz = static_cast<int32_t>(floor(static_cast<double>(v0.z) / static_cast<double>(voxel_size.z)));
            const int8_t step_x = (dir_x > 0.0) ? 1 : -1;
            const int8_t step_y = (dir_y > 0.0) ? 1 : -1;
            const int8_t step_z = (dir_z > 0.0) ? 1 : -1;

            desc.valid = 1;
            desc.segment_length = segment_length;
            desc.start_x = sx;
            desc.start_y = sy;
            desc.start_z = sz;
            desc.step_x = step_x;
            desc.step_y = step_y;
            desc.step_z = step_z;

            if (dir_x == 0.0)
            {
                desc.tmax0_x = CUDART_INF;
                desc.tdelta_x = CUDART_INF;
            }
            else
            {
                const double border = static_cast<double>(voxel_size.x) * static_cast<double>(sx + (step_x > 0 ? 1 : 0));
                desc.tmax0_x = (border - static_cast<double>(v0.x)) / dir_x;
                desc.tdelta_x = static_cast<double>(voxel_size.x) / fabs(dir_x);
            }

            if (dir_y == 0.0)
            {
                desc.tmax0_y = CUDART_INF;
                desc.tdelta_y = CUDART_INF;
            }
            else
            {
                const double border = static_cast<double>(voxel_size.y) * static_cast<double>(sy + (step_y > 0 ? 1 : 0));
                desc.tmax0_y = (border - static_cast<double>(v0.y)) / dir_y;
                desc.tdelta_y = static_cast<double>(voxel_size.y) / fabs(dir_y);
            }

            if (dir_z == 0.0)
            {
                desc.tmax0_z = CUDART_INF;
                desc.tdelta_z = CUDART_INF;
            }
            else
            {
                const double border = static_cast<double>(voxel_size.z) * static_cast<double>(sz + (step_z > 0 ? 1 : 0));
                desc.tmax0_z = (border - static_cast<double>(v0.z)) / dir_z;
                desc.tdelta_z = static_cast<double>(voxel_size.z) / fabs(dir_z);
            }
            return desc;
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

        __global__ void build_indexed_edge_desc_and_jobs_kernel(
            const float *__restrict__ vertices,
            const int32_t *__restrict__ edges,
            int64_t num_edges,
            float3 voxel_size,
            EdgeDesc *__restrict__ edge_desc,
            int32_t *__restrict__ job_edge,
            int32_t *__restrict__ job_xyz,
            double *__restrict__ job_tmax)
        {
            const int64_t eid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (eid >= num_edges)
                return;

            const int32_t v0_id = edges[2 * eid + 0];
            const int32_t v1_id = edges[2 * eid + 1];
            const float3 v0 = make_float3(vertices[3 * static_cast<int64_t>(v0_id) + 0],
                                          vertices[3 * static_cast<int64_t>(v0_id) + 1],
                                          vertices[3 * static_cast<int64_t>(v0_id) + 2]);
            const float3 v1 = make_float3(vertices[3 * static_cast<int64_t>(v1_id) + 0],
                                          vertices[3 * static_cast<int64_t>(v1_id) + 1],
                                          vertices[3 * static_cast<int64_t>(v1_id) + 2]);
            const EdgeDesc desc = make_edge_desc(v0, v1, voxel_size);
            edge_desc[eid] = desc;
            job_edge[eid] = static_cast<int32_t>(eid);
            job_xyz[3 * eid + 0] = desc.start_x;
            job_xyz[3 * eid + 1] = desc.start_y;
            job_xyz[3 * eid + 2] = desc.start_z;
            job_tmax[3 * eid + 0] = desc.tmax0_x;
            job_tmax[3 * eid + 1] = desc.tmax0_y;
            job_tmax[3 * eid + 2] = desc.tmax0_z;
        }

        __global__ void build_boundary_edge_desc_and_jobs_kernel(
            const float *__restrict__ boundaries,
            int64_t num_boundaries,
            float3 voxel_size,
            EdgeDesc *__restrict__ edge_desc,
            int32_t *__restrict__ job_edge,
            int32_t *__restrict__ job_xyz,
            double *__restrict__ job_tmax)
        {
            const int64_t eid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (eid >= num_boundaries)
                return;

            const float *seg = boundaries + 6 * eid;
            const float3 v0 = make_float3(seg[0], seg[1], seg[2]);
            const float3 v1 = make_float3(seg[3], seg[4], seg[5]);
            const EdgeDesc desc = make_edge_desc(v0, v1, voxel_size);
            edge_desc[eid] = desc;
            job_edge[eid] = static_cast<int32_t>(eid);
            job_xyz[3 * eid + 0] = desc.start_x;
            job_xyz[3 * eid + 1] = desc.start_y;
            job_xyz[3 * eid + 2] = desc.start_z;
            job_tmax[3 * eid + 0] = desc.tmax0_x;
            job_tmax[3 * eid + 1] = desc.tmax0_y;
            job_tmax[3 * eid + 2] = desc.tmax0_z;
        }

        __global__ void count_dda_round_kernel(
            const EdgeDesc *__restrict__ edge_desc,
            const int32_t *__restrict__ job_edge,
            const int32_t *__restrict__ job_xyz,
            const double *__restrict__ job_tmax,
            int64_t num_jobs,
            Int3 grid_min,
            Int3 grid_max,
            int chunk_steps,
            int32_t *__restrict__ pair_count,
            int32_t *__restrict__ next_count)
        {
            const int64_t jid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (jid >= num_jobs)
                return;

            const int32_t eid = job_edge[jid];
            const EdgeDesc desc = edge_desc[eid];
            if (!desc.valid)
            {
                pair_count[jid] = 0;
                next_count[jid] = 0;
                return;
            }

            int x = job_xyz[3 * jid + 0];
            int y = job_xyz[3 * jid + 1];
            int z = job_xyz[3 * jid + 2];
            double tx = job_tmax[3 * jid + 0];
            double ty = job_tmax[3 * jid + 1];
            double tz = job_tmax[3 * jid + 2];

            int32_t pairs = in_bounds(x, y, z, grid_min, grid_max) ? 1 : 0;
            bool alive = true;
            for (int step_idx = 0; step_idx < chunk_steps; ++step_idx)
            {
                const int axis = argmin_axis_strict(tx, ty, tz);
                const double t_axis = (axis == 0) ? tx : (axis == 1 ? ty : tz);
                if (t_axis > desc.segment_length)
                {
                    alive = false;
                    break;
                }
                if (axis == 0)
                {
                    x += static_cast<int32_t>(desc.step_x);
                    tx += desc.tdelta_x;
                }
                else if (axis == 1)
                {
                    y += static_cast<int32_t>(desc.step_y);
                    ty += desc.tdelta_y;
                }
                else
                {
                    z += static_cast<int32_t>(desc.step_z);
                    tz += desc.tdelta_z;
                }
                if (in_bounds(x, y, z, grid_min, grid_max))
                    pairs += 1;
            }

            pair_count[jid] = pairs;
            next_count[jid] = alive ? 1 : 0;
        }

        __global__ void emit_dda_round_kernel(
            const EdgeDesc *__restrict__ edge_desc,
            const int32_t *__restrict__ job_edge,
            const int32_t *__restrict__ job_xyz,
            const double *__restrict__ job_tmax,
            int64_t num_jobs,
            Int3 grid_min,
            Int3 grid_max,
            int chunk_steps,
            const int32_t *__restrict__ pair_offset,
            const int32_t *__restrict__ next_offset,
            int32_t *__restrict__ out_edge,
            uint64_t *__restrict__ out_voxel_key,
            int32_t *__restrict__ next_edge,
            int32_t *__restrict__ next_xyz,
            double *__restrict__ next_tmax)
        {
            const int64_t jid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (jid >= num_jobs)
                return;

            const int32_t eid = job_edge[jid];
            const EdgeDesc desc = edge_desc[eid];
            if (!desc.valid)
                return;

            int x = job_xyz[3 * jid + 0];
            int y = job_xyz[3 * jid + 1];
            int z = job_xyz[3 * jid + 2];
            double tx = job_tmax[3 * jid + 0];
            double ty = job_tmax[3 * jid + 1];
            double tz = job_tmax[3 * jid + 2];

            int32_t pair_out = pair_offset[jid];
            if (in_bounds(x, y, z, grid_min, grid_max))
            {
                out_edge[pair_out] = eid;
                out_voxel_key[pair_out] = lex_voxel_key(x, y, z, grid_min, grid_max);
                pair_out += 1;
            }

            bool alive = true;
            for (int step_idx = 0; step_idx < chunk_steps; ++step_idx)
            {
                const int axis = argmin_axis_strict(tx, ty, tz);
                const double t_axis = (axis == 0) ? tx : (axis == 1 ? ty : tz);
                if (t_axis > desc.segment_length)
                {
                    alive = false;
                    break;
                }
                if (axis == 0)
                {
                    x += static_cast<int32_t>(desc.step_x);
                    tx += desc.tdelta_x;
                }
                else if (axis == 1)
                {
                    y += static_cast<int32_t>(desc.step_y);
                    ty += desc.tdelta_y;
                }
                else
                {
                    z += static_cast<int32_t>(desc.step_z);
                    tz += desc.tdelta_z;
                }
                if (in_bounds(x, y, z, grid_min, grid_max))
                {
                    out_edge[pair_out] = eid;
                    out_voxel_key[pair_out] = lex_voxel_key(x, y, z, grid_min, grid_max);
                    pair_out += 1;
                }
            }

            if (alive)
            {
                const int32_t next_out = next_offset[jid];
                next_edge[next_out] = eid;
                next_xyz[3 * next_out + 0] = x;
                next_xyz[3 * next_out + 1] = y;
                next_xyz[3 * next_out + 2] = z;
                next_tmax[3 * next_out + 0] = tx;
                next_tmax[3 * next_out + 1] = ty;
                next_tmax[3 * next_out + 2] = tz;
            }
        }

        __global__ void mark_unique_edge_voxel_kernel(
            const int32_t *__restrict__ edge_id,
            const uint64_t *__restrict__ voxel_key,
            int64_t n,
            int32_t *__restrict__ flags)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            if (i == 0)
            {
                flags[i] = 1;
                return;
            }
            flags[i] = (edge_id[i] != edge_id[i - 1] || voxel_key[i] != voxel_key[i - 1]) ? 1 : 0;
        }

        __global__ void scatter_unique_edge_voxel_kernel(
            const int32_t *__restrict__ edge_id,
            const uint64_t *__restrict__ voxel_key,
            const int32_t *__restrict__ flags,
            const int32_t *__restrict__ offsets,
            int64_t n,
            int32_t *__restrict__ unique_edge_id,
            uint64_t *__restrict__ unique_voxel_key)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n || flags[i] == 0)
                return;
            const int32_t out = offsets[i];
            unique_edge_id[out] = edge_id[i];
            unique_voxel_key[out] = voxel_key[i];
        }

        __global__ void decode_edge_pairs_kernel(
            const int32_t *__restrict__ edge_id,
            const uint64_t *__restrict__ voxel_key,
            int64_t n,
            Int3 grid_min,
            Int3 grid_max,
            int32_t *__restrict__ out_edge_id,
            int32_t *__restrict__ out_voxels)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const Int3 v = decode_lex_voxel_key(voxel_key[i], grid_min, grid_max);
            out_edge_id[i] = edge_id[i];
            out_voxels[3 * i + 0] = v.x;
            out_voxels[3 * i + 1] = v.y;
            out_voxels[3 * i + 2] = v.z;
        }

        __global__ void build_surface_lookup_kernel(
            const int32_t *__restrict__ voxels,
            int64_t n,
            Int3 grid_min,
            Int3 grid_max,
            uint64_t *__restrict__ keys,
            int32_t *__restrict__ rows)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const int x = voxels[3 * i + 0];
            const int y = voxels[3 * i + 1];
            const int z = voxels[3 * i + 2];
            keys[i] = lex_voxel_key(x, y, z, grid_min, grid_max);
            rows[i] = static_cast<int32_t>(i);
        }

        __global__ void map_unique_edge_voxels_kernel(
            const int32_t *__restrict__ edge_id,
            const uint64_t *__restrict__ voxel_key,
            int64_t n,
            const uint64_t *__restrict__ surface_keys,
            const int32_t *__restrict__ surface_rows,
            int64_t num_voxels,
            uint64_t *__restrict__ pair_keys_all,
            int32_t *__restrict__ valid)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const uint64_t key = voxel_key[i];
            const int32_t pos = lower_bound_u64(surface_keys, num_voxels, key);
            if (pos < num_voxels && surface_keys[pos] == key)
            {
                pair_keys_all[i] = pack_pair_key(surface_rows[pos], edge_id[i]);
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
            int64_t n,
            uint64_t *__restrict__ pair_keys)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n || valid[i] == 0)
                return;
            pair_keys[offsets[i]] = pair_keys_all[i];
        }

        __device__ __forceinline__ SymQEF10 boundary_qef_from_segment(float3 p0, float3 p1, float boundary_weight)
        {
            const double dx = static_cast<double>(p1.x) - static_cast<double>(p0.x);
            const double dy = static_cast<double>(p1.y) - static_cast<double>(p0.y);
            const double dz = static_cast<double>(p1.z) - static_cast<double>(p0.z);
            const double length = sqrt(dx * dx + dy * dy + dz * dz);
            if (length < 1e-6)
                return qef_zero();

            const double ux = dx / length;
            const double uy = dy / length;
            const double uz = dz / length;
            const double a00 = 1.0 - ux * ux;
            const double a01 = -ux * uy;
            const double a02 = -ux * uz;
            const double a11 = 1.0 - uy * uy;
            const double a12 = -uy * uz;
            const double a22 = 1.0 - uz * uz;
            const double bx = -(a00 * p0.x + a01 * p0.y + a02 * p0.z);
            const double by = -(a01 * p0.x + a11 * p0.y + a12 * p0.z);
            const double bz = -(a02 * p0.x + a12 * p0.y + a22 * p0.z);
            const double c = p0.x * (a00 * p0.x + a01 * p0.y + a02 * p0.z) +
                             p0.y * (a01 * p0.x + a11 * p0.y + a12 * p0.z) +
                             p0.z * (a02 * p0.x + a12 * p0.y + a22 * p0.z);
            return SymQEF10{
                static_cast<float>(boundary_weight * a00),
                static_cast<float>(boundary_weight * a01),
                static_cast<float>(boundary_weight * a02),
                static_cast<float>(boundary_weight * bx),
                static_cast<float>(boundary_weight * a11),
                static_cast<float>(boundary_weight * a12),
                static_cast<float>(boundary_weight * by),
                static_cast<float>(boundary_weight * a22),
                static_cast<float>(boundary_weight * bz),
                static_cast<float>(boundary_weight * c),
            };
        }

        __global__ void build_boundary_qef_contrib_kernel(
            const uint64_t *__restrict__ pair_keys,
            int64_t n,
            const float *__restrict__ boundaries,
            float boundary_weight,
            int32_t *__restrict__ voxel_ids,
            SymQEF10 *__restrict__ qefs)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const uint64_t key = pair_keys[i];
            const int32_t voxel_id = unpack_pair_voxel_id(key);
            const int32_t edge_id = unpack_pair_edge_id(key);
            const float *seg = boundaries + 6 * static_cast<int64_t>(edge_id);
            const float3 p0 = make_float3(seg[0], seg[1], seg[2]);
            const float3 p1 = make_float3(seg[3], seg[4], seg[5]);
            voxel_ids[i] = voxel_id;
            qefs[i] = boundary_qef_from_segment(p0, p1, boundary_weight);
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

        std::tuple<torch::Tensor, torch::Tensor, int64_t> trace_unique_edge_voxels(
            const torch::Tensor &edge_desc_storage,
            torch::Tensor curr_edge,
            torch::Tensor curr_xyz,
            torch::Tensor curr_tmax,
            int64_t curr_size,
            Int3 grid_min,
            Int3 grid_max,
            int chunk_steps,
            const torch::TensorOptions &opts_i32,
            const torch::TensorOptions &opts_u64,
            const torch::TensorOptions &opts_f64,
            const torch::TensorOptions &opts_u8,
            cudaStream_t stream)
        {
            std::vector<torch::Tensor> edge_rounds;
            std::vector<torch::Tensor> key_rounds;
            std::vector<int64_t> round_sizes;
            int64_t total_pairs = 0;
            auto *edge_desc = reinterpret_cast<const EdgeDesc *>(edge_desc_storage.data_ptr<uint8_t>());

            while (curr_size > 0)
            {
                TORCH_CHECK(curr_size <= std::numeric_limits<int>::max(), "DDA job count exceeds CUB int range");
                const int blocks = static_cast<int>(div_up_i64(curr_size, kThreads));
                auto pair_count = torch::empty({curr_size}, opts_i32);
                auto next_count = torch::empty({curr_size}, opts_i32);
                count_dda_round_kernel<<<blocks, kThreads, 0, stream>>>(
                    edge_desc,
                    curr_edge.data_ptr<int32_t>(),
                    curr_xyz.data_ptr<int32_t>(),
                    curr_tmax.data_ptr<double>(),
                    curr_size,
                    grid_min,
                    grid_max,
                    chunk_steps,
                    pair_count.data_ptr<int32_t>(),
                    next_count.data_ptr<int32_t>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();

                auto pair_offset = torch::empty({curr_size}, opts_i32);
                auto next_offset = torch::empty({curr_size}, opts_i32);
                size_t temp_bytes = 0;
                C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                    nullptr,
                    temp_bytes,
                    pair_count.data_ptr<int32_t>(),
                    pair_offset.data_ptr<int32_t>(),
                    static_cast<int>(curr_size),
                    stream));
                auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
                C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                    temp.data_ptr<uint8_t>(),
                    temp_bytes,
                    pair_count.data_ptr<int32_t>(),
                    pair_offset.data_ptr<int32_t>(),
                    static_cast<int>(curr_size),
                    stream));

                temp_bytes = 0;
                C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                    nullptr,
                    temp_bytes,
                    next_count.data_ptr<int32_t>(),
                    next_offset.data_ptr<int32_t>(),
                    static_cast<int>(curr_size),
                    stream));
                temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
                C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                    temp.data_ptr<uint8_t>(),
                    temp_bytes,
                    next_count.data_ptr<int32_t>(),
                    next_offset.data_ptr<int32_t>(),
                    static_cast<int>(curr_size),
                    stream));

                const int64_t num_pairs = read_compact_total_i32(pair_count, pair_offset, curr_size, stream);
                const int64_t num_next = read_compact_total_i32(next_count, next_offset, curr_size, stream);
                TORCH_CHECK(num_pairs <= std::numeric_limits<int>::max(), "DDA pair count exceeds CUB int range");

                auto out_edge = torch::empty({num_pairs}, opts_i32);
                auto out_key = torch::empty({num_pairs}, opts_u64);
                auto next_edge = torch::empty({num_next}, opts_i32);
                auto next_xyz = torch::empty({num_next, 3}, opts_i32);
                auto next_tmax = torch::empty({num_next, 3}, opts_f64);
                emit_dda_round_kernel<<<blocks, kThreads, 0, stream>>>(
                    edge_desc,
                    curr_edge.data_ptr<int32_t>(),
                    curr_xyz.data_ptr<int32_t>(),
                    curr_tmax.data_ptr<double>(),
                    curr_size,
                    grid_min,
                    grid_max,
                    chunk_steps,
                    pair_offset.data_ptr<int32_t>(),
                    next_offset.data_ptr<int32_t>(),
                    out_edge.data_ptr<int32_t>(),
                    out_key.data_ptr<uint64_t>(),
                    next_edge.data_ptr<int32_t>(),
                    next_xyz.data_ptr<int32_t>(),
                    next_tmax.data_ptr<double>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();

                if (num_pairs > 0)
                {
                    edge_rounds.push_back(out_edge);
                    key_rounds.push_back(out_key);
                    round_sizes.push_back(num_pairs);
                    total_pairs += num_pairs;
                }
                curr_edge = next_edge;
                curr_xyz = next_xyz;
                curr_tmax = next_tmax;
                curr_size = num_next;
            }

            if (total_pairs == 0)
                return std::make_tuple(torch::empty({0}, opts_i32), torch::empty({0}, opts_u64), int64_t{0});

            auto raw_edge = torch::empty({total_pairs}, opts_i32);
            auto raw_key = torch::empty({total_pairs}, opts_u64);
            int64_t cursor = 0;
            for (size_t i = 0; i < round_sizes.size(); ++i)
            {
                const int64_t n = round_sizes[i];
                C10_CUDA_CHECK(cudaMemcpyAsync(
                    raw_edge.data_ptr<int32_t>() + cursor,
                    edge_rounds[i].data_ptr<int32_t>(),
                    static_cast<size_t>(n) * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice,
                    stream));
                C10_CUDA_CHECK(cudaMemcpyAsync(
                    raw_key.data_ptr<uint64_t>() + cursor,
                    key_rounds[i].data_ptr<uint64_t>(),
                    static_cast<size_t>(n) * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice,
                    stream));
                cursor += n;
            }

            TORCH_CHECK(total_pairs <= std::numeric_limits<int>::max(), "DDA total pair count exceeds CUB int range");
            auto key_by_voxel = torch::empty({total_pairs}, opts_u64);
            auto edge_by_voxel = torch::empty({total_pairs}, opts_i32);
            size_t temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                nullptr,
                temp_bytes,
                raw_key.data_ptr<uint64_t>(),
                key_by_voxel.data_ptr<uint64_t>(),
                raw_edge.data_ptr<int32_t>(),
                edge_by_voxel.data_ptr<int32_t>(),
                static_cast<int>(total_pairs),
                0,
                64,
                stream));
            auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                temp.data_ptr<uint8_t>(),
                temp_bytes,
                raw_key.data_ptr<uint64_t>(),
                key_by_voxel.data_ptr<uint64_t>(),
                raw_edge.data_ptr<int32_t>(),
                edge_by_voxel.data_ptr<int32_t>(),
                static_cast<int>(total_pairs),
                0,
                64,
                stream));

            auto sorted_edge = torch::empty({total_pairs}, opts_i32);
            auto sorted_key = torch::empty({total_pairs}, opts_u64);
            temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                nullptr,
                temp_bytes,
                edge_by_voxel.data_ptr<int32_t>(),
                sorted_edge.data_ptr<int32_t>(),
                key_by_voxel.data_ptr<uint64_t>(),
                sorted_key.data_ptr<uint64_t>(),
                static_cast<int>(total_pairs),
                0,
                32,
                stream));
            temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                temp.data_ptr<uint8_t>(),
                temp_bytes,
                edge_by_voxel.data_ptr<int32_t>(),
                sorted_edge.data_ptr<int32_t>(),
                key_by_voxel.data_ptr<uint64_t>(),
                sorted_key.data_ptr<uint64_t>(),
                static_cast<int>(total_pairs),
                0,
                32,
                stream));

            auto unique_flags = torch::empty({total_pairs}, opts_i32);
            int blocks = static_cast<int>(div_up_i64(total_pairs, kThreads));
            mark_unique_edge_voxel_kernel<<<blocks, kThreads, 0, stream>>>(
                sorted_edge.data_ptr<int32_t>(),
                sorted_key.data_ptr<uint64_t>(),
                total_pairs,
                unique_flags.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            auto unique_offsets = torch::empty({total_pairs}, opts_i32);
            temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                nullptr,
                temp_bytes,
                unique_flags.data_ptr<int32_t>(),
                unique_offsets.data_ptr<int32_t>(),
                static_cast<int>(total_pairs),
                stream));
            temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
            C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                temp.data_ptr<uint8_t>(),
                temp_bytes,
                unique_flags.data_ptr<int32_t>(),
                unique_offsets.data_ptr<int32_t>(),
                static_cast<int>(total_pairs),
                stream));
            const int64_t num_unique = read_compact_total_i32(unique_flags, unique_offsets, total_pairs, stream);

            auto unique_edge = torch::empty({num_unique}, opts_i32);
            auto unique_key = torch::empty({num_unique}, opts_u64);
            scatter_unique_edge_voxel_kernel<<<blocks, kThreads, 0, stream>>>(
                sorted_edge.data_ptr<int32_t>(),
                sorted_key.data_ptr<uint64_t>(),
                unique_flags.data_ptr<int32_t>(),
                unique_offsets.data_ptr<int32_t>(),
                total_pairs,
                unique_edge.data_ptr<int32_t>(),
                unique_key.data_ptr<uint64_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            return std::make_tuple(unique_edge, unique_key, num_unique);
        }

    } // namespace

    std::tuple<torch::Tensor, torch::Tensor> edge_dda_old(
        const torch::Tensor &vertices,
        const torch::Tensor &edges,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        int chunk_steps)
    {
        TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
        TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");
        TORCH_CHECK(chunk_steps > 0, "chunk_steps must be > 0");
        static_assert(sizeof(EdgeDesc) % 8 == 0, "Unexpected EdgeDesc layout");

        const c10::cuda::CUDAGuard guard(vertices.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        const torch::Device device = vertices.device();
        const auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
        const auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(device);
        const auto opts_f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        const auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);

        const int64_t num_edges = edges.size(0);
        if (num_edges == 0)
            return std::make_tuple(torch::empty({0}, opts_i32), torch::empty({0, 3}, opts_i32));
        TORCH_CHECK(num_edges <= std::numeric_limits<int32_t>::max(), "edge count exceeds int32 range");

        const float *voxel_size_ptr = voxel_size.data_ptr<float>();
        const int32_t *grid_range_ptr = grid_range.data_ptr<int32_t>();
        const float3 voxel_size_h = make_float3(voxel_size_ptr[0], voxel_size_ptr[1], voxel_size_ptr[2]);
        const Int3 grid_min{grid_range_ptr[0], grid_range_ptr[1], grid_range_ptr[2]};
        const Int3 grid_max{grid_range_ptr[3], grid_range_ptr[4], grid_range_ptr[5]};

        auto edge_desc = torch::empty({num_edges * static_cast<int64_t>(sizeof(EdgeDesc))}, opts_u8);
        auto curr_edge = torch::empty({num_edges}, opts_i32);
        auto curr_xyz = torch::empty({num_edges, 3}, opts_i32);
        auto curr_tmax = torch::empty({num_edges, 3}, opts_f64);

        int blocks = static_cast<int>(div_up_i64(num_edges, kThreads));
        build_indexed_edge_desc_and_jobs_kernel<<<blocks, kThreads, 0, stream>>>(
            vertices.data_ptr<float>(),
            edges.data_ptr<int32_t>(),
            num_edges,
            voxel_size_h,
            reinterpret_cast<EdgeDesc *>(edge_desc.data_ptr<uint8_t>()),
            curr_edge.data_ptr<int32_t>(),
            curr_xyz.data_ptr<int32_t>(),
            curr_tmax.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto traced = trace_unique_edge_voxels(
            edge_desc,
            curr_edge,
            curr_xyz,
            curr_tmax,
            num_edges,
            grid_min,
            grid_max,
            chunk_steps,
            opts_i32,
            opts_u64,
            opts_f64,
            opts_u8,
            stream);
        torch::Tensor unique_edge = std::get<0>(traced);
        torch::Tensor unique_key = std::get<1>(traced);
        const int64_t num_unique = std::get<2>(traced);

        auto out_edge = torch::empty({num_unique}, opts_i32);
        auto out_voxels = torch::empty({num_unique, 3}, opts_i32);
        if (num_unique > 0)
        {
            blocks = static_cast<int>(div_up_i64(num_unique, kThreads));
            decode_edge_pairs_kernel<<<blocks, kThreads, 0, stream>>>(
                unique_edge.data_ptr<int32_t>(),
                unique_key.data_ptr<uint64_t>(),
                num_unique,
                grid_min,
                grid_max,
                out_edge.data_ptr<int32_t>(),
                out_voxels.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return std::make_tuple(out_edge, out_voxels);
    }

    torch::Tensor boundary_qef_old(
        const torch::Tensor &boundaries,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        float boundary_weight,
        const torch::Tensor &voxels,
        int chunk_steps)
    {
        TORCH_CHECK(boundaries.is_cuda(), "boundaries must be a CUDA tensor");
        TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
        TORCH_CHECK(chunk_steps > 0, "chunk_steps must be > 0");
        static_assert(sizeof(SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");
        static_assert(sizeof(EdgeDesc) % 8 == 0, "Unexpected EdgeDesc layout");

        const c10::cuda::CUDAGuard guard(boundaries.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(boundaries.get_device()).stream();
        const torch::Device device = boundaries.device();
        const auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
        const auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(device);
        const auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const auto opts_f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        const auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);

        const int64_t num_boundaries = boundaries.size(0);
        const int64_t num_voxels = voxels.size(0);
        auto out_qefs = torch::zeros({num_voxels, 10}, opts_f32);
        if (num_boundaries == 0 || num_voxels == 0)
            return out_qefs;
        TORCH_CHECK(num_boundaries <= std::numeric_limits<int32_t>::max(), "boundary count exceeds int32 range");
        TORCH_CHECK(num_voxels <= std::numeric_limits<int>::max(), "surface voxel count exceeds CUB int range");

        const float *voxel_size_ptr = voxel_size.data_ptr<float>();
        const int32_t *grid_range_ptr = grid_range.data_ptr<int32_t>();
        const float3 voxel_size_h = make_float3(voxel_size_ptr[0], voxel_size_ptr[1], voxel_size_ptr[2]);
        const Int3 grid_min{grid_range_ptr[0], grid_range_ptr[1], grid_range_ptr[2]};
        const Int3 grid_max{grid_range_ptr[3], grid_range_ptr[4], grid_range_ptr[5]};

        auto edge_desc = torch::empty({num_boundaries * static_cast<int64_t>(sizeof(EdgeDesc))}, opts_u8);
        auto curr_edge = torch::empty({num_boundaries}, opts_i32);
        auto curr_xyz = torch::empty({num_boundaries, 3}, opts_i32);
        auto curr_tmax = torch::empty({num_boundaries, 3}, opts_f64);

        int blocks = static_cast<int>(div_up_i64(num_boundaries, kThreads));
        build_boundary_edge_desc_and_jobs_kernel<<<blocks, kThreads, 0, stream>>>(
            boundaries.data_ptr<float>(),
            num_boundaries,
            voxel_size_h,
            reinterpret_cast<EdgeDesc *>(edge_desc.data_ptr<uint8_t>()),
            curr_edge.data_ptr<int32_t>(),
            curr_xyz.data_ptr<int32_t>(),
            curr_tmax.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto traced = trace_unique_edge_voxels(
            edge_desc,
            curr_edge,
            curr_xyz,
            curr_tmax,
            num_boundaries,
            grid_min,
            grid_max,
            chunk_steps,
            opts_i32,
            opts_u64,
            opts_f64,
            opts_u8,
            stream);
        torch::Tensor unique_edge = std::get<0>(traced);
        torch::Tensor unique_key = std::get<1>(traced);
        const int64_t num_edge_voxels = std::get<2>(traced);
        if (num_edge_voxels == 0)
            return out_qefs;
        TORCH_CHECK(num_edge_voxels <= std::numeric_limits<int>::max(), "edge voxel pair count exceeds CUB int range");

        auto surface_keys = torch::empty({num_voxels}, opts_u64);
        auto surface_rows = torch::empty({num_voxels}, opts_i32);
        blocks = static_cast<int>(div_up_i64(num_voxels, kThreads));
        build_surface_lookup_kernel<<<blocks, kThreads, 0, stream>>>(
            voxels.data_ptr<int32_t>(),
            num_voxels,
            grid_min,
            grid_max,
            surface_keys.data_ptr<uint64_t>(),
            surface_rows.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto surface_keys_sorted = torch::empty({num_voxels}, opts_u64);
        auto surface_rows_sorted = torch::empty({num_voxels}, opts_i32);
        size_t temp_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_bytes,
            surface_keys.data_ptr<uint64_t>(),
            surface_keys_sorted.data_ptr<uint64_t>(),
            surface_rows.data_ptr<int32_t>(),
            surface_rows_sorted.data_ptr<int32_t>(),
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
            surface_rows.data_ptr<int32_t>(),
            surface_rows_sorted.data_ptr<int32_t>(),
            static_cast<int>(num_voxels),
            0,
            64,
            stream));

        auto pair_keys_all = torch::empty({num_edge_voxels}, opts_u64);
        auto valid = torch::empty({num_edge_voxels}, opts_i32);
        blocks = static_cast<int>(div_up_i64(num_edge_voxels, kThreads));
        map_unique_edge_voxels_kernel<<<blocks, kThreads, 0, stream>>>(
            unique_edge.data_ptr<int32_t>(),
            unique_key.data_ptr<uint64_t>(),
            num_edge_voxels,
            surface_keys_sorted.data_ptr<uint64_t>(),
            surface_rows_sorted.data_ptr<int32_t>(),
            num_voxels,
            pair_keys_all.data_ptr<uint64_t>(),
            valid.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto offsets = torch::empty({num_edge_voxels}, opts_i32);
        temp_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            temp_bytes,
            valid.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            static_cast<int>(num_edge_voxels),
            stream));
        temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            temp.data_ptr<uint8_t>(),
            temp_bytes,
            valid.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            static_cast<int>(num_edge_voxels),
            stream));
        const int64_t num_valid_pairs = read_compact_total_i32(valid, offsets, num_edge_voxels, stream);
        if (num_valid_pairs == 0)
            return out_qefs;

        auto pair_keys = torch::empty({num_valid_pairs}, opts_u64);
        compact_pair_keys_kernel<<<blocks, kThreads, 0, stream>>>(
            pair_keys_all.data_ptr<uint64_t>(),
            valid.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            num_edge_voxels,
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
        build_boundary_qef_contrib_kernel<<<blocks, kThreads, 0, stream>>>(
            unique_pair_keys.data_ptr<uint64_t>(),
            num_unique_pairs,
            boundaries.data_ptr<float>(),
            boundary_weight,
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
