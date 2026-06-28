#include "api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;

        __host__ __device__ __forceinline__ int64_t div_up_i64(int64_t n, int64_t d)
        {
            return (n + d - 1) / d;
        }

        __device__ __forceinline__ bool solve_3x3_ref(const float *A, const float *b, float *x)
        {
            const float a00 = A[4] * A[8] - A[5] * A[7];
            const float a01 = A[5] * A[6] - A[3] * A[8];
            const float a02 = A[3] * A[7] - A[4] * A[6];
            const float det = A[0] * a00 + A[1] * a01 + A[2] * a02;
            if (fabsf(det) < 1e-12f)
                return false;

            const float a10 = A[2] * A[7] - A[1] * A[8];
            const float a11 = A[0] * A[8] - A[2] * A[6];
            const float a12 = A[1] * A[6] - A[0] * A[7];
            const float a20 = A[1] * A[5] - A[2] * A[4];
            const float a21 = A[2] * A[3] - A[0] * A[5];
            const float a22 = A[0] * A[4] - A[1] * A[3];
            const float inv_det = 1.0f / det;

            x[0] = (a00 * b[0] + a10 * b[1] + a20 * b[2]) * inv_det;
            x[1] = (a01 * b[0] + a11 * b[1] + a21 * b[2]) * inv_det;
            x[2] = (a02 * b[0] + a12 * b[1] + a22 * b[2]) * inv_det;
            return true;
        }

        __device__ __forceinline__ bool solve_2x2_ref(const float *A, const float *b, float *x)
        {
            const float det = A[0] * A[3] - A[1] * A[2];
            if (fabsf(det) < 1e-12f)
                return false;
            const float inv_det = 1.0f / det;
            x[0] = (A[3] * b[0] - A[1] * b[1]) * inv_det;
            x[1] = (A[0] * b[1] - A[2] * b[0]) * inv_det;
            return true;
        }

        __device__ __forceinline__ float eval_qef_error_ref(const float *Q, const float *p)
        {
            const float v0 = p[0];
            const float v1 = p[1];
            const float v2 = p[2];
            const float row0 = Q[0] * v0 + Q[1] * v1 + Q[2] * v2 + Q[3];
            const float row1 = Q[4] * v0 + Q[5] * v1 + Q[6] * v2 + Q[7];
            const float row2 = Q[8] * v0 + Q[9] * v1 + Q[10] * v2 + Q[11];
            const float row3 = Q[12] * v0 + Q[13] * v1 + Q[14] * v2 + Q[15];
            return v0 * row0 + v1 * row1 + v2 * row2 + row3;
        }

        __global__ void gather_triangles_kernel(
            const float *__restrict__ vertices,
            const int32_t *__restrict__ faces,
            int64_t num_faces,
            float *__restrict__ triangles)
        {
            const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (tid >= 3 * num_faces)
                return;

            const int64_t f = tid / 3;
            const int lv = static_cast<int>(tid - 3 * f);
            const int32_t vid = faces[3 * f + lv];
            triangles[3 * tid + 0] = vertices[3 * static_cast<int64_t>(vid) + 0];
            triangles[3 * tid + 1] = vertices[3 * static_cast<int64_t>(vid) + 1];
            triangles[3 * tid + 2] = vertices[3 * static_cast<int64_t>(vid) + 2];
        }

        __global__ void solve_qef_ref_kernel(
            const int32_t *__restrict__ voxels,
            const float *__restrict__ mean_sum,
            const float *__restrict__ cnt,
            const float *__restrict__ intersect_qefs,
            const float *__restrict__ face_qefs,
            const float *__restrict__ boundary_qefs,
            int64_t n,
            float3 voxel_size,
            float face_weight,
            float regularization_weight,
            float *__restrict__ dual_vertices)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;

            const float *iq = intersect_qefs + 10 * i;
            const float *fq = face_qefs + 10 * i;
            const float *bq = boundary_qefs + 10 * i;

            const float q00 = iq[0] + face_weight * fq[0] + bq[0];
            const float q01 = iq[1] + face_weight * fq[1] + bq[1];
            const float q02 = iq[2] + face_weight * fq[2] + bq[2];
            const float q03 = iq[3] + face_weight * fq[3] + bq[3];
            const float q11 = iq[4] + face_weight * fq[4] + bq[4];
            const float q12 = iq[5] + face_weight * fq[5] + bq[5];
            const float q13 = iq[6] + face_weight * fq[6] + bq[6];
            const float q22 = iq[7] + face_weight * fq[7] + bq[7];
            const float q23 = iq[8] + face_weight * fq[8] + bq[8];
            const float q33 = iq[9] + face_weight * fq[9] + bq[9];

            float Q[16] = {
                q00, q01, q02, q03,
                q01, q11, q12, q13,
                q02, q12, q22, q23,
                q03, q13, q23, q33};

            const float c = cnt[i];
            if (regularization_weight > 0.0f && c > 0.0f)
            {
                const float px = mean_sum[3 * i + 0] / c;
                const float py = mean_sum[3 * i + 1] / c;
                const float pz = mean_sum[3 * i + 2] / c;
                const float w = regularization_weight * c;
                Q[0] += w;
                Q[5] += w;
                Q[10] += w;
                Q[3] -= w * px;
                Q[7] -= w * py;
                Q[11] -= w * pz;
                Q[12] -= w * px;
                Q[13] -= w * py;
                Q[14] -= w * pz;
                Q[15] += w * (px * px + py * py + pz * pz);
            }

            const int x = voxels[3 * i + 0];
            const int y = voxels[3 * i + 1];
            const int z = voxels[3 * i + 2];
            const float min_corner[3] = {
                x * voxel_size.x,
                y * voxel_size.y,
                z * voxel_size.z};
            const float max_corner[3] = {
                (x + 1) * voxel_size.x,
                (y + 1) * voxel_size.y,
                (z + 1) * voxel_size.z};

            const float A[9] = {
                Q[0], Q[1], Q[2],
                Q[4], Q[5], Q[6],
                Q[8], Q[9], Q[10]};
            const float rhs[3] = {-Q[3], -Q[7], -Q[11]};

            float v_new[3];
            float x_sol[3];
            bool found = false;
            float best = 1e30f;

            if (solve_3x3_ref(A, rhs, x_sol))
            {
                if (x_sol[0] >= min_corner[0] && x_sol[0] <= max_corner[0] &&
                    x_sol[1] >= min_corner[1] && x_sol[1] <= max_corner[1] &&
                    x_sol[2] >= min_corner[2] && x_sol[2] <= max_corner[2])
                {
                    v_new[0] = x_sol[0];
                    v_new[1] = x_sol[1];
                    v_new[2] = x_sol[2];
                    found = true;
                }
            }

            if (!found)
            {
                for (int fixed_axis = 0; fixed_axis < 3; ++fixed_axis)
                {
                    const int ax1 = (fixed_axis + 1) % 3;
                    const int ax2 = (fixed_axis + 2) % 3;
                    const float A2[4] = {
                        Q[4 * ax1 + ax1], Q[4 * ax1 + ax2],
                        Q[4 * ax2 + ax1], Q[4 * ax2 + ax2]};
                    const float B2[4] = {
                        Q[4 * ax1 + fixed_axis], Q[4 * ax1 + 3],
                        Q[4 * ax2 + fixed_axis], Q[4 * ax2 + 3]};

                    for (int bound_type = 0; bound_type < 2; ++bound_type)
                    {
                        const float q0 = bound_type ? min_corner[fixed_axis] : max_corner[fixed_axis];
                        const float b2[2] = {
                            -(B2[0] * q0 + B2[1]),
                            -(B2[2] * q0 + B2[3])};
                        float x2[2];
                        if (solve_2x2_ref(A2, b2, x2))
                        {
                            if (x2[0] >= min_corner[ax1] && x2[0] <= max_corner[ax1] &&
                                x2[1] >= min_corner[ax2] && x2[1] <= max_corner[ax2])
                            {
                                float p[3];
                                p[fixed_axis] = q0;
                                p[ax1] = x2[0];
                                p[ax2] = x2[1];
                                const float err = eval_qef_error_ref(Q, p);
                                if (err < best)
                                {
                                    best = err;
                                    v_new[0] = p[0];
                                    v_new[1] = p[1];
                                    v_new[2] = p[2];
                                    found = true;
                                }
                            }
                        }
                    }
                }

                for (int free_axis = 0; free_axis < 3; ++free_axis)
                {
                    const int ax1 = (free_axis + 1) % 3;
                    const int ax2 = (free_axis + 2) % 3;
                    const float a_diag = Q[4 * free_axis + free_axis];
                    if (fabsf(a_diag) < 1e-12f)
                        continue;
                    const float b0 = Q[4 * free_axis + ax1];
                    const float b1 = Q[4 * free_axis + ax2];
                    const float b2 = Q[4 * free_axis + 3];

                    for (int c0 = 0; c0 < 2; ++c0)
                    {
                        for (int c1 = 0; c1 < 2; ++c1)
                        {
                            const float q0 = c0 ? min_corner[ax1] : max_corner[ax1];
                            const float q1 = c1 ? min_corner[ax2] : max_corner[ax2];
                            const float value = -(b0 * q0 + b1 * q1 + b2) / a_diag;
                            if (value >= min_corner[free_axis] && value <= max_corner[free_axis])
                            {
                                float p[3];
                                p[free_axis] = value;
                                p[ax1] = q0;
                                p[ax2] = q1;
                                const float err = eval_qef_error_ref(Q, p);
                                if (err < best)
                                {
                                    best = err;
                                    v_new[0] = p[0];
                                    v_new[1] = p[1];
                                    v_new[2] = p[2];
                                    found = true;
                                }
                            }
                        }
                    }
                }

                for (int bx = 0; bx < 2; ++bx)
                {
                    for (int by = 0; by < 2; ++by)
                    {
                        for (int bz = 0; bz < 2; ++bz)
                        {
                            const float p[3] = {
                                bx ? min_corner[0] : max_corner[0],
                                by ? min_corner[1] : max_corner[1],
                                bz ? min_corner[2] : max_corner[2]};
                            const float err = eval_qef_error_ref(Q, p);
                            if (err < best)
                            {
                                best = err;
                                v_new[0] = p[0];
                                v_new[1] = p[1];
                                v_new[2] = p[2];
                                found = true;
                            }
                        }
                    }
                }
            }

            dual_vertices[3 * i + 0] = v_new[0];
            dual_vertices[3 * i + 1] = v_new[1];
            dual_vertices[3 * i + 2] = v_new[2];
        }

    } // namespace

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    mesh_to_flexible_dual_grid_ref(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        float face_weight,
        float boundary_weight,
        float regularization_weight,
        int64_t intersect_chunk_triangles)
    {
        TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
        TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
        TORCH_CHECK(intersect_chunk_triangles > 0, "intersect_chunk_triangles must be positive");
        static_assert(sizeof(SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");

        const c10::cuda::CUDAGuard guard(vertices.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        const torch::Device device = vertices.device();
        const auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        const int64_t num_faces = faces.size(0);
        auto triangles = torch::empty({num_faces, 3, 3}, opts_f32);
        if (num_faces > 0)
        {
            const int blocks = static_cast<int>(div_up_i64(num_faces * 3, kThreads));
            gather_triangles_kernel<<<blocks, kThreads, 0, stream>>>(
                vertices.data_ptr<float>(),
                faces.data_ptr<int32_t>(),
                num_faces,
                triangles.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        auto intersect = intersect_qef_ref(triangles, voxel_size, grid_range, intersect_chunk_triangles);
        torch::Tensor voxels = std::get<0>(intersect);
        torch::Tensor mean_sum = std::get<1>(intersect);
        torch::Tensor cnt = std::get<2>(intersect);
        torch::Tensor intersected = std::get<3>(intersect);
        torch::Tensor intersect_qefs = std::get<4>(intersect);
        torch::Tensor hash_keys = std::get<5>(intersect);
        torch::Tensor hash_vals = std::get<6>(intersect);

        const int64_t num_voxels = voxels.size(0);
        if (num_voxels == 0)
        {
            auto dual_vertices = torch::empty({0, 3}, opts_f32);
            return std::make_tuple(voxels, dual_vertices, intersected);
        }

        torch::Tensor face_qefs = face_weight > 0.0f
                                      ? face_qef_ref(triangles, voxel_size, grid_range, voxels, hash_keys, hash_vals)
                                      : torch::zeros({num_voxels, 10}, opts_f32);
        torch::Tensor boundary_qefs = boundary_qef_ref(
            vertices,
            faces,
            voxel_size,
            grid_range,
            boundary_weight,
            voxels,
            hash_keys,
            hash_vals);

        const float *voxel_size_ptr = voxel_size.data_ptr<float>();
        const float3 voxel_size_h = make_float3(voxel_size_ptr[0], voxel_size_ptr[1], voxel_size_ptr[2]);
        auto dual_vertices = torch::empty({num_voxels, 3}, opts_f32);
        const int blocks = static_cast<int>(div_up_i64(num_voxels, kThreads));
        solve_qef_ref_kernel<<<blocks, kThreads, 0, stream>>>(
            voxels.data_ptr<int32_t>(),
            mean_sum.data_ptr<float>(),
            cnt.data_ptr<float>(),
            intersect_qefs.data_ptr<float>(),
            face_qefs.data_ptr<float>(),
            boundary_qefs.data_ptr<float>(),
            num_voxels,
            voxel_size_h,
            face_weight,
            regularization_weight,
            dual_vertices.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return std::make_tuple(voxels, dual_vertices, intersected);
    }

} // namespace o_voxel::fdg
