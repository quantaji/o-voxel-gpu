#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace o_voxel::fdg
{

    struct Int3
    {
        int x;
        int y;
        int z;

        __host__ __device__ int &operator[](int i) { return (&x)[i]; }
        __host__ __device__ int operator[](int i) const { return (&x)[i]; }
    };

    struct GridSpec
    {
        float3 voxel_size;
        Int3 grid_min;
        Int3 grid_max;
    };

    struct SymQEF10
    {
        float q00;
        float q01;
        float q02;
        float q03;
        float q11;
        float q12;
        float q13;
        float q22;
        float q23;
        float q33;
    };

    __host__ __device__ __forceinline__ uint64_t pack_voxel_key(
        int x,
        int y,
        int z,
        Int3 grid_min,
        Int3 grid_max)
    {
        const uint64_t sx = static_cast<uint64_t>(grid_max.x - grid_min.x);
        const uint64_t sy = static_cast<uint64_t>(grid_max.y - grid_min.y);
        const uint64_t ux = static_cast<uint64_t>(x - grid_min.x);
        const uint64_t uy = static_cast<uint64_t>(y - grid_min.y);
        const uint64_t uz = static_cast<uint64_t>(z - grid_min.z);
        return ux + sx * (uy + sy * uz);
    }

    __host__ __device__ __forceinline__ Int3 unpack_voxel_key(
        uint64_t key,
        Int3 grid_min,
        Int3 grid_max)
    {
        const uint64_t sx = static_cast<uint64_t>(grid_max.x - grid_min.x);
        const uint64_t sy = static_cast<uint64_t>(grid_max.y - grid_min.y);
        const uint64_t yz = sx * sy;
        const uint64_t z = key / yz;
        const uint64_t rem = key - z * yz;
        const uint64_t y = rem / sx;
        const uint64_t x = rem - y * sx;
        return Int3{
            static_cast<int>(x) + grid_min.x,
            static_cast<int>(y) + grid_min.y,
            static_cast<int>(z) + grid_min.z,
        };
    }

} // namespace o_voxel::fdg
