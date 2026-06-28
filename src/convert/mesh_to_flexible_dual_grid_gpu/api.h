#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <tuple>

namespace o_voxel::fdg
{

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    mesh_to_flexible_dual_grid_old(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        float face_weight,
        float boundary_weight,
        float regularization_weight,
        int64_t intersect_chunk_triangles,
        int boundary_chunk_steps);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    mesh_to_flexible_dual_grid_ref(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        float face_weight,
        float boundary_weight,
        float regularization_weight,
        int64_t intersect_chunk_triangles);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    mesh_to_flexible_dual_grid(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        float face_weight,
        float boundary_weight,
        float regularization_weight,
        int64_t intersect_chunk_triangles);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    intersect_qef_old(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        int64_t chunk_triangles);

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
        int64_t chunk_triangles);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    intersect_qef(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        int64_t chunk_triangles);

    torch::Tensor
    intersect_occ(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        int64_t chunk_triangles);

    torch::Tensor
    intersect_occ_old(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        int64_t chunk_triangles);

    std::tuple<torch::Tensor, torch::Tensor>
    voxelize_mesh_octree(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range);

    std::tuple<torch::Tensor, torch::Tensor>
    edge_dda_old(
        const torch::Tensor &vertices,
        const torch::Tensor &edges,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        int chunk_steps);

    torch::Tensor face_qef_old(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        const torch::Tensor &voxels);

    torch::Tensor face_qef_ref(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        const torch::Tensor &voxels,
        const torch::Tensor &hash_keys,
        const torch::Tensor &hash_vals);

    torch::Tensor face_qef(
        const torch::Tensor &triangles,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        const torch::Tensor &voxels,
        const torch::Tensor &brick_hash_keys,
        const torch::Tensor &brick_hash_vals,
        const torch::Tensor &brick_bits,
        const torch::Tensor &brick_base);

    torch::Tensor boundary_qef_old(
        const torch::Tensor &boundaries,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        float boundary_weight,
        const torch::Tensor &voxels,
        int chunk_steps);

    torch::Tensor boundary_qef_ref(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        float boundary_weight,
        const torch::Tensor &voxels,
        const torch::Tensor &hash_keys,
        const torch::Tensor &hash_vals);

    torch::Tensor boundary_qef(
        const torch::Tensor &boundaries,
        const torch::Tensor &voxel_size,
        const torch::Tensor &grid_range,
        float boundary_weight,
        const torch::Tensor &voxels,
        const torch::Tensor &brick_hash_keys,
        const torch::Tensor &brick_hash_vals,
        const torch::Tensor &brick_bits,
        const torch::Tensor &brick_base);

} // namespace o_voxel::fdg
