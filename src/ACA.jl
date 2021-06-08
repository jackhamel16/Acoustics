@views function computeRHSContributionSoundSoft!(pulse_mesh::PulseMesh,
                                                 wavenumber,
                                                 distance_to_edge_tol,
                                                 near_singular_tol,
                                                 test_node::Node,
                                                 src_node::Node,
                                                 J_vec::AbstractArray{T,1},
                                                 V_vec::AbstractArray{T,1}) where T
    # Computes the RHS contributions for interactions between all elements
    # in node1 and node2 (non-self-node interactions)
    @unpack areas,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    for local_testele_idx = 1:length(test_node.element_idxs)
        global_testele_idx = test_node.element_idxs[local_testele_idx]
        for local_srcele_idx = 1:length(src_node.element_idxs)
            global_srcele_idx = src_node.element_idxs[local_srcele_idx]
            is_singular = global_testele_idx == global_srcele_idx
            testIntegrand1(x,y,z) = scalarGreensIntegration(pulse_mesh,
                                                           global_srcele_idx,
                                                           wavenumber,
                                                           [x,y,z],
                                                           distance_to_edge_tol,
                                                           near_singular_tol,
                                                           is_singular)
            Z_entry = gaussQuadrature(areas[global_testele_idx],
                                      testIntegrand1,
                                      test_quadrature_points[global_testele_idx],
                                      test_quadrature_weights)
            V_vec[global_testele_idx] += Z_entry * J_vec[global_srcele_idx]
        end
    end
end # computeRHSContributionSoundSoft!
# function computeRHSContributionSoundSoft!(pulse_mesh::PulseMesh, wavenumber, distance_to_edge_tol, near_singular_tol, node1::Node, node2::Node, J_vec::AbstractArray{T,1}, V_vec::AbstractArray{T,1}) where T
#     # Computes the RHS contributions for interactions between all elements
#     # in node1 and node2 (non-self-node interactions)
#     @unpack areas,
#             test_quadrature_points,
#             test_quadrature_weights = pulse_mesh
#     is_not_singular = false
#     for local_node1ele_idx = 1:length(node1.element_idxs)
#         global_node1ele_idx = node1.element_idxs[local_node1ele_idx]
#         for local_node2ele_idx = 1:length(node2.element_idxs)
#             global_node2ele_idx = node2.element_idxs[local_node2ele_idx]
#             # for node 1 elements as srcs, node 2 elements as tests
#             testIntegrand1(x,y,z) = scalarGreensIntegration(pulse_mesh,
#                                                            global_node1ele_idx,
#                                                            wavenumber,
#                                                            [x,y,z],
#                                                            distance_to_edge_tol,
#                                                            near_singular_tol,
#                                                            is_not_singular)
#             # for node 2 elements as srcs, node 1 elements as tests
#             testIntegrand2(x,y,z) = scalarGreensIntegration(pulse_mesh,
#                                                             global_node2ele_idx,
#                                                             wavenumber,
#                                                             [x,y,z],
#                                                             distance_to_edge_tol,
#                                                             near_singular_tol,
#                                                             is_not_singular)
#             Z_entry1 = gaussQuadrature(areas[global_node2ele_idx],
#                                       testIntegrand1,
#                                       test_quadrature_points[global_node2ele_idx],
#                                       test_quadrature_weights)
#             Z_entry2 = gaussQuadrature(areas[global_node1ele_idx],
#                                        testIntegrand2,
#                                        test_quadrature_points[global_node1ele_idx],
#                                        test_quadrature_weights)
#             V_vec[global_node2ele_idx] += Z_entry1 * J_vec[global_node1ele_idx]
#             V_vec[global_node1ele_idx] += Z_entry2 * J_vec[global_node2ele_idx]
#         end
#     end
# end # computeRHSContributionSoundSoft!

# function computeRHSContributionSoundSoft!(pulse_mesh::PulseMesh, wavenumber, distance_to_edge_tol, near_singular_tol, node::Node, J_vec::AbstractArray{T,1}, V_vec::AbstractArray{T,1}) where T
#     # Computes the RHS contributions for interactions between all elements
#     # node with themselves (self-node interactions)
#     @unpack areas,
#             test_quadrature_points,
#             test_quadrature_weights = pulse_mesh
#     for local_src_idx = 1:length(node.element_idxs)
#         global_src_idx = node.element_idxs[local_src_idx]
#         for local_test_idx = 1:length(node.element_idxs)
#             global_test_idx = node.element_idxs[local_test_idx]
#             is_singular = global_src_idx == global_test_idx
#             testIntegrand(x,y,z) = scalarGreensIntegration(pulse_mesh,
#                                                            global_src_idx,
#                                                            wavenumber,
#                                                            [x,y,z],
#                                                            distance_to_edge_tol,
#                                                            near_singular_tol,
#                                                            is_singular)
#             Z_entry = gaussQuadrature(areas[global_test_idx],
#                                       testIntegrand,
#                                       test_quadrature_points[global_test_idx],
#                                       test_quadrature_weights)
#             V_vec[global_test_idx] += Z_entry * J_vec[global_src_idx]
#         end
#     end
# end # computeRHSContributionSoundSoft!

function computeZJMatVec(pulse_mesh::PulseMesh,
                         octree::Octree,
                         wavenumber,
                         distance_to_edge_tol,
                         near_singular_tol,
                         J_vec::AbstractArray{T,1}) where T
    @unpack num_elements = pulse_mesh
    @unpack leaf_node_idxs,
            nodes = octree
    leaf_nodes = octree.nodes[octree.leaf_node_idxs]
    num_leaves = length(leaf_nodes)
    V_vec = zeros(ComplexF64, num_elements)
    for local_test_node_idx = 1:num_leaves
        global_test_node_idx = octree.leaf_node_idxs[local_test_node_idx]
        test_node = octree.nodes[global_test_node_idx]
        for local_src_node_idx = 1:num_leaves
            global_src_node_idx = octree.leaf_node_idxs[local_src_node_idx]
            src_node = octree.nodes[global_src_node_idx]
            computeRHSContributionSoundSoft!(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, test_node, src_node, J_vec, V_vec)
        end
    end
    return(V_vec)
end # computeZJMatVec
