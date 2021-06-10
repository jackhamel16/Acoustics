using LinearAlgebra

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
            # is_singular = global_testele_idx == global_srcele_idx
            # testIntegrand1(x,y,z) = scalarGreensIntegration(pulse_mesh,
            #                                                global_srcele_idx,
            #                                                wavenumber,
            #                                                [x,y,z],
            #                                                distance_to_edge_tol,
            #                                                near_singular_tol,
            #                                                is_singular)
            # Z_entry = gaussQuadrature(areas[global_testele_idx],
            #                           testIntegrand1,
            #                           test_quadrature_points[global_testele_idx],
            #                           test_quadrature_weights)
            Z_entry = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_testele_idx, global_srcele_idx)
            V_vec[global_testele_idx] += Z_entry * J_vec[global_srcele_idx]
        end
    end
end # computeRHSContributionSoundSoft!

@views function computeZArray(pulse_mesh::PulseMesh,
                                           wavenumber,
                                           distance_to_edge_tol,
                                           near_singular_tol,
                                           global_test_idx,
                                           global_src_idxs)
    num_src_eles = length(global_src_idxs)
    Z_array = Array{ComplexF64,1}(undef, num_src_eles)
    for local_src_idx = 1:num_src_eles
        global_src_idx = global_src_idxs[local_src_idx]
        Z_array[local_src_idx] = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_idx, global_src_idx)
    end
    return(Z_array)
end

@views function computeRHSContributionACA(pulse_mesh::PulseMesh,
                                           wavenumber,
                                           distance_to_edge_tol,
                                           near_singular_tol,
                                           approximation_tol,
                                           test_node::Node,
                                           src_node::Node)
    @unpack num_elements = pulse_mesh
    num_ele_test_node = length(test_node.element_idxs)
    num_ele_src_node = length(src_node.element_idxs)
    U = Array{ComplexF64, 2}(undef, num_ele_test_node, 1)
    V = Array{ComplexF64, 2}(undef, 1, num_ele_src_node)
    I1 = 1
    R_tilde = zeros(ComplexF64, num_ele_test_node, num_ele_src_node) # change to not store entire matrix?
    #initialization
    global_test_ele_idx = test_node.element_idxs[I1]
    global_src_ele_idxs = src_node.element_idxs
    R_tilde[I1,:] = computeZArray(pulse_mesh, wavenumber, distance_to_edge_tol,
                             near_singular_tol, global_test_ele_idx, global_src_ele_idxs)
    R_tilde_J1, J1 = findmax(abs.(R_tilde[I1,:]))
    V[1,:] = R_tilde[I1,:] ./ R_tilde_J1
    global_src_ele_idx = src_node.element_idxs[J1]
    global_test_ele_idxs = test_node.element_idxs
    R_tilde[:,J1] = computeZArray(pulse_mesh, wavenumber, distance_to_edge_tol,
                             near_singular_tol, global_src_ele_idx, global_test_ele_idxs)
    U[:,1] = R_tilde[:,J1]
    norm_Z_tilde_1 = sqrt(norm(U[:,1])^2*norm(V[1,:])^2)
    # end initialization
    norm_Z_tilde_k = norm_Z_tilde_1
    k = 2
    while norm(U[:,k-1])*norm(V[k-1,:]) > approximation_tol * norm_Z_tilde_k #&& k <=3
        Ik = findmax(abs.(R_tilde[:,J1]))[2]
        Z_Ik_row = zeros(ComplexF64, num_ele_src_node)
        global_test_ele_idx = test_node.element_idxs[Ik]
        for src_ele_idx = 1:num_ele_src_node
            global_src_ele_idx = src_node.element_idxs[src_ele_idx]
            Z_Ik_row[src_ele_idx] = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_ele_idx, global_src_ele_idx)
        end
        sum_uv_term = zeros(ComplexF64, num_ele_src_node)
        for idx = 1:k-1
            sum_uv_term += U[Ik,idx] * V[idx,:]
        end
        R_tilde[Ik,:] = Z_Ik_row - sum_uv_term
        R_tilde_Jk, Jk = findmax(abs.(R_tilde[Ik,:]))
        V = cat(V, transpose(R_tilde[Ik,:] ./ R_tilde_Jk), dims=1)
        Z_Jk_col = zeros(ComplexF64, num_ele_test_node)
        global_src_ele_idx = src_node.element_idxs[Jk]
        for test_ele_idx = 1:num_ele_test_node
            global_test_ele_idx = test_node.element_idxs[test_ele_idx]
            Z_Jk_col[test_ele_idx] = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_ele_idx, global_src_ele_idx)
        end
        sum_uv_term = zeros(ComplexF64, num_ele_test_node)
        for idx = 1:k-1
            sum_uv_term += V[idx,Jk] * U[:,idx]
        end
        R_tilde[:,Jk] = Z_Jk_col - sum_uv_term
        U = cat(U, R_tilde[:,Jk], dims=2)
        # sum_term = 0
        # for j = 1:k-1
        #     sum_term += abs(transpose(U[:,j])*U[:,k]) * abs(transpose(V[j,:])*V[k,:])
        # end
        # norm_Z_tilde_k = sqrt(norm_Z_tilde_k + 2*sum_term + norm(U[:,k])^2*norm(V[k,:])^2)
        sum_term = 0
        for j = 1:k-1
            sum_term += abs(transpose(U[:,j])*U[:,k]) * abs(transpose(V[j,:])*V[k,:])
        end
        norm_Z_tilde_k = sqrt(norm_Z_tilde_k^2 + 2*sum_term + norm(U[:,k])^2*norm(V[k,:])^2)
        # println("compare inside ", norm_Z_tilde_k)
        # # norm_Z_tilde_k = norm(U*V) #bad way, but should be accurate
        # println("               ", norm_Z_tilde_k)
        k += 1
    end # while
    return(U,V)
    # start off reviewing ACA paper and implementing the algorithm here given two nodes

####### left off with this primitive version of the algorithm. It doesnt pass test 3 so it may have a bug or the nodes just arent well separated enough (they are touching...)

end #computeRHSContributionACA

@views function computeZEntrySoundSoft(pulse_mesh::PulseMesh, wavenumber, distance_to_edge_tol, near_singular_tol, test_ele_idx::Int64, src_ele_idx::Int64)
    # currently no unit test
    @unpack areas,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    is_singular = test_ele_idx == src_ele_idx
    testIntegrand(x,y,z) = scalarGreensIntegration(pulse_mesh,
                                                   src_ele_idx,
                                                   wavenumber,
                                                   [x,y,z],
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    Z_entry = gaussQuadrature(areas[test_ele_idx],
                              testIntegrand,
                              test_quadrature_points[test_ele_idx],
                              test_quadrature_weights)
    return(Z_entry)
end # computeZEntry
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
