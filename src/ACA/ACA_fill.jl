# dependencies: ACA.jl greens_functions.jl mesh.jl octree.jl quadrature.jl

################################################################################
# The file contains functions that fill node2node_Z_matrices and               #
# node2node_dZdk_matrices the belong to the Nodes in an Octree.  These are     #
# analagous to the fill functions in fill.jl, but more complicated due to ACA. #
################################################################################

@views function computeZEntrySoundSoft(pulse_mesh::PulseMesh,
                                       wavenumber,
                                       distance_to_edge_tol,
                                       near_singular_tol,
                                       test_ele_idx::Int64,
                                       src_ele_idx::Int64)::ComplexF64
    # Computes the sounds soft IE Z matrix values for the interaction between the test element
    #   at global index test_ele_idx and source element at global index src_ele_idx.
    # Returns the requested entry of the Z matrix
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
                              test_quadrature_weights)::ComplexF64
    return(Z_entry)
end # function computeZEntrySoundSoft

@views function computeZEntrySoundSoft(pulse_mesh::PulseMesh,
                                       test_node::Node,
                                       src_node::Node,
                                       wavenumber,
                                       distance_to_edge_tol,
                                       near_singular_tol,
                                       test_idx::Int64,
                                       src_idx::Int64)
    # Alternative implementation of computeZEntrySoundSoft that can be wrapped up to give to
    #   computeMatrixACA easily. Rather than pass in global element idxs, the function assumes
    #   you are computing a sub-matrix of Z for all elements between test_node and src_node.
    #   test_idx and src_idx are local element idxs for elements contained in test_node and
    #   src_node, respectively.  The function assumes test_idx and src_idx are valid indices
    #   (i.e. not larger than the total number of elements in test_node or src_node)
    # Returns the requested entry of the sub-Z matrix
    global_test_idx = test_node.element_idxs[test_idx]
    global_src_idx = src_node.element_idxs[src_idx]
    Z_entry = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_idx, global_src_idx)
    return(Z_entry)
end # function computeZEntrySoundSoft

@views function computeZEntrySoftCFIE(pulse_mesh::PulseMesh,
                                       test_node::Node,
                                       src_node::Node,
                                       wavenumber,
                                       softIE_weight,
                                       distance_to_edge_tol,
                                       near_singular_tol,
                                       test_idx::Int64,
                                       src_idx::Int64)
    # Computes the sound soft CFIE Z matrix values for the interaction between a test
    #   element in test_node and src element in src_node.  The element is determined by
    #   test_idx (local to test_node) and src_idx (local to src_node). The function
    #   assumes test_idx and src_idx are valid indices (i.e. not larger than the total
    #   number of elements in test_node and src_node, respectively)
    # Returns the requested entry of the Z matrix
    global_test_idx = test_node.element_idxs[test_idx]
    global_src_idx = src_node.element_idxs[src_idx]
    @unpack areas,
            normals,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    is_singular = global_test_idx == global_src_idx
    test_normal = normals[global_test_idx,:]
    testIntegrand(x,y,z) = softIE_weight *
                           scalarGreensIntegration(pulse_mesh,
                                                   global_src_idx,
                                                   wavenumber,
                                                   [x,y,z],
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular) +
                           (1-softIE_weight) * im *
                           scalarGreensNormalDerivativeIntegration(pulse_mesh,
                                                                   global_src_idx,
                                                                   wavenumber,
                                                                   [x,y,z],
                                                                   test_normal,
                                                                   is_singular)
    Z_entry = gaussQuadrature(areas[global_test_idx],
                              testIntegrand,
                              test_quadrature_points[global_test_idx],
                              test_quadrature_weights)::ComplexF64
    return(Z_entry)
end # function computeZEntrySoftCFIE

@views function computedZdkEntrySoundSoft(pulse_mesh::PulseMesh,
                                          test_node::Node,
                                          src_node::Node,
                                          wavenumber,
                                          test_idx::Int64,
                                          src_idx::Int64)
    # Computes an entry of the matrix dZ/dk for an interaction between a src element
    #   in src_node and test element in test_node localted at src_idx and test_idx.
    #   Assumes you are computing a sub-matrix of dZ/dk for all elements between test_node and src_node.
    #   test_idx and src_idx are local element idxs for elements contained in test_node and
    #   src_node, respectively.  The function assumes test_idx and src_idx are valid indices
    #   (i.e. not larger than the total number of elements in test_node or src_node)
    # Returns the requested entry of the sub-dZ/dk matrix
    global_test_idx = test_node.element_idxs[test_idx]
    global_src_idx = src_node.element_idxs[src_idx]
    @unpack areas,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    is_singular = global_test_idx == global_src_idx
    testIntegrand(x,y,z) = scalarGreensKDerivIntegration(pulse_mesh,
                                                         global_src_idx,
                                                         wavenumber,
                                                         [x,y,z],
                                                         is_singular)
    Z_entry = gaussQuadrature(areas[global_test_idx],
                              testIntegrand,
                              test_quadrature_points[global_test_idx],
                              test_quadrature_weights)::ComplexF64
    return(Z_entry)
end # function computedZdkEntrySoundSoft

@views function fillOctreedZdkMatricesSoundSoft!(pulse_mesh::PulseMesh,
                                                 octree::Octree,
                                                 wavenumber,
                                                 compression_distance,
                                                 ACA_approximation_tol)
    # Computes the sub dZ/dk matrices for interactions between nodes using ACA if the nodes
    #   are sufficiently far apart or directly computing the sub-Z matrix if too close
    # octree is the Octree object for which the sub-Z matrices will be computed and stored in
    #   i.e. it has empyt arrays stored for node2node_Z_matrices when passed as argument
    # compression distance is the number of node edge lengths between centroids of nodes dictating when ACA can be used
    # ACA_approximation_tol determines how accurately the compressed matrices represent Z
    # returns nothing
    z_entry_datatype = ComplexF64
    soundSoftDerivTestIntegrand(r_test, global_src_idx, is_singular) = scalarGreensKDerivIntegration(pulse_mesh,
                                                                                global_src_idx,
                                                                                wavenumber,
                                                                                r_test,
                                                                                is_singular)
    num_leaves = length(octree.leaf_node_idxs)
    leaf_edge_length = octree.nodes[octree.leaf_node_idxs[1]].bounds[1][2] -
                       octree.nodes[octree.leaf_node_idxs[1]].bounds[1][1]
    min_separation = compression_distance * leaf_edge_length
    for local_test_node_idx = 1:num_leaves
        global_test_node_idx = octree.leaf_node_idxs[local_test_node_idx]
        test_node = octree.nodes[global_test_node_idx]
        for local_src_node_idx = 1:num_leaves
            global_src_node_idx = octree.leaf_node_idxs[local_src_node_idx]
            src_node = octree.nodes[global_src_node_idx]
            if norm(src_node.centroid-test_node.centroid) > min_separation # use ACA
                num_rows = length(test_node.element_idxs)
                num_cols = length(src_node.element_idxs)
                computeMatrixEntry(test_idx, src_idx) = computedZdkEntrySoundSoft(pulse_mesh,
                                                            test_node, src_node, wavenumber,
                                                            test_idx, src_idx)
                compressed_sub_Z = computeMatrixACA(Val(z_entry_datatype), computeMatrixEntry,
                                                    ACA_approximation_tol, num_rows, num_cols)
                append!(test_node.node2node_dZdk_matrices, [compressed_sub_Z])
            else # use direct Z calculation
                sub_Z_matrix = Array{ComplexF64,2}(undef, length(test_node.element_idxs), length(src_node.element_idxs))
                nodeMatrixFill!(pulse_mesh, test_node, src_node, soundSoftDerivTestIntegrand, sub_Z_matrix)
                append!(test_node.node2node_dZdk_matrices, [sub_Z_matrix])
            end # if-else
        end
    end
end #function fillOctreedZdkMatricesSoundSoft!

function fillOctreeZMatricesGeneral!(pulse_mesh::PulseMesh,
                                     octree::Octree,
                                     computeZEntry::Function,
                                     testIntegrand::Function,
                                     compression_distance,
                                     ACA_approximation_tol,
                                     use_normal=false)
    # Core general purpose routine to fill sub-Z matrices of octree nodes
    # computeZEntry is prewrapped so that its only arguments are as shown below
    #   and is the function to compute a single entry of the Z matrix used in ACA
    # testIntegrand is the integrand of the test integral in the MoM formulation
    #   which is used to fill a sub-Z matrix when it cannot be compressed with ACA
    z_entry_datatype = ComplexF64
    num_leaves = length(octree.leaf_node_idxs)
    leaf_edge_length = octree.nodes[octree.leaf_node_idxs[1]].bounds[1][2] -
                       octree.nodes[octree.leaf_node_idxs[1]].bounds[1][1]
    min_separation = compression_distance * leaf_edge_length
    for local_test_node_idx = 1:num_leaves
        global_test_node_idx = octree.leaf_node_idxs[local_test_node_idx]
        test_node = octree.nodes[global_test_node_idx]
        for local_src_node_idx = 1:num_leaves
            global_src_node_idx = octree.leaf_node_idxs[local_src_node_idx]
            src_node = octree.nodes[global_src_node_idx]
            if norm(src_node.centroid-test_node.centroid) > min_separation # use ACA
                num_rows = length(test_node.element_idxs)
                num_cols = length(src_node.element_idxs)
                computeMatrixEntry(test_idx, src_idx) = computeZEntry(test_node, src_node,
                                                                      test_idx, src_idx)
                compressed_sub_Z = computeMatrixACA(Val(z_entry_datatype), computeMatrixEntry,
                                                    ACA_approximation_tol, num_rows, num_cols)
                append!(test_node.node2node_Z_matrices, [compressed_sub_Z])
            else # use direct Z calculation
                # sub_Z_matrix = zeros(ComplexF64, length(test_node.element_idxs), length(src_node.element_idxs))
                sub_Z_matrix = Array{ComplexF64,2}(undef, length(test_node.element_idxs), length(src_node.element_idxs))
                if use_normal == false
                    nodeMatrixFill!(pulse_mesh, test_node, src_node, testIntegrand, sub_Z_matrix)
                else
                    nodeMatrixNormalDerivFill!(pulse_mesh, test_node, src_node, testIntegrand, sub_Z_matrix)
                end
                append!(test_node.node2node_Z_matrices, [sub_Z_matrix])
            end # if-else
        end
    end
end # function fillOctreeZMatricesGeneral!

@views function fillOctreeZMatricesSoundSoft!(pulse_mesh::PulseMesh,
                                              octree::Octree,
                                              wavenumber,
                                              distance_to_edge_tol,
                                              near_singular_tol,
                                              compression_distance,
                                              ACA_approximation_tol)
    # Computes the sub Z matrices for interactions between nodes using ACA if the nodes
    #   are sufficiently far apart or directly computing the sub-Z matrix if too close
    # octree is the Octree object for which the sub-Z matrices will be computed and stored in
    #   i.e. it has empyt arrays stored for node2node_Z_matrices when passed as argument
    # compression distance is the number of node edge lengths between centroids of nodes dictating when ACA can be used
    # ACA_approximation_tol determines how accurately the compressed matrices represent Z
    # returns nothing
    z_entry_datatype = ComplexF64
    soundSoftTestIntegrand(r_test, global_src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, global_src_idx,
                                                                      wavenumber, r_test, distance_to_edge_tol,
                                                                      near_singular_tol, is_singular)
    computeZEntryIntermediate(test_node, src_node, test_idx, src_idx) = computeZEntrySoundSoft(pulse_mesh, test_node, src_node,
                                                                            wavenumber, distance_to_edge_tol,
                                                                            near_singular_tol, test_idx, src_idx)
    fillOctreeZMatricesGeneral!(pulse_mesh, octree, computeZEntryIntermediate,
                                soundSoftTestIntegrand, compression_distance, ACA_approximation_tol)
end

@views function fillOctreeZMatricesSoundSoftCFIE!(pulse_mesh::PulseMesh,
                                              octree::Octree,
                                              wavenumber,
                                              softIE_weight,
                                              distance_to_edge_tol,
                                              near_singular_tol,
                                              compression_distance,
                                              ACA_approximation_tol)
    # Computes the sub Z matrices for interactions between nodes using ACA if the nodes
    #   are sufficiently far apart or directly computing the sub-Z matrix if too close
    # octree is the Octree object for which the sub-Z matrices will be computed and stored in
    #   i.e. it has empyt arrays stored for node2node_Z_matrices when passed as argument
    # compression distance is the number of node edge lengths between centroids of nodes dictating when ACA can be used
    # ACA_approximation_tol determines how accurately the compressed matrices represent Z
    # returns nothing
    z_entry_datatype = ComplexF64
    use_normal = true
    soundSoftCFIETestIntegrand(r_test, global_src_idx, test_normal, is_singular) = softIE_weight *
                                                                      scalarGreensIntegration(pulse_mesh, global_src_idx,
                                                                          wavenumber, r_test, distance_to_edge_tol,
                                                                          near_singular_tol, is_singular) +
                                                                      (1-softIE_weight) * im *
                                                                      scalarGreensNormalDerivativeIntegration(pulse_mesh,
                                                                          global_src_idx, wavenumber, r_test, test_normal,
                                                                          is_singular)
    computeZEntryForACA(test_node, src_node, test_idx, src_idx) = computeZEntrySoftCFIE(pulse_mesh, test_node, src_node,
                                                                            wavenumber, softIE_weight, distance_to_edge_tol,
                                                                            near_singular_tol, test_idx, src_idx)
    fillOctreeZMatricesGeneral!(pulse_mesh, octree, computeZEntryForACA,
                                soundSoftCFIETestIntegrand, compression_distance,
                                ACA_approximation_tol, use_normal)
end

@views function nodeMatrixFill!(pulse_mesh::PulseMesh,
                        test_node::Node,
                        src_node::Node,
                        testIntegrand::Function,
                        sub_z_matrix::AbstractArray{ComplexF64, 2})
    # Directly computes the sub-Z matrix for interactions between elements in test_node
    #   and src_node.
    # Results stored in sub_z_matrix
    @unpack elements,
            areas,
            nodes,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    num_src_elements = length(src_node.element_idxs)
    num_test_elements = length(test_node.element_idxs)
    for local_src_idx in 1:num_src_elements
        global_src_idx = src_node.element_idxs[local_src_idx]
        for local_test_idx in 1:num_test_elements
            global_test_idx = test_node.element_idxs[local_test_idx]
            is_singular = (global_src_idx == global_test_idx)
            test_tri_nodes = getTriangleNodes(global_test_idx, elements, nodes)
            src_tri_nodes = getTriangleNodes(global_src_idx, elements, nodes)
            testIntegrandXYZ(x,y,z) = testIntegrand([x,y,z], global_src_idx, is_singular)
            sub_z_matrix[local_test_idx, local_src_idx] = gaussQuadrature(areas[global_test_idx],
                                                           testIntegrandXYZ,
                                                           test_quadrature_points[global_test_idx],
                                                           test_quadrature_weights)
        end
    end
end # function nodeMatrixFill!

@views function nodeMatrixNormalDerivFill!(pulse_mesh::PulseMesh,
                        test_node::Node,
                        src_node::Node,
                        testIntegrand::Function,
                        sub_z_matrix::AbstractArray{ComplexF64, 2})
    # Directly computes the sub-Z matrix for interactions between elements in test_node
    #   and src_node when integrand needs the test normal vector.
    # Results stored in sub_z_matrix
    @unpack elements,
            areas,
            normals,
            nodes,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    num_src_elements = length(src_node.element_idxs)
    num_test_elements = length(test_node.element_idxs)
    for local_src_idx in 1:num_src_elements
        global_src_idx = src_node.element_idxs[local_src_idx]
        for local_test_idx in 1:num_test_elements
            global_test_idx = test_node.element_idxs[local_test_idx]
            is_singular = (global_src_idx == global_test_idx)
            test_tri_nodes = getTriangleNodes(global_test_idx, elements, nodes)
            src_tri_nodes = getTriangleNodes(global_src_idx, elements, nodes)
            testIntegrandXYZ(x,y,z) = testIntegrand([x,y,z], global_src_idx, normals[global_test_idx,:], is_singular)
            sub_z_matrix[local_test_idx, local_src_idx] = gaussQuadrature(areas[global_test_idx],
                                                           testIntegrandXYZ,
                                                           test_quadrature_points[global_test_idx],
                                                           test_quadrature_weights)
        end
    end
end # function nodeMatrixNormalDerivFill!
