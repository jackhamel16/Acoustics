# dependencies: fill.jl greens_functions.jl mesh.jl ACA.jl octree.jl

@views function fullMatvecACA(pulse_mesh::PulseMesh, octree::Octree, J::AbstractArray{T,1}) where T
    # This function implicitly computes Z*J=V for the entire Z matrix using the
    # sub-Z matrices computed directly or compressed as U and V for nodes of the octree.

    #initialize V
    # for each test leaf node
        #for each src leaf node
            # get element idxs in src node and then build sub-J by indexing J
            # get interaction z matrix
            # if Z, multiply Z*J=V
            # if U and V, multiply U*(V*J)=V
            # get element idxs in test node and add sub-V elements to correct V elements
    @unpack num_elements = pulse_mesh
    V = zeros(ComplexF64, num_elements)
    leaf_nodes = octree.nodes[octree.leaf_node_idxs]
    num_nodes = length(leaf_nodes)
    for test_node_idx = 1:num_nodes
        test_node = leaf_nodes[test_node_idx]
        for src_node_idx = 1:num_nodes
            src_node = leaf_nodes[src_node_idx]
            sub_J = J[src_node.element_idxs]
            sub_Z = test_node.node2node_Z_matrices[src_node_idx]
            sub_V = subMatvecACA(sub_Z, sub_J)
            V[test_node.element_idxs] += sub_V
        end
    end
    return(V)
end # fullMatvecACA

@views function subMatvecACA(sub_Z::AbstractArray{T,2}, sub_J::AbstractArray{T,1}) where T
    return(sub_Z * sub_J)
end #subMatvecACA

@views function subMatvecACA(sub_Z::Tuple{Array{T,2},Array{T,2}}, sub_J::AbstractArray{T,1}) where T
    return(sub_Z[1] * (sub_Z[2] * sub_J))
end #subMatvecACA
