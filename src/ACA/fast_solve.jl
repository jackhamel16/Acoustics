# dependencies: fill.jl greens_functions.jl mesh.jl ACA.jl octree.jl
using LinearAlgebra
using IterativeSolvers
using LinearMaps
using Parameters

@views function fullMatvecACA(pulse_mesh::PulseMesh, octree::Octree, J::AbstractArray{T,1})::Array{T,1} where T
    # This function implicitly computes Z*J=V for the entire Z matrix using the
    #   sub-Z matrices computed directly or compressed as U and V for interactions
    #   between the elements in nodes of the octree.
    @unpack num_elements = pulse_mesh
    V = zeros(ComplexF64, num_elements)
    leaf_nodes = octree.nodes[octree.leaf_node_idxs]
    num_nodes = length(leaf_nodes)
    for test_node_idx = 1:num_nodes
        test_node = leaf_nodes[test_node_idx]
        for src_node_idx = 1:num_nodes
            src_node = leaf_nodes[src_node_idx]
            sub_J = J[src_node.element_idxs]
            sub_Z = test_node.node2node_Z_matrices[src_node_idx] # this produces type instability, but unavoidable
            sub_V = subMatvecACA(sub_Z, sub_J)
            V[test_node.element_idxs] += sub_V
        end
    end
    return(V)
end # fullMatvecACA

@views function solveSoundSoftIEACA(pulse_mesh::PulseMesh,
                                    num_levels::Int64,
                                    excitation::Function,
                                    wavenumber,
                                    distance_to_edge_tol,
                                    near_singular_tol,
                                    compression_distance,
                                    ACA_approximation_tol)
    # Solves for the surface unknowns for the problem geometry described by pulse_mesh
    #   and excitation using ACA when suitable
    # num_levels determines the number of levels in the octree for ACA
    # distance_to_edge_tol and near_singular_tol are paramters for integration routines
    # compression_distance determines when to use ACA (see lower lvl func descriptions)
    # ACA_approximation_tol lower means higher rank approximations are used in ACA
    # returns an array of the unknowns named sources
    @unpack num_elements = pulse_mesh
    octree = createOctree(num_levels, pulse_mesh)
    fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                  distance_to_edge_tol, near_singular_tol,
                                  compression_distance, ACA_approximation_tol)
    rhs = zeros(ComplexF64, num_elements)
    rhsFill!(pulse_mesh, excitation, rhs)
    fullMatvecWrapped(J) = fullMatvecACA(pulse_mesh, octree, J)
    fullMatvecLinearMap = LinearMap(fullMatvecWrapped, num_elements)
    sources = zeros(ComplexF64, num_elements)
    gmres!(sources, fullMatvecLinearMap, rhs)
    return(sources)
end #solveSoundSoftIEACA

@views function subMatvecACA(sub_Z::AbstractArray{T,2}, sub_J::AbstractArray{T,1})::Array{T,1} where T
    # computes the matrix-vector product between sub_Z and sub_J when sub_Z is
    #   not decomposed
    return(sub_Z * sub_J)
end #subMatvecACA

@views function subMatvecACA(sub_Z::Tuple{Array{T,2},Array{T,2}}, sub_J::AbstractArray{T,1})::Array{T,1} where T
    # computes the matrix-vector product between sub_Z and sub_J when sub_Z is
    #   decomposed into U and V matrices
    return(sub_Z[1] * (sub_Z[2] * sub_J))
end #subMatvecACA
