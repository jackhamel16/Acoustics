# dependencies: fill.jl greens_functions.jl mesh.jl ACA.jl octree.jl
using LinearAlgebra
using IterativeSolvers
using LinearMaps
using Parameters

struct ACAMetrics
    num_eles_per_node::Array{Int64, 1}
    compressed_size::Int64
    uncompressed_size::Int64
    compression_ratio::Float64
    percentage_matrices_compressed::Float64
    avg_rank::Float64
    min_rank::Int64
    max_rank::Int64
    avg_num_eles::Float64
    min_num_eles::Int64
    max_num_eles::Int64
end

@views function computeACAMetrics(num_elements, octree::Octree)
    num_nodes = length(octree.leaf_node_idxs)
    num_eles_per_node = Array{Int64, 1}(undef, num_nodes)
    ranks = []
    compressed_size = 0
    for local_node_idx = 1:num_nodes
        global_node_idx = octree.leaf_node_idxs[local_node_idx]
        node = octree.nodes[global_node_idx]
        num_eles_per_node[local_node_idx] = length(node.element_idxs)
        for local_node_idx2 = 1:num_nodes
            global_node_idx2 = octree.leaf_node_idxs[local_node_idx2]
            sub_Z = node.node2node_Z_matrices[local_node_idx2]
            if typeof(sub_Z) == Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}
                append!(ranks, size(sub_Z[1], 2))
                matrix_size = length(sub_Z[1]) + length(sub_Z[2])
                compressed_size += matrix_size
            else
                matrix_size = length(sub_Z)
                compressed_size += matrix_size
            end
        end
    end
    uncompressed_size = num_elements^2
    compression_ratio = compressed_size / uncompressed_size
    percentage_matrices_compressed = length(ranks) / num_nodes^2
    num_compressed_matrices = length(ranks)
    if num_compressed_matrices != 0
        avg_rank = sum(ranks) / num_compressed_matrices
        min_rank = min(ranks...)
        max_rank = max(ranks...)
    else
        avg_rank = 0
        min_rank = 0
        max_rank = 0
    end
    avg_num_eles = sum(num_eles_per_node) / num_nodes
    min_num_eles = min(num_eles_per_node...)
    max_num_eles = max(num_eles_per_node...)
    return(ACAMetrics(num_eles_per_node,
                      compressed_size,
                      uncompressed_size,
                      compression_ratio,
                      percentage_matrices_compressed,
                      avg_rank,
                      min_rank,
                      max_rank,
                      avg_num_eles,
                      min_num_eles,
                      max_num_eles))
end # ACAMetrics

function printACAMetrics(metrics::ACAMetrics)
    println("Displaying ACA Metrics:")
    println("  Octree Metrics:")
    println("    Number of elements per node:")
    println("      Mean = ", metrics.avg_num_eles)
    println("      Min = ", metrics.min_num_eles)
    println("      Max = ", metrics.max_num_eles)
    println("  Matrix Metrics:")
    println("    Matrix Rank:")
    println("      Mean = ", metrics.avg_rank)
    println("      Min = ", metrics.min_rank)
    println("      Max = ", metrics.max_rank)
    println("  Compression Metrics:")
    println("    Percentage of Matrices Compressed = ", 100*metrics.percentage_matrices_compressed)
    println("    Compression Ratio = ", metrics.compression_ratio)
end

function printACAMetrics(metrics::ACAMetrics, output_file::IOStream)
    # Prints metrics to an output file
    println(output_file, "Displaying ACA Metrics:")
    println(output_file, "  Octree Metrics:")
    println(output_file, "    Number of elements per node:")
    println(output_file, "      Mean = ", metrics.avg_num_eles)
    println(output_file, "      Min = ", metrics.min_num_eles)
    println(output_file, "      Max = ", metrics.max_num_eles)
    println(output_file, "  Matrix Metrics:")
    println(output_file, "    Matrix Rank:")
    println(output_file, "      Mean = ", metrics.avg_rank)
    println(output_file, "      Min = ", metrics.min_rank)
    println(output_file, "      Max = ", metrics.max_rank)
    println(output_file, "  Compression Metrics:")
    println(output_file, "    Percentage of Matrices Compressed = ", 100*metrics.percentage_matrices_compressed)
    println(output_file, "    Compression Ratio = ", metrics.compression_ratio)
end

@views function fullMatvecACA(pulse_mesh::PulseMesh, octree::Octree, J::AbstractArray{T,1}, use_dZdk=false)::Array{T,1} where T
    # This function implicitly computes Z*J=V for the entire Z matrix using the
    #   sub-Z matrices computed directly or compressed as U and V for interactions
    #   between the elements in nodes of the octree.  if use_dZdk is true then
    #   the function computes dZ/dk*J
    @unpack num_elements = pulse_mesh
    V = zeros(ComplexF64, num_elements)
    leaf_nodes = octree.nodes[octree.leaf_node_idxs]
    num_nodes = length(leaf_nodes)
    if use_dZdk == false
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
    else
        for test_node_idx = 1:num_nodes
            test_node = leaf_nodes[test_node_idx]
            for src_node_idx = 1:num_nodes
                src_node = leaf_nodes[src_node_idx]
                sub_J = J[src_node.element_idxs]
                sub_Z = test_node.node2node_dZdk_matrices[src_node_idx] # this produces type instability, but unavoidable
                sub_V = subMatvecACA(sub_Z, sub_J)
                V[test_node.element_idxs] += sub_V
            end
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
                                    ACA_approximation_tol,
                                    gmres_tol,
                                    gmres_max_iters)
    # Solves for the surface unknowns for the problem geometry described by pulse_mesh
    #   and excitation using ACA when suitable
    # num_levels determines the number of levels in the octree for ACA
    # distance_to_edge_tol and near_singular_tol are paramters for integration routines
    # compression_distance determines when to use ACA (see lower lvl func descriptions)
    # ACA_approximation_tol lower means higher rank approximations are used in ACA
    # returns an array of the unknowns named sources
    # this is different from the below function by allowing control over GMRES params
    @unpack num_elements = pulse_mesh

    println("Filling RHS...")
    rhs = zeros(ComplexF64, num_elements)
    rhs_fill_time = @elapsed rhsFill!(pulse_mesh, excitation, rhs)
    println("  RHS fill time: ", rhs_fill_time)

    println("Filling ACA Matrix...")
    matrix_fill_time = @elapsed begin
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
    end
    println("  Matrix fill time: ", matrix_fill_time)

    println("Solving with ACA...")
    solve_time = @elapsed begin
        fullMatvecWrapped(J) = fullMatvecACA(pulse_mesh, octree, J)
        fullMatvecLinearMap = LinearMap(fullMatvecWrapped, num_elements)
        sources = zeros(ComplexF64, num_elements)
        num_iters = gmres!(sources, fullMatvecLinearMap, rhs, reltol=gmres_tol, maxiter=gmres_max_iters, log=true)[2]
        println("GMRES ", string(num_iters))
    end
    println("  Solve time: ", solve_time)

    return((sources, octree, computeACAMetrics(num_elements, octree)))
    # return(computeACAMetrics(num_elements, octree))
end #solveSoundSoftIEACA

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

    println("Filling RHS...")
    rhs = zeros(ComplexF64, num_elements)
    rhs_fill_time = @elapsed rhsFill!(pulse_mesh, excitation, rhs)
    println("  RHS fill time: ", rhs_fill_time)

    println("Filling ACA Matrix...")
    matrix_fill_time = @elapsed begin
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
    end
    println("  Matrix fill time: ", matrix_fill_time)

    println("Solving with ACA...")
    solve_time = @elapsed begin
        fullMatvecWrapped(J) = fullMatvecACA(pulse_mesh, octree, J)
        fullMatvecLinearMap = LinearMap(fullMatvecWrapped, num_elements)
        sources = zeros(ComplexF64, num_elements)
        num_iters = gmres!(sources, fullMatvecLinearMap, rhs, log=true)[2]
        println("GMRES ", string(num_iters))
    end
    println("  Solve time: ", solve_time)

    return((sources, octree, computeACAMetrics(num_elements, octree)))
    # return(computeACAMetrics(num_elements, octree))
end #solveSoundSoftIEACA

@views function solveSoundSoftIEACA(pulse_mesh::PulseMesh,
                                    octree::Octree,
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
    # different from above function because it is used when an octree already exists
    #   with Z matrices filled
    @unpack num_elements = pulse_mesh
    println("Filling RHS...")
    rhs = zeros(ComplexF64, num_elements)
    rhs_fill_time = @elapsed rhsFill!(pulse_mesh, excitation, rhs)
    println("RHS fill time: ", rhs_fill_time)

    println("Solving with ACA...")
    solve_time = @elapsed begin
        fullMatvecWrapped(J) = fullMatvecACA(pulse_mesh, octree, J)
        fullMatvecLinearMap = LinearMap(fullMatvecWrapped, num_elements)
        sources = zeros(ComplexF64, num_elements)
        num_iters = gmres!(sources, fullMatvecLinearMap, rhs, log=true)[2]
        println("GMRES ", string(num_iters))
    end
    println("Solve time: ", solve_time)

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
