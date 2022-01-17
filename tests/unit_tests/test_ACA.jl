using Test

include("../../src/quadrature.jl")
include("../../src/code_structures/mesh.jl")
include("../../src/code_structures/octree.jl")
include("../../src/greens_functions.jl")
include("../../src/fill.jl")
include("../../src/ACA/ACA_fill.jl")

include("../../src/ACA/ACA.jl")

@testset "ACA tests" begin
    @testset "computeMatrixACA tests" begin
        approximation_tol = 1e-3
        num_rows, num_cols = 5, 5
        sol = I(num_rows)
        computeMatrixEntry(test_idx, src_idx)::Float64 = convert(Float64, test_idx == src_idx)
        func_return_type = Val(Float64)
        test_U, test_V = computeMatrixACA(func_return_type, computeMatrixEntry, approximation_tol, num_rows, num_cols)
        @test isapprox(test_U*test_V, sol, rtol=1e-14)

        # nodes are not well-separated for this test
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        approximation_tol = 1e-3
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        test_node = octree.nodes[6]; src_node = octree.nodes[7]
        sol_sub_Z = z_matrix[test_node.element_idxs,src_node.element_idxs]
        computeMatrixEntry(test_idx,src_idx)=computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        func_return_type = Val(ComplexF64)
        test_U, test_V = computeMatrixACA(func_return_type, computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        @test isapprox(test_U*test_V, sol_sub_Z, rtol=1e-14)

        # in this test, triangles are separated by ~80 elements
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        approximation_tol = 1e-4
        mesh_filename = "examples/test/disjoint_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        test_node = octree.nodes[6]; src_node = octree.nodes[7]
        sol_sub_Z = z_matrix[test_node.element_idxs,src_node.element_idxs]
        # singular_vals = svd(sol_sub_Z).S
        computeMatrixEntry(test_idx,src_idx)=computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        test_U, test_V = computeMatrixACA(func_return_type, computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        @test isapprox(test_U*test_V, sol_sub_Z, rtol=0.17e-5)

        test_node = octree.nodes[6]; src_node = octree.nodes[9]
        sol_sub_Z = z_matrix[test_node.element_idxs,src_node.element_idxs]
        computeMatrixEntry(test_idx,src_idx)=computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        test_U, test_V = computeMatrixACA(func_return_type, computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        @test isapprox(test_U*test_V, sol_sub_Z, rtol=0.22e-5)

        # in this test, triangles are separated by ~800 elements
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        approximation_tol = 1e-16 # uses maximum rank to approximate Z
        mesh_filename = "examples/test/far_apart_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 5
        octree = createOctree(num_levels, pulse_mesh)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        test_node = octree.nodes[14]; src_node = octree.nodes[17]
        sol_sub_Z = z_matrix[test_node.element_idxs,src_node.element_idxs]
        computeMatrixEntry(test_idx,src_idx)=computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        test_U, test_V = computeMatrixACA(func_return_type, computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        @test isapprox(test_U*test_V, sol_sub_Z, rtol=1e-15)
    end # computeMatrixACA
end # ACA tests
