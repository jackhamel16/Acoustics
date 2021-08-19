using Test

include("../../src/quadrature.jl")
include("../../src/code_structures/mesh.jl")
include("../../src/greens_functions.jl")
include("../../src/code_structures/octree.jl")
include("../../src/fill.jl")

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
    end #computeRHSContributionACA
    @testset "computeZEntrySoundSoft tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0

        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        global_test_idx = 1
        global_src_idx = 1
        test_Z_entry = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_idx, global_src_idx)
        @test isapprox(test_Z_entry, z_matrix[global_test_idx, global_src_idx], rtol=1e-14)
        global_test_idx = 2
        global_src_idx = 5
        test_Z_entry = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_idx, global_src_idx)
        @test isapprox(test_Z_entry, z_matrix[global_test_idx, global_src_idx], rtol=1e-14)
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        test_idx = 1
        src_idx = 1
        test_Z_entry = computeZEntrySoundSoft(pulse_mesh, octree.nodes[1], octree.nodes[1], wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        @test isapprox(test_Z_entry, z_matrix[test_idx, src_idx], rtol=1e-14)
    end # computeZEntrySoundSoft tests
    @testset "computedZdkEntrySoundSoft tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0

        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensKDerivIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       is_singular)
        dzdk_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, dzdk_matrix)
        global_test_idx = 1
        global_src_idx = 1
        test_dZdk_entry = computedZdkEntrySoundSoft(pulse_mesh, octree.nodes[1], octree.nodes[1], wavenumber, global_test_idx, global_src_idx)
        @test isapprox(test_dZdk_entry, dzdk_matrix[global_test_idx, global_src_idx], rtol=1e-14)
        global_test_idx = 2
        global_src_idx = 5
        test_dZdk_entry = computedZdkEntrySoundSoft(pulse_mesh, octree.nodes[1], octree.nodes[1], wavenumber, global_test_idx, global_src_idx)
        @test isapprox(test_dZdk_entry, dzdk_matrix[global_test_idx, global_src_idx], rtol=1e-14)
        test_idx = 1
        src_idx = 1
        test_dZdk_entry = computedZdkEntrySoundSoft(pulse_mesh, octree.nodes[1], octree.nodes[1], wavenumber, test_idx, src_idx)
        @test isapprox(test_dZdk_entry, dzdk_matrix[test_idx, src_idx], rtol=1e-14)
    end # computedZdkEntrySoundSoft tests
    @testset "computeZJMatVec tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0

        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        J_vec = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V_vec = z_matrix * J_vec
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        test_V = computeZJMatVec(pulse_mesh, octree, J_vec)
        @test isapprox(test_V, sol_V_vec, rtol=1e-15)
        num_levels = 4
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        test_V = computeZJMatVec(pulse_mesh, octree, J_vec)
        @test isapprox(test_V, sol_V_vec, rtol=1e-15)

        mesh_filename = "examples/test/circular_plate_1m.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        J_vec = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V_vec = z_matrix * J_vec
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        test_V = computeZJMatVec(pulse_mesh, octree, J_vec)
        @test isapprox(test_V, sol_V_vec, rtol=0.2e-5)
    end # computeZJMatVec tests
end # ACA tests
