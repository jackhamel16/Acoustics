using Test

include("../../src/quadrature.jl")
include("../../src/mesh.jl")
include("../../src/greens_functions.jl")
include("../../src/octree.jl")
include("../../src/fill.jl")

include("../../src/ACA.jl")

@testset "ACA tests" begin
    @testset "computeRHSContribution tests" begin
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
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        J_vec = randn(ComplexF64, pulse_mesh.num_elements)

        # test self-interactions of top node
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        node = octree.nodes[1]
        sol_V_vec = z_matrix * J_vec
        V_vec = zeros(ComplexF64, pulse_mesh.num_elements)
        computeRHSContributionSoundSoft!(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, node, node, J_vec, V_vec)
        @test isapprox(V_vec, sol_V_vec, rtol=1e-15)

        # test self-interactions of leaf node
        num_levels = 2
        octree = createOctree(num_levels, pulse_mesh)
        node = octree.nodes[4]
        sub_z = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        for node1_idx = node.element_idxs
            for node2_idx = node.element_idxs
                sub_z[node1_idx,node2_idx] = z_matrix[node1_idx,node2_idx]
            end
        end
        sol_V_vec = sub_z * J_vec
        V_vec = zeros(ComplexF64, pulse_mesh.num_elements)
        computeRHSContributionSoundSoft!(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, node, node, J_vec, V_vec)
        @test isapprox(V_vec, sol_V_vec, rtol=1e-15)

        # test non-self-interactions of leaf nodes
        V_vec = zeros(ComplexF64, pulse_mesh.num_elements)
        num_levels = 2
        octree = createOctree(num_levels, pulse_mesh)
        node1 = octree.nodes[2]; node2 = octree.nodes[3]
        sub_z = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        for node1_idx = node1.element_idxs
            for node2_idx = node2.element_idxs
                sub_z[node1_idx,node2_idx] = z_matrix[node1_idx,node2_idx]
            end
        end
        sol_V_vec = sub_z * J_vec
        computeRHSContributionSoundSoft!(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, node1, node2, J_vec, V_vec)
        @test isapprox(V_vec, sol_V_vec, rtol=1e-15)
    end #computeRHSContribution tests
    @testset "computeMatrixACA tests" begin
        approximation_tol = 1e-3
        num_rows, num_cols = 5, 5
        sol = I(num_rows)
        computeMatrixEntry(test_idx, src_idx) = convert(Float64, test_idx == src_idx)
        test_U, test_V = computeMatrixACA(computeMatrixEntry, approximation_tol, num_rows, num_cols)
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
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        test_node = octree.nodes[6]; src_node = octree.nodes[7]
        sol_sub_Z = z_matrix[test_node.element_idxs,src_node.element_idxs]
        computeMatrixEntry(test_idx,src_idx)=computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        test_U, test_V = computeMatrixACA(computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
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
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        test_node = octree.nodes[6]; src_node = octree.nodes[7]
        sol_sub_Z = z_matrix[test_node.element_idxs,src_node.element_idxs]
        singular_vals = svd(sol_sub_Z).S
        # computeMatrixArrayFunc(global_test_ele_idx, global_src_ele_idxs) = computeZArray(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_ele_idx, global_src_ele_idxs)
        # test_U, test_V = computeRHSContributionACA(computeMatrixArrayFunc, approximation_tol, test_node, src_node)
        computeMatrixEntry(test_idx,src_idx)=computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        test_U, test_V = computeMatrixACA(computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        @test isapprox(test_U*test_V, sol_sub_Z, rtol=0.17e-5)
        test_node = octree.nodes[6]; src_node = octree.nodes[9]
        sol_sub_Z = z_matrix[test_node.element_idxs,src_node.element_idxs]
        # test_U, test_V = computeRHSContributionACA(computeMatrixArrayFunc, approximation_tol, test_node, src_node)
        computeMatrixEntry(test_idx,src_idx)=computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        test_U, test_V = computeMatrixACA(computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
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
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        test_node = octree.nodes[14]; src_node = octree.nodes[17]
        # test_node = octree.nodes[13]; src_node = octree.nodes[11]
        sol_sub_Z = z_matrix[test_node.element_idxs,src_node.element_idxs]
        computeMatrixEntry(test_idx,src_idx)=computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        test_U, test_V = computeMatrixACA(computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        # computeMatrixArrayFunc(global_test_ele_idx, global_src_ele_idxs) = computeZArray(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_ele_idx, global_src_ele_idxs)
        # test_U, test_V = computeRHSContributionACA(computeMatrixArrayFunc, approximation_tol, test_node, src_node)
        @test isapprox(test_U*test_V, sol_sub_Z, rtol=1e-15)

        # println(norm(test_U*test_V-sol_sub_Z)/norm(sol_sub_Z))
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
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
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
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        J_vec = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V_vec = z_matrix * J_vec
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        test_V = computeZJMatVec(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, J_vec)
        @test isapprox(test_V, sol_V_vec, rtol=1e-15)
        num_levels = 4
        octree = createOctree(num_levels, pulse_mesh)
        test_V = computeZJMatVec(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, J_vec)
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
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        J_vec = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V_vec = z_matrix * J_vec
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        test_V = computeZJMatVec(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, J_vec)
        @test isapprox(test_V, sol_V_vec, rtol=1e-14)
    end # computeZJMatVec tests
    @testset "nodeMatrixFillACA tests" begin
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
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        sol = z_matrix[1,2]
        test_node = octree.nodes[6]
        src_node = octree.nodes[7]
        sub_Z = zeros(ComplexF64, 1, 1)
        nodeMatrixFill!(pulse_mesh, test_node, src_node, testIntegrand, sub_Z)
        @test isapprox(sub_Z[1,1], sol, rtol=1e-14)

        sol = z_matrix[2,8]
        test_node = octree.nodes[7]
        src_node = octree.nodes[13]
        sub_Z = zeros(ComplexF64, 1, 1)
        nodeMatrixFill!(pulse_mesh, test_node, src_node, testIntegrand, sub_Z)
        @test isapprox(sub_Z[1,1], sol, rtol=1e-14)

        sol = z_matrix[[1,2],[3,4]]
        test_node = octree.nodes[2]
        src_node = octree.nodes[3]
        sub_Z = zeros(ComplexF64, 2, 2)
        nodeMatrixFill!(pulse_mesh, test_node, src_node, testIntegrand, sub_Z)
        @test isapprox(sub_Z, sol, rtol=1e-14)
    end # nodeMatrixFillACA tests
end # ACA tests
