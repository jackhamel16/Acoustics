using Test

include("../../src/code_structures/mesh.jl")
include("../../src/quadrature.jl")
include("../../src/code_structures/octree.jl")
include("../../src/fill.jl")
include("../../src/greens_functions.jl")
include("../../src/ACA/ACA.jl")
include("../../src/ACA/ACA_fill.jl")


@testset "ACA_fill tests" begin
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
    @testset "computeZEntrySoftCFIE tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        softIE_weight = 0.5

        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = softIE_weight * scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        testIntegrandND(r_test, src_idx, test_normal, is_singular) = (1-softIE_weight) * im *
                                                      scalarGreensNormalDerivativeIntegration(pulse_mesh,
                                                        src_idx, wavenumber, r_test, test_normal, is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        matrixNormalDerivFill!(pulse_mesh, testIntegrandND, z_matrix)
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        test_idx = 1
        src_idx = 1
        test_node = octree.nodes[1]
        src_node = octree.nodes[1]
        global_test_idx = test_node.element_idxs[test_idx]
        global_src_idx = src_node.element_idxs[src_idx]
        test_Z_entry = computeZEntrySoftCFIE(pulse_mesh, test_node, src_node, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        @test isapprox(test_Z_entry, z_matrix[global_test_idx, global_src_idx], rtol=1e-14)
        test_idx = 2
        src_idx = 5
        global_test_idx = test_node.element_idxs[test_idx]
        global_src_idx = src_node.element_idxs[src_idx]
        test_Z_entry = computeZEntrySoftCFIE(pulse_mesh, test_node, src_node, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        @test isapprox(test_Z_entry, z_matrix[global_test_idx, global_src_idx], rtol=1e-14)

        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        test_idx = 1
        src_idx = 1
        test_node = octree.nodes[7]
        src_node = octree.nodes[9]
        global_test_idx = test_node.element_idxs[test_idx]
        global_src_idx = src_node.element_idxs[src_idx]
        test_Z_entry = computeZEntrySoftCFIE(pulse_mesh, test_node, src_node, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        @test isapprox(test_Z_entry, z_matrix[global_test_idx, global_src_idx], rtol=1e-14)
    end # computeZEntrySoftCFIE tests
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
    @testset "fillOctreeZMatricesGeneral! tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        approximation_tol = 1e-3
        compression_distance = 1.5 # should exclude all box touching sides or corners
        ACA_tol = 1e-4
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = 1
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        computeZEntry(test_node, src_node, test_idx, src_idx) = 1
        fillOctreeZMatricesGeneral!(pulse_mesh, octree, computeZEntry, testIntegrand, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[1].node2node_Z_matrices[1], z_matrix,rtol = 1e-14)

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
        computeZEntry(test_node, src_node, test_idx, src_idx) = computeZEntrySoundSoft(pulse_mesh,
                                                                  test_node, src_node, wavenumber,
                                                                  distance_to_edge_tol,
                                                                  near_singular_tol,
                                                                  test_idx, src_idx)
        fillOctreeZMatricesGeneral!(pulse_mesh, octree, computeZEntry, testIntegrand, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[6].node2node_Z_matrices[1][1], z_matrix[1,1], rtol = 1e-14)
        @test isapprox(octree.nodes[6].node2node_Z_matrices[2][1], z_matrix[1,2], rtol = 1e-14)
        @test isapprox(octree.nodes[6].node2node_Z_matrices[3][1] * octree.nodes[6].node2node_Z_matrices[3][2], [z_matrix[1,4]], rtol = 1e-14)
        @test isapprox(octree.nodes[8].node2node_Z_matrices[1][1] * octree.nodes[8].node2node_Z_matrices[1][2], [z_matrix[4,1]], rtol = 1e-14)
        @test isapprox(octree.nodes[8].node2node_Z_matrices[3][1], z_matrix[4,4], rtol = 1e-14)

        use_normal = true
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
        testIntegrandNormal(r_test, src_idx, test_normal, is_singular) = testIntegrand(r_test, src_idx, is_singular) # just test the functionality of passing in a normal vector
        computeZEntry(test_node, src_node, test_idx, src_idx) = computeZEntrySoundSoft(pulse_mesh,
                                                                  test_node, src_node, wavenumber,
                                                                  distance_to_edge_tol,
                                                                  near_singular_tol,
                                                                  test_idx, src_idx)
        fillOctreeZMatricesGeneral!(pulse_mesh, octree, computeZEntry, testIntegrandNormal, compression_distance, ACA_tol, use_normal)
        @test isapprox(octree.nodes[6].node2node_Z_matrices[1][1], z_matrix[1,1], rtol = 1e-14)
        @test isapprox(octree.nodes[6].node2node_Z_matrices[2][1], z_matrix[1,2], rtol = 1e-14)
        @test isapprox(octree.nodes[6].node2node_Z_matrices[3][1] * octree.nodes[6].node2node_Z_matrices[3][2], [z_matrix[1,4]], rtol = 1e-14)
        @test isapprox(octree.nodes[8].node2node_Z_matrices[1][1] * octree.nodes[8].node2node_Z_matrices[1][2], [z_matrix[4,1]], rtol = 1e-14)
        @test isapprox(octree.nodes[8].node2node_Z_matrices[3][1], z_matrix[4,4], rtol = 1e-14)
    end # fillOctreeZMatricesGeneral! tests
    @testset "fillOctreeZMatricesSoundSoft! tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        approximation_tol = 1e-3
        compression_distance = 1.5 # should exclude all box touching sides or corners
        ACA_tol = 1e-4
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
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[1].node2node_Z_matrices[1], z_matrix,rtol = 1e-14)

        num_levels = 2
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[2].node2node_Z_matrices[1], z_matrix[[1,2],[1,2]], rtol = 1e-14)
        @test isapprox(octree.nodes[2].node2node_Z_matrices[2], z_matrix[[1,2],[3,4]], rtol = 1e-14)
        @test isapprox(octree.nodes[2].node2node_Z_matrices[4], z_matrix[[1,2],[7,8]], rtol = 1e-14)
        @test isapprox(octree.nodes[3].node2node_Z_matrices[1], z_matrix[[3,4],[1,2]], rtol = 1e-14)
        @test isapprox(octree.nodes[4].node2node_Z_matrices[4], z_matrix[[5,6],[7,8]], rtol = 1e-14)
        @test isapprox(octree.nodes[5].node2node_Z_matrices[3], z_matrix[[7,8],[5,6]], rtol = 1e-14)

        num_levels = 3
        large_compression_dist = 1e10 # all interactions use direct Z calculation
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, large_compression_dist, ACA_tol)
        @test isapprox(octree.nodes[11].node2node_Z_matrices[5][1,1], z_matrix[5,6], rtol = 1e-14)
        @test isapprox(octree.nodes[11].node2node_Z_matrices[8][1,1], z_matrix[5,8], rtol = 1e-14)

        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        # first test types of elements of node2node_Z_matrices are correct
        UV_type = Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}
        Z_type = Array{ComplexF64,2}
        @test typeof(octree.nodes[10].node2node_Z_matrices[1]) == UV_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[2]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[3]) == UV_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[4]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[5]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[6]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[7]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[8]) == UV_type
        # Now test a few exact values
        test_node = octree.nodes[11]
        src_node = octree.nodes[13]
        @test isapprox(test_node.node2node_Z_matrices[5][1,1], z_matrix[5,6], rtol = 1e-14)
        computeMatrixEntry(test_idx,src_idx) = computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        sol_U, sol_V = computeMatrixACA(Val(ComplexF64), computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        @test isapprox(test_node.node2node_Z_matrices[8][1], sol_U, rtol = 1e-14)
        @test isapprox(test_node.node2node_Z_matrices[8][2], sol_V, rtol = 1e-14)
    end # fillOctreeZMatricesSoundSoft! tests
    @testset "fillOctreeZMatricesSoundSoftCFIE! tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        approximation_tol = 1e-3
        compression_distance = 1.5 # should exclude all box touching sides or corners
        softIE_weight = 0.5
        ACA_tol = 1e-4
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = softIE_weight * scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        testIntegrandND(r_test, src_idx, test_normal, is_singular) = (1 - softIE_weight) * im *
                                                        scalarGreensNormalDerivativeIntegration(pulse_mesh, src_idx,
                                                          wavenumber,
                                                          r_test,
                                                          test_normal,
                                                          is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        matrixNormalDerivFill!(pulse_mesh, testIntegrandND, z_matrix)
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoftCFIE!(pulse_mesh, octree, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[1].node2node_Z_matrices[1], z_matrix,rtol = 1e-14)

        num_levels = 2
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoftCFIE!(pulse_mesh, octree, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[2].node2node_Z_matrices[1], z_matrix[[1,2],[1,2]], rtol = 1e-14)
        @test isapprox(octree.nodes[2].node2node_Z_matrices[2], z_matrix[[1,2],[3,4]], rtol = 1e-14)
        @test isapprox(octree.nodes[2].node2node_Z_matrices[4], z_matrix[[1,2],[7,8]], rtol = 1e-14)
        @test isapprox(octree.nodes[3].node2node_Z_matrices[1], z_matrix[[3,4],[1,2]], rtol = 1e-14)
        @test isapprox(octree.nodes[4].node2node_Z_matrices[4], z_matrix[[5,6],[7,8]], rtol = 1e-14)
        @test isapprox(octree.nodes[5].node2node_Z_matrices[3], z_matrix[[7,8],[5,6]], rtol = 1e-14)

        num_levels = 3
        large_compression_dist = 1e10 # all interactions use direct Z calculation
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoftCFIE!(pulse_mesh, octree, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, large_compression_dist, ACA_tol)
        @test isapprox(octree.nodes[11].node2node_Z_matrices[5][1,1], z_matrix[5,6], rtol = 1e-14)
        @test isapprox(octree.nodes[11].node2node_Z_matrices[8][1,1], z_matrix[5,8], rtol = 1e-14)

        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoftCFIE!(pulse_mesh, octree, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        # first test types of elements of node2node_Z_matrices are correct
        UV_type = Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}
        Z_type = Array{ComplexF64,2}
        @test typeof(octree.nodes[10].node2node_Z_matrices[1]) == UV_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[2]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[3]) == UV_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[4]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[5]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[6]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[7]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[8]) == UV_type
        # Now test a few exact values
        test_node = octree.nodes[11]
        src_node = octree.nodes[13]
        @test isapprox(test_node.node2node_Z_matrices[5][1,1], z_matrix[5,6], rtol = 1e-14)
        computeMatrixEntry(test_idx,src_idx) = computeZEntrySoftCFIE(pulse_mesh, test_node, src_node, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        sol_U, sol_V = computeMatrixACA(Val(ComplexF64), computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        @test isapprox(test_node.node2node_Z_matrices[8][1], sol_U, rtol = 1e-14)
        @test isapprox(test_node.node2node_Z_matrices[8][2], sol_V, rtol = 1e-14)
    end # fillOctreeZMatricesSoundSoft! tests
    @testset "fillOctreedZdkMatricesSoundSoft! tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        approximation_tol = 1e-3
        compression_distance = 1.5 # should exclude all box touching sides or corners
        ACA_tol = 1e-4
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensKDerivIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       is_singular)
        dZdk_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, dZdk_matrix)
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[1].node2node_dZdk_matrices[1], dZdk_matrix, rtol = 1e-14)

        num_levels = 2
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[2].node2node_dZdk_matrices[1], dZdk_matrix[[1,2],[1,2]], rtol = 1e-14)
        @test isapprox(octree.nodes[2].node2node_dZdk_matrices[2], dZdk_matrix[[1,2],[3,4]], rtol = 1e-14)
        @test isapprox(octree.nodes[2].node2node_dZdk_matrices[4], dZdk_matrix[[1,2],[7,8]], rtol = 1e-14)
        @test isapprox(octree.nodes[3].node2node_dZdk_matrices[1], dZdk_matrix[[3,4],[1,2]], rtol = 1e-14)
        @test isapprox(octree.nodes[4].node2node_dZdk_matrices[4], dZdk_matrix[[5,6],[7,8]], rtol = 1e-14)
        @test isapprox(octree.nodes[5].node2node_dZdk_matrices[3], dZdk_matrix[[7,8],[5,6]], rtol = 1e-14)

        num_levels = 3
        large_compression_dist = 1e10 # all interactions use direct Z calculation
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber, large_compression_dist, ACA_tol)
        @test isapprox(octree.nodes[11].node2node_dZdk_matrices[5][1,1], dZdk_matrix[5,6], rtol = 1e-14)
        @test isapprox(octree.nodes[11].node2node_dZdk_matrices[8][1,1], dZdk_matrix[5,8], rtol = 1e-14)

        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber, compression_distance, ACA_tol)
        # first test types of elements of node2node_Z_matrices are correct
        UV_type = Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}
        Z_type = Array{ComplexF64,2}
        @test typeof(octree.nodes[10].node2node_dZdk_matrices[1]) == UV_type
        @test typeof(octree.nodes[10].node2node_dZdk_matrices[2]) == Z_type
        @test typeof(octree.nodes[10].node2node_dZdk_matrices[3]) == UV_type
        @test typeof(octree.nodes[10].node2node_dZdk_matrices[4]) == Z_type
        @test typeof(octree.nodes[10].node2node_dZdk_matrices[5]) == Z_type
        @test typeof(octree.nodes[10].node2node_dZdk_matrices[6]) == Z_type
        @test typeof(octree.nodes[10].node2node_dZdk_matrices[7]) == Z_type
        @test typeof(octree.nodes[10].node2node_dZdk_matrices[8]) == UV_type
        # Now test a few exact values
        test_node = octree.nodes[11]
        src_node = octree.nodes[13]
        @test isapprox(test_node.node2node_dZdk_matrices[5][1,1], dZdk_matrix[5,6], rtol = 1e-14)
        computeMatrixEntry(test_idx,src_idx) = computedZdkEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, test_idx, src_idx)
        sol_U, sol_V = computeMatrixACA(Val(ComplexF64), computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        @test isapprox(test_node.node2node_dZdk_matrices[8][1], sol_U, rtol = 1e-14)
        @test isapprox(test_node.node2node_dZdk_matrices[8][2], sol_V, rtol = 1e-14)
    end # fillOctreedZdkMatricesSoundSoft! tests
    @testset "nodeMatrixFill! tests" begin
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
    end # nodeMatrixFill! tests
    @testset "nodeMatrixNormalDerivFill! tests" begin
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
        testIntegrandND(r_test, src_idx, test_normal, is_singular) = scalarGreensNormalDerivativeIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       test_normal,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixNormalDerivFill!(pulse_mesh, testIntegrandND, z_matrix)
        sol = z_matrix[1,2]
        test_node = octree.nodes[6]
        src_node = octree.nodes[7]
        sub_Z = zeros(ComplexF64, 1, 1)
        nodeMatrixNormalDerivFill!(pulse_mesh, test_node, src_node, testIntegrandND, sub_Z)
        @test isapprox(sub_Z[1,1], sol, rtol=1e-14)

        sol = z_matrix[2,8]
        test_node = octree.nodes[7]
        src_node = octree.nodes[13]
        sub_Z = zeros(ComplexF64, 1, 1)
        nodeMatrixNormalDerivFill!(pulse_mesh, test_node, src_node, testIntegrandND, sub_Z)
        @test isapprox(sub_Z[1,1], sol, rtol=1e-14)

        sol = z_matrix[[1,2],[3,4]]
        test_node = octree.nodes[2]
        src_node = octree.nodes[3]
        sub_Z = zeros(ComplexF64, 2, 2)
        nodeMatrixNormalDerivFill!(pulse_mesh, test_node, src_node, testIntegrandND, sub_Z)
        @test isapprox(sub_Z, sol, rtol=1e-14)
    end # nodeMatrixNormalDerivFill! tests
end # ACA_fill tests
