using Test

include("../../src/math.jl")
include("../../src/excitation.jl")
include("../../src/code_structures/mesh.jl")
include("../../src/quadrature.jl")
include("../../src/code_structures/octree.jl")
include("../../src/fill.jl")
include("../../src/greens_functions.jl")
include("../../src/solve.jl")
include("../../src/ACA/ACA.jl")
include("../../src/ACA/ACA_fill.jl")

include("../../src/ACA/fast_solve.jl")

@testset "fast_solve tests" begin
    @testset "computeACAMetrics tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1.5 # not using and ACA compression
        ACA_tol = 1e-4
        mesh_filename = "examples/test/disjoint_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        metrics = computeACAMetrics(pulse_mesh.num_elements, octree)
        @test isapprox(length(metrics.num_eles_per_node), 1, rtol=1e-14)
        @test isapprox(metrics.num_eles_per_node[1], pulse_mesh.num_elements, rtol=1e-14)
        @test isapprox(metrics.compressed_size, pulse_mesh.num_elements^2, rtol=1e-14)
        @test isapprox(metrics.uncompressed_size, pulse_mesh.num_elements^2, rtol=1e-14)
        @test isapprox(metrics.compression_ratio, 1, rtol=1e-14)
        @test isapprox(metrics.percentage_matrices_compressed, 0, rtol=1e-14)
        @test isapprox(metrics.avg_rank, 0, rtol=1e-14)
        @test isapprox(metrics.min_rank, 0, rtol=1e-14)
        @test isapprox(metrics.max_rank, 0, rtol=1e-14)
        @test isapprox(metrics.avg_num_eles, pulse_mesh.num_elements, rtol=1e-14)
        @test isapprox(metrics.min_num_eles, pulse_mesh.num_elements, rtol=1e-14)
        @test isapprox(metrics.max_num_eles, pulse_mesh.num_elements, rtol=1e-14)


        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1.5 # not using and ACA compression
        ACA_tol = 1e-4
        mesh_filename = "examples/test/disjoint_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        metrics = computeACAMetrics(pulse_mesh.num_elements, octree)
        compressed_size = 10240
        uncompressed_size = pulse_mesh.num_elements^2
        avg_rank = 4
        @test isapprox(metrics.num_eles_per_node, [37, 35, 27, 29], rtol=1e-14)
        @test isapprox(metrics.compressed_size, compressed_size, rtol=1e-14)
        @test isapprox(metrics.uncompressed_size, uncompressed_size, rtol=1e-14)
        @test isapprox(metrics.compression_ratio, compressed_size / pulse_mesh.num_elements^2, rtol=1e-14)
        @test isapprox(metrics.percentage_matrices_compressed, 0.5, rtol=1e-14)
        @test isapprox(metrics.avg_rank, 4, rtol=1e-14)
        @test isapprox(metrics.min_rank, 4, rtol=1e-14)
        @test isapprox(metrics.max_rank, 4, rtol=1e-14)
        @test isapprox(metrics.avg_num_eles, 32, rtol=1e-14)
        @test isapprox(metrics.min_num_eles, 27, rtol=1e-14)
        @test isapprox(metrics.max_num_eles, 37, rtol=1e-14)
    end # computeACAMetrics tests
    @testset "fullMatvecACA tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1e100 # not using and ACA compression
        ACA_tol = 1e-4
        mesh_filename = "examples/test/disjoint_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber, compression_distance, ACA_tol)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        testIntegrand2(r_test, src_idx, is_singular) = scalarGreensKDerivIntegration(pulse_mesh, src_idx,
                                                      wavenumber,
                                                      r_test,
                                                      is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        dZdk_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        matrixFill!(pulse_mesh, testIntegrand2, dZdk_matrix)
        rand_J = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V = z_matrix * rand_J
        sol_V_dZdk = dZdk_matrix * rand_J
        test_V = fullMatvecACA(pulse_mesh, octree, rand_J)
        test_V_dZdk = fullMatvecACA(pulse_mesh, octree, rand_J, true)
        @test isapprox(test_V, sol_V, rtol=1e-14)
        @test isapprox(test_V_dZdk, sol_V_dZdk, rtol=1e-14)

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1.5
        ACA_tol = 1e-10
        mesh_filename = "examples/test/disjoint_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber, compression_distance, ACA_tol)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        testIntegrand2(r_test, src_idx, is_singular) = scalarGreensKDerivIntegration(pulse_mesh, src_idx,
                                                      wavenumber,
                                                      r_test,
                                                      is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        dZdk_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        matrixFill!(pulse_mesh, testIntegrand2, dZdk_matrix)
        rand_J = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V = z_matrix * rand_J
        sol_V_dZdk = dZdk_matrix * rand_J
        test_V = fullMatvecACA(pulse_mesh, octree, rand_J)
        test_V_dZdk = fullMatvecACA(pulse_mesh, octree, rand_J, true)
        @test isapprox(test_V, sol_V, rtol=1e-12)
        @test isapprox(test_V_dZdk, sol_V_dZdk, rtol=0.22e-9)

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1.5
        ACA_tol = 1e-4
        mesh_filename = "examples/test/rectangular_strips/rectangular_strip.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber, compression_distance, ACA_tol)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        testIntegrand2(r_test, src_idx, is_singular) = scalarGreensKDerivIntegration(pulse_mesh, src_idx,
                                                      wavenumber,
                                                      r_test,
                                                      is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        dZdk_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        matrixFill!(pulse_mesh, testIntegrand2, dZdk_matrix)
        rand_J = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V = z_matrix * rand_J
        sol_V_dZdk = dZdk_matrix * rand_J
        test_V = fullMatvecACA(pulse_mesh, octree, rand_J)
        test_V_dZdk = fullMatvecACA(pulse_mesh, octree, rand_J, true)
        @test isapprox(test_V, sol_V, rtol=0.6e-5)
        @test isapprox(test_V_dZdk, sol_V_dZdk, rtol=0.4e-3)
    end # fullMatvecACA tests
    @testset "solveSoftIEACA tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1e100 # not using and ACA compression
        ACA_approximation_tol = 1e-4
        mesh_filename = "examples/test/disjoint_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        num_levels = 3
        test_J = solveSoftIEACA(pulse_mesh, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        sol_J = solveSoftIE(pulse_mesh, excitation, wavenumber, distance_to_edge_tol, near_singular_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.13e-6) # not exact because of GMRES
        @test typeof(test_J[2]) == Octree
        @test typeof(test_J[3]) == ACAMetrics
        rm("solveSoftIEACA_GMRES_residual_history.txt")
        rm("solveSoftIEACA_GMRES_residual_history.png")

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1
        ACA_approximation_tol = 1e-4
        mesh_filename = "examples/test/disjoint_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        num_levels = 3
        test_J = solveSoftIEACA(pulse_mesh, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.15e-6)
        rm("solveSoftIEACA_GMRES_residual_history.txt")
        rm("solveSoftIEACA_GMRES_residual_history.png")

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangular_strips/rectangular_strip.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        sol_J = solveSoftIE(pulse_mesh, excitation, wavenumber, distance_to_edge_tol, near_singular_tol)

        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-2
        test_J = solveSoftIEACA(pulse_mesh, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.42e-2)
        rm("solveSoftIEACA_GMRES_residual_history.txt")
        rm("solveSoftIEACA_GMRES_residual_history.png")

        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-6
        test_J = solveSoftIEACA(pulse_mesh, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.34e-6)
        rm("solveSoftIEACA_GMRES_residual_history.txt")
        rm("solveSoftIEACA_GMRES_residual_history.png")

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1.5
        ACA_approximation_tol = 1e-4
        mesh_filename = "examples/test/spheres/sphere_1m_1266.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        test_J = solveSoftIEACA(pulse_mesh, octree, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        sol_J = solveSoftIE(pulse_mesh, excitation, wavenumber, distance_to_edge_tol, near_singular_tol)
        @test isapprox(sol_J, test_J, rtol=0.21e-3)
        rm("solveSoftIEACA_GMRES_residual_history.txt")
        rm("solveSoftIEACA_GMRES_residual_history.png")
    end # solveSoftIEACA tests
    @testset "solveSoftCFIEACA tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        ACA_approximation_tol = 1e-4
        softIE_weight = 0.5
        mesh_filename = "examples/test/disjoint_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, wavenumber, [x_test,y_test,z_test], l, m)
        excitation_normal_deriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(1.0, wavenumber, [x_test,y_test,z_test], l, m, normal)
        sol_J = solveSoftCFIE(pulse_mesh, excitation, excitation_normal_deriv, wavenumber, distance_to_edge_tol, near_singular_tol, softIE_weight)
        # one octree level, no compression
        num_levels = 1
        compression_distance = 1e100 # not using and ACA compression
        test_J = solveSoftCFIEACA(pulse_mesh, num_levels, excitation, excitation_normal_deriv, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.12e-7) # not exact because of GMRES
        @test typeof(test_J[2]) == Octree
        @test typeof(test_J[3]) == ACAMetrics
        rm("solveSoftCFIEACA_GMRES_residual_history.txt")
        rm("solveSoftCFIEACA_GMRES_residual_history.png")
        # 3 octree levels, no compression
        num_levels = 3
        compression_distance = 1e100
        test_J = solveSoftCFIEACA(pulse_mesh, num_levels, excitation, excitation_normal_deriv, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.13e-7)
        rm("solveSoftCFIEACA_GMRES_residual_history.txt")
        rm("solveSoftCFIEACA_GMRES_residual_history.png")
        # 3 octree levels, with compression of non-touching nodes
        num_levels = 3
        compression_distance = 1.5
        test_J = solveSoftCFIEACA(pulse_mesh, num_levels, excitation, excitation_normal_deriv, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.15e-7)
        rm("solveSoftCFIEACA_GMRES_residual_history.txt")
        rm("solveSoftCFIEACA_GMRES_residual_history.png")

        # Switch to rectangular strip mesh
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        softIE_weight = 0.5
        mesh_filename = "examples/test/rectangular_strips/rectangular_strip.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 1,-1
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        excitation_normal_deriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(1.0, wavenumber, [x_test,y_test,z_test], l, m, normal)
        sol_J = solveSoftCFIE(pulse_mesh, excitation, excitation_normal_deriv, wavenumber, distance_to_edge_tol, near_singular_tol, softIE_weight)
        # 1 octree level, no compression
        num_levels = 1
        compression_distance = 1e100
        ACA_approximation_tol = 1e-6
        test_J = solveSoftCFIEACA(pulse_mesh, num_levels, excitation, excitation_normal_deriv, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.8e-8)
        rm("solveSoftCFIEACA_GMRES_residual_history.txt")
        rm("solveSoftCFIEACA_GMRES_residual_history.png")
        # 3 levels, with compression, low approx tol
        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-2
        test_J = solveSoftCFIEACA(pulse_mesh, num_levels, excitation, excitation_normal_deriv, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.5e-3)
        rm("solveSoftCFIEACA_GMRES_residual_history.txt")
        rm("solveSoftCFIEACA_GMRES_residual_history.png")
        # 4 levels, with compression, high approx tol, tests when octree already is made
        num_levels = 4
        compression_distance = 1.5
        ACA_approximation_tol = 1e-6
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoftCFIE!(pulse_mesh, octree, wavenumber, softIE_weight,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        test_J = solveSoftCFIEACA(pulse_mesh, octree, num_levels, excitation, excitation_normal_deriv, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J, rtol=0.13e-7)
        rm("solveSoftCFIEACA_GMRES_residual_history.txt")
        rm("solveSoftCFIEACA_GMRES_residual_history.png")

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        softIE_weight = 0.5
        mesh_filename = "examples/test/spheres/sphere_1m_1266.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        excitation_normal_deriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(1.0, wavenumber, [x_test,y_test,z_test], l, m, normal)
        sol_J = solveSoftCFIE(pulse_mesh, excitation, excitation_normal_deriv, wavenumber, distance_to_edge_tol, near_singular_tol, softIE_weight)
        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-5
        test_J = solveSoftCFIEACA(pulse_mesh, num_levels, excitation, excitation_normal_deriv, wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_J, test_J[1], rtol=0.9e-6)
        rm("solveSoftCFIEACA_GMRES_residual_history.txt")
        rm("solveSoftCFIEACA_GMRES_residual_history.png")
    end # solveSoftCFIEACA tests
    @testset "subMatvecACA tests" begin
        sub_Z = zeros(5,5)
        sub_J = zeros(5)
        test_sub_V = subMatvecACA(sub_Z, sub_J)
        sol_sub_V = zeros(5)
        @test isapprox(test_sub_V, sol_sub_V, rtol=1e-14)

        sub_Z = randn(5,5)
        sub_J = randn(5)
        test_sub_V = subMatvecACA(sub_Z, sub_J)
        sol_sub_V = sub_Z * sub_J
        @test isapprox(test_sub_V, sol_sub_V, rtol=1e-14)

        sub_Z = randn(ComplexF64, 8, 8)
        sub_J = randn(ComplexF64, 8)
        test_sub_V = subMatvecACA(sub_Z, sub_J)
        sol_sub_V = sub_Z * sub_J
        @test isapprox(test_sub_V, sol_sub_V, rtol=1e-14)

        U = randn(5,2)
        V = randn(2,5)
        sub_Z = (U,V)
        sub_J = randn(5)
        test_sub_V = subMatvecACA(sub_Z, sub_J)
        sol_sub_V = U * (V * sub_J)
        @test isapprox(test_sub_V, sol_sub_V, rtol=1e-14)

        U = randn(ComplexF64, 8, 4)
        V = randn(ComplexF64, 4, 8)
        sub_Z = (U,V)
        sub_J = randn(ComplexF64, 8)
        test_sub_V = subMatvecACA(sub_Z, sub_J)
        sol_sub_V = U * (V * sub_J)
        @test isapprox(test_sub_V, sol_sub_V, rtol=1e-14)
    end # fullMatvecACA tests
end # fast_solve tests
