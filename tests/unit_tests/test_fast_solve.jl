using Test

include("../../src/math.jl")
include("../../src/excitation.jl")
include("../../src/mesh.jl")
include("../../src/quadrature.jl")
include("../../src/octree.jl")
include("../../src/fill.jl")
include("../../src/greens_functions.jl")
include("../../src/solve.jl")
include("../../src/ACA.jl")

include("../../src/fast_solve.jl")

@testset "fast_solve tests" begin
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
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        rand_J = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V = z_matrix * rand_J
        test_V = fullMatvecACA(pulse_mesh, octree, rand_J)
        @test isapprox(test_V, sol_V, rtol=1e-14)

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
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        rand_J = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V = z_matrix * rand_J
        test_V = fullMatvecACA(pulse_mesh, octree, rand_J)
        @test isapprox(test_V, sol_V, rtol=1e-12)

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1.5
        ACA_tol = 1e-4
        mesh_filename = "examples/test/rectangular_strip.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        rand_J = randn(ComplexF64, pulse_mesh.num_elements)
        sol_V = z_matrix * rand_J
        test_V = fullMatvecACA(pulse_mesh, octree, rand_J)
        @test isapprox(test_V, sol_V, rtol=0.5e-5)
    end # fullMatvecACA tests
    @testset "solveSoundSoftIEACA tests" begin
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
        # octree = createOctree(num_levels, pulse_mesh)
        test_J = solveSoundSoftIEACA(pulse_mesh, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        sol_J = solveSoftIE(pulse_mesh, excitation, wavenumber, distance_to_edge_tol, near_singular_tol)
        @test isapprox(sol_J, test_J, rtol=0.13e-6)

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1.5
        ACA_approximation_tol = 1e-4
        mesh_filename = "examples/test/disjoint_triangles.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        num_levels = 3
        # octree = createOctree(num_levels, pulse_mesh)
        test_J = solveSoundSoftIEACA(pulse_mesh, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        # sol_J = solveSoftIE(pulse_mesh, excitation, wavenumber, distance_to_edge_tol, near_singular_tol)
        @test isapprox(sol_J, test_J, rtol=0.15e-6)

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1
        ACA_approximation_tol = 1e-2
        mesh_filename = "examples/test/rectangular_strip.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        num_levels = 3
        # octree = createOctree(num_levels, pulse_mesh)
        test_J = solveSoundSoftIEACA(pulse_mesh, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        sol_J = solveSoftIE(pulse_mesh, excitation, wavenumber, distance_to_edge_tol, near_singular_tol)
        @test isapprox(sol_J, test_J, rtol=0.51e-2)

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1.5
        ACA_approximation_tol = 1e-6
        mesh_filename = "examples/test/rectangular_strip.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        num_levels = 3
        # octree = createOctree(num_levels, pulse_mesh)
        test_J = solveSoundSoftIEACA(pulse_mesh, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        # sol_J = solveSoftIE(pulse_mesh, excitation, wavenumber, distance_to_edge_tol, near_singular_tol)
        @test isapprox(sol_J, test_J, rtol=0.34e-6)

        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        compression_distance = 1.5
        ACA_approximation_tol = 1e-10
        mesh_filename = "examples/test/sphere_1m_1266.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        l, m = 0, 0
        excitation(x_test, y_test, z_test) = sphericalWave(1.0, real(wavenumber), [x_test,y_test,z_test], l, m)
        num_levels = 3
        # octree = createOctree(num_levels, pulse_mesh)
        test_J = solveSoundSoftIEACA(pulse_mesh, num_levels, excitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        sol_J = solveSoftIE(pulse_mesh, excitation, wavenumber, distance_to_edge_tol, near_singular_tol)
        @test isapprox(sol_J, test_J, rtol=0.21e-5)

    end # solveSoundSoftIEACA tests
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
