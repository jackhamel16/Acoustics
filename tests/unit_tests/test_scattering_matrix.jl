using Test

include("../../src/includes.jl")

@testset "scattering_matrix tests" begin
    @testset "calculateScatteringMatrix tests" begin
        max_l = 1
        lambda=20.0
        wavenumber = 2*pi/lambda + 0*im
        a = 1.0 # sphere radius
        excitation_amplitude = 1.0
        num_harmonics = 4
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/spheres/sphere_1m_1266.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        V_solution = zeros(ComplexF64, num_harmonics, pulse_mesh.num_elements)
        J_solution = zeros(ComplexF64, pulse_mesh.num_elements, num_harmonics)
        l, m = 0, 0
        solution = zeros(ComplexF64, num_harmonics, num_harmonics)
        for tl=0:max_l
            pl = tl
            for tm=-tl:tl
                pm = -tm
                t_idx = tl^2 + tl + tm + 1
                p_idx = pl^2 + pl + pm + 1
                solution[t_idx,p_idx] = (-1)^tm * (1 - 2 * sphericalBesselj(tl, real(wavenumber) * a) / sphericalHankel2(tl, real(wavenumber) * a))
            end
        end
        no_coupling_matrix = [1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0] # mag of zero in spots corresponding to coupling between different harmonics
        pulse_mesh.Z_factors = lu(calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol))
        test_S, test_Js = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test Array{ComplexF64, 2} == typeof(test_S)
        @test size(test_S) == (num_harmonics, num_harmonics)
        @test size(test_Js) == (pulse_mesh.num_elements, num_harmonics)
        @test isapprox(solution, test_S, rtol=0.95e-3)
        @test isapprox(test_S, transpose(test_S), rtol=0.13e-8)
        @test isapprox(1, abs(det(test_S) * det(adjoint(test_S))), rtol=0.2e-11)
        @test isapprox(abs.(test_S), no_coupling_matrix, rtol = 0.5e-5)

        max_l = 8
        lambda=10.0
        wavenumber = 2*pi/lambda + 0*im
        excitation_amplitude = 1.0
        num_harmonics = max_l^2 + 2*max_l + 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/circular_plate_1m.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        pulse_mesh.Z_factors = lu(calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol))
        test_S, test_Js = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test size(test_S) == (num_harmonics, num_harmonics)
        @test size(test_Js) == (pulse_mesh.num_elements, num_harmonics)
        @test isapprox(test_S, transpose(test_S), rtol=0.15e-7)
        @test isapprox(1, abs(det(test_S) * det(adjoint(test_S))), rtol=1e-15)

        # Test using a flat plate and S should still be unitary
        # takes a few minutes to run
        # max_l = 19
        # lambda = 5 # two wavelengths across plate
        # wavenumber = 2*pi/lambda + 0*im
        # num_harmonics = 400
        # src_quadrature_rule = gauss7rule
        # test_quadrature_rule = gauss1rule
        # distance_to_edge_tol = 1e-12
        # near_singular_tol = 1.0
        # mesh_filename = "examples/test/rectangle_plate_10m.msh"
        # pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        # pulse_mesh.Z_factors = lu(calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol))
        # test_S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        # @test size(test_S) == (num_harmonics, num_harmonics)
        # @test isapprox(test_S, transpose(test_S), rtol=1e-4)
        # @test isapprox(1, abs(det(test_S) * det(adjoint(test_S))), rtol=0.8e-1)
    end # calculateScatteringMatrix tests

    @testset "calculateScatteringMatrixACA tests" begin
        max_l = 1
        lambda=20.0
        wavenumber = 2*pi/lambda + 0*im
        a = 1.0 # sphere radius
        excitation_amplitude = 1.0
        num_harmonics = 4
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/spheres/sphere_1m_1266.msh"
        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-5
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        V_solution = zeros(ComplexF64, num_harmonics, pulse_mesh.num_elements)
        J_solution = zeros(ComplexF64, pulse_mesh.num_elements, num_harmonics)
        l, m = 0, 0
        solution = zeros(ComplexF64, num_harmonics, num_harmonics)
        for tl=0:max_l
            pl = tl
            for tm=-tl:tl
                pm = -tm
                t_idx = tl^2 + tl + tm + 1
                p_idx = pl^2 + pl + pm + 1
                solution[t_idx,p_idx] = (-1)^tm * (1 - 2 * sphericalBesselj(tl, real(wavenumber) * a) / sphericalHankel2(tl, real(wavenumber) * a))
            end
        end
        no_coupling_matrix = [1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0] # mag of zero in spots corresponding to coupling between different harmonics
        # test_S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        # pulse_mesh.Z_factors = lu(calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol))
        test_S, test_Js = calculateScatteringMatrixACA(max_l, wavenumber, pulse_mesh, octree, distance_to_edge_tol, near_singular_tol)
        @test Array{ComplexF64, 2} == typeof(test_S)
        @test size(test_S) == (num_harmonics, num_harmonics)
        @test size(test_Js) == (pulse_mesh.num_elements, num_harmonics)
        @test isapprox(solution, test_S, rtol=0.95e-3)
        @test isapprox(test_S, transpose(test_S), rtol=0.12e-7)
        @test isapprox(1, abs(det(test_S) * det(adjoint(test_S))), rtol=0.34e-7)
        @test isapprox(abs.(test_S), no_coupling_matrix, rtol = 0.43e-5)

        max_l = 8
        lambda=10.0
        wavenumber = 2*pi/lambda + 0*im
        excitation_amplitude = 1.0
        num_harmonics = max_l^2 + 2*max_l + 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/circular_plate_1m.msh"
        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-5
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        test_S, test_Js = calculateScatteringMatrixACA(max_l, wavenumber, pulse_mesh, octree, distance_to_edge_tol, near_singular_tol)
        @test size(test_S) == (num_harmonics, num_harmonics)
        @test isapprox(test_S, transpose(test_S), rtol=0.16e-7)
        @test isapprox(1, abs(det(test_S) * det(adjoint(test_S))), rtol=0.53e-7)

        # Test using a flat plate and S should still be unitary
        # takes a few minutes to run
        # max_l = 19
        # lambda = 5 # two wavelengths across plate
        # wavenumber = 2*pi/lambda + 0*im
        # num_harmonics = 400
        # src_quadrature_rule = gauss7rule
        # test_quadrature_rule = gauss7rule
        # distance_to_edge_tol = 1e-12
        # near_singular_tol = 1.0
        # mesh_filename = "examples/test/rectangle_plate_10m.msh"
        # num_levels = 3
        # compression_distance = 1.5
        # ACA_approximation_tol = 1e-5
        # pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        # octree = createOctree(num_levels, pulse_mesh)
        # fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
        #                               distance_to_edge_tol, near_singular_tol,
        #                               compression_distance, ACA_approximation_tol)
        # test_S, test_Js = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, octree, distance_to_edge_tol, near_singular_tol)
        # @test size(test_S) == (num_harmonics, num_harmonics)
        # @test isapprox(test_S, transpose(test_S), rtol=1e-4)
        # @test isapprox(1, abs(det(test_S) * det(adjoint(test_S))), rtol=0.8e-1)

    end # calculateScatteringMatrixACA tests

    @testset "calculateScatteringMatrixDerivative tests" begin
        max_l = 1
        wavenumber = 1.0 + 0.0im
        num_harmonics = 4
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        @unpack num_elements = pulse_mesh
        Z = calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol)
        Z_factors = lu(Z)
        pulse_mesh.Z_factors = Z_factors
        dZdk = calculateZKDerivMatrix(pulse_mesh, wavenumber)
        Vs_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
        Js = Array{ComplexF64}(undef, num_elements, num_harmonics)
        dVsdk_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
        harmonic_idx = 1
        for l = 0:max_l
            for m=-l:l
                Vs_trans[harmonic_idx,:] = calculateVlm(pulse_mesh, wavenumber, l, m)
                Js[:,harmonic_idx] = Z_factors \ Vs_trans[harmonic_idx,:]
                dVsdk_trans[harmonic_idx, :] = calculateVlmKDeriv(pulse_mesh, wavenumber, l, m)
                harmonic_idx += 1
            end
        end
        solution_dSdk = -im/(2*wavenumber^2)*Vs_trans*Js + im/(2*wavenumber)*dVsdk_trans*Js +
                        im/(2*wavenumber)*(Vs_trans/Z_factors)*(transpose(dVsdk_trans)-dZdk*Js)
        test_dSdk = calculateScatteringMatrixDerivative(max_l, num_harmonics, wavenumber, pulse_mesh, Js, distance_to_edge_tol, near_singular_tol)
        @test isapprox(solution_dSdk, test_dSdk, rtol=1e-5)

        max_l = 10
        wavenumber = 4.5 + 0.0im
        deltak = 0.001 * wavenumber
        k_high = wavenumber + deltak/2
        k_low = wavenumber - deltak/2
        num_harmonics = 121
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        Z_high = lu(calculateZMatrix(pulse_mesh, k_high, distance_to_edge_tol, near_singular_tol))
        Z_low = lu(calculateZMatrix(pulse_mesh, k_low, distance_to_edge_tol, near_singular_tol))
        pulse_mesh.Z_factors = Z_high
        S_high, test_Js = calculateScatteringMatrix(max_l, k_high, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        pulse_mesh.Z_factors = Z_low
        S_low, test_Js = calculateScatteringMatrix(max_l, k_low, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        solution_dSdk = (S_high - S_low) / deltak
        Z_factors = lu(calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol))
        pulse_mesh.Z_factors = Z_factors
        Js = Array{ComplexF64}(undef, num_elements, num_harmonics)
        harmonic_idx = 1
        for l = 0:max_l
            for m=-l:l
                Js[:,harmonic_idx] = Z_factors \ calculateVlm(pulse_mesh, wavenumber, l, m)
                harmonic_idx += 1
            end
        end
        test_dSdk = calculateScatteringMatrixDerivative(max_l, num_harmonics, wavenumber, pulse_mesh, Js, distance_to_edge_tol, near_singular_tol)
        @test size(test_dSdk) == (num_harmonics, num_harmonics)
        @test isapprox(solution_dSdk, test_dSdk, rtol=0.15e-3)

        max_l = 10
        lambda = 1.5
        wavenumber = 2*pi/lambda + 0.0im
        deltak = 0.001 * wavenumber
        k_high = wavenumber + deltak/2
        k_low = wavenumber - deltak/2
        num_harmonics = max_l^2 + 2*max_l + 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangular_strips/rectangular_strip.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        Z_high = lu(calculateZMatrix(pulse_mesh, k_high, distance_to_edge_tol, near_singular_tol))
        Z_low = lu(calculateZMatrix(pulse_mesh, k_low, distance_to_edge_tol, near_singular_tol))
        pulse_mesh.Z_factors = Z_high
        S_high, test_Js = calculateScatteringMatrix(max_l, k_high, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        pulse_mesh.Z_factors = Z_low
        S_low, test_Js = calculateScatteringMatrix(max_l, k_low, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        solution_dSdk = (S_high - S_low) / deltak
        Z_factors = lu(calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol))
        pulse_mesh.Z_factors = Z_factors
        Js = Array{ComplexF64}(undef, pulse_mesh.num_elements, num_harmonics)
        harmonic_idx = 1
        for l = 0:max_l
            for m=-l:l
                Js[:,harmonic_idx] = Z_factors \ calculateVlm(pulse_mesh, wavenumber, l, m)
                harmonic_idx += 1
            end
        end
        test_dSdk = calculateScatteringMatrixDerivative(max_l, num_harmonics, wavenumber, pulse_mesh, Js, distance_to_edge_tol, near_singular_tol)
        @test size(test_dSdk) == (num_harmonics, num_harmonics)
        @test isapprox(solution_dSdk, test_dSdk, rtol=0.4e-4)
    end # calculateScatteringMatrixDerivative tests

    @testset "calculateScatteringMatrixDerivativeACA tests" begin
        max_l = 1
        wavenumber = 1.0 + 0.0im
        num_harmonics = 4
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-5
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                         compression_distance, ACA_approximation_tol)
        @unpack num_elements = pulse_mesh
        Z = calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol)
        Z_factors = lu(Z)
        pulse_mesh.Z_factors = Z_factors
        dZdk = calculateZKDerivMatrix(pulse_mesh, wavenumber)
        Vs_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
        Js = Array{ComplexF64}(undef, num_elements, num_harmonics)
        dVsdk_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
        harmonic_idx = 1
        for l = 0:max_l
            for m=-l:l
                Vs_trans[harmonic_idx,:] = calculateVlm(pulse_mesh, wavenumber, l, m)
                Js[:,harmonic_idx] = Z_factors \ Vs_trans[harmonic_idx,:]
                dVsdk_trans[harmonic_idx, :] = calculateVlmKDeriv(pulse_mesh, wavenumber, l, m)
                harmonic_idx += 1
            end
        end
        solution_dSdk = -im/(2*wavenumber^2)*Vs_trans*Js + im/(2*wavenumber)*dVsdk_trans*Js +
                        im/(2*wavenumber)*(Vs_trans/Z_factors)*(transpose(dVsdk_trans)-dZdk*Js)
        test_dSdk = calculateScatteringMatrixDerivativeACA(max_l, num_harmonics, wavenumber, pulse_mesh, Js, octree)
        @test isapprox(solution_dSdk, test_dSdk, rtol=0.4e-5)

        max_l = 10
        wavenumber = 4.5 + 0.0im
        deltak = 0.001 * wavenumber
        k_high = wavenumber + deltak/2
        k_low = wavenumber - deltak/2
        num_harmonics = 121
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        num_levels = 3
        compression_distance = 100
        ACA_approximation_tol = 1e-5
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                         compression_distance, ACA_approximation_tol)
        Z_high = lu(calculateZMatrix(pulse_mesh, k_high, distance_to_edge_tol, near_singular_tol))
        Z_low = lu(calculateZMatrix(pulse_mesh, k_low, distance_to_edge_tol, near_singular_tol))
        pulse_mesh.Z_factors = Z_high
        S_high, test_Js = calculateScatteringMatrix(max_l, k_high, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        pulse_mesh.Z_factors = Z_low
        S_low, test_Js = calculateScatteringMatrix(max_l, k_low, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        solution_dSdk = (S_high - S_low) / deltak
        Z_factors = lu(calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol))
        Js = Array{ComplexF64}(undef, pulse_mesh.num_elements, num_harmonics)
        harmonic_idx = 1
        for l = 0:max_l
            for m=-l:l
                Js[:,harmonic_idx] = Z_factors \ calculateVlm(pulse_mesh, wavenumber, l, m)
                harmonic_idx += 1
            end
        end
        test_dSdk = calculateScatteringMatrixDerivativeACA(max_l, num_harmonics, wavenumber, pulse_mesh, Js, octree)
        @test size(test_dSdk) == (num_harmonics, num_harmonics)
        @test isapprox(solution_dSdk, test_dSdk, rtol=0.11e-3)

        max_l = 10
        lambda = 1.5
        wavenumber = 2*pi/lambda + 0.0im
        deltak = 0.001 * wavenumber
        k_high = wavenumber + deltak/2
        k_low = wavenumber - deltak/2
        num_harmonics = max_l^2 + 2*max_l + 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangular_strips/rectangular_strip.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                      distance_to_edge_tol, near_singular_tol,
                                      compression_distance, ACA_approximation_tol)
        fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                         compression_distance, ACA_approximation_tol)
        Z_high = lu(calculateZMatrix(pulse_mesh, k_high, distance_to_edge_tol, near_singular_tol))
        Z_low = lu(calculateZMatrix(pulse_mesh, k_low, distance_to_edge_tol, near_singular_tol))
        pulse_mesh.Z_factors = Z_high
        S_high, test_Js = calculateScatteringMatrix(max_l, k_high, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        pulse_mesh.Z_factors = Z_low
        S_low, test_Js = calculateScatteringMatrix(max_l, k_low, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        solution_dSdk = (S_high - S_low) / deltak
        Z_factors = lu(calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol))
        pulse_mesh.Z_factors = Z_factors
        Js = Array{ComplexF64}(undef, pulse_mesh.num_elements, num_harmonics)
        harmonic_idx = 1
        for l = 0:max_l
            for m=-l:l
                Js[:,harmonic_idx] = Z_factors \ calculateVlm(pulse_mesh, wavenumber, l, m)
                harmonic_idx += 1
            end
        end
        test_dSdk = calculateScatteringMatrixDerivativeACA(max_l, num_harmonics, wavenumber, pulse_mesh, Js, octree)
        @test size(test_dSdk) == (num_harmonics, num_harmonics)
        @test isapprox(solution_dSdk, test_dSdk, rtol=0.4e-4)
    end # calculateScatteringMatrixDerivativeACA tests

    @testset "calculateVlm tests" begin
        wavenumber = 0.5 + 0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        l, m = 1, 0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(2*wavenumber,
                                                                        wavenumber,
                                                                        [x_test,y_test,z_test],
                                                                        l, m)
        solution_Vlm = zeros(ComplexF64, pulse_mesh.num_elements)
        rhsFill!(pulse_mesh, sphericalWaveExcitation, solution_Vlm)
        test_Vlm = calculateVlm(pulse_mesh, wavenumber, l, m)
        @test isapprox(solution_Vlm, test_Vlm, rtol=1e-14)
    end # calculateVlm tests

    @testset "calculateVlmKDeriv tests" begin
        wavenumber = 2.0 + 0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        l, m = 1, 0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        sphericalWaveKDerivIntegrand(x_test, y_test, z_test) = sphericalWaveKDerivative(wavenumber,
                                                                        [x_test,y_test,z_test],
                                                                        l, m)
        solution_dVdk = zeros(ComplexF64, pulse_mesh.num_elements)
        rhsFill!(pulse_mesh, sphericalWaveKDerivIntegrand, solution_dVdk)
        test_dVdk = calculateVlmKDeriv(pulse_mesh, wavenumber, l, m)
        @test isapprox(solution_dVdk, test_dVdk, rtol=1e-14)

        wavenumber = 1.0 + 0.0im
        delta_k = real(wavenumber) * 0.01
        k_high = wavenumber + delta_k/2
        k_low = wavenumber - delta_k/2
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        l, m = 0, 0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        sphericalWaveLow(x_test, y_test, z_test) = sphericalWave(2*k_low,
                                                                  k_low,
                                                                        [x_test,y_test,z_test],
                                                                        l,
                                                                        m)
        V_low = zeros(ComplexF64, pulse_mesh.num_elements)
        rhsFill!(pulse_mesh, sphericalWaveLow, V_low)
        sphericalWaveHigh(x_test, y_test, z_test) = sphericalWave(2*k_high,
                                                                  k_high,
                                                                        [x_test,y_test,z_test],
                                                                        l,
                                                                        m)
        V_high = zeros(ComplexF64, pulse_mesh.num_elements)
        rhsFill!(pulse_mesh, sphericalWaveHigh, V_high)
        solution_dVdk = (V_high - V_low) ./ delta_k
        test_dVdk = calculateVlmKDeriv(pulse_mesh, wavenumber, l, m)
        @test isapprox(solution_dVdk, test_dVdk, rtol=1e-5)

        wavenumber = 2.3 + 0.0im
        delta_k = real(wavenumber) * 0.01
        k_high = wavenumber + delta_k/2
        k_low = wavenumber - delta_k/2
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        l, m = 4,-1
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        sphericalWaveLow(x_test, y_test, z_test) = sphericalWave(2*k_low,
                                                                  k_low,
                                                                        [x_test,y_test,z_test],
                                                                        l,
                                                                        m)
        V_low = zeros(ComplexF64, pulse_mesh.num_elements)
        rhsFill!(pulse_mesh, sphericalWaveLow, V_low)
        sphericalWaveHigh(x_test, y_test, z_test) = sphericalWave(2*k_high,
                                                                  k_high,
                                                                        [x_test,y_test,z_test],
                                                                        l,
                                                                        m)
        V_high = zeros(ComplexF64, pulse_mesh.num_elements)
        rhsFill!(pulse_mesh, sphericalWaveHigh, V_high)
        solution_dVdk = (V_high - V_low) ./ delta_k
        test_dVdk = calculateVlmKDeriv(pulse_mesh, wavenumber, l, m)
        @test isapprox(solution_dVdk, test_dVdk, rtol=1e-4)
    end # calculateVlmKDeriv tests

    @testset "calculateZKDerivMatrix tests" begin
        wavenumber = 1.0 + 0.0im
        delta_k = real(wavenumber) * 0.01
        k_high = wavenumber + delta_k/2
        k_low = wavenumber - delta_k/2
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrandLow(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       k_low,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        Z_low = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrandLow, Z_low)
        testIntegrandHigh(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       k_high,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        Z_high = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrandHigh, Z_high)
        solution_dZdk = (Z_high - Z_low) ./ delta_k
        test_dZdk = calculateZKDerivMatrix(pulse_mesh, wavenumber)
        @test isapprox(solution_dZdk, test_dZdk, rtol=1e-4)

        wavelength = 1.25
        wavenumber = 2*pi/wavelength + 0.0im
        delta_k = real(wavenumber) * 0.01
        k_high = wavenumber + delta_k/2
        k_low = wavenumber - delta_k/2
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/circular_plate_1m.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrandLow(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       k_low,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        Z_low = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrandLow, Z_low)
        testIntegrandHigh(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       k_high,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        Z_high = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrandHigh, Z_high)
        solution_dZdk = (Z_high - Z_low) ./ delta_k
        test_dZdk = calculateZKDerivMatrix(pulse_mesh, wavenumber)
        @test isapprox(solution_dZdk, test_dZdk, rtol=0.15e-3)
    end # calculateZKDerivMatrix tests

    @testset "calculateZMatrix tests" begin
        wavenumber = 0.5 + 0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        solution_Z = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, solution_Z)
        test_Z = calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol)
        @test isapprox(solution_Z, test_Z, rtol=1e-14)
    end # calculateZMatrix tests
end # scattering_matrix tests
