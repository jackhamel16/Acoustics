using Test

include("../../src/includes.jl")

@testset "scattering_matrix tests" begin
    @testset "calculateScatteringMatrix tests" begin
        # Test exact solution for l=0
        max_l = 0
        lambda=20.0
        wavenumber = 2*pi/lambda + 0*im
        num_harmonics = 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/sphere_1m_1266.msh"
        a = 1.0 # sphere radius
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        test_S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        solution_S = 1 - 2 * sphericalBesselj(0, real(wavenumber) * a) / sphericalHankel2(0, real(wavenumber) * a)
        @test Array{ComplexF64, 2} == typeof(test_S)
        @test size(test_S) == (num_harmonics, num_harmonics)
        @test isapprox(abs(test_S[1,1]), 1.0, rtol=1e-5)
        @test isapprox(test_S[1,1], solution_S, rtol=0.25e-2)

        max_l = 1
        lambda=20.0
        wavenumber = 2*pi/lambda + 0*im
        a = 1.0 # sphere radius
        excitation_amplitude = 1.0
        num_harmonics = 4
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/sphere_1m_1266.msh"
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
        test_S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test Array{ComplexF64, 2} == typeof(test_S)
        @test size(test_S) == (num_harmonics, num_harmonics)
        @test isapprox(solution, test_S, rtol=1e-2)
        @test isapprox(test_S, transpose(test_S), rtol=1e-7)
        @test isapprox(1, abs(det(test_S) * det(adjoint(test_S))), rtol=0.5e-4)
        @test isapprox(abs.(test_S), no_coupling_matrix, rtol = 1e-5)

        max_l = 8
        lambda=10.0
        wavenumber = 2*pi/lambda + 0*im
        excitation_amplitude = 1.0
        num_harmonics = 81
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/sphere_1m_1266.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

        test_S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test size(test_S) == (num_harmonics, num_harmonics)
        @test isapprox(test_S, transpose(test_S), rtol=1e-5)
        @test isapprox(1, abs(det(test_S) * det(adjoint(test_S))), rtol=1e-3)

        # Test using a flat plate and S should still be unitary
        # takes a few minutes to run
        max_l = 19
        lambda = 5 # two wavelengths across plate
        wavenumber = 2*pi/lambda + 0*im
        num_harmonics = 400
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_10m.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

        test_S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test size(test_S) == (num_harmonics, num_harmonics)
        @test isapprox(test_S, transpose(test_S), rtol=1e-4)
        @test isapprox(1, abs(det(test_S) * det(adjoint(test_S))), rtol=0.8e-1)

    end # calculateScatteringMatrix tests
    @testset "calculateScatteringMatrixDerivative tests" begin
        max_l = 1
        wavenumber = 1.0 + 0.0im
        num_harmonics = 4
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        solution_dSdk = -im/(2*wavenumber)*calculateVJMatrix(max_l, num_harmonics, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        test_dSdk = calculateScatteringMatrixDerivative(max_l, num_harmonics, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test isapprox(solution_dSdk, test_dSdk, rtol=1e-14)
    end # calculateScatteringMatrixDerivative tests
    @testset "calculateVJMatrix tests" begin
        max_l = 1
        wavenumber = 1.0 + 0.0im
        excitation_amplitude = 1.0
        num_harmonics = 4
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_elements = 8
        V_solution = zeros(ComplexF64, num_harmonics, num_elements)
        J_solution = zeros(ComplexF64, num_elements, num_harmonics)
        l, m = 0, 0
        for harmonic_idx=1:num_harmonics
            sphericalWaveExcitation(x_test, y_test, z_test) = 2*wavenumber*
                                                              sphericalWave(excitation_amplitude,
                                                                            real(wavenumber),
                                                                            [x_test,y_test,z_test],
                                                                            l,
                                                                            m)
            sources, z, rhs = solveSoftIE(pulse_mesh,
                                          sphericalWaveExcitation,
                                          wavenumber,
                                          distance_to_edge_tol,
                                          near_singular_tol,
                                          true)
            V_solution[harmonic_idx, :] = rhs
            J_solution[:, harmonic_idx] = sources
            if m < l
                m += 1
            else
                l+= 1
                m = -l
            end
        end
        test_VJ = calculateVJMatrix(max_l, num_harmonics, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        solution_VJ = V_solution * J_solution
        @test size(test_VJ) == (num_harmonics, num_harmonics)
        @test isapprox(solution_VJ, test_VJ, rtol=1e-14)
    end # calculateVJMatrix tests
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
        matrixFill(pulse_mesh, testIntegrandLow, Z_low)
        testIntegrandHigh(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       k_high,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        Z_high = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill(pulse_mesh, testIntegrandHigh, Z_high)
        solution_dZdk = (Z_high - Z_low) ./ delta_k
        test_dZdk = calculateZKDerivMatrix(pulse_mesh, wavenumber)
        @test isapprox(solution_dZdk, test_dZdk, rtol=1e-4)
    end # calculateScatteringMatrixDerivative tests
end # scattering_matrix tests
