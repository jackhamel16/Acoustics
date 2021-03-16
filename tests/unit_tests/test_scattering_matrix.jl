using Test

include("../../src/includes.jl")

@testset "scattering_matrix tests" begin
    @testset "calculateScatteringMatrix tests" begin
        # max_l = 1
        # wavenumber = 1;
        # num_harmonics = 4 #max_l^2 + 2*max_l + 1
        # solution_S = [1 0 0 0; 0 0 0 -1; 0 0 1 0; 0 -1 0 0] + im*5*ones(4,4)
        max_l = 1
        lambda=50.0
        wavenumber = 2*pi/lambda + 0*im
        excitation_amplitude = 1.0
        num_harmonics = 4
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        # mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        mesh_filename = "examples/test/sphere_1m_1266.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        V_solution = zeros(ComplexF64, num_harmonics, pulse_mesh.num_elements)
        J_solution = zeros(ComplexF64, pulse_mesh.num_elements, num_harmonics)
        l, m = 0, 0
        for harmonic_idx=1:num_harmonics
            sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m)
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
        solution_S = [1 0 0 0; 0 0 0 -1; 0 0 1 0; 0 -1 0 0] + im/(2*wavenumber)*V_solution*J_solution
        test_S = calculateScatteringMatrix(max_l, wavenumber, excitation_amplitude, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test Array{ComplexF64, 2} == typeof(test_S)
        @test (num_harmonics, num_harmonics) == size(test_S)
        @test isapprox(solution_S, test_S, rtol=1e-14)
        @test isapprox(test_S, adjoint(test_S), rtol=1e-4)
    end # calculateScatteringMatrix tests
    @testset "calculateVMatrix tests" begin
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
            sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m)
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
        @test isapprox(V_solution * J_solution, calculateVJMatrix(max_l, wavenumber, excitation_amplitude, pulse_mesh, distance_to_edge_tol, near_singular_tol), rtol=1e-14)
    end # calculateScatteringMatrix tests
end # scattering_matrix tests
