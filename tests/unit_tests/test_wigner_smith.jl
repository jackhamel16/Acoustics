using Test

include("../../src/includes.jl")

@testset "wigner_smith tests" begin
    @testset "calculateWSMatrix tests" begin
        max_l = 10
        lambda = 1
        wavenumber = 2*pi/lambda + 0.0im
        num_harmonics = max_l^2 + 2*max_l + 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/circular_plate_1m.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        @unpack num_elements = pulse_mesh
        Z = calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol)
        Z_factors = lu(Z)
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
        Q_sca_inc = -1/2/wavenumber * adjoint(Js)*transpose(dVsdk_trans)
        Q_inc_sca = adjoint(Q_sca_inc)
        Q_sca_sca_i = 1/2/wavenumber^2 * adjoint(Js)*real.(Z + wavenumber .* dZdk)*Js
        Q_sca_sca_d = im/8/wavenumber^2 * adjoint(Js)*(conj.(transpose(Vs_trans))*dVsdk_trans - conj.(transpose(dVsdk_trans))*Vs_trans)*Js
        sol_Q = Q_sca_inc+Q_inc_sca+Q_sca_sca_i+Q_sca_sca_d

        S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        dSdk = calculateScatteringMatrixDerivative(max_l, num_harmonics, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        test_Q = calculateWSMatrix(S, dSdk)
        @test isapprox(sol_Q, test_Q, rtol=0.8e-5)

        max_l = 4
        lambda = 10.0
        wavenumber = 2*pi/lambda + 0.0im
        num_harmonics = 25
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/sphere_1m_1266.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        @unpack num_elements = pulse_mesh
        Z = calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol)
        Z_factors = lu(Z)
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
        Q_sca_inc = -1/2/wavenumber * adjoint(Js)*transpose(dVsdk_trans)
        Q_inc_sca = adjoint(Q_sca_inc)
        Q_sca_sca_i = 1/2/wavenumber^2 * adjoint(Js)*real.(Z + wavenumber .* dZdk)*Js
        Q_sca_sca_d = im/8/wavenumber^2 * adjoint(Js)*(conj.(transpose(Vs_trans))*dVsdk_trans - conj.(transpose(dVsdk_trans))*Vs_trans)*Js
        sol_Q = Q_sca_inc+Q_inc_sca+Q_sca_sca_i+Q_sca_sca_d

        S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        dSdk = calculateScatteringMatrixDerivative(max_l, num_harmonics, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        test_Q = calculateWSMatrix(S, dSdk)
        @test isapprox(sol_Q, test_Q, rtol=0.13e-7)
    end # calculateWSMatrix tests
end
