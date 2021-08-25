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

        test_Q = calculateWSMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test isapprox(sol_Q, test_Q, rtol=0.8e-5)
        @test isapprox(test_Q, adjoint(test_Q), rtol=0.2e-4) # Q should be Hermitian
        test_eigen_Q = eigen(test_Q)
        @test isapprox(num_harmonics, length(test_eigen_Q.values), rtol=1e-14)
        @test isapprox(sum(abs.(real.(test_eigen_Q.values))), sum(abs.(test_eigen_Q.values)), rtol=1e-10) # test eigenvalues are real
        real_values = real.(test_eigen_Q.values)
        @test isapprox(real_values, sort(real_values), rtol=1e-14) # test eigenvalues ordered least to greatest
        reconstructed_Q = test_eigen_Q.vectors * diagm(test_eigen_Q.values) * adjoint(test_eigen_Q.vectors)
        @test isapprox(test_Q, reconstructed_Q, rtol = 0.2e-2)

        ##### DONT RUN REGULARLY AS THIS IS VERY SLOW #####
        # max_l = 13
        # lambda = 1
        # wavenumber = 2*pi/lambda + 0.0im
        # num_harmonics = max_l^2 + 2*max_l + 1
        # src_quadrature_rule = gauss7rule
        # test_quadrature_rule = gauss7rule
        # distance_to_edge_tol = 1e-12
        # near_singular_tol = 1.0
        # mesh_filename = "examples/test/circular_plate_fine.msh"
        # pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        #
        # test_Q = calculateWSMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        # @test isapprox(test_Q, adjoint(test_Q), rtol=0.45e-1) # Q should be Hermitian
        # test_eigen_Q = eigen(test_Q)
        # @test isapprox(num_harmonics, length(test_eigen_Q.values), rtol=1e-14)
        # @test isapprox(sum(abs.(real.(test_eigen_Q.values))), sum(abs.(test_eigen_Q.values)), rtol=0.13e-3) # test eigenvalues are real
        # real_values = real.(test_eigen_Q.values)
        # @test isapprox(real_values, sort(real_values), rtol=1e-14) # test eigenvalues ordered least to greatest
        # reconstructed_Q = test_eigen_Q.vectors * diagm(test_eigen_Q.values) * adjoint(test_eigen_Q.vectors)
        # @test isapprox(test_Q, reconstructed_Q, rtol = 0.12e-1)
        #
        # max_l = 17
        # lambda = 2
        # wavenumber = 2*pi/lambda + 0.0im
        # num_harmonics = max_l^2 + 2*max_l + 1
        # src_quadrature_rule = gauss7rule
        # test_quadrature_rule = gauss7rule
        # distance_to_edge_tol = 1e-12
        # near_singular_tol = 1.0
        # mesh_filename = "examples/test/rectangular_strips/rectangular_strip_fine.msh"
        # pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        #
        # test_Q = calculateWSMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        # @test isapprox(test_Q, adjoint(test_Q), rtol=0.5e-6) # Q should be Hermitian
        # test_eigen_Q = eigen(test_Q)
        # @test isapprox(num_harmonics, length(test_eigen_Q.values), rtol=1e-14)
        # @test isapprox(sum(abs.(real.(test_eigen_Q.values))), sum(abs.(test_eigen_Q.values)), rtol=1e-14) # test eigenvalues are real
        # real_values = real.(test_eigen_Q.values)
        # @test isapprox(real_values, sort(real_values), rtol=1e-14) # test eigenvalues ordered least to greatest
        # reconstructed_Q = test_eigen_Q.vectors * diagm(test_eigen_Q.values) * adjoint(test_eigen_Q.vectors)
        # @test isapprox(test_Q, reconstructed_Q, rtol = 0.5e-5)
        ##########

        max_l = 4
        lambda = 10.0
        wavenumber = 2*pi/lambda + 0.0im
        num_harmonics = 25
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/spheres/sphere_1m_1266.msh"
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
        Q_sca_inc = -1/2/wavenumber * adjoint(Js)*transpose(dVsdk_trans)
        Q_inc_sca = adjoint(Q_sca_inc)
        Q_sca_sca_i = 1/2/wavenumber^2 * adjoint(Js)*real.(Z + wavenumber .* dZdk)*Js
        Q_sca_sca_d = im/8/wavenumber^2 * adjoint(Js)*(conj.(transpose(Vs_trans))*dVsdk_trans - conj.(transpose(dVsdk_trans))*Vs_trans)*Js
        sol_Q = Q_sca_inc+Q_inc_sca+Q_sca_sca_i+Q_sca_sca_d

        S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)#, Z_factors)
        dSdk = calculateScatteringMatrixDerivative(max_l, num_harmonics, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)#, Z_factors)
        test_Q = calculateWSMatrix(S, dSdk)
        @test isapprox(sol_Q, test_Q, rtol=0.13e-7)
    end # calculateWSMatrix tests
    @testset "calculateWSMatrixACA tests" begin
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
        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-5
        octree = createOctree(num_levels, pulse_mesh)
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

        test_Q = calculateWSMatrixACA(max_l, wavenumber, pulse_mesh, octree, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_Q, test_Q, rtol=0.8e-5)
        @test isapprox(test_Q, adjoint(test_Q), rtol=0.2e-4) # Q should be Hermitian
        test_eigen_Q = eigen(test_Q)
        @test isapprox(num_harmonics, length(test_eigen_Q.values), rtol=1e-14)
        @test isapprox(sum(abs.(real.(test_eigen_Q.values))), sum(abs.(test_eigen_Q.values)), rtol=1e-10) # test eigenvalues are real
        real_values = real.(test_eigen_Q.values)
        @test isapprox(real_values, sort(real_values), rtol=1e-14) # test eigenvalues ordered least to greatest
        reconstructed_Q = test_eigen_Q.vectors * diagm(test_eigen_Q.values) * adjoint(test_eigen_Q.vectors)
        @test isapprox(test_Q, reconstructed_Q, rtol = 0.3e-2)

        ##### DONT RUN REGULARLY AS THIS IS VERY SLOW #####
        max_l = 13
        lambda = 1
        wavenumber = 2*pi/lambda + 0.0im
        num_harmonics = max_l^2 + 2*max_l + 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/circular_plate_fine.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

        test_Q = calculateWSMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test isapprox(test_Q, adjoint(test_Q), rtol=0.45e-1) # Q should be Hermitian
        test_eigen_Q = eigen(test_Q)
        @test isapprox(num_harmonics, length(test_eigen_Q.values), rtol=1e-14)
        @test isapprox(sum(abs.(real.(test_eigen_Q.values))), sum(abs.(test_eigen_Q.values)), rtol=0.13e-3) # test eigenvalues are real
        real_values = real.(test_eigen_Q.values)
        @test isapprox(real_values, sort(real_values), rtol=1e-14) # test eigenvalues ordered least to greatest
        reconstructed_Q = test_eigen_Q.vectors * diagm(test_eigen_Q.values) * adjoint(test_eigen_Q.vectors)
        @test isapprox(test_Q, reconstructed_Q, rtol = 0.12e-1)

        max_l = 17
        lambda = 2
        wavenumber = 2*pi/lambda + 0.0im
        num_harmonics = max_l^2 + 2*max_l + 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangular_strips/rectangular_strip_fine.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

        test_Q = calculateWSMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test isapprox(test_Q, adjoint(test_Q), rtol=0.5e-6) # Q should be Hermitian
        test_eigen_Q = eigen(test_Q)
        @test isapprox(num_harmonics, length(test_eigen_Q.values), rtol=1e-14)
        @test isapprox(sum(abs.(real.(test_eigen_Q.values))), sum(abs.(test_eigen_Q.values)), rtol=1e-14) # test eigenvalues are real
        real_values = real.(test_eigen_Q.values)
        @test isapprox(real_values, sort(real_values), rtol=1e-14) # test eigenvalues ordered least to greatest
        reconstructed_Q = test_eigen_Q.vectors * diagm(test_eigen_Q.values) * adjoint(test_eigen_Q.vectors)
        @test isapprox(test_Q, reconstructed_Q, rtol = 0.5e-5)
        ##########

        max_l = 4
        lambda = 10.0
        wavenumber = 2*pi/lambda + 0.0im
        num_harmonics = 25
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/spheres/sphere_1m_1266.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-5
        octree = createOctree(num_levels, pulse_mesh)
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
        Q_sca_inc = -1/2/wavenumber * adjoint(Js)*transpose(dVsdk_trans)
        Q_inc_sca = adjoint(Q_sca_inc)
        Q_sca_sca_i = 1/2/wavenumber^2 * adjoint(Js)*real.(Z + wavenumber .* dZdk)*Js
        Q_sca_sca_d = im/8/wavenumber^2 * adjoint(Js)*(conj.(transpose(Vs_trans))*dVsdk_trans - conj.(transpose(dVsdk_trans))*Vs_trans)*Js
        sol_Q = Q_sca_inc+Q_inc_sca+Q_sca_sca_i+Q_sca_sca_d

        test_Q = calculateWSMatrixACA(max_l, wavenumber, pulse_mesh, octree, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
        @test isapprox(sol_Q, test_Q, rtol=0.27e-6)
    end # calculateWSMatrixACA tests
    @testset "solveWSMode tests" begin
        # this isnt a great test as the solution is copied from the output
        # therefore it, at best, only tells me if I changed something in the code
        max_l = 2
        mode_idx = 1
        lambda = 10
        wavenumber = 2*pi/lambda + 0.0im
        num_harmonics = max_l^2 + 2*max_l + 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/circular_plate_1m.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        test_sources = solveWSMode(max_l, mode_idx, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
        @test isapprox(test_sources[1], -0.5313576866296218 - 0.21990746085878485im, rtol=1e-14)
        @test isapprox(test_sources[468], -0.6792050632304026 - 0.28109552129052373im, rtol=1e-14)
    end # calculateWSMatrix tests
    @testset "solveWSModeACA tests" begin
        # this isnt a great test as the solution is copied from the output of non-ACA solve
        # therefore it, at best, only tells me if I changed something in the code
        max_l = 2
        mode_idx = 1
        lambda = 10
        wavenumber = 2*pi/lambda + 0.0im
        num_harmonics = max_l^2 + 2*max_l + 1
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        num_levels = 3
        compression_distance = 1.5
        ACA_approximation_tol = 1e-5
        mesh_filename = "examples/test/circular_plate_1m.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        test_sources = solveWSModeACA(max_l, mode_idx, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol, num_levels, compression_distance, ACA_approximation_tol)
        @test isapprox(test_sources[1][1], -0.5313576866296218 - 0.21990746085878485im, rtol=1e-5)
        @test isapprox(test_sources[1][468], -0.6792050632304026 - 0.28109552129052373im, rtol=0.8e-6)
    end # calculateWSMatrix tests
    @testset "sphericalWaveWSMode tests" begin
        wavenumber = 2.0
        max_l = 0
        mode_vector = [1]
        x, y, z = 1.0, 1.0, 1.0
        sol_wave = sphericalWave(2*wavenumber, wavenumber, [x,y,z], 0, 0)
        test_wave = sphericalWaveWSMode(x, y, z, max_l, wavenumber, mode_vector)
        @test isapprox(sol_wave, test_wave, rtol=1e-14)

        wavenumber = 2.0
        max_l = 1
        mode_vector = [1, 0, 0, 0.5+0.2im]
        x, y, z = 1.0, 1.0, 1.0
        sol_wave = 0
        sol_wave += 1 * sphericalWave(2*wavenumber, wavenumber, [x,y,z], 0, 0)
        sol_wave += (0.5+0.2im) * sphericalWave(2*wavenumber, wavenumber, [x,y,z], 1, 1)
        test_wave = sphericalWaveWSMode(x, y, z, max_l, wavenumber, mode_vector)
        @test isapprox(sol_wave, test_wave, rtol=1e-14)
    end # sphericalWaveWSMode tests
end
