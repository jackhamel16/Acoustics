using Test
using GSL

include("../../src/includes.jl")

@testset "solve tests" begin
    @testset "solveSoftCFIE tests" begin
        # All these test do is make sure CFIE is softIE with softIE_weight is 1
        # and is softIENormalDeriv when softIE_weight is zeros
        excitation_amplitude = 1.0
        lambda=2.5
        wavenumber = 2*pi/lambda + 0*im
        wavevector = [wavenumber, 0.0, wavenumber] ./ sqrt(2)
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0

        mesh_filename = "examples/test/spheres/sphere_1m_1266.msh"
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        planeWaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
        planeWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = planeWaveNormalDerivative(excitation_amplitude, wavevector, [x_test,y_test,z_test], normal)

        sourcesIE = solveSoftIE(pulse_mesh,
                        planeWaveExcitation,
                        wavenumber,
                        distance_to_edge_tol,
                        near_singular_tol)
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        sourcesIEnd = solveSoftIENormalDeriv(pulse_mesh,
                        planeWaveExcitationNormalDeriv,
                        wavenumber)
        soft_IE_only = 1.0
        soft_IE_nd_only = 0.0
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        sourcesCFIE_soft_IE_only = solveSoftCFIE(pulse_mesh,
                        planeWaveExcitation,
                        planeWaveExcitationNormalDeriv,
                        wavenumber,
                        distance_to_edge_tol,
                        near_singular_tol,
                        soft_IE_only)
        pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        sourcesCFIE_soft_IE_nd_only = solveSoftCFIE(pulse_mesh,
                        planeWaveExcitation,
                        planeWaveExcitationNormalDeriv,
                        wavenumber,
                        distance_to_edge_tol,
                        near_singular_tol,
                        soft_IE_nd_only)
        @test isapprox(sourcesCFIE_soft_IE_only, sourcesIE, rtol=1e-12)
        @test isapprox(sourcesCFIE_soft_IE_only, sourcesIEnd, rtol=1e-1)
        @test isapprox(sourcesCFIE_soft_IE_nd_only, sourcesIEnd, rtol=1e-12)
        @test isapprox(sourcesCFIE_soft_IE_nd_only, sourcesIE, rtol=1e-1)
    end #solveSoftCFIE tests
end
