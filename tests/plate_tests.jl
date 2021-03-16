using Test

include("../src/includes.jl")

@testset "Tests with a flat rectangular plate" begin
    @testset "solveSoftIE Symmetry tests" begin
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        excitation_amplitude = 1.0
        wavenumber = 2*pi/20+0*im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0

        wavevector1 = [0.0, 0.0, wavenumber]
        planewaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector1, [x_test,y_test,z_test])
        sources = solveSoftIE(mesh_filename,
                        planewaveExcitation,
                        wavenumber,
                        src_quadrature_rule,
                        test_quadrature_rule,
                        distance_to_edge_tol,
                        near_singular_tol)
        # Test the 4 corner elements have same source
        @test isapprox(sources[1], sources[5], rtol=1e-15)
        @test isapprox(sources[5], sources[8], rtol=1e-15)
        @test isapprox(sources[8], sources[4], rtol=1e-15)
        # Test the 4 inner elements have same source
        @test isapprox(sources[2], sources[6], rtol=1e-14)
        @test isapprox(sources[6], sources[7], rtol=1e-15)
        @test isapprox(sources[7], sources[3], rtol=1e-15)
        # test that some are not equal
        @test false == isapprox(sources[1], sources[2], rtol=1e-8)
        @test false == isapprox(sources[4], sources[6], rtol=1e-8)

        wavevector2 = [wavenumber, 0.0, 0.0]
        planewaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector2, [x_test,y_test,z_test])
        sources = solveSoftIE(mesh_filename,
                        planewaveExcitation,
                        wavenumber,
                        src_quadrature_rule,
                        test_quadrature_rule,
                        distance_to_edge_tol,
                        near_singular_tol)
        @test isapprox(sources[1], sources[5], rtol=1e-14)
        @test isapprox(sources[2], sources[6], rtol=1e-14)
        @test isapprox(sources[7], sources[3], rtol=1e-14)
        @test isapprox(sources[8], sources[4], rtol=1e-14)
        # test that some are not equal
        @test false == isapprox(sources[5], sources[6], rtol=1e-8)
        @test false == isapprox(sources[2], sources[4], rtol=1e-8)
    end
end
