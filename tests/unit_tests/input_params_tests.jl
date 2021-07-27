using Test

include("../../src/code_structures/input_params.jl")

@testset "input_params tests" begin
    @testset "InputParams tests" begin
        mesh_filename = "test_filename"
        src_quadrature_rule = "gauss7"
        test_quadrature_rule = "gauss1"
        excitation_params = ExcitationParams()
        test_params = InputParams(mesh_filename=mesh_filename,
                                  src_quadrature_rule=src_quadrature_rule,
                                  test_quadrature_rule=test_quadrature_rule,
                                  excitation_params=excitation_params)
        @test test_params.mesh_filename == mesh_filename
        @test test_params.src_quadrature_rule == src_quadrature_rule
        @test test_params.test_quadrature_rule == test_quadrature_rule
        @test test_params.excitation_params == excitation_params
        @test isapprox(test_params.distance_to_edge_tol, 1e-12, rtol = 1e-14)
        @test isapprox(test_params.near_singular_tol, 1.0, rtol = 1e-14)
    end # InputParams tests
    @testset "ExcitationParams tests" begin
        excitation_type = "planewave"
        test_lambda = 1.0
        test_wavenumber = 2*pi/test_lambda
        test_wavevector = test_wavenumber .* [0.0, 1.0, 0.0]
        test_amplitude = 2.5
        test_l = 1
        test_m = 1
        test_excitation_params = ExcitationParams(excitation_type,
                                                  test_lambda,
                                                  test_wavenumber,
                                                  test_wavevector,
                                                  test_amplitude,
                                                  test_l,
                                                  test_m)
        @test test_excitation_params.type == excitation_type
        @test test_excitation_params.lambda == test_lambda
        @test test_excitation_params.wavenumber == test_wavenumber
        @test test_excitation_params.wavevector == test_wavevector
        @test test_excitation_params.amplitude == test_amplitude
        @test test_excitation_params.l == test_l
        @test test_excitation_params.m == test_m
    end # ExcitationParams tests
    @testset "ACAParams tests" begin
        test_num_levels = 2
        test_compression_distance = 2.2
        test_ACA_params = ACAParams(num_levels=test_num_levels,
                                    compression_distance=test_compression_distance)
        @test isapprox(test_ACA_params.num_levels, test_num_levels, rtol = 1e-14)
        @test isapprox(test_ACA_params.compression_distance, test_compression_distance, rtol = 1e-14)
        @test isapprox(test_ACA_params.approximation_tol, 1e-4, rtol = 1e-14)
    end # ExcitationParams tests
end #input_params tests
