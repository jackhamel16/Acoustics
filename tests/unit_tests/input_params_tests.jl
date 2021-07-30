using Test

include("../../src/code_structures/input_params.jl")

@testset "input_params tests" begin
    @testset "InputParams tests" begin
        mesh_filename = "test_filename"
        equation = "sound soft IE"
        src_quadrature_string = "gauss7"
        test_quadrature_string = "gauss1"
        excitation_params = ExcitationParams()
        ACA_params = ACAParams()
        test_params = InputParams(mesh_filename=mesh_filename,
                                  equation=equation,
                                  src_quadrature_string=src_quadrature_string,
                                  test_quadrature_string=test_quadrature_string,
                                  excitation_params=excitation_params)
        @test test_params.mesh_filename == mesh_filename
        @test test_params.equation == equation
        @test test_params.src_quadrature_string == src_quadrature_string
        @test test_params.test_quadrature_string == test_quadrature_string
        @test test_params.excitation_params == excitation_params
        @test isapprox(test_params.distance_to_edge_tol, 1e-12, rtol = 1e-14)
        @test isapprox(test_params.near_singular_tol, 1.0, rtol = 1e-14)
        @test test_params.ACA_params == ACA_params
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
        test_use_ACA = true
        test_num_levels = 2
        test_compression_distance = 2.2
        test_ACA_params = ACAParams(use_ACA=test_use_ACA,
                                    num_levels=test_num_levels,
                                    compression_distance=test_compression_distance)
        @test test_ACA_params.use_ACA == test_use_ACA
        @test isapprox(test_ACA_params.num_levels, test_num_levels, rtol = 1e-14)
        @test isapprox(test_ACA_params.compression_distance, test_compression_distance, rtol = 1e-14)
        @test isapprox(test_ACA_params.approximation_tol, 0, rtol = 1e-14)
    end # ExcitationParams tests
    @testset "parseACAParams tests" begin
        inputs_filename = "examples/test/test_inputs1.txt"
        file = open(inputs_filename, "r")
        file_lines = split(read(file, String), "\r")
        test_ACA_params = parseACAParams(file_lines)
        @test test_ACA_params == ACAParams()

        inputs_filename = "examples/test/test_inputs2.txt"
        file = open(inputs_filename, "r")
        file_lines = split(read(file, String), "\r")
        test_ACA_params = parseACAParams(file_lines)
        sol_ACA_params = ACAParams(true, 3, 1.6, 1e-6)
        @test test_ACA_params == sol_ACA_params
    end #parseACAParams
    @testset "parseExcitationParams tests" begin
        inputs_filename = "examples/test/test_inputs1.txt"
        file = open(inputs_filename, "r")
        file_lines = split(read(file, String), "\r")
        excitation_params = parseExcitationParams(file_lines)
        @test excitation_params.type == "planewave"
        @test isapprox(excitation_params.lambda, 1.1, rtol=1e-14)
        @test isapprox(excitation_params.amplitude, -1.5, rtol=1e-14)
        @test isapprox(excitation_params.wavenumber, 2 * pi / 1.1, rtol=1e-14)
        @test isapprox(excitation_params.wavevector, [ 2 * pi / 1.1, 0, 0], rtol=1e-14)
        @test isapprox(excitation_params.l, 0, rtol=1e-14)
        @test isapprox(excitation_params.m, 0, rtol=1e-14)

        inputs_filename = "examples/test/test_inputs2.txt"
        file = open(inputs_filename, "r")
        file_lines = split(read(file, String), "\r")
        excitation_params = parseExcitationParams(file_lines)
        @test excitation_params.type == "sphericalwave"
        @test isapprox(excitation_params.lambda, 1.1, rtol=1e-14)
        @test isapprox(excitation_params.amplitude, -1.5, rtol=1e-14)
        @test isapprox(excitation_params.wavenumber, 2 * pi / 1.1, rtol=1e-14)
        @test isapprox(excitation_params.wavevector, [ 2 * pi / 1.1, 0, 0], rtol=1e-14)
        @test isapprox(excitation_params.l, 2, rtol=1e-14)
        @test isapprox(excitation_params.m, 1, rtol=1e-14)
    end # parseExcitationParams tests
    @testset "parseInputParams tests" begin
        inputs_filename = "examples/test/test_inputs1.txt"
        test_input_params = parseInputParams(inputs_filename)
        @test test_input_params.mesh_filename == "gibberish.msh"
        @test test_input_params.equation == "sound soft IE"
        @test test_input_params.src_quadrature_string == "gauss7"
        @test test_input_params.test_quadrature_string == "gauss1"
        @test test_input_params.distance_to_edge_tol == 1e-10
        @test test_input_params.near_singular_tol == 2.2
        @test test_input_params.excitation_params.type == "planewave"
        @test isapprox(test_input_params.excitation_params.lambda, 1.1, rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.amplitude, -1.5, rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.wavenumber, 2 * pi / 1.1, rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.wavevector, [2 * pi / 1.1, 0, 0], rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.l, 0, rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.m, 0, rtol=1e-14)
        @test test_input_params.ACA_params.use_ACA == false

        inputs_filename = "examples/test/test_inputs2.txt"
        test_input_params = parseInputParams(inputs_filename)
        @test test_input_params.mesh_filename == "gibberish2.msh"
        @test test_input_params.equation == "sound soft normal derivative IE"
        @test test_input_params.src_quadrature_string == "gauss3"
        @test test_input_params.test_quadrature_string == "gauss7"
        @test test_input_params.distance_to_edge_tol == 1e-8
        @test test_input_params.near_singular_tol == 2.0
        @test test_input_params.excitation_params.type == "sphericalwave"
        @test isapprox(test_input_params.excitation_params.lambda, 1.1, rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.amplitude, -1.5, rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.wavenumber, 2 * pi / 1.1, rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.wavevector, [2 * pi / 1.1, 0, 0], rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.l, 2, rtol=1e-14)
        @test isapprox(test_input_params.excitation_params.m, 1, rtol=1e-14)
        @test test_input_params.ACA_params.use_ACA == true
        @test isapprox(test_input_params.ACA_params.num_levels, 3, rtol=1e-14)
        @test isapprox(test_input_params.ACA_params.compression_distance, 1.6, rtol=1e-14)
        @test isapprox(test_input_params.ACA_params.approximation_tol, 1e-6, rtol=1e-14)
    end # parseInputParams tests
end #input_params tests
