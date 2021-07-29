using Test

include("../../src/code_structures/input_params.jl")

@testset "input_params tests" begin
    @testset "InputParams tests" begin
        mesh_filename = "test_filename"
        equation = "sound soft IE"
        src_quadrature_rule = "gauss7"
        test_quadrature_rule = "gauss1"
        excitation_params = ExcitationParams()
        use_ACA = false
        ACA_params = ACAParams()
        test_params = InputParams(mesh_filename=mesh_filename,
                                  equation=equation,
                                  src_quadrature_rule=src_quadrature_rule,
                                  test_quadrature_rule=test_quadrature_rule,
                                  excitation_params=excitation_params,
                                  use_ACA=use_ACA)
        @test test_params.mesh_filename == mesh_filename
        @test test_params.equation == equation
        @test test_params.src_quadrature_rule == src_quadrature_rule
        @test test_params.test_quadrature_rule == test_quadrature_rule
        @test test_params.excitation_params == excitation_params
        @test isapprox(test_params.distance_to_edge_tol, 1e-12, rtol = 1e-14)
        @test isapprox(test_params.near_singular_tol, 1.0, rtol = 1e-14)
        @test test_params.use_ACA == use_ACA
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
        test_num_levels = 2
        test_compression_distance = 2.2
        test_ACA_params = ACAParams(num_levels=test_num_levels,
                                    compression_distance=test_compression_distance)
        @test isapprox(test_ACA_params.num_levels, test_num_levels, rtol = 1e-14)
        @test isapprox(test_ACA_params.compression_distance, test_compression_distance, rtol = 1e-14)
        @test isapprox(test_ACA_params.approximation_tol, 0, rtol = 1e-14)
    end # ExcitationParams tests
    @testset "parseACAParams tests" begin

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
        filename = "examples/test/test_inputs1.txt"
        excitation_params = parseExcitationParams(split(read(open(inputs_filename, "r"), String), "\r"))
        test = parseInputParams(filename)
        @test test[1] == "gibberish.msh"
        @test test[2] == "sound soft IE"
        @test test[3] == "gauss7"
        @test test[4] == "gauss1"
        @test test[5] == 1e-10
        @test test[6] == 2.2
        @test test[7].type == "planewave"
        @test isapprox(test[7].lambda, 1.1, rtol=1e-14)
        @test isapprox(test[7].amplitude, -1.5, rtol=1e-14)
        @test isapprox(test[7].wavenumber, 2 * pi / 1.1, rtol=1e-14)
        @test isapprox(test[7].wavevector, [2 * pi / 1.1, 0, 0], rtol=1e-14)
        @test isapprox(test[7].l, 0, rtol=1e-14)
        @test isapprox(test[7].m, 0, rtol=1e-14)
        @test test[8] == false
    end # parseInputParams tests
end #input_params tests
