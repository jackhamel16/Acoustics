using Test

include("../../src/includes.jl")

function getSources(filename::String)
    file = open(filename, "r")
    file_lines = split(read(file, String), "\n")
    num_lines = length(file_lines)
    num_sources = num_lines - 3
    source_file_lines = file_lines[2:num_sources+1]
    sources = Array{Float64,1}(undef, num_sources)
    for line_idx = 1:num_sources
        source_line = source_file_lines[line_idx]
        sources[line_idx] = parse.(Float64, split(split(split(source_line,"{")[2],"}")[1],","))[1]
    end
    close(file)
    return(sources)
end

@testset "run tests" begin
    excitation_amplitude = 1.0
    lambda=1.0
    wavenumber = 2*pi/lambda + 0*im
    wavevector = [wavenumber, 0.0, 0.0]
    src_quadrature_rule = gauss7rule
    test_quadrature_rule = gauss1rule
    distance_to_edge_tol = 1e-12
    near_singular_tol = 1.0
    mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    planeWaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
    sol_sources = solveSoftIE(pulse_mesh,
                    planeWaveExcitation,
                    wavenumber,
                    distance_to_edge_tol,
                    near_singular_tol)
    test_output = readchomp(`julia run.jl examples/basic_soundsoft.txt`)
    test_sources = getSources("sources_real.pos")
    @test isapprox(test_sources, real.(sol_sources), rtol=1e-14)

    planeWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = planeWaveNormalDerivative(excitation_amplitude, wavevector, [x_test,y_test,z_test], normal)
    sol_sources = solveSoftIENormalDeriv(pulse_mesh,
                                         planeWaveExcitationNormalDeriv,
                                         wavenumber)
    test_output = readchomp(`julia run.jl examples/basic_soundsoftnormalderiv.txt`)
    test_sources = getSources("sources_real.pos")
    @test isapprox(test_sources, real.(sol_sources), rtol=1e-14)

    excitation_amplitude = 1.5
    lambda=2.0
    wavenumber = 2*pi/lambda + 0*im
    l, m = 1, 1
    sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
    num_levels = 3
    compression_distance = 1.5
    ACA_approximation_tol = 1e-6
    sol_sources, octree, metrics = solveSoundSoftIEACA(pulse_mesh, num_levels,
                                                       sphericalWaveExcitation, wavenumber,
                                                       distance_to_edge_tol, near_singular_tol,
                                                       compression_distance, ACA_approximation_tol)
    test_output = readchomp(`julia run.jl examples/basic_ACA.txt`)
    test_sources = getSources("sources_real.pos")
    @test isapprox(test_sources, real.(sol_sources), rtol=1e-14)

    sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
    soundsoftIE_weight = 0.5
    sol_sources = solveSoftCFIE(pulse_mesh,
                           sphericalWaveExcitation,
                           sphericalWaveExcitationNormalDeriv,
                           wavenumber,
                           distance_to_edge_tol,
                           near_singular_tol,
                           soundsoftIE_weight)
    test_output = readchomp(`julia run.jl examples/basic_soundsoftCFIE.txt`)
    test_sources = getSources("sources_real.pos")
    @test isapprox(test_sources, real.(sol_sources), rtol=1e-14)

    max_l = 5
    mode_idx = 3
    lambda = 1
    wavenumber = 2*pi/lambda
    src_quadrature_rule = gauss7rule
    test_quadrature_rule = gauss7rule
    distance_to_edge_tol = 1e-12
    near_singular_tol = 1.0
    mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    sol_sources = solveWSMode(max_l, mode_idx, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    test_output = readchomp(`julia run.jl examples/basic_WS.txt`)
    test_sources = getSources("sources_real.pos")
    @test isapprox(test_sources, real.(sol_sources), rtol=1e-14)

    rm("sources_mag.pos")
    rm("sources_real.pos")
    rm("sources_imag.pos")
    rm("Wigner_Smith_time_delays.txt")
end
