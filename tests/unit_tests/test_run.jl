using Test



@testset "run tests" begin
    excitation_amplitude = 1.0
    lambda=10
    wavenumber = 2*pi/lambda + 0*im
    wavevector = [wavenumber, 0.0, 0.0]
    src_quadrature_rule = gauss7rule
    test_quadrature_rule = gauss1rule
    distance_to_edge_tol = 1e-12
    near_singular_tol = 1.0
    mesh_filename = "examples/test/circular_plate_1m.msh"
    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    planeWaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
    sol_sources = solveSoftIE(pulse_mesh,
                    planeWaveExcitation,
                    wavenumber,
                    distance_to_edge_tol,
                    near_singular_tol)
    test_output = readchomp(`julia run.jl examples/basic.txt`)
    # run(`julia run.jl examples/basic.txt ">" test_output.txt`)
    test_output_lines = split(test_output, "\n")

end
