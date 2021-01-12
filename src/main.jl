include("excitation.jl")
include("mesh.jl")
include("solve.jl")

include("solve_simple.jl")


excitation_amplitude = 1.0
# wavenumber = 32*pi/5+0*im
# wavevector = [wavenumber/sqrt(2), 0.0, wavenumber/sqrt(2)]
lambda=1
wavenumber = 2*pi/lambda + 0*im
wavevector = [wavenumber, 0.0, 0.0]
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss1rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0

mesh_filename = "examples/simple/circular_plate_1m.msh"
planeWaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
# l, m = 0, 0
# sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m)
sources = solve(mesh_filename,
                planeWaveExcitation,
                wavenumber,
                src_quadrature_rule,
                test_quadrature_rule,
                distance_to_edge_tol,
                near_singular_tol)
sources = solveSimple(mesh_filename,
            planeWaveExcitation,
            wavenumber)

real_filename = "sources_real"
imag_filename = "sources_imag"
mag_filename = "sources_mag"
exportSourcesGmsh(mesh_filename, real_filename, real.(sources))
exportSourcesGmsh(mesh_filename, imag_filename, imag.(sources))
exportSourcesGmsh(mesh_filename, mag_filename, abs.(sources))
