include("excitation.jl")
include("mesh.jl")
include("solve.jl")

excitation_amplitude = 1.0
wavenumber = 32*pi/5+0*im
wavevector = [wavenumber/sqrt(2), 0.0, wavenumber/sqrt(2)]
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss1rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0

mesh_filename = "examples/simple/rectangle_plate_symmetric.msh"
# planewaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
l, m = 1, 0
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m)
sources = solve(mesh_filename,
                sphericalWaveExcitation,
                wavenumber,
                src_quadrature_rule,
                test_quadrature_rule,
                distance_to_edge_tol,
                near_singular_tol)

real_filename = "sources_real"
imag_filename = "sources_imag"
mag_filename = "sources_mag"
exportSourcesGmsh(mesh_filename, real_filename, real.(sources))
exportSourcesGmsh(mesh_filename, imag_filename, imag.(sources))
exportSourcesGmsh(mesh_filename, mag_filename, abs.(sources))
