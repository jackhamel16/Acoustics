include("includes.jl")

excitation_amplitude = 1.0
lambda=10
wavenumber = 2*pi/lambda + 0*im
wavevector = [wavenumber, 0.0, 0.0]
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss7rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0

# mesh_filename = "examples/test/rectangular_strip_fine.msh"
# mesh_filename = "examples/simple/plates/rectangular_strip_super_fine.msh"
mesh_filename = "examples/test/spheres/sphere_1m_328.msh"
# mesh_filename = "examples/test/circular_plate_1m.msh"
pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
planeWaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
# planeWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = planeWaveNormalDerivative(excitation_amplitude, wavevector, [x_test,y_test,z_test], normal)
# l, m = 0, 0
# sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m)
# sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m, normal)
# sources = solveSoftCFIE(pulse_mesh,
#                 sphericalWaveExcitation,
#                 sphericalWaveExcitationNormalDeriv,
#                 wavenumber,
#                 distance_to_edge_tol,
#                 near_singular_tol,
#                 nd_scale_factor)
println("Running...")
println("Number of Elements = ", pulse_mesh.num_elements)
println("Direct Solver")
@time sources1 = solveSoftIE(pulse_mesh,
                planeWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
# sources = solveSoftIENormalDeriv(pulse_mesh,
#                 sphericalWaveExcitationNormalDeriv,
#                 wavenumber)
# max_l = 6
# mode_idx = 1
# sources = solveWSMode(max_l, mode_idx, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
println("Compressed Solver")
num_levels = 3
compression_distance = 1.5
ACA_approximation_tol = 1e-4
@time sources2, octree, metrics = solveSoundSoftIEACA(pulse_mesh, num_levels, planeWaveExcitation, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_approximation_tol)
printACAMetrics(metrics)

error = norm(sources1 - sources2) / norm(sources1)
println("Relative l2-norm Error = ", error)
#
# real_filename = "sources_real"
# imag_filename = "sources_imag"
# mag_filename = "sources_mag"
# exportSourcesGmsh(mesh_filename, real_filename, real.(sources))
# exportSourcesGmsh(mesh_filename, imag_filename, imag.(sources))
# exportSourcesGmsh(mesh_filename, mag_filename, abs.(sources))
