using BenchmarkTools

include("../src/includes.jl")

# max_l = 8
# lambda=10.0
# wavenumber = 2*pi/lambda + 0*im
# excitation_amplitude = 1.0
# num_harmonics = 81
# src_quadrature_rule = gauss7rule
# test_quadrature_rule = gauss1rule
# distance_to_edge_tol = 1e-12
# near_singular_tol = 1.0
# # mesh_filename= "examples/test/rectangle_plate_8elements_symmetric.msh"
# mesh_filename = "examples/simple/circular_plate_1m.msh"
# pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

# test_S = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)

# max_l = 8
# lambda = 5
# wavenumber = 2*pi/lambda + 0.0im
# num_harmonics = 81
# src_quadrature_rule = gauss7rule
# test_quadrature_rule = gauss7rule
# distance_to_edge_tol = 1e-12
# near_singular_tol = 1.0
# mesh_filename = "examples/simple/circular_plate_1m.msh"
# pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
#
# test_dSdk = calculateScatteringMatrixDerivative(max_l, num_harmonics, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)


max_l = 10
lambda = 1
wavenumber = 2*pi/lambda + 0.0im
num_harmonics = max_l^2 + 2*max_l + 1
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss7rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0
mesh_filename = "examples/test/circular_plate_1m.msh"
pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
test_Q = calculateWSMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
