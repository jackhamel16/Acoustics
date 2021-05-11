using BenchmarkTools

include("../src/includes.jl")

# n = 7

# scale_factor = 1.1
# l2Norm(x, y, z) = sqrt(x^2 + y^2 + z^2)
# points = hcat(randn(n), randn(n), randn(n))
# weights = randn(n)
# nodes = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
#
# pointsT = hcat(randn(n), randn(n), randn(n), randn(n), randn(n), randn(n), randn(n))


mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
excitation_amplitude = 1.0
wavenumber = 2*pi/20+0*im
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss1rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0

pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

r_test = [10.0, -1.0, 0.0]
nodes = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                               wavenumber,
                                               r_test,
                                               distance_to_edge_tol,
                                               near_singular_tol,
                                               is_singular)
z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
# z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements); matrixFill(pulse_mesh, testIntegrand, z_matrix)
# @benchmark matrixFill(pulse_mesh.num_elements, pulse_mesh.elements, pulse_mesh.nodes, testIntegrand, test_quadrature_rule)
getTriangleNodes(1, pulse_mesh.elements, pulse_mesh.nodes)
