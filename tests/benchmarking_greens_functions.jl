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

pulse_mesh = buildPulseMesh(mesh_filename)

# testIntegrand(r_test, nodes, is_singular)::Complex{Float64} = scalarGreensIntegration(wavenumber,
#                                                r_test,
#                                                nodes,
#                                                src_quadrature_rule,
#                                                distance_to_edge_tol,
#                                                near_singular_tol,
#                                                is_singular)


r_test = [10.0, -1.0, 0.0]
r_src = [0.0, 0.0, 0.0]
k = 1+0.0*im
nodes = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
computeScalarGreensSingularityIntegralParameters(r_test, nodes)
scalarGreensIntegration(wavenumber, r_test, nodes, src_quadrature_rule, distance_to_edge_tol, near_singular_tol, false)
scalarGreensNonSingularIntegral(wavenumber, r_test, nodes, src_quadrature_rule)
