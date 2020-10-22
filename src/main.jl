include("mesh.jl")
include("fill.jl")
include("greens_functions.jl")

wavenumber = 1/20+0*im
wavevector = [0.0, 0.0, wavenumber]
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss1rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0

mesh_filename = "examples/simple/rectangle_plate.msh"
pulse_mesh = buildPulseMesh(mesh_filename)

planewave(x,y,z) = exp(-1*im*dot(wavevector, [x,y,z]))

rhs = rhsFill(pulse_mesh.num_elements, pulse_mesh.elements, pulse_mesh.nodes, planewave, test_quadrature_rule)
testIntegrand(r_test, nodes, is_singular) = scalarGreensIntegration(wavenumber,
                                               r_test,
                                               nodes,
                                               src_quadrature_rule,
                                               distance_to_edge_tol,
                                               near_singular_tol,
                                               is_singular)
z_matrix = matrixFill(pulse_mesh.num_elements, pulse_mesh.elements, pulse_mesh.nodes, testIntegrand, test_quadrature_rule)
source_vec = z_matrix \ rhs
