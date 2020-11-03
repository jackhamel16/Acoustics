include("excitation.jl")
include("fill.jl")
include("greens_functions.jl")
include("mesh.jl")

excitation_amplitude = 1.0
wavenumber = 1/20+0*im
wavevector = [0.0, 0.0, wavenumber]
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss1rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0

mesh_filename = "examples/simple/rectangle_plate_8elements_symmetric.msh"
planewaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
sources = solve(mesh_filename,
                planewaveExcitation,
                wavenumber,
                src_quadrature_rule,
                test_quadrature_rule,
                distance_to_edge_tol,
                near_singular_tol)
