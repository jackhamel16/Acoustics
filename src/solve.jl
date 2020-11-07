include("fill.jl")
include("greens_functions.jl")
include("mesh.jl")

function solve(mesh_filename::String,
               excitation::Function,
               wavenumber::Complex{Float64},
               src_quadrature_rule::Array{Float64, 2},
               test_quadrature_rule::Array{Float64, 2},
               distance_to_edge_tol::Float64,
               near_singular_tol::Float64)

    pulse_mesh = buildPulseMesh(mesh_filename)
    rhs = rhsFill(pulse_mesh.num_elements, pulse_mesh.elements, pulse_mesh.nodes, excitation, test_quadrature_rule)
    testIntegrand(r_test, nodes, is_singular) = scalarGreensIntegration(wavenumber,
                                                   r_test,
                                                   nodes,
                                                   src_quadrature_rule,
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    z_matrix = matrixFill(pulse_mesh.num_elements, pulse_mesh.elements, pulse_mesh.nodes, testIntegrand, test_quadrature_rule)
    source_vec = z_matrix \ rhs
    source_vec
end
