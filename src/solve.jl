# dependencies: fill.jl greens_functions.jl mesh.jl

function solve(mesh_filename::String,
               excitation::Function,
               wavenumber::Complex{Float64},
               src_quadrature_rule::AbstractArray{Float64, 2},
               test_quadrature_rule::AbstractArray{Float64, 2},
               distance_to_edge_tol::Float64,
               near_singular_tol::Float64)

    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    println("Filling RHS...")
    rhs = rhsFill(pulse_mesh, excitation)
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                   wavenumber,
                                                   r_test,
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    println("Filling Matrix...")
    z_matrix = matrixFill(pulse_mesh, testIntegrand)
    println("Inverting Matrix...")
    source_vec = z_matrix \ rhs
    source_vec
end
