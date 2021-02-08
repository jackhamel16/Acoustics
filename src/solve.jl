# dependencies: fill.jl greens_functions.jl mesh.jl

function solveSoftIE(mesh_filename::String,
                     excitation::Function,
                     wavenumber::Complex{Float64},
                     src_quadrature_rule::AbstractArray{Float64, 2},
                     test_quadrature_rule::AbstractArray{Float64, 2},
                     distance_to_edge_tol::Float64,
                     near_singular_tol::Float64)

    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    @unpack num_elements = pulse_mesh
    println("Filling RHS...")
    rhs = zeros(ComplexF64, num_elements)
    rhsFill(pulse_mesh, excitation, rhs)
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                   wavenumber,
                                                   r_test,
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    println("Filling Matrix...")
    z_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill(pulse_mesh, testIntegrand, z_matrix)
    println("Inverting Matrix...")
    source_vec = z_matrix \ rhs
    source_vec
end

function solveSoftIENormalDeriv(mesh_filename::String,
                                excitation::Function,
                                excitation_normal_derivative::Function,
                                wavenumber::Complex{Float64},
                                src_quadrature_rule::AbstractArray{Float64, 2},
                                test_quadrature_rule::AbstractArray{Float64, 2})

    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    @unpack num_elements = pulse_mesh
    println("Filling RHS...")
    rhs = zeros(ComplexF64, num_elements)
    rhsFill(pulse_mesh, excitation_normal_derivative, rhs, true)
    testIntegrandNormalDerivative(r_test, src_idx, is_singular) =
                            scalarGreensNormalDerivativeIntegration(pulse_mesh,
                                                                    src_idx,
                                                                    wavenumber,
                                                                    r_test,
                                                                    is_singular)
    println("Filling Matrix...")
    z_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill(pulse_mesh, testIntegrandNormalDerivative, z_matrix)
    println("Inverting Matrix...")
    source_vec = z_matrix \ rhs
    source_vec
end

function solveSoftCFIE(mesh_filename::String,
                       excitation::Function,
                       excitation_normal_derivative::Function,
                       wavenumber::Complex{Float64},
                       src_quadrature_rule::AbstractArray{Float64, 2},
                       test_quadrature_rule::AbstractArray{Float64, 2},
                       distance_to_edge_tol::Float64,
                       near_singular_tol::Float64,
                       nd_scale_factor::Float64)

    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    @unpack num_elements = pulse_mesh
    println("Filling RHS...")
    rhs = zeros(ComplexF64, num_elements)
    rhsFill(pulse_mesh, excitation_normal_derivative, rhs, true)
    rhsFill(pulse_mesh, excitation, nd_scale_factor .* rhs)
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh,
                                                        src_idx,
                                                        wavenumber,
                                                        r_test,
                                                        distance_to_edge_tol,
                                                        near_singular_tol,
                                                        is_singular)
    testIntegrandNormalDerivative(r_test, src_idx, is_singular) =
                            scalarGreensNormalDerivativeIntegration(pulse_mesh,
                                                                    src_idx,
                                                                    wavenumber,
                                                                    r_test,
                                                                    is_singular)
    println("Filling Matrix...")
    z_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill(pulse_mesh, testIntegrandNormalDerivative, z_matrix)
    matrixFill(pulse_mesh, testIntegrand, nd_scale_factor .* z_matrix)
    println("Inverting Matrix...")
    source_vec = z_matrix \ rhs
    source_vec
end
