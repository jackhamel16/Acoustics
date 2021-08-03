# dependencies: fill.jl greens_functions.jl mesh.jl

function solveSoftIE(pulse_mesh::PulseMesh,
                     excitation::Function,
                     wavenumber::Number,
                     distance_to_edge_tol::Float64,
                     near_singular_tol::Float64,
                     return_z_rhs=false)

    @unpack num_elements = pulse_mesh
    println("Filling RHS...")
    rhs = zeros(ComplexF64, num_elements)
    rhsFill!(pulse_mesh, excitation, rhs)
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                   wavenumber,
                                                   r_test,
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    println("Filling Matrix...")
    z_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill!(pulse_mesh, testIntegrand, z_matrix)
    println("Solving...")
    source_vec = z_matrix \ rhs
    if return_z_rhs == false
        return(source_vec)
    else
        return(source_vec, z_matrix, rhs)
    end
end

function solveSoftIENormalDeriv(pulse_mesh::PulseMesh,
                                excitation_normal_derivative::Function,
                                wavenumber::Complex{Float64},
                                return_z=false)

    @unpack num_elements = pulse_mesh
    println("Filling RHS...")
    rhs = zeros(ComplexF64, num_elements)
    rhsFill!(pulse_mesh, excitation_normal_derivative, rhs, true)
    testIntegrandNormalDerivative(r_test, src_idx, is_singular) =
                            scalarGreensNormalDerivativeIntegration(pulse_mesh,
                                                                    src_idx,
                                                                    wavenumber,
                                                                    r_test,
                                                                    is_singular)
    println("Filling Matrix...")
    z_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill!(pulse_mesh, testIntegrandNormalDerivative, z_matrix)
    println("Solving...")
    source_vec = z_matrix \ rhs
    if return_z == false
        return(source_vec)
    else
        return(source_vec, z_matrix)
    end
end

function solveSoftCFIE(pulse_mesh::PulseMesh,
                       excitation::Function,
                       excitation_normal_derivative::Function,
                       wavenumber::Complex{Float64},
                       distance_to_edge_tol::Float64,
                       near_singular_tol::Float64,
                       softIE_weight::Float64,
                       return_z=false)

    @unpack num_elements = pulse_mesh
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
    z_matrix_nd = zeros(ComplexF64, num_elements, num_elements)
    matrixFill!(pulse_mesh, testIntegrandNormalDerivative, z_matrix_nd)
    z_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill!(pulse_mesh, testIntegrand, z_matrix)

    avg_z_nd = sum(abs.(z_matrix_nd))./length(z_matrix_nd)
    avg_z = sum(abs.(z_matrix))./length(z_matrix)
    nd_scale_factor = im*avg_z / avg_z_nd
    z_matrix = softIE_weight * z_matrix + (1-softIE_weight) * nd_scale_factor * z_matrix_nd

    println("Filling RHS...")
    rhs_nd = zeros(ComplexF64, num_elements)
    rhsFill!(pulse_mesh, excitation_normal_derivative, rhs_nd, true)
    rhs = zeros(ComplexF64, num_elements)
    rhsFill!(pulse_mesh, excitation, rhs)
    rhs = softIE_weight * rhs + (1-softIE_weight) * nd_scale_factor * rhs_nd

    println("Solving...")
    source_vec = z_matrix \ rhs
    if return_z == false
        return(source_vec)
    else
        return(source_vec, z_matrix)
    end
end
