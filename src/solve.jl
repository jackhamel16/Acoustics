# dependencies: fill.jl greens_functions.jl mesh.jl

function solveSoftIE(pulse_mesh::PulseMesh,
                     excitation::Function,
                     wavenumber::Number,
                     distance_to_edge_tol::Float64,
                     near_singular_tol::Float64)
    @unpack num_elements = pulse_mesh
    println("Filling RHS...")
    RHS = zeros(ComplexF64, num_elements)
    rhs_fill_time = @elapsed rhsFill!(pulse_mesh, excitation, RHS)
    pulse_mesh.RHS = RHS
    println("  RHS fill time: ", rhs_fill_time)

    matrix_fill_time = @elapsed begin
        if pulse_mesh.Z_factors == lu(ones(1,1))
            println("Filling Matrix...")
            testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                           wavenumber,
                                                           r_test,
                                                           distance_to_edge_tol,
                                                           near_singular_tol,
                                                           is_singular)
            Z_matrix = zeros(ComplexF64, num_elements, num_elements)
            matrixFill!(pulse_mesh, testIntegrand, Z_matrix)
            Z_factors = lu(Z_matrix)
            pulse_mesh.Z_factors = Z_factors
        end
    end
    println("  Matrix fill time: ", matrix_fill_time)

    println("Solving...")
    solve_time = @elapsed source_vec = pulse_mesh.Z_factors \ pulse_mesh.RHS
    println("  Solve time: ", solve_time)

    return(source_vec)
end

function solveSoftIENormalDeriv(pulse_mesh::PulseMesh,
                                excitation_normal_derivative::Function,
                                wavenumber::Number)

    @unpack num_elements = pulse_mesh
    println("Filling RHS...")
    RHS = zeros(ComplexF64, num_elements)
    rhs_fill_time = @elapsed rhsNormalDerivFill!(pulse_mesh, excitation_normal_derivative, RHS)
    pulse_mesh.RHS = RHS
    println("  RHS fill time: ", rhs_fill_time)

    matrix_fill_time = @elapsed begin
        if pulse_mesh.Z_factors == lu(ones(1,1))
            println("Filling Matrix...")
            testIntegrandNormalDerivative(r_test, src_idx, is_singular) =
                                    scalarGreensNormalDerivativeIntegration(pulse_mesh,
                                                                            src_idx,
                                                                            wavenumber,
                                                                            r_test,
                                                                            is_singular)
            Z_matrix = zeros(ComplexF64, num_elements, num_elements)
            matrixFill!(pulse_mesh, testIntegrandNormalDerivative, Z_matrix)
            Z_factors = lu(Z_matrix)
            pulse_mesh.Z_factors = Z_factors
        end
    end
    println("  Matrix fill time: ", matrix_fill_time)

    println("Solving...")
    solve_time = @elapsed source_vec = pulse_mesh.Z_factors \ pulse_mesh.RHS
    println("  Solve time: ", solve_time)

    return(source_vec)
end

function solveSoftCFIE(pulse_mesh::PulseMesh,
                       excitation::Function,
                       excitation_normal_derivative::Function,
                       wavenumber::Number,
                       distance_to_edge_tol::Float64,
                       near_singular_tol::Float64,
                       softIE_weight::Float64)

    @unpack num_elements = pulse_mesh
    matrix_fill_time = @elapsed begin
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
        # avg_z_nd = sum(abs.(z_matrix_nd))./length(z_matrix_nd)
        # avg_z = sum(abs.(z_matrix))./length(z_matrix)
        # nd_scale_factor = im*avg_z / avg_z_nd # probably shouldn't be here long term
        # println("avg nd Z value = ", avg_z_nd)
        # println("avg Z value = ", avg_z)
        # println("normal derivative scale factor = ", nd_scale_factor,"\n")
        # Z_factors = lu(softIE_weight * z_matrix + (1-softIE_weight) * nd_scale_factor * z_matrix_nd)
        Z_factors = lu(softIE_weight * z_matrix + (1-softIE_weight) * im * z_matrix_nd)
        pulse_mesh.Z_factors = Z_factors
    end
    println("  Matrix fill time: ", matrix_fill_time)

    rhs_fill_time = @elapsed begin
        println("Filling RHS...")
        rhs_nd = zeros(ComplexF64, num_elements)
        rhsNormalDerivFill!(pulse_mesh, excitation_normal_derivative, rhs_nd)
        rhs = zeros(ComplexF64, num_elements)
        rhsFill!(pulse_mesh, excitation, rhs)
        # RHS = softIE_weight * rhs + (1-softIE_weight) * nd_scale_factor * rhs_nd
        RHS = softIE_weight * rhs + (1-softIE_weight) * im * rhs_nd
        pulse_mesh.RHS = RHS
    end
    println("  RHS fill time: ", rhs_fill_time)

    println("Solving...")
    solve_time = @elapsed source_vec = pulse_mesh.Z_factors \ pulse_mesh.RHS
    println("  Solve time: ", solve_time)

    return(source_vec)
end
