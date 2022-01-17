# dependencies: fill.jl greens_functions.jl mesh.jl

################################################################################
# This file contains functions that solve various acoustic integral equations  #
# for the equivalent sources that exist on the surface of a scatterer w/o ACA  #
# See documentation section 3                                                  #
################################################################################

function solveSoftIE(pulse_mesh::PulseMesh,
                     excitation::Function,
                     wavenumber::Number,
                     distance_to_edge_tol::Float64,
                     near_singular_tol::Float64)
    # Top level function for solving the sound-soft IE
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
end # solveSoftIE

function solveSoftIENormalDeriv(pulse_mesh::PulseMesh,
                                excitation_normal_derivative::Function,
                                wavenumber::Number)
    # Top level function for solving the sound-soft IE normal derivative
    @unpack num_elements = pulse_mesh
    println("Filling RHS...")
    RHS = zeros(ComplexF64, num_elements)
    rhs_fill_time = @elapsed rhsNormalDerivFill!(pulse_mesh, excitation_normal_derivative, RHS)
    pulse_mesh.RHS = RHS
    println("  RHS fill time: ", rhs_fill_time)

    matrix_fill_time = @elapsed begin
        if pulse_mesh.Z_factors == lu(ones(1,1))
            println("Filling Matrix...")
            testIntegrandNormalDerivative(r_test, src_idx, test_normal, is_singular) =
                                    scalarGreensNormalDerivativeIntegration(pulse_mesh,
                                                                            src_idx,
                                                                            wavenumber,
                                                                            r_test,
                                                                            test_normal,
                                                                            is_singular)
            Z_matrix = zeros(ComplexF64, num_elements, num_elements)
            matrixNormalDerivFill!(pulse_mesh, testIntegrandNormalDerivative, Z_matrix)
            Z_factors = lu(Z_matrix)
            pulse_mesh.Z_factors = Z_factors
        end
    end
    println("  Matrix fill time: ", matrix_fill_time)

    println("Solving...")
    solve_time = @elapsed source_vec = pulse_mesh.Z_factors \ pulse_mesh.RHS
    println("  Solve time: ", solve_time)

    return(source_vec)
end # solveSoftIENormalDeriv

function solveSoftCFIE(pulse_mesh::PulseMesh,
                       excitation::Function,
                       excitation_normal_derivative::Function,
                       wavenumber::Number,
                       distance_to_edge_tol::Float64,
                       near_singular_tol::Float64,
                       softIE_weight::Float64)
    # Top level function for solving the sound-soft CFIE
    @unpack num_elements = pulse_mesh
    matrix_fill_time = @elapsed begin
        testIntegrand(r_test, src_idx, is_singular) = softIE_weight *
                                                      scalarGreensIntegration(pulse_mesh,
                                                            src_idx, wavenumber, r_test,
                                                            distance_to_edge_tol,
                                                            near_singular_tol, is_singular)
        testIntegrandND(r_test, src_idx, test_normal, is_singular) = (1-softIE_weight) * im *
                                                        scalarGreensNormalDerivativeIntegration(pulse_mesh,
                                                                                                src_idx,
                                                                                                wavenumber,
                                                                                                r_test,
                                                                                                test_normal,
                                                                                                is_singular)
        println("Filling Matrix...")
        z_matrix = zeros(ComplexF64, num_elements, num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        matrixNormalDerivFill!(pulse_mesh, testIntegrandND, z_matrix)
        pulse_mesh.Z_factors = lu(z_matrix)
    end
    println("  Matrix fill time: ", matrix_fill_time)

    rhs_fill_time = @elapsed begin
        println("Filling RHS...")
        rhs_nd_func(x,y,z,normal) = (1-softIE_weight) * im * excitation_normal_derivative(x,y,z,normal)
        rhs_func(x,y,z) = softIE_weight * excitation(x,y,z)
        rhs = zeros(ComplexF64, num_elements)
        rhsNormalDerivFill!(pulse_mesh, rhs_nd_func, rhs)
        rhsFill!(pulse_mesh, rhs_func, rhs)
        pulse_mesh.RHS = rhs
    end
    println("  RHS fill time: ", rhs_fill_time)

    println("Solving...")
    solve_time = @elapsed source_vec = pulse_mesh.Z_factors \ pulse_mesh.RHS
    println("  Solve time: ", solve_time)

    return(source_vec)
end # solveSoftCFIE
