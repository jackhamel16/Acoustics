@views function calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    num_harmonics = max_l^2 + 2*max_l + 1
    S_matrix = zeros(ComplexF64, num_harmonics, num_harmonics)

    for tl=0:max_l
        pl = tl
        for tm=-tl:tl
            pm = -tm
            t_idx = tl^2 + tl + tm + 1
            p_idx = pl^2 + pl + pm + 1
            S_matrix[t_idx, p_idx] = (-1)^tm
        end
    end

    VJ_matrix = calculateVJMatrix(max_l, num_harmonics, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    S_matrix += im/(2*wavenumber) * VJ_matrix

    return(S_matrix)
end

@views function calculateVJMatrix(max_l, num_harmonics, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    @unpack num_elements = pulse_mesh
    excitation_amplitude = 1.0

    V_matrix = Array{ComplexF64}(undef, num_harmonics, num_elements)
    J_matrix = Array{ComplexF64}(undef, num_elements, num_harmonics)
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                   wavenumber,
                                                   r_test,
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    println("Filling Z Matrix for Scattering Matrix...")
    z_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill(pulse_mesh, testIntegrand, z_matrix)
    z_factors = factorize(z_matrix)
    println("Filling V and J Matrices for Scattering Matrix...")
    harmonic_idx = 1
    for l = 0:max_l
        for m=-l:l
            sphericalWaveExcitation(x_test, y_test, z_test) = 2 * wavenumber *
                                                              sphericalWave(excitation_amplitude,
                                                                            real(wavenumber),
                                                                            [x_test,y_test,z_test],
                                                                            l,
                                                                            m)
            rhs = zeros(ComplexF64, num_elements)
            rhsFill(pulse_mesh, sphericalWaveExcitation, rhs)
            V_matrix[harmonic_idx,:] = rhs
            J = z_factors \ rhs
            J_matrix[:,harmonic_idx] = J
            harmonic_idx += 1
        end
    end
    return(V_matrix * J_matrix)
end
