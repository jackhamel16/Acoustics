function calculateScatteringMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    num_harmonics = max_l^2 + 2*max_l + 1
    S_matrix = zeros(ComplexF64, num_harmonics, num_harmonics)
    # might be able to just loop through when delta function is 1 to save time
    for tl=0:max_l
        for tm=-tl:tl
            t_idx = tl^2 + tl + tm + 1 # try replacing with a counter, might be faster
            for pl=0:max_l
                for pm=-pl:pl
                    p_idx = pl^2 + pl + pm + 1 # try replacing with a counter, might be faster
                    S_matrix[t_idx, p_idx] = (-1)^tm * ((tl==pl) & (tm==-pm))
                end # pm
            end # pl
        end # tm
    end # tl

    VJ_matrix = calculateVJMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    S_matrix += im/(2*wavenumber) * VJ_matrix

    return(S_matrix)
end

function calculateVJMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    @unpack num_elements = pulse_mesh
    excitation_amplitude = 1.0
    num_harmonics = max_l^2 + 2*max_l + 1

    V_matrix = Array{ComplexF64}(undef, num_harmonics, num_elements)
    J_matrix = Array{ComplexF64}(undef, num_elements, num_harmonics)
    harmonic_idx = 1
    # compute Z
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                   wavenumber,
                                                   r_test,
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    println("Filling Z Matrix for Scattering Matrix...")
    z_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill(pulse_mesh, testIntegrand, z_matrix)
    # facotrize Z
    z_factors = factorize(z_matrix)
    # for each RHS
    println("Filling V and J Matrices for Scattering Matrix...")
    for l = 0:max_l
        for m=-l:l
            # compute RHS
            sphericalWaveExcitation(x_test, y_test, z_test) = 2 * wavenumber * sphericalWave(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m)
            rhs = zeros(ComplexF64, num_elements)
            rhsFill(pulse_mesh, sphericalWaveExcitation, rhs)
            # Store RHS in V matrix
            V_matrix[harmonic_idx,:] = rhs
            # compute J and sotre in J matrix
            J = z_factors \ rhs
            # J, Z, RHS = solveSoftIE(pulse_mesh,
            #                      sphericalWaveExcitation,
            #                      wavenumber,
            #                      distance_to_edge_tol,
            #                      near_singular_tol,
            #                      true)
            # V_matrix[harmonic_idx,:] = RHS
            J_matrix[:,harmonic_idx] = J
            harmonic_idx += 1
        end
    end
    return(V_matrix * J_matrix)
end
