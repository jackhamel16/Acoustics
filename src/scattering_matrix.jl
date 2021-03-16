function calculateScatteringMatrix(max_l, wavenumber, excitation_amplitude, pulse_mesh, distance_to_edge_tol, near_singular_tol)
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

    VJ_matrix = calculateVJMatrix(max_l, wavenumber, excitation_amplitude, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    S_matrix += im/(2*wavenumber) * VJ_matrix

    return(S_matrix)
end

function calculateVJMatrix(max_l, wavenumber, excitation_amplitude, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    @unpack num_elements = pulse_mesh
    num_harmonics = max_l^2 + 2*max_l + 1

    V_matrix = Array{ComplexF64}(undef, num_harmonics, num_elements)
    J_matrix = Array{ComplexF64}(undef, num_elements, num_harmonics)
    harmonic_idx = 1
    for l = 0:max_l
        for m=-l:l
            sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m)
            J, Z, RHS = solveSoftIE(pulse_mesh,
                                 sphericalWaveExcitation,
                                 wavenumber,
                                 distance_to_edge_tol,
                                 near_singular_tol,
                                 true)
            V_matrix[harmonic_idx,:] = RHS
            J_matrix[:,harmonic_idx] = J
            harmonic_idx += 1
        end
    end
    return(V_matrix * J_matrix)
end
