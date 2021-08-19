# Dependencies: solve.jl

@views function calculateScatteringMatrix(max_l::Int64, wavenumber, pulse_mesh::PulseMesh, distance_to_edge_tol, near_singular_tol)
    @unpack num_elements,
            Z_factors = pulse_mesh
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
    Vs_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
    Js = Array{ComplexF64}(undef, num_elements, num_harmonics)
    harmonic_idx = 1
    for l = 0:max_l
        for m=-l:l
            Vs_trans[harmonic_idx,:] = calculateVlm(pulse_mesh, wavenumber, l, m)
            Js[:,harmonic_idx] = Z_factors \ Vs_trans[harmonic_idx,:]
            harmonic_idx += 1
        end
    end
    S_matrix += im/(2*wavenumber) * Vs_trans * Js
    return(S_matrix)
end

@views function calculateScatteringMatrixACA(max_l::Int64, wavenumber, pulse_mesh::PulseMesh, octree::Octree, distance_to_edge_tol, near_singular_tol)
    @unpack num_elements = pulse_mesh
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
    Vs_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
    Js = Array{ComplexF64}(undef, num_elements, num_harmonics)
    harmonic_idx = 1
    for l = 0:max_l
        for m=-l:l
            Vs_trans[harmonic_idx,:] = calculateVlm(pulse_mesh, wavenumber, l, m)
            fullMatvecWrapped(J) = fullMatvecACA(pulse_mesh, octree, J)
            fullMatvecLinearMap = LinearMap(fullMatvecWrapped, num_elements)
            sources = zeros(ComplexF64, num_elements)
            num_iters = gmres!(sources, fullMatvecLinearMap, Vs_trans[harmonic_idx,:], log=true)[2]
            println("GMRES ", string(num_iters))
            Js[:,harmonic_idx] = sources
            harmonic_idx += 1
        end
    end
    S_matrix += im/(2*wavenumber) * Vs_trans * Js
    return(S_matrix)
end

@views function calculateScatteringMatrixDerivative(max_l::Int64, num_harmonics::Int64, wavenumber, pulse_mesh::PulseMesh, distance_to_edge_tol, near_singular_tol)
    @unpack num_elements,
            Z_factors = pulse_mesh
    dZdk = calculateZKDerivMatrix(pulse_mesh, wavenumber)
    Vs_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
    Js = Array{ComplexF64}(undef, num_elements, num_harmonics)
    dVsdk_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
    harmonic_idx = 1
    for l = 0:max_l
        for m=-l:l
            Vs_trans[harmonic_idx,:] = calculateVlm(pulse_mesh, wavenumber, l, m)
            Js[:,harmonic_idx] = Z_factors \ Vs_trans[harmonic_idx,:]
            dVsdk_trans[harmonic_idx, :] = calculateVlmKDeriv(pulse_mesh, wavenumber, l, m)
            harmonic_idx += 1
        end
    end
    j_over_2k = im*0.5 / wavenumber
    term1 = -j_over_2k/wavenumber * Vs_trans* Js
    term2 = j_over_2k * dVsdk_trans * Js
    # old_term3 = j_over_2k * (Vs_trans / z_factors) * (transpose(dVsdk_trans) - dZdk*Js) # true without full galerkin
    term3 = j_over_2k*(transpose(Js)*transpose(dVsdk_trans) - transpose(Js)*dZdk*Js) # requires full galerkin testing
    return(term1 + term2 + term3)
end

@views function calculateScatteringMatrixDerivativeACA(max_l::Int64, num_harmonics::Int64, wavenumber, pulse_mesh::PulseMesh, octree::Octree, distance_to_edge_tol, near_singular_tol)
    @unpack num_elements = pulse_mesh
    dZdk_octree = copy(octree)
    fillOctreedZdkMatricesSoundSoft!(pulse_mesh, dZdk_octree, wavenumber,
                                     compression_distance, ACA_approximation_tol) # could maybe speed this up not having to recheck whether to compress or not
    Vs_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
    Js = Array{ComplexF64}(undef, num_elements, num_harmonics)
    dVsdk_trans = Array{ComplexF64}(undef, num_harmonics, num_elements)
    dZdk_times_Js = Array{ComplexF64}(undef, num_elements, num_harmonics)
    harmonic_idx = 1
    for l = 0:max_l
        for m=-l:l
            Vs_trans[harmonic_idx,:] = calculateVlm(pulse_mesh, wavenumber, l, m)
            fullMatvecWrapped(J) = fullMatvecACA(pulse_mesh, octree, J)
            fullMatvecLinearMap = LinearMap(fullMatvecWrapped, num_elements)
            sources = zeros(ComplexF64, num_elements)
            num_iters = gmres!(sources, fullMatvecLinearMap, Vs_trans[harmonic_idx,:], log=true)[2]
            println("GMRES ", string(num_iters))
            Js[:,harmonic_idx] = sources
            dVsdk_trans[harmonic_idx, :] = calculateVlmKDeriv(pulse_mesh, wavenumber, l, m)
            dZdk_times_Js[:,harmonic_idx] = computeZJMatVec(pulse_mesh, dZdk_octree, Js[:,num_harmonics])
            harmonic_idx += 1
        end
    end
    j_over_2k = im*0.5 / wavenumber
    term1 = -j_over_2k/wavenumber * Vs_trans* Js
    term2 = j_over_2k * dVsdk_trans * Js
    # old_term3 = j_over_2k * (Vs_trans / z_factors) * (transpose(dVsdk_trans) - dZdk*Js) # true without full galerkin
    term3 = j_over_2k*(transpose(Js)*transpose(dVsdk_trans) - transpose(Js)*dZdk_times_Js) # requires full galerkin testing
    return(term1 + term2 + term3)
end

@views function calculateVlm(pulse_mesh::PulseMesh, wavenumber, l, m)
    # calculates the rhs vector for a spherical wave excitation of degree l and order m
    @unpack num_elements = pulse_mesh
    sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(2 * wavenumber,
                                                                    real(wavenumber),
                                                                    [x_test,y_test,z_test],
                                                                    l,
                                                                    m)
    Vlm = zeros(ComplexF64, num_elements)
    rhsFill!(pulse_mesh, sphericalWaveExcitation, Vlm)
    return(Vlm)
end

@views function calculateVlmKDeriv(pulse_mesh::PulseMesh, wavenumber, l, m)
    # calculates the derivative with respect to k of the rhs vector for a
    # spherical wave excitation of degree l and order m
    @unpack num_elements = pulse_mesh
    dVlmdk = zeros(ComplexF64, num_elements)
    sphericalWaveKDerivIntegrand(x,y,z) = sphericalWaveKDerivative(wavenumber, [x,y,z], l, m)
    rhsFill!(pulse_mesh, sphericalWaveKDerivIntegrand, dVlmdk)
    return(dVlmdk)
end

@views function calculateZKDerivMatrix(pulse_mesh::PulseMesh, wavenumber)
    # Computes the derivative with respect to k of the Z matrix using sound-soft
    # IE to construct the scattering matrix
    @unpack num_elements = pulse_mesh
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensKDerivIntegration(pulse_mesh,
                                                                                src_idx,
                                                                                wavenumber,
                                                                                r_test,
                                                                                is_singular)
    dZdk_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill!(pulse_mesh, testIntegrand, dZdk_matrix)
    return(dZdk_matrix)
end

@views function calculateZKDerivMatrixACA(pulse_mesh::PulseMesh, wavenumber)
    # Computes the derivative with respect to k of the Z matrix using sound-soft
    # IE to construct the scattering matrix
    @unpack num_elements = pulse_mesh
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensKDerivIntegration(pulse_mesh,
                                                                                src_idx,
                                                                                wavenumber,
                                                                                r_test,
                                                                                is_singular)
    dZdk_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill!(pulse_mesh, testIntegrand, dZdk_matrix)
    return(dZdk_matrix)
end

@views function calculateZMatrix(pulse_mesh::PulseMesh, wavenumber, distance_to_edge_tol, near_singular_tol)
    # Computes the Z matrix using sound-soft IE to construct the scattering matrix
    @unpack num_elements = pulse_mesh
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                   wavenumber,
                                                   r_test,
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    Z_matrix = zeros(ComplexF64, num_elements, num_elements)
    matrixFill!(pulse_mesh, testIntegrand, Z_matrix)
    return(Z_matrix)
end
