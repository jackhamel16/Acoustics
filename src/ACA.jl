using LinearAlgebra
using Parameters

@views @fastmath @inbounds function computeMatrixACA(func_return_type::Val{T},
                                 computeMatrixEntryFunc::Function,
                                 approximation_tol,
                                 num_rows::Int64,
                                 num_cols::Int64) where T
    # This function follows the algorithm described in "The Adaptive Cross Approximation Algorithm
    #   for Accelerated Method of Moments Computations of EMC Problems" by Zhao, Vouvakis and Lee
    #   for decomposing a rank deficient subset of the Z matrix into U and V matrices to accelerate
    #   matrix-vector products with the Z matrix.  In my applications this is used on groups of
    #   interactions for well-separated elements.
    # func_return type is Val(DataType) where DataType is the type returned by computeMatrixEntryFunc
    #   This is necessary so computeMatrixACA can dispatch on the type avoiding type instabilities
    # computeMatrixEntryFunc tells ACA how to compute an entry of the matrix it is approximating
    #   computeMatrixEntryFunc takes only local col and row idxs of the matrix as args for generalization.
    # approximation_tol: the condition that stop iterations state that the norm of the approximate
    #   error matrix must be less than the norm of the approximated matrix given by UV times this tolerance
    # num_rows and num_cols give the dimensions of the matrix
    U = Array{T, 2}(undef, num_rows, 1)
    V = Array{T, 2}(undef, 1, num_cols)
    #initialization
    Ik = 1
    R_tilde_Ik = zeros(T, num_cols)
    R_tilde_Jk = zeros(T, num_rows)
    for col_idx = 1:num_cols
        R_tilde_Ik[col_idx] = computeMatrixEntryFunc(Ik, col_idx)
    end
    Jk = findmax(abs.(R_tilde_Ik))[2]
    V[1,:] = R_tilde_Ik ./ R_tilde_Ik[Jk]
    for row_idx = 1:num_rows
        R_tilde_Jk[row_idx] = computeMatrixEntryFunc(row_idx, Jk)
    end
    U[:,1] = R_tilde_Jk
    norm_Z_tilde_k = sqrt(norm(U[:,1])^2*norm(V[1,:])^2)
    k = 2
    # end initialization
    max_rank = min(num_rows, num_cols)
    used_Iks, used_Jks = [Ik], [Jk]
    while (norm(U[:,k-1])*norm(V[k-1,:]) > approximation_tol * norm_Z_tilde_k) && (k <= max_rank)
        # Update Ik
        max_Ik_val, Ik = -1.0, 0
        for row_idx = 1:num_rows
            abs_val = abs(R_tilde_Jk[row_idx])
            if (abs_val > max_Ik_val) && ((row_idx in used_Iks) == false)
                Ik = row_idx
                max_Ik_val = abs_val
            end
        end
        append!(used_Iks, Ik)
        R_tilde_Ik = zeros(T, num_cols)
        R_tilde_Jk = zeros(T, num_rows)
        # compute R_tilde_Ik (the Ikth row of approximate error matrix)
        for col_idx = 1:num_cols
            R_tilde_Ik[col_idx] = computeMatrixEntryFunc(Ik, col_idx)
        end
        sum_uv_term = zeros(T, num_cols)
        for idx = 1:k-1
            sum_uv_term += U[Ik,idx] * V[idx,:] #math here could be implemented incorrectly
        end
        R_tilde_Ik = R_tilde_Ik - sum_uv_term
        # Update Jk
        max_Jk_val, Jk = -1.0, 0
        for col_idx = 1:num_cols
            abs_val = abs(R_tilde_Ik[col_idx])
            if (abs_val > max_Jk_val) && ((col_idx in used_Jks) == false)
                Jk = col_idx
                max_Jk_val = abs_val
            end
        end
        append!(used_Jks, Jk)
        V = [V; transpose(R_tilde_Ik ./ R_tilde_Ik[Jk])]
        # compute R_tilde_Jk (the Jkth col of approximate error matrix)
        for row_idx = 1:num_rows
            R_tilde_Jk[row_idx] = computeMatrixEntryFunc(row_idx, Jk)
        end
        sum_uv_term = zeros(T, num_rows)
        for idx = 1:k-1
            sum_uv_term += V[idx,Jk] * U[:,idx]
        end
        R_tilde_Jk = R_tilde_Jk - sum_uv_term
        U = [U R_tilde_Jk]
        # Update norm_Z_tilde_k for while loop
        sum_term = 0.0
        for j = 1:k-1
            sum_term += abs(transpose(U[:,j])*U[:,k]) * abs(transpose(V[j,:])*V[k,:])
        end
        norm_Z_tilde_k = sqrt(norm_Z_tilde_k^2 + 2*sum_term + norm(U[:,k])^2*norm(V[k,:])^2)
        k += 1
    end # while
    return(U,V)
end #computeRHSContributionACA

@views function computeZEntrySoundSoft(pulse_mesh::PulseMesh,
                                       wavenumber,
                                       distance_to_edge_tol,
                                       near_singular_tol,
                                       test_ele_idx::Int64,
                                       src_ele_idx::Int64)::ComplexF64
    # Computes the sounds soft IE Z matrix values for the interaction between the test element
    #   at global index test_ele_idx and source element at global index src_ele_idx.
    # Returns the requested entry of the Z matrix
    @unpack areas,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    is_singular = test_ele_idx == src_ele_idx
    testIntegrand(x,y,z) = scalarGreensIntegration(pulse_mesh,
                                                   src_ele_idx,
                                                   wavenumber,
                                                   [x,y,z],
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    Z_entry = gaussQuadrature(areas[test_ele_idx],
                              testIntegrand,
                              test_quadrature_points[test_ele_idx],
                              test_quadrature_weights)::ComplexF64
    return(Z_entry)
end # computeZEntry

@views function computeZEntrySoundSoft(pulse_mesh::PulseMesh,
                                       test_node::Node,
                                       src_node::Node,
                                       wavenumber,
                                       distance_to_edge_tol,
                                       near_singular_tol,
                                       test_idx::Int64,
                                       src_idx::Int64)
    # Alternative implementation of computeZEntrySoundSoft that can be wrapped up to give to
    #   computeMatrixACA easily. Rather than pass in global element idxs, the function assumes
    #   you are computing a sub-matrix of Z for all elements between test_node and src_node.
    #   test_idx and src_idx are local element idxs for elements contained in test_node and
    #   src_node, respectively.  The function assumes test_idx and src_idx are valid indices
    #   (i.e. not larger than the total number of elements in test_node or src_node)
    # Returns the requested entry of the sub-Z matrix
    global_test_idx = test_node.element_idxs[test_idx]
    global_src_idx = src_node.element_idxs[src_idx]
    Z_entry = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_idx, global_src_idx)
    return(Z_entry)
end
