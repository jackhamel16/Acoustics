function inner(x)
    return(2*x)
end

function outer(func::Function)
    func(3)
end

function test()
    wavenumber = 1.0+0.0im
    src_quadrature_rule = gauss7rule
    test_quadrature_rule = gauss7rule
    distance_to_edge_tol = 1e-12
    near_singular_tol = 1.0
    approximation_tol = 1e-3
    mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
    pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    num_levels = 3
    octree = createOctree(num_levels, pulse_mesh)
    testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                   wavenumber,
                                                   r_test,
                                                   distance_to_edge_tol,
                                                   near_singular_tol,
                                                   is_singular)
    z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
    matrixFill(pulse_mesh, testIntegrand, z_matrix)
    test_node = octree.nodes[6]; src_node = octree.nodes[7]
    sol_sub_Z = z_matrix[test_node.element_idxs,src_node.element_idxs]
    computeMatrixEntry(test_idx,src_idx)=computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
    @code_warntype computeMatrixACA(computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
end

# doing the below for computeMatrixACA, might let me dispatch on the return type of the matrix entry function if Float64 is the return type
function test2(return_type::Val{T}, x) where T
    y = convert(T, x)
    println(y, " ", typeof(y))
end


@views function computeMatrixACA(func_return_type::Val{T},
                                 computeMatrixEntryFunc::Function,
                                 approximation_tol,
                                 num_rows::Int64,
                                 num_cols::Int64) where T
    # computeMatrixEntryFunc tells ACA how to compute an entry of the matrix it is approximating
    # This function takes only local col and row idxs of the matrix so this function is generalized.
    # approximation_tol: the condition that stop iterations state that the norm of the approximate
    # error matrix must be less than the norm of the approximated matrix given by UV times this tolerance
    # num_rows and num_cols give the dimensions of the matrix
    U = Array{T, 2}(undef, num_rows, 1)
    V = Array{T, 2}(undef, 1, num_cols)
    #initialization
    Ik = 1
    R_tilde_Ik = zeros(T, num_cols)
    R_tilde_Jk = zeros(T, num_rows)
    for col_idx = 1:num_cols
        R_tilde_Ik[col_idx] = computeMatrixEntry(Ik, col_idx)
    end
    Jk = findmax(abs.(R_tilde_Ik))[2]
    V[1,:] = R_tilde_Ik ./ R_tilde_Ik[Jk]
    trans = transpose(R_tilde_Ik ./ R_tilde_Ik[Jk])
    # V = cat(V::Array{T,2}, trans::Transpose{T,Array{T,1}}, dims=1)::Array{T,2}
    for row_idx = 1:num_rows
        R_tilde_Jk[row_idx] = computeMatrixEntry(row_idx, Jk)
    end
    U[:,1] = R_tilde_Jk
    U = cat(U, R_tilde_Jk, dims=2)::Array{T,2}
    # norm_Z_tilde_k = sqrt(norm(U[:,1])^2*norm(V[1,:])^2)
    # k = 2
    # # end initialization
    # max_rank = min(num_rows, num_cols)
    # used_Iks, used_Jks = [Ik], [Jk]
    # while (norm(U[:,k-1])*norm(V[k-1,:]) > approximation_tol * norm_Z_tilde_k) && (k <= max_rank)
    #     # Update Ik
    #     max_Ik_val, Ik = -1.0, 0
    #     for row_idx = 1:num_rows
    #         abs_val = abs(R_tilde_Jk[row_idx])
    #         if (abs_val > max_Ik_val) && ((row_idx in used_Iks) == false)
    #             Ik = row_idx
    #             max_Ik_val = abs_val
    #         end
    #     end
    #     append!(used_Iks, Ik)
    #     R_tilde_Ik = zeros(T, num_cols)
    #     R_tilde_Jk = zeros(T, num_rows)
    #     # compute R_tilde_Ik (the Ikth row of approximate error matrix)
    #     for col_idx = 1:num_cols
    #         R_tilde_Ik[col_idx] = computeMatrixEntryFunc(Ik, col_idx)
    #     end
    #     sum_uv_term = zeros(T, num_cols)
    #     for idx = 1:k-1
    #         sum_uv_term += U[Ik,idx] * V[idx,:] #math here could be implemented incorrectly
    #     end
    #     R_tilde_Ik = R_tilde_Ik - sum_uv_term
    #     # Update Jk
    #     max_Jk_val, Jk = -1.0, 0
    #     for col_idx = 1:num_cols
    #         abs_val = abs(R_tilde_Ik[col_idx])
    #         if (abs_val > max_Jk_val) && ((col_idx in used_Jks) == false)
    #             Jk = col_idx
    #             max_Jk_val = abs_val
    #         end
    #     end
    #     append!(used_Jks, Jk)
    #     V = cat(V, transpose(R_tilde_Ik ./ R_tilde_Ik[Jk])::Transpose{T,Array{T,1}}, dims=1)::Array{T,2}
    #     # compute R_tilde_Jk (the Jkth col of approximate error matrix)
    #     for row_idx = 1:num_rows
    #         R_tilde_Jk[row_idx] = computeMatrixEntryFunc(row_idx, Jk)
    #     end
    #     sum_uv_term = zeros(T, num_rows)
    #     for idx = 1:k-1
    #         sum_uv_term += V[idx,Jk] * U[:,idx]
    #     end
    #     R_tilde_Jk = R_tilde_Jk - sum_uv_term
    #     U = cat(U, R_tilde_Jk, dims=2)::Array{T,2}
    #     # Update norm_Z_tilde_k for while loop
    #     sum_term = 0.0
    #     for j = 1:k-1
    #         sum_term += abs(transpose(U[:,j])*U[:,k]) * abs(transpose(V[j,:])*V[k,:])
    #     end
    #     norm_Z_tilde_k = sqrt(norm_Z_tilde_k^2 + 2*sum_term + norm(U[:,k])^2*norm(V[k,:])^2)
    #     k += 1
    # end # while
    # return(U,V)
end #computeRHSContributionACA
