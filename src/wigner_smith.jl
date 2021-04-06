@views function calculateWSMatrix(S_matrix::AbstractArray{ComplexF64,2},
                                  dSdk_matrix::AbstractArray{ComplexF64,2})
    return(im*adjoint(S_matrix)*dSdk_matrix)
end
