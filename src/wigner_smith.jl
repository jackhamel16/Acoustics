# Dependencies: scattering_matrix.jl solve.jl

################################################################################
# This file serves two purposes:                                               #
#     1) To compute the Wigner-Smith matrix, Q, with or without ACA.           #
#     2) To diagonalize Q and use the time-delays to build an excitation and   #
#        solve any of the avilable integral equations with that excitation     #
#        with or without using ACA.                                            #
################################################################################

using LinearAlgebra
using Parameters

@views function calculateWSMatrix(S_matrix::AbstractArray{ComplexF64,2},
                                  dSdk_matrix::AbstractArray{ComplexF64,2})
    return(im*adjoint(S_matrix)*dSdk_matrix)
end # function calculateWSMatrix

@views function calculateWSMatrix(max_l::Int64,
                                  wavenumber,
                                  pulse_mesh::PulseMesh,
                                  distance_to_edge_tol,
                                  near_singular_tol)
    num_harmonics = max_l^2 + 2*max_l + 1
    println("Filling Z Matrix...")
    Z_matrix = calculateZMatrix(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol)
    pulse_mesh.Z_factors = lu(Z_matrix)
    println("Calculating Scattering Matrix...")
    S, Js = calculateScatteringMatrix(max_l, wavenumber, pulse_mesh,
                                  distance_to_edge_tol, near_singular_tol)
    println("Calculating Scattering Matrix Derivative...")
    dSdk = calculateScatteringMatrixDerivative(max_l, num_harmonics, wavenumber,
                                               pulse_mesh, Js, distance_to_edge_tol,
                                               near_singular_tol)
    println("Calculating Wigner-Smith Matrix...")
    Q = calculateWSMatrix(S, dSdk)
    return(Q)
end # function calculateWSMatrix

@views function calculateWSMatrixACA(max_l::Int64,
                                  wavenumber,
                                  pulse_mesh::PulseMesh,
                                  octree::Octree,
                                  distance_to_edge_tol,
                                  near_singular_tol,
                                  compression_distance,
                                  ACA_approximation_tol)
    num_harmonics = max_l^2 + 2*max_l + 1
    println("Filling Z and dZ/dk matrices...")
    fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol,
                                  near_singular_tol, compression_distance, ACA_approximation_tol)
    fillOctreedZdkMatricesSoundSoft!(pulse_mesh, octree, wavenumber,
                                     compression_distance, ACA_approximation_tol)
    # mkdir("scattering_matrix_GMRES_residuals")
    println("Calculating Scattering Matrix...")
    S, Js = calculateScatteringMatrixACA(max_l, wavenumber, pulse_mesh, octree,
                                  distance_to_edge_tol, near_singular_tol)
    println("Calculating Scattering Matrix Derivative...")
    dSdk = calculateScatteringMatrixDerivativeACA(max_l, num_harmonics, wavenumber,
                                                  pulse_mesh, Js, octree)
    println("Calculating Wigner-Smith Matrix...")
    Q = calculateWSMatrix(S, dSdk)
    return(Q)
end # function calculateWSMatrix

@views function solveWSModeSoft(max_l::Int64,
                            mode_idxs::AbstractArray{Int64,1},
                            wavenumber,
                            pulse_mesh::PulseMesh,
                            distance_to_edge_tol,
                            near_singular_tol)
    # Solves sound-soft IE with an incident field for the WS mode indicated by mode_idx
    Q = calculateWSMatrix(max_l, wavenumber, pulse_mesh, distance_to_edge_tol, near_singular_tol)
    eigen_Q = eigen(Q)
    writeWSTimeDelays(eigen_Q.values)
    sources_WS = Array{Array{ComplexF64, 1}, 1}(undef, length(mode_idxs))
    for local_mode_idx = 1:length(mode_idxs)
        mode_idx = mode_idxs[local_mode_idx]
        mode_vector = eigen_Q.vectors[:,mode_idx]
        excitationWSMode(x,y,z) = sphericalWaveWSMode(x, y, z, max_l, wavenumber, mode_vector)
        println("Solving at WS Mode ", mode_idx,"...")
        println("Wigner-Smith Time Delay = ", eigen_Q.values[mode_idx])
        sources_WS[local_mode_idx] = solveSoftIE(pulse_mesh,
                                 excitationWSMode,
                                 wavenumber,
                                 distance_to_edge_tol,
                                 near_singular_tol)
    end
    return(sources_WS)
end # function solveWSModeSoft

@views function solveWSModeSoftACA(max_l::Int64,
                               mode_idxs::AbstractArray{Int64,1},
                               wavenumber,
                               pulse_mesh::PulseMesh,
                               distance_to_edge_tol,
                               near_singular_tol,
                               num_levels,
                               compression_distance,
                               ACA_approximation_tol)
    # Solves sound-soft IE with an incident field for the WS mode indicated by mode_idx
    #   using ACA
    octree = createOctree(num_levels, pulse_mesh)
    Q = calculateWSMatrixACA(max_l, wavenumber, pulse_mesh, octree, distance_to_edge_tol,
                             near_singular_tol, compression_distance, ACA_approximation_tol)
    eigen_Q = eigen(Q)
    writeWSTimeDelays(eigen_Q.values)
    sources_WS = Array{Array{ComplexF64, 1}, 1}(undef, length(mode_idxs))
    for local_mode_idx = 1:length(mode_idxs)
        mode_idx = mode_idxs[local_mode_idx]
        mode_vector = eigen_Q.vectors[:,mode_idx]
        excitationWSMode(x,y,z) = sphericalWaveWSMode(x, y, z, max_l, wavenumber, mode_vector)
        println("Solving at WS Mode ", mode_idx,"...")
        println("Wigner-Smith Time Delay = ", eigen_Q.values[mode_idx])
        sources_WS[local_mode_idx] = solveSoftIEACA(pulse_mesh, octree, num_levels, excitationWSMode,
                                         wavenumber, distance_to_edge_tol, near_singular_tol,
                                         compression_distance, ACA_approximation_tol)
    end
    metrics = computeACAMetrics(pulse_mesh.num_elements, octree)
    return(sources_WS, octree, metrics)
end # function solveWSModeSoftACA

@views function solveWSModeSoftCFIEACA(max_l::Int64,
                               mode_idxs::AbstractArray{Int64,1},
                               wavenumber,
                               pulse_mesh::PulseMesh,
                               distance_to_edge_tol,
                               near_singular_tol,
                               softIE_weight,
                               num_levels,
                               compression_distance,
                               ACA_approximation_tol)
    # Solves sound-soft CFIE with an incident field for the WS mode indicated by
    #   mode_idx using ACA
    octree = createOctree(num_levels, pulse_mesh)
    Q = calculateWSMatrixACA(max_l, wavenumber, pulse_mesh, octree, distance_to_edge_tol,
                             near_singular_tol, compression_distance, ACA_approximation_tol)
    eigen_Q = eigen(Q)
    writeWSTimeDelays(eigen_Q.values)
    sources_WS = Array{Array{ComplexF64, 1}, 1}(undef, length(mode_idxs))
    for local_mode_idx = 1:length(mode_idxs)
        mode_idx = mode_idxs[local_mode_idx]
        mode_vector = eigen_Q.vectors[:,mode_idx]
        excitationWSMode(x,y,z) = sphericalWaveWSMode(x, y, z, max_l, wavenumber, mode_vector)
        excitationNormalDerivWSMode(x,y,z,normal) = sphericalWaveNormalDerivWSMode(x, y, z, max_l, wavenumber, normal, mode_vector)
        println("Solving at WS Mode ", mode_idx,"...")
        println("Wigner-Smith Time Delay = ", eigen_Q.values[mode_idx])
        sources_WS[local_mode_idx] = solveSoftCFIEACA(pulse_mesh, octree, num_levels, excitationWSMode,
                                      excitationNormalDerivWSMode,
                                         wavenumber, softIE_weight, distance_to_edge_tol, near_singular_tol,
                                         compression_distance, ACA_approximation_tol)
    end
    metrics = computeACAMetrics(pulse_mesh.num_elements, octree)
    return(sources_WS, octree, metrics)
end # function solveWSModeSoftCFIEACA

function sphericalWaveWSMode(x, y, z, max_l::Int64, wavenumber, mode_vector::AbstractArray{T,1}) where T
    # Sums spherical incident waves weighted by eigenvector elements of the desired WS mode
    excitation_amplitude = 2 * real(wavenumber)
    total_wave = 0
    harmonic_idx = 1
    for l = 0:max_l
        for m=-l:l
            total_wave += mode_vector[harmonic_idx] * sphericalWave(excitation_amplitude, real(wavenumber), [x,y,z], l, m)
            harmonic_idx += 1
        end
    end
    total_wave
end # function sphericalWaveWSMode

function sphericalWaveNormalDerivWSMode(x, y, z, max_l::Int64, wavenumber, normal::Array{Float64,1}, mode_vector::AbstractArray{T,1}) where T
    # sums normal derivatives of spherical incident waves weighted by eigenvector
    #   elements of the desired WS mode
    excitation_amplitude = 2 * real(wavenumber)
    total_wave = 0
    harmonic_idx = 1
    for l = 0:max_l
        for m=-l:l
            total_wave += mode_vector[harmonic_idx] * sphericalWaveNormalDerivative(excitation_amplitude, real(wavenumber), [x,y,z], l, m, normal)
            harmonic_idx += 1
        end
    end
    total_wave
end # function sphericalWaveWSMode

function writeWSTimeDelays(time_delays::AbstractArray{T, 1}) where T
    # no unit test for this
    num_modes = length(time_delays)
    output_file = open("Wigner_Smith_time_delays.txt", "w")
    println(output_file, "WS Mode    Time Delay")
    for mode_idx in 1:num_modes
        println(output_file, string(mode_idx, " ", time_delays[mode_idx]))
    end
    close(output_file)
end # function writeWSTimeDelays

function readWSMatrix(WS_matrix_filename::String)
    # reads the WS matrix, Q, from the fiel generated by writeWSMatrix
    input_file = open(WS_matrix_filename, "r")
    file_lines = strip.(split(read(input_file, String), "\n"))
    num_modes = length(file_lines) - 1 # last line is empty
    Q = Array{ComplexF64, 2}(undef, num_modes, num_modes)
    for row_idx = 1:num_modes
        Q_row_data = transpose(reshape(parse.(Float64, split(file_lines[row_idx])), 2, num_modes))
        for col_idx = 1:num_modes
            Q[row_idx,col_idx] = Q_row_data[col_idx,1] + Q_row_data[col_idx,2] * 1im
        end
    end
    close(input_file)
    return(Q)
end # readWSMatrix

function writeWSMatrix(Q::AbstractArray{T, 2}) where T
    # writes the WS matrix, Q, to a txt file row by row. Each element separated by spaces.
    #                                            File Format:
    #                                          Re(q11) Im(q11) ... Re(q1n) Im(q1n)
    # Q = [q11 ... 11n; ...; qn1 ... qnn] -->        ...       ...       ...
    #                                          Re(qn1) Im(qn1) ... Re(qnn) Im(qnn)
    #
    output_file = open("Wigner_Smith_matrix.txt", "w")
    for row_idx in 1:size(Q,1)
        row_string = ""
        for col_idx = 1:size(Q,2)
            val = Q[row_idx, col_idx]
            row_string = string(row_string, real(val), " ", imag(val), " ")
        end
        println(output_file, row_string)
        # println(row_string)
    end
    close(output_file)
end # writeWSMatrix
