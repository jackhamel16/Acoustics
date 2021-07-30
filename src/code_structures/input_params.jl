using LinearAlgebra
using Parameters

@with_kw struct ExcitationParams
    type::String = ""
    lambda::Float64 = 0.0
    wavenumber::Number = 0.0
    wavevector::Array{Number, 1} = [0.0]
    amplitude::Float64 = 0.0
    l::Int64 = 0
    m::Int64 = 0
end

@with_kw struct ACAParams
    use_ACA::Bool = false
    num_levels::Int64 = 0
    compression_distance::Float64 = 0.0
    approximation_tol::Float64 = 0.0
end

@with_kw struct InputParams
    mesh_filename::String = ""
    equation::String = ""
    src_quadrature_string::String = ""
    test_quadrature_string::String = ""
    excitation_params::ExcitationParams = ExcitationParams()
    distance_to_edge_tol::Float64 = 1e-12
    near_singular_tol::Float64 = 1.0
    ACA_params::ACAParams = ACAParams()
end

function parseACAParams(input_file_lines::AbstractArray{T,1}) where T <: AbstractString
    ACA_idx = findall(x -> x == "\nACA:",input_file_lines)[1]
    use_ACA = getAttribute(input_file_lines[ACA_idx+1]) == "yes"
    if use_ACA == false
        return(ACAParams())
    else
        num_levels = parse(Int64, getAttribute(input_file_lines[ACA_idx+2]))
        compression_distance = parse(Float64, getAttribute(input_file_lines[ACA_idx+3]))
        approximation_tol = parse(Float64, getAttribute(input_file_lines[ACA_idx+4]))
        return(ACAParams(use_ACA=true,
                         num_levels=num_levels,
                         compression_distance=compression_distance,
                         approximation_tol=approximation_tol))
    end
end #parseACAParams

function parseExcitationParams(input_file_lines::AbstractArray{T,1}) where T <: AbstractString
    excitation_idx = findall(x -> x == "\nexcitation:",input_file_lines)[1]
    type = getAttribute(input_file_lines[excitation_idx+1])
    lambda = parse(Float64, getAttribute(input_file_lines[excitation_idx+2]))
    amplitude = parse(Float64, getAttribute(input_file_lines[excitation_idx+3]))
    wavevector_raw = parse.(Float64, split(getAttribute(input_file_lines[excitation_idx+4]), ' '))
    wavenumber = 2 * pi / lambda
    wavevector = wavenumber .* wavevector_raw ./ norm(wavevector_raw)
    if type == "sphericalwave"
        l = parse(Int64, getAttribute(input_file_lines[excitation_idx+5]))
        m = parse(Int64, getAttribute(input_file_lines[excitation_idx+6]))
        return(ExcitationParams(type=type,
                                lambda=lambda,
                                amplitude=amplitude,
                                wavenumber=wavenumber,
                                wavevector=wavevector,
                                l=l,
                                m=m))
    end
    if type == "planewave"
        return(ExcitationParams(type=type,
                                lambda=lambda,
                                amplitude=amplitude,
                                wavenumber=wavenumber,
                                wavevector=wavevector))
    end
end #parseExcitationParams

function parseInputParams(inputs_filename::String)
    file = open(inputs_filename, "r")
    file_lines = split(read(file, String), "\r")
    mesh_filename = getAttribute(file_lines[1])
    equation = getAttribute(file_lines[2])
    src_quadrature_string = getAttribute(file_lines[3])
    test_quadrature_string = getAttribute(file_lines[4])
    distance_to_edge_tol = parse(Float64, getAttribute(file_lines[5]))
    near_singular_tol = parse(Float64, getAttribute(file_lines[6]))
    excitation_params = parseExcitationParams(file_lines)
    ACA_params = parseACAParams(file_lines)
    return(InputParams(mesh_filename=mesh_filename,
                       equation=equation,
                       src_quadrature_string=src_quadrature_string,
                       test_quadrature_string=test_quadrature_string,
                       excitation_params=excitation_params,
                       distance_to_edge_tol=distance_to_edge_tol,
                       near_singular_tol=near_singular_tol,
                       ACA_params=ACA_params))
end # parseInputParams

function getAttribute(input_file_line::AbstractString)
    # separates extra wording and symbols from input parameter
    return(strip(split(input_file_line, ":")[2], ' '))
end
