# dependencies: none
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

@with_kw struct WignerSmithParams
    lambda::Float64 = 0.0
    wavenumber::Number = 0.0
    max_l::Int64 = 0
    mode_idx::Int64 = 0
end

@with_kw struct InputParams
    mesh_filename::String = ""
    equation::String = ""
    CFIE_weight::Float64 = 0.0
    src_quadrature_string::String = ""
    test_quadrature_string::String = ""
    distance_to_edge_tol::Float64 = 1e-12
    near_singular_tol::Float64 = 1.0
    excitation_params::ExcitationParams = ExcitationParams()
    ACA_params::ACAParams = ACAParams()
    WS_params::WignerSmithParams = WignerSmithParams()
end

function parseACAParams(input_file_lines::AbstractArray{T,1}) where T <: AbstractString
    # parses the string elements of input_file_lines for the ACA settings provided
    # in the input file
    ACA_idxs = findall(x -> x == "\nACA:",input_file_lines)
    if length(ACA_idxs) == 0
        return(ACAParams())
    else
        ACA_idx = ACA_idxs[1]
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
    end
end #parseACAParams

function parseExcitationParams(input_file_lines::AbstractArray{T,1}) where T <: AbstractString
    # parses the string elements of input_file_lines for the excitation settings
    # provided in the input file
    excitation_idxs = findall(x -> x == "\nexcitation:",input_file_lines)
    if length(excitation_idxs) == 0
        return(ExcitationParams())
    else
        excitation_idx = excitation_idxs[1]
        type = getAttribute(input_file_lines[excitation_idx+1])
        lambda = parse(Float64, getAttribute(input_file_lines[excitation_idx+2]))
        amplitude = parse(Float64, getAttribute(input_file_lines[excitation_idx+3]))
        wavenumber = 2 * pi / lambda
        if type == "sphericalwave"
            l = parse(Int64, getAttribute(input_file_lines[excitation_idx+4]))
            m = parse(Int64, getAttribute(input_file_lines[excitation_idx+5]))
            return(ExcitationParams(type=type,
                                    lambda=lambda,
                                    amplitude=amplitude,
                                    wavenumber=wavenumber,
                                    l=l,
                                    m=m))
        elseif type == "planewave"
            wavevector_raw = parse.(Float64, split(getAttribute(input_file_lines[excitation_idx+4]), ' '))
            wavevector = wavenumber .* wavevector_raw ./ norm(wavevector_raw)
            return(ExcitationParams(type=type,
                                    lambda=lambda,
                                    amplitude=amplitude,
                                    wavenumber=wavenumber,
                                    wavevector=wavevector))
        end
    end
end #parseExcitationParams

function parseWignerSmithParams(input_file_lines::AbstractArray{T,1}) where T <: AbstractString
    # parses the string elements of input_file_lines for the Wigner Smith settings
    # provided in the input file
    WS_idxs = findall(x -> x == "\nWigner Smith:",input_file_lines)
    if length(WS_idxs) == 0
        return(WignerSmithParams())
    else
        WS_idx = WS_idxs[1]
        lambda = parse(Float64, getAttribute(input_file_lines[WS_idx+1]))
        max_l = parse(Int64, getAttribute(input_file_lines[WS_idx+2]))
        mode_idx = parse(Int64, getAttribute(input_file_lines[WS_idx+3]))
        return(WignerSmithParams(lambda=lambda,
                                 wavenumber = 2 * pi / lambda,
                                 max_l=max_l,
                                 mode_idx=mode_idx))
    end
end #parseWignerSmithParams

function parseInputParams(inputs_filename::String)
    # Top-level parsing function.  Provided with the input filename as string, returns
    # an InputParams instance containg all the information in inputs_filename
    file = open(inputs_filename, "r")
    file_lines = split(read(file, String), "\r")
    line_idx = 1
    mesh_filename = getAttribute(file_lines[line_idx])
    equation = getAttribute(file_lines[line_idx+1])
    if equation == "sound soft CFIE"
        CFIE_weight = parse(Float64, getAttribute(file_lines[line_idx+2]))
        line_idx += 1
    else
        CFIE_weight = 0.0
    end
    src_quadrature_string = getAttribute(file_lines[line_idx+2])
    test_quadrature_string = getAttribute(file_lines[line_idx+3])
    distance_to_edge_tol = parse(Float64, getAttribute(file_lines[line_idx+4]))
    near_singular_tol = parse(Float64, getAttribute(file_lines[line_idx+5]))
    excitation_params = parseExcitationParams(file_lines)
    ACA_params = parseACAParams(file_lines)
    WS_params = parseWignerSmithParams(file_lines)
    return(InputParams(mesh_filename=mesh_filename,
                       equation=equation,
                       CFIE_weight=CFIE_weight,
                       src_quadrature_string=src_quadrature_string,
                       test_quadrature_string=test_quadrature_string,
                       excitation_params=excitation_params,
                       distance_to_edge_tol=distance_to_edge_tol,
                       near_singular_tol=near_singular_tol,
                       ACA_params=ACA_params,
                       WS_params=WS_params))
end # parseInputParams

function getAttribute(input_file_line::AbstractString)
    # separates extra wording and symbols from input parameter
    return(strip(split(input_file_line, ":")[2], ' '))
end
