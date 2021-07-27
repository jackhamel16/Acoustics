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

@with_kw struct InputParams
    mesh_filename::String = ""
    src_quadrature_rule::String = ""
    test_quadrature_rule::String = ""
    excitation_params::ExcitationParams = ExcitationParams()
    distance_to_edge_tol::Float64 = 1e-12
    near_singular_tol::Float64 = 1.0
end

@with_kw struct ACAParams
    num_levels::Int64 = 0
    compression_distance::Float64 = 1.5 # default compresses when nodes dont shre edge/corner
    approximation_tol::Float64 = 1e-4
end
