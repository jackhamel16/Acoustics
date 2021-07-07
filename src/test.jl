using BenchmarkTools

include("includes.jl")

@views function computeZEntrySoundSoft(areas,test_quadrature_points,test_quadrature_weights,
                                       wavenumber,
                                       distance_to_edge_tol,
                                       near_singular_tol,
                                       test_ele_idx::Int64,
                                       src_ele_idx::Int64)::ComplexF64
    # Computes the sounds soft IE Z matrix values for the interaction between the test element
    #   at global index test_ele_idx and source element at global index src_ele_idx.
    # Returns the requested entry of the Z matrix
    # @unpack areas,
    #         test_quadrature_points,
    #         test_quadrature_weights = pulse_mesh
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

function computeZEntrySoundSoft(pulse_mesh::PulseMesh,
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
    # area = areas[test_ele_idx]::Float64
    # println(typeof(areas[test_ele_idx]))
    Z_entry = gaussQuadrature(areas[test_ele_idx],
                              testIntegrand,
                              test_quadrature_points[test_ele_idx],
                              test_quadrature_weights)::ComplexF64
    return(Z_entry)
end # computeZEntry

function test()
    wavenumber = 1.0+0.0im
    src_quadrature_rule = gauss7rule
    test_quadrature_rule = gauss7rule
    distance_to_edge_tol = 1e-12
    near_singular_tol = 1.0

    mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
    pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    global_test_idx = 1
    global_src_idx = 1
    @code_warntype computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_idx, global_src_idx)
end

#benchmark no changes
# minimum time:     24.899 μs (0.00% GC)
# median time:      29.101 μs (0.00% GC)
# mean time:        36.775 μs (9.21% GC)
# maximum time:     4.703 ms (98.29% GC)

# added type of z_entry
# minimum time:     25.400 μs (0.00% GC)
# median time:      33.399 μs (0.00% GC)
# mean time:        38.066 μs (8.01% GC)
# maximum time:     3.727 ms (98.29% GC)
