function computeRHSContribution(pulse_mesh::PulseMesh, wavenumber, distance_to_edge_tol, near_singular_tol, node1::Node, node2::Node, J_vec::AbstractArray{T,1}, V_vec::AbstractArray{T,1}) where T
    # Computes the RHS contributions for interactions between all elements
    # in node1 and node2
    @unpack num_elements,
            elements,
            areas,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    node1_elements = elements[node1.element_idxs,:]
    Z = Array{ComplexF64,2}(undef, num_elements, num_elements)
    for local_test_idx = 1:num_elements
        global_test_idx = node1.element_idxs[local_test_idx]
        for local_src_idx = 1:num_elements
            global_src_idx = node1.element_idxs[local_src_idx]
            is_singular = global_test_idx == global_src_idx
            testIntegrand(x,y,z) = scalarGreensIntegration(pulse_mesh,
                                                           global_src_idx,
                                                           wavenumber,
                                                           [x,y,z],
                                                           distance_to_edge_tol,
                                                           near_singular_tol,
                                                           is_singular)
            Z_entry = gaussQuadrature(areas[global_test_idx],
                                      testIntegrand,
                                      test_quadrature_points[global_test_idx],
                                      test_quadrature_weights)
            Z[global_test_idx,global_src_idx] = Z_entry
        end
    end
    Z
end
