# dependencies: mesh.jl quadrature.jl

using LinearAlgebra

function rhsFill(pulse_mesh::PulseMesh, fieldFunc::Function)
    @unpack num_elements,
            elements,
            nodes,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    rhs = Array{Complex{Float64}, 1}(undef, num_elements)
    for element_idx in 1:num_elements
        triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
        rhs[element_idx] = -1 * integrateTriangle(triangle_nodes, fieldFunc, test_quadrature_points[element_idx], test_quadrature_weights)
    end
    rhs
end

function matrixFill(pulse_mesh::PulseMesh, testIntegrand::Function)
    @unpack num_elements,
            elements,
            nodes,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    z_matrix = Array{Complex{Float64}, 2}(undef, num_elements, num_elements)
    for src_idx in 1:num_elements
        for test_idx in 1:num_elements
            is_singular = (src_idx == test_idx)
            test_nodes = getTriangleNodes(test_idx, elements, nodes)
            src_nodes = getTriangleNodes(src_idx, elements, nodes)
            testIntegrandXYZ(x,y,z) = testIntegrand([x,y,z], src_idx, is_singular)
            z_matrix[test_idx, src_idx] = integrateTriangle(test_nodes, testIntegrandXYZ, test_quadrature_points[test_idx], test_quadrature_weights)
        end
    end
    z_matrix
end
