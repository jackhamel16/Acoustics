# dependencies: mesh.jl quadrature.jl

using LinearAlgebra

function rhsFill(num_elements::Int64, elements::Array{Int64, 2}, nodes::Array{Float64, 2}, fieldFunc::Function, quadrature_rule::Array{Float64, 2})
    rhs = Array{Complex{Float64}, 1}(undef, num_elements)
    for element_idx in 1:num_elements
        triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
        rhs[element_idx] = -1 * integrateTriangle(triangle_nodes, fieldFunc, quadrature_rule[:,1:3], quadrature_rule[:,4])
    end
    rhs
end

function matrixFill(num_elements::Int64, elements::Array{Int64, 2}, nodes::Array{Float64, 2}, testIntegrand::Function, quadrature_rule::Array{Float64, 2})
    # @unpack num_elements,
    #         elements,
    #         nodes,
    #         test_quadrature_points,
    #         test_quadrature_weights = pulse_mesh
    z_matrix = Array{Complex{Float64}, 2}(undef, num_elements, num_elements)
    for src_idx in 1:num_elements
        for test_idx in 1:num_elements
            is_singular = (src_idx == test_idx)
            test_nodes = getTriangleNodes(test_idx, elements, nodes)
            src_nodes = getTriangleNodes(src_idx, elements, nodes)
            testIntegrandXYZ(x,y,z) = testIntegrand([x,y,z], src_nodes, is_singular)
            z_matrix[test_idx, src_idx] = integrateTriangle(test_nodes, testIntegrandXYZ, quadrature_rule[:,1:3], quadrature_rule[:,4])
        end
    end
    z_matrix
end
