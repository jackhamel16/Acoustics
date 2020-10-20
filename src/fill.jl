using LinearAlgebra

include("quadrature.jl")

function getTriangleNodes(element_idx::Int64, elements::Array{Int64, 2}, nodes::Array{Float64, 2})
    triangle_nodes = Array{Float64, 2}(undef, 3, 3)
    for node_idx_local in 1:3
        node_idx_global = elements[element_idx, node_idx_local]
        triangle_nodes[node_idx_local,:] = nodes[node_idx_global,:]
    end
    triangle_nodes
end

function rhsFill(num_elements::Int64, elements::Array{Int64, 2}, nodes::Array{Float64, 2}, fieldFunc::Function, quadrature_rule::Array{Float64, 2})
    rhs = Array{Complex{Float64}, 1}(undef, num_elements)
    for element_idx in 1:num_elements
        triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
        rhs[element_idx] = integrateTriangle(triangle_nodes, fieldFunc, quadrature_rule[:,1:3], quadrature_rule[:,4])
    end
    rhs
end

function matrixFill(num_elements::Int64, elements::Array{Int64, 2}, nodes::Array{Float64, 2}, testIntegrand::Function, quadrature_rule::Array{Float64, 2})
    z_matrix = Array{Complex{Float64}, 2}(undef, num_elements, num_elements)
    for src_idx in 1:num_elements
        for test_idx in 1:num_elements
            is_singular = (src_idx == test_idx)
            test_nodes = getTriangleNodes(test_idx, elements, nodes)
            src_nodes = getTriangleNodes(src_idx, elements, nodes)
            testIntegrandXYZ(x,y,z) = testIntegrand([x,y,z], src_nodes, is_singular)
            println("indices: ",test_idx, src_idx)
            z_matrix[test_idx, src_idx] = integrateTriangle(test_nodes, testIntegrandXYZ, quadrature_rule[:,1:3], quadrature_rule[:,4])
        end
    end
    z_matrix
end
